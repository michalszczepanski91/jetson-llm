#!/usr/bin/env python3
"""End-to-end wall latency/cold-start/thermal/power for an orchestrator-LLM
candidate (any configs/models.yaml row - vLLM or llama.cpp, Orin or Thor),
through benchmarks/harness.py (see that module's docstring for what it
measures and why - a copy of jetson-vlm-lab's/embedded-ai-chain's harness).

Fixed test input: one fixed transcript-shaped prompt, non-streaming, no
tools attached - this measures raw chat-completion latency, not tool-calling
behavior (see scripts/validate_tool_calling.py for that) or
tokens/sec/TTFT (see scripts/benchmark_streaming.py).

setup_fn starts the local coordinator and waits for /health - this *is*
cold start for --target local. teardown_fn stops it. For --target remote,
cold_start_ms measures nothing (the server was already running) - flagged
in extra_metadata, not silently reported as real, same convention
jetson-vlm-lab's benchmark.py uses.

rss_mb is not meaningful for --target local either: the model lives inside
the Docker container, so this script's own process RSS is just an HTTP
client's footprint. system_ram_mb from tegrastats is the real number on
this unified-memory hardware.

Usage:
    uv run python scripts/benchmark.py --model-config 1.5b-awq-vllm-orin
    uv run python scripts/benchmark.py --model-config 1.5b-q4-llamacpp-orin --runs 30 \\
        --results-json output/benchmark_1.5b_llamacpp.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from harness import run_benchmark  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_client import call_llm  # noqa: E402
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from model_config import load_model_config  # noqa: E402

_DEFAULT_PROMPT = "What do you see in front of you right now?"
_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="1.5b-awq-vllm-orin", help="key into configs/models.yaml")
    p.add_argument("--prompt", default=_DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--label", default=None)
    p.add_argument("--warmup-min", type=int, default=3)
    p.add_argument("--warmup-max", type=int, default=10)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--ready-timeout", type=float, default=600.0)
    p.add_argument("--results-json", default=None)
    add_target_args(p)
    return p.parse_args()


def main():
    args = parse_args()

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")

    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)

    messages = [{"role": "user", "content": args.prompt}]

    def setup():
        coordinator.start()
        if not coordinator.wait_ready():
            raise TimeoutError(f"{variant['backend']} server did not become ready")
        return coordinator

    def work(_coordinator):
        call_llm(_coordinator, messages, max_tokens=args.max_tokens)

    def teardown(_coordinator):
        coordinator.stop()

    extra_metadata = {
        "model_config": args.model_config,
        "backend": variant["backend"],
        "platform": variant["platform"],
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "target": args.target,
        "rss_mb_caveat": "not meaningful - the model runs inside the server's own "
        "container, this process is just an HTTP client.",
    }
    if args.target == "remote":
        extra_metadata["remote_host"] = args.remote_host
        extra_metadata["remote_port"] = args.remote_port
        extra_metadata["cold_start_caveat"] = (
            "not meaningful - RemoteCoordinator.start() is a no-op, the server "
            "was already running before this script started; cold_start_ms below "
            "measures nothing"
        )
        extra_metadata["tegrastats_caveat"] = (
            "system_ram_mb/power_mw/thermal_c/nvpmodel_mode/jetson_clocks_locked "
            "below describe the machine running THIS script, not the remote "
            "inference server actually doing the work"
        )

    run_benchmark(
        setup,
        work,
        teardown,
        label=args.label or f"orchestrator_llm_{args.model_config}",
        model=variant["model"],
        precision=variant["precision"],
        warmup_min=args.warmup_min,
        warmup_max=args.warmup_max,
        runs=args.runs,
        results_json=args.results_json,
        extra_metadata=extra_metadata,
    )


if __name__ == "__main__":
    main()
