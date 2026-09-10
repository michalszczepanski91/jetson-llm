#!/usr/bin/env python3
"""Time-to-first-token and tokens/sec for an orchestrator-LLM candidate, via
the server's streaming OpenAI-compatible API (vLLM and llama-server both
support `"stream": true` the same way).

Separate script from scripts/benchmark.py (which measures one number:
end-to-end wall latency, through benchmarks/harness.py) because
run_benchmark() only times a call, it has no channel for a second per-rep
number - reuses harness.percentiles() directly, same as jetson-vlm-lab's
benchmark_streaming.py, so TTFT/tokens-per-sec are reported the same way
("never means") without duplicating that math.

No tools attached (see scripts/validate_tool_calling.py for tool-calling
correctness) - this measures raw generation speed on a fixed prompt.

Usage:
    uv run python scripts/benchmark_streaming.py --model-config 1.5b-awq-vllm-orin
    uv run python scripts/benchmark_streaming.py --model-config 1.5b-q4-llamacpp-orin --runs 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from harness import percentiles  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from model_config import load_model_config  # noqa: E402

_DEFAULT_PROMPT = "What do you see in front of you right now?"
_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="1.5b-awq-vllm-orin", help="key into configs/models.yaml")
    p.add_argument("--prompt", default=_DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--ready-timeout", type=float, default=600.0)
    p.add_argument("--label", default=None)
    p.add_argument("--results-json", default=None)
    add_target_args(p)
    return p.parse_args()


def _stream_one(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    import urllib.request

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_token_time = None
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                n_tokens += 1
    t_end = time.perf_counter()

    ttft = (first_token_time - t0) if first_token_time else None
    total = t_end - t0
    decode_seconds = (t_end - first_token_time) if first_token_time else None
    tokens_per_sec = (n_tokens / decode_seconds) if decode_seconds and decode_seconds > 0 else None
    return {"ttft_seconds": ttft, "total_seconds": total, "n_tokens": n_tokens, "tokens_per_sec": tokens_per_sec}


def main():
    args = parse_args()

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")

    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)
    print("Connecting to remote server..." if args.target == "remote" else f"Starting {variant['backend']} server...")
    coordinator.start()
    try:
        if not coordinator.wait_ready():
            raise TimeoutError(f"{variant['backend']} server did not become ready")

        print(f"Running {args.runs} streaming reps...")
        results = []
        # coordinator.model, NOT variant["model"]: for a local-checkpoint-path
        # Edge-LLM row (self-quantized checkpoints), the launch path and the
        # name the server actually registers requests under can differ - see
        # EdgeLlmCoordinator's served_model_name. Using the raw config path
        # here 404'd every streaming request for such a row (2026-09-09) -
        # silent everywhere else because vLLM/llama-cpp rows use an HF repo id
        # as both, so the two values happen to be identical there.
        for i in range(args.runs):
            result = _stream_one(coordinator.base_url, coordinator.model, args.prompt, args.max_tokens)
            results.append(result)
            if result["ttft_seconds"]:
                print(f"  [{i + 1}/{args.runs}] ttft={result['ttft_seconds']:.3f}s "
                      f"tokens/s={result['tokens_per_sec']:.1f}")
            else:
                print(f"  [{i + 1}] no tokens?")

        ttft_values = [r["ttft_seconds"] for r in results if r["ttft_seconds"] is not None]
        tps_values = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] is not None]
        row = {
            "label": args.label or f"orchestrator_llm_{args.model_config}_streaming",
            "model_config": args.model_config,
            "model": variant["model"],
            "backend": variant["backend"],
            "n_runs": len(results),
            "ttft_seconds": percentiles(ttft_values),
            "tokens_per_sec": percentiles(tps_values),
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "target": args.target,
        }
        if args.target == "remote":
            row["remote_host"] = args.remote_host
            row["remote_port"] = args.remote_port
        print(json.dumps(row, indent=2))

        if args.results_json:
            out = Path(args.results_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            rows = json.loads(out.read_text()) if out.exists() else []
            rows.append(row)
            out.write_text(json.dumps(rows, indent=2))
    finally:
        if args.target != "remote":
            print(f"Stopping {variant['backend']} server...")
        coordinator.stop()


if __name__ == "__main__":
    main()
