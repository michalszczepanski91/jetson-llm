#!/usr/bin/env python3
"""One-off single-cell benchmark: TTFT / throughput / memory / power / energy
for one candidate at one workload point.

This is the ad-hoc entry point - useful for probing a new candidate or
debugging a coordinator. **For anything that will be reported, use
`scripts/run_experiment.py`** with a `configs/benchmarks/*.yaml` file:
docs/TODO.md GATE 3 requires a benchmark to be re-runnable from (model config
key, experiment config, git SHA) alone, and a flag typed at a shell prompt is
not a record of what was measured.

Both scripts share `benchmarks/runner.py:measure_cell()`, so they cannot
disagree about how a measurement is taken. That matters here specifically:
this repo has already lost a debugging session to `docker-compose.yml` and
`VllmCoordinator` drifting apart on `--enable-auto-tool-choice`, and a
second entry point re-implementing the measurement loop would be the same
mistake in a more expensive place.

Usage:
    uv run python scripts/benchmark_streaming.py \\
        --model-config 1.5b-awq-vllm-orin --execution-condition standalone

    uv run python scripts/benchmark_streaming.py \\
        --model-config 1.5b-q4-llamacpp-orin --execution-condition co-resident \\
        --co-resident yolo stt tts --runs 30
"""

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # a redirected/backgrounded run fully
# buffers stdout otherwise, which silently hid a live campaign's progress from a
# `tail -f` monitor for 20+ minutes on 2026-09-04 - the process was fine, only the
# log looked idle.

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness import TegrastatsSampler  # noqa: E402
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from manifest import assert_condition_matches_reality, write_result  # noqa: E402
from model_config import load_model_config  # noqa: E402
from runner import Cell, measure_cell  # noqa: E402

_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="1.5b-awq-vllm-orin", help="key into configs/models.yaml")
    p.add_argument("--experiment", default="streaming",
                   help="experiment name, becomes part of the experiment_id")
    p.add_argument("--input-tokens", type=int, default=None,
                   help="approximate prompt length to build (default: the fixed historical prompt)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=30,
                   help="minimum measured repetitions (docs/note.md 48 wants >=30)")
    p.add_argument("--warmup-min", type=int, default=3)
    p.add_argument("--warmup-max", type=int, default=20)
    p.add_argument("--min-measurement-s", type=float, default=10.0,
                   help="wall-clock floor for the measurement window. tegrastats samples at 500ms, "
                        "so a fast cell can finish its repetitions before enough power samples "
                        "exist for a meaningful energy figure")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="pinned for performance runs; server defaults change behaviour, not just wording")
    p.add_argument("--execution-condition", choices=["standalone", "co-resident"], required=True,
                   help="REQUIRED, no default (docs/note.md 15) - a standalone and a co-resident "
                        "number are different physical quantities on unified memory")
    p.add_argument("--co-resident", nargs="*", default=[],
                   help="which components ran alongside, e.g. --co-resident yolo stt tts")
    p.add_argument("--prompt-uniqueness", choices=["unique-per-run", "identical-per-run"],
                   default="unique-per-run",
                   help="unique-per-run (default) prefixes each request with a per-run marker so "
                        "the KV cache cannot be reused; identical-per-run reproduces the older "
                        "behaviour and measures WARM-CACHE prefill, which must be labelled as such")
    p.add_argument("--replicate", type=int, default=1,
                   help="replicate number; re-running a cell means a new one, never an overwrite")
    p.add_argument("--results-root", default=None, help="default: <repo>/results")
    p.add_argument("--ready-timeout", type=float, default=900.0)
    add_target_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    if args.execution_condition == "co-resident" and not args.co_resident:
        raise SystemExit(
            'error: --execution-condition co-resident requires --co-resident naming what ran '
            'alongside (e.g. --co-resident yolo stt tts). "Under load" is not a reproducible '
            "condition, and the schema rejects a co-resident result without it."
        )

    if args.target == "local":
        assert_condition_matches_reality(
            args.execution_condition,
            own_containers={"vllm-llm-lab", "llamacpp-llm-lab"},
            declared_co_resident=args.co_resident,
        )

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")
    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)

    sampler = TegrastatsSampler()
    sampler.start()
    try:
        print("Connecting to remote server..." if args.target == "remote"
              else f"Starting {variant['backend']} server...")
        coordinator.start()
        if not coordinator.wait_ready():
            raise TimeoutError(f"{variant['backend']} server did not become ready")
        cold_start = coordinator.cold_start_breakdown()
        if cold_start["measured"]:
            print(f"cold start: {cold_start['total_s']}s (weights_cached={cold_start['weights_cached']})")

        def progress(i, record):
            if record.get("ttft_ms"):
                print(f"  [{i + 1}/{args.runs}] ttft={record['ttft_ms']:.1f}ms "
                      f"decode={record.get('decode_tok_s') or float('nan'):.1f}tok/s")

        result = measure_cell(
            coordinator=coordinator,
            variant=variant,
            model_config_key=args.model_config,
            cell=Cell(output_tokens=args.max_tokens, input_tokens=args.input_tokens),
            sampler=sampler,
            cold_start=cold_start,
            execution_condition=args.execution_condition,
            co_resident=args.co_resident,
            experiment=args.experiment,
            command=shlex.join([sys.executable, *sys.argv]),
            temperature=args.temperature,
            runs=args.runs,
            warmup_min=args.warmup_min,
            warmup_max=args.warmup_max,
            min_measurement_s=args.min_measurement_s,
            prompt_uniqueness=args.prompt_uniqueness,
            replicate=args.replicate,
            target=args.target,
            on_progress=progress,
        )
        path = write_result(result, results_root=args.results_root or (REPO_ROOT / "results"))
        print(f"\nwrote {path}")
        _validate(result)
        _print_summary(result)
    finally:
        sampler.stop()
        if args.target != "remote":
            print(f"Stopping {variant['backend']} server...")
        coordinator.stop()


def _validate(result: dict) -> None:
    """Fail loudly here rather than at analysis time. Optional import: the
    benchmark must still run where jsonschema isn't installed, but it says so
    instead of quietly skipping the check."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("! jsonschema not installed - result NOT validated against the schema")
        return
    schema_path = REPO_ROOT / "schemas" / "benchmark_result.schema.json"
    errors = sorted(
        Draft202012Validator(json.loads(schema_path.read_text())).iter_errors(result),
        key=lambda e: list(e.path),
    )
    if errors:
        print(f"! result does NOT conform to {schema_path.name}:")
        for e in errors[:10]:
            print(f"    {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}")
    else:
        print(f"OK - conforms to {schema_path.name}")


def _print_summary(result: dict) -> None:
    a, p = result["aggregates"], result["power"]
    print("\n--- summary (p50 / p95 / p99) ---")
    for key in ("ttft_ms", "e2e_latency_ms", "decode_tok_s"):
        if key in a:
            s = a[key]
            print(f"  {key:<18} {s['p50']:.1f} / {s['p95']:.1f} / {s['p99']:.1f}   (n={s['n']})")
    if "energy_per_output_token_j" in p:
        print(f"  {'J/output-token':<18} {p['energy_per_output_token_j']:.4f}"
              f"   (rails: {', '.join(p['rails_included'])})")
    else:
        print("  J/output-token     not computed - no power samples or no token counts")
    if result["validity"]["flags"]:
        print(f"  flags: {', '.join(result['validity']['flags'])}")


if __name__ == "__main__":
    main()
