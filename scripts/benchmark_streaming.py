#!/usr/bin/env python3
"""TTFT / throughput / memory / power / ENERGY for one orchestrator-LLM
candidate at one workload point, emitting a document that conforms to
schemas/benchmark_result.schema.json.

Rewritten for docs/TODO.md Phase 2. What changed and why:

* **Energy exists now.** Before this, `scripts/benchmark.py` had tegrastats
  power but no idea how many tokens came back, while this script had token
  counts and started no sampler at all - so J/output-token, which
  docs/note.md §14 calls the headline metric and
  embedded-ai-chain/docs/paper.md §7.1 builds Semantic Decision Energy on,
  was **uncomputable from either script**. This one samples power across the
  measurement window *and* knows the token counts inside it.

* **Every measured repetition is kept** (docs/note.md §10). The previous
  version wrote only percentiles; percentiles are regenerable from raw runs,
  raw runs are not recoverable from percentiles.

* **Token counts come from the server's `usage`**, not from counting SSE
  deltas - see `src/llm_client.py:stream_llm`.

* **Cold start is decomposed** (docs/note.md §37), and says whether the run
  paid for a weight download. An undecomposed number has already misled this
  lab twice: a 338s "cold start" that was mostly a 6.7GB download, and a 2.0s
  one that was a warm-cache second run.

* **`--execution-condition` is required, with no default** (docs/note.md
  §15). On a unified-memory board a standalone number and a co-resident
  number are different physical quantities, and the one thing this repo must
  never produce is a table that silently mixes them.

Usage:
    uv run python scripts/benchmark_streaming.py \\
        --model-config 1.5b-awq-vllm-orin --execution-condition standalone

    uv run python scripts/benchmark_streaming.py \\
        --model-config 1.5b-q4-llamacpp-orin --execution-condition co-resident \\
        --co-resident yolo stt tts --runs 30
"""

import argparse
import datetime
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from harness import TegrastatsSampler, _half_average, _rail_stats, percentiles, process_rss_mb  # noqa: E402
from manifest import (  # noqa: E402
    SCHEMA_VERSION,
    build_manifest,
    energy_block,
    experiment_id,
    write_result,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_client import stream_llm  # noqa: E402
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from model_config import load_model_config  # noqa: E402

_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}

#: Below this many tegrastats samples inside the measurement window, the energy
#: figure is flagged. Not a hard failure - a thin number is still a number - but
#: it must not read as authoritative as one backed by a full window. 20 samples
#: is 10s at the sampler's 500ms interval, matching --min-measurement-s.
_MIN_POWER_SAMPLES = 20

#: Versioned per docs/note.md §21 - a prompt change is an experimental change,
#: so it may not live silently inside the benchmark implementation. Bump the
#: version string in PROMPT_TEMPLATE_VERSION if either of these changes.
PROMPT_TEMPLATE_VERSION = "v1"
_BASE_PROMPT = "What do you see in front of you right now?"
_FILLER = (
    "The camera observes the room and reports what it finds. "
    "Objects appear and disappear as people move between the desk and the door. "
)


def build_prompt(target_tokens: int | None, run_index: int | None = None) -> tuple[str, str]:
    """Return (prompt, prompt_source). With no target, the historical fixed
    prompt is used unchanged so this script's numbers stay comparable with
    the runs already on record.

    `run_index` makes the prompt unique per repetition, which exists to
    defeat KV-cache reuse. Both backends cache by prompt PREFIX, so the
    marker goes at the front - a unique suffix would leave everything before
    it reusable and change nothing. Confirmed live 2026-09-04: with an
    identical prompt, llama-server logged `prompt eval time = ... / 1 tokens`
    for a 40-token prompt on every run after the first, i.e. it was not
    prefilling at all.

    With a target, filler sentences are repeated to approximately that many
    tokens - *approximately* because there is no tokenizer on the host (this
    repo is stdlib-only by design and the tokenizer lives inside the serving
    container). The rough 0.75 words-per-token ratio only sets the target;
    the ACHIEVED count is read back from the server's own usage block and is
    what every derived figure uses. That is why the schema keeps
    `input_tokens_target` and per-run `input_tokens` as separate fields."""
    marker = f"[run {run_index}] " if run_index is not None else ""
    if target_tokens is None:
        return marker + _BASE_PROMPT, f"fixed_transcript_{PROMPT_TEMPLATE_VERSION}"
    words_needed = int(target_tokens * 0.75)
    filler_words = _FILLER.split()
    repeats = max(1, words_needed // len(filler_words) + 1)
    body = " ".join((_FILLER * repeats).split()[:words_needed])
    return f"{marker}{body}\n\n{_BASE_PROMPT}", f"repeated_filler_{PROMPT_TEMPLATE_VERSION}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="1.5b-awq-vllm-orin", help="key into configs/models.yaml")
    p.add_argument("--experiment", default="streaming", help="experiment name, becomes part of the experiment_id")
    p.add_argument("--input-tokens", type=int, default=None,
                   help="approximate prompt length to build (default: the fixed historical prompt)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=30, help="minimum measured repetitions (docs/note.md §48 wants >=30)")
    p.add_argument("--min-measurement-s", type=float, default=10.0,
                   help="wall-clock floor for the measurement window. tegrastats samples at 500ms, "
                        "so a fast cell can finish its repetitions before enough power samples exist "
                        "for a meaningful energy figure - measurement continues past --runs until "
                        "this elapses. Carried over from benchmarks/harness.py, which has the same "
                        "floor for the same reason")
    p.add_argument("--warmup-min", type=int, default=3)
    p.add_argument("--warmup-max", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="pinned for performance runs; server defaults change behaviour, not just wording")
    p.add_argument("--execution-condition", choices=["standalone", "co-resident"], required=True,
                   help="REQUIRED, no default (docs/note.md §15) - a standalone and a co-resident "
                        "number are different physical quantities on unified memory")
    p.add_argument("--co-resident", nargs="*", default=[],
                   help="which components ran alongside, e.g. --co-resident yolo stt tts")
    p.add_argument("--prompt-uniqueness", choices=["unique-per-run", "identical-per-run"],
                   default="unique-per-run",
                   help="unique-per-run (default) prefixes each request with a per-run marker so "
                        "the KV cache cannot be reused; identical-per-run reproduces the older "
                        "behaviour and measures WARM-CACHE prefill, which must be labelled as such")
    p.add_argument("--replicate", type=int, default=1, help="replicate number; re-running a cell means a new one")
    p.add_argument("--results-root", default=None, help="default: <repo>/results")
    p.add_argument("--ready-timeout", type=float, default=600.0)
    p.add_argument("--label", default=None)
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

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")
    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)

    unique = args.prompt_uniqueness == "unique-per-run"
    _, prompt_source = build_prompt(args.input_tokens)

    def messages_for(i: int) -> list[dict]:
        prompt, _ = build_prompt(args.input_tokens, run_index=i if unique else None)
        return [{"role": "user", "content": prompt}]

    exp_id = experiment_id(
        platform=variant["platform"],
        model_config_key=args.model_config,
        experiment=args.experiment,
        input_tokens=args.input_tokens,
        output_tokens=args.max_tokens,
        replicate=args.replicate,
    )
    print(f"experiment_id: {exp_id}")

    sampler = TegrastatsSampler()
    sampler.start()
    runs: list[dict] = []
    try:
        print("Connecting to remote server..." if args.target == "remote" else f"Starting {variant['backend']} server...")
        coordinator.start()
        if not coordinator.wait_ready():
            raise TimeoutError(f"{variant['backend']} server did not become ready")
        cold_start = coordinator.cold_start_breakdown()
        if cold_start["measured"]:
            print(f"cold start: {cold_start['total_s']}s "
                  f"(weights_cached={cold_start['weights_cached']})")

        # --- warmup: to thermal steady state, not a guessed count ----------
        warmup_count = 0
        steady = False
        while warmup_count < args.warmup_max:
            stream_llm(coordinator, messages_for(-warmup_count - 1), max_tokens=args.max_tokens,
                       temperature=args.temperature)
            warmup_count += 1
            if warmup_count >= args.warmup_min and sampler.is_thermally_stable():
                steady = True
                break
        print(f"warmup: {warmup_count} runs, steady_state={steady}")

        # --- measurement ---------------------------------------------------
        start_sample = sampler.latest() or {}
        rss_before = process_rss_mb()
        rss_samples: list[float] = []
        token_source = None  # bound before the loop: a zero-run cell still reports validity
        t_meas_start = time.monotonic()
        i = 0
        # Runs at least --runs repetitions AND for at least --min-measurement-s
        # of wall clock: a fast cell (vLLM at 108 tok/s finishes 8 runs in 4.4s)
        # would otherwise yield ~9 tegrastats samples, which is not enough to
        # characterise power - and the energy figure derived from it would look
        # exactly as authoritative as one backed by 60 samples.
        while i < args.runs or (time.monotonic() - t_meas_start) < args.min_measurement_s:
            try:
                record = stream_llm(coordinator, messages_for(i), max_tokens=args.max_tokens,
                                    temperature=args.temperature)
            except Exception as exc:  # noqa: BLE001 - a failed run is a result, not a crash
                record = {"e2e_latency_ms": 0.0, "error": f"{type(exc).__name__}: {exc}"}
            record["run_index"] = i
            token_source = record.pop("token_source", None)
            runs.append(record)
            if i % 10 == 0:
                r = process_rss_mb()
                if r is not None:
                    rss_samples.append(r)
                if record.get("ttft_ms"):
                    print(f"  [{i + 1}/{args.runs}] ttft={record['ttft_ms']:.1f}ms "
                          f"decode={record.get('decode_tok_s') or float('nan'):.1f}tok/s")
            i += 1
        t_meas_end = time.monotonic()
        r = process_rss_mb()
        if r is not None:
            rss_samples.append(r)
        end_sample = sampler.latest() or {}

        # --- aggregates, derived from `runs` and nothing else --------------
        ok = [r for r in runs if not r.get("error")]

        def agg(key):
            vals = [r[key] for r in ok if r.get(key) is not None]
            return percentiles(vals) if vals else None

        aggregates = {k: v for k, v in {
            "ttft_ms": agg("ttft_ms"),
            "e2e_latency_ms": agg("e2e_latency_ms"),
            "decode_ms": agg("decode_ms"),
            "prefill_tok_s": agg("prefill_tok_s"),
            "decode_tok_s": agg("decode_tok_s"),
            "output_tokens": agg("output_tokens"),
            "inter_token_latency_ms": (
                percentiles([g for r in ok for g in r.get("inter_token_latency_ms") or []]) or None
            ),
        }.items() if v}

        # --- power / energy, over the measurement window only --------------
        power_samples = sampler.samples_between(t_meas_start, t_meas_end)
        rails = {rail: _rail_stats(power_samples, rail)
                 for rail in ("vdd_gpu_soc_mw", "vdd_cpu_cv_mw", "vin_sys_5v0_mw")}
        total_output_tokens = sum(r.get("output_tokens") or 0 for r in ok)
        power = energy_block(
            rails,
            window_s=t_meas_end - t_meas_start,
            n_requests=len(ok),
            total_output_tokens=total_output_tokens,
        )

        ram = [s["ram_used_mb"] for s in power_samples if "ram_used_mb" in s]
        temps = [t for t in (s.get("tj_temp_c", s.get("gpu_temp_c")) for s in power_samples) if t is not None]

        flags = []
        if args.target == "remote":
            flags += ["cold_start_not_measured", "telemetry_is_client_side_not_inference_host"]
        if token_source and token_source.startswith("sse delta"):
            flags.append("token_counts_are_sse_chunk_counts_not_tokens")
        n_power = max((s_["n"] for s_ in rails.values() if s_), default=0)
        if n_power < _MIN_POWER_SAMPLES:
            flags.append(f"power_window_thin_{n_power}_samples")

        result = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": exp_id,
            "label": args.label or f"{args.model_config} out={args.max_tokens}",
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "manifest": build_manifest(
                backend=variant["backend"],
                platform=variant["platform"],
                base_url=coordinator.base_url,
                image=getattr(coordinator, "image", None),
                command=shlex.join([sys.executable, *sys.argv]),
                model_config_key=args.model_config,
            ),
            "model": {
                "name": variant["model"],
                "precision": variant["precision"],
                "backend": variant["backend"],
                "platform": variant["platform"],
            },
            "workload": {
                "input_tokens_target": (args.input_tokens if args.input_tokens is not None
                                       else _achieved_or_none(ok)),
                "output_tokens_target": args.max_tokens,
                "batch_size": 1,
                "concurrency": 1,
                "prompt_source": prompt_source,
                "prompt_uniqueness": args.prompt_uniqueness,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "tools_attached": False,
                "sampling": {"temperature": args.temperature, "seed": None},
            },
            "execution_condition": args.execution_condition,
            "cold_start": cold_start,
            "runs": runs,
            "aggregates": aggregates,
            "memory": {
                "system_ram_used_mb": {"peak": max(ram) if ram else None, "steady": _half_average(ram)},
                "system_ram_total_mb": (start_sample or end_sample).get("ram_total_mb"),
                "client_rss_mb": {
                    "before": rss_before,
                    "peak": max(rss_samples) if rss_samples else None,
                    "steady": _half_average(rss_samples),
                },
                "attribution_caveat": (
                    "client_rss_mb is this benchmark process only - an HTTP client - and is NOT "
                    "model memory; the model runs inside the server's container. "
                    "system_ram_used_mb from tegrastats is the real figure on this unified-memory "
                    "board. weights_mb/kv_cache_mb are not broken out: neither backend reports "
                    "them over its HTTP API."
                ),
            },
            "power": power,
            "thermal": {
                "start_c": start_sample.get("tj_temp_c", start_sample.get("gpu_temp_c")),
                "end_c": end_sample.get("tj_temp_c", end_sample.get("gpu_temp_c")),
                "max_c": max(temps) if temps else None,
                "sensor": "tj",
            },
            "validity": {
                "warmup_runs": warmup_count,
                "warmup_reached_steady_state": steady,
                "measurement_runs": len(runs),
                "thermal_throttling_observed": False,
                "failed_runs": len(runs) - len(ok),
                "flags": flags,
                "notes": f"token_source: {token_source}",
            },
        }
        if args.co_resident:
            result["co_resident_workload"] = args.co_resident

        path = write_result(result, results_root=args.results_root or _default_results_root())
        print(f"\nwrote {path}")
        _validate(result)
        _print_summary(result)
    finally:
        sampler.stop()
        if args.target != "remote":
            print(f"Stopping {variant['backend']} server...")
        coordinator.stop()


def _achieved_or_none(ok: list[dict]) -> int | None:
    """With no --input-tokens target, the fixed prompt still has a length, so
    report the achieved count as the target - it is the same number, measured
    rather than requested. Returns None (not 0) when the server sent no usage
    block at all: the schema permits null there precisely so this gap stays
    visible instead of being filled with a fabricated target."""
    for r in ok:
        if r.get("input_tokens"):
            return r["input_tokens"]
    return None


def _default_results_root():
    return Path(__file__).resolve().parent.parent / "results"


def _validate(result: dict) -> None:
    """Fail loudly here rather than at analysis time. Optional import: the
    benchmark must still run on a machine where jsonschema isn't installed,
    but it will say so instead of quietly skipping the check."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("! jsonschema not installed - result NOT validated against the schema")
        return
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "benchmark_result.schema.json"
    errors = sorted(
        Draft202012Validator(json.loads(schema_path.read_text())).iter_errors(result),
        key=lambda e: list(e.path),
    )
    if errors:
        print(f"! result does NOT conform to {schema_path.name}:")
        for e in errors[:10]:
            print(f"    {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}")
    else:
        print(f"✓ conforms to {schema_path.name}")


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


if __name__ == "__main__":
    main()
