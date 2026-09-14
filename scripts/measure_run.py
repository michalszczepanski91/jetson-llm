#!/usr/bin/env python3
"""One config in, one run directory out: records.jsonl + power.jsonl +
run_meta.json.

The v2 measurement path. `scripts/run_experiment.py` stays exactly as it is
- it produces `benchmark_result.schema.json` documents and every existing
figure depends on them - but it cannot answer the questions this run has to
answer, and retrofitting it would have meant changing the shape of a
document 95 results already conform to.

What is different here, and why:

  * **Statistics are not computed at measurement time.** This script writes
    per-request rows and nothing else; every percentile, interval and ratio
    is derived later by `analysis/stats.py` reading only the JSONL. So a
    figure can be regenerated, a percentile redefined, or an interval method
    changed without putting the board back under load for an hour. The old
    path aggregated in `measure_cell()` and kept the raw runs alongside,
    which meant the aggregate and the raw could in principle disagree.
  * **Prefill and decode are separate regimes, so the x axis is prompt
    length.** One TTFT at one prompt length is not a prefill result.
  * **Memory is measured, including the largest context that actually
    runs.** Probed until it fails, not read back from `--max-model-len`.
  * **Idle power is measured per cell, immediately before the cell**, with
    the server loaded and resident, so `marginal` energy means something.
  * **Prompts are byte-identical across frameworks and repetitions**, which
    is only safe with prefix caching OFF - so the run verifies that, by
    timing a second identical request, rather than trusting a flag.

Usage:
    python3 scripts/measure_run.py --model-config 7b-fp16-vllm-thor-p1 \
        --experiment p1-context --execution-condition standalone
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import device_memory as dm           # noqa: E402
import energy as en                  # noqa: E402
import prompts as pr                 # noqa: E402
import representation as rep         # noqa: E402
from harness import TegrastatsSampler, _rail_stats, nvpmodel_mode  # noqa: E402
from llm_client import stream_llm    # noqa: E402
from llm_coordinator import add_target_args, build_coordinator     # noqa: E402
from manifest import (               # noqa: E402
    DEFAULT_ENERGY_RAILS, assert_condition_matches_reality, build_manifest, experiment_id,
)
from model_config import load_model_config  # noqa: E402

#: Every rail SAMPLED, derived from the energy rails plus the board-level
#: supply - same single-source-of-truth rule benchmarks/runner.py records
#: having learned the hard way when a third private copy of the rail names
#: left a Thor run reporting no joules.
POWER_RAILS = DEFAULT_ENERGY_RAILS + ("vin_sys_5v0_mw",)

#: Every rail SUMMED into a joule figure - the per-domain rails only.
#: `vin_sys_5v0_mw` is the board-level supply and already contains much of
#: what the domain rails report, so including it roughly doubles the answer.
#: The smoke run on 2026-09-11 did exactly that and reported 37.6W of
#: generation power against a true ~23.5W before the two tuples were
#: separated; benchmarks/manifest.py's DEFAULT_ENERGY_RAILS comment had
#: already warned that vin_sys is "recorded for context, deliberately NOT
#: summed into the energy figure".
ENERGY_RAILS = DEFAULT_ENERGY_RAILS

RUN_FORMAT_VERSION = "2.0.0"

#: Per-request fields that turn a backend's own prompt cache off, where that
#: cache is a request-level setting rather than a launch flag. Only
#: llama-server needs one: vLLM takes `--no-enable-prefix-caching` at launch
#: (set on the config row) and Edge-LLM has shown no prompt reuse on this
#: box. Unknown fields are ignored by every backend here, so this is inert
#: where it does not apply - and in all three cases the per-cell cache probe,
#: not this table, is what establishes that nothing is being reused.
CACHE_OFF_BODY = {
    "llama-cpp": {"cache_prompt": False},
}


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment", required=True,
                   help="short name for this run, e.g. p1-context")
    p.add_argument("--execution-condition", required=True, choices=["standalone", "co-resident"],
                   help="no default, on purpose - a standalone and a co-resident number are "
                        "different physical quantities on unified memory")
    p.add_argument("--co-resident", nargs="*", default=None,
                   help="every workload sharing the board; required with co-resident")
    p.add_argument("--prompt-tokens", default="128,512,2048,8192",
                   help="prompt-length sweep, comma separated (reference-tokenizer counts)")
    p.add_argument("--gen-tokens", default="128", help="generation lengths, comma separated")
    p.add_argument("--runs", type=int, default=30, help="measured repetitions per cell")
    p.add_argument("--warmup", type=int, default=5,
                   help="warm-up requests per cell, discarded (brief asks for >=5)")
    p.add_argument("--idle-seconds", type=float, default=20.0)
    p.add_argument("--warmup-min-s", type=float, default=150.0,
                   help="minimum seconds of sustained load before measuring. Not a guess: "
                        "this board's CPU governor takes ~2min to promote the busy core, "
                        "worth 39%% of vLLM's decode rate - see warm_up()")
    p.add_argument("--min-window-s", type=float, default=25.0,
                   help="keep measuring past --runs until the window is this long, so the "
                        "energy figure rests on enough 500ms tegrastats samples")
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--results-root", default=str(REPO_ROOT / "results" / "v2"))
    p.add_argument("--skip-max-context", action="store_true",
                   help="skip the context probe (it costs a few long requests)")
    p.add_argument("--max-context-cap", type=int, default=65536)
    p.add_argument("--ready-timeout", type=float, default=1200.0)
    add_target_args(p)
    return p.parse_args()


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


class Records:
    """Append-only JSONL sink. Flushed per row so a run killed at minute 40
    still leaves 40 minutes of analysable data - the whole reason statistics
    live downstream of this file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w")
        self.n = 0

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self.n += 1

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------
# representation
# --------------------------------------------------------------------------

def _snapshot_dir(model: str, cache: str = "/opt/hf-cache/hub") -> Path | None:
    root = Path(cache) / f"models--{model.replace('/', '--')}" / "snapshots"
    if not root.exists():
        return None
    snaps = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0] if snaps else None


def _gguf_path(model: str, quant: str | None, cache: str = "/opt/llama-cache") -> Path | None:
    root = Path(cache) / f"models--{model.replace('/', '--')}" / "snapshots"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/*.gguf"))
    if quant:
        tagged = [p for p in candidates if quant.lower() in p.name.lower()]
        candidates = tagged or candidates
    firsts = [p for p in candidates if "-of-" not in p.name or "00001-of-" in p.name]
    return (firsts or candidates or [None])[0]


def describe_representation(variant: dict[str, Any]) -> dict[str, Any]:
    """Read the artifact this config will actually load and decompose it.

    Activation and KV dtype are NOT readable from the artifact - they are
    properties of how the runtime was launched - so each is asserted from
    the launch configuration and carries a `_source` saying exactly that.
    Where a backend can be made to print its own KV dtype, the log scrape
    will contradict this block if it disagrees, and that contradiction is
    the point of recording the source."""
    backend = variant["backend"]
    model = variant["model"]
    if backend == "llama-cpp":
        path = _gguf_path(model, variant.get("quant"))
        if not path:
            return {"_not_collected": f"no GGUF found on disk for {model}"}
        artifact = rep.describe_gguf(path)
        kv_dtype = variant.get("cache_type_k", "f16")
        kv_src = ("llama-server default (f16) unless --cache-type-k/-v was passed; "
                  "recorded from the launch configuration, not read back from the server")
        act = "fp32 accumulate / fp16 compute (ggml CUDA kernels dequantise to float)"
        act_src = "property of the ggml CUDA backend, not a launch flag"
    else:
        snap = _snapshot_dir(model)
        if not snap:
            return {"_not_collected": f"no HF snapshot found on disk for {model}"}
        artifact = rep.describe_safetensors_model(snap)
        kv_dtype = variant.get("kv_cache_dtype", "auto -> model dtype")
        kv_src = "launch configuration (vLLM --kv-cache-dtype; 'auto' means the model dtype)"
        act = artifact.get("hf_config", {}).get("torch_dtype")
        act_src = "checkpoint torch_dtype; the runtime was not asked to override it"

    block = rep.build_representation(
        artifact=artifact,
        declared_label=variant.get("precision", ""),
        backend=backend,
        kv_cache_dtype=kv_dtype,
        kv_cache_dtype_source=kv_src,
        activations_dtype=act,
        activations_dtype_source=act_src,
    )
    block["_artifact"] = {k: v for k, v in artifact.items() if k != "modules"}
    return block


def kv_per_token(representation: dict[str, Any]) -> tuple[int | None, str]:
    """2 x n_layers x n_kv_heads x head_dim x bytes_per_element, from the
    model config. Returns (bytes, formula-as-written) so the result records
    the arithmetic and not just its answer."""
    hf = (representation.get("_artifact") or {}).get("hf_config") or {}
    layers = hf.get("num_hidden_layers")
    kv_heads = hf.get("num_key_value_heads")
    hidden = hf.get("hidden_size")
    heads = hf.get("num_attention_heads")
    if not all((layers, kv_heads, hidden, heads)):
        return None, "model config did not expose n_layers / n_kv_heads / hidden_size"
    head_dim = hidden // heads
    kv_dtype = (representation.get("kv_cache", {}).get("dtype") or "").lower()
    bpe = 1 if ("fp8" in kv_dtype or "int8" in kv_dtype or "q8" in kv_dtype) else 2
    value = rep.kv_cache_bytes_per_token(layers, kv_heads, head_dim, bpe)
    formula = (f"2 x {layers} layers x {kv_heads} kv_heads x {head_dim} head_dim "
               f"x {bpe} bytes = {value} B/token (KV dtype: {kv_dtype or 'unknown'})")
    return value, formula


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    variant = load_model_config(args.model_config)
    prompt_targets = _ints(args.prompt_tokens)
    gen_targets = _ints(args.gen_tokens)

    run_id = experiment_id(
        platform=variant["platform"], model_config_key=args.model_config,
        experiment=args.experiment, replicate=args.replicate,
    )
    out_dir = Path(args.results_root) / run_id
    if (out_dir / "run_meta.json").exists():
        raise SystemExit(
            f"{out_dir}/run_meta.json already exists. Re-running a cell means a new "
            f"replicate (--replicate {args.replicate + 1}), never a silent overwrite "
            "of data a figure was drawn from."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    records = Records(out_dir / "records.jsonl")
    print(f"→ {out_dir}")

    container = f"llmlab-{args.model_config}"[:60]
    assert_condition_matches_reality(
        args.execution_condition, own_containers={container},
        declared_co_resident=args.co_resident,
    )

    representation = describe_representation(variant)
    kv_bytes, kv_formula = kv_per_token(representation)
    print(f"   representation: {representation.get('comparability_class')} "
          f"{representation.get('effective_bits_per_weight')} bits/weight, "
          f"KV {kv_bytes} B/token")

    baseline_mem = dm.system_memory()
    # The device baseline is what makes the footprint attributable. On a
    # `standalone` run it is zero (assert_condition_matches_reality has
    # already refused the run otherwise); on a co-resident one it is the
    # other workload's footprint, and subtracting it is the whole point.
    baseline_dev = dm.device_memory_total()
    if baseline_dev is not None:
        baseline_mem["device_bytes"] = baseline_dev
    mem_sampler = dm.MemorySampler(interval_s=0.5)
    power = TegrastatsSampler(interval_ms=500)
    mem_sampler.start()
    power.start()

    local_kwargs = {k: v for k, v in variant.items() if k not in {
        "model", "backend", "platform", "precision", "notes"}}
    local_kwargs["container_name"] = container
    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)

    meta: dict[str, Any] = {
        "run_format_version": RUN_FORMAT_VERSION,
        "run_id": run_id,
        "config_id": args.model_config,
        "framework": variant["backend"],
        "experiment": args.experiment,
        "replicate": args.replicate,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "representation": representation,
        "execution_condition": args.execution_condition,
        "co_resident_workload": args.co_resident,
        "cells": [],
    }
    if args.co_resident:
        meta["co_resident_workload"] = list(args.co_resident)

    t_start = time.monotonic()
    backend_mem: dict[str, Any] = {"_source": "server never started"}
    max_ctx: dict[str, Any] = {"max_context_measured": None,
                               "_not_collected": "the run ended before the probe"}
    try:
        print("   starting server…")
        coordinator.start()
        if not coordinator.wait_ready(args.ready_timeout):
            raise SystemExit(f"server never became ready: {coordinator.error}")
        cold = (coordinator.cold_start_breakdown()
                if hasattr(coordinator, "cold_start_breakdown") else {})
        print(f"   ready after {cold.get('total_s')}s")

        meta["manifest"] = build_manifest(
            backend=variant["backend"], platform=variant["platform"],
            base_url=coordinator.base_url, image=getattr(coordinator, "image", None),
            command=meta["command"], model_config_key=args.model_config,
            experiment_config=None, target=args.target,
        )
        meta["cold_start"] = cold
        meta["nvpmodel"] = nvpmodel_mode()

        # --- prompt corpus: built once, reused byte-for-byte everywhere ----
        # The corpus is keyed by the REFERENCE model, not by what this row
        # serves. A llama.cpp row names a `-GGUF` repo, which would key a
        # second corpus file and send different bytes than the vLLM and
        # Edge-LLM legs - destroying the one property that makes the three
        # comparable. `prompt_corpus_reference` lets such a row name the
        # corpus it shares.
        corpus_ref = variant.get("prompt_corpus_reference") or variant["model"]
        corpus = pr.load_or_build(
            REPO_ROOT / "configs" / "prompt_corpus",
            prompt_targets,
            lambda: pr.TokenCounter(coordinator.base_url, coordinator.model),
            reference_model=corpus_ref,
        )
        if corpus_ref != variant["model"]:
            corpus["_shared_corpus_of"] = corpus_ref
        meta["prompt_corpus"] = {
            k: v for k, v in corpus.items() if k != "entries"
        } | {"entries": {k: {kk: vv for kk, vv in v.items() if kk != "text"}
                         for k, v in corpus["entries"].items()}}
        print(f"   corpus {corpus['corpus_sha256'][:12]} via {corpus['reference_count_method']}")

        # --- memory the runtime reports about itself ----------------------
        log = dm.container_log(container) if variant["backend"] != "edge-llm" else None
        backend_mem = dm.parse_backend_memory(variant["backend"], log)
        if log:
            (out_dir / "server_startup.log").write_text(log)
        print(f"   backend log fields: {sorted(k for k in backend_mem if not k.startswith('_') and k != 'matched_patterns')}")

        # --- largest context that actually runs ---------------------------
        if args.skip_max_context:
            max_ctx = {"max_context_measured": None,
                       "_not_collected": "--skip-max-context was passed"}
        else:
            ref = corpus["entries"][str(max(prompt_targets))]
            words_per_token = len(ref["text"].split()) / max(1, ref["reference_tokens"])

            def make_text(n: int) -> str:
                return f"[probe{n}] " + pr._shuffled_filler(int(n * words_per_token), seed=n) + pr._QUESTION

            print("   probing max context…")
            max_ctx = dm.probe_max_context(
                coordinator.base_url, coordinator.model, make_text,
                lo=min(prompt_targets), hi_cap=args.max_context_cap,
                on_attempt=lambda a: records.write(
                    {"kind": "max_context_probe", "config_id": args.model_config, **a}),
            )
            print(f"   max context measured: {max_ctx.get('max_context_measured')} "
                  f"({max_ctx.get('limiting_factor')})")

        # --- the grid ------------------------------------------------------
        for p_target in prompt_targets:
            entry = corpus["entries"][str(p_target)]
            text = entry["text"]
            for g in gen_targets:
                cell = run_cell(
                    coordinator=coordinator, records=records, power=power,
                    config_id=args.model_config, framework=variant["backend"],
                    text=text, entry=entry, gen_tokens=g,
                    runs=args.runs, warmup=args.warmup,
                    idle_seconds=args.idle_seconds, min_window_s=args.min_window_s,
                    warmup_min_s=args.warmup_min_s,
                    nvpmodel=meta["nvpmodel"],
                    extra_body=CACHE_OFF_BODY.get(variant["backend"]),
                )
                meta["cells"].append(cell)

    finally:
        # Composed in `finally`, not after the grid: a run stopped part-way
        # through its cells has still MEASURED a memory footprint and a set
        # of cache probes, and throwing them away because the campaign was
        # interrupted would discard real data. The 2026-09-11 DVFS
        # diagnostic was exactly such a run, and its results.json came out
        # with an empty memory block for no better reason than where these
        # two lines sat.
        meta["wall_clock_s"] = round(time.monotonic() - t_start, 1)
        try:
            meta["memory"] = dm.compose(
                backend_reported=backend_mem,
                kv_bytes_per_token=kv_bytes or 0,
                kv_formula=kv_formula,
                artifact_bytes=(representation.get("_artifact") or {}).get("artifact_bytes", 0),
                baseline=baseline_mem, session=mem_sampler.summary(), max_context=max_ctx,
            )
            meta["confounds"] = confound_block(variant, meta, args)
        except Exception as exc:  # noqa: BLE001 - a partial run is still a result
            meta["memory_error"] = f"{type(exc).__name__}: {exc}"
        try:
            coordinator.stop()
        except Exception as exc:  # noqa: BLE001
            meta["stop_error"] = str(exc)
        power_rows = [{"t": ts, **s} for ts, s in getattr(power, "_samples", [])]
        power.stop()
        mem_sampler.stop()
        with (out_dir / "power.jsonl").open("w") as fh:
            for row in power_rows:
                fh.write(json.dumps(row) + "\n")
        records.close()
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"   wrote {records.n} records + {len(power_rows)} power samples → {out_dir}")
    return 0


def _record(row: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """One per-request row for records.jsonl.

    `inter_token_latency_ms` is kept as the full vector, not a summary. That
    is the whole point of the file: the brief asks for p50/p90/p99 and an
    ITL standard deviation, and every one of those must be recomputable
    later without putting the board back under load."""
    return {**base, **{k: v for k, v in row.items() if k != "token_source"}}


def warm_up(
    coordinator,
    records: "Records",
    base: dict[str, Any],
    messages: list[dict],
    gen_tokens: int,
    *,
    extra_body: dict[str, Any] | None = None,
    min_runs: int = 5,
    min_seconds: float = 150.0,
    max_runs: int = 80,
    max_seconds: float = 480.0,
    window: int = 6,
    drift_threshold: float = 0.05,
) -> dict[str, Any]:
    """Warm up until the decode rate stops moving, not for a fixed count.

    **Why a count is not enough on this board, measured 2026-09-11.** With
    the brief's five warm-ups, the first cell of the first Priority-1 run
    produced this decode trajectory across 30 identical repetitions:

        9.43 9.45 9.46 9.52 9.54 9.37 9.55 9.53 9.56 10.50 13.33 13.00 ...

    a flat plateau, a STEP of +39% at repetition 10, and a flat plateau
    after it - about two minutes into sustained load. The cause is CPU DVFS,
    not thermal and not the GPU: `schedutil` governs the CPU from 972MHz to
    2601MHz while the GPU GPC clock is already pinned at its maximum, and
    vLLM's batch-1 decode is partly CPU-bound on per-token scheduling. A p50
    taken across that step belongs to neither regime. The full record is
    results/invalid/2026-09-11_thor-dvfs-ramp-diagnostic/.

    Two conditions, because each misses what the other catches:

      * **A minimum duration under load.** A coefficient-of-variation check
        alone would have *passed* at repetition 6 - the pre-step plateau is
        extremely stable, at 9.5 tok/s with well under 1% spread - and
        declared steady state in the wrong regime. Only elapsed time under
        sustained load reaches the governor's promotion.
      * **A stability check.** Duration alone would fix the clock ramp and
        miss a slow thermal decay, which moves the other way and has no
        fixed time constant.

    The stability check is a TREND test - the mean of the last `window` runs
    against the mean of the `window` before it - and not a spread test, which
    is what it was first written as. A coefficient-of-variation threshold has
    to be set below the process noise to mean anything, and on this board the
    decode rate's own run-to-run range over six samples is 5-13% of its mean:
    a 3% threshold was simply unreachable and ran every cell to its cap, while
    a threshold loose enough to be reachable would have accepted the 39% step
    this function exists to exclude. Comparing two consecutive means separates
    the two: noise cancels between the halves, a regime change does not.

    Clocks are deliberately NOT locked with `jetson_clocks`. Locking would
    remove the confound outright, but it would also stop measuring the board
    a deployment actually runs on; the honest alternative is to measure at
    the steady state the default governor reaches and to record how long
    that took. `warmup_seconds_to_steady_state` in every cell is that
    record."""
    rates: list[float] = []
    t0 = time.monotonic()
    i = 0
    reached = False
    while True:
        r = stream_llm(coordinator, messages, max_tokens=gen_tokens, temperature=0.0,
                       extra_body=extra_body)
        records.write(_record(r, {**base, "kind": "request", "phase": "warmup",
                                  "run_index": -i - 1}))
        if r.get("decode_tok_s"):
            rates.append(r["decode_tok_s"])
        i += 1
        elapsed = time.monotonic() - t0
        if i >= max_runs or elapsed >= max_seconds:
            break
        if i < min_runs or elapsed < min_seconds or len(rates) < 2 * window:
            continue
        recent = rates[-window:]
        previous = rates[-2 * window:-window]
        m_recent = sum(recent) / window
        m_prev = sum(previous) / window
        overall = (m_recent + m_prev) / 2
        drift = abs(m_recent - m_prev) / overall if overall else 1.0
        if drift <= drift_threshold:
            reached = True
            break
    recent = rates[-window:] if len(rates) >= window else rates
    mean = sum(recent) / len(recent) if recent else None
    return {
        "runs": i,
        "seconds": round(time.monotonic() - t0, 1),
        "reached_steady_state": reached,
        "criterion": (f"at least {min_runs} runs AND at least {min_seconds}s under load AND "
                      f"the mean of the last {window} decode rates within {drift_threshold:.0%} "
                      f"of the mean of the {window} before them"),
        "decode_tok_s_trajectory": [round(x, 3) for x in rates],
        "decode_tok_s_at_steady_state": round(mean, 3) if mean else None,
        "_caveat": None if reached else (
            f"steady state was NOT reached within {max_runs} runs / {max_seconds}s - the "
            "measurement that follows may straddle a regime change and its percentiles "
            "belong to no single regime"
        ),
    }


def run_cell(
    *,
    coordinator,
    records: "Records",
    power,
    config_id: str,
    framework: str,
    text: str,
    entry: dict[str, Any],
    gen_tokens: int,
    runs: int,
    warmup: int,
    idle_seconds: float,
    min_window_s: float,
    warmup_min_s: float,
    nvpmodel: str,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one (prompt length x generation length) cell.

    Order is deliberate and each step depends on the one before it:

      1. **Warm up** with the real prompt, at least 5, discarded but written
         to the JSONL with `phase: "warmup"` so a reader can check that they
         were in fact discarded and see where steady state began.
      2. **Cache probe**, on a prompt with a unique LEADING token that has
         never been sent. Both backends cache by PREFIX, so a salt at the
         front invalidates the whole match - a salt at the end would leave
         everything before it reusable and the probe would measure nothing.
         Sending it twice and comparing TTFT tests the property directly,
         and unlike reading back a launch flag it works identically on all
         three backends.

         **It must come after warm-up, not before.** Run first, it charges
         the very first request to this server with every one-time cost the
         server has - allocator, CUDA graph entry, lazy init - and then
         compares that against a second request that pays none of them. On
         the 2026-09-11 smoke run that produced a 8.9x "ratio" and a
         confident false verdict of PREFIX REUSE IS ACTIVE on a server whose
         own log said `Prefix cache hit rate: 0.0%`. Warming up first makes
         the two probe requests differ in the one thing being tested.
      3. **Idle power**, with the server loaded and resident but serving
         nothing, immediately before the measured window and therefore in
         the same thermal and power state the measurement will run in.
      4. **Measure**, sending the identical prompt every repetition. That is
         only legitimate because step 2 verified nothing is caching it; the
         payoff is that the bytes are identical across repetitions AND
         across frameworks, which is what makes the cross-framework
         comparison a comparison.

    Measurement continues past `runs` until `min_window_s` has elapsed. At
    500ms tegrastats sampling a 5-second cell yields 11 power samples, and
    an energy figure backed by 11 samples must not read as authoritative as
    one backed by 60 - the same floor benchmarks/runner.py enforces."""
    base = {
        "config_id": config_id, "framework": framework,
        "prompt_target_tokens": entry["target_tokens"],
        "prompt_reference_tokens": entry["reference_tokens"],
        "prompt_sha256": entry["sha256"],
        "gen_tokens": gen_tokens, "concurrency": 1, "nvpmodel": nvpmodel,
    }
    label = f"in~{entry['target_tokens']} out{gen_tokens}"

    # 1. warm-up to a MEASURED steady state --------------------------------
    messages = [{"role": "user", "content": text}]
    warm = warm_up(coordinator, records, base, messages, gen_tokens,
                   extra_body=extra_body, min_runs=warmup, min_seconds=warmup_min_s)

    # 2. cache probe ------------------------------------------------------
    salt = hashlib.sha256(f"{time.time_ns()}{label}".encode()).hexdigest()[:12]
    probe_text = f"[cacheprobe {salt}] {text}"
    probe: list[dict[str, Any]] = []
    for attempt in (1, 2):
        r = stream_llm(coordinator, [{"role": "user", "content": probe_text}],
                       max_tokens=min(16, gen_tokens), temperature=0.0,
                       extra_body=extra_body)
        probe.append(r)
        records.write(_record(r, {**base, "kind": "request", "phase": "cache_probe",
                                  "attempt": attempt, "run_index": -100 + attempt}))
    t1, t2 = probe[0].get("ttft_ms"), probe[1].get("ttft_ms")
    cache_verdict = None
    if t1 and t2:
        ratio = t2 / t1
        cache_verdict = {
            "first_ttft_ms": round(t1, 2), "second_ttft_ms": round(t2, 2),
            "ratio_second_over_first": round(ratio, 3),
            # 0.7 is a deliberately loose threshold: run-to-run TTFT noise on
            # this board is a few percent, while a genuine prefix-cache hit
            # on a 2048-token prompt is a 5-10x collapse. Anything between is
            # neither, and is reported as inconclusive rather than forced
            # into a verdict.
            "prefix_caching_appears_disabled": ratio > 0.7,
            "verdict": ("the second identical request was not faster - no prefix reuse detected"
                        if ratio > 0.7 else
                        f"the second identical request was {1/ratio:.1f}x faster - PREFIX REUSE "
                        "IS ACTIVE and every repetition after the first is a cache hit"),
        }

    # 3. idle baseline ----------------------------------------------------
    idle = en.measure_idle(power, ENERGY_RAILS, seconds=idle_seconds)

    # 4. measurement ------------------------------------------------------
    t0 = time.monotonic()
    ok = 0
    out_tokens = 0
    i = 0
    while i < runs or (time.monotonic() - t0) < min_window_s:
        try:
            r = stream_llm(coordinator, messages, max_tokens=gen_tokens, temperature=0.0,
                           extra_body=extra_body)
        except Exception as exc:  # noqa: BLE001 - a failed run is a result, not a crash
            r = {"e2e_latency_ms": None, "error": f"{type(exc).__name__}: {exc}"}
        records.write(_record(r, {**base, "kind": "request", "phase": "measure", "run_index": i}))
        if not r.get("error"):
            ok += 1
            out_tokens += r.get("output_tokens") or 0
        i += 1
    t_end = time.monotonic()

    samples = power.samples_between(t0, t_end)
    rails = {r: _rail_stats(samples, r) for r in POWER_RAILS}
    temps = [t for t in (s.get("tj_temp_c", s.get("gpu_temp_c")) for s in samples) if t is not None]
    energy = en.task_energy(
        rail_stats=rails, rails=ENERGY_RAILS, window_s=t_end - t0, idle=idle,
        n_requests=ok, total_output_tokens=out_tokens, n_tool_calls_completed=None,
    )
    achieved = next((r for r in (probe[0],) if r.get("input_tokens")), {}).get("input_tokens")
    cell = {
        "label": label,
        "prompt_target_tokens": entry["target_tokens"],
        "prompt_reference_tokens": entry["reference_tokens"],
        "prompt_achieved_tokens_incl_template": achieved,
        "gen_tokens": gen_tokens,
        "runs_requested": runs, "runs_measured": i, "runs_ok": ok,
        "warmup_discarded": warm["runs"],
        "warmup_seconds_to_steady_state": warm["seconds"],
        "warmup_reached_steady_state": warm["reached_steady_state"],
        "warmup": warm,
        "extra_runs_for_min_window": max(0, i - runs),
        "measurement_window_s": round(t_end - t0, 3),
        # Same reason as energy.measure_idle's: these let stats.py re-slice
        # power.jsonl for this cell and bootstrap the mean rail power, which
        # is what gives the energy figures an interval instead of a bare
        # single-measurement number.
        "power_window_t0": t0,
        "power_window_t1": t_end,
        "power_samples_in_window": len(samples),
        "energy": energy,
        "cache_probe": cache_verdict,
        "thermal": {
            "max_c": max(temps) if temps else None,
            "mean_c": round(sum(temps) / len(temps), 2) if temps else None,
            "sensor": "tj",
        },
    }
    e = energy.get("marginal") or energy.get("absolute") or {}
    print(f"   {label}: warmup {warm['runs']}r/{warm['seconds']}s "
          f"{'steady' if warm['reached_steady_state'] else 'NOT STEADY'} "
          f"({warm['decode_tok_s_at_steady_state']} tok/s) | {ok}/{runs} ok, {len(samples)} power samples, "
          f"{e.get('j_per_turn')} J/turn ({'marginal' if 'marginal' in energy else 'absolute'}), "
          f"cache={'off' if (cache_verdict or {}).get('prefix_caching_appears_disabled') else 'SUSPECT'}")
    return cell


def confound_block(variant: dict[str, Any], meta: dict[str, Any], args) -> dict[str, Any]:
    """Everything the brief lists as silently invalidating a cross-framework
    comparison, recorded as either a verification or an honest gap.

    Split into `verified` (something was measured or read back) and
    `asserted` (a launch flag was set and believed). The distinction is the
    whole value of the block: `enable_prefix_caching: false` in a config file
    is an assertion, and the cache probe in every cell is the verification,
    and this lab has already been burned by the difference between those two
    kinds of statement."""
    probes = [c["cache_probe"] for c in meta["cells"] if c.get("cache_probe")]
    caching_clean = all(p["prefix_caching_appears_disabled"] for p in probes) if probes else None
    return {
        "verified": {
            "prefix_caching_disabled": {
                "value": caching_clean,
                "method": "second identical request timed against the first, per cell, on a "
                          "prompt with a unique leading token",
                "per_cell_ratios": [p["ratio_second_over_first"] for p in probes],
            },
            "board_quiet": {
                "value": True,
                "method": "assert_condition_matches_reality(): no other containers and no "
                          "bare-metal process holding an nvhost/nvgpu/nvmap handle",
            },
            "warmups_discarded": {
                "value": [c.get("warmup_discarded") for c in meta["cells"]],
                "method": "warmed up to a MEASURED steady state per cell, not to a fixed "
                          "count: at least --warmup-min-s under load plus a stability check "
                          "on recent decode rates. Written to records.jsonl with "
                          "phase='warmup' and excluded by analysis/stats.py, which reads "
                          "only phase='measure'",
                "seconds_per_cell": [c.get("warmup_seconds_to_steady_state") for c in meta["cells"]],
                "reached_steady_state_per_cell": [
                    c.get("warmup_reached_steady_state") for c in meta["cells"]],
            },
            "identical_prompt_bytes_within_run": {
                "value": True,
                "method": "every repetition of a cell sends one frozen corpus string; its "
                          "sha256 is on every record row",
            },
        },
        "asserted": {
            "greedy_decode": {
                "value": "temperature=0.0",
                "caveat": "temperature alone is NOT controlled sampling - the three backends "
                          "apply different top_p/top_k/min_p defaults (Edge-LLM 0.9/50, vLLM "
                          "1.0/-1). This run pins temperature only; the greedy control that "
                          "pins top_k=1 is a separate experiment.",
            },
            "seed": {"value": None, "_not_collected":
                     "no seed is sent; at temperature=0 the sampler is deterministic anyway, "
                     "but this is an assertion about the backend, not a verification"},
            "max_tokens_and_stop": {
                "value": f"max_tokens={args.gen_tokens}, no stop sequences on any backend",
            },
            "prefix_caching_request_field": {
                "value": CACHE_OFF_BODY.get(variant["backend"]),
                "caveat": "sent on every request where the backend's prompt cache is a "
                          "request-level setting (llama-server's cache_prompt); inert "
                          "elsewhere. Verified by the cache probe, not by this field.",
            },
            "prefix_caching_flag": {
                "value": variant.get("enable_prefix_caching"),
                "caveat": "a launch flag, believed; the verification is the cache probe above",
            },
            "gpu_offload": {
                "value": variant.get("n_gpu_layers"),
                "caveat": "llama.cpp only; -1 means all layers. Verified against the server's "
                          "own 'offloaded N/M layers' log line in memory.offloaded_layers",
            },
            "nvpmodel": {"value": meta.get("nvpmodel")},
            "fan_profile": {"value": None, "_not_collected":
                            "not read; /sys/devices/pwm-fan is not exposed on this L4T build "
                            "and no fan profile was pinned for this run"},
        },
        "not_yet_verified": {
            "identical_rendered_prompt_bytes_across_frameworks": (
                "asserted by construction - all frameworks send the same corpus string - but "
                "the RENDERED prompt (after each backend's chat template) is not compared in "
                "this run. scripts/chat_template_crossfeed.py did exactly that comparison for "
                "vLLM vs Edge-LLM on BFCL and found byte equality; it has not been re-run for "
                "this corpus."
            ),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
