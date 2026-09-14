#!/usr/bin/env python3
"""records.jsonl -> results.json. Every statistic in this lab is computed
here and nowhere else.

The separation is the point. `scripts/measure_run.py` writes per-request rows
and stops; this script turns them into percentiles, intervals and derived
ratios; `analysis/figures.py` reads only what this produces. So a percentile
can be redefined, an interval method changed, or a figure regenerated without
putting the board back under load for an hour - and, more importantly, the
published aggregate and the raw data can never drift apart, because the
aggregate has no independent existence.

Two rules it enforces on the way through:

  * **Only `phase == "measure"` rows count.** Warm-ups are written to the
    JSONL so a reader can see them and check where steady state began, and
    they are excluded here. Cache probes and max-context probes likewise:
    they are diagnostics, and one of them is deliberately a 16-token request
    that would drag a decode percentile down if it were counted.
  * **A missing input produces a missing output, never a zero.** Every leaf
    goes through `benchmarks/envelope.py`, whose absent case carries a reason
    string. A partial honest file is the goal.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import envelope as E  # noqa: E402
from manifest import DEFAULT_ENERGY_RAILS  # noqa: E402

RESULTS_FORMAT_VERSION = "2.0.0"

#: Peak DRAM bandwidth, for the memory-bandwidth-utilisation figure.
#: Datasheet values, NOT measured on this board - decode MBU computed from
#: them is an upper-bound-relative number and is labelled as such wherever it
#: appears. A measured STREAM-style figure would be the honest denominator
#: and this lab does not have one.
PEAK_DRAM_GBPS = {
    "thor": 273.0,   # AGX Thor Developer Kit, 128GB LPDDR5X
    "orin": 204.8,   # AGX Orin 64GB, LPDDR5
}

#: Percentiles reported for every latency distribution. p99 on n=30 is a
#: weak estimate and its bootstrap interval says so; it is reported because
#: an orchestrator's worst turn is what a user actually notices, and
#: omitting it would hide the tail this whole block exists to expose.
PCTS = (50, 90, 99)


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """(run metadata, per-request rows, power samples).

    power.jsonl is read because the energy figures need an interval and a
    window mean has no computable spread on its own. Each tegrastats sample
    inside a cell's window is a draw from that cell's power distribution, so
    bootstrapping their mean is a legitimate interval - and it is the only
    way "error bars on everything" does not quietly except energy."""
    meta = json.loads((run_dir / "run_meta.json").read_text())
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()
               if line.strip()]
    power_path = run_dir / "power.jsonl"
    power = [json.loads(line) for line in power_path.read_text().splitlines()
             if line.strip()] if power_path.exists() else []
    return meta, records, power


def _vals(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None and not r.get("error")]


def _single(value: float | int | None, unit: str, reason: str) -> dict[str, Any]:
    """A quantity measured exactly once (a memory footprint, a cold start).

    Given `n=1` and `method="single measurement"` rather than a fabricated
    interval: one reading has no spread, and dressing it in a ci95 of
    [value, value] would claim a precision nobody measured."""
    if value is None:
        return E.not_collected(reason, unit)
    return E.measured(value, unit, n=1, method="single measurement")


def latency_block(rows: list[dict]) -> dict[str, Any]:
    """TTFT, inter-token latency, turn time - percentiles, not means.

    Jitter matters more than the mean for an on-device orchestrator: a
    stuttering 40 tok/s is worse to listen to than a steady 25, and only the
    ITL spread distinguishes them. So the ITL standard deviation is reported
    alongside its percentiles, and the raw per-request ITL vectors stay in
    the JSONL so any other spread statistic can be computed later."""
    ttft = _vals(rows, "ttft_ms")
    e2e = _vals(rows, "e2e_latency_ms")
    decode = _vals(rows, "decode_ms")
    itl = [g for r in rows if not r.get("error") for g in (r.get("inter_token_latency_ms") or [])]

    block: dict[str, Any] = {"n_requests": len(rows), "n_ok": len([r for r in rows if not r.get("error")])}
    for name, vals, unit in (("ttft_ms", ttft, "ms"), ("turn_ms", e2e, "ms"),
                             ("decode_ms", decode, "ms"), ("inter_token_ms", itl, "ms")):
        for p in PCTS:
            block[f"{name}_p{p}"] = E.percentile_envelope(vals, p, unit)
    block["inter_token_ms_stdev"] = E.stdev_envelope(itl, "ms")
    block["inter_token_ms_n_gaps"] = len(itl)
    # The ECDF the latency-tail figure draws, precomputed here as a quantile
    # grid. figures.py reads only results.json and must never open a
    # records.jsonl, so the distribution has to travel WITH the result - and
    # a 101-point grid is a faithful ECDF at plot resolution while costing a
    # few hundred bytes instead of the tens of thousands of raw gaps.
    block["inter_token_ms_ecdf"] = (
        {"q": list(range(0, 101)),
         "ms": [round(E.percentile(itl, q), 4) for q in range(0, 101)],
         "n": len(itl)}
        if len(itl) >= 10 else
        {"q": [], "ms": [], "n": len(itl),
         "_not_collected": "fewer than 10 inter-token gaps in this cell"}
    )
    block["_inter_token_caveat"] = (
        "gaps between successive SSE content deltas, not between tokens. A backend may "
        "coalesce several tokens into one delta or split one across two, and the three "
        "backends do not agree on it - so an ITL comparison ACROSS frameworks is confounded "
        "by chunking, while within one framework it is a clean jitter measurement."
    )
    return block


def throughput_block(rows: list[dict]) -> dict[str, Any]:
    """Prefill and decode reported separately, because they are different
    regimes: prefill is compute-bound and decode is bandwidth-bound, and
    quantization moves them in opposite directions. Folding prefill into
    TTFT - one number, at one prompt length - hides exactly that."""
    block: dict[str, Any] = {}
    for name, key, unit in (("prefill_tok_s", "prefill_tok_s", "tokens/s"),
                            ("decode_tok_s", "decode_tok_s", "tokens/s"),
                            ("output_tokens", "output_tokens", "tokens"),
                            ("input_tokens", "input_tokens", "tokens")):
        vals = _vals(rows, key)
        block[f"{name}_p50"] = E.percentile_envelope(vals, 50, unit)
    block["_prefill_caveat"] = (
        "prefill_tok_s is input_tokens / TTFT, so it charges prefill with every fixed "
        "per-request cost the server has - scheduling, tokenisation, the first sampling "
        "step. At 128 tokens that overhead dominates and the figure understates the "
        "kernel; at 8192 it is negligible. Compare prefill rates across prompt LENGTHS "
        "with that in mind, and across frameworks only at the same length."
    )
    return block


def _window_power_mw(power: list[dict], t0: float | None, t1: float | None) -> list[float]:
    """Total summed-rail power for each tegrastats sample inside a window.

    Only the per-domain rails are summed. `vin_sys_5v0_mw` is sampled and
    recorded for context but must NOT enter a joule figure - it is the
    board-level supply and already contains much of what the domain rails
    report. Including it once cost this campaign a 37.6W reading against a
    true 23.5W (2026-09-11 smoke run)."""
    if t0 is None or t1 is None:
        return []
    out = []
    for s in power:
        t = s.get("t")
        if t is None or not (t0 <= t <= t1):
            continue
        present = [s[r] for r in DEFAULT_ENERGY_RAILS if s.get(r) is not None]
        if present:
            out.append(sum(present))
    return out


def _power_envelope(samples: list[float], reason: str) -> dict[str, Any]:
    return E.mean_envelope(samples, "mW", reason_if_empty=reason) if samples \
        else E.not_collected(reason, "mW")


def _scaled(leaf: dict[str, Any], factor: float, unit: str) -> dict[str, Any]:
    """Multiply an envelope by an exact constant, carrying its interval.

    Energy is power x time and per-turn energy is that over a counted number
    of turns; both multipliers are exact, so the relative interval is
    unchanged and scaling the bounds is the correct propagation rather than
    an approximation."""
    if not E.is_measured(leaf):
        return E.not_collected(leaf.get("_not_collected", "input not measured"), unit)
    ci = [c * factor for c in leaf["ci95"]] if leaf.get("ci95") else None
    return E.measured(leaf["value"] * factor, unit, ci95=ci, n=leaf.get("n"),
                      method=leaf.get("method"))


def energy_block(cell: dict[str, Any], quality: dict[str, Any] | None,
                 power: list[dict]) -> dict[str, Any]:
    """Absolute and marginal energy in each denominator the decision uses,
    every figure carrying an interval derived from the power samples
    themselves.

    Joules per correct answer is assembled HERE and only here: it needs an
    accuracy from a quality run that the measurement script never saw, and
    joining the two at analysis time is what keeps the performance and
    quality document types independently measurable (docs/note.md §5)."""
    raw = cell.get("energy") or {}
    idle = raw.get("idle") or {}
    window_s = raw.get("measurement_window_s") or 0.0
    n_turns = cell.get("runs_ok") or 0
    gen_samples = _window_power_mw(power, cell.get("power_window_t0"), cell.get("power_window_t1"))
    idle_samples = _window_power_mw(power, idle.get("window_t0"), idle.get("window_t1"))

    p_gen = _power_envelope(gen_samples, "no power samples inside the measurement window")
    p_idle = _power_envelope(idle_samples, "no power samples inside the idle window")
    # Fall back to the window means the measurement script already computed
    # when a run predates the window-bound fields. Reported without an
    # interval and labelled, rather than silently dropped.
    if not E.is_measured(p_gen) and raw.get("generation_power_mw") is not None:
        p_gen = E.measured(raw["generation_power_mw"], "mW", n=1,
                           method="single window mean (this run recorded no power window "
                                  "bounds, so no interval is computable)")
    if not E.is_measured(p_idle) and idle.get("total_power_mw") is not None:
        p_idle = E.measured(idle["total_power_mw"], "mW", n=1,
                            method="single window mean (no idle window bounds recorded)")

    out: dict[str, Any] = {
        "rails_included": raw.get("rails_included"),
        "rails_summed_into_joules": list(DEFAULT_ENERGY_RAILS),
        "measurement_window_s": window_s,
        "conventions": raw.get("conventions"),
        "idle_power_mw": p_idle,
        "generation_power_mw": p_gen,
        "n_turns_in_window": n_turns,
    }

    total_out_tokens = None
    if raw.get("absolute", {}).get("j_per_output_token") and raw["absolute"].get("energy_j"):
        total_out_tokens = raw["absolute"]["energy_j"] / raw["absolute"]["j_per_output_token"]

    variants: dict[str, dict[str, Any]] = {"absolute": p_gen}
    if E.is_measured(p_gen) and E.is_measured(p_idle):
        marg_ci = None
        if p_gen.get("ci95") and p_idle.get("ci95"):
            marg_ci = [max(0.0, p_gen["ci95"][0] - p_idle["ci95"][1]),
                       max(0.0, p_gen["ci95"][1] - p_idle["ci95"][0])]
        variants["marginal"] = E.measured(
            max(0.0, p_gen["value"] - p_idle["value"]), "mW", ci95=marg_ci,
            n=min(p_gen.get("n") or 1, p_idle.get("n") or 1),
            method="difference of two window means; interval by interval arithmetic, so "
                   "it is wider than a joint resampling would give")
    else:
        out["marginal_skipped_reason"] = (
            raw.get("marginal_skipped_reason") or "no idle baseline was measured for this cell")

    acc = (quality or {}).get("accuracy")
    for name, p_leaf in variants.items():
        conv = (raw.get(name) or {}).get("convention") or (raw.get("conventions") or {}).get(name)
        block: dict[str, Any] = {"power_mw": p_leaf, "convention": conv}
        joules = _scaled(p_leaf, window_s / 1000.0, "J")
        block["energy_j"] = joules
        block["j_per_turn"] = _scaled(joules, 1.0 / n_turns, "J/turn") if n_turns else \
            E.not_collected("no successful turns in this window", "J/turn")
        block["j_per_output_token"] = _scaled(joules, 1.0 / total_out_tokens, "J/token") \
            if total_out_tokens else E.not_collected(
                "no output token count for this window", "J/token")
        src = raw.get(name) or {}
        block["j_per_completed_tool_call"] = E.not_collected(
            src.get("_j_per_completed_tool_call_not_collected",
                    "this workload attached no tools"), "J/tool-call") \
            if src.get("j_per_completed_tool_call") is None else \
            E.measured(src["j_per_completed_tool_call"], "J/tool-call", n=1,
                       method="single measurement")
        if acc and E.is_measured(acc) and E.is_measured(block["j_per_turn"]):
            block["j_per_correct_answer"] = E.ratio_envelope(
                block["j_per_turn"], acc, "J/correct-answer")
            block["_j_per_correct_answer_join"] = (quality or {}).get("_join")
        else:
            block["j_per_correct_answer"] = E.not_collected(
                "no quality run has been joined to this config yet; run the accuracy suite "
                "and re-run analysis/stats.py with --quality", "J/correct-answer")
        out[name] = block

    if "marginal" not in out:
        out["marginal"] = {"_not_collected": out.get("marginal_skipped_reason",
                                                     "marginal energy was not computed")}
    return out


def memory_block(meta: dict[str, Any]) -> dict[str, Any]:
    """One footprint per config, repeated onto every row of that config.

    Repeated rather than stored once because a row is meant to be
    self-contained for plotting - `figures.py` should never have to join two
    tables to draw a memory bar - and because the cost is a few hundred
    bytes."""
    m = meta.get("memory") or {}
    total = m.get("system_total_bytes")
    mx = m.get("max_context") or {}
    block = {
        "weights_bytes_on_device": _single(m.get("weights_bytes_on_device"), "bytes", "not measured"),
        "weights_bytes_source": m.get("weights_bytes_source"),
        "kv_cache_bytes_per_token": _single(m.get("kv_cache_bytes_per_token"), "bytes/token", "not measured"),
        "kv_cache_bytes_per_token_formula": m.get("kv_cache_bytes_per_token_formula"),
        "kv_cache_bytes_reserved": _single(m.get("kv_cache_bytes_reserved"), "bytes",
                                           "the runtime did not report a KV reservation"),
        "peak_device_bytes": _single(
            m.get("peak_device_bytes"), "bytes",
            m.get("_peak_device_not_collected", "no device-memory samples")),
        "peak_device_source": m.get("peak_device_source"),
        "peak_attributable_bytes": _single(
            m.get("peak_attributable_bytes"), "bytes",
            "needs a device-memory baseline taken before the server started"),
        "free_bytes_remaining": _single(
            m.get("free_bytes_remaining"), "bytes", "no device-memory samples"),
        "runtime_overhead_bytes": _single(
            m.get("runtime_overhead_bytes"), "bytes",
            "needs both a measured device peak and a runtime component breakdown"),
        "component_breakdown_bytes": m.get("component_breakdown_bytes"),
        "breakdown_note": m.get("breakdown_note"),
        "board_accounting_caveat": m.get("board_accounting_caveat"),
        "system_total_bytes": _single(total, "bytes", "not read"),
        "max_context_measured": _single(mx.get("max_context_measured"), "tokens",
                                        mx.get("_not_collected", "not probed")),
        "max_context_limiting_factor": mx.get("limiting_factor"),
        "max_context_configured": _single(m.get("max_context_configured"), "tokens",
                                          "the runtime did not print its configured limit"),
        "attribution_note": m.get("board_accounting_caveat"),
    }
    for k in ("graph_capture_bytes", "non_torch_bytes", "activation_peak_bytes",
              "offloaded_layers", "total_layers"):
        if k in m:
            block[k] = _single(m[k], "bytes" if k.endswith("_bytes") else "layers", "")
    return block


def efficiency_block(
    meta: dict[str, Any],
    memory: dict[str, Any],
    throughput: dict[str, Any],
    mean_context_tokens: float,
) -> dict[str, Any]:
    """Memory-bandwidth utilisation for decode; MFU for prefill.

    MBU = bytes read per token x tokens/s / peak DRAM bandwidth. For a
    batch-1 autoregressive decode the bytes read per token are essentially
    the whole weight set plus the KV rows for the current context: every
    weight is touched exactly once per token and nothing is reused across
    tokens, which is what makes decode bandwidth-bound in the first place.
    The KV term grows from the prompt length to prompt+gen over a turn, so
    the mean context - prompt + gen/2 - is what goes in.

    Why it is here at all: anything below ~40% MBU is a backend problem
    being reported as a precision result. A config that looks 20% slower
    because its runtime leaves half the bandwidth on the floor has said
    nothing about its quantization scheme, and without this number the two
    are indistinguishable in a latency table.

    The denominator is a DATASHEET peak, not a measured STREAM figure, so
    every MBU here is relative to a bound no real kernel reaches. Read the
    ordering and the order of magnitude; do not read the absolute percent as
    a hardware efficiency."""
    platform = ((meta.get("manifest") or {}).get("hardware") or {}).get("platform")
    peak = PEAK_DRAM_GBPS.get(platform)
    w = memory["weights_bytes_on_device"]
    kv = memory["kv_cache_bytes_per_token"]
    tok = throughput.get("decode_tok_s_p50", {})
    block: dict[str, Any] = {
        "prefill_mfu": E.not_collected(
            "needs a FLOP count per prefill token and a datasheet peak FLOP/s at this "
            "board's achieved clocks; neither is established for Thor in this lab yet",
            "fraction"),
    }
    if not (peak and E.is_measured(w) and E.is_measured(kv) and E.is_measured(tok)):
        block["decode_mbu"] = E.not_collected(
            f"needs a datasheet peak bandwidth for platform {platform!r} plus measured "
            "weights, KV/token and decode rate", "fraction")
        return block

    bytes_per_token = w["value"] + kv["value"] * mean_context_tokens
    achieved_gbps = bytes_per_token * tok["value"] / 1e9
    mbu = achieved_gbps / peak
    ci = [bytes_per_token * c / 1e9 / peak for c in tok["ci95"]] if tok.get("ci95") else None
    block["decode_mbu"] = E.measured(mbu, "fraction", ci95=ci, n=tok.get("n"),
                                     method="derived from the bootstrapped decode rate")
    block["decode_achieved_gbps"] = E.measured(achieved_gbps, "GB/s", n=tok.get("n"),
                                               method="derived")
    block["peak_dram_gbps"] = E.measured(peak, "GB/s", n=1,
                                         method="datasheet for this board, NOT measured")
    block["bytes_read_per_token"] = E.measured(bytes_per_token, "bytes", n=1, method="derived")
    block["mean_context_tokens"] = mean_context_tokens
    block["backend_bound_flag"] = bool(mbu < 0.40)
    block["_interpretation"] = (
        "MBU below ~0.40 means the runtime is leaving most of the memory bandwidth unused, "
        "so a latency difference against another config is at least partly a backend "
        "difference and not a representation difference. The denominator is a datasheet peak "
        "no real kernel reaches, so treat the absolute value as a lower bound on true "
        "utilisation and compare configs to each other rather than to 1.0."
    )
    return block


def quality_for(config_id: str, quality_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The quality block for one config, or an honest empty one.

    Quality and performance are separate document types on purpose
    (docs/note.md §5) and join here on `config_id`. Accuracy carries a Wilson
    interval; anything the suite did not score is absent with a reason."""
    q = quality_runs.get(config_id)
    if not q:
        return {
            "accuracy": E.not_collected(
                "no quality run joined to this config; run the accuracy suite and pass "
                "--quality to analysis/stats.py", "fraction"),
            "suites": {},
        }
    return q


def build_rows(
    meta: dict[str, Any],
    records: list[dict[str, Any]],
    quality_runs: dict[str, dict[str, Any]],
    power: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One row per (config x condition) cell."""
    measured = [r for r in records if r.get("kind") == "request" and r.get("phase") == "measure"]
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in measured:
        by_cell[(r["prompt_target_tokens"], r["gen_tokens"], r.get("concurrency", 1))].append(r)

    cells_meta = {(c["prompt_target_tokens"], c["gen_tokens"]): c for c in meta.get("cells", [])}
    memory = memory_block(meta)
    quality = quality_for(meta["config_id"], quality_runs)
    rows = []
    for (p_tokens, g_tokens, conc), rows_in_cell in sorted(by_cell.items()):
        cell = cells_meta.get((p_tokens, g_tokens), {})
        achieved_in = E.percentile(_vals(rows_in_cell, "input_tokens"), 50) \
            if _vals(rows_in_cell, "input_tokens") else None
        achieved_out = E.percentile(_vals(rows_in_cell, "output_tokens"), 50) \
            if _vals(rows_in_cell, "output_tokens") else None
        thr = throughput_block(rows_in_cell)
        mean_context = (achieved_in or p_tokens) + (achieved_out or g_tokens) / 2
        row = {
            "config_id": meta["config_id"],
            "run_id": meta["run_id"],
            "framework": meta["framework"],
            "representation": {k: v for k, v in meta["representation"].items() if k != "_artifact"},
            "condition": {
                "prompt_tokens_target": p_tokens,
                "prompt_tokens_achieved_p50": achieved_in,
                "gen_tokens_requested": g_tokens,
                "gen_tokens_achieved_p50": achieved_out,
                "concurrency": conc,
                "nvpmodel": meta.get("nvpmodel"),
                "execution_condition": meta.get("execution_condition"),
                "prompt_sha256": rows_in_cell[0].get("prompt_sha256"),
            },
            "latency": latency_block(rows_in_cell),
            "throughput": thr,
            "memory": memory,
            "energy": energy_block(cell, quality, power or []),
            "quality": quality,
            "thermal": {
                "max_c": _single((cell.get("thermal") or {}).get("max_c"), "degC", "not sampled"),
                "mean_c": _single((cell.get("thermal") or {}).get("mean_c"), "degC", "not sampled"),
                "time_to_throttle_s": E.not_collected(
                    "needs the 30-minute sustained run; this cell is a few minutes long and "
                    "never reached a throttle point", "s"),
                "sensor": (cell.get("thermal") or {}).get("sensor"),
            },
            "validity": {
                "cache_probe": cell.get("cache_probe"),
                "warmup_discarded": cell.get("warmup_discarded"),
                "runs_ok": cell.get("runs_ok"),
                "power_samples_in_window": cell.get("power_samples_in_window"),
                "confounds": meta.get("confounds"),
                "cold_start": meta.get("cold_start"),
            },
        }
        row["efficiency"] = efficiency_block(meta, memory, thr, mean_context)
        rows.append(row)
    return rows


def check_envelopes(rows: list[dict[str, Any]]) -> list[str]:
    """Every numeric leaf must be a legal envelope: a value with a unit, or a
    null with a reason. Run on every write, because the whole contract with
    `figures.py` - that it never has to guess - is worth nothing if one
    branch of one function can emit a bare float."""
    problems = []
    for i, row in enumerate(rows):
        for path, leaf in E.flatten_leaves(row):
            if leaf.get("value") is None and not leaf.get("_not_collected"):
                problems.append(f"row {i} {path}: null with no _not_collected reason")
            if leaf.get("value") is not None and not leaf.get("unit"):
                problems.append(f"row {i} {path}: value with no unit")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="combined results.json across every run dir (default: none, "
                         "each run dir still gets its own)")
    ap.add_argument("--quality", type=Path, default=None,
                    help="JSON mapping config_id -> quality block, from the accuracy suite")
    args = ap.parse_args()

    quality_runs = json.loads(args.quality.read_text()) if args.quality else {}
    all_rows: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        meta, records, power = load_run(run_dir)
        rows = build_rows(meta, records, quality_runs, power)
        doc = {
            "results_format_version": RESULTS_FORMAT_VERSION,
            "generated": datetime.datetime.now().astimezone().isoformat(),
            "source_runs": [meta["run_id"]],
            "rows": rows,
        }
        problems = check_envelopes(rows)
        if problems:
            doc["_envelope_problems"] = problems
            print(f"! {len(problems)} envelope problems in {run_dir.name}", file=sys.stderr)
            for p in problems[:10]:
                print(f"    {p}", file=sys.stderr)
        (run_dir / "results.json").write_text(json.dumps(doc, indent=2))
        print(f"{run_dir/'results.json'}: {len(rows)} rows")
        all_rows.extend(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "results_format_version": RESULTS_FORMAT_VERSION,
            "generated": datetime.datetime.now().astimezone().isoformat(),
            "source_runs": [d.name for d in args.run_dirs],
            "rows": all_rows,
        }, indent=2))
        print(f"{args.out}: {len(all_rows)} rows from {len(args.run_dirs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
