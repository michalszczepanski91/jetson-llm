#!/usr/bin/env python3
"""Human-readable rendering of a results.json - the terminal counterpart to
analysis/figures.py, reading exactly the same file and inventing nothing.

Exists so that "show me the numbers" and "draw the figure" cannot disagree:
both read `results.json`, neither reaches back to a records.jsonl, and an
envelope with no value renders as its `_not_collected` reason rather than as
a blank or a zero."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
import envelope as E  # noqa: E402


def fmt(leaf: Any, scale: float = 1.0, digits: int = 1, width: int = 22) -> str:
    """Render one envelope as `value [lo, hi]`, or as why it is missing."""
    if not isinstance(leaf, dict):
        return str(leaf)
    if leaf.get("value") is None:
        reason = leaf.get("_not_collected", "not collected")
        return f"— ({reason[:width]}…)" if len(reason) > width else f"— ({reason})"
    v = leaf["value"] * scale
    out = f"{v:,.{digits}f}"
    if leaf.get("ci95"):
        lo, hi = (c * scale for c in leaf["ci95"])
        out += f" [{lo:,.{digits}f}, {hi:,.{digits}f}]"
    return out


GIB = 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--section", default="all",
                    choices=["all", "memory", "latency", "energy", "representation", "validity"])
    args = ap.parse_args()
    doc = json.loads(args.results.read_text())
    rows = doc["rows"]
    if not rows:
        print("no rows")
        return 1
    sec = args.section

    if sec in ("all", "representation"):
        r = rows[0]["representation"]
        print("═" * 100)
        print(f"REPRESENTATION  {rows[0]['config_id']}  ({rows[0]['framework']})")
        print("═" * 100)
        w, a, kv = r["weights"], r["activations"], r["kv_cache"]
        print(f"  declared label          {r['declared_label']}")
        print(f"  comparability class     {r['comparability_class']}")
        print(f"  weights                 dtype={w['dtype']}  method={w['method']}  "
              f"granularity={w['granularity']}  group_size={w['group_size']}")
        print(f"  activations             dtype={a['dtype']}   [{a['_source']}]")
        print(f"  kv cache                dtype={kv['dtype']}   [{kv['_source']}]")
        print(f"  modules kept FP         {', '.join(r['modules_kept_fp']) or 'none'}")
        print(f"  modules quantized       {', '.join(r['modules_quantized']) or 'none'}")
        print(f"  module storage types    {r['module_storage_types']}")
        print(f"  calibration             {r['calibration'].get('dataset') or r['calibration'].get('_not_collected')}")
        print(f"  param count             {r['param_count']:,}   ({r['param_count_method']})")
        print(f"  artifact bytes          {r['artifact_bytes']:,}")
        print(f"  EFFECTIVE BITS/WEIGHT   {r['effective_bits_per_weight']}")

    if sec in ("all", "memory"):
        m = rows[0]["memory"]
        print("\n" + "═" * 100)
        print("MEMORY  (one footprint per config; GiB unless stated)")
        print("═" * 100)
        print(f"  weights on device       {fmt(m['weights_bytes_on_device'], 1/GIB, 2)} GiB")
        print(f"    source                {m['weights_bytes_source']}")
        print(f"  KV bytes / token        {fmt(m['kv_cache_bytes_per_token'], 1, 0)} B")
        print(f"    formula               {m['kv_cache_bytes_per_token_formula']}")
        print(f"  KV reserved by runtime  {fmt(m['kv_cache_bytes_reserved'], 1/GIB, 2)} GiB")
        print(f"  PEAK DEVICE MEMORY      {fmt(m['peak_device_bytes'], 1/GIB, 2)} GiB")
        print(f"    source                {m.get('peak_device_source')}")
        print(f"  peak attributable       {fmt(m['peak_attributable_bytes'], 1/GIB, 2)} GiB")
        print(f"  runtime overhead        {fmt(m['runtime_overhead_bytes'], 1/GIB, 2)} GiB")
        print(f"  FREE LEFT OVER          {fmt(m['free_bytes_remaining'], 1/GIB, 2)} GiB "
              f"of {fmt(m['system_total_bytes'], 1/GIB, 2)} GiB total")
        if m.get("component_breakdown_bytes"):
            for k, v in m["component_breakdown_bytes"].items():
                print(f"      {k:<28} {v/GIB:>7.2f} GiB")
        print(f"  MAX CONTEXT measured    {fmt(m['max_context_measured'], 1, 0)} tokens")
        print(f"    limiting factor       {m['max_context_limiting_factor']}")
        print(f"    configured            {fmt(m['max_context_configured'], 1, 0)} tokens")
        if m.get("breakdown_note"):
            print(f"  ! {m['breakdown_note']}")
        if m.get("board_accounting_caveat"):
            print(f"  ! {m['board_accounting_caveat']}")

    if sec in ("all", "latency"):
        print("\n" + "═" * 100)
        print("LATENCY & THROUGHPUT  by prompt length   (p50 [95% CI], bootstrap n as shown)")
        print("═" * 100)
        hdr = f"  {'prompt':>7} {'n':>4} │ {'TTFT p50':>20} {'TTFT p90':>20} {'TTFT p99':>20}"
        print(hdr)
        for r in rows:
            c, l = r["condition"], r["latency"]
            print(f"  {c['prompt_tokens_achieved_p50'] or c['prompt_tokens_target']:>7.0f} "
                  f"{l['n_ok']:>4} │ {fmt(l['ttft_ms_p50']):>20} {fmt(l['ttft_ms_p90']):>20} "
                  f"{fmt(l['ttft_ms_p99']):>20}")
        print(f"\n  {'prompt':>7}      │ {'prefill tok/s':>22} {'decode tok/s':>22} {'turn p90 ms':>20}")
        for r in rows:
            c, t, l = r["condition"], r["throughput"], r["latency"]
            print(f"  {c['prompt_tokens_achieved_p50'] or c['prompt_tokens_target']:>7.0f}      │ "
                  f"{fmt(t['prefill_tok_s_p50']):>22} {fmt(t['decode_tok_s_p50'], 1, 2):>22} "
                  f"{fmt(l['turn_ms_p90']):>20}")
        print(f"\n  {'prompt':>7}      │ {'ITL p50':>16} {'ITL p90':>16} {'ITL p99':>16} {'ITL stdev':>16} {'MBU':>14}")
        for r in rows:
            c, l, e = r["condition"], r["latency"], r["efficiency"]
            print(f"  {c['prompt_tokens_achieved_p50'] or c['prompt_tokens_target']:>7.0f}      │ "
                  f"{fmt(l['inter_token_ms_p50']):>16} {fmt(l['inter_token_ms_p90']):>16} "
                  f"{fmt(l['inter_token_ms_p99']):>16} {fmt(l['inter_token_ms_stdev']):>16} "
                  f"{fmt(e.get('decode_mbu', {}), 100, 1):>14}")

    if sec in ("all", "energy"):
        print("\n" + "═" * 100)
        print("ENERGY  (idle measured per cell, server loaded and resident, immediately before)")
        print("═" * 100)
        print(f"  {'prompt':>7} │ {'P_idle W':>9} {'P_gen W':>9} │ {'ABS J/turn':>11} {'ABS J/tok':>10} │"
              f" {'MRG J/turn':>11} {'MRG J/tok':>10} │ {'J/correct':>12}")
        for r in rows:
            c, e = r["condition"], r["energy"]
            a, mg = e.get("absolute", {}), e.get("marginal", {})
            def g(block, k, scale=1.0, d=3):
                leaf = block.get(k)
                return fmt(leaf, scale, d, width=8) if isinstance(leaf, dict) else "—"
            print(f"  {c['prompt_tokens_achieved_p50'] or c['prompt_tokens_target']:>7.0f} │ "
                  f"{g(e,'idle_power_mw',1/1000,2):>9} {g(e,'generation_power_mw',1/1000,2):>9} │ "
                  f"{g(a,'j_per_turn',1,2):>11} {g(a,'j_per_output_token',1,4):>10} │ "
                  f"{g(mg,'j_per_turn',1,2):>11} {g(mg,'j_per_output_token',1,4):>10} │ "
                  f"{g(mg,'j_per_correct_answer',1,2):>12}")
        print(f"\n  absolute = {rows[0]['energy']['conventions']['absolute']}")
        print(f"  marginal = {rows[0]['energy']['conventions']['marginal']}")

    if sec in ("all", "validity"):
        print("\n" + "═" * 100)
        print("VALIDITY")
        print("═" * 100)
        v = rows[0]["validity"]
        cf = v.get("confounds") or {}
        for name, entry in (cf.get("verified") or {}).items():
            print(f"  [verified] {name:<42} {entry.get('value')}")
        for name, entry in (cf.get("asserted") or {}).items():
            print(f"  [asserted] {name:<42} {entry.get('value')}"
                  f"{'  ⚠ ' + entry['caveat'][:60] if entry.get('caveat') else ''}")
        for r in rows:
            cp = r["validity"].get("cache_probe")
            tag = f"  cache probe in={r['condition']['prompt_tokens_target']:>5}: "
            if not cp:
                # "no probe data" and "reuse detected" are different facts and
                # must not render the same way - the second is an alarm.
                print(tag + "— (no probe recorded for this cell)")
                continue
            print(tag + f"1st {cp.get('first_ttft_ms')}ms → 2nd {cp.get('second_ttft_ms')}ms "
                  f"(ratio {cp.get('ratio_second_over_first')}) — "
                  f"{'OK, no reuse' if cp.get('prefix_caching_appears_disabled') else 'REUSE DETECTED'}")
        for name, text in (cf.get("not_yet_verified") or {}).items():
            print(f"  [NOT verified] {name}:\n      {text}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
