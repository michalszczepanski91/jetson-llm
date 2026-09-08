#!/usr/bin/env python3
"""Collate this lab's Thor result JSON into the comparison tables that go into
docs/thor-framework-comparison.md.

Exists because those tables were hand-assembled from a dozen separate result
files during the first (shared-box) pass, which is both slow and a
transcription-error risk once there are two passes to compare - the shared-box
numbers (output/*_sharedbox.json) and the exclusive-window numbers of record
(output/*.json, no suffix).

Reads whatever is present and says PENDING for what is not, so it can be run
mid-campaign to see how far along things are.

Usage:
    uv run python scripts/collate_thor_results.py            # both passes
    uv run python scripts/collate_thor_results.py --pass clean
"""

import argparse
import json
from pathlib import Path

_OUT = Path(__file__).resolve().parent.parent / "output"

# (label, config key) in the order the tables should read.
_FRAMEWORKS_1_5B = [
    ("Edge-LLM", "1.5b-fp16-edgellm-thor"),
    ("vLLM", "1.5b-fp16-vllm-thor"),
    ("llama.cpp", "1.5b-fp16-llamacpp-thor"),
]
_FRAMEWORKS_7B = [
    ("Edge-LLM", "7b-fp16-edgellm-thor"),
    ("vLLM", "7b-fp16-vllm-thor"),
    ("llama.cpp", "7b-fp16-llamacpp-thor"),
]
_SIZES_EDGELLM = [
    ("Qwen2.5-1.5B", "1.5b-fp16-edgellm-thor"),
    ("Qwen2.5-7B", "7b-fp16-edgellm-thor"),
    ("Qwen2.5-14B", "14b-fp16-edgellm-thor"),
]


def _load(name: str):
    """Result files are a list of runs; the last one is the run of record."""
    path = _OUT / name
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return blob[-1] if isinstance(blob, list) else blob


def _suffix(pass_name: str) -> str:
    return "_sharedbox" if pass_name == "shared" else ""


def _pct(x, places: int = 0):
    # MMLU gets a decimal place: its whole point is comparing backends against
    # each other on the same model, where the spread is a few points (38.5 vs
    # 40.5 vs 42.0) and rounding to whole percent throws the signal away.
    return f"{x * 100:.{places}f}%" if isinstance(x, (int, float)) else "–"


def _row(cfg: str, pass_name: str) -> dict:
    s = _suffix(pass_name)
    bfcl = _load(f"tool_calling_{cfg}.json")          # accuracy is pass-independent
    mmlu = _load(f"mmlu_{cfg}.json")
    stream = _load(f"streaming_{cfg}{s}.json")
    bench = _load(f"benchmark_{cfg}{s}.json")
    short = _load(f"benchmark_{cfg}_short32{s}.json")

    out = {}
    if bfcl:
        out["simple"] = _pct(bfcl.get("simple_accuracy"))
        out["irrelevance"] = _pct(bfcl.get("irrelevance_accuracy"))
        out["overall"] = _pct(bfcl.get("overall_accuracy"))
    if mmlu:
        out["mmlu"] = _pct(mmlu.get("accuracy"), places=1)
    if stream:
        out["ttft"] = f"{stream['ttft_seconds']['p50'] * 1000:.0f} ms"
        out["toks"] = f"{stream['tokens_per_sec']['p50']:.1f}"
    if bench:
        out["lat_p50"] = f"{bench['latency_ms']['p50']:.0f} ms"
        out["lat_p95"] = f"{bench['latency_ms']['p95']:.0f} ms"
        out["cold"] = f"{bench['cold_start_ms'] / 1000:.1f} s"
        power = (bench.get("power_mw") or {}).get("vin_sys_5v0_mw") or {}
        if power.get("avg"):
            out["power"] = f"{power['avg'] / 1000:.1f} W"
        # A run that never reached thermal steady state must not be reported as
        # if it had - the harness knows, so surface it rather than hiding it.
        if bench.get("warmup_reached_steady_state") is False:
            out["lat_p50"] += " ⚠"
            out["lat_p95"] += " ⚠"
    if short:
        out["turn32"] = f"{short['latency_ms']['p50']:.0f} ms"
        if short.get("warmup_reached_steady_state") is False:
            out["turn32"] += " ⚠"
    return out


def _table(title: str, rows: list, cols: list, pass_name: str) -> str:
    headers = {
        "simple": "BFCL simple", "irrelevance": "BFCL irrelevance", "overall": "BFCL overall",
        "mmlu": "MMLU", "ttft": "TTFT p50", "toks": "tok/s p50",
        "lat_p50": "latency p50", "lat_p95": "latency p95", "cold": "cold start",
        "power": "power", "turn32": "32-tok turn",
    }
    lines = [f"\n### {title}\n"]
    lines.append("| | " + " | ".join(headers[c] for c in cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for label, cfg in rows:
        data = _row(cfg, pass_name)
        if not data:
            lines.append(f"| **{label}** |" + " PENDING |" * len(cols))
            continue
        lines.append(f"| **{label}** | " + " | ".join(data.get(c, "–") for c in cols) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="pass_name", choices=["shared", "clean", "both"], default="both")
    args = p.parse_args()

    passes = ["shared", "clean"] if args.pass_name == "both" else [args.pass_name]
    for pass_name in passes:
        label = ("SHARED BOX (indicative — another user's container resident)"
                 if pass_name == "shared" else
                 "EXCLUSIVE BOX (numbers of record)")
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print(_table(
            "Framework comparison, model fixed at Qwen2.5-1.5B-Instruct FP16",
            _FRAMEWORKS_1_5B,
            ["overall", "simple", "irrelevance", "mmlu", "ttft", "toks", "lat_p95", "cold", "power"],
            pass_name,
        ))
        print(_table(
            "Framework comparison at 7B",
            _FRAMEWORKS_7B,
            ["overall", "simple", "irrelevance", "mmlu", "ttft", "toks", "lat_p95", "cold"],
            pass_name,
        ))
        print(_table(
            "Size sweep, framework fixed at Edge-LLM",
            _SIZES_EDGELLM,
            ["overall", "irrelevance", "mmlu", "ttft", "toks", "turn32", "cold"],
            pass_name,
        ))
    print("\n⚠ = the harness reported warmup_reached_steady_state=false; re-run before reporting.\n")


if __name__ == "__main__":
    main()
