#!/usr/bin/env python3
"""Quality results -> the join file analysis/stats.py consumes.

Quality and performance are deliberately separate document types in this lab
(docs/note.md §5): they are measured by different scripts, at different
times, under different workloads, and merging them at measurement time would
make each unmeasurable without the other. They join at ANALYSIS time, on
`model_config_key`, and this is where that join is defined.

What it adds on the way through is the interval. The existing
`quality_result` documents record `overall_accuracy` and `n` and nothing
about uncertainty, so every table built from them so far has reported a
point estimate as if it were exact. Here each score is re-expressed as a
Wilson envelope from its own successes and n, which is what makes the
report's central rule checkable: do not report an ordering the intervals do
not support.

**Counts, not rates.** Wilson needs successes and n, and a rounded rate
times n is not the same number - at n=240 a rate rounded to three decimals
can be off by a fifth of an item, which moves a bound. Where
`outcomes.jsonl` exists the successes are counted from it directly; where
only a rate survives, the reconstruction is flagged in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
import envelope as E  # noqa: E402


def _suite_name(doc: dict[str, Any]) -> str:
    ds = doc.get("dataset") or {}
    return (ds.get("name") or "unknown").lower()


def _count_outcomes(run_dir: Path) -> dict[str, tuple[int, int]]:
    """(successes, n) per category, counted from the per-item log."""
    path = run_dir / "outcomes.jsonl"
    if not path.exists():
        return {}
    tally: dict[str, list[int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in (row.get("category") or "uncategorised", "__all__"):
            slot = tally.setdefault(key, [0, 0])
            slot[0] += bool(row.get("correct"))
            slot[1] += 1
    return {k: (v[0], v[1]) for k, v in tally.items()}


def load_quality(results_root: Path) -> dict[str, dict[str, Any]]:
    """Every quality_result under `results_root`, keyed by model config."""
    out: dict[str, dict[str, Any]] = {}
    for result in sorted(results_root.glob("*/result.json")):
        doc = json.loads(result.read_text())
        scores = doc.get("scores")
        if not scores or "overall_accuracy" not in scores:
            continue  # a benchmark_result, not a quality_result
        cfg = (doc.get("manifest") or {}).get("model_config_key")
        if not cfg:
            continue
        suite = _suite_name(doc)
        counted = _count_outcomes(result.parent)
        entry = out.setdefault(cfg, {"suites": {}, "_join": [], "_sources": []})
        entry["_sources"].append(str(result.parent.name))

        def envelope_for(name: str, rate: float | None, n: int | None,
                         counts: tuple[int, int] | None) -> dict[str, Any]:
            if counts:
                k, total = counts
                leaf = E.proportion_envelope(k, total)
                leaf["_successes"] = k
                return leaf
            if rate is None or not n:
                return E.not_collected(f"{name}: no score recorded")
            leaf = E.proportion_envelope(round(rate * n), n)
            leaf["_reconstructed_from_rate"] = (
                "successes were reconstructed as round(rate x n) because this result kept no "
                "per-item outcomes; the interval is therefore accurate to within one item"
            )
            return leaf

        entry["suites"][f"{suite}_overall"] = envelope_for(
            f"{suite} overall", scores.get("overall_accuracy"), scores.get("n"),
            counted.get("__all__"))
        for cat, cat_scores in (scores.get("by_category") or {}).items():
            entry["suites"][f"{suite}_{cat}"] = envelope_for(
                f"{suite} {cat}", cat_scores.get("accuracy"), cat_scores.get("n"),
                counted.get(cat) or (
                    (cat_scores["n_correct"], cat_scores["n"])
                    if "n_correct" in cat_scores and "n" in cat_scores else None))
        entry["_join"].append({
            "suite": suite, "result_id": doc.get("result_id"),
            "n_evaluated": (doc.get("dataset") or {}).get("n_evaluated"),
            "n_available": (doc.get("dataset") or {}).get("n_available"),
            "sampling_method": (doc.get("dataset") or {}).get("sampling_method"),
            "scorer": (doc.get("protocol") or {}).get("scorer"),
        })
    for entry in out.values():
        entry["accuracy"] = _headline(entry["suites"])
    return out


#: Which suite is THE accuracy when a config has several, in order of
#: preference. Tool-call judgment first: this lab's whole recommendation is a
#: tool-call recommendation, so joules per correct answer means joules per
#: correct tool-call JUDGMENT unless nothing measured one.
HEADLINE_ORDER = ("bfcl_overall", "bfcl_irrelevance", "mmlu_overall")


def _headline(suites: dict[str, Any]) -> dict[str, Any]:
    for name in HEADLINE_ORDER:
        if name in suites and E.is_measured(suites[name]):
            leaf = dict(suites[name])
            leaf["_headline_suite"] = name
            return leaf
    return E.not_collected(
        "no suite in " + ", ".join(HEADLINE_ORDER) + " was measured for this config",
        "fraction")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "v2" / "quality.json")
    args = ap.parse_args()
    data = load_quality(args.results_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2))
    print(f"wrote {args.out}: {len(data)} configs")
    for cfg, entry in sorted(data.items()):
        acc = entry["accuracy"]
        if E.is_measured(acc):
            lo, hi = acc["ci95"]
            print(f"  {cfg:<34} {acc['value']:.3f} [{lo:.3f}, {hi:.3f}] n={acc['n']} "
                  f"({acc.get('_headline_suite')})")
        else:
            print(f"  {cfg:<34} — {acc['_not_collected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
