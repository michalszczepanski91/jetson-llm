#!/usr/bin/env python3
"""results.json -> publication figures. No data lives in this file.

Every number a figure draws comes from `results.json`; this module holds
layout and encoding and nothing else. That is why `analysis/stats.py`
precomputes the inter-token ECDF as a quantile grid - a figure that had to
reopen a records.jsonl to draw a distribution would be a second, silently
divergent analysis path.

A figure whose inputs were never measured is SKIPPED with the reason
printed, not drawn with a gap or a zero. The brief's rule about never
inventing a value applies to figures at least as strongly as to files: a
bar chart with a missing bar reads as "that config scored nothing".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import envelope as E   # noqa: E402
import style           # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

GIB = 1024 ** 3


def _leaf(row: dict, path: str) -> dict[str, Any]:
    node: Any = row
    for part in path.split("."):
        node = (node or {}).get(part) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}


def _v(row: dict, path: str):
    return _leaf(row, path).get("value")


def _err(row: dict, path: str):
    """Asymmetric error bars in matplotlib's (below, above) form, from a
    ci95. Returned as a 2x1 array so a single point can carry it."""
    leaf = _leaf(row, path)
    v, ci = leaf.get("value"), leaf.get("ci95")
    if v is None or not ci:
        return None
    return [[max(0.0, v - ci[0])], [max(0.0, ci[1] - v)]]


def _by_config(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["config_id"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["condition"]["prompt_tokens_target"])
    return out


def _skip(name: str, reason: str) -> None:
    print(f"  SKIP {name}: {reason}")


# --------------------------------------------------------------------------
# 1. quality-energy Pareto
# --------------------------------------------------------------------------

def fig_quality_energy(rows: list[dict], out: Path) -> None:
    """Joules per CORRECT answer against accuracy, with the Pareto front.

    J per correct answer rather than J per token because that is the axis a
    deployment actually trades against quality: it folds in output-length
    drift (a quantized model that gets chattier costs more per turn at the
    same J/token) and the accuracy itself (a config that is cheap and wrong
    has to answer twice). The x axis is log because the configs this lab
    compares span more than an order of magnitude."""
    pts = [(r, _v(r, "energy.marginal.j_per_correct_answer"), _v(r, "quality.accuracy"))
           for r in rows]
    pts = [(r, x, y) for r, x, y in pts if x is not None and y is not None]
    if not pts:
        reason = _leaf(rows[0], "energy.marginal.j_per_correct_answer").get("_not_collected") \
            or "no config has both an energy and an accuracy measurement"
        return _skip(out.name, reason)

    fig, ax = plt.subplots(figsize=(style.SINGLE_COL, 2.6))
    for r, x, y in pts:
        ax.errorbar(x, y, yerr=_err(r, "quality.accuracy"), xerr=_err(r, "energy.marginal.j_per_correct_answer"),
                    marker=style.marker_for(r), color=style.color_for(r),
                    linestyle="none", markerfacecolor="none")
        style.direct_label(ax, x, y, style.label_for(r), style.color_for(r))
    # Pareto front: cheaper AND more accurate dominates.
    front = sorted(
        [(x, y, r) for r, x, y in pts
         if not any(x2 <= x and y2 >= y and (x2, y2) != (x, y) for _, x2, y2 in pts)],
        key=lambda t: t[0])
    if len(front) > 1:
        ax.step([f[0] for f in front], [f[1] for f in front], where="post",
                color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("marginal energy per correct answer (J, log scale)")
    ax.set_ylabel("accuracy")
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# 2. context scaling
# --------------------------------------------------------------------------

def fig_context_scaling(rows: list[dict], out: Path) -> None:
    """Two panels, because one TTFT at one prompt length hides the story.

    Left: TTFT and prefill rate against prompt length - the compute-bound
    regime. Right: decode rate against context - the bandwidth-bound one.
    Quantization moves these in opposite directions, so a single "speed"
    number for a config is a claim about neither."""
    by_cfg = _by_config(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.DOUBLE_COL, 2.5))
    ax1b = ax1.twinx()
    ax1b.grid(False)
    drew = False
    for cfg, series in by_cfg.items():
        xs = [s["condition"]["prompt_tokens_achieved_p50"] or s["condition"]["prompt_tokens_target"]
              for s in series]
        ttft = [_v(s, "latency.ttft_ms_p50") for s in series]
        pre = [_v(s, "throughput.prefill_tok_s_p50") for s in series]
        dec = [_v(s, "throughput.decode_tok_s_p50") for s in series]
        c, m = style.color_for(series[0]), style.marker_for(series[0])
        if any(t is not None for t in ttft):
            drew = True
            lo = [_err(s, "latency.ttft_ms_p50") for s in series]
            ax1.errorbar(xs, ttft, marker=m, color=c, markerfacecolor="none",
                         yerr=[[e[0][0] if e else 0 for e in lo],
                               [e[1][0] if e else 0 for e in lo]])
            style.direct_label(ax1, xs[-1], ttft[-1], style.label_for(series[0]), c)
        if any(p is not None for p in pre):
            ax1b.plot(xs, pre, marker=m, color=c, linestyle=":", alpha=0.55,
                      markersize=3, markerfacecolor="none")
        if any(d is not None for d in dec):
            de = [_err(s, "throughput.decode_tok_s_p50") for s in series]
            ax2.errorbar(xs, dec, marker=m, color=c, markerfacecolor="none",
                         yerr=[[e[0][0] if e else 0 for e in de],
                               [e[1][0] if e else 0 for e in de]])
            style.direct_label(ax2, xs[-1], dec[-1], style.label_for(series[0]), c)
    if not drew:
        plt.close(fig)
        return _skip(out.name, "no TTFT measurements in any row")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("prompt length (tokens, achieved)")
    ax1.set_yscale("log")
    ax1.set_ylabel("TTFT p50 (ms, solid)")
    ax1b.set_ylabel("prefill rate (tok/s, dotted)")
    ax1.set_title("prefill: compute-bound", loc="left")
    ax2.set_ylabel("decode rate p50 (tok/s)")
    ax2.set_title("decode: bandwidth-bound", loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# 3. memory composition
# --------------------------------------------------------------------------

def fig_memory(rows: list[dict], out: Path) -> None:
    """Stacked footprint against the board's whole pool.

    The bar that matters on a unified-memory SoC is the white one at the
    top: what a co-resident perception stack would actually have found free.
    Weights and KV are drawn as MEASURED-RESIDENT, with the runtime's
    RESERVATION marked separately where the two differ - on this board they
    differ by more than 20GB, and only one of the two answers the
    co-tenancy question."""
    by_cfg = _by_config(rows)
    first = {cfg: series[0] for cfg, series in by_cfg.items()}
    usable = {c: r for c, r in first.items() if _v(r, "memory.peak_attributable_bytes")}
    if not usable:
        return _skip(out.name, "no measured peak memory in any row")

    fig, ax = plt.subplots(figsize=(style.SINGLE_COL, 2.6))
    names = list(usable)
    for i, cfg in enumerate(names):
        r = usable[cfg]
        total = (_v(r, "memory.system_total_bytes") or 0) / GIB
        w = (_v(r, "memory.weights_bytes_on_device") or 0) / GIB
        kv = (_v(r, "memory.kv_cache_bytes_resident_at_peak") or 0) / GIB
        attributable = (_v(r, "memory.peak_attributable_bytes") or 0) / GIB
        overhead = max(0.0, attributable - w - kv)
        baseline = max(0.0, (_v(r, "memory.peak_device_bytes") or 0) / GIB - attributable)
        c = style.color_for(r)
        bottom = 0.0
        for height, color, hatch, lbl in (
            (baseline, "0.85", None, "OS + page cache"),
            (w, c, None, "weights"),
            (kv, c, "///", "KV (resident)"),
            (overhead, c, "...", "runtime overhead"),
        ):
            ax.bar(i, height, bottom=bottom, color=color, edgecolor="white",
                   linewidth=0.4, hatch=hatch, label=lbl if i == 0 else None)
            bottom += height
        ax.bar(i, max(0.0, total - bottom), bottom=bottom, color="white",
               edgecolor="0.6", linewidth=0.4, label="free" if i == 0 else None)
        reserved = _v(r, "memory.kv_cache_bytes_reserved")
        if reserved:
            y = baseline + w + reserved / GIB
            ax.plot([i - 0.42, i + 0.42], [y, y], color="0.2", linewidth=0.9, linestyle="--")
            if i == 0:
                style.direct_label(ax, i + 0.42, y, "KV reserved", "0.2", dx=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([style.label_for(usable[c]) for c in names], rotation=20, ha="right")
    ax.set_ylabel("unified memory (GiB)")
    ax.legend(loc="upper left", ncol=2, fontsize=6)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# 4. latency tail
# --------------------------------------------------------------------------

def fig_latency_tail(rows: list[dict], out: Path) -> None:
    """ECDF of inter-token latency, with p50/p90/p99 marked.

    A mean would hide the thing a listener actually notices. The tail is the
    figure: a steady 25 tok/s is a better orchestrator than a stuttering 40,
    and only the shape of this curve to the right of p90 distinguishes
    them."""
    fig, ax = plt.subplots(figsize=(style.SINGLE_COL, 2.4))
    drew = False
    for r in rows:
        ecdf = _leaf(r, "latency.inter_token_ms_ecdf")
        if not ecdf.get("ms"):
            continue
        drew = True
        c = style.color_for(r)
        ax.plot(ecdf["ms"], [q / 100 for q in ecdf["q"]], color=c, linewidth=1.0,
                alpha=0.9)
        for p, ls in ((50, ":"), (90, "--"), (99, "-.")):
            v = _v(r, f"latency.inter_token_ms_p{p}")
            if v is not None:
                ax.plot([v], [p / 100], marker=style.marker_for(r), color=c,
                        markersize=3.5, markerfacecolor="none")
        label = f"{style.label_for(r)} in{r['condition']['prompt_tokens_target']}"
        style.direct_label(ax, ecdf["ms"][-1], 1.0, label, c)
    if not drew:
        plt.close(fig)
        return _skip(out.name, "no inter-token ECDF in any row")
    for p in (0.5, 0.9, 0.99):
        ax.axhline(p, color="0.8", linewidth=0.4, zorder=0)
        ax.annotate(f"p{int(p*100)}", (ax.get_xlim()[0], p), fontsize=6, color="0.5",
                    va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("inter-token latency (ms, log scale)")
    ax.set_ylabel("cumulative fraction of gaps")
    ax.set_ylim(0, 1.02)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# 5. accuracy with Wilson intervals
# --------------------------------------------------------------------------

def fig_accuracy(rows: list[dict], out: Path) -> None:
    """Per-suite accuracy with Wilson intervals and significance marks.

    The intervals are the point, not decoration. At n=200 a 67% score
    carries about +-6.5pp, which is wider than most of the orderings this
    lab has been tempted to report; a bar without its interval invites
    exactly the claim the data does not support. Configs whose intervals
    overlap the best config's are marked, so the reader can see which part
    of the ranking is real."""
    by_cfg = _by_config(rows)
    suites: dict[str, list[tuple[str, dict, dict]]] = {}
    for cfg, series in by_cfg.items():
        for suite, leaf in (_leaf(series[0], "quality.suites") or {}).items():
            if E.is_measured(leaf):
                suites.setdefault(suite, []).append((cfg, series[0], leaf))
    if not suites:
        reason = _leaf(rows[0], "quality.accuracy").get("_not_collected") or "no quality suites"
        return _skip(out.name, reason)

    fig, axes = plt.subplots(1, len(suites), figsize=(style.DOUBLE_COL, 2.4), sharey=True)
    axes = [axes] if len(suites) == 1 else list(axes)
    for ax, (suite, entries) in zip(axes, sorted(suites.items())):
        entries.sort(key=lambda e: e[2]["value"], reverse=True)
        best = entries[0][2]
        for i, (cfg, row, leaf) in enumerate(entries):
            yerr = [[leaf["value"] - leaf["ci95"][0]], [leaf["ci95"][1] - leaf["value"]]] \
                if leaf.get("ci95") else None
            ax.barh(i, leaf["value"], color=style.color_for(row), alpha=0.85, height=0.6)
            ax.errorbar(leaf["value"], i, xerr=yerr, color="0.2", linestyle="none", linewidth=0.8)
            if i and E.intervals_overlap(leaf, best):
                # n.s. against the leader - the ordering below this mark is
                # not supported by the intervals and must not be reported.
                ax.annotate("n.s.", (leaf["ci95"][1], i), fontsize=6, color="0.35",
                            xytext=(3, 0), textcoords="offset points", va="center")
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([style.label_for(r) for _, r, _ in entries], fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f"{suite}  (n={best.get('n')})", loc="left")
        ax.set_xlabel("accuracy")
        ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# 6. latency-power Pareto across nvpmodel levels
# --------------------------------------------------------------------------

def fig_latency_power(rows: list[dict], out: Path) -> None:
    """One operating point is not a power story; the Pareto curve is.

    Needs the same core sweep repeated at each nvpmodel level. With one
    level measured there is no curve to draw, and drawing a single point as
    if it were one would be the exact overclaim this figure exists to
    prevent."""
    frameworks: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        frameworks.setdefault(r["framework"], {}).setdefault(
            r["condition"].get("nvpmodel") or "unknown", []).append(r)
    levels = {lvl for fw in frameworks.values() for lvl in fw}
    if len(levels) < 2:
        return _skip(out.name,
                     f"only one nvpmodel level measured ({levels or 'none'}); a latency-power "
                     "Pareto needs the core sweep repeated at each level")

    fig, axes = plt.subplots(1, len(frameworks), figsize=(style.DOUBLE_COL, 2.4),
                             sharey=True, squeeze=False)
    for ax, (fw, by_level) in zip(axes[0], sorted(frameworks.items())):
        pts = []
        for lvl, rs in sorted(by_level.items()):
            p = [_v(r, "energy.absolute.power_mw") for r in rs]
            t = [_v(r, "latency.turn_ms_p90") for r in rs]
            p = [x for x in p if x is not None]
            t = [x for x in t if x is not None]
            if p and t:
                pts.append((sum(p) / len(p) / 1000, sum(t) / len(t), lvl, rs[0]))
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="0.5", linewidth=0.8, zorder=0)
        for x, y, lvl, r in pts:
            ax.plot(x, y, marker=style.marker_for(r), color=style.color_for(r),
                    markerfacecolor="none", linestyle="none")
            style.direct_label(ax, x, y, lvl, style.color_for(r))
        ax.set_title(style.FRAMEWORK_LABEL.get(fw, fw), loc="left")
        ax.set_xlabel("board power (W)")
    axes[0][0].set_ylabel("turn time p90 (ms)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# 7. framework x representation heatmap
# --------------------------------------------------------------------------

def fig_heatmap(rows: list[dict], metric: str, out: Path) -> None:
    """Framework x representation, normalised to the FP16 baseline.

    The hatched cells are the reason this figure is drawn as a matrix at all:
    a cell is marked NON-COMPARABLE whenever the two configs it would
    compare belong to different `comparability_class`es, so nothing on this
    page can be read as "llama.cpp's INT4 beats Edge-LLM's INT4" when the
    two INT4s are a mixed K-quant at 5.0 effective bits and a uniform
    grouped scheme at 4.3."""
    import numpy as np
    cells: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["framework"], (r["representation"] or {}).get("comparability_class"))
        if key not in cells and _v(r, metric) is not None:
            cells[key] = r
    fws = sorted({k[0] for k in cells})
    reps = sorted({k[1] for k in cells})
    if len(fws) < 2 and len(reps) < 2:
        return _skip(out.name,
                     f"only {len(fws)} framework(s) x {len(reps)} representation(s) measured; "
                     "a normalised matrix needs at least a second row or column")

    baselines = {fw: _v(cells[(fw, "fp16_baseline")], metric)
                 for fw in fws if (fw, "fp16_baseline") in cells}
    grid = np.full((len(reps), len(fws)), np.nan)
    for (fw, rep), row in cells.items():
        base = baselines.get(fw)
        if base:
            grid[reps.index(rep)][fws.index(fw)] = _v(row, metric) / base

    fig, ax = plt.subplots(figsize=(style.SINGLE_COL, 0.5 * len(reps) + 1.2))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
    for i in range(len(reps)):
        for j in range(len(fws)):
            if np.isnan(grid[i][j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           hatch="xxx", edgecolor="0.7", linewidth=0))
                ax.text(j, i, "n/c", ha="center", va="center", fontsize=6, color="0.4")
            else:
                ax.text(j, i, f"{grid[i][j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(grid[i][j] - 1) > 0.35 else "black")
    ax.set_xticks(range(len(fws)))
    ax.set_xticklabels([style.FRAMEWORK_LABEL.get(f, f) for f in fws], rotation=20, ha="right")
    ax.set_yticks(range(len(reps)))
    ax.set_yticklabels(reps, fontsize=6)
    ax.set_title(f"{metric}, relative to each framework's own FP16", loc="left", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("x FP16 baseline", fontsize=6)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------

FIGURES: list[tuple[str, Callable]] = [
    ("fig1_quality_energy_pareto.pdf", fig_quality_energy),
    ("fig2_context_scaling.pdf", fig_context_scaling),
    ("fig3_memory_composition.pdf", fig_memory),
    ("fig4_latency_tail.pdf", fig_latency_tail),
    ("fig5_accuracy_wilson.pdf", fig_accuracy),
    ("fig6_latency_power_pareto.pdf", fig_latency_power),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "output" / "figures")
    args = ap.parse_args()
    style.apply()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.results.read_text())["rows"]
    print(f"{len(rows)} rows from {args.results}")
    for name, fn in FIGURES:
        try:
            fn(rows, args.out_dir / name)
        except Exception as exc:  # noqa: BLE001 - one broken figure must not kill the set
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    try:
        fig_heatmap(rows, "latency.turn_ms_p90", args.out_dir / "fig7_heatmap_turn_p90.pdf")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR fig7: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
