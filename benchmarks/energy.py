"""Energy per *task*, and the idle baseline that makes "marginal" mean
something.

`benchmarks/manifest.py:energy_block()` already turns average rail power
over a window into joules and joules per output token. Two things it cannot
express, and both change conclusions:

**1. Per token is the wrong denominator for a decision.** A quantized model
that answers the same question in 40% more tokens is not 40% cheaper per
token in any sense a battery cares about; it costs more. The quantity a
deployment budgets is joules per *completed turn* - and, once accuracy
enters, joules per *correct* answer, which folds output-length drift and
quality into one number and can reverse a ranking that J/token puts the
other way up. J/token stays (it is the right axis for a decode-efficiency
question) but it is no longer the only one.

**2. There is no idle baseline.** An absolute board-power reading includes
the SoC's floor, the fans, the display, the rest of the kernel. On this
board that floor is a large fraction of the total at 7B, so an absolute
J/token silently compares "the model" against "the model plus the board"
depending on how long the window was. Both numbers are legitimate and they
serve different decisions, so both are reported and each is labelled:

    absolute  P_gen            - budgets a power envelope and a thermal
                                 design. The question it answers is "can
                                 this board sustain this workload".
    marginal  P_gen - P_idle   - budgets the incremental cost of giving the
                                 orchestrator turn to this config on a board
                                 that is already powered and already running
                                 the perception stack. The question it
                                 answers is "what does this choice cost me".

The idle reading is taken with the server loaded and resident but serving
nothing, immediately before the measurement window, in the same power state
(same nvpmodel, same fan profile, same thermal neighbourhood). Not at the
start of the campaign, and not with the server stopped: a cold board idles
lower than a warm one, and an unloaded board idles lower than one holding
15GB of weights, so either shortcut would inflate every marginal figure.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

#: Which decision each convention serves. Carried into the result so a
#: figure's axis label can be generated from the data rather than typed by
#: hand next to a number that might be the other one.
CONVENTIONS = {
    "absolute": (
        "P_gen: total board rail power during generation. Budgets the power "
        "envelope and thermal design - 'can this board sustain this workload'."
    ),
    "marginal": (
        "P_gen - P_idle, idle measured with this same server loaded and resident "
        "but serving nothing. Budgets the incremental cost of the orchestrator turn "
        "on a board that is already on - 'what does this choice cost me'."
    ),
}


def measure_idle(
    sampler,
    rails: Sequence[str],
    *,
    seconds: float = 20.0,
    settle_s: float = 3.0,
) -> dict[str, Any]:
    """Average rail power with the server up and idle.

    `settle_s` is discarded before the window starts: the board is still
    coming down from whatever request preceded this call, and including that
    decay would bias the idle figure upward, which biases every marginal
    energy figure downward - the flattering direction, which is the one this
    lab treats as a defect (see `runner.py`'s prompt-uniqueness note for the
    same reasoning applied to TTFT)."""
    from harness import _rail_stats  # local import: stdlib-only module boundary

    time.sleep(settle_s)
    t0 = time.monotonic()
    time.sleep(seconds)
    t1 = time.monotonic()
    samples = sampler.samples_between(t0, t1)
    stats = {r: _rail_stats(samples, r) for r in rails}
    present = {r: s for r, s in stats.items() if s}
    total_mw = sum(s["avg"] for s in present.values()) if present else None
    temps = [t for t in (s.get("tj_temp_c", s.get("gpu_temp_c")) for s in samples) if t is not None]
    return {
        "measured": bool(present),
        # Monotonic bounds, kept so analysis/stats.py can re-slice power.jsonl
        # and put a real interval on the power figure. Without them the mean
        # power over a window is a single number with no computable spread,
        # and "error bars on everything" quietly excepts energy.
        "window_t0": t0,
        "window_t1": t1,
        "window_s": round(t1 - t0, 3),
        "settle_discarded_s": settle_s,
        "n_samples": len(samples),
        "rails_mw": {r: s for r, s in present.items()},
        "rails_included": list(present),
        "total_power_mw": round(total_mw, 1) if total_mw is not None else None,
        "tj_temp_c_mean": round(sum(temps) / len(temps), 2) if temps else None,
        "server_state": "loaded and resident, serving no requests",
        "_not_collected": None if present else "no power rails present in the idle window",
    }


def task_energy(
    *,
    rail_stats: dict[str, dict[str, float] | None],
    rails: Sequence[str],
    window_s: float,
    idle: dict[str, Any],
    n_requests: int,
    total_output_tokens: int | None,
    n_tool_calls_completed: int | None = None,
) -> dict[str, Any]:
    """Absolute and marginal energy for one measurement window, expressed in
    every denominator the decision actually uses.

    `n_tool_calls_completed` is the count of turns in THIS window that
    produced a structurally valid tool call. When the workload carried no
    tools it is None, and the per-tool-call figures come back as honest
    absences rather than as the per-request figure wearing a different name.
    Joules per *correct* answer is deliberately not computed here at all: it
    needs an accuracy denominator that comes from a different run, and it is
    assembled at analysis time by joining on config_id - see
    `analysis/stats.py`."""
    present = {r: rail_stats.get(r) for r in rails if rail_stats.get(r)}
    block: dict[str, Any] = {
        "rails_included": list(present),
        "rails_mw": {r: s for r, s in rail_stats.items() if s},
        "measurement_window_s": round(window_s, 3),
        "idle": idle,
        "conventions": CONVENTIONS,
    }
    if not present or window_s <= 0:
        block["_not_collected"] = "no power rails sampled in the measurement window"
        return block

    p_gen_mw = sum(s["avg"] for s in present.values())
    block["generation_power_mw"] = round(p_gen_mw, 1)

    variants: dict[str, float] = {"absolute": p_gen_mw}
    idle_mw = idle.get("total_power_mw") if idle.get("measured") else None
    if idle_mw is not None:
        # Rail sets must match for a subtraction to be meaningful: subtracting
        # an idle figure built from two rails off a generation figure built
        # from three is not a marginal power, it is an arithmetic accident.
        if set(idle.get("rails_included") or []) != set(present):
            block["marginal_skipped_reason"] = (
                f"idle rails {sorted(idle.get('rails_included') or [])} differ from generation "
                f"rails {sorted(present)}; a subtraction across different rail sets is not a "
                "marginal power"
            )
        else:
            variants["marginal"] = max(0.0, p_gen_mw - idle_mw)
            block["idle_power_mw"] = round(idle_mw, 1)
            if p_gen_mw <= idle_mw:
                block["marginal_warning"] = (
                    "generation power did not exceed the idle baseline - the marginal figure "
                    "is clamped at zero and should be read as 'below this measurement's noise "
                    "floor', not as zero energy"
                )
    else:
        block["marginal_skipped_reason"] = "no idle baseline was measured for this cell"

    for name, mw in variants.items():
        joules = mw / 1000.0 * window_s
        entry: dict[str, Any] = {
            "power_mw": round(mw, 1),
            "energy_j": round(joules, 4),
            "convention": CONVENTIONS[name],
        }
        if n_requests > 0:
            entry["j_per_turn"] = round(joules / n_requests, 5)
        if total_output_tokens:
            entry["j_per_output_token"] = round(joules / total_output_tokens, 7)
        if n_tool_calls_completed:
            entry["j_per_completed_tool_call"] = round(joules / n_tool_calls_completed, 5)
        elif n_tool_calls_completed == 0:
            entry["j_per_completed_tool_call"] = None
            entry["_j_per_completed_tool_call_not_collected"] = (
                "this workload attached tools but produced no structurally valid tool call"
            )
        else:
            entry["j_per_completed_tool_call"] = None
            entry["_j_per_completed_tool_call_not_collected"] = (
                "this workload attached no tools; measured by the tool-calling run instead, "
                "joined on config_id at analysis time"
            )
        entry["j_per_correct_answer"] = None
        entry["_j_per_correct_answer_not_collected"] = (
            "needs an accuracy denominator from the quality suite; computed in "
            "analysis/stats.py by joining this row to its quality run on config_id"
        )
        block[name] = entry
    return block
