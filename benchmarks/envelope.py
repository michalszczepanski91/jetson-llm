"""The numeric envelope every leaf of a results.json wears, plus the
interval estimators that fill it in.

Why an envelope at all: the plotting code must never have to guess whether
a number came with an interval, how many samples back it, or what units it
is in. `docs/note.md` §5 already keeps measurement layers separable; this
does the same one level down, at the individual number.

    {"value": 40.7, "unit": "tokens/s", "ci95": [39.1, 42.2], "n": 30,
     "method": "bootstrap"}

and the only other legal shape is the honest absence:

    {"value": null, "unit": "J", "_not_collected": "<short reason>"}

**Stdlib only, on purpose.** This module is imported by BOTH the measurement
path (which is stdlib-only by design - see benchmarks/harness.py's docstring
for why this repo keeps its .venv minimal) and the analysis path (which may
use numpy/matplotlib freely). Sharing one implementation is what stops the
two sides from disagreeing about what a p90 is, which is exactly the class
of bug `benchmarks/runner.py`'s `_POWER_RAILS` comment records having been
paid for once already.

Estimator choice per quantity is not cosmetic:

  * **Wilson** for accuracy. A proportion's normal-approximation interval is
    wrong in precisely the regime this lab reports in - near 0 or 1, at
    n in the hundreds - and it can produce bounds outside [0, 1]. At the
    n=200 the task brief calls out, a 67% score carries roughly +-6.5pp;
    that has to be visible on the figure, not inferred by the reader.
  * **Bootstrap (percentile method)** for latency percentiles. Latency is
    not normal, and a p99 in particular has no closed-form interval worth
    trusting. Resampling makes no distributional claim.
  * **Normal** only for a mean of many independent samples, and it is
    labelled so a reader can discount it.

Every interval records its `method` because a reader comparing two bars
needs to know they were built the same way.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Sequence

#: Resamples for a bootstrap interval. 2000 is where the percentile method's
#: own Monte-Carlo noise falls below the width of the intervals this lab
#: actually reports; going to 10000 changes published digits by less than
#: rounding does, and costs 5x on a board that is also the device under test.
DEFAULT_BOOTSTRAP = 2000

#: Fixed so a results.json is reproducible from its records.jsonl. An
#: interval that moves when you re-run the analysis is not a result.
BOOTSTRAP_SEED = 20260911


def measured(
    value: float | int | None,
    unit: str,
    *,
    ci95: Sequence[float] | None = None,
    n: int | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """A measured leaf. `value=None` is legal only via `not_collected()` -
    a None here means the estimator had nothing to work with, which is
    reported as such rather than as a zero."""
    leaf: dict[str, Any] = {"value": value, "unit": unit}
    if ci95 is not None:
        lo, hi = ci95
        leaf["ci95"] = [lo, hi]
    if n is not None:
        leaf["n"] = n
    if method is not None:
        leaf["method"] = method
    return leaf


def not_collected(reason: str, unit: str | None = None) -> dict[str, Any]:
    """The only legal way to write a missing number.

    The task brief's rule - "never invent a value; anything not measured is
    null with a short reason next to it" - is enforced by making the absent
    case as easy to write as the present one. A partial honest file is the
    goal; a file where a zero and an unmeasured quantity look alike is not."""
    leaf: dict[str, Any] = {"value": None, "_not_collected": reason}
    if unit:
        leaf["unit"] = unit
    return leaf


def is_measured(leaf: Any) -> bool:
    return isinstance(leaf, dict) and leaf.get("value") is not None


# --------------------------------------------------------------------------
# point estimators
# --------------------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, `p` in [0, 100].

    Interpolating rather than nearest-rank because a p99 of 30 samples is
    otherwise just max(), which reports the sample maximum as if it were a
    tail estimate. The bootstrap interval around it is what actually says
    how little 30 samples know about a p99 - see `percentile_envelope`."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(ordered[int(k)])
    return float(ordered[lo] * (hi - k) + ordered[hi] * (k - lo))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation (n-1). Reported for inter-token latency
    specifically, where the brief's point stands: a stuttering 40 tok/s is
    worse for an on-device orchestrator than a steady 25, and only the
    spread says which one you have."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


# --------------------------------------------------------------------------
# interval estimators
# --------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, returned on [0, 1].

    Used for every accuracy number in this lab. The normal approximation is
    not acceptable here: it misbehaves exactly where the interesting scores
    are (Edge-LLM's 94% irrelevance is close enough to the boundary that the
    two intervals differ visibly), and it can emit a lower bound below zero.
    Wilson cannot."""
    if n <= 0:
        raise ValueError("wilson_ci needs n > 0")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = centre - half, centre + half
    # At k=0 and k=n the analytic bound is exactly 0 and exactly 1, but the
    # expression above reaches it by cancellation and leaves ~1e-17 behind.
    # Harmless arithmetically, and still worth removing: a published table
    # should read "0.0", not "6.9e-18", and a reader should not have to
    # decide which of those two a figure meant.
    if successes == 0:
        lo = 0.0
    if successes == n:
        hi = 1.0
    return (max(0.0, lo), min(1.0, hi))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile-method bootstrap interval for an arbitrary statistic.

    Returns None below 3 samples rather than an interval: with two points a
    resampling interval is a number-shaped artifact of the two points, and
    printing it would imply a precision that was never measured.

    The RNG is seeded per call from a module constant, so re-running the
    analysis over an unchanged records.jsonl reproduces the published
    interval exactly."""
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(stat(sample))
    stats.sort()
    lo = percentile(stats, 100 * alpha / 2)
    hi = percentile(stats, 100 * (1 - alpha / 2))
    return (lo, hi)


def normal_ci(values: Sequence[float], z: float = 1.959963985) -> tuple[float, float] | None:
    """Textbook interval on a mean. Offered, and labelled `normal` in the
    envelope, only for quantities where a mean is the thing being reported
    (average power over a window). Never used for a percentile."""
    if len(values) < 2:
        return None
    m = mean(values)
    half = z * stdev(values) / math.sqrt(len(values))
    return (m - half, m + half)


# --------------------------------------------------------------------------
# envelope constructors
# --------------------------------------------------------------------------

def percentile_envelope(
    values: Sequence[float],
    p: float,
    unit: str,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    reason_if_empty: str = "no successful runs in this cell",
) -> dict[str, Any]:
    if not values:
        return not_collected(reason_if_empty, unit)
    ci = bootstrap_ci(values, lambda s: percentile(s, p), n_boot=n_boot)
    return measured(
        percentile(values, p), unit,
        ci95=list(ci) if ci else None,
        n=len(values),
        method="bootstrap" if ci else "point (n<3, no interval)",
    )


def mean_envelope(values: Sequence[float], unit: str, *, reason_if_empty: str = "no samples") -> dict[str, Any]:
    if not values:
        return not_collected(reason_if_empty, unit)
    ci = normal_ci(values)
    return measured(
        mean(values), unit,
        ci95=list(ci) if ci else None,
        n=len(values),
        method="normal" if ci else "point (n<2, no interval)",
    )


def stdev_envelope(values: Sequence[float], unit: str, *, n_boot: int = DEFAULT_BOOTSTRAP) -> dict[str, Any]:
    if len(values) < 2:
        return not_collected("fewer than 2 samples", unit)
    ci = bootstrap_ci(values, stdev, n_boot=n_boot)
    return measured(
        stdev(values), unit,
        ci95=list(ci) if ci else None,
        n=len(values),
        method="bootstrap" if ci else "point (n<3, no interval)",
    )


def proportion_envelope(successes: int, n: int, unit: str = "fraction") -> dict[str, Any]:
    """Accuracy and other pass-rates. Always Wilson, always with n visible,
    because the brief's rule - don't report an ordering the intervals don't
    support - can only be checked by a reader who can see the interval."""
    if n <= 0:
        return not_collected("no items scored", unit)
    lo, hi = wilson_ci(successes, n)
    return measured(successes / n, unit, ci95=[lo, hi], n=n, method="wilson")


def ratio_envelope(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
    unit: str,
    *,
    reason: str = "one of the two inputs was not collected",
) -> dict[str, Any]:
    """A derived quantity built from two envelopes (e.g. J per correct
    answer = J per task / accuracy).

    The interval is propagated by *interval arithmetic on the two ci95s*,
    not by pretending the ratio has a bootstrap of its own: the two inputs
    generally come from different runs (a power run and a quality run) and
    are not resampleable jointly from anything this lab records. That makes
    the result conservative - wider than a joint bootstrap would give - and
    it is labelled `interval-arithmetic` so nobody reads it as tighter than
    it is."""
    if not (is_measured(numerator) and is_measured(denominator)):
        return not_collected(reason, unit)
    d = denominator["value"]
    if not d:
        return not_collected("denominator is zero", unit)
    value = numerator["value"] / d
    n_ci = numerator.get("ci95")
    d_ci = denominator.get("ci95")
    if n_ci and d_ci and d_ci[0] > 0:
        candidates = [a / b for a in n_ci for b in d_ci]
        ci = [min(candidates), max(candidates)]
        return measured(value, unit, ci95=ci, method="interval-arithmetic")
    return measured(value, unit, method="point (an input carried no interval)")


def intervals_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool | None:
    """Do two envelopes' 95% intervals overlap?

    The blunt significance test the report's tables use. Non-overlap implies
    a difference at roughly the 95% level; overlap does NOT imply no
    difference (two overlapping 95% intervals can still differ
    significantly), so callers must phrase a negative as "the intervals do
    not support an ordering", never as "the configs are equal". Returns None
    when either side has no interval to compare."""
    if not (is_measured(a) and is_measured(b)):
        return None
    ca, cb = a.get("ci95"), b.get("ci95")
    if not ca or not cb:
        return None
    return not (ca[1] < cb[0] or cb[1] < ca[0])


def flatten_leaves(obj: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    """Walk a result row and yield (dotted.path, leaf) for every envelope.

    Used by the schema check and by figures.py to look a metric up by name
    without hard-coding the nesting."""
    if isinstance(obj, dict):
        if "value" in obj and ("unit" in obj or "_not_collected" in obj):
            yield prefix, obj
            return
        for k, v in obj.items():
            yield from flatten_leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten_leaves(v, f"{prefix}[{i}]")
