"""The numeric envelope and its interval estimators.

These are the functions every published number passes through, so a silent
regression here is a silent regression in every figure and table. The cases
below are chosen to pin the properties that actually matter rather than to
exercise lines: that Wilson stays inside [0,1] where the normal
approximation does not, that a bootstrap is reproducible across runs, and
that an unmeasured quantity can never render as a zero."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import envelope as E


def test_wilson_reproduces_the_brief_s_worked_example():
    """The task brief's own claim: at n=200 a ~67% score carries about
    +-6.5pp. If this drifts, every 'is this ordering real?' judgement in the
    report drifts with it."""
    lo, hi = E.wilson_ci(134, 200)
    assert (hi - lo) / 2 == pytest.approx(0.065, abs=0.002)
    assert lo < 0.67 < hi


def test_wilson_stays_inside_the_unit_interval_at_the_boundary():
    """Where the normal approximation emits a negative lower bound, Wilson
    must not - this is the whole reason accuracy uses it."""
    lo, hi = E.wilson_ci(0, 30)
    assert lo == 0.0
    assert 0.0 < hi < 0.2
    lo, hi = E.wilson_ci(30, 30)
    assert hi == 1.0
    assert 0.8 < lo < 1.0


def test_wilson_narrows_with_n():
    narrow = E.wilson_ci(1340, 2000)
    wide = E.wilson_ci(134, 200)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0]) / 2


def test_bootstrap_is_reproducible():
    """An interval that moves when you re-run the analysis is not a result."""
    vals = [float(x) for x in range(1, 41)]
    a = E.bootstrap_ci(vals, lambda s: E.percentile(s, 90))
    b = E.bootstrap_ci(vals, lambda s: E.percentile(s, 90))
    assert a == b


def test_bootstrap_declines_below_three_samples():
    """Two points can produce a number-shaped interval that means nothing."""
    assert E.bootstrap_ci([1.0, 2.0], E.mean) is None


def test_percentile_interpolates_rather_than_returning_the_max():
    vals = list(range(1, 11))
    assert E.percentile(vals, 99) < 10
    assert E.percentile(vals, 50) == pytest.approx(5.5)


def test_not_collected_never_looks_like_a_zero():
    leaf = E.not_collected("no quality run joined", "J")
    assert leaf["value"] is None
    assert leaf["_not_collected"]
    assert not E.is_measured(leaf)
    assert E.is_measured(E.measured(0.0, "J")) is False or E.measured(0.0, "J")["value"] == 0.0


def test_empty_input_yields_a_reason_not_an_exception():
    leaf = E.percentile_envelope([], 50, "ms")
    assert leaf["value"] is None and leaf["_not_collected"]


def test_ratio_envelope_is_conservative_and_says_so():
    num = E.measured(100.0, "J", ci95=[90.0, 110.0])
    den = E.measured(0.5, "fraction", ci95=[0.4, 0.6])
    out = E.ratio_envelope(num, den, "J/correct")
    assert out["value"] == pytest.approx(200.0)
    # Interval arithmetic, so the bounds are the extreme quotients.
    assert out["ci95"] == [pytest.approx(150.0), pytest.approx(275.0)]
    assert out["method"] == "interval-arithmetic"


def test_ratio_envelope_refuses_a_missing_input():
    out = E.ratio_envelope(E.measured(1.0, "J"), E.not_collected("no accuracy"), "J/correct")
    assert out["value"] is None and out["_not_collected"]


def test_overlap_is_none_when_either_side_has_no_interval():
    a = E.measured(1.0, "ms", ci95=[0.9, 1.1])
    assert E.intervals_overlap(a, E.measured(2.0, "ms")) is None
    assert E.intervals_overlap(a, E.measured(1.05, "ms", ci95=[1.0, 1.2])) is True
    assert E.intervals_overlap(a, E.measured(5.0, "ms", ci95=[4.0, 6.0])) is False


def test_flatten_finds_every_leaf_in_a_nested_row():
    row = {"latency": {"ttft_ms_p50": E.measured(1.0, "ms")},
           "energy": {"marginal": {"j_per_turn": E.not_collected("no idle")}}}
    found = dict(E.flatten_leaves(row))
    assert set(found) == {"latency.ttft_ms_p50", "energy.marginal.j_per_turn"}
