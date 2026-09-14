"""The quality -> performance join.

The join is where joules per correct answer becomes computable, and it is
the one place in the pipeline where two independently-measured documents
meet. What must hold: successes are COUNTED where the per-item log exists,
reconstruction from a rounded rate is flagged where it does not, and a
config with no quality run produces an honest absence rather than a default."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "analysis"))

import envelope as E   # noqa: E402
import quality         # noqa: E402


def _quality_run(root: Path, name: str, cfg: str, suite: str, scores: dict,
                 outcomes: list[dict] | None = None) -> None:
    run = root / name
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({
        "schema_version": "1.0.0", "result_id": name,
        "manifest": {"model_config_key": cfg, "platform": "thor"},
        "dataset": {"name": suite, "n_available": 640, "n_evaluated": scores["n"],
                    "sampling_method": "first-n"},
        "protocol": {"scorer": "simplified-ast"},
        "scores": scores,
    }))
    if outcomes is not None:
        (run / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes) + "\n")


def test_successes_are_counted_from_outcomes_not_derived_from_a_rate(tmp_path):
    outcomes = [{"category": "irrelevance", "correct": i < 47} for i in range(50)]
    _quality_run(tmp_path, "run_a", "cfg", "BFCL",
                 {"overall_accuracy": 0.94, "n": 50,
                  "by_category": {"irrelevance": {"accuracy": 0.94, "n": 50, "n_correct": 47}}},
                 outcomes)
    got = quality.load_quality(tmp_path)
    leaf = got["cfg"]["suites"]["bfcl_irrelevance"]
    assert leaf["_successes"] == 47
    assert leaf["n"] == 50
    assert "_reconstructed_from_rate" not in leaf


def test_reconstruction_from_a_rate_is_flagged(tmp_path):
    _quality_run(tmp_path, "run_b", "cfg", "MMLU", {"overall_accuracy": 0.63, "n": 1000})
    got = quality.load_quality(tmp_path)
    leaf = got["cfg"]["suites"]["mmlu_overall"]
    assert leaf["value"] == pytest.approx(0.63)
    assert "_reconstructed_from_rate" in leaf


def test_every_score_carries_a_wilson_interval(tmp_path):
    _quality_run(tmp_path, "run_c", "cfg", "MMLU", {"overall_accuracy": 0.63, "n": 1000})
    leaf = quality.load_quality(tmp_path)["cfg"]["suites"]["mmlu_overall"]
    assert leaf["method"] == "wilson"
    assert leaf["ci95"][0] < leaf["value"] < leaf["ci95"][1]


def test_headline_prefers_the_tool_call_suite(tmp_path):
    """This lab's recommendation is a tool-call recommendation, so joules per
    correct answer must mean joules per correct tool-call judgment whenever
    one was measured."""
    _quality_run(tmp_path, "run_d", "cfg", "MMLU", {"overall_accuracy": 0.63, "n": 1000})
    _quality_run(tmp_path, "run_e", "cfg", "BFCL", {"overall_accuracy": 0.94, "n": 100})
    got = quality.load_quality(tmp_path)
    assert got["cfg"]["accuracy"]["_headline_suite"] == "bfcl_overall"


def test_a_config_with_no_quality_run_is_simply_absent(tmp_path):
    _quality_run(tmp_path, "run_f", "other", "BFCL", {"overall_accuracy": 0.9, "n": 100})
    assert "cfg" not in quality.load_quality(tmp_path)


def test_benchmark_results_are_not_mistaken_for_quality_results(tmp_path):
    run = tmp_path / "perf"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({
        "manifest": {"model_config_key": "cfg"}, "aggregates": {"ttft_ms": {"p50": 1}}}))
    assert quality.load_quality(tmp_path) == {}


def test_narrow_and_wide_intervals_come_out_in_the_right_order(tmp_path):
    """The property every ordering claim in the report rests on."""
    _quality_run(tmp_path, "small", "a", "BFCL", {"overall_accuracy": 0.94, "n": 50})
    _quality_run(tmp_path, "large", "b", "BFCL", {"overall_accuracy": 0.94, "n": 2000})
    got = quality.load_quality(tmp_path)
    small, large = got["a"]["accuracy"], got["b"]["accuracy"]
    assert (small["ci95"][1] - small["ci95"][0]) > 4 * (large["ci95"][1] - large["ci95"][0])
    assert E.intervals_overlap(small, large) is True
