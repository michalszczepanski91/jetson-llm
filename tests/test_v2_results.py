"""The contract between analysis/stats.py and everything that reads its
output.

The schema is validated against REAL analysis output built from a synthetic
records.jsonl, not against a hand-written example - an example can be kept
conformant by editing the example. What must stay true is that the code path
figures.py depends on emits conformant documents."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "analysis"))

import envelope as E     # noqa: E402
import stats             # noqa: E402


def _interpreter_with_matplotlib() -> str | None:
    """The one dependency this repo deliberately does not put in its venv.

    `analysis/style.py` needs matplotlib and every figure consumer imports
    it; the venv stays stdlib+4 wheels because a native aarch64 wheel is
    exactly the shadowing risk `docs/environment.md` warns about, so the
    `figures`/`analyze` Makefile targets run under the system interpreter
    that already carries it. This test follows the same rule rather than
    asserting one interpreter has everything: try the venv first, fall back
    to the system `python3`, and skip only if neither can plot.
    """
    for exe in (sys.executable, shutil.which("python3")):
        if exe and subprocess.run([exe, "-c", "import matplotlib"],
                                  capture_output=True).returncode == 0:
            return exe
    return None

SCHEMA = json.loads((REPO / "schemas" / "v2_results.schema.json").read_text())


def _fake_run(tmp_path: Path, n: int = 12) -> Path:
    run = tmp_path / "2026-01-01_thor_fake_p1_bs1_r01"
    run.mkdir()
    meta = {
        "run_format_version": "2.0.0", "run_id": run.name, "config_id": "fake",
        "framework": "vllm", "experiment": "p1", "replicate": 1,
        "nvpmodel": "120W", "execution_condition": "standalone",
        "manifest": {"hardware": {"platform": "thor"}},
        "representation": {
            "declared_label": "bf16", "backend": "vllm",
            "weights": {"dtype": "bf16", "method": "none (unquantized)",
                        "granularity": "n/a (unquantized)", "group_size": None,
                        "symmetric": None, "storage_dtype_histogram_params": {"BF16": 10},
                        "_granularity_source": "safetensors tensor table"},
            "activations": {"dtype": "bfloat16", "_source": "checkpoint torch_dtype"},
            "kv_cache": {"dtype": "auto", "_source": "launch configuration"},
            "modules_kept_fp": ["lm_head"], "modules_quantized": [],
            "module_storage_types": {"lm_head": ["BF16"]},
            "calibration": {"_not_collected": "unquantized"},
            "artifact_bytes": 16, "param_count": 8,
            "param_count_method": "safetensors tensor table",
            "effective_bits_per_weight": 16.0, "comparability_class": "fp16_baseline",
        },
        "memory": {
            "weights_bytes_on_device": 100, "weights_bytes_source": "log",
            "kv_cache_bytes_per_token": 57344, "kv_cache_bytes_per_token_formula": "2 x ...",
            "system_total_bytes": 1000, "peak_device_bytes": 400,
            "peak_attributable_bytes": 300, "free_bytes_remaining": 600,
            "max_context": {"max_context_measured": 16161, "limiting_factor": "configuration"},
        },
        "cells": [{
            "prompt_target_tokens": 128, "gen_tokens": 16, "runs_ok": n,
            "warmup_discarded": 5, "power_samples_in_window": 60,
            "power_window_t0": 100.0, "power_window_t1": 130.0,
            "thermal": {"max_c": 52.0, "mean_c": 51.0, "sensor": "tj"},
            "cache_probe": {"ratio_second_over_first": 0.99,
                            "prefix_caching_appears_disabled": True},
            "energy": {
                "rails_included": ["vdd_gpu_mw"], "measurement_window_s": 30.0,
                "conventions": {"absolute": "a", "marginal": "m"},
                "generation_power_mw": 20000, "idle_power_mw": 8000,
                "idle": {"measured": True, "window_t0": 60.0, "window_t1": 80.0,
                         "total_power_mw": 8000, "rails_included": ["vdd_gpu_mw"]},
                "absolute": {"power_mw": 20000, "energy_j": 600, "j_per_turn": 50.0,
                             "j_per_output_token": 3.0, "j_per_completed_tool_call": None,
                             "_j_per_completed_tool_call_not_collected": "no tools",
                             "convention": "a"},
                "marginal": {"power_mw": 12000, "energy_j": 360, "j_per_turn": 30.0,
                             "j_per_output_token": 1.8, "j_per_completed_tool_call": None,
                             "_j_per_completed_tool_call_not_collected": "no tools",
                             "convention": "m"},
            },
        }],
        "confounds": {"verified": {}, "asserted": {}, "not_yet_verified": {}},
    }
    (run / "run_meta.json").write_text(json.dumps(meta))
    with (run / "records.jsonl").open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "kind": "request", "phase": "measure", "config_id": "fake",
                "framework": "vllm", "prompt_target_tokens": 128,
                "prompt_reference_tokens": 128, "prompt_sha256": "deadbeef",
                "gen_tokens": 16, "concurrency": 1, "nvpmodel": "120W", "run_index": i,
                "ttft_ms": 100.0 + i, "e2e_latency_ms": 1000.0 + i, "decode_ms": 900.0 + i,
                "input_tokens": 157, "output_tokens": 16,
                "prefill_tok_s": 1500.0 + i, "decode_tok_s": 12.0 + i * 0.01,
                "inter_token_latency_ms": [60.0 + (i % 3), 61.0, 80.0],
                "finish_reason": "length", "error": None,
            }) + "\n")
        # a warm-up row that must NOT reach any statistic
        fh.write(json.dumps({
            "kind": "request", "phase": "warmup", "config_id": "fake", "framework": "vllm",
            "prompt_target_tokens": 128, "gen_tokens": 16, "concurrency": 1,
            "run_index": -1, "ttft_ms": 99999.0, "e2e_latency_ms": 99999.0,
            "decode_tok_s": 0.1, "inter_token_latency_ms": [9999.0], "error": None,
        }) + "\n")
    with (run / "power.jsonl").open("w") as fh:
        # Idle window 60-80s then the measurement window 100-130s, at the
        # real sampler's 500ms cadence, so the interval machinery is
        # exercised on a realistic number of samples rather than on two.
        for k in range(40):
            fh.write(json.dumps({"t": 60.0 + k * 0.5, "vdd_gpu_mw": 3000 + (k % 4) * 20,
                                 "vdd_cpu_soc_mss_mw": 5000 + (k % 3) * 15,
                                 "vin_sys_5v0_mw": 6000, "tj_temp_c": 48.0}) + "\n")
        for k in range(60):
            fh.write(json.dumps({"t": 100.0 + k * 0.5, "vdd_gpu_mw": 12000 + (k % 5) * 100,
                                 "vdd_cpu_soc_mss_mw": 8000 + (k % 4) * 50,
                                 "vin_sys_5v0_mw": 14000, "tj_temp_c": 52.0}) + "\n")
    return run


@pytest.fixture
def rows(tmp_path):
    meta, records, power = stats.load_run(_fake_run(tmp_path))
    return stats.build_rows(meta, records, {}, power)


def test_output_validates_against_the_schema(rows):
    from jsonschema import Draft202012Validator
    doc = {"results_format_version": "2.0.0", "generated": "now", "rows": rows}
    errors = list(Draft202012Validator(SCHEMA).iter_errors(doc))
    assert not errors, [f"{'.'.join(map(str, e.path))}: {e.message}" for e in errors[:5]]


def test_every_leaf_is_a_legal_envelope(rows):
    assert stats.check_envelopes(rows) == []


def test_warmups_are_excluded_from_every_statistic(rows):
    """The 99999ms warm-up row would be visible in any percentile that let
    it through - which is the point of writing warm-ups to the JSONL at all,
    so their exclusion is checkable rather than asserted."""
    l = rows[0]["latency"]
    assert l["n_requests"] == 12
    assert l["ttft_ms_p99"]["value"] < 200
    assert l["inter_token_ms_p99"]["value"] < 100


def test_percentiles_carry_bootstrap_intervals(rows):
    leaf = rows[0]["latency"]["ttft_ms_p90"]
    assert leaf["method"] == "bootstrap"
    assert leaf["ci95"][0] <= leaf["value"] <= leaf["ci95"][1]


def test_ecdf_travels_with_the_result_so_figures_need_no_raw_data(rows):
    ecdf = rows[0]["latency"]["inter_token_ms_ecdf"]
    assert len(ecdf["ms"]) == 101
    assert ecdf["ms"] == sorted(ecdf["ms"])


def test_j_per_correct_answer_is_absent_with_a_reason_until_quality_joins(rows):
    leaf = rows[0]["energy"]["marginal"]["j_per_correct_answer"]
    assert leaf["value"] is None
    assert "quality" in leaf["_not_collected"]


def test_j_per_correct_answer_appears_once_a_quality_run_is_joined(tmp_path):
    meta, records, power = stats.load_run(_fake_run(tmp_path))
    quality = {"fake": {"accuracy": E.proportion_envelope(60, 100), "suites": {}}}
    rows = stats.build_rows(meta, records, quality, power)
    energy = rows[0]["energy"]
    leaf = energy["marginal"]["j_per_correct_answer"]
    # Derived from this row's own J/turn rather than a hard-coded constant:
    # the point under test is the JOIN, not the energy arithmetic.
    assert leaf["value"] == pytest.approx(energy["marginal"]["j_per_turn"]["value"] / 0.6)
    assert leaf["method"] == "interval-arithmetic"
    assert leaf["ci95"][0] < leaf["value"] < leaf["ci95"][1]


def test_ratio_propagates_an_interval_when_both_inputs_have_one():
    num = E.measured(30.0, "J/turn", ci95=[28.0, 32.0], method="bootstrap")
    den = E.proportion_envelope(60, 100)
    out = E.ratio_envelope(num, den, "J/correct-answer")
    assert out["method"] == "interval-arithmetic"
    assert out["ci95"][0] < out["value"] < out["ci95"][1]


def test_energy_carries_an_interval_derived_from_the_power_samples(rows):
    """A window mean has no spread of its own; the tegrastats samples inside
    the window do, and bootstrapping their mean is what stops energy from
    being the one quantity on the page without error bars."""
    e = rows[0]["energy"]
    assert e["generation_power_mw"]["ci95"]
    assert e["generation_power_mw"]["n"] == 60
    assert e["idle_power_mw"]["n"] == 40
    j = e["marginal"]["j_per_turn"]
    assert j["ci95"][0] < j["value"] < j["ci95"][1]


def test_vin_sys_is_sampled_but_never_summed_into_joules(rows):
    """The board-level supply already contains much of what the domain rails
    report. Summing it once read 37.6W against a true 23.5W."""
    e = rows[0]["energy"]
    assert "vin_sys_5v0_mw" not in e["rails_summed_into_joules"]
    # 12000-12400 + 8000-8150 per sample, so the mean must be well under the
    # ~34W it would reach if the 14W supply rail were included.
    assert 20000 < e["generation_power_mw"]["value"] < 21000


def test_marginal_is_generation_minus_idle(rows):
    e = rows[0]["energy"]
    assert e["marginal"]["power_mw"]["value"] == pytest.approx(
        e["generation_power_mw"]["value"] - e["idle_power_mw"]["value"])


def test_mbu_is_derived_and_flags_a_backend_bound_config(rows):
    eff = rows[0]["efficiency"]
    assert eff["decode_mbu"]["value"] > 0
    assert eff["peak_dram_gbps"]["method"].endswith("NOT measured")
    assert isinstance(eff["backend_bound_flag"], bool)


def test_figures_and_table_run_on_this_document(tmp_path, rows):
    """Both consumers must survive a document in which most quantities are
    honestly absent - that is the normal state of a partial campaign."""
    doc = tmp_path / "results.json"
    doc.write_text(json.dumps({"results_format_version": "2.0.0", "generated": "now",
                               "rows": rows}))
    exe = _interpreter_with_matplotlib()
    if exe is None:
        pytest.skip("no interpreter on this box has matplotlib - see make figures")
    for script, extra in (("analysis/figures.py", ["--out-dir", str(tmp_path / "fig")]),
                          ("analysis/table.py", ["--out", str(tmp_path / "t.tex")])):
        out = subprocess.run([exe, str(REPO / script), str(doc), *extra],
                             capture_output=True, text=True, cwd=REPO)
        assert out.returncode == 0, out.stderr
        assert "ERROR" not in out.stdout, out.stdout
