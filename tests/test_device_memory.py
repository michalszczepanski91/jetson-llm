"""Memory probing: the log scrapers, the residency accounting, and the
measured max-context search.

The scraper tests use log text captured from a real Thor startup on
2026-09-11 rather than text invented to match the regexes, which is the only
way this suite can fail when an image bump changes the phrasing - the
failure mode these patterns exist to catch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import device_memory as dm

REAL_VLLM_LOG = """
(EngineCore pid=102) INFO 09-11 08:55:25 [gpu_model_runner.py:4820] Model loading took 14.25 GiB memory and 13.038451 seconds
(EngineCore pid=102) INFO 09-11 08:51:01 [gpu_worker.py:447] Available KV cache memory: 31.92 GiB
(EngineCore pid=102) INFO 09-11 08:51:01 [kv_cache_utils.py:1319] GPU KV cache size: 597,776 tokens
(EngineCore pid=102) INFO 09-11 08:51:01 [kv_cache_utils.py:1323] Maximum concurrency for 16,384 tokens per request: 36.48x
(EngineCore pid=102) INFO 09-11 08:51:17 [gpu_worker.py:608] CUDA graph pool memory: 0.32 GiB (actual), 0.1 GiB (estimated)
"""

REAL_LLAMACPP_LOG = """
load_tensors: offloaded 29/29 layers to GPU
load_tensors:        CUDA0 model buffer size = 14531.94 MiB
llama_kv_cache_unified:      CUDA0 KV buffer size =   112.00 MiB
llama_context:      CUDA0 compute buffer size =   304.00 MiB
"""


def test_vllm_weights_and_kv_are_read_off_the_real_log():
    got = dm.parse_backend_memory("vllm", REAL_VLLM_LOG)
    assert got["weights_bytes_on_device"] == int(14.25 * 1024 ** 3)
    assert got["kv_cache_tokens_total"] == 597776
    assert got["graph_capture_bytes"] == int(0.32 * 1024 ** 3)
    assert got["max_context_configured"] == 16384
    assert got["matched_patterns"]


def test_llamacpp_offload_is_read_so_the_ngl_claim_is_verified_not_asserted():
    """-ngl -1 in a config file is an assertion; 'offloaded 29/29 layers' is
    the server's own report, and a partial offload silently makes a
    llama.cpp latency number a CPU number."""
    got = dm.parse_backend_memory("llama-cpp", REAL_LLAMACPP_LOG)
    assert got["offloaded_layers"] == 29
    assert got["total_layers"] == 29
    assert got["weights_bytes_on_device"] == int(14531.94 * 1024 ** 2)
    assert got["kv_cache_bytes_reserved"] == int(112.0 * 1024 ** 2)


def test_no_log_yields_no_fields_rather_than_zeros():
    got = dm.parse_backend_memory("vllm", None)
    assert got["matched_patterns"] == []
    assert "weights_bytes_on_device" not in got


def test_unmatched_log_yields_no_fields():
    got = dm.parse_backend_memory("vllm", "nothing resembling a memory line here")
    assert "weights_bytes_on_device" not in got


def _compose(**over):
    base = dict(
        backend_reported={"weights_bytes_on_device": 15 * 1024 ** 3,
                          "kv_cache_bytes_reserved": 32 * 1024 ** 3,
                          "kv_cache_tokens_total": 597776, "_source": "test"},
        kv_bytes_per_token=57344, kv_formula="test",
        artifact_bytes=15_231_271_888,
        baseline={"total_bytes": 128 * 1024 ** 3, "available_bytes": 96 * 1024 ** 3,
                  "used_bytes": 32 * 1024 ** 3, "device_bytes": 0},
        session={"n_samples": 100, "peak_used_bytes": 59 * 1024 ** 3,
                 "min_available_bytes": 72 * 1024 ** 3,
                 "n_device_samples": 50, "peak_device_bytes": 49 * 1024 ** 3},
        max_context={"max_context_measured": 16161},
    )
    base.update(over)
    return dm.compose(**base)


def test_footprint_comes_from_device_memory_not_from_meminfo():
    """The instrument choice is the whole finding. On this Tegra, /proc/meminfo
    cannot see the GPU carveout and cgroup accounting cannot see nvmap, so a
    board-level delta under-reports a 14 GiB weight load as 4.5 GiB. Only
    nvidia-smi's per-process figure attributes."""
    out = _compose()
    assert out["peak_device_bytes"] == 49 * 1024 ** 3
    assert out["peak_attributable_bytes"] == 49 * 1024 ** 3
    assert out["free_bytes_remaining"] == (128 - 49) * 1024 ** 3
    assert "nvidia-smi" in out["peak_device_source"]


def test_board_figures_are_kept_but_marked_non_attributing():
    out = _compose()
    assert out["board_used_peak_bytes"] == 59 * 1024 ** 3
    assert "must not" in out["board_accounting_caveat"]
    assert "peak_device_bytes" in out["board_accounting_caveat"]


def test_component_breakdown_reconciles_against_the_measured_total():
    """The runtime itemises weights and KV; the residual is CUDA context and
    allocator slack. Naming it is what makes a stacked memory bar add up to
    the total actually measured."""
    out = _compose()
    parts = out["component_breakdown_bytes"]
    assert parts["weights_bytes_on_device"] == 15 * 1024 ** 3
    assert parts["kv_cache_bytes_reserved"] == 32 * 1024 ** 3
    assert out["runtime_overhead_bytes"] == (49 - 15 - 32) * 1024 ** 3
    assert out["runtime_overhead_bytes"] + sum(parts.values()) == out["peak_device_bytes"]


def test_a_co_resident_baseline_is_subtracted():
    """peak_attributable is the rise above whatever else already held the GPU."""
    out = _compose(baseline={"total_bytes": 128 * 1024 ** 3, "available_bytes": 96 * 1024 ** 3,
                             "used_bytes": 32 * 1024 ** 3, "device_bytes": 9 * 1024 ** 3})
    assert out["peak_attributable_bytes"] == (49 - 9) * 1024 ** 3


def test_kv_formula_is_cross_checked_against_the_runtime():
    out = _compose()
    assert out["kv_formula_agrees_with_runtime"] is True
    wrong = _compose(kv_bytes_per_token=28672)   # half the truth, e.g. fp8 assumed
    assert wrong["kv_formula_agrees_with_runtime"] is False


def test_weights_fall_back_to_the_artifact_and_say_so():
    out = _compose(backend_reported={"_source": "no log available"})
    assert out["weights_bytes_on_device"] == 15_231_271_888
    assert out["weights_bytes_source"].startswith("FALLBACK")


def test_missing_samples_produce_a_reason_not_a_zero():
    out = _compose(session={"n_samples": 0})
    assert out["peak_device_bytes"] is None
    assert "nvidia-smi" in out["_peak_device_not_collected"]


def test_max_context_probe_classifies_a_config_limit_apart_from_an_oom():
    """A server refusing a prompt because --max-model-len says so and a
    server dying on a KV allocation have different fixes, and the token
    count alone cannot tell them apart."""
    calls = []

    def fake(url, model, text, timeout):
        n = len(text)
        calls.append(n)
        if n > 5000:
            return {"ok": False, "status": 400,
                    "detail": "This model's maximum context length is 16384 tokens"}
        return {"ok": True, "prompt_tokens": n}

    dm._attempt, real = fake, dm._attempt
    try:
        out = dm.probe_max_context("http://x", "m", lambda n: "x" * n, lo=128, hi_cap=65536)
    finally:
        dm._attempt = real
    assert out["max_context_measured"] <= 5000
    assert out["limiting_factor"].startswith("configuration")
    assert out["search_tolerance_tokens"] >= 64
    assert len(out["attempts"]) == len(calls)


def test_max_context_probe_reports_a_floor_failure_rather_than_guessing():
    def fake(url, model, text, timeout):
        return {"ok": False, "status": 500, "detail": "CUDA out of memory"}

    dm._attempt, real = fake, dm._attempt
    try:
        out = dm.probe_max_context("http://x", "m", lambda n: "x" * n, lo=128)
    finally:
        dm._attempt = real
    assert out["max_context_measured"] is None
    assert "floor probe" in out["_not_collected"]
