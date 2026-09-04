"""Tests for benchmarks/manifest.py - reproducibility metadata, experiment
IDs, immutable result writing, and the energy arithmetic (docs/TODO.md
Phase 2).

The bias throughout is towards asserting that a *failed probe degrades
honestly* rather than that a successful one works: `docker` may be absent,
a server may expose no version endpoint, tegrastats may produce no samples.
docs/note.md §54 is mostly a list of ways benchmark metadata gets quietly
fabricated, and every one of those is a place where a `None` should have
been returned instead.

No Docker/GPU needed.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import manifest  # noqa: E402


# --- experiment IDs --------------------------------------------------------


def test_experiment_id_is_human_readable_and_ordered():
    """docs/note.md §30 wants an ID traceable by eye, not an opaque UUID."""
    exp = manifest.experiment_id(
        platform="orin",
        model_config_key="1.5b-awq-vllm-orin",
        experiment="output-sweep",
        input_tokens=512,
        output_tokens=128,
        replicate=1,
        on=date(2026, 9, 4),
    )
    assert exp == "2026-09-04_orin_1.5b-awq-vllm-orin_output-sweep_in512_out128_bs1_r01"


def test_experiment_id_omits_unset_workload_axes():
    """A cell that set no input-token target must not claim `in0`."""
    exp = manifest.experiment_id(
        platform="orin", model_config_key="k", experiment="streaming",
        output_tokens=128, on=date(2026, 9, 4),
    )
    # Segment-wise, not substring: "orin" contains "in".
    assert not any(seg.startswith("in") and seg[2:].isdigit() for seg in exp.split("_"))
    assert exp.endswith("_out128_bs1_r01")


def test_replicate_is_zero_padded_so_ids_sort():
    a = manifest.experiment_id(platform="orin", model_config_key="k", experiment="e",
                               replicate=2, on=date(2026, 9, 4))
    b = manifest.experiment_id(platform="orin", model_config_key="k", experiment="e",
                               replicate=10, on=date(2026, 9, 4))
    assert sorted([b, a]) == [a, b]


# --- immutable result writing ---------------------------------------------


def test_write_result_places_by_experiment_id(tmp_path: Path):
    result = {"experiment_id": "2026-09-04_orin_k_e_bs1_r01", "runs": []}
    path = manifest.write_result(result, results_root=tmp_path)
    assert path == tmp_path / "raw" / result["experiment_id"] / "result.json"
    assert json.loads(path.read_text())["experiment_id"] == result["experiment_id"]


def test_write_result_refuses_to_overwrite(tmp_path: Path):
    """docs/note.md §29 - historical results are immutable. The failure this
    prevents is a re-run silently replacing the data a published figure was
    drawn from; re-running a cell means a new replicate, which the ID already
    encodes."""
    result = {"experiment_id": "2026-09-04_orin_k_e_bs1_r01"}
    manifest.write_result(result, results_root=tmp_path)
    with pytest.raises(FileExistsError, match="immutable"):
        manifest.write_result(result, results_root=tmp_path)


def test_write_result_rejects_an_unidentified_document(tmp_path: Path):
    with pytest.raises(ValueError, match="experiment_id"):
        manifest.write_result({"runs": []}, results_root=tmp_path)


def test_write_result_accepts_a_quality_documents_result_id(tmp_path: Path):
    """quality_result documents key on `result_id`, not `experiment_id`."""
    path = manifest.write_result({"result_id": "2026-09-04_orin_k_bfcl_n40"}, results_root=tmp_path)
    assert path.parent.name == "2026-09-04_orin_k_bfcl_n40"


# --- energy ----------------------------------------------------------------

_RAILS = {
    "vdd_gpu_soc_mw": {"avg": 20000.0, "min": 1.0, "max": 2.0, "n": 30},
    "vdd_cpu_cv_mw": {"avg": 5000.0, "min": 1.0, "max": 2.0, "n": 30},
    "vin_sys_5v0_mw": {"avg": 3000.0, "min": 1.0, "max": 2.0, "n": 30},
}


def test_energy_sums_only_the_declared_rails():
    """VIN_SYS_5V0 is a separate board-level supply. Including it silently
    would change the quantity being reported, which is why `rails_included`
    is recorded alongside the number - a GPU-only figure and a board-total
    figure must never share an axis (docs/note.md §14)."""
    block = manifest.energy_block(_RAILS, window_s=10.0, n_requests=5, total_output_tokens=100)
    # (20000 + 5000) mW = 25 W over 10 s = 250 J. VIN_SYS_5V0 excluded.
    assert block["energy_joules"] == pytest.approx(250.0)
    assert block["rails_included"] == ["vdd_gpu_soc_mw", "vdd_cpu_cv_mw"]
    assert block["energy_per_request_j"] == pytest.approx(50.0)
    assert block["energy_per_output_token_j"] == pytest.approx(2.5)


def test_energy_keeps_every_sampled_rail_in_the_record():
    """Excluded from the sum, but not thrown away - a later analysis may want
    board-total, and it cannot recover a rail that was never written down."""
    block = manifest.energy_block(_RAILS, window_s=10.0, n_requests=1, total_output_tokens=1)
    assert set(block["rails_mw"]) == set(_RAILS)


def test_no_power_samples_means_no_energy_figure():
    """Omitted, not zero: a reader must be able to tell "not measured" from
    "measured as zero"."""
    block = manifest.energy_block({}, window_s=10.0, n_requests=5, total_output_tokens=100)
    assert "energy_joules" not in block
    assert block["rails_included"] == []


def test_no_token_count_means_no_per_token_figure():
    block = manifest.energy_block(_RAILS, window_s=10.0, n_requests=5, total_output_tokens=None)
    assert "energy_joules" in block
    assert "energy_per_output_token_j" not in block


def test_zero_length_window_yields_no_energy():
    """Guards the division as well as the physics: a window that never opened
    cannot have integrated any power."""
    block = manifest.energy_block(_RAILS, window_s=0.0, n_requests=5, total_output_tokens=100)
    assert "energy_joules" not in block


# --- probes degrade honestly ----------------------------------------------


def test_missing_command_returns_none_not_a_guess(monkeypatch):
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: None)
    assert manifest.docker_version() is None
    assert manifest.image_digest("whatever:tag") is None


def test_git_info_reports_unknown_rather_than_inventing_a_commit(monkeypatch):
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: None)
    info = manifest.git_info()
    assert info["git_commit"] == "unknown"
    assert info["git_dirty"] is False


def test_clean_tree_is_not_reported_as_dirty(monkeypatch):
    """`git status --porcelain` returns "" on a clean tree and the helper
    turns that into None - so a naive truthiness check would conflate "clean"
    with "couldn't tell", in the direction that hides a dirty tree."""
    monkeypatch.setattr(manifest, "_run",
                        lambda cmd, **k: "abc1234" if "rev-parse" in cmd else "")
    assert manifest.git_info()["git_dirty"] is False

    monkeypatch.setattr(manifest, "_run",
                        lambda cmd, **k: "abc1234" if "rev-parse" in cmd else " M src/x.py")
    assert manifest.git_info()["git_dirty"] is True


def test_backend_version_says_it_looked_rather_than_guessing(monkeypatch):
    """Never derive a version from the image tag:
    `dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04` is a tag, and the builds it
    has shipped (4579, 5058, 5283) behave differently enough that this lab had
    to tell them apart by behaviour."""
    monkeypatch.setattr(manifest, "_get_json", lambda *a, **k: None)
    assert manifest.probe_backend_version("http://x", "vllm").startswith("unknown")
    assert manifest.probe_backend_version("http://x", "llama-cpp").startswith("unknown")


def test_backend_version_reads_the_running_server(monkeypatch):
    monkeypatch.setattr(manifest, "_get_json", lambda url, **k: {"version": "0.10.1"})
    assert manifest.probe_backend_version("http://x", "vllm") == "vllm 0.10.1"

    monkeypatch.setattr(manifest, "_get_json", lambda url, **k: {"build_info": "b5058-6bf28f01"})
    assert manifest.probe_backend_version("http://x", "llama-cpp") == "b5058-6bf28f01"


def test_image_digest_strips_the_repo_prefix(monkeypatch):
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: "dustynv/llama_cpp@sha256:abc")
    assert manifest.image_digest("dustynv/llama_cpp:tag") == "sha256:abc"
