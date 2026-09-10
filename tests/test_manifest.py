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


# --- the standalone honesty check -----------------------------------------


def test_standalone_is_refused_when_other_containers_run(monkeypatch):
    """docs/note.md §15's distinction is only worth anything if a `standalone`
    claim is checked. This device runs a production vllm-orchestrator around
    the clock, so `standalone` is the natural thing to type and nothing about
    the run would look wrong - which is how a mislabelled result nearly got
    committed on 2026-09-04."""
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: "vllm-orchestrator\nsomething-else")
    with pytest.raises(SystemExit) as exc:
        manifest.assert_condition_matches_reality("standalone", own_containers={"vllm-llm-lab"})
    message = str(exc.value)
    # The message must name what it found and show the fix, or it just blocks.
    assert "vllm-orchestrator" in message
    assert "co-resident" in message


def test_standalone_passes_on_a_quiet_board(monkeypatch):
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: None)
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [])
    manifest.assert_condition_matches_reality("standalone", own_containers=set())


def test_our_own_container_does_not_count_as_co_residency(monkeypatch):
    """The lab's own server is the workload, not a co-resident neighbour."""
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: "vllm-llm-lab")
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [])
    manifest.assert_condition_matches_reality("standalone", own_containers={"vllm-llm-lab"})


def test_co_resident_runs_are_never_blocked_when_the_declaration_is_complete(monkeypatch):
    """The check exists to stop a false/incomplete label, not to police what may
    run - a co-resident run declaring the truth is exactly what Phase 9 needs."""
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: "yolo\nstt\ntts")
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [])
    manifest.assert_condition_matches_reality(
        "co-resident", own_containers=set(), declared_co_resident=["yolo", "stt", "tts"]
    )


# --- bare-metal GPU-holding processes (not just containers) ---------------
#
# Real incident, 2026-09-04: a whole Phase 4 campaign ran and was labelled
# `standalone` while embedded-ai-chain's full production pipeline (YOLO +
# orchestrator + STT/TTS) ran as ONE bare-metal process, invisible to
# `docker ps`. Caught only by noticing the process held open nvhost/nvgpu
# file descriptors. Six results were written before this was caught and
# were deleted rather than relabelled - the co-resident workload was never
# declared, so it could not be described honestly after the fact.


def _fake_proc_tree(tmp_path, pid_fds: dict[int, list[str]], cmdlines: dict[int, str] | None = None):
    """Build a throwaway directory shaped like /proc: one subdir per pid,
    each with an `fd/` holding symlinks named after GPU (or non-GPU) anon
    inodes, plus an optional cmdline file. Exercises the real symlink-target
    parsing rather than mocking gpu_holding_pids() itself."""
    cmdlines = cmdlines or {}
    for pid, targets in pid_fds.items():
        fd_dir = tmp_path / str(pid) / "fd"
        fd_dir.mkdir(parents=True)
        for i, target in enumerate(targets):
            (fd_dir / str(i)).symlink_to(f"anon_inode:{target}")
        cmdline_path = tmp_path / str(pid) / "cmdline"
        cmdline_path.write_bytes(cmdlines.get(pid, f"proc{pid}").encode() + b"\x00")
    return tmp_path


def _patch_proc_root(monkeypatch, tmp_path):
    """Redirect every /proc-prefixed path manifest.py touches (gpu_holding_pids'
    own Path("/proc") AND _cmdline's Path(f"/proc/{pid}/cmdline")) into the fake
    tree, so both functions see the same fake filesystem consistently."""
    import pathlib
    real_path = pathlib.Path

    def fake_path(p="/proc"):
        p = str(p)
        if p == "/proc":
            return tmp_path
        if p.startswith("/proc/"):
            return tmp_path / p[len("/proc/"):]
        return real_path(p)

    monkeypatch.setattr(manifest, "Path", fake_path)


def test_gpu_holding_pids_finds_a_real_gpu_handle(tmp_path, monkeypatch):
    _fake_proc_tree(tmp_path, {93363: ["nvhost-17000000.gpu-fd10", "nvgpu-ga10b-tsg17"]},
                    {93363: "/home/michal/dev/embedded-ai-chain/.venv/bin/python tts_consumer.py"})
    _patch_proc_root(monkeypatch, tmp_path)
    found = manifest.gpu_holding_pids()
    assert [pid for pid, _ in found] == [93363]
    assert "tts_consumer.py" in found[0][1]


def test_gpu_holding_pids_ignores_processes_with_no_gpu_fd(tmp_path, monkeypatch):
    """A process with ordinary file descriptors (sockets, regular files) must
    not be reported - only an actual nvhost/nvgpu/nvmap handle counts as
    "holding the GPU", or every process on the board would trip this."""
    _fake_proc_tree(tmp_path, {111: ["socket:[12345]", "pipe:[6789]"]})
    _patch_proc_root(monkeypatch, tmp_path)
    assert manifest.gpu_holding_pids() == []


def test_gpu_holding_pids_degrades_on_unreadable_proc(monkeypatch):
    """A missing/unreadable /proc must not crash the benchmark - it should
    report "found none", the same honest-degradation rule as every other
    probe in this module."""
    monkeypatch.setattr(manifest, "Path", lambda p="/proc": __import__("pathlib").Path("/does/not/exist") if p == "/proc" else __import__("pathlib").Path(p))
    assert manifest.gpu_holding_pids() == []


def test_own_container_pids_resolves_via_docker_top(monkeypatch):
    """docker top reports HOST pids for containerized processes - containers
    share the host kernel and only the PID *namespace* differs - which is
    what lets this exclude the lab's own server without excluding a
    same-looking bare-metal process by name alone."""
    monkeypatch.setattr(manifest, "_run",
                        lambda cmd, **k: "PID\n249955\n250287\n250288" if "top" in cmd else None)
    assert manifest._own_container_pids({"vllm-llm-lab"}) == {249955, 250287, 250288}


def test_standalone_is_refused_for_a_bare_metal_gpu_process(monkeypatch):
    """The container check alone is exactly the gap that let the real
    incident through - this pins the fix."""
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: None)  # no containers, no docker top hits
    monkeypatch.setattr(manifest, "gpu_holding_pids",
                        lambda: [(93363, "python tts_consumer.py")])
    with pytest.raises(SystemExit) as exc:
        manifest.assert_condition_matches_reality("standalone", own_containers=set())
    message = str(exc.value)
    assert "93363" in message
    assert "tts_consumer.py" in message
    assert "co-resident" in message


def test_standalone_is_refused_when_container_and_bare_metal_both_present(monkeypatch):
    """Both signals are surfaced together, not just the first one found - a
    partial report would still under-declare the true co-resident workload."""
    monkeypatch.setattr(manifest, "_run",
                        lambda cmd, **k: "vllm-orchestrator" if cmd[:2] == ["docker", "ps"] else None)
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [(93363, "tts_consumer.py")])
    with pytest.raises(SystemExit) as exc:
        manifest.assert_condition_matches_reality("standalone", own_containers=set())
    message = str(exc.value)
    assert "vllm-orchestrator" in message
    assert "93363" in message


def test_own_containers_gpu_pids_are_excluded_from_the_bare_metal_check(monkeypatch):
    """The lab's own server legitimately holds a GPU handle - it must not
    trip its own guard. own_container_pids() has to resolve to the SAME pids
    gpu_holding_pids() reports for this to work."""
    monkeypatch.setattr(manifest, "_run",
                        lambda cmd, **k: ("" if cmd[:2] == ["docker", "ps"]
                                          else "PID\n249955" if "top" in cmd else None))
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [(249955, "vllm serve ...")])
    manifest.assert_condition_matches_reality("standalone", own_containers={"vllm-llm-lab"})


def test_standalone_passes_on_a_truly_quiet_board_including_bare_metal(monkeypatch):
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: None)
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [])
    manifest.assert_condition_matches_reality("standalone", own_containers=set())


# --- co-resident declarations can be incomplete, not just standalone false -
#
# Real incident, 2026-09-04, on already-COMMITTED data: 5 results correctly
# said co-resident: [vllm-orchestrator], but PID 93363 (tts_consumer.py) was
# running throughout every one of them and was never declared. Found only
# because the user asked about one specific committed result by name - no
# check caught it at write time, because none existed yet.


def test_co_resident_declaration_missing_a_bare_metal_process_is_refused(monkeypatch):
    """Pins the exact incident: 'co-resident: [vllm-orchestrator]' declared,
    but a bare-metal GPU process is also running and not named."""
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: None)
    monkeypatch.setattr(manifest, "gpu_holding_pids",
                        lambda: [(93363, "python tts_consumer.py")])
    with pytest.raises(SystemExit) as exc:
        manifest.assert_condition_matches_reality(
            "co-resident", own_containers=set(), declared_co_resident=["vllm-orchestrator"]
        )
    message = str(exc.value)
    assert "93363" in message
    assert "incomplete" in message


def test_co_resident_declaration_missing_a_container_is_refused(monkeypatch):
    monkeypatch.setattr(manifest, "_run",
                        lambda cmd, **k: "vllm-orchestrator" if cmd[:2] == ["docker", "ps"] else None)
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [])
    with pytest.raises(SystemExit) as exc:
        manifest.assert_condition_matches_reality(
            "co-resident", own_containers=set(), declared_co_resident=["something-else"]
        )
    assert "vllm-orchestrator" in str(exc.value)


def test_co_resident_declaration_matching_reality_passes(monkeypatch):
    monkeypatch.setattr(manifest, "_run",
                        lambda cmd, **k: "vllm-orchestrator" if cmd[:2] == ["docker", "ps"] else None)
    monkeypatch.setattr(manifest, "gpu_holding_pids",
                        lambda: [(93363, "python tts_consumer.py")])
    manifest.assert_condition_matches_reality(
        "co-resident", own_containers=set(),
        declared_co_resident=["vllm-orchestrator", "pid:93363"],
    )


def test_co_resident_declaration_check_ignores_own_containers(monkeypatch):
    """The lab's own server must not have to be declared as part of its own
    co-resident workload - it's the thing being measured, not a neighbour."""
    monkeypatch.setattr(manifest, "_run",
                        lambda cmd, **k: ("" if cmd[:2] == ["docker", "ps"]
                                          else "PID\n249955" if "top" in cmd else None))
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [(249955, "vllm serve ...")])
    manifest.assert_condition_matches_reality(
        "co-resident", own_containers={"vllm-llm-lab"}, declared_co_resident=["something"]
    )


def test_co_resident_with_no_declaration_and_nothing_running_passes(monkeypatch):
    """A co-resident label with an empty declared list is only valid if the
    board genuinely has nothing else on it - degenerate but not a crash."""
    monkeypatch.setattr(manifest, "_run", lambda *a, **k: None)
    monkeypatch.setattr(manifest, "gpu_holding_pids", lambda: [])
    manifest.assert_condition_matches_reality(
        "co-resident", own_containers=set(), declared_co_resident=None
    )


# --- remote-target manifests must not mislabel the client as the host ------
#
# hardware_manifest()/software_manifest() read /proc, /sys, and the local
# docker daemon - all describe the CLIENT machine, not a remote inference
# host reached via RemoteCoordinator (e.g. a Thor on the LAN). Calling them
# unconditionally would silently mislabel the Orin's own hardware state as
# if it described Thor, under a manifest whose own `platform` field claims
# "thor" - found while checking this repo's Thor-readiness, before any real
# Thor run happened to catch it the expensive way.


def test_local_hardware_manifest_reports_this_machine(monkeypatch):
    monkeypatch.setattr(manifest, "board_model", lambda: "Jetson AGX Orin")
    monkeypatch.setattr(manifest, "memory_total_mb", lambda: 30698.0)
    hw = manifest.hardware_manifest("orin", target="local")
    assert hw["board"] == "Jetson AGX Orin"
    assert hw["memory_total_mb"] == 30698.0


def test_remote_hardware_manifest_does_not_claim_client_values(monkeypatch):
    """Even if board_model()/memory_total_mb() would return real (Orin)
    values, a remote-target manifest must not report them under a `platform`
    field that names a different machine."""
    monkeypatch.setattr(manifest, "board_model", lambda: "Jetson AGX Orin")
    monkeypatch.setattr(manifest, "memory_total_mb", lambda: 30698.0)
    hw = manifest.hardware_manifest("thor", target="remote")
    assert hw["platform"] == "thor"
    assert "Orin" not in hw["board"]
    assert hw["memory_total_mb"] == 0.0
    assert "remote" in hw["nvpmodel_mode"]
    assert hw["jetson_clocks_locked"] is None
    assert hw["l4t_version"] is None


def test_remote_software_manifest_omits_local_docker_version(monkeypatch):
    """docker_version() is this client's own daemon - irrelevant to a
    container Thor's own Docker manages, not this client's."""
    monkeypatch.setattr(manifest, "probe_backend_version", lambda *a, **k: "vllm 0.19.0")
    monkeypatch.setattr(manifest, "docker_version", lambda: "27.5.1")
    sw = manifest.software_manifest("vllm", "http://thor:8000", None, target="remote")
    assert "docker" not in sw
    assert sw["backend_version"] == "vllm 0.19.0"  # this one IS remote-safe - queries base_url


def test_local_software_manifest_keeps_docker_version(monkeypatch):
    monkeypatch.setattr(manifest, "probe_backend_version", lambda *a, **k: "vllm 0.19.0")
    monkeypatch.setattr(manifest, "docker_version", lambda: "27.5.1")
    sw = manifest.software_manifest("vllm", "http://127.0.0.1:8000", "some/image:tag", target="local")
    assert sw["docker"] == "27.5.1"


def test_build_manifest_threads_target_through(monkeypatch):
    monkeypatch.setattr(manifest, "board_model", lambda: "Jetson AGX Orin")
    monkeypatch.setattr(manifest, "probe_backend_version", lambda *a, **k: "vllm 0.19.0")
    monkeypatch.setattr(manifest, "docker_version", lambda: "27.5.1")
    m = manifest.build_manifest(
        backend="vllm", platform="thor", base_url="http://thor:8000", image=None,
        command="x", model_config_key="k", target="remote",
    )
    assert "Orin" not in m["hardware"]["board"]
    assert "docker" not in m["software"]


# --- quality_manifest() (docs/TODO.md Phase 5's retrofit) ------------------


def test_quality_manifest_matches_the_schemas_manifest_shape(monkeypatch):
    monkeypatch.setattr(manifest, "probe_backend_version", lambda *a, **k: "vllm 0.19.0")
    m = manifest.quality_manifest(
        backend="vllm", platform="orin", base_url="http://127.0.0.1:8000",
        image="ghcr.io/x:tag", command="cmd", model_config_key="1.5b-awq-vllm-orin",
    )
    assert m["backend"] == "vllm"
    assert m["backend_version"] == "vllm 0.19.0"
    assert m["container_image"] == "ghcr.io/x:tag"
    assert m["platform"] == "orin"
    assert m["target"] == "local"
    assert "git_commit" in m and "git_dirty" in m


def test_quality_manifest_remote_target_flag_is_recorded(monkeypatch):
    monkeypatch.setattr(manifest, "probe_backend_version", lambda *a, **k: "vllm 0.19.0")
    m = manifest.quality_manifest(
        backend="vllm", platform="thor", base_url="http://thor:8000",
        image=None, command="cmd", model_config_key="k", target="remote",
    )
    assert m["target"] == "remote"
    assert "n/a" in m["container_image"]
