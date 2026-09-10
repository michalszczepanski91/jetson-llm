"""Reproducibility metadata and result persistence - docs/TODO.md Phase 2.

Unlike benchmarks/harness.py (a hand-maintained copy of jetson-vlm-lab's /
embedded-ai-chain's), this module is new and repo-specific: it exists because
docs/note.md §31 requires every result to carry enough information to re-run
it, and §29/§30 require a stable experiment ID and a raw-results hierarchy.

Three jobs:

  1. **Manifest** - git commit, backend version, container image digest,
     hardware and software environment. The rule that shapes this file is
     that a version must be *read from the thing that ran*, never inferred
     from a name: `dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04` is a tag, not
     a llama.cpp version, and this lab has already had to tell builds 4579 /
     5058 / 5283 apart by behaviour alone (docs/TODO.md Phase 1).

  2. **Experiment IDs and result paths** - `results/raw/<experiment_id>/`,
     replacing the hand-passed `--results-json` path that carried
     experimental meaning in a CLI flag.

  3. **Energy** - the arithmetic that turns tegrastats' per-rail average
     power over the measurement window into joules and J/output-token
     (docs/note.md §14). Deliberately here rather than in harness.py: the
     harness times a callable and has no idea how many tokens came back, so
     energy has to be assembled by the caller that knows both.

Everything degrades to an honest `None`/`"unknown"` rather than a guess when
a probe fails: docs/note.md §54's list of things not to do is mostly a list
of ways benchmark metadata gets quietly fabricated.

Stdlib-only, same rule as the rest of this repo's src/ and benchmarks/.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

_MEMTOTAL_RE = re.compile(r"MemTotal:\s+(\d+)\s+kB")


# --- small process helpers -------------------------------------------------


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    """Best-effort command output, or None. Never raises: a missing `docker`
    or a non-git checkout must degrade the manifest, not abort a benchmark
    that has already spent 160s on a cold start."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _get_json(url: str, timeout: float = 5.0) -> Any | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError, OSError):
        return None


# --- git -------------------------------------------------------------------


def git_info(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit and dirty state. `git_dirty=True` doesn't invalidate a run, but
    it does mean the commit alone can't reproduce it - so it is recorded
    rather than assumed false."""
    commit = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(repo_root), "status", "--porcelain"])
    return {
        "git_commit": commit or "unknown",
        # status is None on failure and "" on a clean tree - only a non-empty
        # string means genuinely dirty, so don't collapse the two with `or`.
        "git_dirty": bool(status) if status is not None else False,
    }


# --- docker ----------------------------------------------------------------


def docker_version() -> str | None:
    return _run(["docker", "version", "--format", "{{.Server.Version}}"])


def image_digest(image: str) -> str | None:
    """RepoDigest of a locally-present image. This is the only thing that
    actually pins it: `latest-jetson-orin` is a moving tag, so a campaign
    spanning weeks could silently change backends without this."""
    out = _run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image])
    if not out:
        return None
    # "repo@sha256:..." -> "sha256:..."
    return out.split("@", 1)[1] if "@" in out else out


def running_containers(exclude: set[str] | None = None) -> list[str]:
    """Names of currently running containers, minus the ones this lab starts
    itself. Used as a pre-flight honesty check: see `assert_condition_matches_reality`."""
    out = _run(["docker", "ps", "--format", "{{.Names}}"])
    if not out:
        return []
    return [n for n in out.splitlines() if n and n not in (exclude or set())]


def assert_condition_matches_reality(
    execution_condition: str,
    own_containers: set[str],
    declared_co_resident: list[str] | None = None,
) -> None:
    """Refuse a `standalone` claim on a board that isn't, and refuse an
    incomplete `co-resident` claim on one that is - both failure modes have
    happened on this repo's real data in one day.

    docs/note.md §15 makes the standalone/co-resident distinction load-bearing
    for every figure this lab produces, and a *mislabelled* result is strictly
    worse than a missing one - it silently contaminates any table it joins.
    The failure mode is entirely realistic: this device runs a production
    `vllm-orchestrator` around the clock, so the natural thing to type is
    `standalone`, and nothing about the run would have looked wrong. Caught
    exactly that way on the first Phase 3 grid run, 2026-09-04.

    Deliberately an error rather than a warning, and deliberately without an
    override flag: the fix is either to stop the other workload or to declare
    the truth, and both are one command. An override would be used.

    Detects two signals, because they miss different failure modes:

    1. **Other Docker containers** - `running_containers()`.
    2. **Bare-metal processes actually holding GPU device handles** -
       `gpu_holding_pids()`, scanning `/proc/<pid>/fd` for `nvhost`/`nvgpu`/
       `nvmap` symlinks, the same signal used to catch both incidents below.

    Then applies them two ways:

    - `execution_condition == "standalone"`: any detected signal is an error
      (the board must be quiet).
    - `execution_condition == "co-resident"`: every detected signal must
      already be named in `declared_co_resident`, or it's an error too - a
      declared workload a reader trusts to be complete is exactly as load-
      bearing as the top-level label itself.

    **Real incident, 2026-09-04.** The container check alone let a whole
    Phase 4 campaign run and get labelled `standalone` while
    `embedded-ai-chain`'s full production pipeline - YOLO perception,
    orchestrator, STT/TTS - was running as ONE bare-metal process the entire
    time, invisible to `docker ps`. Confirmed after the fact by
    `docker inspect`-style reasoning applied to `/proc`: the process held
    open `nvhost-*.gpu-fd*`/`nvgpu-*-tsg*` file descriptors, i.e. it was
    genuinely using the GPU, not just importing GPU-capable libraries. Six
    results had already been written and labelled `standalone` before this
    was caught. That first version of this function only checked containers,
    and its own docstring said so.

    **Second incident, same day, on already-committed data.** The user asked
    about one specific committed result (`..._phase2-verify_out128_bs1_r01`,
    correctly labelled `co-resident: [vllm-orchestrator]`) and whether it was
    contaminated. It wasn't mislabelled at the TOP level - but PID 93363 was
    running throughout that result's entire measurement window too (started
    2026-09-03, the result was written 2026-09-04 afternoon) and was never
    added to `co_resident_workload`. Five committed results had this gap.
    They were corrected in place (the field value fixed, a `_comment`
    explaining what changed and why, git history holding the original) rather
    than deleted - by then the archive-don't-delete policy below already
    applied, and unlike the six `standalone` results, there was nothing
    uncontrolled about this data: exactly what was running is known, so an
    honest correction was possible where the first incident's wasn't.

    The `declared_co_resident` completeness check exists because of this
    second incident: a co-resident label that's merely *not literally false*
    is not the same as one that's *complete*, and only the second is what a
    reader actually needs from `co_resident_workload`.

    Both incidents' contaminated/incomplete results are recorded in
    `docs/TODO.md`'s Phase 3 incident log, not just in this docstring - this
    function is the code fix, not a substitute for the record of why.

    Deliberately an error rather than a warning, and deliberately without an
    override flag: the fix is either to stop the other workload or to declare
    the truth, and both are one command. An override would be used.

    Still not exhaustive - a GPU-idle-but-about-to-wake process, or a process
    whose GPU access this /proc heuristic doesn't recognise, would still slip
    through. The message reports what was found, not that the board is
    clean."""
    own_pids = _own_container_pids(own_containers)
    other_containers = running_containers(exclude=own_containers)
    other_gpu = [(pid, cmd) for pid, cmd in gpu_holding_pids() if pid not in own_pids]

    if execution_condition == "standalone":
        if not other_containers and not other_gpu:
            return
        lines = ["error: execution_condition is 'standalone' but the board is not quiet:"]
        if other_containers:
            lines.append("  other containers running:")
            lines += [f"    {n}" for n in other_containers]
        if other_gpu:
            lines.append("  bare-metal processes actively holding a GPU device handle:")
            lines += [f"    pid {pid}: {cmd}" for pid, cmd in other_gpu]
        workload = other_containers + [f"pid:{pid}" for pid, _ in other_gpu]
        lines += [
            "On unified memory these compete for the same RAM and GPU, so the resulting",
            "numbers are co-resident numbers. Either stop them, or declare the truth:",
            "    execution_condition: co-resident",
            f"    co_resident_workload: [{', '.join(workload)}]",
            "A mislabelled result is worse than no result - it contaminates every table",
            "it is joined into (docs/note.md §15). See this function's own docstring for",
            "the real incident that made bare-metal detection necessary, not just Docker.",
        ]
        raise SystemExit("\n".join(lines))

    # execution_condition == "co-resident": a declared list can be INCOMPLETE
    # even when the top-level label is correct. Real incident, 2026-09-04: 5
    # results correctly said co-resident but declared only
    # ["vllm-orchestrator"], missing tts_consumer.py (PID 93363) - which was
    # running throughout every one of them. Caught by the user asking about a
    # SPECIFIC already-committed result, not by any check that existed then.
    # This closes that half of the gap the same way the standalone half was
    # closed: compare what's actually running against what was declared.
    declared = set(declared_co_resident or [])
    undeclared_containers = [n for n in other_containers if n not in declared]
    undeclared_gpu = [
        (pid, cmd) for pid, cmd in other_gpu
        if not any(str(pid) in d for d in declared)
    ]
    if not undeclared_containers and not undeclared_gpu:
        return
    lines = ["error: execution_condition is 'co-resident' but the declared workload is incomplete:"]
    if undeclared_containers:
        lines.append("  running but NOT in co_resident_workload:")
        lines += [f"    {n}" for n in undeclared_containers]
    if undeclared_gpu:
        lines.append("  bare-metal GPU-holding processes NOT in co_resident_workload:")
        lines += [f"    pid {pid}: {cmd}" for pid, cmd in undeclared_gpu]
    lines += [
        "An incomplete co-resident declaration is the same failure as a false",
        "standalone claim, one step removed: a reader trusts co_resident_workload",
        "to name everything that could have affected the numbers. Add the missing",
        "entries to co_resident_workload (docs/TODO.md Phase 3's incident record",
        "has the full story of how this was found on already-committed data).",
    ]
    raise SystemExit("\n".join(lines))


def _own_container_pids(container_names: set[str]) -> set[int]:
    """Host-visible PIDs of every process inside our own containers, via
    `docker top` - which reports HOST pids for containerized processes (Docker
    containers share the host kernel; only the PID *namespace* differs, so a
    process's host PID is real and visible in /proc). Used to exclude the
    lab's own server from the GPU-holding-process check, the same way
    `running_containers(exclude=...)` excludes it from the container check."""
    pids: set[int] = set()
    for name in container_names:
        out = _run(["docker", "top", name, "-eo", "pid"])
        if not out:
            continue
        for line in out.splitlines()[1:]:  # skip the "PID" header
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
    return pids


_GPU_FD_RE = re.compile(r"^(nvhost|nvgpu|nvmap)")


def gpu_holding_pids() -> list[tuple[int, str]]:
    """(pid, cmdline) for every process with an open nvhost/nvgpu/nvmap file
    descriptor - i.e. genuinely holding a GPU device handle right now, not
    merely having a GPU-capable library importable. This is the signal that
    caught embedded-ai-chain's production pipeline running bare-metal: see
    `assert_condition_matches_reality`'s docstring for the incident.

    Best-effort by nature: `/proc/<pid>/fd` for another user's process is
    unreadable without privilege and is silently skipped (this lab runs as
    the same user as the production pipeline it is checking for, so that
    has not been a practical limit here) - so an empty result means "found
    none", not "confirmed none"."""
    found: list[tuple[int, str]] = []
    self_pid = 0
    try:
        self_pid = int(Path("/proc/self").resolve().name)
    except (OSError, ValueError):
        pass
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        fd_dir = entry / "fd"
        try:
            has_gpu_fd = any(
                _GPU_FD_RE.match(target.name)
                for fd in fd_dir.iterdir()
                if (target := _readlink_target(fd)) is not None
            )
        except (OSError, PermissionError):
            continue
        if has_gpu_fd:
            cmdline = _cmdline(pid)
            found.append((pid, cmdline))
    return found


def _readlink_target(fd_path: Path) -> Path | None:
    try:
        target = fd_path.readlink()
    except OSError:
        return None
    # anon_inode targets look like "anon_inode:nvhost-17000000.gpu-fd10"
    name = target.name
    if name.startswith("anon_inode:"):
        return Path(name.removeprefix("anon_inode:"))
    return target


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return f"(pid {pid}, cmdline unreadable)"
    text = raw.replace(b"\x00", b" ").decode(errors="replace").strip()
    return text or f"(pid {pid}, no cmdline - kernel thread?)"


# --- backend version (read from the running server) ------------------------


def probe_backend_version(base_url: str, backend: str) -> str:
    """Ask the RUNNING server what it is. Endpoint support differs per
    backend and per build, so every known shape is tried and the first hit
    wins; the fallback is the literal string "unknown", never a guess derived
    from the image tag.

    vLLM serves `GET /version` -> {"version": "0.x.y"}. llama-server serves
    `GET /props`, whose `build_info`/`system_info` carries the build number
    that actually distinguishes 4579 (rejects tool_choice) from 5058 (works)
    - the exact distinction that made this function necessary."""
    if backend == "vllm":
        body = _get_json(f"{base_url}/version")
        if isinstance(body, dict) and body.get("version"):
            return f"vllm {body['version']}"
    elif backend == "llama-cpp":
        body = _get_json(f"{base_url}/props")
        if isinstance(body, dict):
            for key in ("build_info", "system_info"):
                if body.get(key):
                    return str(body[key])
            # Newer builds nest it under default_generation_settings/model_path
            # only; fall through to the generic probe rather than inventing one.
    # Generic last resort: /v1/models exposes an id but not a version - record
    # that we looked and found nothing rather than leaving the field absent.
    return "unknown (server exposed no version endpoint)"


# --- hardware / software ---------------------------------------------------


def memory_total_mb() -> float | None:
    """Real unified memory, read from the device. Never inferred from the
    platform name: this Orin is ~30GB despite earlier notes claiming 64GB,
    and docs/note.md §7's own schema example hardcodes a 128GB Thor."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    m = _MEMTOTAL_RE.search(text)
    return round(int(m.group(1)) / 1024, 1) if m else None


def board_model() -> str | None:
    try:
        return Path("/proc/device-tree/model").read_text().strip("\x00").strip() or None
    except OSError:
        return None


def hardware_manifest(platform: str, target: str = "local") -> dict[str, Any]:
    """The hardware block of a benchmark_result manifest. Imports harness
    lazily so this module stays usable (and unit-testable) without it.

    `target="remote"` (e.g. a Thor reached via RemoteCoordinator) is the case
    that makes every probe here wrong if called blindly: nvpmodel_mode(),
    jetson_clocks_locked_heuristic(), l4t_version(), board_model() and
    memory_total_mb() all read THIS process's own /proc and /sys - the client
    Orin's, not the remote inference host's. Calling them unconditionally
    would silently mislabel the Orin's hardware state as if it described
    Thor, under a manifest whose own `platform` field says "thor" - the same
    class of mislabeling this repo has already found and fixed twice for
    execution_condition (docs/TODO.md Phase 3's incident record). For a
    remote target these fields are honestly reported as not describing the
    inference host, mirroring the old scripts/benchmark.py's
    `tegrastats_caveat` rather than reinventing that caveat differently."""
    if target == "remote":
        return {
            "platform": platform,
            "board": "unknown (remote target - client-side probe would describe the wrong machine)",
            "memory_total_mb": 0.0,
            "nvpmodel_mode": "unknown (remote target - not queryable from this client)",
            "jetson_clocks_locked": None,
            "l4t_version": None,
        }
    from harness import jetson_clocks_locked_heuristic, l4t_version, nvpmodel_mode

    return {
        "platform": platform,
        "board": board_model() or "unknown",
        "memory_total_mb": memory_total_mb() or 0.0,
        "nvpmodel_mode": nvpmodel_mode(),
        "jetson_clocks_locked": jetson_clocks_locked_heuristic(),
        "l4t_version": l4t_version(),
    }


def software_manifest(backend: str, base_url: str, image: str | None, target: str = "local") -> dict[str, Any]:
    import platform as _platform

    block: dict[str, Any] = {
        "backend": backend,
        # backend_version genuinely IS a remote-safe probe unlike the fields
        # below - it queries base_url itself (the remote server), not this
        # client's own state.
        "backend_version": probe_backend_version(base_url, backend),
        "container_image": image or "n/a (remote target - image not owned by this host)",
        "python": _platform.python_version(),
    }
    if image:
        digest = image_digest(image)
        if digest:
            block["container_digest"] = digest
    # docker_version() reports THIS client's Docker daemon. Meaningful for a
    # local run (that daemon runs the container being measured); meaningless
    # for remote (Thor's container isn't managed by this client's Docker at
    # all) - omitted rather than included-and-mislabeled, same reasoning as
    # hardware_manifest()'s remote branch above.
    if target != "remote":
        docker = docker_version()
        if docker:
            block["docker"] = docker
    return block


def build_manifest(
    *,
    backend: str,
    platform: str,
    base_url: str,
    image: str | None,
    command: str,
    model_config_key: str,
    experiment_config: str | None = None,
    target: str = "local",
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        **git_info(),
        "command": command,
        "model_config_key": model_config_key,
        "hardware": hardware_manifest(platform, target=target),
        "software": software_manifest(backend, base_url, image, target=target),
    }
    if experiment_config:
        manifest["experiment_config"] = experiment_config
    return manifest


def quality_manifest(
    *,
    backend: str,
    platform: str,
    base_url: str,
    image: str | None,
    command: str,
    model_config_key: str,
    target: str = "local",
) -> dict[str, Any]:
    """The flat manifest shape schemas/quality_result.schema.json expects -
    quality evaluations (BFCL, MMLU) don't need the full nested
    hardware/software split a performance result does (no power/thermal/RAM
    telemetry to attribute), but DO need the same backend-identity fields:
    a tool-calling score is as backend/version-dependent as a latency
    number - both of this lab's known tool-calling failures (llama.cpp's
    temperature sensitivity, Bielik's empty tool_calls on vLLM) are
    backend/version effects, not model effects, and are uninterpretable
    without this block."""
    info = git_info()
    manifest: dict[str, Any] = {
        "git_commit": info["git_commit"],
        "git_dirty": info["git_dirty"],
        "command": command,
        "model_config_key": model_config_key,
        "backend": backend,
        "backend_version": probe_backend_version(base_url, backend),
        "container_image": image or "n/a (remote target - image not owned by this host)",
        "platform": platform,
        "target": target,
    }
    return manifest


# --- experiment identity ---------------------------------------------------


def experiment_id(
    *,
    platform: str,
    model_config_key: str,
    experiment: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    batch_size: int = 1,
    replicate: int = 1,
    on: date | None = None,
) -> str:
    """docs/note.md §30's convention, generated rather than hand-typed:

        <date>_<platform>_<model-config-key>_<experiment>_in<N>_out<N>_bs<N>_rNN

    Human-readable on purpose - an opaque UUID would satisfy uniqueness but
    not the other half of §30, which is that a result should be traceable by
    eye back to the configuration that produced it."""
    parts = [
        (on or date.today()).isoformat(),
        platform,
        model_config_key,
        experiment,
    ]
    if input_tokens is not None:
        parts.append(f"in{input_tokens}")
    if output_tokens is not None:
        parts.append(f"out{output_tokens}")
    parts.append(f"bs{batch_size}")
    parts.append(f"r{replicate:02d}")
    return "_".join(parts)


def write_result(
    result: dict[str, Any],
    *,
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    filename: str = "result.json",
) -> Path:
    """Write one result document to results/raw/<experiment_id>/<filename>.

    Refuses to overwrite: docs/note.md §29 makes historical results
    immutable, and the failure mode this prevents is a re-run silently
    replacing the data a published figure was drawn from. Re-running the same
    cell means a new replicate (`r02`), which the ID already encodes."""
    exp_id = result.get("experiment_id") or result.get("result_id")
    if not exp_id:
        raise ValueError("result has neither experiment_id nor result_id - cannot place it")
    out_dir = Path(results_root) / "raw" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Historical results are immutable "
            f"(docs/note.md §29) - bump the replicate in the experiment_id "
            f"instead of overwriting."
        )
    path.write_text(json.dumps(result, indent=2))
    return path


# --- energy ----------------------------------------------------------------

#: Rails summed into the energy figures by default. The per-domain rails are
#: the ones the inference workload actually moves; VIN_SYS_5V0 is a separate
#: board-level supply, so including it would silently change the quantity
#: being reported. docs/note.md §14 - and `rails_included` records the choice
#: in the result, because a GPU-only figure and a board-total figure must
#: never end up on one axis.
#:
#: Both boards' names are listed because they are mutually exclusive in
#: practice: Orin emits VDD_GPU_SOC/VDD_CPU_CV, Thor emits
#: VDD_GPU/VDD_CPU_SOC_MSS, and whichever board is running contributes exactly
#: two of these four. Added 2026-09-10 after the first Thor run through this
#: pipeline produced NO energy figure at all - the tuple held only Orin's
#: names, so `present` came back empty on Thor and the run silently reported
#: no joules despite sampling power the whole time.
#:
#: **The two boards' sums are close but not identical in decomposition**: Orin
#: bundles GPU+SoC against CPU+CV, Thor splits GPU alone from CPU+SoC+MSS. The
#: union of domains is nearly the same, so the sums are broadly comparable, but
#: a cross-platform energy claim must cite `rails_included` rather than assume
#: the two figures were built the same way.
DEFAULT_ENERGY_RAILS = (
    "vdd_gpu_soc_mw", "vdd_cpu_cv_mw",      # Orin
    "vdd_gpu_mw", "vdd_cpu_soc_mss_mw",     # Thor
)


def energy_block(
    rail_stats: dict[str, dict[str, float] | None],
    *,
    window_s: float,
    n_requests: int,
    total_output_tokens: int | None,
    rails: tuple[str, ...] = DEFAULT_ENERGY_RAILS,
) -> dict[str, Any]:
    """Turn per-rail average power over the measurement window into joules.

    Returns only what the inputs actually support: with no rail samples there
    is no energy figure, and with no token count there is no J/token - both
    are omitted rather than defaulted to 0, so a reader can tell "not
    measured" from "measured as zero"."""
    present = {r: rail_stats.get(r) for r in rails if rail_stats.get(r)}
    block: dict[str, Any] = {
        "rails_mw": {r: s for r, s in rail_stats.items() if s},
        "measurement_window_s": round(window_s, 3),
        "rails_included": list(present),
    }
    if not present or window_s <= 0:
        return block

    total_mw = sum(s["avg"] for s in present.values())
    energy_j = total_mw / 1000.0 * window_s
    block["energy_joules"] = round(energy_j, 3)
    if n_requests > 0:
        block["energy_per_request_j"] = round(energy_j / n_requests, 4)
    if total_output_tokens:
        block["energy_per_output_token_j"] = round(energy_j / total_output_tokens, 6)
    return block
