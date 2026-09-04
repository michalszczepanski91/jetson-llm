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


def hardware_manifest(platform: str) -> dict[str, Any]:
    """The hardware block of a benchmark_result manifest. Imports harness
    lazily so this module stays usable (and unit-testable) without it."""
    from harness import jetson_clocks_locked_heuristic, l4t_version, nvpmodel_mode

    return {
        "platform": platform,
        "board": board_model() or "unknown",
        "memory_total_mb": memory_total_mb() or 0.0,
        "nvpmodel_mode": nvpmodel_mode(),
        "jetson_clocks_locked": jetson_clocks_locked_heuristic(),
        "l4t_version": l4t_version(),
    }


def software_manifest(backend: str, base_url: str, image: str | None) -> dict[str, Any]:
    import platform as _platform

    block: dict[str, Any] = {
        "backend": backend,
        "backend_version": probe_backend_version(base_url, backend),
        "container_image": image or "n/a (remote target - image not owned by this host)",
        "python": _platform.python_version(),
    }
    if image:
        digest = image_digest(image)
        if digest:
            block["container_digest"] = digest
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
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        **git_info(),
        "command": command,
        "model_config_key": model_config_key,
        "hardware": hardware_manifest(platform),
        "software": software_manifest(backend, base_url, image),
    }
    if experiment_config:
        manifest["experiment_config"] = experiment_config
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

#: Rails summed into the energy figures by default. GPU+SoC and CPU+CV are
#: the two the inference workload actually moves; VIN_SYS_5V0 is a separate
#: board-level supply, so including it would silently change the quantity
#: being reported. docs/note.md §14 - and `rails_included` records the choice
#: in the result, because a GPU-only figure and a board-total figure must
#: never end up on one axis.
DEFAULT_ENERGY_RAILS = ("vdd_gpu_soc_mw", "vdd_cpu_cv_mw")


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
