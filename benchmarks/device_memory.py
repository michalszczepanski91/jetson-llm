"""Does this config fit, and what is left for everything else?

Nothing in this lab's existing results answers that. `benchmarks/runner.py`
writes a `memory` block whose own `attribution_caveat` admits it: system RAM
from tegrastats, client RSS (an HTTP client's, not the model's), and an
explicit "weights_mb/kv_cache_mb are not broken out: neither backend reports
them over its HTTP API". True of the HTTP API, and that turned out to be the
wrong place to look - both containerised backends print exactly these
numbers to their own startup log, and the third writes them into its engine
build config.

On a unified-memory SoC this is the question that decides a deployment.
`embedded-ai-chain` runs YOLO, STT and TTS on the same 122GB pool as the
orchestrator LLM; a config that is 40ms faster per turn and leaves 3GB for
the perception stack is not a better config, it is an unshippable one. So
five numbers, and each is sourced explicitly:

  weights_bytes_on_device   what the runtime says it allocated for weights
  kv_cache_bytes_per_token  computed from the model config, cross-checked
                            against the runtime's own reported cache size
  peak_device_bytes         measured, board-level, over the whole session
  max_context_measured      MEASURED by probing until it fails - not
                            max_model_len read back from the config file
  free_bytes_remaining      total - peak

The distinction that makes `max_context_measured` worth the probe requests:
a server refusing a 9000-token prompt because `--max-model-len 8192` says so
is a *configuration* limit, and a server accepting it and dying on a KV
allocation is a *memory* limit. They have different fixes and the number
alone cannot tell them apart, so `limiting_factor` is recorded next to it.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

_MEMINFO_RE = re.compile(r"^(\w+):\s+(\d+)\s+kB", re.M)


def device_memory_by_pid() -> dict[int, int] | None:
    """Per-process GPU memory in bytes, from `nvidia-smi --query-compute-apps`.

    **This is the only instrument on this board that sees the model.** Two
    plausible alternatives were tested on Thor on 2026-09-11 and both fail:

      * **cgroup v2 accounting misses it entirely.** With vLLM's own log
        reporting `Model loading took 14.25 GiB`, the container's
        `memory.current` read 1.80 GiB and `memory.peak` 5.37 GiB, of which
        `anon` was 1.57 GiB. CUDA allocates through nvmap, which is not
        charged to the container's memory cgroup.
      * **Board-level deltas from /proc/meminfo under-report.** 55.2 GB of
        this board is unaccounted by meminfo at all (MemTotal minus free,
        file cache, anon and slab) - it is the Tegra GPU carveout - and the
        NVIDIA driver's pool does NOT shrink back when a container exits. So
        a second run against a warm pool sees almost no delta: a session that
        allocated 14.25 GiB of weights measured a 4.5 GiB rise in board
        `used`, and tegrastats agreed with that wrong number because it
        measures the same quantity.

    Returns None when nvidia-smi is unavailable rather than zero, so a board
    without it reports the gap instead of a fabricated footprint."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    found: dict[int, int] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            found[int(parts[0])] = int(parts[1]) * 1024 ** 2
    return found


def device_memory_total() -> int | None:
    by_pid = device_memory_by_pid()
    return sum(by_pid.values()) if by_pid is not None else None


def system_memory() -> dict[str, int]:
    """Board memory from /proc/meminfo, in bytes.

    `MemAvailable` rather than `MemFree` is the one a co-tenancy decision
    should use: on this board 80GB sits in page cache that a new allocation
    can reclaim, and MemFree would report that as unavailable and
    understate headroom by an order of magnitude."""
    text = open("/proc/meminfo").read()
    fields = {k: int(v) * 1024 for k, v in _MEMINFO_RE.findall(text)}
    return {
        "total_bytes": fields.get("MemTotal", 0),
        "free_bytes": fields.get("MemFree", 0),
        "available_bytes": fields.get("MemAvailable", 0),
        "cached_bytes": fields.get("Cached", 0),
        "used_bytes": fields.get("MemTotal", 0) - fields.get("MemAvailable", 0),
    }


# --------------------------------------------------------------------------
# what the runtime itself reports
# --------------------------------------------------------------------------

#: Patterns each backend prints at startup. Every one of these was read off
#: a real log from this box rather than from documentation; a pattern that
#: stops matching yields None, never a guess, and `matched_patterns` in the
#: output says which ones fired so a silent regression is visible.
_LOG_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    "vllm": [
        # Confirmed against a real Thor startup log, 2026-09-11. The first
        # pattern is the one that actually fires on vLLM 0.19; the other two
        # are older phrasings kept so this keeps working across image bumps.
        ("weights_bytes_on_device", r"Model loading took ([\d.]+)\s*GiB", "GiB"),
        ("weights_bytes_on_device", r"[Mm]odel weights take ([\d.]+)\s*GiB", "GiB"),
        ("weights_bytes_on_device", r"Loading model weights took ([\d.]+)\s*GiB", "GiB"),
        ("kv_cache_bytes_reserved", r"GPU KV cache size: ([\d,]+) tokens", "tokens"),
        ("kv_cache_bytes_reserved", r"Available KV cache memory[:=]\s*([\d.]+)\s*GiB", "GiB"),
        ("activation_peak_bytes", r"PyTorch activation peak memory takes ([\d.]+)\s*GiB", "GiB"),
        ("non_torch_bytes", r"non-torch memory takes ([\d.]+)\s*GiB", "GiB"),
        ("graph_capture_bytes", r"CUDA graph pool memory: ([\d.]+)\s*GiB \(actual\)", "GiB"),
        ("graph_capture_bytes", r"Graph capturing finished in \d+ secs, took ([\d.]+)\s*GiB", "GiB"),
        ("max_context_configured", r"[Mm]aximum concurrency for ([\d,]+) tokens", "tokens"),
    ],
    "llama-cpp": [
        ("weights_bytes_on_device", r"load_tensors:\s+CUDA\d+ model buffer size\s*=\s*([\d.]+) MiB", "MiB"),
        ("weights_bytes_on_device", r"llm_load_tensors:\s+CUDA\d+ buffer size\s*=\s*([\d.]+) MiB", "MiB"),
        ("kv_cache_bytes_reserved", r"(?:llama_kv_cache\w*|init):\s+CUDA\d+ KV buffer size\s*=\s*([\d.]+) MiB", "MiB"),
        ("compute_buffer_bytes", r"CUDA\d+ compute buffer size\s*=\s*([\d.]+) MiB", "MiB"),
        ("offloaded_layers", r"offloaded (\d+)/(\d+) layers to GPU", "layers"),
    ],
    "edge-llm": [
        ("weights_bytes_on_device", r"[Ww]eights?(?: memory)?[:=]\s*([\d.]+)\s*GiB", "GiB"),
        ("kv_cache_bytes_reserved", r"KV cache(?: memory)?[:=]\s*([\d.]+)\s*GiB", "GiB"),
    ],
}

_UNIT_BYTES = {"GiB": 1024 ** 3, "MiB": 1024 ** 2, "KiB": 1024, "B": 1}


def container_log(container_name: str, tail: int = 4000) -> str | None:
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container_name],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout + out.stderr


def parse_backend_memory(backend: str, log_text: str | None) -> dict[str, Any]:
    """Pull whatever the runtime said about its own allocation out of its log.

    Returns a dict with a `_source` per field and a `matched_patterns` list.
    Fields that no pattern matched are simply absent - a caller turning this
    into a results row is expected to render them as `not_collected`, never
    as zero."""
    found: dict[str, Any] = {}
    matched: list[str] = []
    if not log_text:
        return {"_source": "no log available", "matched_patterns": []}
    for field, pattern, unit in _LOG_PATTERNS.get(backend, []):
        if field in found:
            continue
        m = re.search(pattern, log_text)
        if not m:
            continue
        raw = m.group(1).replace(",", "")
        matched.append(pattern)
        if unit == "tokens":
            found[field.replace("_bytes_reserved", "_tokens_total")] = int(float(raw))
            if field == "max_context_configured":
                found["max_context_configured"] = int(float(raw))
        elif unit == "layers":
            found["offloaded_layers"] = int(raw)
            found["total_layers"] = int(m.group(2))
        else:
            found[field] = int(float(raw) * _UNIT_BYTES[unit])
    found["_source"] = f"{backend} startup log (docker logs)"
    found["matched_patterns"] = matched
    return found


# --------------------------------------------------------------------------
# the measured largest context
# --------------------------------------------------------------------------

def _attempt(base_url: str, model: str, text: str, timeout: float) -> dict[str, Any]:
    """One max_tokens=1 completion. Success carries the server's own
    prompt_tokens, which is what makes the returned context length a
    measurement rather than an estimate."""
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1, "temperature": 0.0,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        return {"ok": True, "prompt_tokens": body.get("usage", {}).get("prompt_tokens")}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        return {"ok": False, "status": exc.code, "detail": detail}
    except Exception as exc:  # noqa: BLE001 - a refusal is data, not a crash
        return {"ok": False, "status": None, "detail": f"{type(exc).__name__}: {exc}"}


_CONFIG_LIMIT_RE = re.compile(
    r"maximum context length|max_model_len|longer than the maximum|"
    r"n_ctx|context (?:size|window)|exceeds|too long|max_input_len",
    re.I,
)
_OOM_RE = re.compile(r"out of memory|OOM|cudaErrorMemoryAllocation|CUDA error", re.I)


def probe_max_context(
    base_url: str,
    model: str,
    make_text: Callable[[int], str],
    *,
    lo: int = 128,
    hi_cap: int = 131072,
    timeout: float = 600.0,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Largest prompt the server actually serves - measured, not read back.

    Doubles from `lo` until something fails, then bisects. Every attempt is
    handed to `on_attempt` so it lands in records.jsonl: the failures are as
    much a result as the successes (docs/note.md §54 - record the OOM, do
    not skip it), and they are what `limiting_factor` is classified from.

    `make_text(n)` must produce a prompt of approximately n tokens; the
    approximation only steers the search, and every reported length is the
    server's own `prompt_tokens` from a request that succeeded."""
    attempts: list[dict[str, Any]] = []

    def try_at(n: int) -> dict[str, Any]:
        res = _attempt(base_url, model, make_text(n), timeout)
        res["requested_tokens"] = n
        attempts.append(res)
        if on_attempt:
            on_attempt(res)
        return res

    first = try_at(lo)
    if not first["ok"]:
        return {
            "max_context_measured": None,
            "_not_collected": f"the floor probe at ~{lo} tokens already failed: {first['detail'][:200]}",
            "attempts": attempts,
        }

    best_ok = first["prompt_tokens"] or lo
    good = lo
    bad: int | None = None
    n = lo
    while n < hi_cap:
        n *= 2
        res = try_at(n)
        if res["ok"]:
            good = n
            best_ok = max(best_ok, res["prompt_tokens"] or n)
        else:
            bad = n
            break
    if bad is None:
        return {
            "max_context_measured": best_ok,
            "limiting_factor": "not found - every probe up to the search cap succeeded",
            "search_cap_tokens": hi_cap,
            "attempts": attempts,
        }

    failure = attempts[-1]
    # Bisect to ~3% of the answer rather than to the token. A long-context
    # probe request costs seconds of prefill, and nobody makes a deployment
    # decision on the difference between a 16,161-token limit and a
    # 16,380-token one - but the residual width must be reported, not
    # implied away by printing an exact-looking number.
    tolerance = max(64, good // 32)
    while bad - good > tolerance:
        mid = (good + bad) // 2
        res = try_at(mid)
        if res["ok"]:
            good = mid
            best_ok = max(best_ok, res["prompt_tokens"] or mid)
        else:
            bad = mid
            failure = res

    detail = failure.get("detail") or ""
    if _OOM_RE.search(detail):
        factor = "memory (allocation failure)"
    elif _CONFIG_LIMIT_RE.search(detail):
        factor = "configuration (the server's own declared context limit, not memory)"
    else:
        factor = f"unclassified - HTTP {failure.get('status')}"
    return {
        "max_context_measured": best_ok,
        "search_tolerance_tokens": tolerance,
        "search_tolerance_note": (
            f"binary search stopped once the bracket closed to {tolerance} tokens, so the "
            f"true limit lies in [{best_ok}, {bad}); the reported value is the largest "
            "prompt VERIFIED to complete, never an extrapolation"
        ),
        "first_failure_at_requested_tokens": bad,
        "limiting_factor": factor,
        "failure_detail": detail[:400],
        "attempts": attempts,
    }


class MemorySampler:
    """Polls /proc/meminfo on a background thread for the life of a run.

    Separate from `TegrastatsSampler` on purpose, and not a duplicate of it.
    tegrastats reports a `RAM x/y MB` figure whose definition is NVIDIA's,
    and this lab needs the two quantities the kernel defines: `MemAvailable`
    (what a new allocation could actually get) and `MemTotal - MemAvailable`
    (what is spoken for). On this board the difference is not academic - an
    idle Thor shows ~79GB in page cache, so a used-based reading calls 32GB
    "used" while 90GB is genuinely available, and a co-tenancy verdict built
    on the first number would reject configs that fit comfortably.

    The trough of `available_bytes` over a session is the number that
    answers the deployment question: it is the least memory the perception
    stack would have found free at the worst moment of the run."""

    def __init__(self, interval_s: float = 0.5, device_every: int = 4):
        self._interval = interval_s
        #: nvidia-smi costs ~50ms per call, so it is sampled every Nth tick
        #: rather than every tick. The quantity it measures is a pre-allocated
        #: pool that changes only at startup, so 2s resolution is ample.
        self._device_every = device_every
        self._samples: list[tuple[float, dict[str, int]]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples.append((time.monotonic(), self._sample(force_device=True)))
        self._thread = threading.Thread(target=self._loop, daemon=True, name="meminfo-sampler")
        self._thread.start()

    def _sample(self, force_device: bool = False) -> dict[str, int]:
        row = system_memory()
        if force_device or len(self._samples) % self._device_every == 0:
            total = device_memory_total()
            if total is not None:
                row["device_bytes"] = total
        return row

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._samples.append((time.monotonic(), self._sample()))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def samples_between(self, start: float, end: float) -> list[dict[str, int]]:
        return [s for ts, s in self._samples if start <= ts <= end]

    def summary(self, start: float | None = None, end: float | None = None) -> dict[str, Any]:
        window = (
            self.samples_between(start, end) if start is not None and end is not None
            else [s for _, s in self._samples]
        )
        if not window:
            return {"n_samples": 0}
        device = [s["device_bytes"] for s in window if "device_bytes" in s]
        return {
            "n_samples": len(window),
            "peak_used_bytes": max(s["used_bytes"] for s in window),
            "min_available_bytes": min(s["available_bytes"] for s in window),
            "max_available_bytes": max(s["available_bytes"] for s in window),
            "n_device_samples": len(device),
            "peak_device_bytes": max(device) if device else None,
            "min_device_bytes": min(device) if device else None,
        }


def compose(
    *,
    backend_reported: dict[str, Any],
    kv_bytes_per_token: int,
    kv_formula: str,
    artifact_bytes: int,
    baseline: dict[str, int],
    session: dict[str, Any],
    max_context: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the memory block, keeping every number's provenance attached.

    `weights_bytes_on_device` prefers the runtime's own figure and falls back
    to the artifact size on disk, saying which it used - they are not the
    same quantity (a runtime may pad, may convert dtype on load, may keep a
    staging copy) and treating them as interchangeable is exactly the kind of
    quiet substitution this block exists to prevent.

    `baseline` is a `system_memory()` reading taken BEFORE the server
    started; `session` is a `MemorySampler.summary()` over the whole server
    session. The two together are what make the footprint attributable: the
    board's absolute numbers alone would report the OS and the page cache as
    if this config had allocated them."""
    reported_w = backend_reported.get("weights_bytes_on_device")
    block: dict[str, Any] = {
        "weights_bytes_on_device": reported_w if reported_w is not None else artifact_bytes,
        "weights_bytes_source": (
            backend_reported.get("_source") if reported_w is not None
            else "FALLBACK: artifact size on disk - the runtime did not report its own "
                 "weight allocation, so this omits any load-time padding or dtype conversion"
        ),
        "weights_bytes_artifact_on_disk": artifact_bytes,
        "kv_cache_bytes_per_token": kv_bytes_per_token,
        "kv_cache_bytes_per_token_formula": kv_formula,
        "system_total_bytes": baseline["total_bytes"],
        "baseline_available_bytes_before_server": baseline["available_bytes"],
        "baseline_used_bytes_before_server": baseline["used_bytes"],
    }
    for k in ("kv_cache_bytes_reserved", "kv_cache_tokens_total", "activation_peak_bytes",
              "non_torch_bytes", "graph_capture_bytes", "offloaded_layers", "total_layers",
              "max_context_configured"):
        if k in backend_reported:
            block[k] = backend_reported[k]

    # Cross-check: the runtime's own cache size divided by the computed
    # per-token cost should equal the token capacity it also printed. When
    # both are present and they disagree by more than a few percent, the
    # formula's assumptions (KV dtype above all) are wrong for this config,
    # and that is worth surfacing rather than averaging away.
    total = backend_reported.get("kv_cache_bytes_reserved")
    tokens = backend_reported.get("kv_cache_tokens_total")
    if total and tokens and kv_bytes_per_token:
        implied = total / tokens
        block["kv_cache_bytes_per_token_implied_by_runtime"] = round(implied, 1)
        block["kv_formula_agrees_with_runtime"] = abs(implied - kv_bytes_per_token) / kv_bytes_per_token < 0.05
    elif tokens and kv_bytes_per_token and not total:
        block["kv_cache_bytes_reserved_implied"] = tokens * kv_bytes_per_token
        block["kv_cache_bytes_reserved_implied_note"] = (
            "the runtime printed a token capacity but not a byte size; this is that "
            "capacity x the computed per-token cost, so it inherits the formula's assumptions"
        )

    # --- the footprint, from the only instrument that can see it --------
    # Device memory first; board-level meminfo second and clearly labelled.
    # On this Tegra the two disagree by more than 10x and only the first is
    # right - see device_memory_by_pid() for the two experiments that
    # established it.
    peak_dev = session.get("peak_device_bytes")
    base_dev = baseline.get("device_bytes")
    if peak_dev:
        block["peak_device_bytes"] = peak_dev
        block["peak_device_source"] = (
            "nvidia-smi --query-compute-apps, summed over every compute process, "
            "sampled through the whole server session"
        )
        block["baseline_device_bytes_before_server"] = base_dev
        if base_dev is not None:
            block["peak_attributable_bytes"] = peak_dev - base_dev
        block["free_bytes_remaining"] = baseline["total_bytes"] - peak_dev
        block["device_sample_count"] = session.get("n_device_samples")

        # Reconcile the runtime's own component breakdown against the total
        # the driver reports. The residual is real - CUDA context, library
        # workspaces, allocator slack - and naming it rather than silently
        # dropping it is what makes the stacked bar in the memory figure add
        # up to the measured total.
        parts = {k: backend_reported[k] for k in
                 ("weights_bytes_on_device", "kv_cache_bytes_reserved",
                  "graph_capture_bytes", "activation_peak_bytes")
                 if backend_reported.get(k)}
        if parts:
            accounted = sum(parts.values())
            block["component_breakdown_bytes"] = parts
            block["runtime_overhead_bytes"] = peak_dev - accounted
            block["breakdown_note"] = (
                f"the runtime itemises {accounted / 2**30:.1f} GiB of the "
                f"{peak_dev / 2**30:.1f} GiB the driver reports resident; the "
                f"{(peak_dev - accounted) / 2**30:.1f} GiB residual is CUDA context, "
                "library workspaces and allocator slack, which no backend breaks out"
            )
    else:
        block["peak_device_bytes"] = None
        block["_peak_device_not_collected"] = (
            "nvidia-smi --query-compute-apps returned nothing for this session; no other "
            "instrument on this board can see GPU-resident memory"
        )

    if session.get("n_samples"):
        # Secondary, and carrying its own health warning. Kept because it is
        # what `free -h` would show an operator, not because it attributes.
        block["board_used_peak_bytes"] = session["peak_used_bytes"]
        block["board_available_trough_bytes"] = session["min_available_bytes"]
        block["board_available_baseline_bytes"] = baseline["available_bytes"]
        block["memory_sample_count"] = session["n_samples"]
        block["board_accounting_caveat"] = (
            "These board-level figures do NOT attribute this config's footprint and must not "
            "be used as if they did. 55.2 GB of this Tegra is invisible to /proc/meminfo (the "
            "GPU carveout), and the NVIDIA driver's pool does not shrink when a server exits - "
            "so a run against a warm pool shows almost no rise. Measured 2026-09-11: a session "
            "that allocated 14.25 GiB of weights moved board `used` by 4.5 GiB, and tegrastats "
            "agreed with that wrong number because it measures the same quantity. Use "
            "peak_device_bytes."
        )

    block["max_context"] = max_context
    return block
