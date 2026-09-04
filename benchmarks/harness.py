"""Benchmark harness - a copy of jetson-vlm-lab's benchmarks/harness.py
(itself a copy of embedded-ai-chain's Phase 2a harness), not an import of
either. Nothing in this file is image/VLM-specific - it never referenced a
frame or a prompt, only setup_fn/work_fn timing plus tegrastats/RSS - so it
carries over unchanged. This repo is meant to be cloned and run standalone,
the same way jetson-yolov8-trt's own scripts/benchmark_engine.py
reimplements its own percentiles() rather than reaching into a parent repo -
see this repo's README for why. Keep this in sync by hand if either
sibling's harness gains a fix or feature worth having here too; there's no
automated sync.

Shared measurement code so each benchmark script only has to supply a
`setup_fn` (one-time, expensive: start the vLLM container) and a `work_fn`
(one repetition of the thing being measured: one VLM query). The harness
supplies everything embedded-ai-chain's docs/TODO.md Phase 2a checklist asks
for, which this repo's benchmarks are held to the same standard as:

  - N repetitions, p50/p95/p99 (never means, per this project's convention)
  - cold start (setup_fn's own duration) reported separately from warm-run
    latency — a TensorRT engine load or a model weights load is not
    inference time
  - thermal steady-state: warmup keeps running until the GPU/SoC junction
    temperature (`tj`) is flat across a rolling window, not a fixed guess —
    falls back to a warmup iteration/time cap and *says so*
    (`warmup_reached_steady_state=False`) rather than silently pretending
    equilibrium was reached
  - power sampling from `tegrastats`'s rails (VDD_GPU_SOC, VDD_CPU_CV,
    VIN_SYS_5V0), averaged over just the measurement window
  - peak/steady process RSS, plus system RAM from tegrastats as the
    unified-memory proxy for "GPU memory" — Orin has no discrete VRAM to
    query separately; tegrastats' RAM figure already includes whatever CUDA/
    TensorRT allocated, while process RSS alone would miss device-side
    allocations
  - every result records model, precision, nvpmodel mode, jetson_clocks
    (best-effort sysfs heuristic — `jetson_clocks --show` isn't queryable
    without sudo from inside a container, see the sibling `benchmark_engine.py`
    in jetson-yolov8-trt for the same tradeoff), and L4T version

Usage as a library:

    from benchmarks.harness import run_benchmark

    def setup():
        return SomeModel.load(...)          # timed as cold start

    def work(model):
        model.infer(fixed_input)            # timed per warmup/measured rep

    result = run_benchmark(setup, work, label="my_component", model="foo", precision="fp16")

See scripts/benchmark.py in this repo for the real caller.
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
_GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")
_GPU_TEMP_RE = re.compile(r"gpu@([\d.]+)C")
_TJ_TEMP_RE = re.compile(r"tj@([\d.]+)C")
_VDD_GPU_SOC_RE = re.compile(r"VDD_GPU_SOC (\d+)mW/(\d+)mW")
_VDD_CPU_CV_RE = re.compile(r"VDD_CPU_CV (\d+)mW/(\d+)mW")
_VIN_SYS_5V0_RE = re.compile(r"VIN_SYS_5V0 (\d+)mW/(\d+)mW")
_VMRSS_RE = re.compile(r"VmRSS:\s+(\d+)\s+kB")
_L4T_RE = re.compile(r"# R(\d+) \(release\), REVISION: ([\d.]+)")


def parse_tegrastats_line(line: str) -> dict[str, float] | None:
    """Best-effort field extraction from one `tegrastats` line. Returns None
    for a line that doesn't look like tegrastats output at all; individual
    fields are simply omitted (not defaulted to 0) if a rail/field isn't
    present, so callers can tell "missing" from "zero"."""
    if "RAM" not in line:
        return None
    fields: dict[str, float] = {}
    if m := _RAM_RE.search(line):
        fields["ram_used_mb"] = float(m.group(1))
        fields["ram_total_mb"] = float(m.group(2))
    if m := _GR3D_RE.search(line):
        fields["gr3d_freq_pct"] = float(m.group(1))
    if m := _GPU_TEMP_RE.search(line):
        fields["gpu_temp_c"] = float(m.group(1))
    if m := _TJ_TEMP_RE.search(line):
        fields["tj_temp_c"] = float(m.group(1))
    if m := _VDD_GPU_SOC_RE.search(line):
        fields["vdd_gpu_soc_mw"] = float(m.group(1))
    if m := _VDD_CPU_CV_RE.search(line):
        fields["vdd_cpu_cv_mw"] = float(m.group(1))
    if m := _VIN_SYS_5V0_RE.search(line):
        fields["vin_sys_5v0_mw"] = float(m.group(1))
    return fields or None


class TegrastatsSampler:
    """Runs `tegrastats` in the background and keeps a timestamped
    (time.monotonic()) sample history — monotonic because this device's NTP
    sync is known-broken (see docs/TODO.md's risk register); nothing here
    depends on wall-clock time being correct."""

    def __init__(self, interval_ms: int = 500):
        self._interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[tuple[float, dict[str, float]]] = []
        self._stop = threading.Event()

    def start(self, ready_timeout_s: float = 3.0) -> None:
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self._interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + ready_timeout_s
        while not self._samples and time.monotonic() < deadline:
            time.sleep(0.05)

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parsed = parse_tegrastats_line(line)
            if parsed:
                with self._lock:
                    self._samples.append((time.monotonic(), parsed))

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def latest(self) -> dict[str, float] | None:
        with self._lock:
            return self._samples[-1][1] if self._samples else None

    def is_thermally_stable(self, window: int = 10, threshold_c: float = 0.5) -> bool:
        """True once the last `window` samples' junction temperature (`tj`,
        falling back to `gpu`) varies by no more than threshold_c — the
        "discard first N runs, run to equilibrium" check from docs/TODO.md,
        driven by an actual temperature reading rather than a guessed
        iteration count."""
        with self._lock:
            recent = list(self._samples[-window:])
        if len(recent) < window:
            return False
        temps = [s.get("tj_temp_c", s.get("gpu_temp_c")) for _, s in recent]
        temps = [t for t in temps if t is not None]
        if len(temps) < window:
            return False
        return (max(temps) - min(temps)) <= threshold_c

    def samples_between(self, start: float, end: float) -> list[dict[str, float]]:
        with self._lock:
            return [p for (ts, p) in self._samples if start <= ts <= end]


def process_rss_mb(pid: int | str = "self") -> float | None:
    """Reads this (or another) process's resident set size from /proc,
    stdlib-only — deliberately not a psutil dependency, since this repo's
    .venv is kept minimal per docs/environment.md's GPU-wheel-shadowing
    concerns and this needs nothing psutil offers beyond one /proc field."""
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    m = _VMRSS_RE.search(text)
    return int(m.group(1)) / 1024 if m else None


def nvpmodel_mode() -> str:
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if "NV Power Mode" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown (nvpmodel not queryable from this environment)"


def jetson_clocks_locked_heuristic() -> bool | None:
    """Best-effort, no-sudo check: True only if CPU frequency (every core)
    and GPU devfreq are actually pinned - `scaling_min_freq == scaling_max_freq`
    (and the GPU's `min_freq == max_freq`) - the real invariant `jetson_clocks`
    creates. Not a substitute for `jetson_clocks --show` (needs sudo, and
    isn't installed inside Docker containers at all).

    Two weaker versions of this check were tried and both gave a wrong
    answer live on this exact device, caught by a parallel session
    investigating why measured latencies looked off (see docs/TODO.md's risk
    register):
      1. `cur_freq == max_freq` (same approach as jetson-yolov8-trt/scripts/
         benchmark_engine.py's identically-named helper) - a **false
         positive**: the default dynamic governors (`schedutil` on CPU,
         `nvhost_podgov` on GPU) can sit at max frequency under sustained
         load without being locked there at all.
      2. `scaling_governor == "performance"` - a **false negative**: on this
         device, `jetson_clocks` pins `scaling_min_freq` up to
         `scaling_max_freq` without ever renaming the governor away from
         `schedutil`/`nvhost_podgov`. Checking the governor's name checks the
         wrong thing entirely.
    min==max is what's actually true when - and only when - frequency is
    genuinely pinned, regardless of what the governor is named."""
    cpu_paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq"))
    if not cpu_paths:
        return None
    try:
        cpu_locked = all(
            int((p / "scaling_min_freq").read_text()) == int((p / "scaling_max_freq").read_text())
            for p in cpu_paths
        )
    except (OSError, ValueError):
        return None

    gpu_locked = None
    for devfreq_dir in Path("/sys/devices").glob("platform/*/*.gpu/devfreq/*.gpu"):
        try:
            gpu_locked = int((devfreq_dir / "min_freq").read_text()) == int((devfreq_dir / "max_freq").read_text())
        except (OSError, ValueError):
            continue
        break
    if gpu_locked is None:
        return None

    return cpu_locked and gpu_locked


def l4t_version() -> str | None:
    try:
        text = Path("/etc/nv_tegra_release").read_text()
    except OSError:
        return None
    m = _L4T_RE.search(text)
    return f"R{m.group(1)}.{m.group(2)}" if m else None


def percentiles(values: list[float]) -> dict[str, float]:
    """p50/p95/p99 + min/max/n. Never a mean, per this project's convention
    (CLAUDE.md / docs/TODO.md)."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        return s[min(int(n * p), n - 1)]

    return {"min": s[0], "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "max": s[-1], "n": n}


def _rail_stats(samples: list[dict[str, float]], key: str) -> dict[str, float] | None:
    vals = [s[key] for s in samples if key in s]
    if not vals:
        return None
    return {"avg": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def _half_average(values: list[float]) -> float | None:
    """Average of the second half of a series - a cheap "has it settled"
    proxy for steady-state RSS/RAM without a second full stability check."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    half = values[len(values) // 2:]
    return sum(half) / len(half)


@dataclass
class BenchmarkResult:
    label: str
    model: str
    precision: str
    timestamp: str
    n_runs: int
    n_warmup: int
    warmup_reached_steady_state: bool
    cold_start_ms: float
    latency_ms: dict[str, float]
    rss_mb: dict[str, float | None] = field(default_factory=dict)
    system_ram_mb: dict[str, float | None] = field(default_factory=dict)
    power_mw: dict[str, dict[str, float] | None] = field(default_factory=dict)
    thermal_c: dict[str, float | None] = field(default_factory=dict)
    nvpmodel_mode: str = ""
    jetson_clocks_locked: bool | None = None
    l4t_version: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _append_results_json(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    rows.append(row)
    out.write_text(json.dumps(rows, indent=2))


def run_benchmark(
    setup_fn: Callable[[], T],
    work_fn: Callable[[T], Any],
    teardown_fn: Callable[[T], None] | None = None,
    *,
    label: str,
    model: str,
    precision: str = "unknown",
    warmup_min: int = 10,
    warmup_max: int = 500,
    warmup_timeout_s: float = 30.0,
    runs: int = 300,
    min_measurement_s: float = 5.0,
    steady_state_window: int = 10,
    steady_state_threshold_c: float = 0.5,
    tegrastats_interval_ms: int = 500,
    results_json: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    quiet: bool = False,
) -> BenchmarkResult:
    """Runs setup_fn() once (timed as cold start), then work_fn(handle)
    repeatedly: first as warmup (discarded, run until `tj` is flat across
    `steady_state_window` samples or a max iteration/time cap is hit,
    whichever first), then measured repetitions: at least `runs` of them,
    and for at least `min_measurement_s` wall-clock seconds - a fast op (a
    few ms) run only `runs` times would otherwise finish before
    `tegrastats` (500ms default interval) produces enough samples for a
    meaningful power/thermal reading, so the floor keeps measuring past
    `runs` until enough wall-clock time has actually elapsed. Power/thermal
    are sampled continuously in the background throughout; RSS is sampled
    every 20th measured repetition."""
    sampler = TegrastatsSampler(tegrastats_interval_ms)
    sampler.start()
    handle: T | None = None
    try:
        t0 = time.perf_counter()
        handle = setup_fn()
        cold_start_ms = (time.perf_counter() - t0) * 1000

        warmup_deadline = time.monotonic() + warmup_timeout_s
        warmup_count = 0
        steady = False
        while warmup_count < warmup_max:
            work_fn(handle)
            warmup_count += 1
            if warmup_count >= warmup_min and sampler.is_thermally_stable(
                steady_state_window, steady_state_threshold_c
            ):
                steady = True
                break
            if time.monotonic() >= warmup_deadline:
                break

        rss_before = process_rss_mb()
        start_sample = sampler.latest() or {}
        thermal_start = start_sample.get("tj_temp_c", start_sample.get("gpu_temp_c"))

        latencies: list[float] = []
        rss_samples: list[float] = []
        t_meas_start = time.monotonic()
        i = 0
        while i < runs or (time.monotonic() - t_meas_start) < min_measurement_s:
            t0 = time.perf_counter()
            work_fn(handle)
            latencies.append((time.perf_counter() - t0) * 1000)
            if i % 20 == 0:
                r = process_rss_mb()
                if r is not None:
                    rss_samples.append(r)
            i += 1
        t_meas_end = time.monotonic()
        r = process_rss_mb()
        if r is not None:
            rss_samples.append(r)

        end_sample = sampler.latest() or {}
        thermal_end = end_sample.get("tj_temp_c", end_sample.get("gpu_temp_c"))

        power_samples = sampler.samples_between(t_meas_start, t_meas_end)
        power_mw = {
            rail: _rail_stats(power_samples, rail)
            for rail in ("vdd_gpu_soc_mw", "vdd_cpu_cv_mw", "vin_sys_5v0_mw")
        }
        ram_samples = [s["ram_used_mb"] for s in power_samples if "ram_used_mb" in s]
        meas_temps = [s.get("tj_temp_c", s.get("gpu_temp_c")) for s in power_samples]
        meas_temps = [t for t in meas_temps if t is not None]

        result = BenchmarkResult(
            label=label,
            model=model,
            precision=precision,
            timestamp=datetime.datetime.now().astimezone().isoformat(),
            n_runs=len(latencies),
            n_warmup=warmup_count,
            warmup_reached_steady_state=steady,
            cold_start_ms=cold_start_ms,
            latency_ms=percentiles(latencies),
            rss_mb={
                "before": rss_before,
                "peak": max(rss_samples) if rss_samples else None,
                "steady": _half_average(rss_samples),
            },
            system_ram_mb={
                "peak": max(ram_samples) if ram_samples else None,
                "steady": _half_average(ram_samples),
            },
            power_mw=power_mw,
            thermal_c={"start": thermal_start, "end": thermal_end, "max": max(meas_temps) if meas_temps else None},
            nvpmodel_mode=nvpmodel_mode(),
            jetson_clocks_locked=jetson_clocks_locked_heuristic(),
            l4t_version=l4t_version(),
            extra_metadata=extra_metadata or {},
        )
    finally:
        sampler.stop()
        if handle is not None and teardown_fn is not None:
            teardown_fn(handle)

    if not quiet:
        print(json.dumps(result.to_dict(), indent=2))
    if results_json:
        _append_results_json(results_json, result.to_dict())
    return result


# No CLI self-test here (embedded-ai-chain's copy has one against a real
# YOLOv8n TensorRT engine - not applicable in this repo). scripts/benchmark.py
# is the real caller; import run_benchmark/percentiles as a library.
