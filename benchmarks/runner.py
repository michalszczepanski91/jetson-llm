"""The measurement core: one workload cell in, one schema-conformant
benchmark_result document out (docs/TODO.md Phase 3).

Extracted from scripts/benchmark_streaming.py so that two callers can share
it - a single-cell CLI and a grid runner driven by configs/benchmarks/*.yaml
- and so the measurement loop itself is unit-testable, which it was not
while it lived inside a `main()` that owned Docker lifecycle and argparse.

**Server lifecycle deliberately stays with the caller.** A cold start is
~136s for vLLM on this hardware, so a grid that restarted the server per
cell would spend most of a campaign starting servers. One server session
serves every cell for its model; the cost is that all those cells share one
cold-start measurement, which is recorded as a validity flag rather than
quietly repeated as if each had been measured (see `_cold_start_for_cell`).

Two things in here are load-bearing methodology rather than plumbing, and
both were found by running the thing rather than reasoning about it:

  * **Prompts vary per repetition by default.** Both backends cache by
    prompt prefix, so an identical prompt makes every repetition after the
    first a cache hit rather than a prefill. Measured on this Orin: TTFT p50
    40.8ms with identical prompts vs 268.1ms with unique ones.
  * **Measurement continues past `runs` until `min_measurement_s`.**
    tegrastats samples at 500ms; vLLM at ~108 tok/s finishes 8 repetitions
    in 4.4s, which is 9 samples - an energy figure that would read exactly
    as authoritative as one backed by 60.
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness import _half_average, _rail_stats, percentiles, process_rss_mb  # noqa: E402
from llm_client import stream_llm  # noqa: E402
from manifest import (  # noqa: E402
    DEFAULT_ENERGY_RAILS, SCHEMA_VERSION, build_manifest, energy_block, experiment_id,
)

#: Below this many tegrastats samples inside the measurement window, the
#: energy figure is flagged. Not a hard failure - a thin number is still a
#: number - but it must not read as authoritative as one backed by a full
#: window. 20 samples is 10s at the sampler's 500ms interval.
MIN_POWER_SAMPLES = 20

#: Versioned per docs/note.md §21: a prompt change is an experimental change,
#: so it may not live silently inside the benchmark implementation.
#: v2 (2026-09-10) adds the per-cell salt - see `build_prompt`. Any `..._v1`
#: result from a multi-cell vLLM session may carry cross-cell prefix-cache
#: hits in its TTFT; `..._v2` cannot.
PROMPT_TEMPLATE_VERSION = "v2"

_BASE_PROMPT = "What do you see in front of you right now?"
_FILLER = (
    "The camera observes the room and reports what it finds. "
    "Objects appear and disappear as people move between the desk and the door. "
)

#: Every rail sampled into a result: the per-domain rails energy is computed
#: from, plus the board-level supply (recorded for context, deliberately NOT
#: summed into the energy figure - see DEFAULT_ENERGY_RAILS).
#:
#: DERIVED from DEFAULT_ENERGY_RAILS rather than restated, because restating it
#: is what caused a real bug: this tuple was a third independent copy of Orin's
#: rail names, so after harness.py and manifest.py both learned Thor's names on
#: 2026-09-10, a Thor run STILL reported `rails_included: []` and no joules -
#: measure_cell() reads this tuple, not those. One source of truth now.
_POWER_RAILS = DEFAULT_ENERGY_RAILS + ("vin_sys_5v0_mw",)


@dataclass(frozen=True)
class Cell:
    """One point of the workload grid - the unit that produces exactly one
    benchmark_result document."""

    output_tokens: int
    input_tokens: int | None = None
    batch_size: int = 1
    concurrency: int = 1


def expand_grid(workload: dict[str, Any]) -> list[Cell]:
    """Cartesian product of the experiment config's workload axes.

    Every list in an experiment config is a full axis, so a two-element list
    doubles the campaign - docs/note.md §4 warns against expanding the whole
    product at once, which is why configs/benchmarks/ keeps one or two axes
    open per experiment and pins the rest. Order is deterministic (config
    order, not set order) so a partially-completed campaign resumes
    predictably."""
    inputs = workload.get("input_tokens") or [None]
    return [
        Cell(output_tokens=o, input_tokens=i, batch_size=b, concurrency=c)
        for i, o, b, c in itertools.product(
            inputs,
            workload["output_tokens"],
            workload.get("batch_size", [1]),
            workload.get("concurrency", [1]),
        )
    ]


def cell_salt(cell: "Cell", replicate: int = 1) -> str:
    """A short fixed-width tag that differs between cells of one session.

    Exists because `build_prompt`'s run-index marker alone was NOT enough to
    defeat prefix reuse, which cost this lab a real 2x error - see that
    function's docstring."""
    key = f"{cell.input_tokens}|{cell.output_tokens}|{cell.batch_size}|{cell.concurrency}|{replicate}"
    return hashlib.sha256(key.encode()).hexdigest()[:4]


def build_prompt(
    target_tokens: int | None,
    run_index: int | None = None,
    salt: str = "0000",
) -> tuple[str, str]:
    """Return (prompt, prompt_source).

    `run_index` makes the prompt unique per repetition and `salt` makes it
    unique per *cell*, both to defeat KV-cache reuse. Both go at the FRONT
    because both backends cache by prefix - a unique suffix would leave
    everything before it reusable and change nothing. The chat template's own
    system prefix stays cacheable even so, which is realistic: a real
    deployment also has a stable system prompt.

    **Why `salt` exists (2026-09-10).** The run-index marker alone was not
    enough, and the gap was live in published data. `build_prompt` is a
    function of `input_tokens` and `run_index` only - not of `output_tokens`
    - so every cell of a sweep that holds input length fixed emitted a
    *byte-identical* prompt sequence. Under vLLM's automatic prefix caching
    (on by default in V1) the first such cell populates the cache and every
    later one reads out of it. Measured on this Orin, `output_sweep`
    (in=512, 6 output lengths, one server session): cell 0 TTFT p50 74.9ms,
    then 29.9 / 29.9 / 31.1 / 31.2 / 30.2ms for cells 1-5 - a step function
    at the cell boundary, not a decay, and flat within every cell. The
    matching `context_sweep` cell, whose input length made its prompts unique,
    read 66.0ms. That 2.1x is the whole of the discrepancy that made every
    Orin TTFT comparison - including the cross-platform one against Thor -
    unusable until it was explained. llama.cpp is the negative control: it
    keeps a single-slot prompt cache, so its two sweeps agreed (309.5 vs
    310.1ms) and only the vLLM rows were affected.

    The salt is fixed-width so it costs the same tokens in every cell, and
    deterministic so a cell is still reproducible from its identity.

    Token targets are approximate. There is no tokenizer on the host (this
    repo is stdlib-only by design and the tokenizer lives inside the serving
    container), so the ~0.75 words-per-token ratio only sets the target; the
    ACHIEVED count is read back from the server's own usage block and is what
    every derived figure uses. That is why the schema keeps
    `input_tokens_target` and per-run `input_tokens` as separate fields."""
    marker = f"[run {run_index} {salt}] " if run_index is not None else ""
    if target_tokens is None:
        return marker + _BASE_PROMPT, f"fixed_transcript_{PROMPT_TEMPLATE_VERSION}"
    words_needed = int(target_tokens * 0.75)
    filler_words = _FILLER.split()
    repeats = max(1, words_needed // len(filler_words) + 1)
    body = " ".join((_FILLER * repeats).split()[:words_needed])
    return f"{marker}{body}\n\n{_BASE_PROMPT}", f"repeated_filler_{PROMPT_TEMPLATE_VERSION}"


#: The historical baseline every result on disk before 2026-09-11 was taken
#: under. A run matching it gets no ID suffix, so those IDs stay reproducible.
_BASELINE_REGIME = ("v1", "unique-per-run")


def _prompt_regime(prompt_uniqueness: str) -> str | None:
    """The ID component for the prompt regime, or None when it matches the
    historical baseline - see `manifest.experiment_id`'s own docstring for why
    this is an identity question and not a labelling one."""
    if (PROMPT_TEMPLATE_VERSION, prompt_uniqueness) == _BASELINE_REGIME:
        return None
    tag = f"pt{PROMPT_TEMPLATE_VERSION}"
    if prompt_uniqueness != "unique-per-run":
        tag += "-identical"
    return tag


def _cold_start_for_cell(cold_start: dict[str, Any], cell_index: int) -> tuple[dict[str, Any], list[str]]:
    """One server session serves many cells, so only the first cell of a
    session actually paid for the cold start. Later cells carry the same
    numbers - they are true of the session - but flagged, so an analysis can
    never treat N cells as N independent cold-start measurements."""
    if cell_index == 0:
        return cold_start, []
    shared = dict(cold_start)
    shared["caveat"] = (
        "shared across every cell of this server session - only the first cell of the "
        "session paid for this startup. Not an independent measurement. "
    ) + cold_start.get("caveat", "")
    return shared, ["cold_start_shared_across_cells_in_this_server_session"]


def measure_cell(
    *,
    coordinator,
    variant: dict[str, Any],
    model_config_key: str,
    cell: Cell,
    sampler,
    cold_start: dict[str, Any],
    execution_condition: str,
    experiment: str,
    command: str,
    co_resident: list[str] | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    runs: int = 30,
    warmup_min: int = 3,
    warmup_max: int = 20,
    min_measurement_s: float = 10.0,
    prompt_uniqueness: str = "unique-per-run",
    replicate: int = 1,
    cell_index: int = 0,
    target: str = "local",
    experiment_config: str | None = None,
    on_progress: Callable[[int, dict], None] | None = None,
) -> dict[str, Any]:
    """Warm up, measure, and assemble one benchmark_result document.

    The server must already be running: `coordinator.wait_ready()` is the
    caller's job, as is stopping it. `cold_start` is the breakdown the
    caller captured for this server session.
    """
    unique = prompt_uniqueness == "unique-per-run"
    _, prompt_source = build_prompt(cell.input_tokens)
    salt = cell_salt(cell, replicate)

    def messages_for(i: int) -> list[dict]:
        prompt, _ = build_prompt(
            cell.input_tokens, run_index=i if unique else None, salt=salt
        )
        return [{"role": "user", "content": prompt}]

    exp_id = experiment_id(
        platform=variant["platform"],
        model_config_key=model_config_key,
        experiment=experiment,
        input_tokens=cell.input_tokens,
        output_tokens=cell.output_tokens,
        batch_size=cell.batch_size,
        replicate=replicate,
        prompt_regime=_prompt_regime(prompt_uniqueness),
    )

    # --- warmup: to thermal steady state, not a guessed count -------------
    warmup_count = 0
    steady = False
    while warmup_count < warmup_max:
        stream_llm(coordinator, messages_for(-warmup_count - 1),
                   max_tokens=cell.output_tokens, temperature=temperature)
        warmup_count += 1
        if warmup_count >= warmup_min and sampler.is_thermally_stable():
            steady = True
            break

    # --- measurement ------------------------------------------------------
    start_sample = sampler.latest() or {}
    rss_before = process_rss_mb()
    rss_samples: list[float] = []
    measured: list[dict] = []
    token_source = None
    t_meas_start = time.monotonic()
    i = 0
    while i < runs or (time.monotonic() - t_meas_start) < min_measurement_s:
        try:
            record = stream_llm(coordinator, messages_for(i),
                                max_tokens=cell.output_tokens, temperature=temperature)
        except Exception as exc:  # noqa: BLE001 - a failed run is a result, not a crash
            record = {"e2e_latency_ms": 0.0, "error": f"{type(exc).__name__}: {exc}"}
        record["run_index"] = i
        token_source = record.pop("token_source", token_source)
        measured.append(record)
        if i % 10 == 0:
            r = process_rss_mb()
            if r is not None:
                rss_samples.append(r)
            if on_progress:
                on_progress(i, record)
        i += 1
    t_meas_end = time.monotonic()
    r = process_rss_mb()
    if r is not None:
        rss_samples.append(r)
    end_sample = sampler.latest() or {}

    # --- aggregates, derived from `measured` and nothing else -------------
    ok = [r for r in measured if not r.get("error")]

    def agg(key: str):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return percentiles(vals) if vals else None

    aggregates = {k: v for k, v in {
        "ttft_ms": agg("ttft_ms"),
        "e2e_latency_ms": agg("e2e_latency_ms"),
        "decode_ms": agg("decode_ms"),
        "prefill_tok_s": agg("prefill_tok_s"),
        "decode_tok_s": agg("decode_tok_s"),
        "output_tokens": agg("output_tokens"),
        "inter_token_latency_ms": (
            percentiles([g for r in ok for g in r.get("inter_token_latency_ms") or []]) or None
        ),
    }.items() if v}

    # --- power / energy, over the measurement window only -----------------
    power_samples = sampler.samples_between(t_meas_start, t_meas_end)
    rails = {rail: _rail_stats(power_samples, rail) for rail in _POWER_RAILS}
    power = energy_block(
        rails,
        window_s=t_meas_end - t_meas_start,
        n_requests=len(ok),
        total_output_tokens=sum(r.get("output_tokens") or 0 for r in ok),
    )

    ram = [s["ram_used_mb"] for s in power_samples if "ram_used_mb" in s]
    temps = [t for t in (s.get("tj_temp_c", s.get("gpu_temp_c")) for s in power_samples) if t is not None]

    cell_cold_start, flags = _cold_start_for_cell(cold_start, cell_index)
    if target == "remote":
        flags += ["cold_start_not_measured", "telemetry_is_client_side_not_inference_host"]
    if token_source and token_source.startswith("sse delta"):
        flags.append("token_counts_are_sse_chunk_counts_not_tokens")
    n_power = max((s_["n"] for s_ in rails.values() if s_), default=0)
    if n_power < MIN_POWER_SAMPLES:
        flags.append(f"power_window_thin_{n_power}_samples")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": exp_id,
        "label": f"{model_config_key} in={cell.input_tokens or 'fixed'} out={cell.output_tokens}",
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "manifest": build_manifest(
            backend=variant["backend"],
            platform=variant["platform"],
            base_url=coordinator.base_url,
            image=getattr(coordinator, "image", None),
            command=command,
            model_config_key=model_config_key,
            experiment_config=experiment_config,
            target=target,
        ),
        "model": {
            "name": variant["model"],
            "precision": variant["precision"],
            "backend": variant["backend"],
            "platform": variant["platform"],
        },
        "workload": {
            "input_tokens_target": (
                cell.input_tokens if cell.input_tokens is not None else _achieved_or_none(ok)
            ),
            "output_tokens_target": cell.output_tokens,
            "batch_size": cell.batch_size,
            "concurrency": cell.concurrency,
            "prompt_source": prompt_source,
            "prompt_uniqueness": prompt_uniqueness,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "tools_attached": False,
            "sampling": {"temperature": temperature, "seed": seed},
        },
        "execution_condition": execution_condition,
        "cold_start": cell_cold_start,
        "runs": measured,
        "aggregates": aggregates,
        "memory": {
            "system_ram_used_mb": {"peak": max(ram) if ram else None, "steady": _half_average(ram)},
            "system_ram_total_mb": (start_sample or end_sample).get("ram_total_mb"),
            "client_rss_mb": {
                "before": rss_before,
                "peak": max(rss_samples) if rss_samples else None,
                "steady": _half_average(rss_samples),
            },
            "attribution_caveat": (
                "client_rss_mb is this benchmark process only - an HTTP client - and is NOT "
                "model memory; the model runs inside the server's container. "
                "system_ram_used_mb from tegrastats is the real figure on this unified-memory "
                "board. weights_mb/kv_cache_mb are not broken out: neither backend reports "
                "them over its HTTP API."
            ),
        },
        "power": power,
        "thermal": {
            "start_c": start_sample.get("tj_temp_c", start_sample.get("gpu_temp_c")),
            "end_c": end_sample.get("tj_temp_c", end_sample.get("gpu_temp_c")),
            "max_c": max(temps) if temps else None,
            "sensor": "tj",
        },
        "validity": {
            "warmup_runs": warmup_count,
            "warmup_reached_steady_state": steady,
            "measurement_runs": len(measured),
            "thermal_throttling_observed": False,
            "failed_runs": len(measured) - len(ok),
            "flags": flags,
            "notes": f"token_source: {token_source}",
        },
    }
    if co_resident:
        result["co_resident_workload"] = list(co_resident)
    return result


def _achieved_or_none(ok: list[dict]) -> int | None:
    """With no input-token target, the fixed prompt still has a length, so
    report the achieved count as the target - the same number, measured
    rather than requested. Returns None (not 0) when the server sent no usage
    block: the schema permits null there precisely so the gap stays visible
    instead of being filled with a fabricated target."""
    for r in ok:
        if r.get("input_tokens"):
            return r["input_tokens"]
    return None
