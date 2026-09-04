# Jetson LLM Benchmark — Publication-Oriented Development Plan

## 0. Purpose

This document defines the long-term plan for evolving `jetson-llm` from an orchestrator-LLM selection lab into a reproducible, publication-quality benchmark suite for LLM inference on NVIDIA Jetson platforms, with NVIDIA Jetson Thor as the primary target. *(Reconciled 2026-09-04: Orin is the platform of record and Thor the cross-platform arm, until Thor access exists — see §0.1 and §26.)*

The benchmark must remain useful for the original purpose of the repository:

> selecting an LLM for `embedded-ai-chain`

while progressively becoming a general benchmark framework capable of comparing:

* model families,
* model sizes,
* quantization formats,
* inference backends,
* serving configurations,
* context lengths,
* batch/concurrency levels,
* latency,
* throughput,
* memory consumption,
* power consumption,
* energy efficiency,
* task quality,
* tool-calling reliability.

The design goal is not to produce a single "best LLM" number.

The goal is to characterize the trade-off:

> **quality × latency × throughput × memory × energy**

under clearly defined and reproducible edge-computing conditions.

---

# 0.1 Status — how this document relates to the rest of the repo

*Added 2026-09-04, after this plan was read against the code and the existing
results. The methodology below stands; this section records what is already
implemented, what turned out to conflict with reality, and where the merged plan
now lives.*

**This document is the methodology. `docs/TODO.md` is the schedule.** Phases 2+ of
`docs/TODO.md` were rewritten around this file on 2026-09-04; its "Scope change"
section carries the reconciliation in full and its decision log records each call.
`docs/project.diagram.md` is the picture, and marks which boxes are built.

**Already implemented — do not rebuild.** §1 above says not to replace the existing
architecture; more of this plan is already in it than the file assumes, because
`benchmarks/harness.py` was not read when this was written. Already true today:

| Asked for here | Already in `benchmarks/harness.py` / the scripts |
|---|---|
| §9 p50/p95/p99, never means | `percentiles()` — a mean is deliberately absent |
| §9 cold start separate from inference | `run_benchmark()`'s `setup_fn` timing |
| §9 "do not report client-process RSS as model memory" | already true, *and* already carried as an explicit `rss_mb_caveat` in the result |
| §9 system RAM vs process RSS on unified memory | `system_ram_mb` vs `rss_mb`, separately |
| §14 power sampled during the measurement window only | `sampler.samples_between(t_meas_start, t_meas_end)` |
| §38 thermal steady state | `is_thermally_stable()` on real `tj` flatness, with an honest `warmup_reached_steady_state=False` when it gives up |
| §3 power mode / clocks / L4T on every result | `nvpmodel_mode()`, `jetson_clocks_locked_heuristic()`, `l4t_version()` |
| §17 a general-capability suite | `scripts/validate_mmlu.py` |
| §36 tool calling on an external dataset | `scripts/validate_tool_calling.py` against real BFCL |
| §28 append-only registry with platform/precision/backend | `configs/models.yaml` and its own confirmed-vs-untested rule |

**The real gaps this plan correctly identifies** (now `docs/TODO.md` Phase 2, and the
highest-value work in the file):

1. §10's raw per-run data — `run_benchmark()` computes percentiles and then discards
   the underlying list, so nothing can be re-analysed later.
2. §14's J/token — currently **uncomputable**, because the script with power
   telemetry doesn't know token counts and the script with token counts has no
   sampler. See `docs/project.diagram.md` §2.
3. Token counts that are really token counts: the streaming path counts SSE content
   deltas, which is a chunk count.
4. §31's manifest — no git SHA, no backend version, no image digest on any result.
5. §37's decomposed cold start — one recorded 338s "cold start" is mostly a 6.7GB
   download; one recorded 2.0s is a warm-cache second run.

**Conflicts with reality, and how they were resolved.** Each is marked inline at the
section it affects, and recorded in `docs/TODO.md`'s decision log:

- Thor is *not* the primary platform (§26, §48) — there is no confirmed Thor access.
  Orin is the platform of record; Thor is the cross-platform arm.
- The §7 example's `"memory_gb": 128` is not this device: this Orin has ~30GB
  unified, verified.
- §10's fixed `warmup: 5` would be a regression against the existing
  temperature-driven steady-state policy. Treated as a floor, not the policy.
- §45/§46's Qwen3 campaign is gated on a real serving check first — vLLM ≥0.11 has
  no JetPack 6.2 / CUDA 12.6 wheels, and a model choice must not force a runtime
  upgrade.
- This document never mentions **Bielik** or **Apertus**, which are this lab's actual
  second and third model families and its two most interesting results. They stay in
  the matrix; see §1.1's file list below and `docs/project.diagram.md` §5.
- The `schemas/` files added alongside this document were example *instances*, not
  JSON Schema. They are now real draft-2020-12 schemas, validated in
  `tests/test_schemas.py`, with the originals preserved in `schemas/examples/`.
- This plan implies a third paper. There are already two sharing one writing window
  to 2027-01-25. Resolved: this repo is the *instrument* for
  `embedded-ai-chain/docs/paper.md`'s propositions P3/P4/P5/P6, not a third paper.
  Revisit only if the data justifies one on its own.

---

# 1. Current repository state

## 1.1 Existing architecture

The repository already contains:

```text
configs/
    models.yaml

src/
    llm_coordinator.py
    llm_client.py
    model_config.py

benchmarks/
    harness.py

scripts/
    benchmark.py
    benchmark_streaming.py
    validate_mmlu.py
    validate_tool_calling.py

docs/
    TODO.md
    promotion-contract.md
    note.md                 <- this file
    project.diagram.md      <- added 2026-09-04

schemas/                    <- added 2026-09-04, now real JSON Schema
    model.schema.json
    experiment.schema.json
    benchmark_result.schema.json
    quality_result.schema.json
    examples/

tests/                      <- 6 suites, no Docker/GPU needed
```

The candidate registry is also broader than this section implies: as well as
Qwen2.5 at 1.5B/3B/7B it carries **Bielik-11B-v3.0-Instruct** (Polish, SpeakLeash —
working on llama.cpp, tool-calling confirmed broken on vLLM) and
**Apertus-8B-Instruct-2509** (blocked outright: llama.cpp does not recognise its GGUF
architecture on any build tried). Both are real results and both stay in the matrix.

The current architecture supports:

* vLLM,
* llama.cpp,
* local execution,
* remote execution,
* model configuration through `models.yaml`,
* OpenAI-compatible HTTP serving,
* latency benchmarking,
* streaming TTFT / tokens-per-second measurement,
* BFCL-based tool-calling evaluation,
* MMLU-based quantization sanity checking.

Do not replace this architecture unless a demonstrated limitation requires it.

---

# 2. Long-term research objective

The final benchmark should answer questions such as:

### Model efficiency

* Which model provides the best quality at a given latency?
* Which model provides the best quality within a fixed memory budget?
* Which model provides the best quality per joule?

### Quantization

* How much quality is lost when moving from FP16/BF16 to INT8/INT4?
* How much latency and memory are saved?
* Does quantization behave differently for different model families?

### Runtime

* How much does the inference backend affect performance?
* Is the same model significantly faster under vLLM than llama.cpp?
* Does the best runtime depend on context length or model size?

### Edge deployment

* What happens at batch=1?
* How does performance change with context length?
* When does KV cache become the dominant memory cost?
* What model sizes are practically deployable on Thor?

### Practical application

* Which model is the best orchestrator for `embedded-ai-chain`?
* Does a larger model improve tool-call judgment enough to justify its cost?

---

# 3. Fundamental benchmark principle

Never report a performance number without its experimental context.

A valid result must identify at minimum:

```text
hardware
platform
power mode
software environment
model
model revision
precision / quantization
backend
backend version
serving configuration
context length
input tokens
output tokens
batch size / concurrency
sampling configuration
warmup policy
number of repetitions
co-resident workload
```

An unqualified statement such as:

> "Model X achieves 50 tok/s"

is not considered a valid benchmark result.

A valid statement is:

> "Model X, quantization Y, served with backend Z, on Jetson Thor under power configuration P, with batch=1 and 2048 input tokens, achieved median decode throughput of N tok/s over R repetitions."

---

# 4. Benchmark dimensions

The benchmark must eventually be organized around the following multidimensional matrix:

```text
MODEL
  ×
MODEL SIZE
  ×
PRECISION / QUANTIZATION
  ×
BACKEND
  ×
HARDWARE
  ×
POWER MODE
  ×
CONTEXT LENGTH
  ×
OUTPUT LENGTH
  ×
BATCH / CONCURRENCY
```

Do not attempt to evaluate the complete Cartesian product immediately.

The matrix should be expanded progressively.

---

# 5. Benchmark layers

The benchmark is divided into five independent layers.

```text
                 JETSON LLM BENCHMARK
                          |
        +-----------------+------------------+
        |                 |                  |
      QUALITY         PERFORMANCE        EFFICIENCY
        |                 |                  |
        |           +-----+------+       +---+---+
        |           |            |       |       |
     Accuracy      Latency    Throughput Memory  Power
        |           |            |       |       |
        +-----------+------------+-------+-------+
                          |
                    APPLICATION
                    TOOL CALLING
```

The layers must remain independently measurable.

Do not collapse them into a single score initially.

---

# 6. Phase 0 — Freeze the current baseline

## Objective

Create a reproducible baseline before changing the benchmark architecture.

## Tasks

[ ] Record the current Git commit.

[ ] Record current benchmark outputs.

[ ] Record current model registry.

[ ] Record current Orin environment.

[ ] Record current software versions.

[ ] Preserve all existing results.

[ ] Do not overwrite historical benchmark results.

## Required environment metadata

At minimum:

```text
JetPack / L4T
CUDA
driver
GPU architecture
CPU architecture
RAM
power mode
jetson_clocks state
temperature
kernel / OS
Docker version
container image
vLLM version
llama.cpp version
Python version
compiler version
```

## Deliverable

Create:

```text
docs/baseline.md
```

containing the exact environment used for the current measurements.

---

# 7. Phase 1 — Define the benchmark data model

## Objective

Make every experiment machine-readable and reproducible.

Do not start with plots.

First define the result schema.

Create:

```text
schemas/
    benchmark_result.schema.json
    model.schema.json
    experiment.schema.json
```

Every benchmark result should contain metadata similar to:

```json
{
  "experiment_id": "...",
  "timestamp": "...",

  "hardware": {
    "platform": "thor",
    "gpu": "...",
    "memory_gb": 128,
    "power_mode": "...",
    "jetson_clocks": true
  },

  "software": {
    "jetpack": "...",
    "cuda": "...",
    "driver": "...",
    "backend": "vllm",
    "backend_version": "..."
  },

  "model": {
    "name": "...",
    "revision": "...",
    "parameters": "...",
    "precision": "..."
  },

  "serving": {
    "context_length": 4096,
    "max_output_tokens": 256,
    "batch_size": 1,
    "concurrency": 1,
    "temperature": 0.0
  },

  "metrics": {
    "ttft_ms": {},
    "latency_ms": {},
    "prefill_tok_s": {},
    "decode_tok_s": {},
    "output_tok_s": {},
    "peak_memory_mb": {},
    "power_w": {},
    "energy_j": {}
  }
}
```

> **Reconciled 2026-09-04.** These schemas now exist for real, as JSON Schema
> draft 2020-12 (`schemas/`, validated in `tests/test_schemas.py`; the sketches
> above are preserved in `schemas/examples/`). Two changes to the example: (a)
> `"memory_gb": 128` is a Thor assumption — the Orin this lab runs on has **~30GB**
> unified, verified, so `memory_total_mb` is read from the device per run and never
> inferred from the platform name; (b) a fourth schema, `quality_result.schema.json`,
> was added, because §5 keeps quality and performance independently measurable and
> that is only enforceable if they are separate document types.

The exact schema can evolve, but the principle must not:

> Results contain enough information to reproduce the experiment.

---

# 8. Phase 2 — Separate experiment configuration from results

Currently `configs/models.yaml` acts primarily as a candidate registry.

Keep that functionality.

Do not overload it with every benchmark parameter.

Introduce:

```text
configs/
    models.yaml
    benchmarks/
        smoke.yaml
        latency.yaml
        streaming.yaml
        context.yaml
        memory.yaml
        power.yaml
        quality.yaml
```

Example:

```yaml
name: streaming_baseline

input_tokens:
  - 128
  - 512
  - 2048

output_tokens:
  - 128

batch_size:
  - 1

warmup_runs: 5
measurement_runs: 30

temperature: 0.0
```

This allows:

```text
model configuration
```

and

```text
experiment configuration
```

to remain separate.

---

# 9. Phase 3 — Establish a canonical performance benchmark

Before adding many datasets, make the hardware benchmark rigorous.

## 9.1 Metrics

Measure:

### Cold start

```text
process start → server ready
```

Keep cold start separate from inference latency.

### TTFT

```text
request start → first generated token
```

### Prefill throughput

```text
input tokens / prefill time
```

### Decode throughput

```text
generated tokens / decode time
```

### End-to-end latency

```text
request start → final token
```

### Inter-token latency

Measure distribution where possible.

Report:

```text
p50
p95
p99
```

Do not use the arithmetic mean as the primary latency metric.

### Peak memory

Measure the actual inference system memory footprint.

On Jetson unified-memory platforms, clearly distinguish:

```text
process RSS
system RAM
GPU memory accounting
KV cache
model weights
runtime overhead
```

Do not report client-process RSS as model memory.

---

# 10. Phase 4 — Make performance measurements statistically robust

Each benchmark should have:

```text
warmup
measurement
repetition count
outlier policy
```

Recommended initial baseline:

```text
warmup: 5
measurements: 30
```

> **Reconciled 2026-09-04.** Treat these as **floors, not the policy**.
> `benchmarks/harness.py` already warms up until junction temperature is flat across
> a rolling window and records `warmup_reached_steady_state=False` when it gives up
> instead — strictly stronger than a fixed count, and adopting `warmup: 5` verbatim
> would be a regression. `measurement_runs: 30` is kept as the floor for a
> publication cell.

For publication-quality experiments, increase repetitions when variance is high.

Store raw per-run measurements.

Do not store only:

```text
mean
median
p95
```

Store:

```text
run_001
run_002
...
run_N
```

Aggregates must be generated from raw data.

---

# 11. Phase 5 — Canonical prompt-length matrix

Introduce controlled prompt lengths.

Initial matrix:

```text
128
512
1024
2048
4096
8192
16384
```

Extend only when supported by the model/backend/hardware.

For each input length measure:

```text
TTFT
prefill tok/s
decode tok/s
E2E latency
peak memory
power
```

This allows analysis of:

> prefill scaling vs context length

and:

> KV-cache / memory scaling vs context length.

---

# 12. Phase 6 — Canonical output-length matrix

Use:

```text
32
128
256
512
1024
```

or the maximum practical range supported by the configuration.

This separates:

```text
prefill cost
```

from:

```text
decode cost
```

and prevents a benchmark from accidentally favoring a model because it generates fewer tokens.

---

# 13. Phase 7 — Batch and concurrency

For edge deployment, batch=1 is the primary scenario.

However, eventually evaluate:

```text
batch = 1
batch = 2
batch = 4
batch = 8
```

and, where the serving backend supports it:

```text
concurrency = 1
2
4
8
```

Report:

```text
request throughput
token throughput
latency
TTFT
peak memory
```

Do not mix batch and concurrency terminology.

They represent different experimental conditions.

---

# 14. Phase 8 — Power and energy benchmark

This phase is essential for Jetson research.

Collect:

```text
power
temperature
clock state
```

during inference.

Calculate:

```text
energy_per_request
energy_per_output_token
joules_per_token
```

The most important metric is:

```text
J / generated token
```

where appropriate.

Power measurements must be synchronized with the inference interval.

Do not simply report instantaneous board power measured at an arbitrary point in time.

---

# 15. Phase 9 — Co-resident workload

Because the original purpose of the repository is `embedded-ai-chain`, introduce two explicit deployment conditions.

## Condition A — Standalone

Only the LLM workload is active.

## Condition B — Application

The same workload that exists in the real pipeline is active.

For example:

```text
YOLO
STT
TTS
LLM
```

when applicable.

Every published number must explicitly identify:

```text
standalone
```

or:

```text
co-resident
```

This distinction is particularly important for unified-memory Jetson platforms.

---

# 16. Phase 10 — Quality benchmark architecture

Quality must be separated into:

```text
general capability
reasoning
coding
instruction following
tool calling
application-specific behavior
```

Do not create a single custom dataset and call it "LLM quality".

Use established public benchmarks wherever possible.

---

# 17. Initial quality benchmark set

Start with a small, stable set.

Recommended initial categories:

| Category               | Benchmark              |
| ---------------------- | ---------------------- |
| General knowledge      | MMLU                   |
| Mathematical reasoning | GSM8K                  |
| Reasoning              | ARC-Challenge          |
| Instruction following  | IFEval                 |
| Coding                 | HumanEval / HumanEval+ |
| Tool calling           | BFCL                   |
| Multilingual           | Belebele               |

Do not necessarily run every benchmark on every experiment.

Define benchmark tiers.

---

# 18. Benchmark tiers

## Tier 0 — Smoke

Used during development.

```text
10–50 samples
```

Purpose:

* detect broken serving,
* broken tokenizer,
* broken chat template,
* broken quantization,
* malformed output.

## Tier 1 — Regression

Used for every candidate configuration.

Small but representative.

Purpose:

> detect quality regressions caused by quantization/backend/configuration.

## Tier 2 — Full evaluation

Used only for final candidates.

Run the complete selected benchmark suites.

## Tier 3 — Holdout

A fixed dataset never used for tuning.

Used only for final evaluation.

---

# 19. Custom Jetson / application dataset

Eventually create:

```text
datasets/
    application/
```

This dataset should represent the actual use case of the orchestrator.

Examples:

```text
scene description
conversation transcript
tool availability
tool descriptions
user request
expected decision
```

Possible expected outcomes:

```text
call tool
do not call tool
call correct tool
extract correct arguments
answer directly
```

This dataset must not replace BFCL.

It complements BFCL.

BFCL provides external comparability.

The application dataset provides relevance to the actual embedded system.

---

# 20. Dataset versioning

Every dataset must have:

```text
dataset_name
dataset_version
source
license
download procedure
preprocessing version
prompt template version
```

Do not modify a dataset silently.

If the dataset changes:

```text
v1.0
v1.1
v2.0
```

must be created.

Results must reference the exact dataset version.

---

# 21. Prompt-template versioning

The chat template can significantly influence model behavior.

Therefore record:

```text
tokenizer revision
chat template
system prompt
user prompt template
reasoning / thinking configuration
tool schema
sampling parameters
```

A change in prompt formatting is an experimental change.

It must not be hidden inside the benchmark implementation.

---

# 22. Reasoning / thinking models

Models with explicit reasoning or thinking modes require special treatment.

Record separately:

```text
input_tokens
thinking_tokens
visible_output_tokens
total_generated_tokens
```

Measure:

```text
TTFT
time_to_answer
total_generation_time
tokens/sec
```

Do not compare a reasoning model and a non-reasoning model only using visible output tokens.

The total computational work must be visible in the results.

---

# 23. Quantization benchmark

Once FP16/BF16 baseline is stable, introduce:

```text
FP16 / BF16
FP8 where supported
INT8
INT4
```

Then investigate specific quantization methods where relevant:

```text
AWQ
GPTQ
SmoothQuant
other backend-native formats
```

Every quantized candidate must identify:

```text
source model
quantization method
quantization source
weight precision
activation precision
KV precision
calibration dataset if applicable
```

Do not mix quantization methods under one generic "INT4" label.

---

# 24. Quantization quality protocol

For every quantized model:

1. Run smoke test.
2. Run regression quality set.
3. Run performance benchmark.
4. Run memory benchmark.
5. Run power benchmark.
6. Compare against same-model higher-precision baseline.

The primary comparison is:

```text
same model
same backend
same hardware
different precision
```

Do not interpret:

```text
7B INT4 > 1.5B FP16
```

as a quantization result.

---

# 25. Backend benchmark

Eventually support:

```text
vLLM
llama.cpp
```

and potentially future backends.

A backend should be added only when:

* it actually runs on the target hardware,
* the configuration is reproducible,
* its version is recorded,
* its limitations are documented.

Do not add a backend only because it exists theoretically.

---

# 26. Thor as the primary experimental platform

Thor should become the primary platform for the publication-oriented benchmark.

> **Reconciled 2026-09-04 — this is inverted in the merged plan.** There is no
> confirmed Thor access from this lab: no address, no `-thor` row in
> `configs/models.yaml`, and `docs/TODO.md`'s Thor phase has never been started. A
> first publication experiment gated on unavailable hardware is not schedulable.
>
> **Orin is the platform of record; Thor is the cross-platform arm.** That is also
> the stronger framing — this section's own best question, "does the preferred
> allocation change with the hardware?", is `embedded-ai-chain/docs/paper.md`'s P5,
> and answering it needs *both* boards rather than Thor alone. Everything else in
> this section stands: encode `platform` explicitly, never infer it from a config
> key's name, and never carry one board's memory budget or power modes onto another.

Maintain Orin support because:

* it provides historical data,
* it enables comparison,
* it is useful for embedded deployment,
* it allows cross-generation analysis.

The registry should therefore explicitly encode:

```text
platform: orin
platform: thor
```

Never infer hardware from the model configuration name alone.

---

# 27. Power-mode matrix

Eventually evaluate multiple power modes.

Example:

```text
low-power
balanced
maximum-performance
```

Use the actual available Thor power configurations rather than assuming Orin power modes transfer directly.

Record:

```text
power mode identifier
CPU clocks
GPU clocks
DLA clocks if relevant
jetson_clocks state
```

Do not compare two results if their hardware performance state is unknown.

---

# 28. Model registry evolution

Keep:

```text
configs/models.yaml
```

as the authoritative candidate registry.

Each entry should eventually contain:

```yaml
model:
revision:
family:
parameters:
architecture:
precision:
quantization:
backend:
backend_version:
platform:
context_length:
serving_args:
sampling:
source:
license:
notes:
status:
```

Possible status values:

```text
untested
smoke-tested
benchmark-ready
fully-evaluated
blocked
deprecated
```

Never delete old configurations.

Historical candidates remain part of the experimental record.

---

# 29. Result storage

Introduce a structured hierarchy.

Recommended:

```text
results/
    raw/
        <experiment_id>/
    processed/
        ...
    reports/
        ...
```

Do not commit huge raw datasets or model outputs if they are not appropriate for the repository.

However, metadata and enough information to reproduce the experiment must be preserved.

---

# 30. Experiment IDs

Every experiment should receive a unique ID.

Example:

```text
2026-09-04_thor_qwen35_7b_int4_vllm_ctx2048_bs1_r01
```

or a shorter generated UUID plus human-readable metadata.

The experiment ID must allow a result to be traced back to:

```text
model config
benchmark config
Git commit
hardware
software environment
```

---

# 31. Reproducibility manifest

Every benchmark campaign should produce:

```text
manifest.json
```

containing:

```text
git commit
model revision
dataset versions
benchmark configuration
hardware information
software versions
environment variables
container image
command line
timestamp
```

A publication result should be traceable to one manifest.

---

# 32. Statistical reporting

For latency-related metrics report:

```text
p50
p95
p99
```

For throughput report:

```text
median
p5
p95
```

where useful.

For quality report:

```text
score
number of samples
benchmark version
```

For repeated stochastic evaluations report:

```text
mean
standard deviation
confidence interval
```

when statistically meaningful.

Do not report excessive decimal precision.

---

# 33. Performance plots

Eventually generate standardized plots.

Minimum plot set:

### Figure 1 — Decode throughput

```text
tokens/s vs model
```

### Figure 2 — TTFT

```text
TTFT vs context length
```

### Figure 3 — Memory

```text
peak memory vs context length
```

### Figure 4 — Energy

```text
J/token vs model
```

### Figure 5 — Quantization trade-off

```text
quality vs tokens/s
```

### Figure 6 — Pareto frontier

```text
quality
   ^
   |
   |       ●
   |    ●
   |  ●
   +----------------> energy / latency
```

### Figure 7 — Context scaling

```text
context length
        vs
TTFT / memory / throughput
```

Plots must be generated automatically from raw benchmark data.

Never manually edit benchmark plots.

---

# 34. Pareto analysis

Do not create a single weighted score initially.

Instead identify Pareto-optimal candidates.

For example:

```text
quality
latency
memory
energy
```

A candidate is dominated if another candidate is:

* at least as good in every relevant metric,
* and strictly better in at least one.

This is more scientifically meaningful than arbitrary weights.

---

# 35. Application score

The original repository still needs a practical selection mechanism.

Therefore maintain a separate:

```text
application scorecard
```

for `embedded-ai-chain`.

This can include:

```text
tool-calling accuracy
latency
memory
energy
quality
integration reliability
```

The application score must never replace the raw benchmark results.

---

# 36. Tool-calling evaluation

BFCL remains the external benchmark.

The application-specific dataset remains the practical benchmark.

Measure separately:

```text
correct tool
correct arguments
correct no-tool decision
invalid tool call
hallucinated tool
formatting failure
```

Eventually report a confusion matrix.

Example:

```text
                 Expected
               Tool     No Tool
Pred Tool        TP        FP
Pred No Tool     FN        TN
```

This is more informative than a single percentage.

---

# 37. Cold-start benchmark

Cold start must be measured separately from warm inference.

Define:

```text
container cold start
model loading time
server readiness
first request
```

where possible.

For remote benchmarks:

```text
cold_start = not measured
```

must be explicitly recorded.

Never report a remote server's inference latency as cold-start latency.

---

# 38. Thermal stability

Long-running benchmark campaigns must monitor:

```text
temperature
clock frequency
power
throttling
```

A benchmark should be considered invalid or flagged if thermal throttling occurs.

For long tests:

```text
warmup
sustained load
steady-state window
```

should be distinguished.

---

# 39. Benchmark protocol

Before a publication campaign:

1. Reboot the device if required by the protocol.
2. Set the documented power mode.
3. Set documented clocks.
4. Verify thermal state.
5. Stop unrelated workloads.
6. Start required co-resident workloads.
7. Start the LLM server.
8. Wait for readiness.
9. Run warmup.
10. Run measurements.
11. Collect telemetry.
12. Save raw data.
13. Save environment manifest.
14. Stop server.
15. Repeat for next configuration.

The protocol must be scripted as much as possible.

---

# 40. Avoid benchmark contamination

Never change:

```text
model
backend
power mode
context
sampling
prompt
```

during a benchmark campaign unless the change is part of the experiment.

Do not manually tune one model more aggressively than another and then call the comparison fair.

Backend-specific tuning is allowed, but:

> every tuning decision must be documented and reproducible.

---

# 41. Fairness rules

When comparing models:

### Same hardware

unless explicitly studying hardware scaling.

### Same workload

same input/output token targets.

### Same sampling

unless the model requires a documented exception.

### Same benchmark dataset

same version.

### Same evaluation protocol

same scoring rules.

### Same measurement methodology

same warmup and repetition strategy.

Exceptions must be explicitly documented.

---

# 42. Model selection matrix

Eventually create a machine-readable matrix:

```text
model
size
backend
precision
hardware
context
batch
quality
TTFT
prefill
decode
latency
memory
power
energy
```

This becomes the central dataset for analysis.

---

# 43. Publication dataset

Before publication, create an immutable benchmark release.

Example:

```text
benchmark-v1.0
```

It should contain:

```text
models evaluated
model revisions
datasets
benchmark configs
software versions
hardware configuration
raw measurements
processed measurements
plots
tables
analysis scripts
```

The benchmark release must be reproducible from the Git repository plus documented external model/dataset downloads.

---

# 44. Publication-oriented experiment design

A potential paper should not simply ask:

> "Which LLM is fastest on Thor?"

A stronger research question is:

> "What are the quality-efficiency trade-offs of modern LLMs on resource-constrained edge accelerators?"

Possible subquestions:

1. How does model scaling affect latency and energy?
2. How does quantization affect quality-efficiency trade-offs?
3. How does context length affect memory and latency?
4. How strongly does runtime choice influence performance?
5. Which models lie on the Pareto frontier for edge deployment?
6. How do benchmark-level capabilities translate into application-specific tool-calling behavior?

---

# 45. Experimental groups for a first paper

Do not start with dozens of models.

A manageable first study could contain:

### Model families

```text
Qwen2.5
Qwen3.x
one or two additional families
```

### Model sizes

Approximately:

```text
small
medium
large
```

where hardware permits.

### Precision

```text
FP16/BF16
INT8
INT4
```

where available.

### Runtime

```text
vLLM
llama.cpp
```

### Hardware

```text
Jetson Thor
Jetson AGX Orin
```

This already produces a substantial experimental matrix.

---

# 46. First publication baseline

The first complete benchmark campaign should prioritize:

```text
Qwen2.5
Qwen3.x
```

because this directly addresses the original motivation.

> **Reconciled 2026-09-04, then corrected the same day.** The first reconciliation
> here said Qwen3 might not be servable at all, because vLLM ≥0.11 has no prebuilt
> JetPack 6.2 / CUDA 12.6 wheels. **Measured:** the pinned container
> `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` reports `vllm 0.19.0` when asked,
> and serves correctly on this board — the wheel constraint is about *pip*, and
> NVIDIA's container is a different distribution path. Qwen3 is therefore very likely
> fine here. It stays gated on a real serving test only because every new row is.
>
> Also: the "one or two additional families" §45 leaves open are already chosen and
> already measured — **Bielik-11B** (working on llama.cpp, tool-calling broken on
> vLLM) and **Apertus-8B** (blocked). A benchmark that drops its only decisive
> negative result is a worse benchmark.

Do not wait for every possible LLM family before producing useful results.

Additional model families can be added later.

---

# 47. Qwen2.5 vs Qwen3.x comparison rules

When comparing generations:

Prefer matched configurations:

```text
similar parameter count
same hardware
same backend
same precision
same context
same output length
```

Then separately analyze:

```text
parameter scaling
```

and:

```text
generation improvement
```

Do not attribute a performance difference to architecture if parameter count, quantization or backend also changed.

---

# 48. Minimum viable publication experiment

The first serious experiment should be:

> **Reconciled 2026-09-04.** Same experiment, run on **Orin** — see §26. It is
> `docs/TODO.md` Phase 4, and its output-length half is *verbatim*
> `embedded-ai-chain/docs/paper.md`'s P3 core experiment, so one campaign serves both
> documents. Note the output ladder there starts at 16 rather than 128, which P3
> needs and this section's fixed `Output: 128` would miss.

```text
Hardware:
    Thor

Models:
    Qwen2.5-X
    Qwen3-X

Backend:
    one stable backend

Precision:
    FP16/BF16

Batch:
    1

Input:
    128 / 512 / 2048 / 4096 tokens

Output:
    128 tokens

Runs:
    >= 30

Metrics:
    TTFT
    prefill tok/s
    decode tok/s
    E2E latency
    peak memory
    power
    energy/token

Quality:
    MMLU
    GSM8K
    BFCL
```

This should be the first complete vertical slice.

---

# 49. Then add quantization

Second experiment:

```text
same models
same hardware
same backend

FP16/BF16
INT8
INT4
```

Measure:

```text
quality
latency
throughput
memory
energy
```

This creates the first strong quality-efficiency study.

---

# 50. Then add backend comparison

Third experiment:

```text
same model
same precision
same hardware

vLLM
llama.cpp
```

This isolates runtime effects.

---

# 51. Then add context scaling

Fourth experiment:

```text
context:
128
512
1k
2k
4k
8k
16k
```

Measure:

```text
TTFT
prefill
decode
memory
power
```

This is especially important for edge deployment.

---

# 52. Then add concurrency

Fifth experiment:

```text
1
2
4
8
```

Measure:

```text
throughput
latency
memory
energy
```

This distinguishes:

```text
single-user edge inference
```

from:

```text
server-style throughput
```

---

# 53. Then add application relevance

Sixth experiment:

```text
BFCL
+
application-specific orchestrator dataset
```

This connects generic LLM benchmarks to the actual use case.

---

# 54. What should NOT be done

Do not:

* create a single arbitrary LLM score;
* tune each model differently without documentation;
* report only tokens/s;
* report only average latency;
* ignore TTFT;
* ignore memory;
* ignore power;
* mix cold and warm inference;
* mix standalone and co-resident measurements;
* modify datasets without versioning;
* change prompt templates silently;
* delete historical configurations;
* overwrite old results;
* claim a model is "better" based on one metric;
* compare quantization methods without a same-model baseline;
* compare different model sizes as if they isolate architecture;
* rely only on a custom benchmark;
* publish results without software/hardware configuration.

---

# 55. Definition of done — benchmark framework

The framework is considered publication-ready when:

[ ] Model registry is versioned.

[ ] Benchmark configurations are versioned.

[ ] Dataset versions are recorded.

[ ] Every result contains environment metadata.

[ ] Raw measurements are preserved.

[ ] Warmup and repetitions are standardized.

[ ] p50/p95/p99 are automatically computed.

[ ] TTFT is measured.

[ ] Prefill throughput is measured.

[ ] Decode throughput is measured.

[ ] Peak memory is measured.

[ ] Power is measured.

[ ] Energy/token is calculated.

[ ] Thermal state is recorded.

[ ] Standalone/co-resident execution is distinguished.

[ ] Quantization experiments use same-model baselines.

[ ] Quality evaluation is separated from performance evaluation.

[ ] BFCL is retained for tool-calling.

[ ] At least one general benchmark suite is included.

[ ] Application-specific evaluation exists.

[ ] Results can be regenerated from raw data.

[ ] Plots are automatically generated.

[ ] Experiment manifests are stored.

[ ] Historical results are immutable.

[ ] A clean benchmark release can be tagged.

---

# 56. Immediate next steps

Do not implement the entire roadmap.

The next implementation sequence should be:

## Step 1

Create:

```text
docs/baseline.md
```

and freeze the current state.

## Step 2

Create the result schema:

```text
schemas/benchmark_result.schema.json
```

## Step 3

Modify the existing benchmark scripts so that every run records a complete environment manifest.

## Step 4

Add raw per-run measurements rather than only aggregate statistics.

## Step 5

Create:

```text
configs/benchmarks/
```

and move benchmark parameters into explicit configuration files.

## Step 6

Implement the canonical context-length experiment:

```text
128
512
1024
2048
4096
8192
```

with:

```text
batch=1
output=128
```

## Step 7

Add reliable Thor telemetry collection.

## Step 8

Only after the above is stable, begin the Qwen2.5 vs Qwen3.x campaign.

---

# 57. Research discipline

This repository should be treated as an experimental laboratory.

Every important change should answer:

```text
What hypothesis does this change test?
What variable changed?
What variables were held constant?
What measurements are expected to change?
```

Prefer small, traceable commits.

Examples:

```text
benchmark: add Thor environment manifest
benchmark: persist raw streaming measurements
benchmark: add context-length sweep
benchmark: add power telemetry
benchmark: add memory telemetry
benchmark: add Qwen3 model registry
benchmark: add INT4 regression protocol
analysis: add Pareto frontier plots
```

Avoid commits such as:

```text
update benchmark
fix stuff
improve results
```

---

# 58. Final architecture

The intended long-term architecture is:

```text
                        MODEL REGISTRY
                             |
                             v
                    EXPERIMENT CONFIG
                             |
                             v
                       BENCHMARK RUNNER
                             |
              +--------------+--------------+
              |              |              |
           QUALITY       PERFORMANCE     TELEMETRY
              |              |              |
          datasets       TTFT/tok/s      memory
          BFCL/MMLU      latency         power
          application    prefill         thermal
                         decode
              |              |              |
              +--------------+--------------+
                             |
                             v
                         RAW DATA
                             |
                             v
                      ANALYSIS PIPELINE
                             |
             +---------------+----------------+
             |               |                |
          TABLES           PLOTS          STATISTICS
             |               |                |
             +---------------+----------------+
                             |
                             v
                     PARETO ANALYSIS
                             |
                             v
                   APPLICATION SELECTION
                             |
                             v
                    PUBLICATION RELEASE
```

---

# 59. Guiding principle

The benchmark should answer three independent questions:

### Question 1 — Can the model do the task?

```text
QUALITY
```

### Question 2 — Can the hardware run it fast enough?

```text
PERFORMANCE
```

### Question 3 — Can the hardware run it efficiently enough?

```text
MEMORY + POWER + ENERGY
```

The final scientific contribution is the relationship between these three.

The benchmark should therefore ultimately enable statements such as:

> Model A provides higher task quality than Model B, but requires significantly more memory and energy.

> Quantization reduces memory and energy substantially while preserving most task quality.

> Runtime X provides lower TTFT for short contexts, while Runtime Y scales better for long contexts.

> The best model for edge deployment is not necessarily the model with the highest benchmark score, but the model lying on the relevant quality-efficiency Pareto frontier.

This is the core objective of the benchmark.
