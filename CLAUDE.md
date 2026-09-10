# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

Orchestrator-LLM selection lab for NVIDIA Jetson boards - **Orin and Thor are both
live platforms as of 2026-09-08** - testing candidate models across **three serving
backends: vLLM, llama.cpp, and TensorRT Edge-LLM**, not one fixed
model/framework/family. Edge-LLM is **Thor-only** (it needs JetPack 7.x; this Orin is
6.2.x): `docs/TODO.md` Phase 0 excluded TensorRT for v1 and that exclusion was
explicitly **superseded for Thor** once Edge-LLM 0.10.1 turned out to ship an
OpenAI-compatible server with tool-calling. Read both halves of that Phase 0 entry
before assuming TensorRT is out of scope.

**The controlled framework comparison is done** (`docs/thor-framework-comparison.md`),
including an exclusive-box timing campaign, a decoding control, and a working
quantization. Result: at 7B the three backends are within ~2% on turn latency and
~16% on marginal energy per token, and differ by **40 points on BFCL `irrelevance`**
(Edge-LLM 94%, llama.cpp 62%, vLLM 54%) - that one axis decides the recommendation:
**Edge-LLM + Qwen2.5-7B, GPTQ-Int4 quantized if the turn budget is tight (769ms/turn,
0.274 J/token), FP16 otherwise (94% vs INT4's 90% `irrelevance`)**. Quantization
needed a real one-line bug found and patched in Edge-LLM's own source
(`patches/edgellm-int4-bias-recipe.patch`) - every int4 checkpoint was blocked on it
before that. Things this repo now knows and should not re-litigate: the framework gap
is **not** a decoding artifact (`top_k=1` greedy reproduced every score exactly - and
note that pinning only `temperature` is NOT controlled sampling, since the three
backends apply three different `top_p`/`top_k`/`min_p` defaults); it is **not** a
parser artifact (Edge-LLM scores identically with `auto` and `hermes`, 100/100
identical per-case verdicts); and **14B is pointless** (98/100 identical verdicts to
7B at 2x the latency and 2.1x the energy); it is **not** chat-template rendering
(2026-09-10 - Edge-LLM and vLLM build **byte-identical** prompts for all 50 BFCL
irrelevance cases, and vLLM fed Edge-LLM's exact bytes through `/v1/completions`
still abstains at 52% against Edge-LLM's 94%); and it is **not** numeric dtype
(vLLM forced to `--dtype float16`, matching Edge-LLM's engine, scored `irrelevance`
54.0% - identical to its own bf16 54.0%). What remains unidentified is *which*
structural difference causes the framework gap, and the list of things it is NOT is
now longer than the list of candidates. See `scripts/chat_template_crossfeed.py` and
`docs/thor-framework-comparison.md`'s "Why the gap exists".

Currently Qwen2.5-Instruct (1.5B/3B/7B/14B), Apertus-8B-Instruct-2509
(data-sovereignty framing - Swiss-public-funded and fully open including training
data, vs Qwen's Alibaba origin - but **BLOCKED**: llama.cpp doesn't recognize its
GGUF architecture at all, confirmed on two build versions, no working serving path
exists), and Bielik-11B-v3.0-Instruct (Polish, SpeakLeash - **llama.cpp: working**,
the strongest llama.cpp result of any family so far; **vLLM: serves correctly but
tool-calling doesn't work** on either of two parsers tried - the opposite pattern
from Qwen2.5). See README's "Model families" table for the full comparison.

(Named `jetson-llm-qwen` until Apertus was added - renamed once a second model family
broadened its scope, mirroring `jetson-vlm-lab`'s own history of starting as
`jetson-qwen-2.5-VL` and being renamed when Qwen3-VL did the same thing there.)

Sibling project to `jetson-yolov8-trt`/`jetson-whisper-trt`/`jetson-piper`/
`jetson-chatterbox`/`jetson-vlm-lab`, meant to be pulled into `embedded-ai-chain` as a
`third_party/` git submodule once a winner is chosen. Evaluated there as a drop-in
replacement for `LlmOrchestrator`'s hardcoded `base_url`/`model` (currently
Qwen2.5-1.5B-Instruct-AWQ on vLLM) - see `docs/promotion-contract.md` for exactly how.

Text-only (transcript in, response + optional tool call out) - the orchestrator never
handles an image itself (that's the separate on-demand VLM tier `jetson-vlm-lab`
evaluates), so unlike that repo, nothing here does JPEG/frame encoding.

## Common Commands

```bash
make serve-1.5b-vllm / make serve-1.5b-llamacpp   # Start a candidate server in the foreground
make benchmark CONFIG=<key>                        # Latency/cold-start/thermal/power (NOT yet retrofitted - see below)
make experiment EXP=configs/benchmarks/<name>.yaml   # Run a full experiment (the reportable path)
make benchmark-streaming CONFIG=<key> CONDITION=standalone   # One ad-hoc cell
make validate-tool-calling CONFIG=<key>             # BFCL tool-call judgment accuracy (needs staged data, see README)
make validate-mmlu CONFIG=<key>                     # MMLU quantization-sanity accuracy (needs staged data, see README)
make test                                           # Unit tests, no Docker/GPU needed
```

## Architecture

**Split deliberately narrow**, same principle as `jetson-vlm-lab`: this repo holds
only the pieces with zero dependency on `embedded-ai-chain`'s own modules.

- `src/llm_coordinator.py` - `VllmCoordinator` (Docker lifecycle for the vLLM
  container), `LlamaCppCoordinator` (Docker lifecycle for llama.cpp's `llama-server`;
  its `server_argv0` field exists because the Orin and Thor image families disagree
  about whether they set an `ENTRYPOINT`), `EdgeLlmCoordinator` (a local **process**,
  not a container - Edge-LLM ships neither image nor wheel, so it is built from source
  on-device and launched from the venv in that tree), `RemoteCoordinator` (talks to an
  already-running server elsewhere - backend-agnostic, since all three real backends
  speak the same OpenAI-compatible wire format). All four share one duck-typed
  interface (`base_url`, `model`, `start()`, `wait_ready()`, `stop()`).
  Two `EdgeLlmCoordinator` details are load-bearing and were each found the hard way:
  it must run with `cwd` set to the Edge-LLM source tree (the TensorRT plugin is
  registered by a *relative* path, and without it the server dies deserializing its
  own cached engine), and `stop()` must signal the process *group* (uvicorn's workers
  otherwise keep the port and the GPU allocation held).
- `src/llm_client.py` - `call_llm(coordinator, messages, tools=None, tool_choice=...)`,
  the blocking chat-completions round trip. Returns the full assistant message (not
  just text), because `scripts/validate_tool_calling.py` needs `message["tool_calls"]`.
- `src/model_config.py` - loads a named variant (`1.5b-awq-vllm-orin`, ...) from
  `configs/models.yaml`, the single place new families/sizes/quantizations/backends
  get registered.

**What stays in `embedded-ai-chain` and does NOT move here**: `orchestrator.py`'s
`LlmOrchestrator`/`_ToolCallingMixin` (the real dialogue-loop integration: scene-state
reads, `ask_vlm`'s non-blocking submission, the speech-gate/echo-dropping run loop).
Those are integration code for a specific project's architecture, not reusable across
contexts the way the coordinator/client primitives are.

**No custom Dockerfile for either backend**: both `docker-compose.yml` (vLLM) and
`docker-compose.llamacpp.yml` (llama.cpp) run a prebuilt vendor/community image as-is
- `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` (confirmed working, reused from
`jetson-vlm-lab`) and `dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04` (from
`dusty-nv/jetson-containers` - confirmed live 2026-09-04; an older bare `r36.4.0` tag
also exists but its llama.cpp build hard-errors on `tool_choice`, see
`src/llm_coordinator.py`'s `_DEFAULT_LLAMACPP_IMAGE` comment for the full story).

**`benchmarks/harness.py` is a hand-maintained copy**, not an import, of
`jetson-vlm-lab/benchmarks/harness.py` (itself a copy of
`embedded-ai-chain/benchmarks/harness.py`) - unchanged here since nothing in it was
ever image/VLM-specific. This repo is meant to `git clone` and run standalone;
reaching back into either sibling repo's modules would break that. Port fixes by hand
if either sibling's harness gains one worth having here.

## Measurement integrity (docs/TODO.md Phase 2)

`scripts/benchmark_streaming.py` is the retrofitted path and the one to use. It emits
a document conforming to `schemas/benchmark_result.schema.json` into
`results/raw/<experiment_id>/`, validated at write time, containing every measured
repetition (not just percentiles), a full manifest, decomposed cold start, real token
counts from the server's own `usage`, and **energy in J/output-token** - which was
uncomputable from either script before, because the one with power telemetry didn't
know token counts and the one with token counts started no sampler.

Three rules the code enforces rather than trusts:

- **`--execution-condition` is required and has no default.** On unified memory a
  standalone and a co-resident number are different physical quantities. A
  `co-resident` run must also name what ran alongside; the CLI and the schema both
  reject it otherwise.
- **`write_result()` refuses to overwrite.** Re-running a cell means a new replicate,
  never a silent replacement of data a figure was drawn from (`docs/note.md` §29).
- **Prompts vary per repetition by default.** Both backends cache by prompt prefix,
  so an identical prompt makes every repetition after the first a cache hit rather
  than a prefill. Measured on this device: TTFT p50 40.8ms with identical prompts vs
  **268.1ms** with unique ones - a 6.6x error in the flattering direction, invisible
  in the aggregates and only caught by reading llama-server's own log
  (`prompt eval time = ... / 1 tokens` for a 40-token prompt). `--prompt-uniqueness`
  controls it and the choice is recorded in every result.
- **Prompts also vary per *cell*, not just per repetition** (`cell_salt()`, added
  2026-09-10 after this exact gap cost a 2.3x error). Per-repetition uniqueness was
  correct and **insufficient**: `build_prompt` was a function of input length and run
  index only, so every cell of a sweep holding input length fixed sent byte-identical
  prompts, and vLLM V1's automatic prefix caching (**on by default**) served cells 1..n
  out of cell 0's blocks. Worse, `build_prompt` truncates one fixed filler, so a shorter
  cell's prompt is a literal *prefix* of a longer one's - contaminating the context
  sweep progressively with length, which flattened a published scaling curve. It was
  invisible because each result document is written and validated **alone**, and the
  reuse axis is between documents. `PROMPT_TEMPLATE_VERSION` is now `v2`: any `..._v1`
  result from a multi-cell vLLM session may carry cross-cell cache hits in its TTFT.
  Full record: `docs/HISTORY.md`, "Phase 4 - prompt-salt contamination".

`scripts/run_experiment.py` is the path for anything that will be reported: it takes
a `configs/benchmarks/*.yaml` file, so an experiment is re-runnable from (model config
key, experiment config, git SHA) with no flag typed at a shell prompt carrying
experimental meaning. `scripts/benchmark_streaming.py` remains for ad-hoc probing.
Both go through `benchmarks/runner.py:measure_cell()` so they cannot disagree about
how a measurement is taken. One server session serves every cell of a model (a vLLM
cold start is ~136s), and cells after the first are flagged as sharing its cold-start
measurement.

A fourth enforced rule, `assert_condition_matches_reality()`, grew from three
incidents in one day - each caught a different way a labelled result can lie:

1. **`standalone` refused when a Docker container is running.** The natural mistake:
   this device runs a production `vllm-orchestrator` around the clock, so typing
   `standalone` looks fine until you check.
2. **`standalone` refused when a bare-metal process holds a GPU device handle.**
   Container check alone wasn't enough: `embedded-ai-chain`'s entire production
   pipeline (YOLO + orchestrator + STT/TTS) runs as one bare-metal process
   (`tts_consumer.py`) invisible to `docker ps`, and a real campaign got labelled
   `standalone` while it was running the whole time - caught only because the user
   asked whether prior runs were contaminated. Detected via `/proc/<pid>/fd` scans
   for `nvhost`/`nvgpu`/`nvmap` handles. The 6 results already written that way were
   deleted (the co-resident workload was never declared or controlled, so there was
   nothing honest to write after the fact).
3. **`co-resident` refused when the *declared* workload is incomplete.** Subtler,
   found on already-committed data: 5 results correctly said
   `co-resident: [vllm-orchestrator]`, but `tts_consumer.py` was running throughout
   every one of them too and was never named. Unlike incident 2, this data wasn't
   uncontrolled - exactly what was running is known - so those 5 were **corrected in
   place** (field fixed, a `_comment` explaining what changed, pre-correction content
   recoverable from git history) rather than deleted.

No override flag on any of the three: the fix is either to stop the other workload or
to declare the truth, and both are one command. **Anything caught this way after the
fact that can't be honestly corrected in place goes to `results/invalid/`, never
`rm -rf`** - archived with a note on what was actually true, per `results/invalid/README.md`.
Full incident record in docs/HISTORY.md's Phase 3 section - stopping the `vllm-orchestrator`
*container* is not sufficient for a real `standalone` claim on this device;
`tts_consumer.py` also has to stop, and that is a materially bigger interruption than
the container-only story assumed.

**`scripts/benchmark.py` is retired** (docs/TODO.md Phase 2's ⟨DECIDE⟩, resolved
2026-09-04) rather than retrofitted: `stream_llm()`-based measurement already covers
a strict superset of what it measured (TTFT + decode + e2e, plus energy and real
token counts, none of which it had). It is now a shim that exits with a pointer to
the replacements, so an old invocation fails loudly rather than silently doing
nothing. `benchmarks/harness.py` itself is untouched - only this script's use of it
retired; the harness stays the shared hand-maintained copy across sibling repos.
`benchmarks/manifest.py` and `benchmarks/runner.py` are new and repo-specific, which
is why the retrofit machinery lives there instead.

## Schemas

`schemas/` holds JSON Schema (draft 2020-12) for the four document types this lab
produces: a registry row (`model`), a benchmark configuration (`experiment`), a
performance result (`benchmark_result`), and an accuracy result (`quality_result`).
Quality and performance are deliberately separate document types, not one merged row
- `docs/note.md` §5 keeps the measurement layers independently measurable, and they
join at analysis time on `model_config_key`.

They are enforced, not decorative: `tests/test_schemas.py` validates **every
`configs/models.yaml` row** against `model.schema.json` on `make test`, so a
malformed registry row fails in seconds rather than mid-campaign. That row schema is
written to what the registry satisfies *today*; the other three describe what
`docs/TODO.md` Phases 2/3/5 must produce, and today's script output does **not**
conform to them yet - that gap is the work list, and `schemas/README.md` documents
the conformance levels.

Two rules worth knowing before editing them: every `$ref` is local (`#/$defs/...`),
never cross-file, so validation needs no resolver and this repo stays clone-and-run
- same reasoning as `benchmarks/harness.py` being a copy rather than an import; and
`additionalProperties: false` is set everywhere, so adding a result field means
editing the schema, which is intended (a new field in a result *is* a change to the
experimental record). The original 2026-09-04 sketches - instance documents named
`*.schema.json`, with nothing a validator could check - are preserved in
`schemas/examples/`, alongside `*.target.json` files showing the conformant shapes.

## Memory constraints (critical)

This device has ~30GB unified memory (verified via `free -h`, see
`embedded-ai-chain/docs/environment.md`). vLLM's `--gpu-memory-utilization` **must** be
passed explicitly and conservatively - its own default (0.9) fails outright at startup
on this hardware. The three `vllm`/`orin` rows in `configs/models.yaml` reuse
`embedded-ai-chain/src/orchestrator_models.py`'s own already-tuned thresholds
(0.15/0.25/0.5 for 1.5B/3B/7B) rather than re-deriving them from scratch - that file's
own docstring explains the reasoning (a fixed 0.08 default caused a real
"No available memory for the cache blocks" flake on 2026-09-03).

`llama-cpp`/`orin` rows use `n_gpu_layers=-1` (offload everything) as an *untested*
starting point - unlike the vLLM rows, none of these have a prior real measurement to
inherit from (this backend is new to the whole fleet, not just this repo). Re-tune per
`docs/TODO.md` Phase 1's pending smoke test if a size doesn't fit.

## Promotion contract

See `docs/promotion-contract.md` for how a candidate evaluated here (model x precision
x backend x platform) is allowed to become the orchestrator LLM `embedded-ai-chain`
actually runs: the integration contract it must meet (OpenAI-compatible HTTP with
structured tool calls, already satisfied by `VllmCoordinator`/`LlamaCppCoordinator`/
`RemoteCoordinator`), the scorecard every candidate is measured against (including the
tool-call judgment accuracy axis unique to this repo), and where the decision gets
recorded.

See `docs/TODO.md` for the phased action list, including the Phase 0 research already
done on why TensorRT is out of scope for v1 (JetPack version mismatches on both the
legacy TensorRT-LLM Jetson branch and the newer TensorRT Edge-LLM, plus Edge-LLM
having no OpenAI-compatible server at all) and what's still pending (Thor access,
`llama_cpp` image tag confirmation, the full benchmark matrix).

## Current State

**Phase 2-4 (measurement integrity + a real campaign)**: done 2026-09-04. See
README's "Current State" for the two real findings (J/output-token amortization with
generation length; a backend-specific context-scaling gap - **corrected 2026-09-10**:
vLLM's TTFT is *linear* in input length 128->1536, not "near flat" as originally
published, and llama.cpp's is clearly super-linear at ~5.5x vLLM's marginal cost per
token) and
`docs/project.diagram.md` §6 for the full tables. `docs/TODO.md` Phases 2-4 have the
complete record, including the three measurement-integrity incidents this section's
"Measurement integrity" above summarizes.

**Phase 1 (candidate serving/tool-calling)**: real smoke tests done 2026-09-04 for
all three families - see README's "Current State" for the summary and `docs/TODO.md`
Phase 1 for the full record. Qwen2.5-1.5B:
vLLM confirmed working; llama.cpp's initial tool-calling failure was resolved (a
real, llama.cpp-specific temperature sensitivity - `scripts/validate_tool_calling.py`
now pins `--temperature 0.1` by default) and a second real bug (BFCL's non-standard
JSON-schema type names crashing llama.cpp's grammar builder) was fixed along the
way - first real BFCL scorecard: 75% overall on a 40-case sample. Apertus-8B: blocked
outright (llama.cpp has no support for its GGUF architecture; no AWQ exists for
vLLM). Bielik-11B: confirmed working on llama.cpp (correctly structured tool call on
the first try) but tool-calling confirmed NOT working on vLLM (two parsers tried,
`hermes` and `llama3_json`, identical failure on both) - plain completion and Polish
work fine on vLLM, just not tool-calling, the opposite pattern from Qwen2.5.
