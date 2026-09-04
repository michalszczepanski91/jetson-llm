# Orchestrator-LLM promotion contract

Defines how a candidate (model x precision x backend x platform) evaluated in this
repo becomes the orchestrator LLM `embedded-ai-chain` actually runs, and what a "no
clean winner" outcome means. Adapted from `jetson-vlm-lab/docs/promotion-contract.md`
(same shape, same reasoning) for a text-only orchestrator LLM instead of a VLM.

## 1. Integration contract (already implemented - do not reinvent)

A candidate is integration-ready the moment it's served behind an OpenAI-compatible
HTTP endpoint:

- `GET /health` -> 200 once ready
- `GET /v1/models`
- `POST /v1/chat/completions`, accepting `tools`/`tool_choice` and returning
  structured `message.tool_calls`

`embedded-ai-chain/src/orchestrator.py`'s `LlmOrchestrator` already talks to exactly
this shape (`base_url`, `model`, `_complete()`'s `/v1/chat/completions` POST) - moving
it from today's hardcoded vLLM/1.5B-AWQ to any other model/backend/platform this repo
proves out should require changing only its constructor's `base_url`/`model` (and
whatever compose/coordinator file starts that server), not `handle_transcript()`,
`_execute_tool()`, or the tool schema itself. That's the existence proof this contract
holds, mirroring how moving jetson-vlm-lab's VLM from local Orin to remote Thor
required zero changes to `VlmProducer`/`VlmSession`/`ask_vlm`.

**Consequence for this repo**: any vLLM-served or llama.cpp-served candidate already
fits, regardless of quantization, as long as its server enables OpenAI-style
structured tool calling (vLLM: `--enable-auto-tool-choice --tool-call-parser hermes`;
llama.cpp: `--jinja`). A TensorRT-served candidate would fit too, *if* it exposed the
same REST shape - see docs/TODO.md's Phase 0 finding for why that isn't attempted in
this lab's v1 (TensorRT-LLM's Jetson support is stale/JetPack-6.1-only; TensorRT
Edge-LLM requires JetPack 7.x this device doesn't have and has no OpenAI-compatible
server at all - a real architectural mismatch, not a missing coordinator class).

## 2. Candidate registry (`configs/models.yaml`)

Each row declares:

- `model` - HF repo id
- `precision` - human-readable (e.g. "AWQ int4 (weights), bf16 activations")
- `backend` - `vllm` | `llama-cpp` (extend only when a backend is actually tried)
- `platform` - `orin` | `thor` (extend only when a platform is actually tried)
- backend-specific serving args: `gpu_memory_utilization`/`max_model_len` (vllm) or
  `quant`/`n_gpu_layers`/`ctx_size` (llama-cpp)
- `extra_args` - optional list of extra flags passed straight to the server command,
  for anything the coordinator's constructor has no dedicated parameter for
- `notes` - must say plainly whether a row is confirmed-working or an untested
  starting point (see `orchestrator_models.py`'s own convention, which this mirrors)

Existing rows stay even after a winner is declared and shipped - historical
measurements, not stale config (append-only, same convention as
`embedded-ai-chain/docs/TODO.md`'s decision log and `jetson-vlm-lab`'s own registry).

## 3. Scorecard

Every candidate is measured on the same axes before it's eligible to be proposed as a
winner:

| Axis | Source |
|---|---|
| Cold start | `benchmarks/harness.py` |
| Latency p50/p95/p99 (never mean) | `benchmarks/harness.py` |
| TTFT p50, steady tokens/sec | `make benchmark-streaming` |
| **Tool-call judgment accuracy** (overall + broken out by ask_vlm-escalation / read_scene_state / no-tool) | `make validate-tool-calling` |
| KV-cache / context headroom at the tuned serving args | server startup log / harness |
| Qualitative spot-check | same fixed transcript/scene-context set across every candidate |

Tool-call judgment accuracy is the axis that didn't exist in `jetson-vlm-lab`'s
scorecard (GQA/TextVQA measure visual QA, not applicable here) and is arguably the
more important number for this lab specifically: it directly answers whether a bigger
model or a different backend's tool-call parser closes the documented `ask_vlm`
escalation gap well enough that `orchestrator.py`'s forced-`tool_choice` workaround
(`_FORCE_ASK_VLM`) could be removed - see `scripts/validate_tool_calling.py`'s
docstring.

Every number must record which `platform` it was measured on and whether it was
standalone or under the same co-resident load the real pipeline puts on that machine
(YOLO + STT + TTS all resident) - an unqualified number is not a valid result, same
rule `embedded-ai-chain/docs/TODO.md` applies to its own latency numbers.

## 4. Declaring a winner

A candidate is proposed as *the* orchestrator LLM only when:

1. It's been measured on every axis above, on the platform it would actually run on
   in production.
2. The comparison is written up in this repo (README's "Current State" section or a
   dedicated benchmarks doc).
3. A pointer/decision-log row is added to `embedded-ai-chain/docs/TODO.md`, same shape
   as that file's existing decision-log rows - this repo does not become the source of
   truth for what ships; the parent repo's decision log still is.

**"No clean winner" is a legitimate outcome, not a failed process** (same framing as
jetson-vlm-lab's own Qwen-vs-Phi finding). If, say, 7B has meaningfully better
tool-call judgment but doesn't fit alongside the rest of the resident pipeline, that's
a real, useful, written-down finding - not a reason to force a pick.

## 5. What does not belong in this repo

- `orchestrator.py`'s `LlmOrchestrator`/`_ToolCallingMixin` (the real dialogue-loop
  integration, scene-state reads, `ask_vlm` non-blocking submission) stays in
  `embedded-ai-chain` - integration code tied to its specific architecture, not
  something a candidate swap should touch.
- Anything that reaches back into `embedded-ai-chain`'s own modules - this repo clones
  and runs standalone (see README's "Why this is a separate repo").
