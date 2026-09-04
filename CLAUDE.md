# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

Orchestrator-LLM selection lab for NVIDIA Jetson boards (Orin, eventually Thor) -
tests candidate models across two serving backends (vLLM, llama.cpp; see
`docs/TODO.md` Phase 0 for why TensorRT is explicitly excluded), not one fixed
model/framework/family. Currently Qwen2.5-Instruct (1.5B/3B/7B) plus
Apertus-8B-Instruct-2509 as a second model family - see README's "Model families"
section for why Apertus specifically (data-sovereignty framing: Swiss-public-funded
and fully open including training data, vs Qwen's Alibaba origin) and what's still
unconfirmed about it (gated download, unconfirmed tool-calling support).

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
make benchmark CONFIG=<key>                        # Latency/cold-start/thermal/power
make benchmark-streaming CONFIG=<key>               # TTFT / tokens-per-sec
make validate-tool-calling CONFIG=<key>             # BFCL tool-call judgment accuracy (needs staged data, see README)
make validate-mmlu CONFIG=<key>                     # MMLU quantization-sanity accuracy (needs staged data, see README)
make test                                           # Unit tests, no Docker/GPU needed
```

## Architecture

**Split deliberately narrow**, same principle as `jetson-vlm-lab`: this repo holds
only the pieces with zero dependency on `embedded-ai-chain`'s own modules.

- `src/llm_coordinator.py` - `VllmCoordinator` (Docker lifecycle for the vLLM
  container), `LlamaCppCoordinator` (Docker lifecycle for llama.cpp's `llama-server`,
  new to this lab), `RemoteCoordinator` (talks to an already-running server elsewhere,
  e.g. Thor - backend-agnostic, since both real backends speak the same
  OpenAI-compatible wire format). All three share one duck-typed interface
  (`base_url`, `model`, `start()`, `wait_ready()`, `stop()`).
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

Real vLLM + llama.cpp smoke tests done for Qwen2.5-1.5B, 2026-09-04 - see README's
"Current State" for the summary and `docs/TODO.md` Phase 1 for the full record,
including two real bugs those tests caught and fixed, and an open, unresolved
tool-calling-reliability question on the llama.cpp side. Apertus is registered in
`configs/models.yaml` but not yet run (gated download, see README).
