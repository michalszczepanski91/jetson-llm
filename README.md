# jetson-llm-qwen

Orchestrator-LLM selection lab for NVIDIA Jetson boards (Orin, and eventually Thor -
see `configs/models.yaml`'s `platform` field). Compares Qwen2.5-Instruct at 1.5B/3B/7B
across two serving backends - vLLM and llama.cpp - so `embedded-ai-chain`'s
Phase 3 orchestrator LLM (currently a hardcoded Qwen2.5-1.5B-Instruct-AWQ on vLLM,
chosen because it was "the only option actually verified working," not because it was
benchmarked) can be picked with real numbers. See `docs/promotion-contract.md` for
exactly how a candidate here becomes that.

See `CLAUDE.md` for the full architecture rationale, in particular why this repo holds
only the container-lifecycle/HTTP-client primitives, not the scene-state/dialogue-loop
integration code, and `docs/TODO.md` for the phased plan (including why TensorRT is
explicitly excluded from v1, not silently skipped).

## Quick start

```bash
uv venv && uv pip install -r requirements.txt   # or: pip install -r requirements.txt

make serve-1.5b-vllm                             # start vLLM serving Qwen2.5-1.5B-Instruct-AWQ
curl http://localhost:8000/health

make serve-1.5b-llamacpp                         # start llama-server serving the Q4_K_M GGUF
curl http://localhost:8080/health

make benchmark CONFIG=1.5b-awq-vllm-orin           # latency/cold-start/thermal/power
make benchmark CONFIG=1.5b-q4-llamacpp-orin
make benchmark-streaming CONFIG=1.5b-awq-vllm-orin # TTFT / tokens-per-sec
make validate-tool-calling CONFIG=1.5b-awq-vllm-orin  # tool-call judgment accuracy
make test                                          # unit tests, no Docker/GPU needed
```

To measure against an already-running remote server (e.g. Thor) instead of starting a
local container, pass `--target remote --remote-host <ip>` to any `scripts/*.py`
directly (no Makefile target - the host is environment-specific):

```bash
uv run python scripts/benchmark.py --model-config 1.5b-awq-vllm-orin --target remote --remote-host <thor-ip>
uv run python scripts/validate_tool_calling.py --model-config 1.5b-awq-vllm-orin --target remote --remote-host <thor-ip>
```

Every `scripts/*.py` accepts `--model-config <name>` (see `configs/models.yaml` for
the full registry) rather than hardcoding a single model, so adding a new
size/quantization/backend to evaluate means adding a row to that file, not editing
every script.

## Why a tool-calling accuracy eval, not GQA/TextVQA

`jetson-vlm-lab` (this repo's sibling, same pattern applied to the VLM slot) measures
GQA/TextVQA because the VLM's job is visual QA. The orchestrator LLM's actual job -
and its actual documented failure mode
(`embedded-ai-chain/src/orchestrator_models.py`'s "weakest tool-calling judgment...the
documented ask_vlm escalation gap") - is tool-call judgment: given a transcript and
scene-state grounding, does it call `read_scene_state`, call `ask_vlm`, or answer
directly, correctly and *unforced* (`tool_choice="auto"`, not the forced-tool_choice
workaround `orchestrator.py` currently needs)? `scripts/validate_tool_calling.py`
measures exactly that, on a fixed 15-case eval set hand-copied from
`orchestrator.py`'s own `VLM_QUERY_PATTERNS`/`VISION_QUERY_PATTERNS`.

## Why this is a separate repo

Mirrors the `third_party/jetson-vlm-lab` (and `jetson-yolov8-trt`) pattern in
`embedded-ai-chain`: a standalone, independently clonable/testable repo for one
hardware-adjacent component, pulled into the parent project as a `third_party/` git
submodule once a winner is chosen. Deliberately standalone, not importing anything
from `embedded-ai-chain` (no `sys.path` reach-back into the parent repo) -
`benchmarks/harness.py` here is a hand-maintained copy of jetson-vlm-lab's own (itself
a copy of `embedded-ai-chain`'s), the same "clone and run standalone" convention every
repo in this fleet follows.

## Watermark / license notice

Qwen2.5: Apache-2.0 (Qwen team), for both the AWQ and GGUF weight sources used here.
No watermarking on output.

This wrapper: no separate license claimed here (internal eval tooling).

## Current State

Scaffolded, not yet run against real hardware - see `docs/TODO.md` Phase 1 for the
two pending smoke tests (vLLM row confirming `configs/models.yaml`'s claimed-good
settings; llama.cpp row confirming the as-yet-unverified `dustynv/llama_cpp` image
tag). No benchmark numbers exist yet.
