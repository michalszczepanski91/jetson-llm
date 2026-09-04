# jetson-llm-lab

Orchestrator-LLM selection lab for NVIDIA Jetson boards (Orin, and eventually Thor -
see `configs/models.yaml`'s `platform` field). Compares candidate models - currently
Qwen2.5-Instruct at 1.5B/3B/7B, with Apertus-8B-Instruct being added as a second model
family (see "Model families" below) - across two serving backends, vLLM and
llama.cpp, so `embedded-ai-chain`'s Phase 3 orchestrator LLM (currently a hardcoded
Qwen2.5-1.5B-Instruct-AWQ on vLLM, chosen because it was "the only option actually
verified working," not because it was benchmarked) can be picked with real numbers.
See `docs/promotion-contract.md` for exactly how a candidate here becomes that.

(Originally scaffolded and named `jetson-llm-qwen` when Qwen2.5 was the only family in
scope - renamed once a second family was added, mirroring `jetson-vlm-lab`'s own
history: that repo started as `jetson-qwen-2.5-VL` and was renamed once Qwen3-VL
broadened its scope the same way.)

See `CLAUDE.md` for the full architecture rationale, in particular why this repo holds
only the container-lifecycle/HTTP-client primitives, not the scene-state/dialogue-loop
integration code, and `docs/TODO.md` for the phased plan (including why TensorRT is
explicitly excluded from v1, not silently skipped).

## Model families

| Family | License | Access | Function-calling | Status |
|---|---|---|---|---|
| Qwen2.5-Instruct (1.5B/3B/7B) | Apache-2.0 | Open, ungated | Explicitly trained for it (vLLM's `--tool-call-parser hermes` works out of the box) | 1.5B smoke-tested for real, 2026-09-04 |
| Apertus-8B-Instruct-2509 | Apache-2.0 | Original repo is **gated** (SNAI Acceptable Use Policy); the GGUF redistribution this lab actually uses (`bartowski/...-GGUF`) is not | **Unconfirmed** - not advertised as a headline feature, no mention found in its model card | Registered in `configs/models.yaml`, not yet run |

Apertus was added as a second family for a reason beyond curiosity: `embedded-ai-chain`
states data-sovereignty as an explicit project goal, and Qwen2.5 is Alibaba-origin,
where Apertus is Swiss-public-funded (ETH Zurich/EPFL) and fully open including its
*training data*, not just its weights - a better philosophical fit if it turns out to
be a viable substitute. Its tool-calling reliability is a genuinely open question this
lab exists to answer, not an assumption either way. No official/trustworthy AWQ
quantization was found for it (see `configs/models.yaml`'s row for what was checked),
so only a llama.cpp/GGUF row exists for it so far - not a vLLM row, to avoid asserting
a fp16-only ~16GB candidate is workable on this device before it's been tried. That
GGUF row uses `bartowski`'s redistribution rather than the original `swiss-ai` repo -
the original is gated behind SNAI's Acceptable Use Policy, but `bartowski`'s re-hosted
GGUF repo is not (confirmed live, not assumed), so staging this row needs no HF
token/gate acceptance at all. A future vLLM row using the original fp16 weights
directly would still need that gate accepted.

## Quick start

```bash
uv venv && uv pip install -r requirements.txt   # or: pip install -r requirements.txt

make serve-1.5b-vllm                             # start vLLM serving Qwen2.5-1.5B-Instruct-AWQ
curl http://localhost:8000/health

make serve-1.5b-llamacpp                         # start llama-server serving the Q4_K_M GGUF
curl http://localhost:8090/health  # not llama.cpp's conventional 8080 - see llm_coordinator.py's port comment

make benchmark CONFIG=1.5b-awq-vllm-orin           # latency/cold-start/thermal/power
make benchmark CONFIG=1.5b-q4-llamacpp-orin
make benchmark-streaming CONFIG=1.5b-awq-vllm-orin # TTFT / tokens-per-sec
make validate-tool-calling CONFIG=1.5b-awq-vllm-orin  # BFCL tool-call judgment accuracy (needs staged data, see below)
make validate-mmlu CONFIG=1.5b-awq-vllm-orin          # MMLU quantization-sanity accuracy (needs staged data, see below)
make test                                          # unit tests, no Docker/GPU needed
```

## Dataset staging

Neither eval dataset is bundled (both are several MB, not appropriate to commit).
Download once per device:

```bash
# BFCL (scripts/validate_tool_calling.py) - 3 files, ~330KB total
huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    BFCL_v3_simple.json BFCL_v3_irrelevance.json possible_answer/BFCL_v3_simple.json \
    --repo-type dataset --local-dir /opt/datasets/BFCL

# MMLU (scripts/validate_mmlu.py) - one parquet file, ~3.5MB
huggingface-cli download cais/mmlu all/test-00000-of-00001.parquet \
    --repo-type dataset --local-dir /opt/datasets/MMLU
```

Both scripts fail fast with a clear message (not a stack trace) if the expected
files aren't at those paths - same convention as jetson-vlm-lab's TextVQA staging.
Staged for real on this device, 2026-09-04.

Apertus's GGUF row (`apertus-8b-q4-llamacpp-orin`) needs no separate staging step or
HF token - it downloads via `bartowski`'s ungated GGUF redistribution, not the
original gated `swiss-ai` repo (see `configs/models.yaml`'s row for how that was
confirmed). Only a future vLLM row using the original fp16 weights directly would
need SNAI's Acceptable Use Policy accepted first at
https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509.

To measure against an already-running remote server (e.g. Thor) instead of starting a
local container, pass `--target remote --remote-host <ip>` to any `scripts/*.py`
directly (no Makefile target - the host is environment-specific):

```bash
uv run python scripts/benchmark.py --model-config 1.5b-awq-vllm-orin --target remote --remote-host <thor-ip>
uv run python scripts/validate_tool_calling.py --model-config 1.5b-awq-vllm-orin --target remote --remote-host <thor-ip>
```

Every `scripts/*.py` accepts `--model-config <name>` (see `configs/models.yaml` for
the full registry) rather than hardcoding a single model, so adding a new
family/size/quantization/backend to evaluate means adding a row to that file, not
editing every script.

## Why BFCL + MMLU, not GQA/TextVQA

`jetson-vlm-lab` (this repo's sibling, same pattern applied to the VLM slot) measures
GQA/TextVQA because the VLM's job is visual QA. The orchestrator LLM's actual job -
and its actual documented failure mode
(`embedded-ai-chain/src/orchestrator_models.py`'s "weakest tool-calling judgment...the
documented ask_vlm escalation gap") - is tool-call judgment: given a transcript and
scene-state grounding, does it call the right tool, or answer directly, correctly and
*unforced* (`tool_choice="auto"`, not the forced-tool_choice workaround
`orchestrator.py` currently needs)?

- **`scripts/validate_tool_calling.py`** measures exactly that using the **Berkeley
  Function-Calling Leaderboard (BFCL)** - the standard, widely-cited benchmark for
  this (`gorilla-llm/Berkeley-Function-Calling-Leaderboard` on HF, Apache-2.0) -
  rather than a hand-rolled case list, so the resulting numbers mean something beyond
  this one repo. Only BFCL's `simple` (one function, AST-matched against the official
  ground truth) and `irrelevance` (correct behavior is calling nothing) categories are
  used - this orchestrator only ever considers one tool call per turn, so
  multi-turn/parallel/multiple-function BFCL categories don't apply. See that script's
  docstring for exactly how scoring works and how it differs from the official
  `bfcl-eval` checker (simplified, good enough for comparing this lab's own candidates
  against each other, not for a leaderboard-exact score).
- **`scripts/validate_mmlu.py`** is a *quantization-regression sanity check*, not a
  model-ranking benchmark - it exists to catch a broken AWQ/GGUF config (wrong quant
  file, broken chat template) by comparing each backend/precision pair against its own
  same-size sibling, not across sizes. A 7B beating a 1.5B on MMLU is not news.

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
Apertus-8B-Instruct-2509: Apache-2.0. The original repo additionally carries SNAI's
Acceptable Use Policy gate; the GGUF redistribution this lab actually uses does not -
see "Dataset staging" above. No watermarking on output mentioned for either family's
model card.

This wrapper: no separate license claimed here (internal eval tooling).

## Current State

Real Docker/GPU smoke tests done, 2026-09-04 (Qwen2.5-1.5B only so far - see
`docs/TODO.md` Phase 1 for the full record):

- **vLLM** (`1.5b-awq-vllm-orin`): cold start ~160s, plain completion and a
  tool-calling completion both succeeded. Two real bugs found and fixed in the
  process (missing `--enable-auto-tool-choice --tool-call-parser hermes` flags on the
  Python coordinator path; a wrong `max_model_len` value in the registry).
- **llama.cpp** (`1.5b-q4-llamacpp-orin`): plain completion succeeded. Tool-calling
  is an **open, unresolved finding** - the server accepted the request (after fixing
  the image tag to one with a new-enough llama.cpp build) but the model described
  wanting to call the tool in free text rather than emitting a structured response,
  where the identical prompt worked correctly on vLLM. Not yet explained by a config
  fix - `make validate-tool-calling` against this row is what will actually quantify
  whether this is a real, consistent gap.

No BFCL/MMLU scorecard numbers exist yet, and Apertus hasn't been run at all (gated -
see "Dataset staging"). Full matrix is `docs/TODO.md` Phase 3.
