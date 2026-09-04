# jetson-llm

Orchestrator-LLM selection lab for NVIDIA Jetson boards (Orin, and eventually Thor -
see `configs/models.yaml`'s `platform` field). Compares candidate models - currently
Qwen2.5-Instruct at 1.5B/3B/7B, plus Apertus-8B-Instruct and Bielik-11B-v3.0-Instruct
as additional model families (see "Model families" below) - across two serving
backends, vLLM and llama.cpp, so `embedded-ai-chain`'s Phase 3 orchestrator LLM (currently a hardcoded
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
| Qwen2.5-Instruct (1.5B/3B/7B) | Apache-2.0 | Open, ungated | Explicitly trained for it - reliable on both backends once `temperature=0.1` is pinned for llama.cpp (see below) | 1.5B smoke-tested for real, 2026-09-04 - first real BFCL scorecard: 75% overall (85% simple / 65% irrelevance, n=40) on llama.cpp |
| Apertus-8B-Instruct-2509 | Apache-2.0 | Original repo is **gated** (SNAI Acceptable Use Policy); the GGUF redistribution this lab actually uses (`bartowski/...-GGUF`) is not | Unconfirmed - moot for now, see Status | **BLOCKED**: llama.cpp doesn't recognize Apertus's GGUF architecture at all (`unknown model architecture: 'apertus'`), confirmed on two build versions, 2026-09-04. No AWQ exists for a vLLM row either - **no working serving path in this lab right now** |
| Bielik-11B-v3.0-Instruct | Apache-2.0 | Original repo is gated; SpeakLeash's own official GGUF repo is not | **Confirmed working**, 2026-09-04 - correctly returned a structured tool call on the first real test | Smoke-tested for real, 2026-09-04 - the strongest first result of any llama.cpp row so far |

Apertus and Bielik were added for different reasons. Apertus for `embedded-ai-chain`'s
own stated data-sovereignty goal: Qwen2.5 is Alibaba-origin, where Apertus is
Swiss-public-funded (ETH Zurich/EPFL) and fully open including its *training data*,
not just its weights - a better philosophical fit if it had turned out to be a viable
substitute, though its llama.cpp path is currently blocked outright (see table).
Bielik was added at direct user request as a Polish-developed model (SpeakLeash/ACK
Cyfronet AGH) and turned out to be this lab's best llama.cpp result yet - see
`docs/TODO.md` Phase 1 for the full comparison against Qwen's own llama.cpp attempt.

Neither Apertus nor Bielik has an official/trustworthy AWQ quantization (checked via
HF Hub search for both), so neither has a vLLM row - only llama.cpp/GGUF, to avoid
asserting an untested fp16-only candidate is workable on this device. Both GGUF rows
use a redistribution rather than each family's original (gated) repo: Apertus via
`bartowski`'s third-party quant, Bielik via SpeakLeash's own official GGUF repo -
neither redistribution is gated (confirmed live for both, not assumed), so staging
either needs no HF token/gate acceptance. A future vLLM row using either family's
original fp16 weights directly would still need that family's own gate accepted.

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

Apertus's and Bielik's GGUF rows need no separate staging step or HF token - both
download via an ungated GGUF redistribution rather than each family's original gated
repo (see `configs/models.yaml` for how that was confirmed for each). Only a future
vLLM row using either family's original fp16 weights directly would need that
family's own gate accepted first (SNAI's Acceptable Use Policy for Apertus at
https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509; SpeakLeash's gate for
Bielik at https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct). Apertus's row
is moot regardless for now - see "Model families" above, it's blocked at the
llama.cpp level entirely, not a staging problem.

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
Apertus-8B-Instruct-2509 and Bielik-11B-v3.0-Instruct: both Apache-2.0. Both original
repos additionally carry their own gate (SNAI's Acceptable Use Policy for Apertus,
SpeakLeash's own gate for Bielik); the GGUF redistributions this lab actually uses do
not - see "Dataset staging" above. No watermarking on output mentioned for any
family's model card.

This wrapper: no separate license claimed here (internal eval tooling).

## Current State

Real Docker/GPU smoke tests done, 2026-09-04 - see `docs/TODO.md` Phase 1 for the
full record:

- **Qwen2.5-1.5B on vLLM** (`1.5b-awq-vllm-orin`): cold start ~160s, plain completion
  and a tool-calling completion both succeeded. Two real bugs found and fixed in the
  process (missing `--enable-auto-tool-choice --tool-call-parser hermes` flags on the
  Python coordinator path; a wrong `max_model_len` value in the registry).
- **Qwen2.5-1.5B on llama.cpp** (`1.5b-q4-llamacpp-orin`): plain completion succeeded.
  The initial tool-calling failure (model narrating instead of calling the tool) was
  **resolved** - 15 repeated identical calls at default sampling temperature (~0.8)
  succeeded only 12/15 (80%), 15/15 (100%) at `temperature=0.1`. vLLM was 100% at
  both temperatures, so this is a real llama.cpp-specific sensitivity, not a
  framework-wide limitation - `scripts/validate_tool_calling.py` now pins
  `--temperature 0.1` by default to fix it (costs vLLM nothing). A second real bug
  found while re-running the real script against actual BFCL data: llama.cpp
  hard-errors on BFCL's non-standard `"float"`/`"tuple"`/`"any"` schema type names
  (vLLM tolerates them silently) - fixed in `_bfcl_function_to_openai_tool()`. First
  real scorecard after both fixes (`--limit 20`, 40 cases): **75% overall** (85%
  simple / 65% irrelevance) - a small sample, not yet the full-corpus number.
- **Apertus-8B on llama.cpp** (`apertus-8b-q4-llamacpp-orin`): **BLOCKED** - llama.cpp
  doesn't recognize Apertus's GGUF architecture at all, confirmed on two build
  versions. No vLLM path exists either (no AWQ). No working serving path in this lab
  right now.
- **Bielik-11B on llama.cpp** (`bielik-11b-q4-llamacpp-orin`): **working**, and the
  strongest first-look llama.cpp result - cold start 338s (incl. first download),
  plain completion succeeded, and a tool-calling completion correctly returned a
  structured call on the first try.

Real BFCL numbers now exist for one row (`1.5b-q4-llamacpp-orin`, above, small
sample). Everything else in `docs/TODO.md` Phase 3's full matrix is still pending.
