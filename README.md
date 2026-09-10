# jetson-llm

Orchestrator-LLM selection lab for NVIDIA Jetson boards - **both Orin and Thor are
live platforms now** (see `configs/models.yaml`'s `platform` field). Compares
candidate models - currently Qwen2.5-Instruct at 1.5B/3B/7B/14B, plus
Apertus-8B-Instruct and Bielik-11B-v3.0-Instruct as additional model families (see
"Model families" below) - across **three serving backends: vLLM, llama.cpp, and
TensorRT Edge-LLM** (Thor-only, since it needs JetPack 7.x), so
`embedded-ai-chain`'s Phase 3 orchestrator LLM (currently a hardcoded
Qwen2.5-1.5B-Instruct-AWQ on vLLM, chosen because it was "the only option actually
verified working," not because it was benchmarked) can be picked with real numbers.
See `docs/promotion-contract.md` for exactly how a candidate here becomes that.

**First controlled comparison is done** - see `docs/thor-framework-comparison.md`.
Two headline results, both on Thor with the model held fixed:

- **Framework** (at 7B, the size that matters): the three backends are within ~2% on
  turn latency and ~16% on energy per token, and differ by **40 points on escalation
  judgement** - Edge-LLM 94% `irrelevance`, llama.cpp 62%, vLLM 54%. That single axis
  decides it. If tool-call judgement did not matter, llama.cpp would win outright
  (fastest, most energy-efficient, one `docker pull` instead of a source build).
  **Not a decoding artifact**: forcing deterministic argmax (`top_k=1`) on all three
  reproduced every score exactly.
- **Size and framework interact - the backend decides whether size helps at all.**
  1.5B -> 7B moves BFCL `irrelevance` (the "correctly call NO tool" category, i.e. the
  documented `ask_vlm` over-escalation gap) from **78% to 94% on Edge-LLM**, but only
  60% -> 62% on llama.cpp, and it makes vLLM *worse* (68% -> 54%). Thor's 122GB is
  what makes a 7B orchestrator purchasable at all - the Orin's ~30GB never could - but
  the parameters only pay off on a backend that lets the model abstain.
- **Stop at 7B**: 14B agrees with 7B on 98/100 cases while costing 2x the latency and
  2.1x the energy per token.

Timing and energy numbers are from an exclusive-box campaign
(`scripts/thor_exclusive_window.sh`); marginal energy subtracts a measured 5.42 W idle
baseline. Worth knowing for co-residency planning: an idle-but-resident model container
raised board idle from 5.4 W to ~17.5 W, i.e. ~12 W to hold weights doing nothing.

(Originally scaffolded and named `jetson-llm-qwen` when Qwen2.5 was the only family in
scope - renamed once a second family was added, mirroring `jetson-vlm-lab`'s own
history: that repo started as `jetson-qwen-2.5-VL` and was renamed once Qwen3-VL
broadened its scope the same way.)

See `CLAUDE.md` for the full architecture rationale, in particular why this repo holds
only the container-lifecycle/HTTP-client primitives, not the scene-state/dialogue-loop
integration code, and `docs/TODO.md` for the phased plan. Note TensorRT was excluded
from v1 on 2026-09-04 and that exclusion was **superseded for Thor only** on
2026-09-08 - Edge-LLM 0.10.1 turned out to ship an OpenAI-compatible server with
tool-calling, and Thor/JetPack 7.1 is an Official row in its support matrix. The
reasoning for both the original decision and the reversal is in Phase 0.

## Model families

| Family | License | Access | Function-calling | Status |
|---|---|---|---|---|
| Qwen2.5-Instruct (1.5B/3B/7B) | Apache-2.0 | Open, ungated | Explicitly trained for it - reliable on both backends once `temperature=0.1` is pinned for llama.cpp (see below) | 1.5B smoke-tested for real, 2026-09-04 - first real BFCL scorecard: 75% overall (85% simple / 65% irrelevance, n=40) on llama.cpp |
| Apertus-8B-Instruct-2509 | Apache-2.0 | Original repo is **gated** (SNAI Acceptable Use Policy); the GGUF redistribution this lab actually uses (`bartowski/...-GGUF`) is not | Unconfirmed - moot for now, see Status | **BLOCKED**: llama.cpp doesn't recognize Apertus's GGUF architecture at all (`unknown model architecture: 'apertus'`), confirmed on two build versions, 2026-09-04. No AWQ exists for a vLLM row either - **no working serving path in this lab right now** |
| Bielik-11B-v3.0-Instruct | Apache-2.0 | Original repo is gated; SpeakLeash's own official GGUF and "-awq" repos are not | **llama.cpp: working** (structured call on the first try). **vLLM: NOT working** - two parsers tried (`hermes`, `llama3_json`), both fail identically | Both backends smoke-tested for real, 2026-09-04 |

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

make experiment EXP=configs/benchmarks/smoke.yaml    # the reportable path - config-driven
make experiment EXP=configs/benchmarks/output_sweep.yaml DRY=1   # see it expand without starting
# One config file defines the whole campaign (candidates, workload grid, sampling,
# warmup/repetition floors, standalone/co-resident) - no CLI flag carries experimental
# meaning. See configs/benchmarks/ for what ships: smoke (Tier 0), output_sweep
# (the P3 experiment - see "Current State"), context_sweep.

make benchmark-streaming CONFIG=1.5b-awq-vllm-orin CONDITION=standalone  # one ad-hoc cell
# CONDITION is required and has no default: on unified memory a standalone and a
# co-resident number are different physical quantities. Refused outright (no override
# flag) if a Docker container OR a bare-metal process is found holding a GPU handle
# and wasn't declared - see CLAUDE.md's "Measurement integrity" for why that bare-metal
# check exists. Results land in results/raw/<experiment_id>/, schema-validated at
# write time; a result caught as mislabelled after the fact is archived to
# results/invalid/, never silently deleted.

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
uv run python scripts/benchmark_streaming.py --model-config 1.5b-awq-vllm-orin \
  --execution-condition standalone --target remote --remote-host <thor-ip>
# scripts/benchmark.py is retired (docs/TODO.md Phase 2) - see CLAUDE.md
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

### Phase 2-4: measurement integrity, then a real campaign (2026-09-04)

This repo grew from a candidate-selection lab into a reproducible benchmark suite the
same day the Phase 1 smoke tests below were done - see `docs/note.md` for the
methodology and `docs/TODO.md` for the phased plan. The short version: three
measurement bugs were found and fixed by actually running the instrument, then a real
campaign ran clean on both backends.

**What got fixed, in the order it was found:**

1. **Prefix-cache contamination - a 6.6x TTFT error.** Sending the identical prompt
   every repetition meant both backends served every rep after the first from a
   cached prefill. TTFT p50 went 40.8ms -> **268.1ms** once prompts vary per run
   (now the default, `--prompt-uniqueness unique-per-run`).
2. **An inert HF cache mount.** `HUGGINGFACE_HUB_CACHE` baked into the vLLM image
   overrode `HF_HOME`, so the `/opt/hf-cache` volume did nothing and every vLLM run
   re-downloaded its weights into a `--rm` container. Cold start: 156.2s -> **136.2s**
   once fixed (both `VllmCoordinator` and `docker-compose.yml`).
3. **A `standalone` claim that wasn't - twice.** A container-only check let a real
   campaign run labelled `standalone` while `embedded-ai-chain`'s entire production
   pipeline (YOLO + orchestrator + STT/TTS) ran as one bare-metal process the whole
   time, invisible to `docker ps`. A second, subtler version of the same gap: 5
   already-committed `co-resident` results had *correctly* said co-resident but
   named only `vllm-orchestrator`, missing that same bare-metal process. Both are now
   closed: `assert_condition_matches_reality()` checks Docker containers **and**
   bare-metal GPU-holding processes (`/proc/<pid>/fd` for `nvhost`/`nvgpu`/`nvmap`
   handles), and checks a `co-resident` declaration for **completeness**, not just
   whether `standalone` is literally false. No override flag on either check. Full
   incident record in `docs/HISTORY.md`, "Phase 3".

Energy (J/output-token) also went from uncomputable to real: the script with power
telemetry didn't know token counts, the one with token counts started no sampler.
`benchmarks/runner.py:measure_cell()` merges the two paths.

**The campaign itself**, once the board was confirmed genuinely quiet (containers
stopped, bare-metal pipeline stopped, by direct user confirmation - this device's
production system is not this repo's to touch unilaterally): `output_sweep` (12
cells) and `context_sweep` (8 cells), both backends, `standalone`, zero failures, zero
errors, all schema-conformant. Two real signals, one replicate each:

- **J/output-token amortizes with generation length**, both backends - highest at the
  shortest output (16 tokens: 0.473 J/tok vLLM, 0.949 llama.cpp), settling lower by
  128+ tokens (~0.297 / ~0.66). The shape `embedded-ai-chain/docs/paper.md`'s P3
  predicts, reproduced cleanly on real hardware.
- **A backend-specific context-scaling difference.** TTFT vs input length (128 to
  1536 tokens): vLLM stays close to flat (40.8ms -> 81.8ms), llama.cpp is clearly
  super-linear (213.2ms -> 897.7ms, roughly doubling from 512->1024 alone). Decode
  throughput is flat on both, so this is specifically a *prefill* difference.
  Confounded by quantization format (AWQ vs GGUF Q4_K_M) like every cross-backend
  comparison in this lab so far - stated, not fixed.

See `docs/project.diagram.md` §6 for the full tables and `docs/TODO.md` Phase 4 for
the complete record, including what's *not* yet done (an analysis script generating
plots from `results/raw/`, and drafting these numbers into `paper.md` itself).

### Phase 1: candidate serving/tool-calling smoke tests (2026-09-04)

Real Docker/GPU smoke tests done, 2026-09-04 - see `docs/TODO.md` Phase 1 for the
full record:

### Thor, 2026-09-08/10 — three backends, a framework comparison, and a working quantization

Full write-up in `docs/thor-framework-comparison.md` (numbers of record - exclusive
box, no other GPU workload); raw JSON in `output/`. Model fixed at
`Qwen2.5-7B-Instruct` (**FP16 for the framework comparison, not the Orin rows' AWQ -
Edge-LLM cannot build that checkpoint**, see below):

| | BFCL `irrelevance` | 32-token turn | marginal J/token |
|---|---|---|---|
| **Edge-LLM** | **94%** | 1945 ms | 0.85 J |
| llama.cpp | 62% | **1899 ms** | **0.73 J** |
| vLLM | 54% | 2539 ms | 0.96 J |

**Recommendation: Edge-LLM serving `Qwen2.5-7B-Instruct`, quantized GPTQ-Int4 if the
turn budget is tight, FP16 otherwise** - a real bug in Edge-LLM's own source blocked
every int4 checkpoint until it was found and patched (`patches/`):

| precision | BFCL `irrelevance` | 32-token turn | marginal J/token | cold start |
|---|---|---|---|---|
| **INT4-GPTQ** | 90% | **769 ms** | **0.274 J** | **0.03 s** ¹ |
| **FP16** | **94%** | 1945 ms | 0.839 J | 12.1 s ¹ |

¹ Cold start is a cache **hit** for both - a miss (first run, or an empty cache)
takes minutes. Only llama.cpp's cold start is unconditionally fast.

Also found: **14B buys nothing** (98/100 identical BFCL verdicts to 7B at 2x the
latency), the backend gap is **not a decoding or parser artifact** (a `top_k=1`
greedy control reproduced every score exactly), and self-quantized FP8/INT8-SQ
needed real debugging before they were trustworthy - see the comparison doc.

Notes on the three Thor backends:

- **Edge-LLM** has no wheel and no image - it is built from source on-device (~1h).
  It **cannot build Qwen2.5's AWQ checkpoint at all** (`external FP16 bias has no
  checkpoint recipe` - a one-line omission in its own source, since patched; see
  "GPTQ-Int4" in the comparison doc), which is why every Thor row is FP16 or
  self-quantized rather than AWQ.
- **llama.cpp on Thor needs a different image family than Orin**: `dustynv/llama_cpp`
  has no r38 tag on Docker Hub - jetson-containers publishes r38-era images to GHCR
  under `nvidia-ai-iot/` instead, and only that repo's *dated* `b10373-*` tag actually
  works (its own rolling tag and upstream's generic CUDA build both fail).
- **vLLM on Thor** needs its own `-jetson-thor` image tag, not `-jetson-orin`.

### Orin, 2026-09-04

Real Docker/GPU smoke tests - see `docs/TODO.md` Phase 1 for the full record:

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
- **Bielik-11B on vLLM** (`bielik-11b-awq-vllm-orin`): serves correctly at
  `gpu_memory_utilization=0.3` (cold start 234-274s), plain completion and a
  Polish-language sanity check both correct - but **tool-calling does not work**.
  Tried two parsers (`--tool-call-parser hermes` and `llama3_json`), both returned
  an empty `tool_calls` list on the same prompt that worked on vLLM+Qwen2.5 and
  llama.cpp+Bielik. Root cause: Bielik's own docs only describe tool use as a
  manual prompt-injection convention, not training on a tagged format any of
  vLLM's per-family parsers detect - the opposite pattern from Qwen2.5, which
  works on both backends.

### Phase 5: the full 7-row scorecard (2026-09-08)

BFCL (bounded ~30/category sample, n=60) and MMLU (n=200) now exist for all 7 active
rows - see `docs/TODO.md` Phase 5 for the full table, the sampling decision (full
BFCL corpus costed out at 6.9-95min/row from measured decode speeds; a uniform limit
safe for the slowest row would starve the fastest, so the scorecard uses a comparable
bounded sample on every row instead), and the tool-calling confusion matrix now
recorded per row (`tool_call_outcomes` in each BFCL result,
`confusion_matrix_and_taxonomy()` in `scripts/validate_tool_calling.py`).

**The BFCL scorer was corrected 2026-09-09** and every row re-run: argument strings
were compared with a plain `.strip().lower()` where official bfcl-eval first strips
` ,./-_*^` and spaces, so `"3*x**2 + 2*x - 1"` scored as wrong against BFCL's accepted
`"3x**2 + 2x - 1"` - the same maths, and the only spelling that is valid Python.
Re-scoring identical model outputs under both rules flipped **21 cases across the 7
rows**, all in the same four maths cases Thor independently hit (+3.3pp on the weakest
row, +10 to +13.3pp on every other). Rankings did not change.

The consequence is that **`simple` is now saturated** - five of seven rows score
exactly 100% - so **`irrelevance` is the only accuracy axis still discriminating**,
independently reproducing Thor's conclusion on different hardware and quantization.
Two findings there, before either family is called a tool-calling winner:
**Bielik-11B abstains on only 10% of irrelevance cases** despite a perfect 100% on
`simple` - a real false-positive/unwanted-actuation risk its headline number hides -
and **7B shows a real backend gap on irrelevance** (40.0% vLLM vs 56.7% llama.cpp,
same model/size), consistent with this lab's running theme that backend, not just
model, moves the number, and with Thor's much larger same-parser version of the same
gap. Confounded here by AWQ vs GGUF Q4_K_M as usual.

Performance/energy is done for all 7 rows at a confirmed-consistent platform state
(`jetson_clocks_locked: true`, MAXN, standalone) - mid-campaign the board was found
to be hard-resetting under sustained load on an underspec'd 65W supply (confirmed via
the Tegra PMC's `reset_reason=SYS_RESET_N` register - measured peak draw hit 58.4W on
3 rails alone against the devkit's specified 90W requirement). Fixed by swapping the
adapter, then the 5 rows Phase 4 never touched were re-measured as a clean replicate.
That correction surfaced a real finding, not just a methodology fix: **llama.cpp's
TTFT roughly halves once `jetson_clocks` is genuinely locked** (Bielik-11B
3181ms->1787ms, 7B 1984ms->1018ms, 3B 1284ms->563ms) - the uncorrected numbers were
partly measuring GPU clock ramp-up (DVFS) latency between requests, not pure
inference. vLLM barely moved, since it keeps the GPU continuously saturated and never
idles down between requests. Full incident, tables, and decision log in
`docs/TODO.md` Phase 5.
