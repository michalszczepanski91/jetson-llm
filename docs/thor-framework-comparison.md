# Thor framework comparison — which serving backend, and which model size?

Answers two questions for the orchestrator LLM on Jetson Thor: which of vLLM,
llama.cpp, and TensorRT Edge-LLM to serve it with, and which precision of
`Qwen2.5-7B-Instruct` to run. Measured 2026-09-08/10 on the user's Thor (JetPack 7.1,
CUDA 13.0, TensorRT 10.13.3.9, 122 GiB unified memory). `docs/thor-precision-sweep.html`
is a visual walkthrough of what each timing/energy column below actually measures -
open it in a browser if the prose definitions in "Reading the numbers" aren't enough
on their own. Raw JSON in `output/`; the
full engineering story — dead-end images, a real bug found in Edge-LLM's own source,
a self-quantization debugging saga — is kept below in "How we got here" rather than
mixed into the numbers.

## The decision

**TensorRT Edge-LLM serving `Qwen2.5-7B-Instruct`, quantized GPTQ-Int4 if the turn
budget is tight, FP16 otherwise.**

| precision | where the checkpoint came from | BFCL `irrelevance` | 32-token turn | marginal J/token | cold start ¹ |
|---|---|---|---|---|---|
| **INT4-GPTQ** | **downloaded — Qwen's own** | 90% | **769 ms** | **0.274 J** | **0.03 s** |
| **FP16** | **downloaded — Qwen's own** | **94%** | 1945 ms | 0.839 J | 12.1 s |
| FP8 | self-quantized here | 90% | 1052 ms | 0.603 J | — |
| INT8-SQ | self-quantized here | 92% ² | 1074 ms | 0.610 J | — |

¹ Cache **hit**; a miss compiles the engine and takes minutes. Only llama.cpp's cold
start is unconditionally fast. ² INT8-SQ still has an unexplained 12.5-point MMLU gap
(54.0% vs FP16's 67.5%) — not trusted for production; see "Still open."

INT4-GPTQ wins every timing and energy column outright, for 4 points of `irrelevance`.
FP8 is essentially lossless on accuracy but only ~1.4× FP16's throughput, nowhere near
INT4's 2.5×. Which to ship is a genuine product call about the turn budget.

**Both recommended precisions are official Qwen checkpoints, downloaded — not
quantized by us.** That matters for reproducing this: `Qwen2.5-7B-Instruct` and
`Qwen2.5-7B-Instruct-GPTQ-Int4` are both `hf download`s, so the recommendation needs
**no quantization pipeline at all** — just the one-line Edge-LLM patch in `patches/`,
without which *every* int4 checkpoint fails to build, Qwen's official ones included.
Self-quantization was necessary only for FP8 and INT8-SQ, because **no published
checkpoint exists for this model in either format** — and neither of those is
recommended. Recipes for them, with the two failure modes that made the first attempt
at each unusable, are in `scripts/quantize_edgellm.sh`.

**Why Edge-LLM and not vLLM or llama.cpp** — same model, same weights, same 100 BFCL
cases, FP16:

| | Edge-LLM | llama.cpp | vLLM |
|---|---|---|---|
| **BFCL `irrelevance`** | **94%** | 62% | 54% |
| Turn latency (32 tok) | 1945 ms | **1899 ms** | 2539 ms |
| Energy (marginal J/token) | 0.85 J | **0.73 J** | 0.96 J |

If tool-call judgment didn't matter, llama.cpp would win outright — it's the fastest
and most energy-efficient, and needs one `docker pull` instead of a source build. It
loses because it structurally cannot abstain (see "Why the gap exists" below). vLLM —
`embedded-ai-chain`'s current production backend — is worst on every axis measured
here, at either precision.

## Reading the numbers

- **TTFT** (time to first token): wall-clock until the first token arrives, dominated
  by prompt processing. What a user perceives as "how long before it starts talking."
- **tok/s**: decode throughput once generation is underway. `1 / tok/s` is the
  average time per token.
- **32-token turn**: end-to-end time for one complete, non-streaming response of 32
  tokens — a realistic reply or tool call, not a 128-token essay and not one token.
  Measured by a *separate* non-streaming call from TTFT/tok-s, so the two aren't
  arithmetically identical (a 32-tok turn isn't exactly `TTFT + 31 × (1/tok⁄s)`).
- **Marginal J/token**: `(board power while generating − idle board power) ÷ tok/s`.
  Isolates what generation itself costs, subtracting the **5.42 W** the board draws
  fully idle (measured with the box quiet, no containers). Board power is the whole
  board's 5 V rail, not GPU-only.
- **`irrelevance`**: the BFCL category where the correct action is to call **no**
  tool. It's the direct analogue of `embedded-ai-chain`'s documented `ask_vlm`
  over-escalation bug, and the only BFCL axis that still discriminates between
  capable models — see "BFCL `simple` is saturated" below.

## Why the gap exists, and what it isn't

**It's abstention, not detection.** All three backends score 90–92% on `simple` (a
tool call is wanted) at 7B — identical, even with the same parser pinned on both
Edge-LLM and vLLM. The entire 40-point spread is in `irrelevance`. llama.cpp
constrains generation to a tool-call grammar: a structurally-forced call leaves no
room to abstain, so its `irrelevance` sits near 60% regardless of model size (60%→62%
from 1.5B→7B). vLLM and Edge-LLM detect tool tags in free generation instead — which
is why Edge-LLM can turn extra capacity into judgment (78%→94%) while vLLM gets
**worse** with more parameters (68%→54%).

**Ruled out, not assumed:**
- *Not the parser* — Edge-LLM scored identically with `auto` and `hermes` pinned,
  100/100 identical per-case verdicts at both sizes.
- *Not decoding/sampling* — the three backends turned out to apply three different
  truncation defaults (a real flaw in the original setup, which had pinned only
  `temperature`). Forcing `top_k=1` (deterministic argmax on all three, making
  `top_p`/`min_p`/temperature inert) reproduced every backend's own score exactly.
- *Not chat-template rendering* — **2026-09-10, and this was the leading candidate
  until it was tested.** `scripts/chat_template_crossfeed.py` captured what each
  backend actually sends (vLLM via `POST /tokenize` with `return_token_strs`,
  llama.cpp via `POST /apply-template`, Edge-LLM through its own
  `ToolChatTemplateFormatter` — the class `runtime/engine.py` calls — checked against
  the live server's own `usage.prompt_tokens`, 224 == 224). **Edge-LLM and vLLM build
  byte-identical prompts on all 50 cases.** Crossfeeding each rendering to each
  backend that has `/v1/completions` moves abstention by at most 4 points, while
  swapping the backend moves it 32–40:

  | serving ↓ / rendering → | Edge-LLM (canonical) | llama.cpp (braces) | own chat path |
  |---|---|---|---|
  | vLLM | 52% | 56% | 54% |
  | llama.cpp | 60% | 62% | 62% |
  | Edge-LLM | *no `/v1/completions`* | — | 94% |

  llama.cpp's rendering *does* differ from the other two on 50/50 cases — it emits
  `{{"name": ...}}` where the canonical Qwen2.5 template emits `{"name": ...}`, a real
  template bug (226 vs 224 tokens, HF and llama.cpp tokenizers agreeing exactly) — and
  it is worth ~2 points, not 32.
- *Not numeric dtype* — the one structural difference the crossfeed turned up:
  Qwen2.5-7B-Instruct's config.json is `torch_dtype: bfloat16`, vLLM follows it, and
  Edge-LLM's built engine is `"dtype": "F16"`. Same bytes in, different numbers.
  Forcing vLLM to `--dtype float16` (`7b-fp16dtype-vllm-thor`) scored `irrelevance`
  **54.0%, identical to bf16** — though not a null result: 6 of 50 irrelevance
  verdicts flip and `simple` improves 92%→98%, the flips simply cancel. llama.cpp is
  already fp16 at 62%, so dtype could not have explained the ordering anyway.

  With weights, parser, decoding, prompt bytes and dtype all controlled, the gap is
  still structural and **still unidentified**. What is left sits inside Edge-LLM's
  own runtime — its TensorRT kernels and KV-cache handling are the untested surface —
  and is no longer cheap to probe from the outside.

**14B buys nothing.** It agrees with 7B on 98/100 BFCL cases — not approximately,
byte-identical aggregates — while costing 2× the latency and 2.1× the energy. Thor's
122 GiB makes 14B *possible* (Orin's ~30 GB never could), but the memory is better
spent on co-residency headroom or a bigger KV cache.

**BFCL `simple` is saturated.** Both 7B and 14B fail the same four cases — and in
every one, the model called the exactly right function; the failures are argument
*formatting* mismatches (`"x**2"` vs `"lambda x: x**2"`) that this repo's simplified
AST checker doesn't normalize, not model errors. `simple` has an ~8-point
false-negative floor; `irrelevance` is the axis actually worth trusting.

**A quiet box is not the fast box, for Edge-LLM specifically.** It ran 27–37% slower
alone than beside a busy neighbor, while drawing *less* power — pointing at Thor's
dynamic GPU clocks rather than contention (vLLM/llama.cpp barely moved). Since
production co-resides with the VLM tier, the shared condition may be closer to
reality than the clean one. Unconfirmed without a `jetson_clocks`-locked run (needs
root).

## How we got here

<details>
<summary>Serving viability — five of eight image/backend paths tried do not work on Thor</summary>

| Path | Outcome |
|---|---|
| **Edge-LLM**, built from source (0.10.1) | Works. No wheel or image exists; ~1h on-device build, no sudo, no GPU needed. |
| **vLLM** `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` | Works, once the memory-profiling assert below is patched around. Tag differs from the Orin default (`-jetson-orin`). |
| **llama.cpp** `ghcr.io/nvidia-ai-iot/llama_cpp:b10373-r38.2...` | Works — the dated `b*` tag from NVIDIA's GHCR. |
| llama.cpp `dustynv/llama_cpp` (Orin's source) | No Thor tag on Docker Hub — jetson-containers publishes r38-era images to GHCR instead. |
| llama.cpp `ghcr.io/ggml-org/llama.cpp:server-cuda` | Enumerates the GPU and loads the model, then dies on the first inference (`cublas_handle`). Generic CUDA arm64 targets discrete GPUs, not Tegra — device enumeration is not a working-backend test. |
| llama.cpp `nvidia-ai-iot/llama_cpp:r38.2...` (rolling tag) | Fails at startup — ships a CUDA-12 binary inside a CUDA-13 image. |
| Edge-LLM + `Qwen2.5-*-Instruct-AWQ` | Cannot build (see the bias-recipe bug below) — this is why every Thor row is FP16, not AWQ. |
| Apertus-8B (Orin) | Blocked — llama.cpp doesn't recognize its GGUF architecture. |

**vLLM's own bug found along the way**: it asserts free memory never *increases*
during startup profiling and kills the engine when it does — observed on a
completely idle box at 7B. `embedded-ai-chain` hit the same assert in August and
ships a patched `gpu_worker.py`; the Thor vLLM rows mount it. Affects startup only,
not inference, so it can't flatter a performance number.

</details>

<details>
<summary>The Edge-LLM quantization bug — root cause, patch, and what it unblocked</summary>

Every int4 checkpoint tried (Qwen's official AWQ, GPTQ-Int4, and a mis-routed
GPTQ-Int8) failed identically: `ValueError: external FP16 bias has no checkpoint
recipe`. Root cause, found by reading Edge-LLM's own source rather than assumed:
`weights.py`'s `linear_metadata()` computes the bias recipe correctly for every
quant type, and the FP16 return path uses it — but the shared int4 return statement
simply omits `bias_recipe=bias_recipe` from its kwargs. Not a checkpoint problem; a
one-line omission that hits any int4 checkpoint on any architecture with a linear
bias (Qwen2's `q_proj`/`k_proj`/`v_proj`).

Patched the local clone (`patches/edgellm-int4-bias-recipe.patch`, pure Python, no
rebuild). Re-ran `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` and it served correctly —
coherent completions, correctly structured unforced tool calls. This unblocked the
INT4-GPTQ row in the decision table above.

</details>

<details>
<summary>Self-quantized FP8/INT8-SQ — needed real debugging before they were trustworthy</summary>

No published checkpoint exists in FP8 or true INT8 format for this model (checked
against NVIDIA's own 0.10.1 supported-models page). Self-quantized both with
Edge-LLM's own `tensorrt-edgellm-quantize` (ModelOpt-based) — and the first attempt
at each came back **damaged**, not merely suboptimal:

| | MMLU | BFCL simple | what that meant |
|---|---|---|---|
| FP8, 1st attempt (`--kv_cache_quantization fp8`) | 46.5% | **0%** | never emitted a single tool call in 50 tries — its "100% irrelevance" was calling nothing, ever, not judgment |
| INT8-SQ, 1st attempt (128 calibration samples) | 44.5% | 90% | tool-calling intact, but 23 points of MMLU lost |

Two fixes, one variable each: dropping `--kv_cache_quantization` (the tool's own
calibration log had warned about exactly this) fixed FP8 completely — 0%→91% BFCL
simple, 46.5%→66.5% MMLU. Raising calibration samples 128→512 for INT8-SQ helped
substantially (44.5%→54.0% MMLU) but didn't fully close the gap to FP16's 67.5% — see
"Still open." **The lesson**: a quantization run completing without error is not
evidence the result is usable; only running the same accuracy suite every other
candidate went through caught this.

Also found and fixed while chasing this: `scripts/benchmark_streaming.py` read the
raw config path instead of the coordinator's resolved model name, 404-ing every
streaming request for a self-quantized (locally-served) checkpoint while
BFCL/MMLU/latency succeeded against the same server. Silent until now because
vLLM/llama.cpp rows use an HF repo id as both values.

</details>

## Still open

- [ ] `jetson_clocks`-locked run to settle the idle-box DVFS effect (needs root).
- [x] **Official `bfcl-eval` checker: RUN 2026-09-11, and the premise was backwards.**
      `scripts/rescore_bfcl_official.py` re-scores recorded `simple` outcomes with the
      real `bfcl_eval` AST checker (installed `--no-deps` into an isolated venv, with
      only `constants.model_config` stubbed - that one module is why the import chain
      pulls torch, and its sole use in `ast_checker.py` is a registry lookup this lab
      does not need). No re-generation, no GPU: the arguments are already in
      `outcomes.jsonl`. Across all 10 re-scorable sets the official checker scores
      **2.0-3.3 points LOWER** than this repo's, never higher. Our matcher is more
      LENIENT, not more strict, so the documented "~8-point false-negative floor" does
      not exist - and swapping checkers would not stop `simple` saturating, because
      official scores still land in a 77-97% band with everything 3B-and-up at 93%+.
      `simple` is uninformative because the models are genuinely good at it, not
      because the scorer is wrong. **Recommendation: keep the simplified checker as the
      default** (making `bfcl-eval` a hard dependency would drag torch into a Jetson
      venv for a <=3.3-point correction) and use this script as the audit tool.
      Almost all of the disagreement is a single case, `simple_13`: the model sends
      `interval: [1, 3]` where the schema declares an array of `float`, which the
      official checker rejects as `type_error:nested` and ours accepts.
- [ ] **Retain per-case arguments in every accuracy result.** Found by the above: the
      9 results written before 2026-09-09 record only the verdict
      (`correct_arguments: true/false`), not `actual_arguments`/`acceptable_arguments`,
      so they cannot be re-scored by any independent checker, ever. Newer runs do.
      Nothing to fix in the code (it already records them); the gap is that those 9
      results are permanently un-auditable and should be re-run rather than trusted if
      their `simple` numbers ever matter.
- [ ] Cold-start-from-empty-cache as its own measurement for Edge-LLM and vLLM.
- [ ] Co-residency run: orchestrator + VLM tier together — the real deployment
      shape. Known price tag so far: an idle-but-resident neighbor container raises
      board idle from 5.4 W to ~17.5 W, ~12 W continuous before any inference.
- [ ] **INT8-SQ's MMLU gap is REAL but it is 7.8 points, not 12.5** — re-measured
      2026-09-11, and the original figure was a sampling artifact. `validate_mmlu.py`'s
      `--limit` takes `all_rows[:limit]` and MMLU's `all/test` parquet is **ordered by
      subject**, so the n=200 runs every published figure came from are 100
      abstract_algebra + 100 anatomy — 2 of 57 subjects, with one of MMLU's hardest
      subjects at half weight. The same two checkpoints, measured three ways:

      | sample | subjects | FP16 | INT8-SQ | gap |
      |---|---|---|---|---|
      | n=200 first-n (as published) | 2 of 57 | 67.5% | 54.0% | 13.5 pt |
      | n=1000 first-n | 8 of 57 | 74.0% | 61.4% | 12.6 pt |
      | **n=1000 random, seed 1234** | **all 57** | **70.8%** | **63.0%** | **7.8 pt** |

      The direction survives decisively (z=3.7), so "do not ship INT8-SQ without
      closing this" stands. The magnitude does not: the biased slice **overstated the
      damage by about 60%**. The three calibration hypotheses (`--text_dataset
      wikitext`, >512 samples, modelopt's SmoothQuant trouble spots) are all still
      untried, and are now **blocked on tooling, not on time**: `modelopt` and `torch`
      are both absent from the Edge-LLM venv, so `scripts/quantize_edgellm.sh` cannot
      run at all on this box today. Re-creating that environment means installing a
      Jetson-specific torch wheel, which is the wheel-shadowing hazard
      `embedded-ai-chain/docs/environment.md` warns about — a deliberate decision, not
      a chore to slip into another task.
- [ ] **Re-measure any MMLU number that is quoted as a magnitude.** Generalises the
      row above. `--sample random` (with the seed recorded in both the document and the
      `result_id`) was added 2026-09-11; `first-n` remains the default so historical
      results stay reproducible. Every MMLU figure written before that date is a
      2-subject score. For *ranking* a quantization against its same-size sibling —
      the only job `docs/promotion-contract.md` §3 gives MMLU — those runs are still a
      controlled comparison, because every row saw the identical 200 questions. What
      they cannot support is an **effect size** or any general-knowledge reading, since
      the damage is subject-dependent: that is precisely how 7.8 points was published
      as 12.5.
- [x] **`int4_awq` against the bias-recipe patch: VERIFIED 2026-09-11 — it builds,
      and the engine is still unusable.** "Should work (identical return statement)"
      was half right. `Qwen2.5-1.5B-Instruct-AWQ` now compiles and serves (the patch
      does cover this format; the 2026-09-08 "cannot build AWQ" record was taken the
      day before the fix existed). But the engine emits **zero tool calls in 100 BFCL
      cases** — 0/50 `simple`, and a meaningless 100% `irrelevance` that is the
      mute-model false positive, not abstention. It is not numerically broken: **MMLU
      42.0%, identical to the FP16 twin's 42.0%**, and plain completion is correct.
      The FP16 twin calls a tool in **50/50** `simple` cases. Same size, same backend,
      precision the only variable. That makes **two** Edge-LLM AWQ engines that build
      and are then silently wrong in different ways (Qwen3-8B-AWQ emits
      `"0000000..."`), against an `int4_gptq` path validated end-to-end — so the fault
      is specific to Edge-LLM's **AWQ** path, not to int4, and not to the patch.
      `int4_awq_modelopt` remains untested and now needs a deliberate decision rather
      than an assumption: producing such a checkpoint needs a modelopt environment that
      no longer exists on this box (see the INT8-SQ item), for a format whose sibling
      just failed twice.
- [x] **Bielik's tool-calling: ROOT-CAUSED 2026-09-11, and the family was excluded on
      a wrong diagnosis.** Not the bias-recipe bug, and not a backend difference at all.
      `speakleash/Bielik-11B-v3.0-Instruct-awq`'s HF `chat_template` is a **209-character
      bare ChatML loop that never mentions `tools`, `tool_call`, `function_call` or
      `function_list`** — and the GGUF ships the identical template. Every backend
      therefore drops the `tools` parameter before the model ever sees it, which
      explains in ONE cause what was recorded as several: vLLM/Orin returning empty
      `tool_calls` with no tool-call block anywhere in `content` (re-checked by the Orin
      arm), both of its parsers "failing" (there was nothing to parse), and llama.cpp on
      Thor scoring 0/9 across temperatures and 0/6 with `--jinja` removed.
      **The model is tool-capable and was trained for it.** Its tokenizer carries
      dedicated `<|function_call|>`, `<|function_list|>`, `<|function_output|>`,
      `<tool_call>` and `</tool_call>` special tokens, and when served with a tool-aware
      ChatML template (Qwen2.5's, structurally compatible) it emits a correct
      `get_weather({"city":"Warsaw"})` call **6/6**. This contradicts the recorded root
      cause, which read the GGUF README's manual prompt-injection convention as evidence
      of "no native training on any tagged format" — the added-token table says
      otherwise. Consequence: the 2026-09-07 decision excluding this family from the
      scorecard rests on a serving defect this lab can fix by supplying a template, not
      on a model limitation. Revisiting that exclusion is a real decision, not a
      formality. (Apertus is untouched by this and remains blocked on llama.cpp not
      recognizing its GGUF architecture at all.)
- [ ] Identify *what* structural difference actually causes the backend gap (leading
      candidate: chat-template rendering). Cheap decisive test: send one identical
      pre-rendered prompt to all three via `/v1/completions` with no `tools`
      parameter and compare raw output.
