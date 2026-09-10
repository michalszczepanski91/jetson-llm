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

| precision | BFCL `irrelevance` | 32-token turn | marginal J/token | cold start |
|---|---|---|---|---|
| **INT4-GPTQ** | 90% | **769 ms** | **0.274 J** | **0.03 s** ¹ |
| **FP16** | **94%** | 1945 ms | 0.839 J | 12.1 s ¹ |
| FP8 (self-quant) | 90% | 1052 ms | 0.603 J | — |
| INT8-SQ (self-quant) | 92% ² | 1074 ms | 0.610 J | — |

¹ Cold start is a cache **hit** for both — a miss (first run, or an empty cache)
takes minutes for either precision. Only llama.cpp's cold start is unconditionally
fast; see the framework table below. ² INT8-SQ still has an unexplained 12.5-point
MMLU gap (54.0% vs FP16's 67.5%) — not yet trusted for production; see "Still open."

INT4-GPTQ wins every timing and energy column outright against all three other
precisions, for 4 points of `irrelevance`. FP8 is essentially lossless on accuracy but
only ~1.4× FP16's throughput, nowhere near INT4's 2.5×. Both are legitimate; which to
ship is a genuine product call about the pipeline's turn budget.

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
  With weights, parser, and decoding all controlled, the gap is structural — most
  likely in how each backend renders the tools prompt or decides a generation counts
  as a tool call. Not yet isolated further.

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
- [ ] Replace the simplified BFCL AST checker with the official `bfcl-eval` so
      `simple` stops saturating.
- [ ] Cold-start-from-empty-cache as its own measurement for Edge-LLM and vLLM.
- [ ] Co-residency run: orchestrator + VLM tier together — the real deployment
      shape. Known price tag so far: an idle-but-resident neighbor container raises
      board idle from 5.4 W to ~17.5 W, ~12 W continuous before any inference.
- [ ] **INT8-SQ's 12.5-point MMLU gap is still unexplained** after the calibration
      fix. Try `--text_dataset wikitext`, more than 512 samples, or check against
      modelopt's documented SmoothQuant trouble spots (attention output /
      embedding layers) before treating this precision as usable.
- [ ] `int4_awq`/`int4_awq_modelopt` against the same bias-recipe patch — should
      work (identical return statement) but not re-verified per format.
- [ ] Whether the same bias-recipe bug was silently blocking int4 for Apertus/Bielik
      on this repo's Orin rows — worth checking before assuming it's Qwen2-specific.
- [ ] Identify *what* structural difference actually causes the backend gap (leading
      candidate: chat-template rendering). Cheap decisive test: send one identical
      pre-rendered prompt to all three via `/v1/completions` with no `tools`
      parameter and compare raw output.
