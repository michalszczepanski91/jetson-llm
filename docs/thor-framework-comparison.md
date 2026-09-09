# Thor framework comparison — which serving backend, and which model size?

**Recommendation: TensorRT Edge-LLM serving Qwen2.5-7B-Instruct FP16.**

The decision rests on one axis where the backends differ by 40 points and every other
axis is close. If tool-call judgement did *not* matter, llama.cpp would win — it is
the fastest and the most energy-efficient of the three, and needs one `docker pull`
rather than a per-device source build. It loses because it cannot abstain.

| 7B, the three axes that matter | Edge-LLM | llama.cpp | vLLM |
|---|---|---|---|
| **Escalation judgement** (BFCL `irrelevance`) | **94%** | 62% | 54% |
| **Real turn latency** (32 tokens) | 1945 ms | **1899 ms** | 2539 ms |
| **Energy** (marginal J/token) | 0.85 J | **0.73 J** | 0.96 J |

Edge-LLM costs ~2% more turn latency and ~16% more energy than llama.cpp, and buys
**+32 points of escalation accuracy** for it. vLLM — what `embedded-ai-chain` ships
today — is worst on all three.

Three findings drive that, and the third would be missed by looking only at headline
scores:

1. **The backend determines whether model size buys you anything — or costs you.**
   Going 1.5B → 7B takes BFCL `irrelevance` from 78% to **94%** on Edge-LLM, moves
   llama.cpp only 60% → 62%, and makes vLLM **worse, 68% → 54%**. Same weights, same
   cases; all three score 90-92% on `simple`. llama.cpp's tool-call path is
   grammar-*constrained*, so a more capable model has no way to express "call
   nothing"; vLLM's tool-call handling appears to misread the richer output of a
   larger model as tool calls. **Not a configuration artifact:** with the *same*
   parser (`hermes`) pinned on both, Edge-LLM and vLLM score an identical 92% on
   `simple` and 94% vs 54% on `irrelevance` — the gap is entirely in abstention.
   Choose llama.cpp or vLLM and extra parameters buy you nothing on the axis that
   matters.
2. **The size curve flattens completely after 7B.** 14B scores *identically* to 7B —
   not approximately, but 98/100 identical per-case verdicts — while costing 2x the
   latency and 2.1x the energy per token. 14B is wasted memory, time and power.
3. **It is not a decoding artifact.** Forcing deterministic argmax (`top_k=1`) on all
   three reproduced every score exactly, so sampling, temperature and truncation
   explain none of the gap — see "Decoding is NOT the cause" below. This mattered to
   check: the campaign originally pinned only `temperature`, and the three backends
   turned out to apply three different truncation defaults.

`irrelevance` is the category where the correct action is to call **no** tool. It is
the direct analogue of `embedded-ai-chain`'s documented `ask_vlm` over-escalation
gap, and it is the only accuracy metric here that still discriminates (see
"Measurement ceiling" below).

**vLLM — what `embedded-ai-chain` ships today — loses on every axis measured.**

Measured 2026-09-08 on the user's Jetson Thor (JetPack 7.1 / L4T R38.4.0, CUDA 13.0,
TensorRT 10.13.3.9, 122 GiB unified memory, `nvpmodel` 120W). Raw JSON in `output/`,
collated by `scripts/collate_thor_results.py`, campaign driver in
`scripts/thor_exclusive_window.sh`.

---

## The numbers of record (exclusive box)

Taken with the box's other user's container stopped and **no other GPU compute
process running** — the campaign script hard-aborts rather than measure otherwise.
Every latency figure below reached thermal steady state
(`warmup_reached_steady_state: true`); any that had not would be marked and re-run.

### Framework, model fixed at Qwen2.5-1.5B-Instruct FP16

| | BFCL overall | simple | **irrelevance** | MMLU | TTFT p50 | tok/s | lat p50 | **lat p95** | cold start | power |
|---|---|---|---|---|---|---|---|---|---|---|
| **Edge-LLM** | **83%** | 88% | **78%** | 42.0% | 34 ms | 48.0 | 1314 ms | **1331 ms** | 4.1 s ¹ | 15.8 W |
| **llama.cpp** | 74% | 88% | 60% | 38.5% | **31 ms** | **54.4** | **972 ms** | 1502 ms | **4.0 s** | 15.1 W |
| **vLLM** | 75% | 82% | 68% | 40.5% | 46 ms | 43.6 | 1338 ms | 1702 ms | 98.1 s ¹ | **14.9 W** |

### Framework at Qwen2.5-7B-Instruct FP16

| | BFCL overall | simple | **irrelevance** | MMLU | TTFT p50 | tok/s | lat p50 | **lat p95** | 32-tok turn | cold start | power |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Edge-LLM** | **93%** | 92% | **94%** | 67.5% | 74 ms | 16.6 | **2706 ms** | **2713 ms** | **1945 ms** | 10.1 s ¹ | 19.5 W |
| **llama.cpp** | 76% | 90% | 62% | 66.5% | **72 ms** | **17.0** | 3254 ms | 4556 ms | 1899 ms | **6.1 s** | 17.8 W |
| **vLLM** | 73% | 92% | **54%** | 68.5% | 144 ms | 11.3 | 4451 ms | 5950 ms | 2539 ms | 214.1 s ¹ | 16.3 W |

### Size sweep, backend fixed at Edge-LLM

| | BFCL overall | **irrelevance** | MMLU | TTFT p50 | tok/s | **32-tok turn** | cold start ¹ | power |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-**1.5B** | 83% | 78% | 42.0% | **34 ms** | **48.0** | **697 ms** | **4.1 s** | **15.8 W** |
| **Qwen2.5-7B** | **93%** | **94%** | **67.5%** | 74 ms | 16.6 | 1945 ms | 10.1 s | 19.5 W |
| Qwen2.5-**14B** | **93%** | **94%** | **67.5%** | 133 ms | 8.4 | 3791 ms | 20.1 s | 20.1 W |

¹ **Cold start is bimodal for Edge-LLM and vLLM and these are the warm numbers.**
Edge-LLM's figures are engine-cache HITS; a miss compiles TensorRT engines for
minutes. vLLM's are with a warm torch.compile cache. Only llama.cpp's is
unconditional. If a deployment cannot guarantee cache persistence across restarts,
this ranking changes — an operational property, not a benchmark artifact.

² vLLM's 7B BFCL first died on the memory-profiling assert described below; the row
above is the re-run after mounting the patched `gpu_worker.py`.

---

## What the numbers mean

### `irrelevance` is the whole story, and it is backend-bound

On `simple` (a tool IS wanted) all three backends land within a few points at both
sizes. On `irrelevance` they separate sharply, and only Edge-LLM converts extra model
capacity into better judgement:

| irrelevance | 1.5B | 7B | change from 4.7x the parameters |
|---|---|---|---|
| **Edge-LLM** | 78% | **94%** | **+16 pts** |
| llama.cpp | 60% | 62% | +2 pts |
| **vLLM** | 68% | **54%** | **−14 pts** |

**vLLM gets worse at abstaining as the model grows.** The backend does not merely gate
the benefit of scale — it can invert it. All three serve identical weights and all
three score 90-92% on `simple`, so this is not a capability difference in the model;
it is what each backend does with the model's output.

Mechanism, in the two directions:

- **llama.cpp** constrains generation to a tool-call grammar. A grammar that forces a
  structurally valid call leaves no room to abstain, so extra capability cannot
  express itself as restraint. Its `irrelevance` is pinned near 60% at both sizes.
- **vLLM and Edge-LLM** *detect* tool tags in free generation instead, which is why
  Edge-LLM can convert capacity into judgement.

**The parser confound is settled, 2026-09-09 — it was not the parser.** Edge-LLM was
re-run with `--tool-call-parser hermes` pinned (the same parser vLLM used), at both
sizes:

| Edge-LLM | parser `auto` | parser `hermes` | per-case agreement |
|---|---|---|---|
| 7B | 92 / 94 / 93% | 92 / 94 / 93% | **100/100 identical** |
| 1.5B | 88 / 78 / 83% | 88 / 78 / 83% | **100/100 identical** |

Not merely equal aggregates — every individual case decided the same way. So the
head-to-head can be stated with no configuration difference left standing:

| 7B, parser `hermes` on both | simple | **irrelevance** | overall |
|---|---|---|---|
| **Edge-LLM** | 92% | **94%** | **93%** |
| **vLLM** | 92% | **54%** | 73% |

`simple` is *identical* at 92%: both backends recognise a wanted tool call equally
well. The whole 40-point gap is **abstention** — vLLM emits tool calls when the
correct action is to emit none, and does so more as the model grows. That is a
property of vLLM's tool-call handling, not of its parser choice, not of the weights,
and not of this lab's configuration of it.

The llama.cpp result reproduces the Orin finding (85/65/75 there) on different
hardware, a different image family, and now at a second model size — it is a property
of the backend, not of a build.

### Decoding is NOT the cause — settled by a greedy control, 2026-09-09

The comparison above pinned `temperature` and treated sampling as controlled. **It was
not**, and that was a real flaw: each server applies its own truncation defaults, and
all three differ.

| | `top_k` | `top_p` | `min_p` |
|---|---|---|---|
| Edge-LLM | 50 | 0.90 | (not supported) |
| vLLM | −1 (off) | 1.00 | 0 |
| llama.cpp | 40 | 0.95 | **0.05** |

Two experiments removed that confound. First, vLLM re-run with Edge-LLM's exact
truncation (`top_p=0.9, top_k=50`): `irrelevance` moved 54% → 58%, i.e. 4 points of a
40-point gap. Then the decisive one — **`top_k=1` on all three, which forces
deterministic argmax and makes `top_p`/`min_p`/temperature inert**
(`scripts/greedy_framework_control.sh`):

| 7B, greedy `top_k=1` | simple | **irrelevance** | overall | vs its own default sampling |
|---|---|---|---|---|
| **Edge-LLM** | 92% | **94%** | 93% | **identical** |
| **vLLM** | 92% | **54%** | 73% | **identical** |
| **llama.cpp** | 90% | **62%** | 76% | **identical** |

Every backend reproduced its own score exactly. **Sampling explains none of the gap.**
With decoding provably identical and deterministic, and the weights identical, the
remaining explanations are structural: what prompt each backend actually renders from
the tool definitions, llama.cpp's grammar-constrained emission, and how each decides a
generation counts as a tool call. Identifying which of those dominates is the obvious
next investigation — the candidate this campaign could not rule out is chat-template
rendering, since Edge-LLM, vLLM and llama.cpp each build the tools prompt themselves.

### Energy per token

Board power is nearly flat across backends (14.9-19.5 W, a 1.3x spread) while
throughput varies 6.5x, so **energy per token is dominated by speed, not draw**.
Marginal figures subtract a measured idle baseline of **5.42 W** (box fully quiet, no
containers, GPU idle) to isolate what inference actually costs:

| 7B | board | marginal | tok/s | marginal J/token | marginal J per 32-tok turn |
|---|---|---|---|---|---|
| **llama.cpp** | 17.8 W | 12.4 W | **17.0** | **0.73 J** | **23.3 J** |
| Edge-LLM | 19.5 W | 14.0 W | 16.6 | 0.85 J | 27.1 J |
| vLLM | 16.3 W | 10.9 W | 11.3 | 0.96 J | 30.8 J |

| 1.5B | marginal J/token | | 14B | marginal J/token |
|---|---|---|---|---|
| llama.cpp | **0.18 J** | | Edge-LLM 14B | 1.75 J |
| Edge-LLM | 0.22 J | | | |
| vLLM | 0.22 J | | | |

llama.cpp is the most energy-efficient at every size — 14% better than Edge-LLM at 7B,
32% better than vLLM. vLLM draws the *least* power yet costs the *most* per token,
because it is slowest. Going 7B → 14B costs 2.1x the energy per token for zero
accuracy gain.

**Worth recording separately:** the other user's idle-but-resident container raised
board idle from 5.4 W to ~17.5 W — roughly 12 W to hold a model in memory doing
nothing. On a power-budgeted board that is a real co-residency cost, and it is the
kind of thing an "orchestrator + VLM tier together" deployment pays continuously.

### Quantized 7B on Edge-LLM: still not possible, for two different reasons

The obvious "best of both worlds" candidate - 7B's 94% `irrelevance` at closer to
1.5B's latency/energy - would be a quantized 7B on Edge-LLM. Tried three real
checkpoints, 2026-09-09, none reached a running server:

| Checkpoint | quant path | Result |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` (Qwen's own, ungated) | `int4_gptq` | **Same bias bug as AWQ**, identical stack trace, same line. Proves the failure is not AWQ-specific - it's any externalized int4 quant hitting Qwen2's attention bias in `int4_linear()`'s bias handling. |
| `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` (llmcompressor, ungated) | `fp8` | **Different failure, and inconclusive**: rejected at config-parsing time - `unsupported compressed-tensors checkpoint format: float-quantized` - before any layer is built. Edge-LLM's `compressed-tensors` parser only accepts that quant_method when its format string contains `nvfp4`; llmcompressor's plain FP8-dynamic format isn't one of the formats it recognizes. Says nothing about whether `fp8_linear`'s bias handling (which, unlike `int4_linear`, calls `_add_bias` without even attempting to pass a recipe - if anything a worse sign) would have worked. |
| NVIDIA ModelOpt FP8/NVFP4 (the code path this parser is actually built for) | — | **Not attempted** - no official/trustworthy ModelOpt-quantized checkpoint exists for base `Qwen2.5-7B-Instruct` as of 2026-09-09, only its VL sibling (`nvidia/Qwen2.5-VL-7B-Instruct-{FP8,NVFP4}`). Unofficial community NVFP4 quants exist but don't meet this lab's provenance bar. |

**Conclusion: FP16 is not a choice, it is what remains.** Every quantized path tried
either hit a real bug or a checkpoint-availability wall; this is not "quantization
wasn't tried" but "quantization was tried and specifically blocked" for int4, and
genuinely untested (not assumed safe or unsafe) for FP8/NVFP4. Revisit if upstream
fixes the bias-recipe wiring for `int4_linear`, or if NVIDIA publishes a ModelOpt
checkpoint for the base text model.

### 14B buys nothing

7B and 14B agree on **98 of 100** BFCL cases. They differ on exactly two
(`simple_13`, `simple_42`) and those offset, producing byte-identical aggregate
scores (46/50 simple, 47/50 irrelevance, 135/200 MMLU). Meanwhile 14B doubles TTFT
(74 → 133 ms), halves throughput (16.6 → 8.4 tok/s), and doubles the real turn cost
(1945 → 3791 ms).

Thor's 122 GiB makes 14B *possible* — the Orin's ~30 GB never could — but this
measurement says the memory is better spent elsewhere: co-residency with the VLM
tier, longer context, or a bigger KV cache.

### Measurement ceiling: BFCL `simple` cannot score capable models

Both 7B and 14B fail the same four `simple` cases — `simple_13/14/15/16`, consecutive
maths problems — and in every one **the model called exactly the right function**
(`calculate_derivative`, `integrate`, `calculus.derivative`, …). They fail on argument
matching: ground truth accepts several string forms of a maths expression
(`"x**2"`, `"lambda x: x**2"`, `"y=x**2"`) and this repo's *simplified* AST checker —
which `scripts/validate_tool_calling.py`'s own docstring warns is not the official
`bfcl-eval` checker — does not match the model's formatting variant.

So `simple` has a systematic ~8-point false-negative floor, and every capable model
sits on it. **`irrelevance` is the only accuracy axis here that still discriminates**,
which is fortunate, because it is also the one that maps to the production failure.
Anyone quoting `simple` across models should use the official checker instead.

### The idle box is not the faster box (for Edge-LLM)

Edge-LLM ran measurably *slower* alone than it did next to a busy neighbour, while the
other two barely moved:

| Edge-LLM 1.5B | exclusive box | shared box | change |
|---|---|---|---|
| TTFT p50 | 34 ms | 26 ms | **+31%** |
| tok/s p50 | 48.0 | 65.9 | **−27%** |
| 32-tok turn | 697 ms | 509 ms | **+37%** |
| power | 15.8 W | 17.1 W | **−8%** |

Slower *and* drawing less power points at GPU clocks: Thor's frequency scaling is
dynamic (`jetson_clocks_locked: null` in every result), so the other user's
crash-looping container was inadvertently holding the GPU in a boosted state that
Edge-LLM — evidently the most clock-sensitive of the three — was riding. vLLM and
llama.cpp keep the GPU busy enough per token to hold clocks themselves.

**Consequence for interpretation:** a quiet box is the *reproducible* environment, not
necessarily the *representative* one. In production this orchestrator co-resides with
the VLM tier, so the GPU will be loaded — closer to the shared condition. The
disambiguating experiment is a `jetson_clocks`-locked run, which needs root and has
not been done.

---

## Serving viability — most of the work was getting here

Five of the eight backend/image paths tried do not work on this board:

| Path | Outcome |
|---|---|
| **Edge-LLM**, built from source (0.10.1) | **Works.** No wheel, no image — an on-device CMake build (~1h, CPU-only, no sudo). Thor + JetPack 7.0/7.1 is an *Official* row in its support matrix. |
| **vLLM** `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` | **Works**, with the patch below. Note the tag differs from this repo's Orin default (`-jetson-orin`). |
| **llama.cpp** `ghcr.io/nvidia-ai-iot/llama_cpp:b10373-r38.2...` | **Works.** The dated `b*` tag from NVIDIA's GHCR. |
| llama.cpp `dustynv/llama_cpp` (the Orin source) | **No Thor tag exists** on Docker Hub — r35/r36 only. jetson-containers publishes r38-era images to GHCR under `nvidia-ai-iot/`. |
| llama.cpp `ghcr.io/ggml-org/llama.cpp:server-cuda` | **Fails at inference.** Enumerates the GPU (`CUDA0: NVIDIA Thor`) and loads the model, then dies on the first request in `cublas_handle`. Generic CUDA arm64 targets sbsa/discrete, not Tegra. |
| llama.cpp `nvidia-ai-iot/llama_cpp:r38.2...` (rolling tag) | **Fails at startup** — `libcudart.so.12: cannot open shared object file`. An Apr-2025 CUDA-12 binary inside a CUDA-13 image. |
| Edge-LLM + `Qwen2.5-*-Instruct-AWQ` | **Cannot build.** `external FP16 bias has no checkpoint recipe` — its direct builder has no recipe for an externalized attention bias on an `int4_awq` `qwen2` graph. Qwen2 carries QKV biases; Qwen3 dropped them. **This is why every Thor row is FP16.** |
| Apertus-8B (Orin, for reference) | Blocked — llama.cpp does not recognise its GGUF architecture. |

**Two method notes worth keeping:**

- `--list-devices` reporting `CUDA0: NVIDIA Thor` is **not** evidence of a working
  backend. The ggml-org image enumerated the GPU, loaded the model, started the
  server, and only died on the first real inference. Device enumeration is not a smoke
  test.
- **vLLM asserts that free memory never increases during startup profiling** and kills
  the engine when it does — observed here at 7B on a *completely idle* box
  (`Initial free memory 52.65 GiB, current free memory 52.83 GiB`). More free memory
  than expected is harmless; the assert cannot tell. `embedded-ai-chain` hit this on
  this board in August and ships a patched `gpu_worker.py`; the Thor vLLM rows now
  mount it via `extra_volumes`. It affects startup profiling only and touches no
  inference path, so it cannot flatter a performance number. **The earlier
  shared-box failure blamed on "contention" was this same assert** — the box being
  busy was incidental.

---

## The decision, stated with its costs

**Edge-LLM + Qwen2.5-7B FP16** gives 93% BFCL overall and 94% `irrelevance` — a
16-point improvement on exactly the failure mode `embedded-ai-chain` documents — at
**1945 ms per orchestrator-shaped turn** (32 generated tokens) versus 697 ms for
1.5B. TTFT stays at 74 ms, so with streaming the user hears speech begin almost
immediately; what grows is turn completion.

**The trade is +1.25 s per turn for +16 points of escalation accuracy.** Whether that
fits is a product judgement about the pipeline's turn budget, not something this
benchmark settles.

**What would argue against it** (the parser confound is no longer one of these —
see above):

- **Operational cost.** Edge-LLM has no wheel and no image; every device needs a
  ~1h source build. llama.cpp is one `docker pull`, starts in 4-6 s unconditionally,
  and is within ~50 ms of Edge-LLM on 7B median turn time — it is simply a much worse
  *judge*, and cannot be improved by a bigger model.
- **Experimental status.** Edge-LLM's OpenAI server is labelled experimental upstream.
- **Cache dependence.** Its fast start assumes a persistent engine cache
  (`/opt/edgellm-cache`, shared group-writable on this box).
- ~~**Parser confound.**~~ **Settled 2026-09-09 and it was not the parser** —
  Edge-LLM scores identically with `auto` and `hermes` (100/100 identical per-case
  verdicts at both sizes), so the 40-point `irrelevance` gap against vLLM stands with
  the same parser pinned on both. See "irrelevance is the whole story" above.

## Still open

- [x] ~~Re-run BFCL with `--tool-call-parser hermes` pinned on both Edge-LLM and
      vLLM.~~ Done 2026-09-09: no effect on Edge-LLM whatsoever (100/100 identical
      per-case verdicts at 1.5B and 7B), so the backend gap is real and not a
      configuration artifact.
- [ ] `jetson_clocks`-locked run to settle the DVFS effect (needs root).
- [ ] Replace the simplified AST checker with official `bfcl-eval` so `simple` stops
      saturating.
- [ ] Cold-start-from-empty-cache as its own measurement for Edge-LLM and vLLM.
- [ ] Co-residency run: orchestrator + VLM tier together, which is the real
      deployment shape and the condition the DVFS finding says matters. Now with a
      measured price tag: the other user's idle-but-resident container raised board
      idle from 5.4 W to ~17.5 W, so co-residency costs ~12 W continuously before any
      inference happens.
