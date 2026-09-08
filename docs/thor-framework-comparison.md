# Thor framework comparison — which serving backend, and which model size?

**Recommendation: TensorRT Edge-LLM serving Qwen2.5-7B-Instruct FP16.**

Two findings drive that, and the second is the one that would be missed by looking
only at headline scores:

1. **The backend determines whether model size buys you anything — or costs you.**
   Going 1.5B → 7B takes BFCL `irrelevance` from 78% to **94%** on Edge-LLM, moves
   llama.cpp only 60% → 62%, and makes vLLM **worse, 68% → 54%**. Same weights, same
   cases; all three score 90-92% on `simple`. llama.cpp's tool-call path is
   grammar-*constrained*, so a more capable model has no way to express "call
   nothing"; vLLM's tag-detection parser appears to misread the richer output of a
   larger model as tool calls. Choose either and extra parameters buy you nothing on
   the axis that matters.
2. **The size curve flattens completely after 7B.** 14B scores *identically* to 7B —
   not approximately, but 98/100 identical per-case verdicts — while costing 2x the
   latency. 14B is wasted memory and wasted time.

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
  Edge-LLM can convert capacity into judgement. vLLM regressing suggests its `hermes`
  parser is over-eager on the richer, more elaborate output a 7B model produces —
  more text that can be mistaken for a tool call. That is a hypothesis about the
  parser, not a measured cause, and pinning `hermes` on Edge-LLM too (see "Still
  open") is what would test it.

The llama.cpp result reproduces the Orin finding (85/65/75 there) on different
hardware, a different image family, and now at a second model size — it is a property
of the backend, not of a build.

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

**What would argue against it:**

- **Operational cost.** Edge-LLM has no wheel and no image; every device needs a
  ~1h source build. llama.cpp is one `docker pull`, starts in 4-6 s unconditionally,
  and is within ~50 ms of Edge-LLM on 7B median turn time — it is simply a much worse
  *judge*, and cannot be improved by a bigger model.
- **Experimental status.** Edge-LLM's OpenAI server is labelled experimental upstream.
- **Cache dependence.** Its fast start assumes a persistent engine cache
  (`/opt/edgellm-cache`, shared group-writable on this box).
- **Parser confound.** The backends did not use identical tool-call parsers
  (Edge-LLM `auto`, vLLM `hermes`, llama.cpp its grammar path). `hermes` exists in
  Edge-LLM's list, so pinning it across both is a cheap follow-up that would separate
  parser from runtime.

## Still open

- [ ] Re-run BFCL with `--tool-call-parser hermes` pinned on both Edge-LLM and vLLM.
- [ ] `jetson_clocks`-locked run to settle the DVFS effect (needs root).
- [ ] Replace the simplified AST checker with official `bfcl-eval` so `simple` stops
      saturating.
- [ ] Cold-start-from-empty-cache as its own measurement for Edge-LLM and vLLM.
- [ ] Co-residency run: orchestrator + VLM tier together, which is the real
      deployment shape and the condition the DVFS finding says matters.
