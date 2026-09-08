# Thor framework comparison — which serving backend for the orchestrator LLM?

**Interim answer: TensorRT Edge-LLM, and it is not close.** It wins the axis this lab
exists to measure — tool-call judgment, 83% BFCL vs vLLM's 75% and llama.cpp's 74% on
identical weights — *and* it leads TTFT (26ms), throughput (65.9 tok/s) and tail
latency (999ms p95). The accuracy lead comes almost entirely from `irrelevance` (78 vs
68 vs 60): knowing when *not* to call a tool, which is `embedded-ai-chain`'s documented
production weakness. The expected "judges better but runs slower" trade-off **did not
materialise**; it judges better and runs faster, for ~15% more board power.

**vLLM — the current production backend — loses on every axis measured here.** That
is the finding with the most direct consequence for `embedded-ai-chain`.

**What would still change the answer**, and why this is not yet a decision:

- Every timing number was taken on a **shared box** and needs an exclusive re-run.
- The BFCL legs did not use identical tool-call parsers (`auto` vs `hermes`), so the
  honest claim today is about Edge-LLM *as configured by default*.
- Edge-LLM's operational cost is real and unmeasured here: **a from-source build per
  device** (~1h, no wheel or image exists), it is the only backend that **cannot serve
  the AWQ checkpoint** the Orin rows use, its OpenAI server is labelled **experimental**
  upstream, and its fast cold start depends on a **persistent engine cache**.
  A fleet that cannot carry that build cost may still prefer llama.cpp, which is
  within 84ms on median latency, starts fast unconditionally, and needs one image.

**Status: accuracy and (shared-box) timing measured. Exclusive-box re-runs pending.**
Started 2026-09-08 on the user's Jetson Thor (JetPack 7.1 / L4T R38.4.0, CUDA 13.0,
TensorRT 10.13.3.9, 122 GiB unified memory, `nvpmodel` 120W mode).

This is the Thor counterpart to the Orin work in `docs/TODO.md` Phase 1. Unlike that
one, it compares **frameworks against each other with the model held fixed**, rather
than models against each other — that is what "which framework is best on Thor"
actually requires.

## The control variable, and why it is FP16 rather than the Orin baseline's AWQ

Every `-orin` row in `configs/models.yaml` uses `Qwen2.5-1.5B-Instruct-AWQ`, so the
obvious choice was to reuse it. **That turned out to be impossible**: TensorRT
Edge-LLM's server cannot build that checkpoint at all (`ValueError: external FP16
bias has no checkpoint recipe` — its direct builder has no recipe for an externalized
attention bias on an `int4_awq` `qwen2` graph; Qwen2-family models carry QKV biases,
Qwen3 dropped them). Details in `configs/models.yaml`'s `1.5b-fp16-edgellm-thor`
block.

So the pinned control is **`Qwen/Qwen2.5-1.5B-Instruct` at FP16** — the one variant
all candidate backends on this box can serve. Consequences to keep in mind:

- These numbers are **not** directly comparable to the Orin AWQ rows. Precision
  differs, and so does hardware. Nothing here answers `paper.md`'s P5 on its own.
- Precision, model, sample, temperature (0.1) and BFCL case set are identical across
  the three legs below, so the legs *are* comparable **to each other**.

## Serving viability — the first result, before any benchmark

Getting each framework to serve at all was itself most of the work, and two of the
five images/paths tried do not work on this board:

| Path | Outcome |
|---|---|
| **TensorRT Edge-LLM** (built from source, 0.10.1) | **Works.** No prebuilt wheel or image exists; built on-device (~1h, CPU-only, no sudo). Thor + JetPack 7.0/7.1 is an *Official* row in its support matrix. |
| **vLLM** `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` | **Works.** Note the image tag differs from this repo's Orin default (`-jetson-orin`). |
| **llama.cpp** `ghcr.io/ggml-org/llama.cpp:server-cuda` | **Fails at inference.** Enumerates the GPU correctly (`CUDA0: NVIDIA Thor (125771 MiB)`) and loads the model, then dies on the first request: `CUDA error: an internal operation failed` in `cublas_handle`. Generic CUDA arm64 builds target sbsa/discrete GPUs, not Tegra. |
| **llama.cpp** `ghcr.io/nvidia-ai-iot/llama_cpp:r38.2.arm64-sbsa-cu130-24.04` | **Fails at startup.** `libcudart.so.12: cannot open shared object file` — this *rolling* tag ships an Apr-2025 binary linked against CUDA 12 inside a CUDA-13 image. |
| **llama.cpp** `ghcr.io/nvidia-ai-iot/llama_cpp:b10373-r38.2.arm64-sbsa-cp312-cu130-24.04` | **Works.** The dated `b*` tag from the same repo. This is the one to use. |
| **llama.cpp** `dustynv/llama_cpp` (the Orin source) | **No Thor tag exists** on Docker Hub — r35.x/r36.x only. jetson-containers publishes r38-era images to GHCR under `nvidia-ai-iot/` instead. |

Three of the five llama.cpp paths tried are dead ends, and the two live ones differ
only by tag within one repo — worth stating plainly for anyone repeating this: on
Thor, prefer `nvidia-ai-iot`'s **dated `b*` tags** over its rolling tags, and do not
expect either the Orin image or upstream's generic CUDA build to work.

**Method note worth keeping:** `--list-devices` reporting `CUDA0: NVIDIA Thor` was
*not* evidence of a working backend. The model loaded and the server came up; it died
only inside cuBLAS on the first real inference. Device enumeration is not a smoke test.

## Measured — accuracy axes (shared box; co-residency does not perturb these)

Both suites ran with another user's production `vllm-vlm-thor` container resident.
That is fine here: correctness is unaffected by contention, only latency is.

| Framework | BFCL simple | BFCL irrelevance | **BFCL overall** | MMLU |
|---|---|---|---|---|
| **TensorRT Edge-LLM** | 88% | **78%** | **83%** | 42.0% |
| **vLLM** | 82% | 68% | 75% | 40.5% |
| **llama.cpp** | 88% | **60%** | 74% | 38.5% |

n=100 for BFCL (50 simple + 50 irrelevance), n=200 for MMLU, `temperature=0.1`,
`tool_choice="auto"` throughout — unforced judgment, not `orchestrator.py`'s forced
`tool_choice` workaround. Zero unparsed MMLU responses on all three.

### Reading the table

**The MMLU column is the control, and it does its job.** 38.5 / 40.5 / 42.0 is a
3.5-point spread at n=200 — within noise, with no unparsed responses anywhere. All
three frameworks load the same weights and apply the same chat template correctly.
So the BFCL spread is **not** an artifact of a broken setup on any leg; it is a real
difference in tool-call handling.

**The whole BFCL story is in the `irrelevance` column.** On `simple` — a tool call is
wanted, produce it — llama.cpp and Edge-LLM tie at 88% and vLLM trails at 82%. On
`irrelevance` — the right answer is to call *nothing* — the three separate sharply:
78% / 68% / **60%**. Edge-LLM's 8-point overall lead over vLLM, and its 9-point lead
over llama.cpp, is almost entirely earned by knowing when not to call a tool.

That is exactly the axis this lab was built to measure. `embedded-ai-chain`'s
documented orchestrator weakness is over-escalation (the "weakest tool-calling
judgment … documented `ask_vlm` escalation gap"), and a backend that scores well on
`simple` while failing `irrelevance` is precisely a backend that will over-escalate in
production.

**llama.cpp's profile is consistent with the Orin result and has a known mechanism.**
Orin's llama.cpp run on the same model family scored 85 simple / 65 irrelevance / 75
overall — the same shape as Thor's 88 / 60 / 74. `configs/models.yaml` already records
why: llama.cpp's tool-call path is grammar-*constrained*, forcing a structurally valid
call, where vLLM's and Edge-LLM's are tag-*detecting*. Constraint helps when a call is
wanted and hurts when abstaining is correct. Two platforms, two image families, same
signature — this is a property of the backend, not of a particular build.

**Confound to name honestly:** the legs did not use identical parsers — Edge-LLM ran
`--tool-call-parser auto` (model-native detection), vLLM ran `hermes` (this repo's
compiled-in default, unchanged from the Orin rows), llama.cpp its own grammar path.
`hermes` exists in Edge-LLM's choice list too, so pinning it on both is a cheap
follow-up that would separate parser from runtime. Until that runs, the honest claim
is "Edge-LLM **as configured by default** judges tool calls best on Thor", not
"Edge-LLM's runtime is inherently better".

## Measured — timing axes (SHARED BOX, indicative only)

**These are not the numbers of record.** They were taken with the other user's
production container resident (idle, but holding ~37GB), so they must be re-run on an
exclusive box before anything is decided on them. They are here because the ranking
they show is stable and large enough to be useful now. Raw JSON: `output/*_sharedbox.json`.

| | TTFT p50 | tok/s p50 | latency p50 | **latency p95** | cold start | power |
|---|---|---|---|---|---|---|
| **Edge-LLM** | **26 ms** | **65.9** | 996 ms | **999 ms** | 4.1 s ¹ | 17.1 W |
| **llama.cpp** | 32 ms | 55.2 | **912 ms** | 1383 ms | **4.0 s** | 15.0 W |
| **vLLM** | 47 ms | 42.8 | 1423 ms | 1894 ms | 96.1 s ¹ | 14.7 W |

n=20 (streaming), n=30 (latency), 128-token responses, all three warmed to thermal
steady state (`warmup_reached_steady_state: true`). Power is `vin_sys_5v0` board draw
in the 120W nvpmodel mode.

¹ **Cold start is bimodal for two of the three and the table shows only the warm
number.** Edge-LLM's 4.1s is an engine-cache HIT; a miss compiles TensorRT engines
for minutes. vLLM's 96.1s is with a warm torch.compile/inductor cache; its true first
start was minutes as well. Only llama.cpp's 4.0s is unconditional. **If a deployment
cannot guarantee cache persistence across restarts, this ranking changes** — that is
an operational property, not a benchmark artifact, and it deserves a decision of its
own rather than a footnote.

### Reading the timing table

- **Edge-LLM leads on TTFT (1.8x vLLM) and throughput (1.5x vLLM)** and, more
  interestingly, on the **tail**: a 999ms p95 against a 996ms p50 is a 3ms spread.
  llama.cpp's median is better (912ms) but its p95 is 1383ms. For an orchestrator in
  a conversational turn-taking loop, tail consistency is worth more than an 84ms
  median edge — it is the difference between predictable turns and occasional stalls.
- **vLLM loses this comparison on every timing axis**, which is worth stating plainly
  because it is `embedded-ai-chain`'s current production backend. Its 96s warm cold
  start is 24x llama.cpp's.
- **Power is nearly a wash** (14.7-17.1W). Edge-LLM draws ~15% more than the other
  two. Nothing here is a power/performance trade; Edge-LLM is simply doing more work
  per second.
- llama.cpp's first latency run reported `warmup_reached_steady_state: false` and a
  p50 of 987ms/p95 1971ms. Re-run with `--warmup-max 25` it reached steady state and
  improved to 912/1383. **The harness's steady-state flag caught a real measurement
  error** — the first numbers would have understated llama.cpp on median and
  overstated its tail. Do not report a run where that flag is false.

## Still not measured

- [ ] All of the above, on an **exclusive box** — these are the numbers of record.
- [ ] Memory-fit / `gpu_memory_utilization` tuning for the vLLM leg.
- [ ] Cold-start-from-empty-cache for Edge-LLM and vLLM, as its own measurement.

## Repo changes this work required

- `EdgeLlmCoordinator` in `src/llm_coordinator.py` — owns a local
  `tensorrt-edgellm-serve` process rather than a container (Edge-LLM ships no image),
  kills by process *group* (uvicorn's workers otherwise keep the port and the GPU
  allocation), and prepends `/usr/local/cuda/bin` to `PATH` because the engine builder
  needs `nvcc` and a login shell on this box does not have it. `build_coordinator`
  dispatches `backend: edge-llm`.
- `LlamaCppCoordinator` gained `server_argv0`, because the two llama.cpp image
  families differ on whether they set an `ENTRYPOINT`.
- Four `-thor` vLLM rows, one `-thor` edge-llm row, one `-thor` llama-cpp row, all on
  ports that avoid the box's production container (8000) — 8001 vLLM, 8002 Edge-LLM,
  8090 llama.cpp.
- 9 new unit tests (61 total, all passing, no GPU needed).
