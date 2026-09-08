# Thor framework comparison — which serving backend for the orchestrator LLM?

**Interim answer: TensorRT Edge-LLM, on tool-call judgment — the axis this lab
exists to measure.** 83% BFCL overall vs vLLM's 75% and llama.cpp's 74%, on identical
weights, with an MMLU control confirming all three load the model correctly. The lead
comes almost entirely from `irrelevance` (78% vs 68% vs 60%) — knowing when *not* to
call a tool, which is `embedded-ai-chain`'s documented production weakness.

**This is not yet a decision.** No latency, throughput, cold-start or power number has
been taken (those need an exclusive box), and Edge-LLM carries real operational costs
the other two do not: it must be built from source per device (~1h), it is the only
backend that cannot serve the AWQ checkpoint the Orin rows use, and its server is
labelled *experimental* upstream. A framework that judges 8 points better but starts
10x slower or cannot be reproduced on a fleet machine may still lose.

**Status: partial — accuracy axes measured, timing/power axes NOT yet run.**
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

## NOT yet measured — everything timing-shaped

These need an **exclusive box** (the other user's containers stopped) or the numbers
are not reproducible, per the discipline in `docs/TODO.md` Phase 6:

- [ ] `benchmark.py` — wall latency, cold start, thermal, power, for each leg
- [ ] `benchmark_streaming.py` — TTFT and tokens/sec, for each leg
- [ ] Memory-fit / `gpu_memory_utilization` tuning for the vLLM leg

Two things already observed that these runs must pin down properly:

- **vLLM's first cold start on Thor is minutes, not seconds** (torch.compile/inductor);
  a second start with a warm compile cache took **27.8s** to init the engine. Cold
  start is therefore bimodal for this backend and must be reported as two numbers,
  not averaged into one.
- **Edge-LLM's cold start is bimodal for a different reason**: an engine-cache miss
  compiles TensorRT engines (minutes); a hit only loads them (seconds). `cache_dir`
  state is part of the measurement.

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
