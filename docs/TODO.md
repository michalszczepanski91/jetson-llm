# TODO — orchestrator-LLM selection lab

Action list for turning this repo into a working multi-candidate orchestrator-LLM
selection lab (`docs/promotion-contract.md`) **and**, since 2026-09-04, a reproducible
Jetson LLM benchmark suite (`docs/note.md`) of which that selection is one subset.
Phase-gated: don't start a phase until the one before it is checked off.

**This file is a status tracker, not a lab notebook.** Each phase below is objective +
checklist + key results + gate. Full incident narratives (root causes, exact
diagnostics, every number tried) live in **`docs/HISTORY.md`** — follow the "full
record" link on any phase that has one. Rewritten to this shape 2026-09-07; the prior
~980-line version's content is unchanged, just split between this file and HISTORY.md
(see that file's own header, or git history, for the original).

## Status at a glance

**Phases 7 and 8 were answered on Thor, not Orin** — it is the only board that can run
all three backends, so the quantization and backend studies happened there first and
the phase numbering (written when Orin was the only platform) understates them. One
status column can't carry two boards, so the table splits them.

| Phase | Orin | Thor | Key result |
|---|---|---|---|
| 0 — Framework survey | ✅ | ✅ | vLLM + llama.cpp in scope; TensorRT excluded on Orin (JetPack 6.2.x), **superseded on Thor** |
| 1 — Foundations | ✅ | ✅ | Qwen2.5 works on every backend; Bielik llama.cpp-only; Apertus blocked entirely |
| 2 — Measurement integrity | ✅ | 🚧 | Raw runs, energy, manifest, IDs, co-residency guard — 3 real bugs found. Thor's campaign bypassed this pipeline; **one** Thor row now goes through it properly — see Phase 6 |
| 3 — Experiment configuration | ✅ | ⚠️ | Config-driven campaigns; guard hardened after 2 more incidents. Thor ran from shell scripts, not `configs/benchmarks/` |
| 4 — Canonical campaign | 🚧 partial | 🚧 | Orin: J/token amortization + a backend context-scaling gap. Thor: one matched point run (P5 answered — Thor buys capacity, not speed), the two sweeps still not reproduced |
| 5 — Full scorecard matrix | ✅ | ✅ | 7 Orin rows fully scored; `irrelevance` is the only accuracy axis that still discriminates |
| 6 — Cross-platform arm | n/a | ✅ | Thor bring-up, three backends, exclusive-box campaign |
| 7 — Quantization study | ⏳ | ✅ | FP16 / INT4-GPTQ / FP8 / INT8-SQ, one model + backend, MMLU-controlled |
| 8 — Backend study | ⏳ | ✅ | Three backends, model **and** precision held fixed, parser + decoding controlled |
| 9 — Co-resident / concurrency | ⏳ | ⏳ | Highest remaining value — feeds both papers |
| 10 — Analysis pipeline | ⏳ | ⏳ | — |
| 11 — Declare a winner | ✅ deferred | 🚧 | Orin: "no clean winner" + tie-breaker recorded (`docs/promotion-decision.md`). Thor: a recommendation, which is not the same as a promotion |

Full reconciliation of this plan with `docs/note.md`'s original benchmark-suite
proposal: `docs/HISTORY.md`, "Scope change, 2026-09-04".

## Not tested yet — and whether there's a reason

Kept here rather than scattered across phases, because "what's missing" is the
question this file gets asked most and it should not take eleven sections to answer.

- **Only Qwen2.5 has been tested on Thor.** Not a blocker — a deliberate scope choice
  that then never got revisited. The framework comparison held the model fixed
  *because* that is what made it controlled; varying family is a different experiment,
  and it simply hasn't been run. Everything below is execution time, not new design.
- **Bielik-11B on Thor**: a row exists (`bielik-11b-awq-vllm-thor`) and has **never
  been smoke-tested**. It was written to answer a real open question — Bielik's
  tool-calling is confirmed broken on vLLM/Orin across two parsers but works on
  llama.cpp, and nobody knows whether that is backend-wide or Orin-specific. Note the
  existing row is the *vLLM* one, i.e. the variant already known to fail on Orin; the
  *working* Bielik path (llama.cpp) has no Thor row at all, and neither does Edge-LLM.
- **Qwen3 on Thor: tested 2026-09-10, and it is BLOCKED** — `Qwen3-8B-AWQ` (Qwen's
  own published int4, in Edge-LLM's supported list) builds and serves, then emits
  degenerate output: `"0000000..."` for "capital of Poland", a punctuation loop under
  greedy decoding. Not a sampling artifact. It did *not* hit the int4 bias-recipe bug,
  consistent with Qwen3 having dropped the QKV biases that trigger it — so that
  hypothesis held, but the engine is unusable for a different reason further down.
  Root cause open; the `patches/` fix is a live suspect and there is a cheap test to
  settle it (see the row's own notes). **No Qwen3 row on Orin yet either.**
- **Apertus-8B**: genuinely blocked, not neglected — llama.cpp doesn't recognize its
  GGUF architecture on any build tried, and no trustworthy AWQ exists. Worth one
  cheap re-check on Thor: the int4 bias-recipe bug patched in Edge-LLM was blocking
  *every* int4 checkpoint, so it may have been masking an Apertus/Bielik path too.
- **The chat-template test — COMPLETE, 2026-09-10, and it REFUTES the hypothesis.**
  The Thor (Edge-LLM) leg ran precision-matched at 7B fp16 across all three backends,
  via `scripts/chat_template_crossfeed.py`; raw documents in
  `output/chat_template_probe/`. Two results, in order of importance:
  **(1) Edge-LLM and vLLM render byte-identically — 50/50 cases.** Same weights, same
  prompt bytes, 94% vs 54%. So rendering cannot be what separates them. (llama.cpp
  differs from both on 50/50, by exactly two characters: it emits
  `{{"name": ...}}` where the canonical Qwen2.5 template emits `{"name": ...}` — a real
  llama.cpp template bug, confirmed by token count 226 vs 224 with the HF and llama.cpp
  tokenizers agreeing exactly.)
  **(2) The crossfeed matrix settles it.** Feeding each captured rendering to each
  backend that has `/v1/completions` (greedy `top_k=1`, `max_tokens=200`,
  `stop=["<|im_end|>"]`, n=50): vLLM scores 52% on Edge-LLM's rendering and 56% on
  llama.cpp's; llama.cpp scores 60% and 62%. Matched chat-path references on the same
  50 cases reproduce the campaign exactly (Edge-LLM 94%, llama.cpp 62%, vLLM 54%).
  **Swapping the prompt moves abstention by ≤4 points; swapping the backend moves it
  32–40.** The cheap Orin win this test was hoping for — render like Edge-LLM, keep
  vLLM at JetPack 6.2 — is not available.
  **Note Edge-LLM cannot be a crossfeed target at all**: its server exposes no
  `/v1/completions` (only `/v1/chat/completions`, `/v1/messages` and audio routes —
  `experimental/server/api/routes.py`, 0.10.1). That is why this needed a
  rendering-capture design rather than `scripts/chat_template_probe.py` unchanged.
  Superseded record of the two-leg state: `scripts/chat_template_probe.py`, full record in
  `results/raw/2026-09-10_orin_chat-template-probe/`. On Orin, with the template removed
  and the prompt bytes verified identical, **the gap survives**: vLLM 33.3%
  correct-abstain vs llama.cpp 60.0% (against 36.7% / 70.0% through the chat path), only
  5/30 byte-identical completions, and all 8 call-vs-abstain disagreements running the
  same direction. So the template/parser layer accounts for little of the difference on
  this board. **But the Orin leg is confounded by precision** — AWQ int4 vs GGUF Q4_K_M
  — and the precision arm the same day measured precision alone moving this axis 30
  points, so quantization format explains it at least as well as backend runtime does.
  The mechanism behind the *Thor* 40-point finding stays unidentified: that comparison
  held precision at fp16, so it is a different question and needs the Edge-LLM leg.
- **dtype is NOT the mechanism either — tested 2026-09-10.** The one structural
  difference the crossfeed did turn up: `Qwen2.5-7B-Instruct`'s config.json is
  `torch_dtype: bfloat16`, vLLM follows it, and **Edge-LLM's built engine is
  `"dtype": "F16"`** — byte-identical prompts, numerically different weights. Tested
  with `7b-fp16dtype-vllm-thor` (vLLM identical to `7b-fp16-vllm-thor` except
  `--dtype float16`; vLLM logs `Casting torch.bfloat16 to torch.float16`).
  Result: `irrelevance` **54.0%, exactly its bf16 score**. Not a null run — the dtype
  change is real and visible (6 of 50 irrelevance verdicts flip, and `simple` improves
  92%→98%) — but the flips cancel and the abstention rate does not move. Full document,
  schema-conforming, at
  `results/raw/2026-09-10_thor_7b-fp16dtype-vllm-thor_bfcl-v3_n100/`.
  Also note llama.cpp is *already* fp16 and sits at 62%, so dtype could never have
  explained the ordering on its own.
- **A truncation artifact worth knowing before trusting any `irrelevance` number.**
  Qwen2.5 on these prompts writes prose FIRST and emits the tool call after it. At
  `max_tokens=64` vLLM measures **92%** `irrelevance` with `finish_reason=length` on
  every case — a 38-point error in the flattering direction, found and corrected here
  the same day. `validate_tool_calling.py`'s default of 200 is load-bearing, not
  incidental. Even at 200 there are 5–9 truncations per 50 cases, and vLLM's hermes
  parser throws `JSONDecodeError: Unterminated string` on a tool call cut off
  mid-JSON, which scores as an abstention. Worst-case bounds (counting every
  truncation as a call) are Edge-LLM 76% / llama.cpp 50% / vLLM 36% — the ordering and
  the size of the gap both survive, but the absolute numbers are upper bounds.
- **Bielik on Thor — row added 2026-09-10, still NOT smoke-tested.**
  `bielik-11b-q4-llamacpp-thor` now exists, so the *working* Bielik path finally has a
  Thor row (the pre-existing `bielik-11b-awq-vllm-thor` is the variant already known to
  fail on Orin). No Edge-LLM Bielik row is possible: `speakleash/Bielik-11B-v3.0-Instruct`
  is still **gated** (re-confirmed live via the HF API, 2026-09-10) and the ungated
  `-awq` sibling is compressed-tensors, which Edge-LLM's builder does not read — so
  that leg is blocked on gate acceptance, not on lab work.
- **Co-residency on either board** (Phase 9) — the single most valuable unmeasured
  quantity in the project, per both `paper.md` and `embedded-ai-chain`'s own TODO.

---

## Phase 0 — Framework survey (research, done before any code was written)

**Objective**: decide which serving backends are in scope before writing a coordinator
for any of them.

- [x] Confirmed Qwen2.5 GGUF availability: official, Apache-2.0, Qwen-published repos
      for 1.5B/3B/7B — no third-party quant needed.
- [x] **TensorRT excluded from v1 on Orin, superseded for Thor.** TensorRT-LLM's
      Jetson support is JetPack-6.1-only (this Orin is 6.2.x — still excluded here).
      TensorRT Edge-LLM needs JetPack 7.x: assumed out of scope entirely on
      2026-09-04, but **the user's Thor turned out to already be JetPack 7.1**
      (checked live, 2026-09-08) — Edge-LLM 0.10.1 builds and serves there, with
      tool-calling support the original survey found missing. Full story (root
      cause, the bugs found along the way) in `docs/thor-framework-comparison.md`.
- [x] Confirmed llama.cpp's Jetson path: `dusty-nv/jetson-containers` prebuilt images;
      `llama-server` speaks the same OpenAI-compatible wire format as vLLM, so no new
      integration pattern was needed. Note for Thor specifically: that Docker Hub repo
      publishes no Thor (r38) tag at all — jetson-containers' r38-era images live on
      GHCR under `nvidia-ai-iot/` instead, a real gotcha documented in the Thor doc.

- [x] **JetPack 7.2 *has* shipped for Jetson AGX Orin — confirmed 2026-09-10, and it
      does not make the Orin reflash cheaper.** This was an open "nobody has checked"
      question gating the Edge-LLM-on-Orin option. It shipped ~June 2026 (7.2.1 followed
      in August), bringing the Orin family onto JetPack 7 for the first time. The catch
      is what it is built on: **Jetson Linux r39.2, Ubuntu 24.04, kernel 6.8, CUDA 13** —
      against this board's r36.4.7 / Ubuntu 22.04 / CUDA 12.6. So it is not a point
      upgrade, it is a different platform generation:
      - every sibling repo's compiled TensorRT engines are invalidated (they are built
        per TensorRT/CUDA version), `jetson-yolov8-trt` and `jetson-whisper-trt` included;
      - `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` and
        `dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04` are both r36-era images with no
        r39 equivalent confirmed to exist;
      - `embedded-ai-chain`'s Python stack pins the Jetson AI Lab **jp6/cu126** wheel
        index (`onnxruntime-gpu` with the TensorRT/CUDA execution providers, the whole
        reason that index exists) — a jp7/cu13 equivalent has not been checked for the
        components this fleet actually needs;
      - and every existing Orin measurement in this repo becomes non-comparable, which
        is the entire dataset behind the promotion decision.
      **Still not scheduled, and the reason is unchanged**: the Thor-side Edge-LLM leg of
      the chat-template test may make it unnecessary. The 2026-09-10 Orin leg made that
      less likely rather than more (the backend gap survived removing the template
      there), so this needs the Thor answer before anyone commits to a reflash.

**GATE 0** — met, 2026-09-04. vLLM + llama.cpp in scope; TensorRT deferred with reasons
recorded, not silently dropped.

## Phase 1 — Foundations

**Objective**: stand up the coordinator/client/harness primitives and confirm every
candidate family actually serves and can tool-call, before benchmarking any of them.

- [x] `docs/promotion-contract.md` written; `configs/models.yaml` seeded (3 vLLM + 3
      llama.cpp rows, Qwen2.5 1.5B/3B/7B).
- [x] `src/llm_coordinator.py` (`VllmCoordinator`, `LlamaCppCoordinator`,
      `RemoteCoordinator`), `src/llm_client.py`, `benchmarks/harness.py`.
- [x] `scripts/validate_tool_calling.py` — real BFCL scoring (not a hand-rolled case
      list); `scripts/validate_mmlu.py` — quantization-regression sanity check.
- [x] Real smoke tests, both backends, all four families in the registry at the time
      (Qwen2.5-1.5B, Apertus-8B, Bielik-11B). Full diagnostics, every bug found and
      fixed: **`docs/HISTORY.md`, "Phase 1 — smoke-test records"**.
- [x] `make test`: 47/47 on real hardware, clean install. BFCL + MMLU staged on-device.

**Key results:**

| Candidate | vLLM | llama.cpp |
|---|---|---|
| Qwen2.5-1.5B | ✅ works, tool-calling reliable | ✅ works, needs `temperature=0.1` for reliable tool-calling |
| Bielik-11B | ✅ serves; ❌ tool-calling confirmed broken (2 parsers) | ✅ works, tool-calling correct on first try |
| Apertus-8B | no row (no trustworthy AWQ) | ❌ BLOCKED — architecture unrecognized on 2 builds |

First real BFCL scorecard (`1.5b-q4-llamacpp-orin`, n=40): 75% overall (85% simple /
65% irrelevance) — small sample, not the full corpus.

**Generalisable finding**: tool-calling reliability is a property of
(model × backend × parser × temperature), not of the model alone — Qwen2.5 works on
both backends, Bielik only on llama.cpp, the opposite pattern. Recorded as first-class
schema fields for exactly this reason.

**GATE 1** — met for Qwen2.5-1.5B (both backends) and Bielik-11B (llama.cpp; vLLM is a
partial pass — benchmarkable, excluded from the tool-calling scorecard). Apertus is a
closed question (blocked, documented), not an open one. Full record:
`docs/HISTORY.md`.

## Phase 2 — Measurement-integrity retrofit

**Objective**: every benchmark result must carry raw per-run data, a full
reproducibility manifest, real energy, and an honest execution-condition label — before
any campaign runs, because a campaign run without these can't be re-analysed later.

- [x] Raw per-run measurements persisted (`runs[]`); aggregates derived from them only.
- [x] Energy: `benchmarks/runner.py:measure_cell()` merges token counts (from the
      server's own `usage`) with tegrastats power sampled across the same window —
      first real figure, 0.996 J/output-token (llama.cpp, co-resident verification run).
- [x] Cold start decomposed (container start / model load / server ready +
      `weights_cached`), via `_ColdStartMixin`.
- [x] Environment manifest: git commit+dirty, container image+digest, backend version
      **read from the running server**, hardware from the device.
- [x] `results/raw/<experiment_id>/`, `write_result()` refuses to overwrite.
      `--execution-condition` required, no default.
- [x] Every result validated against `schemas/benchmark_result.schema.json` at write time.
- [x] `scripts/benchmark.py` retired (not retrofitted) — `measure_cell()` already
      measures a strict superset. Left as a pointer shim, not deleted, so an old
      invocation fails loudly.

**Three real bugs found running this, not by unit tests alone** — full diagnostics in
`docs/HISTORY.md`, "Phase 2 — incident record":

1. **Prefix-cache contamination, 6.6× TTFT error** (40.8ms → 268.1ms once prompts vary
   per run — both backends were serving every rep after the first from a cache hit).
2. **Inert HF cache mount** — vLLM re-downloaded its weights every run
   (156.2s → 136.2s cold start once fixed).
3. **Energy figure backed by 9 power samples** — a `--min-measurement-s` floor was
   missing from the rewrite; restored.

**GATE 2** — met, 2026-09-04: a real run produced raw rows, an energy figure, a
complete manifest, an experiment ID, and passed schema validation. Not yet done: a raw
→ aggregate re-derivation as a standing proof exercise.

## Phase 3 — Experiment configuration

**Objective**: a campaign runs from `(model config key, experiment config, git SHA)`
alone — no CLI flag carries experimental meaning.

- [x] `schemas/*.json` promoted from example instances to real JSON Schema (4 schemas,
      not 3 — `quality_result` added to keep quality/performance independently
      measurable). `schemas/README.md` documents conformance levels.
- [x] `configs/benchmarks/{smoke,output_sweep,context_sweep}.yaml`, each schema-validated.
- [x] `scripts/run_experiment.py` (config-driven grid runner) +
      `benchmarks/runner.py:measure_cell()` (shared core both entry points call, so
      they can't disagree about how a measurement is taken).
- [x] Campaigns resume (existing results skipped, not overwritten); a failing cell is
      recorded, not fatal to the campaign.
- [ ] Backfill `configs/models.yaml` with `note.md` §28's registry fields
      (`family`, `parameters_b`, `revision`, `quantization{}`, ...) that
      `schemas/model.schema.json` already documents as "Phase 3 target" but doesn't
      yet require.
- [ ] Validate emitted `quality_result` documents in the test suite (blocked on Phase 5).

**Four incidents found before/during the first real campaigns** — full diagnostics in
`docs/HISTORY.md`, "Phase 3 — incident record":

1. A config asked for more context than any candidate has (2048 input + 128 output vs
   every row's 2048-token limit) — caught by a test before any server started.
2. A `standalone` claim was false at the container level (`vllm-orchestrator` running).
3. A `standalone` claim was false at the bare-metal level — `embedded-ai-chain`'s
   *entire production pipeline* runs as one process (`tts_consumer.py`), invisible to
   `docker ps`. 6 results deleted (workload was never controlled, so nothing honest
   could be written after the fact).
4. 5 already-*committed* `co-resident` results were correctly labelled but
   *incompletely* declared (missing that same bare-metal process) — corrected in place
   this time, since the workload was known.

**Result**: `assert_condition_matches_reality()` now checks three things — Docker
containers, bare-metal GPU-holding processes (`/proc/<pid>/fd` scan), and completeness
of a `co-resident` declaration. No override flag on any of the three.

**GATE 3** — met. Verified by a real two-cell grid (one server session, per-cell
documents, shared cold-start correctly flagged, a re-run correctly skipping both
completed cells).

## Phase 4 — Canonical Orin campaign (first complete vertical slice)

**Objective**: `docs/note.md`'s minimum-viable publication experiment, retargeted to
real hardware — one platform, one precision per backend, two swept axes.

- [x] **Output-length sweep** (16/32/64/128/256/512 @ input=512): both backends, 12/12
      cells, 30+ reps each, zero failures. This *is*
      `embedded-ai-chain/docs/paper.md`'s P3 core experiment, run in the form P3 needs.
- [x] **Context-length sweep** (128/512/1024/1536 @ output=128): both backends, 8/8
      cells, 30 reps each, zero failures.
- [x] Both sweeps measure TTFT, prefill/decode tok/s, E2E latency, inter-token
      distribution, peak RAM, power, J/output-token. No throttling observed.

**Key results:**

*J/output-token amortizes with generation length, both backends* — the predicted P3
shape:

| output tokens | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| vLLM J/tok | 0.473 | 0.298 | 0.295 | 0.297 | 0.297 | 0.297 |
| llama.cpp J/tok | 0.949 | 0.757 | 0.682 | 0.647 | 0.673 | 0.657 |

*A real backend-specific context-scaling difference* — TTFT vs input length, decode
throughput held flat on both. **Every vLLM number in the original version of this
table was wrong, and the shape it showed was an artifact** — corrected 2026-09-10,
see "Prompt-salt contamination" below:

| input tokens | 128 | 512 | 1024 | 1536 |
|---|---|---|---|---|
| vLLM TTFT ms (v1, WITHDRAWN) | 40.8 | ~~66.0~~ | ~~83.2~~ | ~~81.8~~ |
| **vLLM TTFT ms (v2, corrected)** | **41.1** | **80.4** | **132.3** | **186.4** |
| llama.cpp TTFT ms (v1) | 213.2 | 310.1 | 630.3 | 897.7 |
| llama.cpp TTFT ms (v2, control) | 216.4 | 313.6 | 637.0 | 897.9 |

**The corrected reading is the opposite of the original one on vLLM's half.** vLLM is
not "near-flat" — it is textbook *linear* in input length: a least-squares fit of the
512/1536 points, `27.4ms + 0.1035ms/token`, predicts 40.7ms at 128 tokens against
41.1ms measured. The v1 row fitted nothing (it "predicted" 60ms at 128 where 40.8 was
measured) and was even non-monotonic at the top, 83.2 → 81.8 — the tell that should
have been caught at the time. llama.cpp is still clearly super-linear and its marginal
prefill cost is **5.5×** vLLM's (0.574 vs 0.104 ms/token).

So the backend difference is real but is a *marginal-cost and curvature* difference,
not the "flat vs super-linear" qualitative gap this table used to claim. Still
confounded by quantization format (AWQ vs GGUF), stated not fixed, per this lab's
fairness rules.

**llama.cpp is the control that makes the correction trustworthy**: the identical code
change moved vLLM by up to 2.3× and moved llama.cpp by <1.5% at every point, because
llama-server keeps a single-slot prompt cache that 30 rotating unique prompts evict,
while vLLM V1 has multi-block automatic prefix caching on by default.

**Also surfaced, not yet a controlled finding**: llama.cpp's standalone decode here
(36-43 tok/s) is roughly double Phase 2's co-resident measurement at the same config
(21 tok/s) — one data point against another, exactly the shape Phase 9 exists to
produce properly.

**Mapping to `embedded-ai-chain/docs/paper.md`:**

| Proposition | Evidence this repo produces |
|---|---|
| P3 — generative extent dominates | Output-length sweep, above, verbatim |
| P4 — deciding to perceive has a cost | Energy above + Phase 5's BFCL no-tool rate |
| P5 — hardware changes the allocation | Phase 6 (Orin vs Thor) |
| P6 — memory is an allocation resource | Phase 9 co-resident runs + OOMs as results |

**GATE 4** — partially met. Both sweeps complete, both backends, real numbers. Not yet
done: an analysis script regenerating plots (Phase 10), and drafting the P3 table into
`paper.md` itself (a parent-repo decision, left explicit rather than done unprompted).

## Phase 5 — Complete the scorecard matrix

**Objective**: every candidate that can serve gets a full benchmark + BFCL + MMLU
scorecard entry.

**Scope decision, 2026-09-07**: Apertus-8B and `bielik-11b-awq-vllm-orin` excluded from
active experimentation — Apertus has no working serving path at all; Bielik-vLLM's
tool-calling is confirmed broken, and `bielik-11b-q4-llamacpp-orin` (working) carries
the family forward. **Rows are not deleted** from `configs/models.yaml` — both marked
excluded in their own `notes`, per this repo's append-only convention. Active matrix:
7 rows (down from 9) — see decision log.

- [x] **Step 0 — smoke-test Qwen2.5-3B/7B first** (never tested on any backend;
      Phase 1's gate was met on 1.5B + Bielik alone). Done 2026-09-07, **4/4 PASS**,
      board confirmed quiet each time; no `gpu_memory_utilization` re-tuning needed on
      either vLLM row. Uncached cold starts 158-327s.
- [x] **Step 1 — the scorecard itself**, all 7 active rows, done 2026-09-08/09. One
      table, since the promotion decision needs speed and accuracy side by side:

  | Row | TTFT p50 | tok/s | J/out-tok | simple | **irrelevance** | MMLU |
  |---|---:|---:|---:|---:|---:|---:|
  | `1.5b-awq-vllm-orin` **(shipping default)** | **31.1ms** | **108.2** | **0.297** | 80.0% | **36.7%** | 40.0% |
  | `3b-awq-vllm-orin` | 132.5ms | 64.4 | 0.566 | 96.7% | **70.0%** | 54.0% |
  | `7b-awq-vllm-orin` | 274.3ms | 33.5 | 1.224 | 96.7% | **40.0%** | 66.0% |
  | `1.5b-q4-llamacpp-orin` | 309.5ms | 38.7 | 0.647 | 100.0% | **70.0%** | 41.5% |
  | `3b-q4-llamacpp-orin` | 562.7ms | 22.2 | 1.178 | 100.0% | **46.7%** | 53.0% |
  | `7b-q4-llamacpp-orin` | 1017.8ms | 11.5 | 2.347 | 100.0% | **56.7%** | 64.0% |
  | `bielik-11b-q4-llamacpp-orin` | 1787.1ms | 7.4 | 4.293 | 100.0% | **10.0%** | 40.0% |

  Performance at `jetson_clocks_locked: true`, MAXN, standalone; BFCL under the
  corrected scorer (2026-09-09). n=60 BFCL, n=200 MMLU. Superseded replicate-1 and
  pre-fix-scorer results are kept on disk, distinguished by `protocol.scorer` — never
  mix them.

  **One case is 3.3 points at n=30**, so `irrelevance` gaps under ~7pp are sampling
  noise. Findings:

  - **Locking `jetson_clocks` roughly halved llama.cpp's TTFT** (Bielik 3181→1787ms,
    7B 1984→1018ms, 3B 1284→563ms) — replicate 1 was partly measuring GPU clock
    ramp-up between idle gaps, not inference. vLLM barely moved, since it keeps the
    GPU saturated and never gives DVFS an idle gap to ramp from. A real finding that
    came out of a methodology correction, and the reason platform state is recorded
    on every result rather than assumed.
  - **`simple` is saturated and no longer discriminates.** Five of seven rows score
    exactly 100%, the other two 96.7%. **`irrelevance` is the only accuracy axis here
    that still separates candidates** — and it is the one that maps to the production
    failure (`ask_vlm` over-escalation). This independently reproduces Thor's
    conclusion (`docs/thor-framework-comparison.md`) on different hardware and
    different quantization.
  - **Bielik-11B almost never abstains on irrelevance cases (10%)** — it calls a tool
    on cases where no tool applies far more often than every Qwen2.5 row, while
    scoring a perfect 100% on `simple`. Exactly the risk a `simple`-only reading hides.
  - **7B shows a real backend gap on irrelevance**: 40.0% on vLLM vs 56.7% on
    llama.cpp, same model size. Consistent with Thor's much larger, same-parser
    version of this gap (94% Edge-LLM vs 54% vLLM at 7B). Confounded by AWQ vs GGUF
    Q4_K_M like every cross-backend comparison in this lab.
  - **The scorer fix did not change the ranking** — every row rose, order held — so
    the promotion decision below stands unchanged.

- [x] **BFCL scorer fix, 2026-09-09.** Argument strings were compared with a plain
      `.strip().lower()` where official bfcl-eval first strips ` ,./-_*^` and spaces,
      so `"3*x**2 + 2*x - 1"` — the same maths as BFCL's accepted spelling, and the
      only form that is valid Python — scored WRONG. Ported official's
      `standardize_string` (Apache-2.0, attributed in the script). Re-scoring
      **identical captured outputs** under both rules: **21 cases flipped across the 7
      rows**, all in `simple_13/14/15/16` — the same four maths cases Thor
      independently hit. **Rankings did not change**, so no conclusion turned on it.
      `protocol.scorer` is now versioned so old and new scores can never be mixed.
      Full record incl. the remaining deliberate deviations: **`docs/HISTORY.md`**.
- [x] ~~Full-corpus BFCL, not `--limit 20`~~ **superseded 2026-09-07** — decided
      against. Full corpus (640 cases) costed out per row from measured decode
      speeds: 6.9min (1.5B vLLM) up to ~95min (Bielik llama.cpp), and a uniform limit
      safe for the slowest row would leave the fastest rows at ~8 cases/category —
      too small to compare. **User decision: bounded ~30/category sample (n=60), all
      7 rows** — comparable sample size across every row beats a full corpus on one
      row and a token sample on the rest.
- [x] Report tool-calling as a confusion matrix (correct tool / correct arguments /
      correct no-tool / invalid call / hallucinated tool / formatting failure), not one
      percentage. Implemented in `scripts/validate_tool_calling.py`'s
      `confusion_matrix_and_taxonomy()`; `tool_call_outcomes` in every BFCL result,
      per-case detail in each result's sibling `outcomes.jsonl`.
- [x] Write the comparison up (README "Current State", this section).

**Hardware incident, 2026-09-08**: the campaign kept dying with no error trace — root
cause was **the board browning out under sustained MAXN load on an undersized 65W
adapter** (the AGX Orin devkit is specified for 90W), confirmed via the Tegra PMC's
`reset_reason=SYS_RESET_N`. Fixed by swapping the supply; the campaign then ran 7+
consecutive rows with zero resets. It also cost hours to a wrong first diagnosis
(a backgrounding-technique bug). Full record: **`docs/HISTORY.md`**.

**GATE 5** — met. Every active row has BFCL, MMLU, and performance/energy at a
confirmed, consistent platform state (`jetson_clocks_locked: true`, MAXN, standalone).

This data drove the promotion decision — **"no clean winner"**, recorded in
`docs/promotion-decision.md`. See Phase 11 for the outcome and its tie-breaker.

## Phase 6 — Thor access and the cross-platform arm

**Objective**: reproduce Phase 4's methodology on a second platform to answer P5 —
does the preferred allocation change with the hardware?

**Done.** Full engineering story in `docs/thor-framework-comparison.md`; the numbers
that drive decisions are below rather than only behind that link. Environment:
JetPack 7.1, CUDA 13.0, TensorRT 10.13.3.9, **122 GiB** unified memory (vs Orin's
~30 GB — `gpu_memory_utilization` values do not transfer between the boards).

**Three backends, `Qwen2.5-7B-Instruct` FP16, same 100 BFCL cases, exclusive box:**

| | Edge-LLM | llama.cpp | vLLM |
|---|---:|---:|---:|
| **BFCL `irrelevance`** | **94%** | 62% | 54% |
| Turn latency (32 tok) | 1945 ms | **1899 ms** | 2539 ms |
| Energy (marginal J/token) | 0.85 J | **0.73 J** | 0.96 J |

Latency and energy are near-ties; the backends differ by **40 points on escalation
judgment**, and that single axis decides it. If tool-call judgment didn't matter,
llama.cpp would win outright (fastest, most efficient, one `docker pull` instead of a
source build). It loses because grammar-constrained generation structurally cannot
abstain. vLLM — `embedded-ai-chain`'s current production backend — is worst on every
axis measured here, at either precision.

**Size**: 1.5B→7B moves `irrelevance` 78%→94% on Edge-LLM, but only 60%→62% on
llama.cpp and *backwards* (68%→54%) on vLLM — the backend decides whether parameters
buy judgment at all. **14B buys nothing**: 98/100 identical verdicts to 7B at 2× the
latency and 2.1× the energy. Thor's 122 GiB makes 7B purchasable where Orin's ~30 GB
never could; that is the real cross-platform finding.

**Recommendation: Edge-LLM + `Qwen2.5-7B-Instruct`, GPTQ-Int4 if the turn budget is
tight, FP16 otherwise.** Precision table in Phase 7 below.

**Two structural gaps this campaign left open** — both real, neither cosmetic:

- [x] **Evidence preserved, 2026-09-10** — the immediate half of the problem below.
      All 68 result JSON files are now committed under `results/thor-precampaign/`
      with a README stating exactly what they are and are not. Previously they sat
      only in `output/`, which `.gitignore` excludes, so **zero Thor results were
      tracked in git** against 82 tracked Orin ones — every published Thor number was
      backed by data that existed on one box's local disk and nowhere else.
- [ ] **Thor's results still have not gone through the measurement-integrity
      pipeline.** The archive above is evidence, not conformance. The campaign ran
      from `scripts/thor_exclusive_window.sh` rather than `run_experiment.py`, so
      those documents carry no `experiment_id`, no manifest, no git SHA, no
      `execution_condition` field, and no per-repetition raw rows — they fail both
      result schemas, and `scripts/analyze.py` (Phase 10) cannot read them. **Thor
      therefore does not yet meet Gates 2 and 3, which Orin met before any campaign
      ran.** Re-emit through `run_experiment.py` (which handles `edge-llm` rows since
      the branch merge) before any Thor number is published as pipeline output.
- [ ] **`assert_condition_matches_reality()` never ran on any Thor result.** Every
      Thor `standalone` claim in these docs is *asserted from the operator's own
      observation* (`docker ps` + a `/proc` scan run by hand in
      `thor_exclusive_window.sh`), not enforced by the guard that has already caught
      three false `standalone` labels on Orin. The shell script's own hard GPU-idle
      gate is a real check and it did abort rather than measure — but it is a
      different, weaker mechanism, and the result documents record no
      `execution_condition` at all. This is the same class of gap as the
      `--target remote` one below, and it applies to data already published.
- [x] **P5's first genuine single-variable answer, 2026-09-10.** `1.5b-awq-vllm-thor`
      vs `1.5b-awq-vllm-orin`: identical checkpoint, precision, backend,
      `max_model_len` and workload point (input=512/output=128), so the **board is the
      only variable**. Every other Thor/Orin pair in this repo varies something else
      too. Run via `configs/benchmarks/p5_cross_platform.yaml` through
      `run_experiment.py` — the first Thor result with a real `experiment_id`,
      manifest and `execution_condition`. 3 replicates, n=30 each.

  | | Thor | Orin | verdict |
  |---|---:|---:|---|
  | decode tok/s | **123.6** | 108.2 / 108.7 | **~14% faster on Thor** — consistent against both Orin runs |
  | TTFT p50 | 54.9 ms | 31.1 / **66.0** ms | **inconclusive** — Thor lands *inside* Orin's own spread |
  | J/output-token | 0.308 | 0.297 / 0.321 | **inconclusive** — Thor inside Orin's spread, and see the rail caveat |

  **The honest headline: Thor buys capacity, not speed, at this size.** Its 4×
  memory is what makes 7B/14B purchasable at all (the real cross-platform finding,
  Phase 6 above) — but for a model that already fits on Orin, decode improves ~14%
  and nothing else clearly moves. Two caveats, both load-bearing rather than
  boilerplate:
  - **Orin's own two runs at this nominally identical point disagree 2× on TTFT**
    (31.1 ms from `output_sweep` vs 66.0 ms from `context_sweep`, same model, same
    day, same input/output). Something uncontrolled differs between those sweeps.
    Until that is explained, no TTFT comparison against this row means anything —
    **and that is a defect in the Orin baseline, not in the Thor measurement.**
  - Thor ran at `jetson_clocks` unlocked (needs root there) against Orin's locked.
    Phase 5 measured that lock as near-irrelevant for vLLM specifically — it keeps
    the GPU saturated, leaving DVFS no idle gap — which is why this comparison is
    worth making at all, but it is not zero.
- [ ] **The two sweeps themselves still not reproduced on Thor.** The row above is
      one workload point, not `output_sweep`/`context_sweep`, so the *shapes* —
      J/token amortization and the backend context-scaling gap — remain unmeasured
      there. Both configs already run against Thor rows; this is execution time.

Also open, tracked in the comparison doc's own "Still open": the `jetson_clocks`-locked
timing run, the official `bfcl-eval` checker, a co-residency run with the VLM tier,
INT8-SQ's unexplained MMLU gap, and identifying the exact mechanism behind the
framework gap (chat-template rendering is the leading candidate).

**Two bugs found 2026-09-10 by actually running an Edge-LLM row**, both silently
live until then — recorded here because each broke something wider than its own row:

- [x] **`EdgeLlmCoordinator` had no `cold_start_breakdown()`**, so `smoke_test.py` and
      `benchmark_streaming.py` crashed on **every** edge-llm row — the whole Thor
      backend was broken for the main measurement path. A merge-integration gap: the
      Orin branch added `_ColdStartMixin` to its two container coordinators, the Thor
      branch added this class, git merged both cleanly (different lines), and every
      test passed. Implemented for a process-owning coordinator: the bimodal cost is a
      TensorRT **engine-cache** miss, not a weight download.
- [x] **`smoke_test.py` returned `PASS` for an engine emitting pure garbage** — its
      only criterion was that the HTTP call hadn't raised. Same hole that let a damaged
      FP8 checkpoint through earlier. Now checks for degenerate output, tuned to flag
      *broken* and never merely *bad*.

**Measurement integrity note, since this Thor is a shared box**: wall latency, TTFT,
tokens/sec, cold start, and any `tegrastats` power/thermal reading need the box
exclusive — another resident GPU process contaminates all of them. Accuracy
(`validate_tool_calling.py`, `validate_mmlu.py`) is unaffected by co-residency and is
fine to run anytime. Record which regime a number came from; an unflagged
shared-box latency figure is worse than no figure.

- [x] **Pre-emptive manifest fix, 2026-09-07** (found auditing Thor-readiness before
      any real Thor row existed): `hardware_manifest()`/`software_manifest()` would
      have silently reported the Orin's own hardware/Docker info under a manifest
      labelled `"platform": "thor"` for any `--target remote` run. Fixed — remote
      runs now report honest `"unknown (remote target...)"` values instead. Full
      record: `docs/HISTORY.md`. 5 new tests.
- [ ] **Still open, and NOT resolved by the work above**: `assert_condition_matches_reality()`
      is skipped entirely for `--target remote` — it can only see the client's own
      processes, not Thor's. Everything in the comparison doc was measured with
      `--target local` running directly on Thor, which never exercises this path, so
      the co-residency guard still does not extend to a genuine `--target remote`
      run against Thor. Needs an SSH-based remote check or a manual on-Thor
      companion check — a real design decision, not a one-line fix.

**GATE 6** — met for the local-on-Thor arm: one real run of each script succeeded
end-to-end, with a full exclusive-box campaign on top. The `--target remote`
co-residency gap above is separate and still open.

## Phase 7 — Quantization study

**Objective**: same model/backend/hardware, precision as the only variable.

**Done on Thor, 2026-09-09/10** — `Qwen2.5-7B-Instruct` on Edge-LLM, precision as the
only variable, MMLU carried throughout as the regression control this phase asks for:

| precision | BFCL `irrelevance` | MMLU | 32-token turn | marginal J/token | cold start ¹ |
|---|---:|---:|---:|---:|---:|
| **INT4-GPTQ** | 90% | 66.5% | **769 ms** | **0.274 J** | **0.03 s** |
| **FP16** (baseline) | **94%** | **67.5%** | 1945 ms | 0.839 J | 12.1 s |
| FP8 (self-quantized) | 90% | 66.5% | 1052 ms | 0.603 J | — |
| INT8-SQ (self-quantized) | 92% | 54.0% ² | 1074 ms | 0.610 J | — |

¹ Cache **hit** for both; a miss takes minutes either way. ² **INT8-SQ keeps an
unexplained 12.5-point MMLU gap** even after raising calibration 128→512 samples
(44.5%→54.0%) — not trusted for production; see the comparison doc's "Still open".

**INT4-GPTQ wins every timing and energy column outright** — 2.5× the throughput and
3× the energy efficiency of FP16 — for 4 points of `irrelevance`. FP8 is essentially
lossless on accuracy but only ~1.4× FP16's throughput. Both are legitimate; which to
ship is a genuine product call about the turn budget, and both are recorded rather
than one being declared correct.

- [x] Label quantization methods precisely, never "INT4" generically. Every Thor row
      names its actual method (`gptq int4`, `fp8 static`, `int8 smoothquant W8A8`),
      and the self-quantized ones record calibration dataset and sample count — which
      is what made the INT8 gap diagnosable at all.
- [x] Per candidate: smoke → MMLU regression → performance → memory → power against
      the same-model higher-precision baseline. Ran in exactly that order; the FP8
      first attempt was caught by it (100% `irrelevance` that turned out to be **zero
      tool calls in 50 tries** — a broken checkpoint, not a good score).
- [x] **Both recommended precisions are official Qwen checkpoints, downloaded — not
      quantized by us.** Worth stating plainly, because the debugging story below
      makes self-quantization look load-bearing and it isn't: `Qwen2.5-7B-Instruct`
      and `Qwen2.5-7B-Instruct-GPTQ-Int4` are ordinary `hf download`s. Reproducing the
      recommendation needs **no quantization pipeline** — only the one-line Edge-LLM
      patch (`patches/edgellm-int4-bias-recipe.patch`), without which *every* int4
      checkpoint fails to build, Qwen's own official ones included. That patch is the
      real enabler here, not the quantizer.
- [x] **Self-quantization capability built** for the two formats where **no published
      checkpoint exists for this model at all** (FP8, INT8-SQ) — neither of which is
      recommended. Recipes now reproducible via `scripts/quantize_edgellm.sh`, which
      records the exact flags plus the two failure modes that made the first attempt
      at each unusable. They were previously ad-hoc shell commands recorded only in
      prose, which made the four checkpoints in `/home/michal/dev/quantize-work`
      irreproducible the moment that directory went away.
- [ ] **Orin arm not started**: `7b-awq-vllm-orin` vs a 7B FP16 row (record the OOM as
      the result if it doesn't fit in ~30 GB). This is the interesting comparison
      precisely *because* Thor's answer may not transfer — 122 GiB makes FP16 free
      there and impossible here.
- [ ] **Qwen3 has never been tested on either board**, and on Thor it is now the
      highest-value single experiment left. Three reasons, all checked, not assumed:
      Edge-LLM's supported-models list includes **official `Qwen3-8B-AWQ` and
      `Qwen3-14B-AWQ`** published by Qwen; Qwen3 **dropped the attention QKV biases**
      that caused the int4 bias-recipe bug, so it likely avoids that path entirely;
      and it would be this lab's first *published* int4 checkpoint on Edge-LLM rather
      than a self-quantized one. The old "vLLM has no JetPack 6.2 wheels" blocker was
      measured false in 2026-09-04 (`vllm 0.19.0` confirmed live) — nothing is
      actually stopping this but execution time.

**GATE 7** — met on Thor (one model, one backend, four precisions, MMLU-controlled).
Not met on Orin, and not met for any second model family.

## Phase 8 — Backend study

**Objective**: same model/precision-class/hardware, backend as the only variable.

**Done on Thor, 2026-09-08/10 — and done more strictly than this phase was written to
ask for.** Table in Phase 6 above. Three backends rather than two, and where this
phase only asked to *state* the AWQ-vs-GGUF confound honestly, the Thor campaign
**eliminated** it: all three backends served the identical FP16 weights, so precision
is not a confound at all.

- [x] Backend as the only variable — same model, same weights, same 100 BFCL cases.
- [x] ~~State the AWQ-vs-GGUF confound~~ **superseded**: removed rather than stated,
      by holding FP16 across all three backends.
- [x] **Parser ruled out as the explanation.** Edge-LLM re-run with `hermes` (the
      parser vLLM used) scored identically to `auto` — 100/100 identical per-case
      verdicts, at both model sizes.
- [x] **Decoding ruled out as the explanation**, and this one exposed a real
      methodology flaw worth keeping: the campaign had pinned only `temperature`, but
      the three backends apply three *different* truncation defaults (Edge-LLM
      top_k=50/top_p=0.9; vLLM top_k=-1/top_p=1.0; llama.cpp
      top_k=40/top_p=0.95/min_p=0.05). **Pinning temperature alone is not controlled
      sampling.** Re-running all three at `top_k=1` (deterministic argmax, making the
      rest inert) reproduced every score exactly.
- [x] **Mechanism identified qualitatively**: it is abstention, not detection. All
      three score 90–92% on `simple`; the entire spread is `irrelevance`. llama.cpp
      constrains generation to a tool-call grammar, so a structurally-forced call
      leaves no room to abstain; vLLM and Edge-LLM detect tool tags in free
      generation instead.
- [ ] **Which structural difference exactly, still unidentified.** With weights,
      parser, and decoding all controlled, the gap is structural — leading candidate is
      chat-template rendering, since each backend builds the tools prompt itself.
      Cheap decisive test, designed but not run: send one identical pre-rendered prompt
      to all three via `/v1/completions` (no `tools` parameter, no template) and
      compare raw output.
- [ ] **Orin arm not started**: vLLM vs llama.cpp temperature-robustness as a measured
      curve. Still worth running — it is the one backend question Thor's FP16
      comparison cannot answer, since Orin's rows are AWQ vs GGUF and that confound is
      real there.

**GATE 8** — met on Thor, with confounds eliminated rather than merely disclosed. Not
met on Orin; the causal mechanism is narrowed but not pinned.

## Phase 9 — Co-resident / concurrency (highest external value)

**Objective**: the number both `paper.md` and `embedded-ai-chain`'s own TODO call the
single most important unmeasured quantity in the whole project.

- [ ] Condition A (standalone) vs B (co-resident with the real pipeline: YOLO + STT +
      TTS + VLM), same model/workload/methodology.
- [ ] Report both directions: what co-residency does to LLM TTFT/decode/energy, **and**
      what the LLM does to `yolo_frame` p50/p95/p99 (the number `embedded-ai-chain`
      actually needs).
- [ ] Batch/concurrency sweep 1/2/4/8 — batch=1 is the deployment reality, the rest is
      characterization; never conflate the two terms.
- [ ] Every OOM is a recorded result, including the ones already on record (VLM OOM at
      `gpu_memory_utilization=0.6`, the 276MB-free thin-margin finding).

**GATE 9** — not met.

## Phase 10 — Analysis pipeline

- [ ] `scripts/analyze.py` regenerating every table/plot from `results/raw/`. Never
      hand-edit a plot.
- [ ] Pareto frontier over (quality, latency, memory, energy) — no weighted score.
- [ ] Keep the application scorecard separate from the raw benchmark.

**GATE 10** — not met.

## Phase 11 — Declare (or defer) a winner

**Orin: done, 2026-09-08 — `docs/promotion-decision.md`.** The contract's §4 allows
two outcomes, and this campaign produced the second one legitimately rather than
failing to reach the first.

- [x] **Declared: "no clean winner", with the tie-breaker condition recorded.** Every
      candidate with better tool-calling accuracy than the shipping default is also
      slower, and the shipping default *already* misses `embedded-ai-chain`'s ≤1000ms
      budget (that repo measured 1.06s p50 end-to-end), so an accuracy-motivated swap
      would deepen an already-open latency miss. Tie-breaker: **if that project accepts
      a revised tool-calling latency budget, `3b-awq-vllm-orin` is the strongest
      candidate measured** — 70.0% `irrelevance` vs the default's 36.7%, a 33-point gap
      well outside the ~7pp noise band, with no MMLU red flag.
- [x] Pointer/decision-log row added to `embedded-ai-chain/docs/TODO.md`, per §4.3.
- [x] **A quantified risk delivered even without a promotion**: the shipping default
      wrongly calls a tool on **63.3%** of cases where none applies. The `ask_vlm`
      over-escalation bug was previously known only anecdotally; this is its first
      real frequency, and it is worth acting on independently of any model swap.
- [ ] Update `embedded-ai-chain/src/orchestrator_models.py` — **correctly not done**:
      nothing was promoted, so there is nothing to change. This stays unchecked as the
      honest state, not as an oversight.

**Thor: a recommendation exists, but it is deliberately not a promotion.** Phase 6
recommends Edge-LLM + `Qwen2.5-7B` (GPTQ-Int4 or FP16). That is the right answer to
*"what should serve an orchestrator on Thor"* — but `embedded-ai-chain` runs on the
**Orin**, which cannot run Edge-LLM at all (JetPack 6.2.x). So the Thor result cannot
promote anything into that project today; it is a platform finding and a strong
argument for what future hardware buys, not a shipping decision.

- [ ] Re-run the promotion contract against Thor properly *if* the parent project ever
      targets Thor — which would need the co-resident measurement §3 requires
      (everything measured so far is `standalone`) and the `results/raw/` gap in
      Phase 6 closed first, since a promotion cannot rest on evidence that isn't in
      the repo.

**GATE 11** — met for Orin (a decision was reached, justified, and recorded). Open for
Thor by design, not by omission.

---

## Deferred / explicitly not doing yet

- **TensorRT-LLM / TensorRT Edge-LLM** — revisit only if Thor moves to JetPack 7.x.
- **A declarative multi-machine launch config** — premature for a 2-machine reality.
- **How many papers this becomes** — decided after the results exist, not now; this
  repo feeds `paper.md`/`PAPER_PLAN.md` regardless.
- **From `note.md`, deliberately not scheduled**: GSM8K/ARC/IFEval/HumanEval/Belebele
  (MMLU+BFCL already cover this lab's two decision axes); a power-mode sweep (defer
  until one campaign is complete at a documented mode); benchmark tiers 0-3 (cheap to
  add to `configs/benchmarks/` later, not a phase); a custom application dataset
  (needs real co-resident traffic from Phase 9 first); reasoning-token accounting
  (mandatory once a thinking-mode row exists, none does yet); an immutable
  `benchmark-v1.0` release (a packaging step, correct to do last).

## Decision log (append-only)

Rows are never deleted, even when superseded.

| Date | Decision | Reason |
|---|---|---|
| 2026-09-04 | Repo becomes a benchmark suite that also selects the orchestrator LLM, not one or the other | `docs/note.md`; the selection scorecard is a strict subset of the benchmark data |
| 2026-09-04 | Orin is the platform of record; Thor is the cross-platform arm | No Thor access confirmed yet; and "does the ranking change across hardware" needs both boards anyway (paper.md P5) — **later overtaken by events**: the user cloned this repo directly onto Thor on 2026-09-08 and ran the cross-platform arm from there; see that date's rows below |
| 2026-09-04 | Existing harness warmup/steady-state policy kept over note.md §4's fixed `warmup: 5` | Temperature-driven steady state with an honest `warmup_reached_steady_state` flag is strictly stronger than a guessed count |
| 2026-09-04 | Measurement-integrity retrofit (raw rows, energy, manifest, IDs, co-residency flag) precedes all campaigns | These are unrecoverable if skipped — a campaign run without them cannot be re-analysed |
| 2026-09-04 | ~~No third paper from this repo~~ **superseded same day** | Two papers already share one writing window |
| 2026-09-04 | Paper count decided after results exist, not now | Direct user decision; nothing in Phases 2-10 changes based on the answer |
| 2026-09-04 | Prompts vary per repetition by default | Identical prompts caused a 6.6x TTFT error via KV-cache reuse |
| 2026-09-04 | `HUGGINGFACE_HUB_CACHE`/`HF_HUB_CACHE` set explicitly (coordinator + compose) | The image's own baked-in value overrode `HF_HOME`, making the cache mount inert |
| 2026-09-04 | `scripts/benchmark.py` retired, not retrofitted | `stream_llm()`-based measurement already covers a strict superset |
| 2026-09-04 | Measurement continues past `--runs` until `--min-measurement-s` elapses | A fast cell produced an energy figure backed by 9 power samples |
| 2026-09-04 | `write_result()` refuses to overwrite | Immutability enforced, not trusted — a re-run must be a new replicate |
| 2026-09-04 | Apertus and Bielik stay in the matrix despite note.md not mentioning them | This lab's actual 2nd/3rd families and its only decisive negative result |
| 2026-09-04 | ~~Qwen3 gated because vLLM has no JetPack 6.2 wheels~~ **superseded same day** | Premise measured false for this container path |
| 2026-09-04 | Qwen3 stays gated on a real serving test, as routine practice not a known blocker | `vllm 0.19.0` confirmed running; second time a version was wrongly assumed from a name |
| 2026-09-04 | Co-residency guard also scans `/proc/<pid>/fd` for bare-metal GPU processes | Container-only check missed `tts_consumer.py` running the entire production pipeline |
| 2026-09-04 | ~~Contaminated results are deleted~~ **corrected same day** — archive/correct, never delete | A contaminated run is still data; the 6 files from that incident were already gone, policy applies forward |
| 2026-09-04 | Co-residency guard also checks `co-resident` declarations for completeness | 5 committed results were correctly labelled but incompletely declared |
| 2026-09-07 | Apertus and `bielik-11b-awq-vllm-orin` excluded from active experimentation; rows kept | No working path (Apertus) / broken tool-calling (Bielik-vLLM); append-only convention preserved |
| 2026-09-07 | Qwen2.5-3B/7B smoke-tested before Phase 5's scorecard touches them | Never tested at all; same bar every row was held to before its first heavy benchmark |
| 2026-09-07 | `docs/TODO.md` split: incident narratives moved to `docs/HISTORY.md` | File had grown to ~980 lines and was no longer scannable; nothing deleted, just relocated |
| 2026-09-08 (Orin) | Phase 5 campaign blocked on hardware: board browns out under sustained MAXN load | PMC `reset_reason=SYS_RESET_N`, `reset_level=L0` — external reset line asserted, i.e. undervoltage. Rules out panic (`SW_MAIN`), watchdog (`*WDT`), thermal (`SENSOR`) and OOM (no reboot at all). Root cause: a 65W supply on a devkit specified for 90W (19V/4.74A) |
| 2026-09-08 (Orin) | Benchmark runs must be hosted outside the Claude Code session (tmux) | Not sufficient on its own — the reboot kills tmux too — but it removes session teardown as a confound, which masked the real cause for hours |
| 2026-09-08 (Orin) | `jetson_clocks` state must be re-asserted and re-verified after every reboot | It does not persist; the governor returns to `schedutil`. A crash-and-resume campaign can silently change platform state mid-experiment, violating the rule that every number records its power mode and clock state |
| 2026-09-08 (Orin) | BFCL sample size for the Phase 5 scorecard: bounded ~30/category (n=60), all 7 rows, not full-corpus on any row | Full corpus (640 cases) costed 6.9-95min per row from measured decode speeds; a uniform limit safe for the slowest row would starve the fastest to ~8 cases/category. Direct user decision |
| 2026-09-08 (Orin) | Board's 65W power adapter replaced with the devkit's specified 90W unit | `SYS_RESET_N` hard resets under sustained MAXN load, measured peak draw 58.4W on 3 rails alone (not full-board). Confirmed root cause of the whole campaign's earlier "dies at ~7-10min" pattern, previously misdiagnosed as a backgrounding-technique bug |
| 2026-09-08 (Orin) | Step 1's 5 scorecard performance/energy rows re-measured as replicate 2 at `jetson_clocks_locked: true` | Replicate 1 was taken before the power-adapter fix at `false`; `jetson_clocks` doesn't persist across the reboot that preceded those runs. Old results kept (immutability), not deleted. Correction also surfaced a real finding: llama.cpp's TTFT roughly halves once clocks are genuinely locked, not just an accuracy fix |
| 2026-09-08 (Thor) | Thor's vllm rows get their own image tag and a non-8000 port, not copies of the orin rows' values | `vllm-vlm-thor` is already live in production on Thor at port 8000 using the `-jetson-thor` image tag, not `-jetson-orin` - confirmed via `docker ps`, not assumed |
| 2026-09-08 (Thor) | ~~llama-cpp/thor stays row-less (blocked)~~ **superseded the same day** | See 2026-09-09 (Thor) row below — was checking Docker Hub only, not GHCR |
| 2026-09-08 (Thor) | TensorRT Edge-LLM becomes a real backend leg, on Thor only | Direct user direction; and Phase 0's three blockers no longer hold on Thor - JetPack 7.1 is an Official row, Edge-LLM 0.10.1 ships an OpenAI-compatible server with tool-calling, and Qwen2.5-*-AWQ is in its supported models |
| 2026-09-08 (Thor) | Thor timing/power/memory-fit runs require an exclusive box; accuracy runs do not | Thor is shared with another user running a production vLLM container - co-residency changes latency/thermal/fit numbers but not BFCL/MMLU correctness |
| 2026-09-09 (Thor) | llama.cpp on Thor works after all, via a different image family than Orin's | `dustynv/llama_cpp` genuinely has no r38 tag on Docker Hub, but jetson-containers publishes r38-era images to GHCR under `nvidia-ai-iot/` instead - checking one registry was not checking the project |
| 2026-09-09 (Thor) | GPTQ-Int4 quantization added to the recommendation, not just FP16 | A one-line bug found in Edge-LLM's own source (`patches/edgellm-int4-bias-recipe.patch`) unblocked it; 2.5x lower latency and 3x lower energy than FP16 for 4 points of `irrelevance` |
| 2026-09-09/10 (Thor) | Self-quantized FP8/INT8-SQ added via `tensorrt-edgellm-quantize`, not left untested | No published checkpoint exists in either format for this model (confirmed against NVIDIA's own supported-models page) — self-quantizing was the only path, and both needed real debugging before the checkpoints were trustworthy |
