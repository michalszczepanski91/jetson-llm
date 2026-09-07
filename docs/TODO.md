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

| Phase | Status | Gate | Key result |
|---|---|---|---|
| 0 — Framework survey | ✅ done | met | vLLM + llama.cpp in scope; TensorRT explicitly excluded (JetPack mismatch) |
| 1 — Foundations | ✅ met for active candidates | met (see caveats) | Qwen2.5-1.5B works both backends; Bielik works llama.cpp only; Apertus blocked entirely |
| 2 — Measurement-integrity retrofit | ✅ done | met | Raw runs, energy, manifest, IDs, co-residency guard — 3 real bugs found & fixed |
| 3 — Experiment configuration | ✅ done | met | Config-driven campaigns; co-residency guard hardened after 2 more incidents |
| 4 — Canonical Orin campaign | 🚧 partial | partial | 20 real cells, 2 findings (J/token amortization, backend context-scaling gap) |
| 5 — Full scorecard matrix | 🚧 in progress | not met | Exclusions decided; 3B/7B smoke-tested (4/4 pass); scorecard itself not started |
| 6 — Thor cross-platform arm | ⏳ not started | not met | Pre-emptive manifest fix landed; co-residency guard gap on Thor still open |
| 7 — Quantization study | ⏳ not started | not met | — |
| 8 — Backend study | ⏳ not started | not met | — |
| 9 — Co-resident/concurrency | ⏳ not started | not met | Highest external value — feeds both papers |
| 10 — Analysis pipeline | ⏳ not started | not met | — |
| 11 — Declare a winner | ⏳ not started | not met | — |

Full reconciliation of this plan with `docs/note.md`'s original benchmark-suite
proposal: `docs/HISTORY.md`, "Scope change, 2026-09-04".

---

## Phase 0 — Framework survey (research, done before any code was written)

**Objective**: decide which serving backends are in scope before writing a coordinator
for any of them.

- [x] Confirmed Qwen2.5 GGUF availability: official, Apache-2.0, Qwen-published repos
      for 1.5B/3B/7B — no third-party quant needed.
- [x] **TensorRT excluded from v1.** TensorRT-LLM's Jetson support is JetPack-6.1-only
      (this Orin is 6.2.x); TensorRT Edge-LLM needs JetPack 7.x and has no
      OpenAI-compatible server. Revisit only if/when Thor moves to JetPack 7.x.
- [x] Confirmed llama.cpp's Jetson path: `dusty-nv/jetson-containers` prebuilt images;
      `llama-server` speaks the same OpenAI-compatible wire format as vLLM, so no new
      integration pattern was needed.

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
throughput held flat on both:

| input tokens | 128 | 512 | 1024 | 1536 |
|---|---|---|---|---|
| vLLM TTFT ms | 40.8 | 66.0 | 83.2 | 81.8 |
| llama.cpp TTFT ms | 213.2 | 310.1 | 630.3 | 897.7 |

vLLM near-flat over the range; llama.cpp clearly super-linear (roughly doubles
512→1024 alone). A *prefill*-specific difference, not general speed — confounded by
quantization format (AWQ vs GGUF), stated not fixed, per this lab's fairness rules.

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

- [x] **Step 0 — smoke-test Qwen2.5-3B/7B first** (never tested on any backend; Phase 1's
      gate was met on 1.5B + Bielik alone). **Done 2026-09-07, `scripts/smoke_test.py`,
      board confirmed quiet before each run — 4/4 PASS:**

  | Row | Cold start (uncached) | Result |
  |---|---|---|
  | `3b-awq-vllm-orin` | 188.3s | ✅ PASS, `gpu_memory_utilization=0.25` needs no re-tuning |
  | `3b-q4-llamacpp-orin` | 158.3s | ✅ PASS |
  | `7b-awq-vllm-orin` | 212.3s | ✅ PASS, `gpu_memory_utilization=0.5` needs no re-tuning — note: a benign `NvMapMemAllocInternalTagged` allocator warning appeared during startup, worth watching given this row's own "high-risk" flag |
  | `7b-q4-llamacpp-orin` | 326.6s | ✅ PASS (7.6GB two-part GGUF download included) |

- [ ] **Step 1 — the scorecard itself**, now unblocked for all 7 active rows: run
      `benchmark`/`benchmark-streaming`/`validate-tool-calling`/`validate-mmlu` for
      each. Record an OOM as a real result, not a skip.
- [ ] Full-corpus BFCL, not `--limit 20` (the only real scorecard so far is a sample).
- [ ] Report tool-calling as a confusion matrix (correct tool / correct arguments /
      correct no-tool / invalid call / hallucinated tool / formatting failure), not one
      percentage.
- [ ] Write the comparison up (README "Current State" or a dedicated benchmarks doc).

**GATE 5** — not yet met. Every plausible row needs a scorecard entry (pass,
fail-with-reason, or doesn't-fit-with-reason). Step 0 is done; Step 1 hasn't started.

## Phase 6 — Thor access and the cross-platform arm

**Objective**: reproduce Phase 4's methodology on a second platform to answer P5 —
does the preferred allocation change with the hardware?

Demoted from "next" to "after the Orin slice is real" — a hardware-availability
dependency this repo doesn't control; nothing above is blocked on it.

- [ ] Confirm SSH/LAN reachability to the user's Thor (address unconfirmed).
- [ ] Confirm what's already running on Thor (avoid port/GPU-memory collision).
- [ ] Record Thor's real power modes and unified-memory size — never assume Orin's
      transfer.
- [ ] Add `-thor` rows to `configs/models.yaml` once real serving args are confirmed —
      don't guess `gpu_memory_utilization`/`n_gpu_layers` ahead of a real run.
- [ ] Re-run Phase 4's two sweeps on Thor, same methodology.
- [x] **Pre-emptive manifest fix, 2026-09-07** (found auditing Thor-readiness before any
      real Thor row exists): `hardware_manifest()`/`software_manifest()` would have
      silently reported this Orin's own hardware/Docker info under a manifest labelled
      `"platform": "thor"` for any `--target remote` run. Fixed — remote runs now report
      honest `"unknown (remote target...)"` values instead. Full record:
      `docs/HISTORY.md`. 5 new tests.
- [ ] **Still open**: `assert_condition_matches_reality()` is skipped entirely for
      `--target remote` — it can only see this client's own processes, not Thor's. The
      co-residency guard Phase 3 built does **not** extend to Thor yet. Needs an
      SSH-based remote check or a manual on-Thor companion check — a real design
      decision, not a one-line fix.

**GATE 6** — not met. One real run of each script against Thor, end-to-end, with the
co-residency gap above resolved or explicitly accepted — not discovered after a
campaign the way Phase 3's incidents were.

## Phase 7 — Quantization study

**Objective**: same model/backend/hardware, precision as the only variable.

- [ ] `7b-awq-vllm-orin` vs a 7B fp16 row (if it fits in ~30GB alongside nothing else;
      record the OOM as the result if not).
- [ ] Label quantization methods precisely, never "INT4" generically —
      `bielik-11b-awq-vllm-orin`'s row already does this right (`compressed-tensors
      int4, group_size=128` despite being named `-awq`).
- [ ] Per candidate: smoke → MMLU regression → performance → memory → power, against
      the same-model higher-precision baseline.
- [ ] Qwen3 row — the expected vLLM-version blocker doesn't exist (pinned image reports
      `vllm 0.19.0`, confirmed live); still add behind a real serving test like every
      other row.

**GATE 7** — not met.

## Phase 8 — Backend study

**Objective**: same model/precision-class/hardware, backend as the only variable.

- [ ] vLLM vs llama.cpp: turn the qualitative finding (vLLM's tool-call parser is
      temperature-robust, llama.cpp's isn't) into a measured curve.
- [ ] State the AWQ-vs-GGUF confound honestly — this isn't a clean precision match.

**GATE 8** — not met.

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

- [ ] Declare per `docs/promotion-contract.md` §4, or explicitly record "no clean
      winner, deferred + tie-breaker condition".
- [ ] Pointer/decision-log row in `embedded-ai-chain/docs/TODO.md`.
- [ ] If the winner changes the shipping model/backend, update
      `embedded-ai-chain/src/orchestrator_models.py` — that repo's own change, made
      with a real number behind it.

**GATE 11** — not met.

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
| 2026-09-04 | Repo becomes a benchmark suite that also selects the orchestrator LLM | `docs/note.md`; the selection scorecard is a strict subset of the benchmark data |
| 2026-09-04 | Orin is the platform of record; Thor is the cross-platform arm | No Thor access confirmed; "does ranking change with hardware" needs both boards anyway (paper.md P5) |
| 2026-09-04 | Existing harness warmup/steady-state policy kept over note.md's fixed `warmup: 5` | Temperature-driven steady state with an honest failure flag is strictly stronger than a guessed count |
| 2026-09-04 | Measurement-integrity retrofit precedes all campaigns | Unrecoverable if skipped — a campaign run without it can't be re-analysed |
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
