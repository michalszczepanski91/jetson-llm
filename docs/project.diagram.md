# Architecture diagrams — jetson-llm

Companion to `docs/note.md` (the methodology) and `docs/TODO.md` (the phased plan).
This file is the *picture*: what the pieces are, what exists today, and what is still
a plan.

Everything below marks its own status, because the single most useful thing a diagram
in this repo can do is stop a reader assuming a box is built when it isn't:

```
[✓]  built and exercised on real hardware
[~]  built, but not yet complete per docs/TODO.md Phase 2
[ ]  planned, does not exist yet
```

Renamed from `progect.diagram.md` on 2026-09-04 (typo, never-committed file). Sections
1-4 revised 2026-09-04, same day: the Phase 2/3 retrofit and a real Phase 4 campaign
both landed within the day this diagram was first drawn, so several `[ ]`/`[~]` boxes
below are now `[✓]` with a real run behind them, not a guess.

---

## 1. Pipeline — config in, results out

```
   configs/models.yaml [✓]              configs/benchmarks/*.yaml [✓]
   "WHICH candidate?"                   "WHAT experiment?"
   model x precision x backend          input/output token grids,
   x platform, append-only,             batch/concurrency, sampling,
   9 rows today                         warmup + repetition floors
                                         3 configs: smoke, output_sweep,
                                         context_sweep
          │                                        │
          │  validated by                          │  validated by
          │  schemas/model.schema.json [✓]         │  schemas/experiment.schema.json [✓]
          │                                        │
          └────────────────┬───────────────────────┘
                           │
                           │  one (candidate, workload point) pair = one cell
                           ▼
              ┌─────────────────────────────┐
              │   scripts/run_experiment.py │  [✓]  the grid runner — GATE 3
              │   scripts/benchmark_streaming.py │  [✓]  single-cell ad-hoc probe
              │                             │
              │  both call one shared core: │
              │  benchmarks/runner.py       │  [✓]  measure_cell() — warmup,
              │    :measure_cell()          │       measurement, aggregation,
              │  benchmarks/manifest.py     │  [✓]  manifest, experiment ID,
              │                             │       energy, the co-residency
              │  src/llm_coordinator.py [✓] │       guards (§3)
              │  src/llm_client.py      [✓] │  OpenAI-compatible round trip,
              │  benchmarks/harness.py  [✓] │  telemetry (TegrastatsSampler,
              │                             │  percentiles()) — the same
              │                             │  hand-maintained copy shared
              │                             │  with sibling repos
              └──────────────┬──────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   VllmCoordinator    LlamaCppCoordinator   RemoteCoordinator
        [✓]                  [✓]                  [~]
   Docker, vLLM        Docker, llama-server   already-running server
   :8000               :8090 (not 8080 —      elsewhere, e.g. Thor
                       collides with          — untested, no Thor
                       embedded-ai-chain's    access confirmed yet
                       dashboard)

   All three share one duck-typed interface: base_url, model,
   start(), wait_ready(), stop(). Both real backends speak the same
   OpenAI-compatible wire format, which is why adding llama.cpp needed
   a new coordinator and no new integration pattern.

   scripts/benchmark.py [retired] — the pre-retrofit non-streaming path.
   Now a shim that exits pointing at the two entries above; kept as a
   file (not deleted) so an old invocation fails loudly, not silently.
```

## 2. What a run measures — one merged path (was: two that had to merge)

This was the most important structural gap in the repo when this diagram was first
drawn; it's closed as of the same day. Kept here as the record of what was wrong and
why the fix looks the way it does — `docs/TODO.md` Phase 2's own record has the full
incident list.

```
  BEFORE (two scripts, split-brain):

  scripts/benchmark.py                  scripts/benchmark_streaming.py (old)
  non-streaming, one number             streaming
          │                                        │
          ▼                                        ▼
  ┌───────────────────┐                  ┌───────────────────┐
  │ harness.py         │                 │ per-run TTFT and  │
  │  · cold start      │                 │ tokens/sec        │
  │  · latency p50/95/99                 │                   │
  │  · tegrastats:      │                │  NO tegrastats    │
  │      power rails    │                │  sampler at all   │
  └────────┬──────────┘                  └─────────┬─────────┘
           │                                       │
     knows POWER                             knows TOKENS
     but not tokens                          but not power
           │                                       │
           └───────────────  ✗  ──────────────────┘
                             │
              J / generated token was UNCOMPUTABLE
              from either script.

  AFTER (one merged path, 2026-09-04):

              benchmarks/runner.py:measure_cell()  [✓]
              runs stream_llm() (knows tokens, from the
              server's own `usage`) WHILE a TegrastatsSampler
              is live across the same measurement window
              (knows power) — one function, both signals
                             │
                             ▼
              power.energy_per_output_token_j  [✓]
              first real figures, 2026-09-04:
                0.306 J/tok  (vLLM, standalone)
                0.657-0.996 J/tok (llama.cpp, varies with
                  execution_condition - see §5)
              docs/note.md §14 calls this the headline
              metric; embedded-ai-chain/docs/paper.md
              §7.1 builds Semantic Decision Energy on it.

              scripts/benchmark.py [retired] rather than
              retrofitted a second time - see §1.
```

## 3. Result lifecycle — where a number goes

```
              one cell runs
                    │
                    ▼
        ┌───────────────────────────┐
        │  RAW per-run measurements │  [✓]  Phase 2, done 2026-09-04
        │  every repetition, kept   │       runs[] holds every rep;
        │  ttft / e2e / decode /    │       aggregates derived from it
        │  tokens / finish_reason   │       and nothing else
        └─────────────┬─────────────┘
                      │
                      ├──────────────► aggregates  (p50/p95/p99, never a mean)
                      │                 regenerable, never the only copy
                      │
                      ▼
        ┌───────────────────────────┐
        │  + manifest               │  [✓]  Phase 2
        │  git commit, backend       │      backend_version read from the
        │  version, image digest,    │      RUNNING server (confirmed:
        │  dataset version, command  │      "vllm 0.19.0", "b5058-...")
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  + experiment_id          │  [✓]  Phase 2/3
        │  + standalone/co-resident │       no default permitted, AND
        │  + co-resident workload   │       checked against reality —
        │    declaration            │       see the guard box below
        └─────────────┬─────────────┘
                      │
                      │  PRE-FLIGHT, before anything expensive starts:
                      │  benchmarks/manifest.py:                    [✓]
                      │  assert_condition_matches_reality()
                      │    1. running_containers()      — Docker
                      │    2. gpu_holding_pids()         — bare-metal,
                      │       /proc/<pid>/fd nvhost/nvgpu/nvmap scan
                      │    3. co-resident completeness   — declared
                      │       workload must name everything (1)+(2) find
                      │  Three incidents on 2026-09-04 built these three
                      │  checks, in this order — see docs/TODO.md Phase 3
                      │
                      ▼
                      │  validated against
                      │  schemas/benchmark_result.schema.json [✓]
                      ▼
              results/raw/<experiment_id>/   [✓]  25 real results,
                                                    2026-09-04
              results/invalid/<experiment_id>/  [✓]  archive, not delete,
                                                    for anything caught
                                                    after the fact
                      │
                      ▼
              scripts/analyze.py  [ ]  Phase 10 — not started
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     tables        plots        Pareto frontier
                                (quality × latency ×
                                 memory × energy —
                                 not a weighted score)
```

Quality results take a parallel path and stay a separate document type
(`schemas/quality_result.schema.json` [~] — the one schema still level-2, see
`schemas/README.md`), joining performance results at analysis time on
`model_config_key`. `docs/note.md` §5 keeps the layers independently measurable;
§34 warns against collapsing them into one number too early.

```
   scripts/validate_tool_calling.py [✓]   BFCL simple + irrelevance
   scripts/validate_mmlu.py         [✓]   MMLU all/test
                    │
                    ▼
        quality_result documents  [~]
        score + n + dataset version + protocol
        + tool-call confusion matrix [ ]
```

## 4. Why this repo exists — the two consumers

A benchmark suite with no consumer drifts. This one has two, and they want
different things from the same data:

```
                    jetson-llm
                (this repo — the instrument)
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
  PROMOTION DECISION                 PAPER EVIDENCE
  docs/promotion-contract.md         embedded-ai-chain/docs/paper.md
                                     embedded-ai-chain/docs/PAPER_PLAN.md
  "which model should                
   embedded-ai-chain's               P3  generative extent dominates
   orchestrator actually run?"           → output-length sweep [✓] RUN
                                            2026-09-04: J/tok highest at
  weighted, application-                   out=16, settles lower by
  specific, allowed to trade               out>=128, both backends —
  axes against each other                  the predicted P3 shape, one
                                            replicate, not yet a claim
  → embedded-ai-chain/src/              P4  deciding to perceive has a cost
    orchestrator_models.py                 → energy + BFCL no-tool rate [ ]
                                       P5  hardware changes allocation
                                           → Orin vs Thor, same method [ ]
                                           (Orin half only so far — see
                                           §5's backend-scaling finding,
                                           a same-platform analogue)
                                       P6  memory is an allocation resource
                                           → co-resident runs, OOMs as
                                             results [ ]
```

The promotion scorecard is allowed to weight axes. The benchmark is not
(`docs/note.md` §35). Keeping them separate is why there are two consumers on this
diagram rather than one.

## 5. Real state of the candidate matrix, 2026-09-04

Not a plan — this is what has actually been run. Full record in `docs/TODO.md`
Phase 1.

```
                       vLLM                    llama.cpp
                 ┌────────────────────┬────────────────────────┐
  Qwen2.5-1.5B   │ ✓ serves           │ ✓ serves               │
                 │ ✓ tool calls       │ ✓ tool calls @ T=0.1   │
                 │   (temp-robust)    │   ✗ unreliable @ T~0.8 │
                 │                    │ ✓ BFCL 75% (n=40)      │
                 ├────────────────────┼────────────────────────┤
  Qwen2.5-3B/7B  │ not run            │ not run                │
                 ├────────────────────┼────────────────────────┤
  Bielik-11B     │ ✓ serves           │ ✓ serves               │
                 │ ✗ tool calls FAIL  │ ✓ tool calls, first try│
                 │   (hermes AND      │   ("Generic" grammar-  │
                 │    llama3_json)    │    constrained path)   │
                 ├────────────────────┼────────────────────────┤
  Apertus-8B     │ no row             │ ✗ BLOCKED — llama.cpp  │
                 │ (no trustworthy    │   doesn't know the     │
                 │  AWQ exists)       │   'apertus' arch, on   │
                 │                    │   two builds           │
                 └────────────────────┴────────────────────────┘
```

The two Bielik cells are the interesting result so far, and they point opposite ways
from Qwen2.5: vLLM's tool-call *detection* parsers find nothing to detect in a model
never trained to emit their tags, while llama.cpp's grammar-*constrained* path forces
valid structure regardless. Tool-calling reliability is a property of
(model × backend × parser × temperature), not of the model — which is why every one
of those four is a recorded field in `schemas/`.

## 6. Real Phase 4 campaign data, 2026-09-04

Not the candidate matrix above (that's Phase 1, tool-calling/serving) — this is
`output_sweep`/`context_sweep`, the performance sweeps, both for `1.5b-awq-vllm-orin`
and `1.5b-q4-llamacpp-orin` only, `standalone`, board confirmed quiet before and after
both campaigns (see docs/TODO.md Phase 3's incident record for what it took to be
able to say that with confidence). 20 cells, zero failures, zero errors.

**Output-length sweep — J/output-token amortizes, both backends:**

```
  output tokens        16     32     64    128    256    512
  vLLM J/tok         0.473  0.298  0.295  0.297  0.297  0.297
  llama.cpp J/tok     0.949  0.757  0.682  0.647  0.673  0.657
```

Highest at the shortest generation on both backends, settling by out≥128 — fixed
per-request overhead amortizing over more generated tokens. The predicted P3 shape
(§4), one replicate.

**Context-length sweep — a real backend-scaling difference, decode held flat:**

```
  input tokens         128    512   1024   1536
  vLLM TTFT ms         40.8   66.0   83.2   81.8
  llama.cpp TTFT ms   213.2  310.1  630.3  897.7
```

vLLM: close to flat (~2x over the full range). llama.cpp: clearly super-linear
(roughly doubles 512→1024 alone). Decode throughput stays close to constant on both
backends across the sweep — this is a *prefill*-scaling difference specifically, not
a general speed gap. Confounded by quantization format (AWQ vs GGUF Q4_K_M), same as
every cross-backend comparison in this lab — stated per docs/note.md §41, not fixed.

**Also surfaced, not yet a controlled finding**: llama.cpp's standalone decode here
(36-43 tok/s) is roughly double Phase 2's co-resident verification measurement at the
same config (21 tok/s). One data point against another — exactly the shape of result
Phase 9's standalone-vs-co-resident design exists to produce properly, not a claim on
its own yet.
