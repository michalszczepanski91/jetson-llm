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

Renamed from `progect.diagram.md` on 2026-09-04 (typo, never-committed file).

---

## 1. Pipeline — config in, results out

```
   configs/models.yaml [✓]              configs/benchmarks/*.yaml [ ]
   "WHICH candidate?"                   "WHAT experiment?"
   model x precision x backend          input/output token grids,
   x platform, append-only,             batch/concurrency, sampling,
   9 rows today                         warmup + repetition floors
          │                                        │
          │  validated by                          │  validated by
          │  schemas/model.schema.json [✓]         │  schemas/experiment.schema.json [~]
          │                                        │
          └────────────────┬───────────────────────┘
                           │
                           │  one (candidate, workload point) pair = one cell
                           ▼
              ┌─────────────────────────────┐
              │      BENCHMARK RUNNER       │
              │                             │
              │  src/llm_coordinator.py [✓] │  container lifecycle
              │  src/llm_client.py      [✓] │  OpenAI-compatible round trip
              │  benchmarks/harness.py  [✓] │  timing, warmup, telemetry
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
```

## 2. What a run measures — two paths that must merge

This is the most important structural fact in the repo today, and the reason
`docs/TODO.md` Phase 2 exists:

```
  scripts/benchmark.py [✓]              scripts/benchmark_streaming.py [✓]
  non-streaming, one number             streaming
          │                                        │
          ▼                                        ▼
  ┌───────────────────┐                  ┌───────────────────┐
  │ harness.py        │                  │ per-run TTFT and  │
  │  · cold start     │                  │ tokens/sec        │
  │  · latency p50/95/99                 │                   │
  │  · tegrastats:    │                  │  NO tegrastats    │
  │      power rails  │                  │  sampler at all   │
  │      thermal      │                  │                   │
  │      system RAM   │                  │                   │
  │  · nvpmodel,      │                  │                   │
  │    jetson_clocks, │                  │                   │
  │    L4T            │                  │                   │
  └────────┬──────────┘                  └─────────┬─────────┘
           │                                       │
     knows POWER                             knows TOKENS
     but not tokens                          but not power
           │                                       │
           └───────────────  ✗  ──────────────────┘
                             │
              J / generated token is currently
              UNCOMPUTABLE from either script.
              docs/note.md §14 calls it the headline
              metric; embedded-ai-chain/docs/paper.md
              §7.1 builds Semantic Decision Energy on it.

              Phase 2 [ ] merges these two paths.
```

## 3. Result lifecycle — where a number goes

```
              one cell runs
                    │
                    ▼
        ┌───────────────────────────┐
        │  RAW per-run measurements │  [ ]  Phase 2
        │  every repetition, kept   │       today: harness computes
        │  ttft / e2e / decode /    │       percentiles(latencies)
        │  tokens / finish_reason   │       and DISCARDS latencies
        └─────────────┬─────────────┘
                      │
                      ├──────────────► aggregates  (p50/p95/p99, never a mean)
                      │                 regenerable, never the only copy
                      │
                      ▼
        ┌───────────────────────────┐
        │  + manifest               │  [ ]  Phase 2
        │  git commit, backend       │      git SHA, backend version read
        │  version, image digest,    │      from the RUNNING server, image
        │  dataset version, command  │      digest, dataset version
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  + experiment_id          │  [ ]  Phase 2
        │  + standalone/co-resident │      no default permitted
        └─────────────┬─────────────┘
                      │
                      │  validated against
                      │  schemas/benchmark_result.schema.json [~]
                      ▼
              results/raw/<experiment_id>/   [ ]
                      │
                      ▼
              scripts/analyze.py  [ ]  Phase 10
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     tables        plots        Pareto frontier
                                (quality × latency ×
                                 memory × energy —
                                 not a weighted score)
```

Quality results take a parallel path and stay a separate document type
(`schemas/quality_result.schema.json` [~]), joining performance results at analysis
time on `model_config_key`. `docs/note.md` §5 keeps the layers independently
measurable; §34 warns against collapsing them into one number too early.

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
   orchestrator actually run?"           → output-length sweep [ ]
                                     P4  deciding to perceive has a cost
  weighted, application-               → energy + BFCL no-tool rate [ ]
  specific, allowed to trade        P5  hardware changes allocation
  axes against each other               → Orin vs Thor, same method [ ]
                                     P6  memory is an allocation resource
  → embedded-ai-chain/src/              → co-resident runs, OOMs as
    orchestrator_models.py                 results [ ]
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
