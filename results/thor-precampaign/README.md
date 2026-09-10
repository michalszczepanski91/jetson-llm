# Thor campaign evidence — pre-pipeline, NOT schema-conformant

The 68 JSON files here are the raw evidence behind every Thor number in
`docs/thor-framework-comparison.md` and `docs/TODO.md` Phase 6/7/8. They are
committed so those numbers remain checkable if this Thor is ever reimaged.

**Read this before treating them like `results/raw/`:**

- They do **not** conform to `schemas/benchmark_result.schema.json` or
  `schemas/quality_result.schema.json`. They are the ad-hoc output of
  `scripts/thor_exclusive_window.sh` and friends, which predate this repo's
  measurement-integrity pipeline being wired up for Thor.
- They therefore carry **no `experiment_id`, no manifest, no git SHA, no
  `execution_condition` field**, and no per-repetition raw rows in the shape
  `results/raw/` guarantees. `scripts/analyze.py` (Phase 10) cannot read them.
- Filename is the only index. `_sharedbox` in a name means another user's
  container was resident — those timing numbers are contaminated and were
  superseded by the exclusive-box re-run; the accuracy numbers are unaffected
  by co-residency and stand.

**Why they exist in this shape at all**: the Thor campaign ran directly on the
box under time pressure during an exclusive window, driven by shell scripts
rather than `scripts/run_experiment.py`. That produced correct numbers and a
correct decision, but skipped the machinery Phase 2/3 built on the Orin side —
a real gap, recorded honestly in `docs/TODO.md` Phase 6 rather than papered
over.

**What replaces this**: re-running the campaign through
`scripts/run_experiment.py`, which since the Orin/Thor branch merge handles
`edge-llm` rows. Until that happens these files are the evidence of record, and
no Thor number should be published as if it came from the validated pipeline.
