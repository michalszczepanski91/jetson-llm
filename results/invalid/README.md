# results/invalid/

Runs discovered *after the fact* to have been mislabeled or otherwise measured under
conditions that don't match what their document claims - kept here rather than
deleted.

**Why this directory exists.** The first time this happened (2026-09-04, a
`standalone` `output_sweep` campaign that was actually co-resident with
`embedded-ai-chain`'s full production pipeline running bare-metal - see
`docs/TODO.md`'s Phase 3 incident record), the contaminated results were deleted
outright. That was the wrong call: `docs/note.md`'s own principle is to never
silently discard data, and a contaminated run is still evidence of something (at
minimum, of the contamination itself) even when it can't be trusted as a clean
measurement. Corrected 2026-09-04, by direct user instruction: archive, don't delete.

**What goes here vs. `results/raw/`:**

- `results/raw/<experiment_id>/` — results whose `execution_condition` (and every
  other schema-required field) is believed accurate at write time. This is what
  `scripts/run_experiment.py`/`scripts/benchmark_streaming.py` write by default, and
  what Phase 10's analysis pipeline reads from.
- `results/invalid/<experiment_id>/` — results moved here after a labeling problem is
  discovered. Never analyzed as if they were `results/raw/` data. Each moved result
  should get a sibling `NOTES.md` (or an entry in this README) stating what was
  actually true and how it was discovered.

**Not a substitute for prevention.** `benchmarks/manifest.py:assert_condition_matches_reality()`
is the real fix — it refuses to let a `standalone` claim be written in the first
place when either Docker containers or bare-metal GPU-holding processes are found.
This directory is the fallback for whatever that check doesn't yet catch, not a
license to skip fixing the check itself.
