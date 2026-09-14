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

---

## Index

Each entry names what was actually true and what fixed it.

| Directory | What was wrong | Prevention that followed |
|---|---|---|
| `2026-09-11_thor-dvfs-ramp-diagnostic/` | Warmed up for a fixed 5 requests. Decode rate held 9.5 tok/s for nine repetitions, then **stepped +39% to 13.0** about two minutes into sustained load — the CPU governor (`schedutil`, 972→2601 MHz) promoting the core vLLM's batch-1 decode loop runs on. A p50 across that step belongs to neither regime. | `scripts/measure_run.py:warm_up()` — warm up to a *measured* steady state, and record `warmup_seconds_to_steady_state` and whether it was reached. |
| `2026-09-11_thor-unreachable-stability-threshold/` | The first version of that fix used a 3% coefficient-of-variation test, which sits **below this board's own run-to-run decode spread** (5–13% over six samples) and so could never pass — every cell would have run to its cap and self-flagged. No cell completed. | Same function, trend test instead of spread test: mean of the last six repetitions against the six before them. Noise cancels between the halves; a regime change does not. |

**A note on the second entry, because it is a different species from everything
above it.** The other runs archived here were *mislabelled* — the measurement
was fine and the document lied about the conditions. These two were *badly
measured*: the label was true and the number was not. Both belong here for the
same reason, which is that the evidence is worth more than the disk space, but
`assert_condition_matches_reality()` cannot catch either one. The check that
catches this class is the one that reports how a warm-up ended, and it is now
in every cell of every v2 result.
