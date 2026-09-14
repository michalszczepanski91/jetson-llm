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
| `2026-09-14_thor-llamacpp-corpus-tokenizer-mismatch/` | `TokenCounter` posted `prompt` to `/tokenize`; **llama-server reads `content`**, so it tokenised the empty string and returned a well-formed `{"tokens": []}`. Every corpus entry recorded `reference_tokens: 0`. Separately, the row keyed its corpus by its own `-GGUF` model id and so built a **second** corpus — the prompts were not byte-identical across backends, which is the one property the three-backend comparison is built to guarantee. | `/tokenize` bodies carry both spellings; a zero count for non-empty text demotes to the `usage` fallback instead of being believed; `build_corpus` raises on a miss larger than `max(4, target/4)`; and a new `prompt_corpus_reference` registry field lets a row name the corpus it shares. `tests/test_prompt_corpus.py`. |

**A note on these three, because they are a different species from everything above
them.** The other runs archived here were *mislabelled*: the measurement was fine and
the document lied about the conditions, which is the class
`assert_condition_matches_reality()` exists to catch.

These three were labelled truthfully. The two 2026-09-11 entries were **badly
measured** — the label was true and the number was not — and the check that catches
that class is a cell reporting how its own warm-up ended, which every v2 result now
does. The 2026-09-14 entry is a third kind again: correctly executed, correctly
labelled, and **incomparable**, because what broke was the object that makes two runs
mean the same thing. No condition guard reaches any of them. What reaches the third is
a corpus builder that refuses to emit entries it could not converge, and a registry
that lets every leg of a comparison name one shared corpus.

All three are kept for the same reason: the evidence is worth more than the disk space,
and a defect is easier to believe with the trajectory that produced it sitting next to
the write-up.
