# Orchestrator-LLM promotion decision — 2026-09-08

Applies `docs/promotion-contract.md`'s §3 scorecard and §4 winner-declaration process to
the Phase 5 data (`docs/TODO.md` Phase 5). **Outcome: no clean winner** — a legitimate
result per the contract's own §4, not a failed process.

## The scorecard (all 7 rows, standalone, MAXN + `jetson_clocks` locked)

| Row | TTFT p50 | decode tok/s | J/out-tok | BFCL simple | **BFCL irrelevance** | MMLU |
|---|---:|---:|---:|---:|---:|---:|
| `1.5b-awq-vllm-orin` **(current default)** | ~~31.1ms~~ **80.5ms** | 109.6 | ~~0.297~~ **0.329** | 80.0% | **36.7%** | 40.0% |
| `3b-awq-vllm-orin` | 132.5ms | 64.4 | 0.566 | 96.7% | **70.0%** | 54.0% |
| `7b-awq-vllm-orin` | 274.3ms | 33.5 | 1.224 | 96.7% | **40.0%** | 66.0% |
| `1.5b-q4-llamacpp-orin` | 309.5ms | 38.7 | 0.647 | 100.0% | **70.0%** | 41.5% |
| `3b-q4-llamacpp-orin` | 562.7ms | 22.2 | 1.178 | 100.0% | **46.7%** | 53.0% |
| `7b-q4-llamacpp-orin` | 1017.8ms | 11.5 | 2.347 | 100.0% | **56.7%** | 64.0% |
| `bielik-11b-q4-llamacpp-orin` | 1787.1ms | 7.4 | 4.293 | 100.0% | **10.0%** | 40.0% |

> **Correction, 2026-09-10 — the current-default row's latency and energy figures.**
> The `1.5b-awq-vllm-orin` row's TTFT and J/output-token were taken from `output_sweep`,
> whose vLLM cells were contaminated by cross-cell prefix-cache reuse (root cause and
> control in `docs/HISTORY.md`, "Phase 4 — prompt-salt contamination"). Corrected by
> re-measurement at `--replicate 2` under the fix: **TTFT 31.1 → 80.5ms**, **J/out-tok
> 0.297 → 0.329**, decode 108.2 → 109.6 tok/s (decode was never exposed). **No other row
> is affected** — every other row's performance figures came from `scorecard.yaml`, which
> is single-cell and so never had a second cell to reuse. No BFCL or MMLU number changes.
>
> **This does change the argument below, and §"Why this isn't a simple pick" is now
> partly wrong as written.** It says `3b-awq-vllm-orin` costs "4.3x the TTFT" of the
> current default. Against the corrected baseline it costs **1.65×** (132.5 vs 80.5ms),
> a difference of ~52ms on a turn that misses its budget by ~60ms. That does not by
> itself make 3B affordable — its decode is still 1.7× slower per token, and decode
> dominates a two-call round trip — but "dramatically slower on every latency axis" is
> no longer a fair description of the 1.5B→3B step, and the recommendation was written
> on the un-corrected number. See "Reopened" at the end of this file.

> **Caveat on the whole MMLU column, added 2026-09-11.** Every MMLU figure in this table
> is an `--limit 200` run, and `validate_mmlu.py` takes `all_rows[:limit]` from a split
> that is **ordered by subject**. Verified on this board's staged copy: the first 200 rows
> are 100 `abstract_algebra` + 100 `anatomy` — **2 of MMLU's 57 subjects**, one of them
> among its hardest. These are therefore *not* MMLU accuracies and must not be quoted as
> such anywhere, this document included.
>
> **What survives is the ordering, and only the ordering.** The first version of this
> caveat said the comparison stays "controlled" because every row saw the identical 200
> questions, and that was too generous — the Thor session then measured the point
> directly, re-running one 7B pair three ways:
>
> | sample | FP16 | INT8-SQ | gap |
> |---|---:|---:|---:|
> | n=200 first-n (2 of 57 subjects) | 67.5% | 54.0% | 13.5 pt |
> | n=1000 first-n (8 of 57 subjects) | 74.0% | 61.4% | 12.6 pt |
> | **n=1000 random, seed 1234 (all 57)** | 70.8% | 63.0% | **7.8 pt** |
>
> A controlled comparison on an unrepresentative slice yields a controlled but **biased
> effect size**, because quantization damage is subject-dependent. The slice overstated
> that gap by ~60%. So: **rankings hold, magnitudes do not.** Every difference in this
> table is an upper bound of unknown tightness, not a measured gap.
>
> That kills more than absolute readings — it kills this document's own
> "sane scaling from the 1.5B row's 40.0%" reasoning below, which argues from the *size*
> of a 14-point step to conclude the quantization is not broken. On these two subjects
> that step's magnitude is not interpretable; only "3B scores higher than 1.5B" is. Read
> it as an ordering claim.
>
> Anywhere this lab reasons from the *size* of an MMLU gap, the number needs re-measuring
> with `--sample random`, not caveating.

BFCL scores are the 2026-09-09 re-run under the corrected scorer (see `docs/TODO.md`
Phase 5's "BFCL scorer fix"). **`simple` is saturated — five rows at exactly 100% — so
`irrelevance` is the only accuracy axis that still discriminates**, which is fortunate,
because it is also the one that maps to the production failure. One case is 3.3 points
at n=30, so `irrelevance` gaps under ~7pp are within sampling noise. The scorer fix
raised every row and **changed no rankings**, so nothing below turns on it.

## Why this isn't a simple "pick the best BFCL row" call

`embedded-ai-chain/docs/TODO.md` §3h already measured the **current shipping default**
end-to-end, live, with the real tool-calling round trip (forced `read_scene_state` +
narration, two LLM calls per turn) — not a single-completion benchmark like the table
above:

> `end_to_end` p50 **1.06s**, p95 **1.97s** — still at/over the ≤1000ms budget... the
> `orchestrator` stage's own min (0.226s) confirms even the fastest possible turn spends
> a real fraction of the budget on LLM round-trip overhead alone.

**The current default already misses its own latency target**, and that miss is carried
into Phase 4 as an explicitly open question (risk register) — not something this repo's
model comparison can quietly resolve by itself.

Every row in the table above with better tool-calling accuracy than the current default
is also **slower** on every latency axis, in most cases dramatically so:

- `3b-awq-vllm-orin` (best accuracy/latency trade of the alternatives): 4.3x the TTFT,
  ~1.7x the decode time per token. Given the round trip is two calls, this pushes an
  already-over-budget number further over, not closer.
- `7b-awq-vllm-orin`, every `llama-cpp` row, and `bielik-11b-q4-llamacpp-orin`: all
  strictly worse on both TTFT and decode speed than even `3b-awq-vllm-orin` — disqualified
  on latency alone, regardless of any accuracy strength. `bielik-11b-q4-llamacpp-orin` in
  particular also has this campaign's *worst* tool-calling judgment (a 10%
  irrelevance-abstain rate — it calls a tool on 90% of cases where none applies) despite
  a perfect 100% on `simple`, so it is not even a live accuracy contender on its own
  terms. It is the clearest single illustration of why `simple` must not be read alone.

So promoting any candidate on accuracy grounds alone, without a separate resolution to
the open latency question, would trade a known, documented failure mode for a novel,
still-worse one.

## What this data does add: a real number for an already-known risk

The risk register already flagged (2026-07-31, live-tested) that the current default
"does not reliably escalate to `ask_vlm` for questions that genuinely need it," discovered
anecdotally (a handful of manually-tried phrasings). **BFCL's irrelevance category is
the same failure mode, now quantified**: `1.5b-awq-vllm-orin` incorrectly calls a tool
on 63.3% of cases where none applies (100% − 36.7% correct-abstain). This is the first
hard measurement of that risk's real frequency, not just its existence.

## Recommendation

1. **Do not swap the shipping default from this data alone.** The blocking constraint is
   latency, and every accuracy-improving alternative measured here makes the already-open
   latency problem worse. Resolving that is Phase 4's decision to make (accept a revised
   budget for tool-calling turns / optimize the round-trip / explore a faster serving
   config — `docs/TODO.md`'s risk register lists all three options), not a side effect of
   picking a different model.
2. **If Phase 4 accepts a revised latency budget for tool-calling turns**,
   `3b-awq-vllm-orin` is the strongest candidate measured: best irrelevance-abstain rate
   among any row that isn't already disqualified on latency (70.0%, vs the current
   default's 36.7% — a 33-point gap, well outside the ~7pp noise band), and no MMLU
   red flag (54.0%, i.e. it *ranks above* the 1.5B row's 40.0% — but see the MMLU caveat
   above: on a 2-subject slice the ordering is interpretable and the 14-point size of the
   step is not, so this supports "not obviously broken", not "sanely scaled"). Promoting it would
   still require the co-resident/end-to-end measurement §3 of the contract calls for
   (everything in this table is `standalone` — Phase 5 did not repeat the co-resident run
   the current default was measured under) before being declared final per §4.
3. **The 63.3% irrelevance false-positive rate on the current default is worth acting on
   independently of any model swap** — it's a quantified, real production risk
   (unwanted actuation) that exists today, regardless of which promotion path is chosen.

## Record

Per contract §4.3, a pointer row is added to `embedded-ai-chain/docs/TODO.md`'s decision
log referencing this document.

## Reopened, 2026-09-10

Two things surfaced after this decision was recorded. Neither is settled here; both
belong to the still-open latency question this document defers to Phase 4.

1. **The baseline was wrong** (see the correction above). The gap between the shipping
   default and `3b-awq-vllm-orin` on TTFT is 1.65×, not 4.3×.

2. **A precision arm was measured and never entered the record.** A BFCL run for
   `1.5b-fp16-vllm-orin` exists, dated 2026-09-09, and scores **66.7% irrelevance**
   against the shipping AWQ twin's 36.7% — same model, same backend, same board, the
   only variable being AWQ int4 vs fp16. It was never committed: the registry rows lived
   only in a `git stash` and the result document sat untracked, so its `model_config_key`
   did not resolve and no analysis could have found it. Recovered and committed
   2026-09-10.

   | row | BFCL irrelevance | false positives (of 30) | MMLU |
   |---|---:|---:|---:|
   | `1.5b-awq-vllm-orin` (shipping) | 36.7% | 19 | 40.0% |
   | `1.5b-fp16-vllm-orin` | **66.7%** | **10** | not yet measured |
   | `3b-awq-vllm-orin` | 70.0% | 9 | 54.0% |

   **`1.5b-fp16` and `3b-awq` are one case apart on the axis that decides this
   comparison** — inside the ~7pp noise band this document defines — while `1.5b-fp16`
   is the same 1.5B model the pipeline already runs. That is a direct challenge to this
   document's framing that "every accuracy-improving alternative is slower": the
   alternative that may not be slower was never on the table.

   **Measured 2026-09-10 (`configs/benchmarks/precision_arm.yaml`, same-day matched
   pair, one cell per model, v2 salt) — and it is not affordable.** The suspicion that
   fp16's ~4× weight traffic would hurt a bandwidth-bound decode was right, and by more
   than enough to settle it:

   | row | TTFT p50 | decode tok/s | J/out-tok | BFCL irrelevance |
   |---|---:|---:|---:|---:|
   | `1.5b-awq-vllm-orin` (shipping) | 80.3ms | 109.3 | 0.322 | 36.7% |
   | `1.5b-fp16-vllm-orin` | 95.6ms | **48.7** | 0.661 | 66.7% |
   | `3b-awq-vllm-orin` | 132.5ms | 64.4 | 0.566 | 70.0% |

   **`1.5b-fp16-vllm-orin` is dominated by `3b-awq-vllm-orin`** — worse decode (48.7 vs
   64.4 tok/s), worse energy (0.661 vs 0.566 J/token) and no better judgment (66.7% vs
   70.0%, one case apart). Its only advantage is TTFT, the smaller term. Dropping AWQ
   costs more than doubling the parameter count and keeping it.

   So the precision arm closes as a **negative result, and a useful one**: it identifies
   AWQ int4 as *a real cause* of the shipping default's judgment failure - 36.7% → 66.7%
   with model, backend, board and prompt all held fixed - while ruling fp16 out as the
   *remedy* on this board. The mechanism and the fix are different questions and this
   separates them. MMLU for the fp16 row is still unmeasured; it would sharpen the causal
   story but cannot revive fp16 as a candidate.

## Overturned in part, 2026-09-11 — recommendation 2 does not survive production measurement

`embedded-ai-chain/docs/round-trip-measurement.md` ran this repo's two leading Orin rows
through the **real** `LlmOrchestrator.handle_transcript()` — production tool loop,
production prompt, production `_FORCE_ASK_VLM` gating — rather than through BFCL. On
turns where no tool is needed:

| | BFCL `irrelevance` (this repo) | spurious tool calls, 40 real turns |
|---|---:|---:|
| `1.5b-awq-vllm-orin` (shipping) | 36.7% correct-abstain | **0 / 40** |
| `3b-awq-vllm-orin` (recommended above) | 70.0% correct-abstain | **21 / 40** |

**The ranking reverses on the exact failure this scorecard exists to predict.** Not a
single-phrase artifact: 3B fires `ask_vlm` on 4–5 of 5 samples for four of five
`needs_none` phrases and leaks into chit-chat, while 1.5B is clean on every phrase. And
3B costs 2.6× the turn latency (633ms p50 vs 241ms), most of it from making 1.62 calls
per turn against 1.14 — not from the +104ms of TTFT this document's corrected table shows.

So **recommendation 2 above should not be acted on**: promoting `3b-awq-vllm-orin` on the
strength of its `irrelevance` score would roughly double unwanted VLM actuation while
also costing latency.

### The methodological finding, which outlives this particular pick

`docs/promotion-contract.md` §3 elevates BFCL `irrelevance` as "arguably the more
important number for this lab specifically", on the grounds that it maps to the
production `ask_vlm` escalation failure. **It does not map to it — here it inverts it.**
Two reasons, both structural rather than incidental:

- **BFCL measures the model naked.** Production wraps it in `_grounding_context()`, which
  supplies the scene data with "already up to date — no need to call `read_scene_state`
  again". That prompt alone suppresses the spurious-call failure completely for 1.5B. So
  §"What this data does add" above is **overstated**: 63.3% is a real property of the
  model, not "the first hard measurement of that risk's real frequency" in production.
- **The tool set differs.** BFCL offers one generic tool; production offers a cheap one
  (`read_scene_state`) beside an expensive actuating one (`ask_vlm`). All 21 of 3B's
  spurious calls were `ask_vlm`. "Calls a tool when it shouldn't" is not one failure mode
  — which tool it reaches for is most of the cost.

This does not make BFCL worthless here: it remains a real, external, comparable benchmark
and it is still the right instrument for comparing candidates *to each other* on raw
judgment. What it cannot do is stand in for the production risk, and the contract's §3
should say so. **A candidate should not be promoted on `irrelevance` alone without a
round-trip measurement against the real orchestrator** — which is cheap, is now scripted
(`embedded-ai-chain/benchmarks/measure_round_trip.py`), and would have caught this.

One thing the production run does confirm in 3B's favour: it escalated on 10/10 turns
that genuinely needed the VLM, 5/5 of them unforced, where 1.5B managed 2 of its 5
unforced. 3B is genuinely better on the false-*negative* axis and would allow
`_FORCE_ASK_VLM` to be retired. That is a real trade, not a tie-break — and per §4 it is
the parent repo's decision to make, not this document's.
