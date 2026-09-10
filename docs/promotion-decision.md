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
   red flag (54.0% — sane
   scaling from the 1.5B row's 40.0%, not a broken quantization). Promoting it would
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

   It is **not** a recommendation yet. `1.5b-fp16-vllm-orin` has no latency, energy or
   MMLU measurement at all, and fp16 weights are ~4× AWQ int4's, which on a
   bandwidth-bound decode is the axis most likely to hurt. Measuring it is the cheapest
   input to the latency decision and should happen before that decision is made.
