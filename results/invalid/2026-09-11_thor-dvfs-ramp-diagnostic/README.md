# 2026-09-11 — Thor vLLM p1-context r01: invalidated by a CPU DVFS ramp

**What is here.** The first Priority-1 run of `7b-fp16-vllm-thor-v2`, stopped
after its second cell. It is kept because it is the *evidence* for a
measurement-validity finding, not because its numbers are usable.

**What went wrong.** The run warmed up with a fixed 5 requests, per the task
brief's "≥5 warm-up requests discarded". Five is not enough on this board.
Decode rate inside the very first cell (in≈128, out=128, 30 repetitions,
identical prompt, prefix caching verified off) reads:

```
9.43 9.45 9.46 9.52 9.54 9.37 9.55 9.53 9.56 10.50 13.33 13.00 12.49 13.26
12.91 12.90 12.97 12.72 12.89 12.69 12.68 12.65 13.17 13.22 13.32 13.14
12.77 12.67 13.14 12.50     (tokens/s)
```

That is not noise and it is not a thermal decay — it is a **step**, at
repetition 10, of **+39%**, after roughly two minutes of sustained load, and
the rate is flat on both sides of it. The next cell's warm-ups start at 13.1
tok/s, so the step is paid once per server session, not once per cell.

**Cause.** `/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor` is
`schedutil`, ranging 972 MHz → 2601 MHz, while the GPU GPC clock is already
pinned at its 1386 MHz maximum (`/sys/class/devfreq/gpu-gpc-0`). During the
run `tegrastats` shows two CPU cores at 2601 MHz and the rest at 972 MHz.
vLLM's batch-1 decode is partly CPU-bound — per-token scheduling and sampling
in Python — so the governor's promotion of the busy core to its top bin is
worth 39% of the decode rate, and the governor takes about two minutes of
sustained load to get there.

**Why it matters beyond this run.** A p50 taken across that step is a
number from neither regime. `results/thor-precampaign/streaming_7b-fp16-vllm-thor.json`
reports 11.31 tok/s from 20 repetitions — which sits between the two plateaus
here (9.5 and 13.0), exactly where a 20-run measurement that straddles the
step would land. That number, and any cross-framework comparison resting on
it, should be re-taken under the steady-state warm-up rather than trusted.

**The fix.** `scripts/measure_run.py` now warms up to a *measured* steady
state — a minimum wall-clock duration under load plus a coefficient-of-
variation check on recent decode rates — and records the warm-up trajectory
and whether steady state was actually reached, instead of trusting a count.
See that file's `warm_up()`.

**Not deleted, not corrected in place.** Per `results/invalid/README.md`, a
result that cannot be honestly corrected after the fact is archived with a
note on what was actually true. Correcting it in place is not possible: the
contaminated quantity is the measurement itself, not a label on it.
