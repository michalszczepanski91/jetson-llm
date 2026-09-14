# 2026-09-11 — Thor vLLM p1-context r02: stopped, warm-up criterion unreachable

**What is here.** The second Priority-1 run of `7b-fp16-vllm-thor-v2`, stopped
during its first cell. No cell completed, so there are no measurements to
invalidate — this directory exists for the warm-up trajectory, which is the
evidence for a calibration fix.

**What went wrong.** The steady-state warm-up added after
`2026-09-11_thor-dvfs-ramp-diagnostic` used a coefficient-of-variation test:
stop once the last six decode rates span no more than 3% of their mean. On
this board that threshold is below the process noise, so it can never be met.
Twenty-two warm-up repetitions produced:

```
12.12 11.58 11.14 11.21 11.08 11.19 11.35 11.10 11.95 12.16 12.03 11.60
11.33 11.32 11.44 11.47 11.86 12.30 12.28 12.21 10.71 11.29   (tokens/s)
```

with the rolling six-sample range running 2.4%–13.5% of the mean and no
downward trend. The cell would have run to its 80-repetition / 480-second cap
and then flagged itself `reached_steady_state: False` — for every cell, of
every framework, forever.

**The deeper problem with a spread test.** It has to sit below the process
noise to mean anything, and a threshold that loose enough to be reachable
here (say 15%) would happily have accepted the *pre-step* plateau of the r01
diagnostic, which was extremely stable at 9.5 tok/s — the exact regime this
warm-up exists to leave.

**The fix.** A trend test instead of a spread test: the mean of the last six
repetitions against the mean of the six before them, steady when they agree
within 5%. Noise cancels between the two halves; a regime change does not.
Against the r01 data the +39% step is 30% of the mean and would be rejected;
against this run's wander the two half-means agree to about 1% and it would
be accepted within a dozen repetitions. The minimum-duration condition stays,
because a trend test alone would also have passed on r01's flat pre-step
plateau.

**Also note for the record**: this session never showed r01's clean step at
all. Its decode rate wandered between 10.7 and 12.3 tok/s for the whole
warm-up. So the 39% step is a *first-load* effect of a session, not something
every session pays, while a persistent ±7% run-to-run wander is normal on this
board and is what the reported confidence intervals must reflect.
