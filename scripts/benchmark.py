#!/usr/bin/env python3
"""RETIRED 2026-09-04 (docs/TODO.md Phase 2's ⟨DECIDE⟩, resolved).

This was the non-streaming latency/cold-start/thermal/power path through
benchmarks/harness.py. It predates the Phase 2 measurement-integrity
retrofit and had none of it applied: no raw per-run rows, no manifest, no
experiment ID, and - like the streaming script before its own fix - it sent
an identical prompt on every repetition, which the Phase 2 verification run
measured as a 6.6x TTFT error from prefix-cache reuse (docs/TODO.md Phase 2).

Retired rather than retrofitted: benchmarks/runner.py:measure_cell() (via
stream_llm()) already measures a strict superset of what this script did -
TTFT, decode time, AND end-to-end latency, plus energy and real token
counts, none of which this script ever had. Retrofitting it would have
produced a second, weaker copy of the same measurement, and this repo has
already spent one debugging session on two implementations of the same
thing drifting apart (docker-compose.yml vs VllmCoordinator on
--enable-auto-tool-choice).

Use instead:
    uv run python scripts/run_experiment.py configs/benchmarks/<name>.yaml
    uv run python scripts/benchmark_streaming.py --model-config <key> \\
        --execution-condition standalone|co-resident

benchmarks/harness.py itself is unaffected - it is a hand-maintained copy
shared with jetson-vlm-lab/embedded-ai-chain and is not this repo's to
retire. Only this script's use of it (via run_benchmark()) is retired.
"""

import sys

sys.exit(
    "scripts/benchmark.py is retired - see this file's own docstring "
    "(docs/TODO.md Phase 2). Use scripts/run_experiment.py or "
    "scripts/benchmark_streaming.py instead."
)
