#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Thor exclusive-window runbook — the numbers of RECORD.
# ─────────────────────────────────────────────────────────────────────────────
# Everything measured before 2026-09-08 17:00 was taken on a shared box (the
# other user's production vllm-vlm-thor container resident, ~37GB of the 122GB
# pool). Those results live in output/*_sharedbox.json and are indicative only.
# This script re-takes them with the box to itself, and adds the runs that
# could not be attempted at all under co-residency.
#
# It stops the other user's container at the start and ALWAYS restarts it at
# the end, including on failure or Ctrl-C (see the trap). That container has
# AutoRemove=false and RestartPolicy=unless-stopped, verified 2026-09-08, so
# stop/start preserves it exactly - it is NOT recreated from a command line
# here, which would risk dropping its patched-gpu_worker bind mount.
#
# Usage:  bash scripts/thor_exclusive_window.sh 2>&1 | tee output/window.log
set -u

REPO=/home/michal/dev/jetson-llm
LOGDIR="$REPO/output/window_logs"
OTHER_CONTAINER=vllm-vlm-thor
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

say() { echo; echo "───── $* ─────"; date '+      %H:%M:%S'; }

restore_other_container() {
    say "RESTORING $OTHER_CONTAINER"
    if docker ps -a --format '{{.Names}}' | grep -qx "$OTHER_CONTAINER"; then
        docker start "$OTHER_CONTAINER" >/dev/null 2>&1 \
            && echo "      started - verifying it answers /health..." \
            || echo "      !! docker start FAILED - tell the user immediately"
        for _ in $(seq 1 60); do
            if curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
                echo "      OK: $OTHER_CONTAINER is serving on :8000 again"; return 0
            fi
            sleep 5
        done
        echo "      !! it did not answer /health within 5min - NEEDS A HUMAN"
    else
        echo "      !! container is GONE from docker ps -a - NEEDS A HUMAN"
    fi
}
# Runs on normal exit, on error, and on Ctrl-C. The other user's production
# service must never be left down because our measurement run died.
trap restore_other_container EXIT INT TERM

# ── Phase 1: take the box ────────────────────────────────────────────────────
say "PHASE 1 — taking exclusive control"
docker stop "$OTHER_CONTAINER" >/dev/null 2>&1 && echo "      stopped $OTHER_CONTAINER"
pkill -f tensorrt-edgellm-serve 2>/dev/null
docker rm -f vllm-llm-lab llamacpp-llm-lab llamacpp-prefetch vllm-probe 2>/dev/null
sleep 10

# HARD GATE. Every number below is invalid if anything else is on the GPU, so
# refuse to measure rather than quietly produce contaminated results. The whole
# point of this window is that the box is ours.
echo "      containers up:"; docker ps --format '        {{.Names}}' | sed 's/^$/        (none)/'
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$GPU_PROCS" ]; then
    echo "      !! GPU IS NOT IDLE:"; echo "$GPU_PROCS" | sed 's/^/        /'
    echo "      !! ABORTING - measuring now would repeat the shared-box problem."
    echo "      !! Investigate, then re-run. (The trap still restores $OTHER_CONTAINER.)"
    exit 1
fi
echo "      GPU compute processes: none — box is ours"
echo "      memory now:"; free -g | sed -n 2p | sed 's/^/        /'
echo "      other user's container restart count at start: $(docker inspect "$OTHER_CONTAINER" --format '{{.RestartCount}}' 2>/dev/null)"

# ── Phase 2: the numbers of record, 1.5B across all three frameworks ─────────
# Same six runs as the shared-box pass, so the two are directly comparable.
say "PHASE 2 — 1.5B timing, all three frameworks"
for CFG in 1.5b-fp16-edgellm-thor 1.5b-fp16-vllm-thor 1.5b-fp16-llamacpp-thor; do
    say "  $CFG streaming (TTFT / tok-s)"
    uv run python scripts/benchmark_streaming.py --model-config "$CFG" \
        --runs 20 --ready-timeout 1800 \
        --results-json "output/streaming_${CFG}.json" > "$LOGDIR/stream_$CFG.log" 2>&1
    echo "      exit=$?"
    say "  $CFG latency / cold start / power"
    # warmup-max 40: on the shared box 25 was not always enough to reach
    # thermal steady state. Never report a run whose
    # warmup_reached_steady_state is false.
    uv run python scripts/benchmark.py --model-config "$CFG" \
        --runs 30 --warmup-max 40 --ready-timeout 1800 \
        --results-json "output/benchmark_${CFG}.json" > "$LOGDIR/bench_$CFG.log" 2>&1
    echo "      exit=$?"
    say "  $CFG orchestrator-shaped turn (32 tokens)"
    uv run python scripts/benchmark.py --model-config "$CFG" \
        --runs 30 --warmup-max 40 --max-tokens 32 --ready-timeout 1800 \
        --results-json "output/benchmark_${CFG}_short32.json" > "$LOGDIR/short32_$CFG.log" 2>&1
    echo "      exit=$?"
done

# ── Phase 3: does the framework gap survive at 7B? ───────────────────────────
# The open question from the shared-box pass: 7B lifted BFCL irrelevance from
# 78% to 94% on Edge-LLM. If capacity washes out the FRAMEWORK difference too,
# the choice can be made on operational grounds (llama.cpp: one image, no
# source build, unconditional 4s cold start) rather than on accuracy.
say "PHASE 3 — 7B across frameworks"
for CFG in 7b-fp16-edgellm-thor 7b-fp16-vllm-thor 7b-fp16-llamacpp-thor; do
    say "  $CFG BFCL"
    uv run python scripts/validate_tool_calling.py --model-config "$CFG" \
        --limit 50 --ready-timeout 3600 \
        --results-json "output/tool_calling_${CFG}.json" > "$LOGDIR/bfcl_$CFG.log" 2>&1
    echo "      exit=$?"
    say "  $CFG MMLU"
    uv run python scripts/validate_mmlu.py --model-config "$CFG" \
        --limit 200 --ready-timeout 3600 \
        --results-json "output/mmlu_${CFG}.json" > "$LOGDIR/mmlu_$CFG.log" 2>&1
    echo "      exit=$?"
    say "  $CFG streaming + latency + 32-token turn"
    uv run python scripts/benchmark_streaming.py --model-config "$CFG" \
        --runs 20 --ready-timeout 3600 \
        --results-json "output/streaming_${CFG}.json" > "$LOGDIR/stream_$CFG.log" 2>&1
    uv run python scripts/benchmark.py --model-config "$CFG" \
        --runs 30 --warmup-max 40 --ready-timeout 3600 \
        --results-json "output/benchmark_${CFG}.json" > "$LOGDIR/bench_$CFG.log" 2>&1
    uv run python scripts/benchmark.py --model-config "$CFG" \
        --runs 30 --warmup-max 40 --max-tokens 32 --ready-timeout 3600 \
        --results-json "output/benchmark_${CFG}_short32.json" > "$LOGDIR/short32_$CFG.log" 2>&1
    echo "      done"
done

# ── Phase 4: 14B — impossible under co-residency, possible alone ─────────────
# ~28GB of FP16 weights. With the other container's ~37GB resident this had no
# room; alone on a 122GB pool it should fit comfortably. This is the run that
# most directly answers "what does Thor's memory actually buy".
say "PHASE 4 — 14B on Edge-LLM (needs the box to itself)"
uv run python scripts/validate_tool_calling.py --model-config 14b-fp16-edgellm-thor \
    --limit 50 --ready-timeout 5400 \
    --results-json output/tool_calling_14b-fp16-edgellm-thor.json > "$LOGDIR/bfcl_14b.log" 2>&1
echo "      BFCL exit=$?"
uv run python scripts/validate_mmlu.py --model-config 14b-fp16-edgellm-thor \
    --limit 200 --ready-timeout 5400 \
    --results-json output/mmlu_14b-fp16-edgellm-thor.json > "$LOGDIR/mmlu_14b.log" 2>&1
echo "      MMLU exit=$?"
uv run python scripts/benchmark_streaming.py --model-config 14b-fp16-edgellm-thor \
    --runs 20 --ready-timeout 5400 \
    --results-json output/streaming_14b-fp16-edgellm-thor.json > "$LOGDIR/stream_14b.log" 2>&1
uv run python scripts/benchmark.py --model-config 14b-fp16-edgellm-thor \
    --runs 30 --warmup-max 40 --max-tokens 32 --ready-timeout 5400 \
    --results-json output/benchmark_14b-fp16-edgellm-thor_short32.json > "$LOGDIR/short32_14b.log" 2>&1
echo "      timing done"

say "ALL PHASES DONE — the trap now restores $OTHER_CONTAINER"
