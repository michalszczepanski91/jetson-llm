#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Autonomous precision sweep — runs unattended, survives disconnection.
# ─────────────────────────────────────────────────────────────────────────────
# WHY THIS EXISTS: the first self-quantized INT8/FP8 checkpoints came back
# damaged (INT8 lost 23 points of MMLU; FP8 could not emit a single tool call
# in 50 attempts), while Qwen's own GPTQ-Int4 was essentially lossless. Two
# candidate causes were identified and are tested here:
#   1. FP8 was quantized WITH --kv_cache_quantization fp8, and the quantizer
#      itself warned "Large KV activations detected. Quantized KV cache may
#      lead to higher accuracy drop." -> retry without it.
#   2. Both used --num_samples 128; the tool's default is 512 and SmoothQuant
#      scales are calibration-sensitive -> retry at 512.
#
# NVIDIA lists NO pre-quantized FP8/INT8 checkpoint for Qwen2.5 (only AWQ and
# GPTQ-Int4, both already tested), and its docs direct users to self-quantize
# for FP8 - so fixing our own quantization is the only available path.
#
# DESIGN FOR UNATTENDED USE:
#   - Started with setsid+nohup, so it survives the terminal, the SSH session,
#     and the agent session that launched it.
#   - Everything it needs is already on local disk (checkpoints, datasets,
#     containers). No internet required for quantize/build/test - only the
#     final `git push` needs the network, and it is the LAST step and
#     non-fatal if it fails.
#   - Serialized on purpose: one GPU, and overlapping runs would contaminate
#     the timing measurements.
#   - Every step logs to output/window_logs/ and results land as JSON, so the
#     work is recoverable/inspectable even if nothing is watching.
#
# Usage:
#   setsid nohup bash scripts/autonomous_precision_sweep.sh \
#       > output/autonomous_sweep.log 2>&1 < /dev/null &
set -u
cd /home/michal/dev/jetson-llm || exit 1
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
L=output/window_logs
mkdir -p "$L"

say() { echo; echo "═════ $* ═════"; date '+       %Y-%m-%d %H:%M:%S'; }

# ── Wait for any in-flight quantization to finish ───────────────────────────
say "PHASE 0 — waiting for any running quantization to finish"
while pgrep -f "tensorrt_edgellm.scripts.quantize" > /dev/null; do sleep 30; done
echo "       no quantization running"

# ── Make sure nothing else holds the GPU ────────────────────────────────────
pkill -f "tensorrt-edgellm-serve" 2>/dev/null
sleep 8
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
if [ -n "$GPU_PROCS" ]; then
    echo "       WARNING: GPU not idle:"; echo "$GPU_PROCS" | sed 's/^/         /'
    echo "       continuing anyway - accuracy is unaffected by co-residency;"
    echo "       treat any TIMING number from this run as indicative only."
else
    echo "       GPU idle"
fi

# ── The rows to evaluate, in order ──────────────────────────────────────────
# Each must already exist as a row in configs/models.yaml.
ROWS="7b-fp8-nokv-selfquant-edgellm-thor 7b-int8sq-512-selfquant-edgellm-thor"

for CFG in $ROWS; do
    say "EVALUATING $CFG"

    # Skip cleanly if the checkpoint never got produced, rather than failing
    # the whole sweep - a missing checkpoint is a quantization problem, and
    # the other row is still worth measuring.
    CKPT=$(uv run python -c "
import sys; sys.path.insert(0,'src')
from model_config import load_model_config
try: print(load_model_config('$CFG')['model'])
except SystemExit: print('MISSING_ROW')
" 2>/dev/null | tail -1)
    if [ "$CKPT" = "MISSING_ROW" ] || [ ! -d "$CKPT" ]; then
        echo "       SKIP: checkpoint or row missing ($CKPT)"
        continue
    fi

    for STEP in "validate_tool_calling --limit 50:tool_calling:bfcl" \
                "validate_mmlu --limit 200:mmlu:mmlu" \
                "benchmark_streaming --runs 20:streaming:stream" \
                "benchmark --runs 30 --warmup-max 40 --max-tokens 32:benchmark:bench"; do
        SCRIPT="${STEP%%:*}"; REST="${STEP#*:}"; PREFIX="${REST%%:*}"; TAG="${REST#*:}"
        SUFFIX=""
        [ "$PREFIX" = "benchmark" ] && SUFFIX="_short32"
        echo "   → $SCRIPT"
        # shellcheck disable=SC2086
        uv run python "scripts/${SCRIPT%% *}.py" ${SCRIPT#* } \
            --model-config "$CFG" --ready-timeout 2400 \
            --results-json "output/${PREFIX}_${CFG}${SUFFIX}.json" \
            > "$L/${TAG}_${CFG}.log" 2>&1
        echo "     exit=$?"
        pkill -f "tensorrt-edgellm-serve" 2>/dev/null; sleep 5
    done
done

# ── Collate, record, commit ─────────────────────────────────────────────────
say "COLLATING"
uv run python scripts/collate_thor_results.py --pass clean > output/precision_sweep_results.txt 2>&1
cat output/precision_sweep_results.txt

say "COMMITTING"
git add -A
git commit -q -F - <<'COMMITMSG'
Autonomous precision sweep: FP8-without-KV-quant and INT8 at full calibration

Follow-up to the first self-quantized results, which came back damaged: INT8
lost 23 points of MMLU (67.5% -> 44.5%) and FP8 could not emit a single tool
call in 50 attempts (0% BFCL simple, and its "100% irrelevance" is therefore
meaningless rather than good), while Qwen's own GPTQ-Int4 was essentially
lossless (66.0% MMLU). That INT4 beat INT8 and FP8 is backwards from theory
and pointed at this lab's own quantization rather than at the precisions.

Two candidate causes tested, both suggested by the evidence rather than
guessed:
  1. FP8 had been quantized WITH --kv_cache_quantization fp8, and modelopt
     itself warned "Large KV activations detected. Quantized KV cache may
     lead to higher accuracy drop." Re-quantized without it.
  2. Both had used --num_samples 128 against the tool's own default of 512,
     and SmoothQuant's per-channel scales are calibration-sensitive.
     Re-quantized at 512.

Also confirmed against NVIDIA's own supported-models page for 0.10.1: there
is NO pre-quantized FP8 or INT8 checkpoint published for Qwen2.5 - only AWQ
and GPTQ-Int4, both already tested here - and the docs direct users to
self-quantize for FP8. So fixing this lab's own quantization was the only
available path, not a matter of finding a better checkpoint.

Run unattended via scripts/autonomous_precision_sweep.sh.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
COMMITMSG
echo "       commit: $(git log --oneline -1)"

# Push last and treat failure as non-fatal: everything above is already
# committed locally, and the network is the only part of this that depends on
# something outside this machine.
git push origin thor-edge-llm 2>&1 | tail -2 || echo "       push failed (no network?) - work is committed locally"

say "SWEEP COMPLETE"
