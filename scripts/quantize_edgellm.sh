#!/usr/bin/env bash
# Reproduce this lab's self-quantized Edge-LLM checkpoints (Thor only).
#
# WHEN YOU NEED THIS - and when you do NOT:
#
#   You do NOT need this script for the recommended configuration. The
#   recommendation (docs/thor-framework-comparison.md) is FP16 or **GPTQ-Int4**,
#   and the Int4 one is `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` - Qwen's OWN
#   published checkpoint, downloaded like any other model. Reproducing the
#   recommendation needs a `hf download` and the one-line Edge-LLM patch in
#   `patches/`, nothing more.
#
#   You need this script only for FP8 and INT8-SmoothQuant, because **no
#   published checkpoint exists for this model in either format** (checked
#   against NVIDIA's 0.10.1 supported-models page, 2026-09-09). Neither is
#   currently recommended - FP8 is accuracy-clean but only ~1.4x FP16
#   throughput, and INT8-SQ still carries an unexplained 12.5-point MMLU gap.
#   They exist so the precision sweep covers the space, not because they won.
#
# WHY THIS FILE EXISTS AT ALL: the recipes below were originally run as ad-hoc
# shell commands and recorded only in prose. That made the four checkpoints in
# /home/michal/dev/quantize-work irreproducible the moment that directory went
# away - a real gap for a lab whose whole point is reproducibility. The exact
# flags are here now.
#
# PREREQUISITES: Edge-LLM built from source with its venv active, and modelopt
# 0.45.0+ in that venv. The container ships 0.39.0, whose `quant_cfg` is a dict
# where Edge-LLM's code expects a list; install with `--no-deps` so it cannot
# pull a generic PyPI torch over the platform-specific one (the wheel-shadowing
# hazard embedded-ai-chain/docs/environment.md warns about).
#
# Usage:
#   scripts/quantize_edgellm.sh fp8       /path/to/Qwen2.5-7B-Instruct /out/dir
#   scripts/quantize_edgellm.sh int8_sq   /path/to/Qwen2.5-7B-Instruct /out/dir

set -euo pipefail

RECIPE="${1:?usage: $0 <fp8|int8_sq> <model_dir> <output_dir>}"
MODEL_DIR="${2:?missing model_dir}"
OUTPUT_DIR="${3:?missing output_dir}"

# 512 is the quantizer's OWN default. Recorded explicitly because lowering it
# was a real, measured mistake: the first INT8-SQ run used 128 (reduced only to
# save time) and lost 23 points of MMLU. Raising it back to 512 recovered most
# of that (44.5% -> 54.0%). SmoothQuant derives per-channel activation scales
# from calibration, so starving it is not a free speedup.
NUM_SAMPLES="${NUM_SAMPLES:-512}"

case "$RECIPE" in
  fp8)
    # NOTE THE ABSENCE of --kv_cache_quantization. That is the whole fix, not an
    # oversight: the first FP8 attempt passed `--kv_cache_quantization fp8` and
    # produced a checkpoint that emitted ZERO tool calls in 50 tries. It scored
    # a meaningless "100% irrelevance" - it wasn't abstaining, it was mute.
    # The quantizer's own calibration log had warned about this. Dropping the
    # flag fixed it completely: 0% -> 91% BFCL simple, 46.5% -> 66.5% MMLU.
    exec tensorrt-edgellm-quantize llm \
      --model_dir "$MODEL_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --quantization fp8 \
      --text_dataset cnn_dailymail \
      --num_samples "$NUM_SAMPLES"
    ;;
  int8_sq)
    # W8A8 SmoothQuant, per-channel. Produces a usable-but-not-trusted
    # checkpoint: 92% BFCL irrelevance but MMLU 54.0% against FP16's 67.5%.
    # Do not ship this without closing that gap - see the comparison doc's
    # "Still open" for the three things to try next.
    exec tensorrt-edgellm-quantize llm \
      --model_dir "$MODEL_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --quantization int8_sq \
      --text_dataset cnn_dailymail \
      --num_samples "$NUM_SAMPLES"
    ;;
  *)
    echo "error: unknown recipe '$RECIPE' (expected fp8 or int8_sq)" >&2
    exit 2
    ;;
esac

# AFTER QUANTIZING, ALWAYS RE-RUN THE FULL ACCURACY SUITE. A quantization run
# that completes without error is NOT evidence the checkpoint is usable - both
# first attempts above exited 0 and were damaged. Only running the same BFCL +
# MMLU every other candidate went through caught it:
#
#   uv run python scripts/validate_tool_calling.py --model-config <row> \
#       --execution-condition standalone --limit 50
#   uv run python scripts/validate_mmlu.py --model-config <row> \
#       --execution-condition standalone
#
# Register the result as its own configs/models.yaml row with `served_model_name`
# set - a locally-served checkpoint directory registers under its BASENAME, not
# the path it was launched with, and omitting it 404s every request.
