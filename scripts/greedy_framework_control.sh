#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  The sampling-controlled framework comparison.
# ─────────────────────────────────────────────────────────────────────────────
# The 2026-09-08 campaign pinned `temperature` across backends and assumed that
# controlled sampling. It did not. Each server applied its OWN truncation
# defaults, and they differ three ways:
#
#     backend      top_k        top_p   min_p
#     Edge-LLM     50           0.90    (none)
#     vLLM         -1 (off)     1.00    0
#     llama.cpp    40           0.95    0.05
#
# So the measured "framework" gap was confounded by three different sampling
# regimes. This script removes that confound the strongest way available:
#
#     --top-k 1  ->  pure argmax on all three backends.
#
# With a single surviving candidate token, top_p and min_p are inert and
# temperature is irrelevant, so all three decode deterministically and
# identically. Whatever difference REMAINS is genuinely the framework:
# llama.cpp's grammar-constrained tool-call path, chat-template rendering
# differences, or kernel numerics - not sampling luck.
#
# top_k=1 is chosen over temperature=0 deliberately: it is expressible on all
# three (Edge-LLM's schema requires top_k >= 1, so 0/-1 "disabled" sentinels
# are not portable) and it forces argmax regardless of how each backend
# interprets temperature=0.
#
# 7B only - that is the size the recommendation rests on.
#
# Usage:  bash scripts/greedy_framework_control.sh 2>&1 | tee output/greedy_control.log
set -u
cd /home/michal/dev/jetson-llm || exit 1
export PATH="$HOME/.local/bin:$PATH"
LOG=output/window_logs
mkdir -p "$LOG"

for CFG in 7b-fp16-edgellm-thor 7b-fp16-vllm-thor 7b-fp16-llamacpp-thor; do
    echo "───── $CFG (greedy: top_k=1) ─────"
    date '+      %H:%M:%S'
    uv run python scripts/validate_tool_calling.py \
        --model-config "$CFG" --limit 50 \
        --temperature 0 --top-k 1 --top-p 1.0 \
        --ready-timeout 1800 \
        --results-json "output/tool_calling_${CFG}_greedy.json" \
        > "$LOG/bfcl_${CFG}_greedy.log" 2>&1
    echo "      exit=$?"
    python3 -c "
import json
d = json.load(open('output/tool_calling_${CFG}_greedy.json'))[-1]
print(f\"      simple={d['simple_accuracy']:.0%}  irrelevance={d['irrelevance_accuracy']:.0%}  overall={d['overall_accuracy']:.0%}\")
" 2>/dev/null || echo "      (no result - see $LOG/bfcl_${CFG}_greedy.log)"
done

echo
echo "───── comparison: default sampling vs greedy ─────"
python3 -c "
import json
def load(p):
    try: return json.load(open(p))[-1]
    except Exception: return None
print(f\"{'backend':12s} {'default irrel':>14s} {'greedy irrel':>13s} {'default overall':>16s} {'greedy overall':>15s}\")
for lbl, cfg in (('Edge-LLM','7b-fp16-edgellm-thor'), ('vLLM','7b-fp16-vllm-thor'), ('llama.cpp','7b-fp16-llamacpp-thor')):
    a = load(f'output/tool_calling_{cfg}.json')
    b = load(f'output/tool_calling_{cfg}_greedy.json')
    f = lambda d, k: f\"{d[k]:.0%}\" if d else '-'
    print(f\"{lbl:12s} {f(a,'irrelevance_accuracy'):>14s} {f(b,'irrelevance_accuracy'):>13s} {f(a,'overall_accuracy'):>16s} {f(b,'overall_accuracy'):>15s}\")
"
