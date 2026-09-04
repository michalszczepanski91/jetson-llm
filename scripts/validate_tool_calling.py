#!/usr/bin/env python3
"""Tool-call judgment accuracy for an orchestrator-LLM candidate - the
accuracy axis for this lab, replacing jetson-vlm-lab's GQA/TextVQA (which
measure visual QA - not this pipeline's job at all).

embedded-ai-chain's own orchestrator_models.py already documents the real,
live-confirmed failure mode this eval targets: at 1.5B, the model's own
"auto" tool_choice judgment misses `ask_vlm` escalations often enough
(2026-09-03 finding: three phrasings, zero ask_vlm calls in 23 real turns)
that orchestrator.py had to add a forced-tool_choice workaround
(_FORCE_ASK_VLM) rather than trust the model. This script measures whether
a bigger model and/or a different backend's tool-call parser (vLLM's
--tool-call-parser hermes vs llama.cpp's --jinja template path) can close
that gap well enough that the forced-tool_choice workaround could be
removed.

EVAL_CASES, _TOOLS, _SYSTEM_PROMPT, and _SCENE_CONTEXT below are hand-copied
(not imported - this repo stays standalone, see README) from
embedded-ai-chain/src/orchestrator.py's VLM_QUERY_PATTERNS,
VISION_QUERY_PATTERNS, _LLM_TOOLS, _LLM_SYSTEM_PROMPT, and
_grounding_context()'s output shape. Every call here uses tool_choice="auto"
only - unlike orchestrator.py's own forced-tool_choice path, since the whole
point is measuring the model's unforced judgment, not the workaround around
it.

Usage:
    uv run python scripts/validate_tool_calling.py --model-config 1.5b-awq-vllm-orin
    uv run python scripts/validate_tool_calling.py --model-config 7b-q4-llamacpp-orin \\
        --results-json output/tool_calling_7b_llamacpp.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_client import call_llm  # noqa: E402
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from model_config import load_model_config  # noqa: E402

_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}

# Hand-copied from embedded-ai-chain/src/orchestrator.py's _LLM_TOOLS.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_scene_state",
            "description": (
                "Read the robot's current, already-computed perception state. Fast, no "
                "delay, always safe to call first. Use for questions about what's "
                "currently visible, how many of something, or a detected person's "
                "posture/activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["yolo", "pose"],
                        "description": (
                            "'yolo': object/person detection counts, updated ~30 times/sec. "
                            "'pose': detected people's posture (standing/sitting/lying) and "
                            "activity (walking/waving/falling) - only populated while a "
                            "person is in frame."
                        ),
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_vlm",
            "description": (
                "Request a much slower vision-language model to look closely at the "
                "camera image and answer a specific question that needs real image "
                "understanding - colors, reading text, identifying an object/brand, or a "
                "detailed description. May need to cold-start (tens of seconds to "
                "minutes) before answering, so only use this when read_scene_state "
                "genuinely cannot answer the question. The answer is NOT returned by "
                "this call - it arrives later as a separate follow-up once ready."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to ask about the current camera view.",
                    },
                },
                "required": ["question"],
            },
        },
    },
]

# Hand-copied from embedded-ai-chain/src/orchestrator.py's _LLM_SYSTEM_PROMPT.
_SYSTEM_PROMPT = (
    "You are the voice of a robot. Speak conversationally, in one or two short "
    "sentences - you are talking out loud, not writing. The next message gives you "
    "the current perception state (object/person counts and posture/activity) - "
    "already fetched, no need to call read_scene_state again unless you specifically "
    "want a fresher reading. Use ask_vlm only when the question needs something that "
    "data cannot answer (color, identity, reading text, a detailed look) - it is much "
    "slower. If ask_vlm reports it accepted the request, tell the user you're taking a "
    "closer look and the answer will follow shortly - keep it very short. If it reports "
    "busy, say you're still working on the previous request."
)

# Fixed scene grounding, same shape as orchestrator.py's _grounding_context()
# output (one person standing, a couple of chairs) - held constant across
# every eval case so only the transcript varies.
_SCENE_CONTEXT = (
    "Current perception state, already up to date - no need to call "
    "read_scene_state again unless you specifically want a fresher "
    "reading:\n"
    'yolo (object/person detection counts): {"available": true, "detections": '
    '{"person": 1, "chair": 2}}\n'
    'pose (per-person posture/activity, empty if no one\'s in frame): {"available": '
    'true, "people": [{"track_id": 1, "posture": "standing", "action": "walking"}]}'
)

# expected_tool is None for "should answer directly, no tool call".
# Transcripts are drawn from real VLM_QUERY_PATTERNS/VISION_QUERY_PATTERNS
# matches in orchestrator.py, not invented fresh - these are the exact
# phrasings that pattern-matching already handles correctly in the
# rule-based fallback, so a tool-calling model that can't match that bar is
# a real regression, not just "different."
EVAL_CASES: list[dict[str, Any]] = [
    # --- should call ask_vlm (needs real vision the yolo/pose data can't give) ---
    {"transcript": "what color is my shirt", "expected_tool": "ask_vlm"},
    {"transcript": "what is this in my hand", "expected_tool": "ask_vlm"},
    {"transcript": "can you read that sign", "expected_tool": "ask_vlm"},
    {"transcript": "take a closer look at the object on the table", "expected_tool": "ask_vlm"},
    {"transcript": "what brand is this laptop", "expected_tool": "ask_vlm"},
    # --- should call read_scene_state (answerable from yolo/pose counts) ---
    {"transcript": "what do you see", "expected_tool": "read_scene_state"},
    {"transcript": "how many people are there", "expected_tool": "read_scene_state"},
    {"transcript": "is anyone in the room", "expected_tool": "read_scene_state"},
    {"transcript": "describe the scene", "expected_tool": "read_scene_state"},
    {"transcript": "who's in the room", "expected_tool": "read_scene_state"},
    # --- should answer directly, no tool call needed ---
    {"transcript": "what's your name", "expected_tool": None},
    {"transcript": "tell me a joke", "expected_tool": None},
    {"transcript": "how are you today", "expected_tool": None},
    {"transcript": "thank you", "expected_tool": None},
    {"transcript": "goodbye", "expected_tool": None},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="1.5b-awq-vllm-orin", help="key into configs/models.yaml")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ready-timeout", type=float, default=600.0)
    p.add_argument("--results-json", default=None)
    add_target_args(p)
    return p.parse_args()


def _first_tool_call_name(message: dict[str, Any]) -> str | None:
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return None
    return tool_calls[0].get("function", {}).get("name")


def main():
    args = parse_args()

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")

    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)
    print("Connecting to remote server..." if args.target == "remote" else f"Starting {variant['backend']} server...")
    coordinator.start()
    try:
        if not coordinator.wait_ready():
            raise TimeoutError(f"{variant['backend']} server did not become ready")

        outcomes = []
        for case in EVAL_CASES:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "system", "content": _SCENE_CONTEXT},
                {"role": "user", "content": case["transcript"]},
            ]
            message = call_llm(
                coordinator, messages, max_tokens=args.max_tokens, tools=_TOOLS, tool_choice="auto",
            )
            actual = _first_tool_call_name(message)
            correct = actual == case["expected_tool"]
            outcomes.append({**case, "actual_tool": actual, "correct": correct})
            status = "OK" if correct else "WRONG"
            print(f"  [{status}] {case['transcript']!r}: expected={case['expected_tool']} actual={actual}")

        def _accuracy(subset):
            return (sum(1 for o in subset if o["correct"]) / len(subset)) if subset else None

        ask_vlm_cases = [o for o in outcomes if o["expected_tool"] == "ask_vlm"]
        scene_cases = [o for o in outcomes if o["expected_tool"] == "read_scene_state"]
        no_tool_cases = [o for o in outcomes if o["expected_tool"] is None]

        row = {
            "model_config": args.model_config,
            "model": variant["model"],
            "backend": variant["backend"],
            "platform": variant["platform"],
            "target": args.target,
            "n_cases": len(outcomes),
            "overall_accuracy": _accuracy(outcomes),
            "ask_vlm_escalation_accuracy": _accuracy(ask_vlm_cases),
            "read_scene_state_accuracy": _accuracy(scene_cases),
            "no_tool_accuracy": _accuracy(no_tool_cases),
            "outcomes": outcomes,
            "note": (
                "tool_choice='auto' throughout - does NOT use orchestrator.py's forced "
                "tool_choice workaround, since the point is measuring unforced judgment"
            ),
        }
        if args.target == "remote":
            row["remote_host"] = args.remote_host
            row["remote_port"] = args.remote_port
        print(json.dumps({k: v for k, v in row.items() if k != "outcomes"}, indent=2))

        if args.results_json:
            out = Path(args.results_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            rows = json.loads(out.read_text()) if out.exists() else []
            rows.append(row)
            out.write_text(json.dumps(rows, indent=2))
    finally:
        if args.target != "remote":
            print(f"Stopping {variant['backend']} server...")
        coordinator.stop()


if __name__ == "__main__":
    main()
