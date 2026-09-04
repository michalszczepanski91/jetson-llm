#!/usr/bin/env python3
"""Tool-call judgment accuracy against real BFCL (Berkeley Function-Calling
Leaderboard, gorilla-llm/Berkeley-Function-Calling-Leaderboard on HF) data -
the accuracy axis for this lab, replacing jetson-vlm-lab's GQA/TextVQA
(which measure visual QA - not applicable to a text orchestrator).

embedded-ai-chain's own orchestrator_models.py already documents the real,
live-confirmed failure mode this eval targets: at 1.5B, the model's own
"auto" tool_choice judgment misses `ask_vlm` escalations often enough
(2026-09-03 finding) that orchestrator.py had to add a forced-tool_choice
workaround (_FORCE_ASK_VLM) rather than trust the model. This script
measures whether a bigger model and/or a different backend's tool-call
parser can close that gap well enough that the workaround could be removed
- using a real, external, recognized benchmark instead of a hand-rolled one
(an earlier version of this script used 15 hand-copied cases; BFCL gives
hundreds of real cases with an established scoring convention instead).

Two BFCL categories only, both single-turn/single-function - this
orchestrator only ever considers one tool call per turn, so BFCL's
multi-turn/parallel/multiple-function categories don't apply here:

  - "simple": exactly one function offered, exactly one correct call
    expected - scored by AST-style argument matching against
    possible_answer/BFCL_v3_simple.json (see _params_match()'s docstring)
  - "irrelevance": the offered function does NOT apply to the question -
    correct behavior is calling nothing at all

**Not the official bfcl-eval checker** - that package's real AST matcher
handles many more type/language-specific edge cases (see
https://github.com/ShishirPatil/gorilla). `_params_match()` below is a
simplified reimplementation, good enough for *relative* comparison across
this lab's own candidates (does 7B beat 1.5B, does llama.cpp's parser beat
vLLM's), not for submitting a leaderboard-comparable score.

`--temperature` defaults to 0.1, not the server's own default sampling
temperature (~0.7-0.8) - a real, measured fix, not an arbitrary choice:
repeated identical Qwen2.5-1.5B/llama.cpp tool-calling calls at default
temperature produced a structured tool call only 80% of the time (12/15),
100% (15/15) at 0.1; the same repeated test against vLLM was 100% at BOTH
settings, so pinning this low costs vLLM nothing while fixing a real
llama.cpp gap. See `docs/TODO.md` Phase 1 for the full diagnostic
(including why raising `max_tokens` alone did NOT fix it - the failures
weren't truncation, the model genuinely chose not to call the tool at that
sampling temperature on some fraction of calls).

Dataset staging - not bundled (BFCL is thousands of samples across
categories, several MB total). Download the three files this script
actually uses:

    huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \\
        BFCL_v3_simple.json BFCL_v3_irrelevance.json possible_answer/BFCL_v3_simple.json \\
        --repo-type dataset --local-dir /opt/datasets/BFCL

Expected layout:
    /opt/datasets/BFCL/BFCL_v3_simple.json
    /opt/datasets/BFCL/BFCL_v3_irrelevance.json
    /opt/datasets/BFCL/possible_answer/BFCL_v3_simple.json

Fails fast with a clear message if missing, same convention as
jetson-vlm-lab's validate_textvqa.py.

Usage:
    uv run python scripts/validate_tool_calling.py --model-config 1.5b-awq-vllm-orin --limit 50
    uv run python scripts/validate_tool_calling.py --model-config 1.5b-q4-llamacpp-orin --limit 50 \\
        --results-json output/tool_calling_1.5b_llamacpp.json
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
_DEFAULT_DATA_DIR = "/opt/datasets/BFCL"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"missing {path} - BFCL isn't bundled with this repo, see this script's "
            "module docstring for the huggingface-cli download command to stage it"
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# BFCL's function defs use several non-standard JSON-schema type names
# (its own convention, predating widespread OpenAI-schema tooling) that
# aren't valid JSON Schema types (object, array, string, number, integer,
# boolean, null). vLLM tolerates these silently, but llama.cpp's server
# hard-errors building its tool-call grammar - confirmed live 2026-09-04:
# "JSON schema conversion failed: Unrecognized schema: {"type":"float",...}"
# on every BFCL case using a float-typed parameter (215 of ~1400 type
# occurrences across the staged simple+irrelevance files - common enough to
# silently wreck a huge fraction of any llama.cpp scorecard run if left
# unfixed, not a rare edge case). Counted every type actually present in
# the staged corpus (`python3 -c "..."` one-off scan, 2026-09-04) rather
# than guessing which ones might appear: dict(658), float(215), tuple(2),
# any(1) beyond the standard types.
_BFCL_TYPE_RENAMES = {"dict": "object", "float": "number", "tuple": "array"}


def _bfcl_function_to_openai_tool(func: dict[str, Any]) -> dict[str, Any]:
    """Recursively normalizes every BFCL-specific type name to a valid
    JSON-schema one - not just at the top-level `parameters` node, since a
    nested param can carry its own `type` too. `"any"` has no JSON-schema
    equivalent (the idiomatic way to express "unconstrained" is to omit
    `type` entirely, not invent a fake type name for it), so that key is
    dropped rather than renamed."""
    def _convert(node):
        if isinstance(node, dict):
            converted = {k: _convert(v) for k, v in node.items()}
            node_type = converted.get("type")
            if node_type in _BFCL_TYPE_RENAMES:
                converted["type"] = _BFCL_TYPE_RENAMES[node_type]
            elif node_type == "any":
                del converted["type"]
            return converted
        if isinstance(node, list):
            return [_convert(v) for v in node]
        return node

    return {"type": "function", "function": _convert(func)}


def _first_tool_call(message: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return None, None
    call = tool_calls[0]
    name = call.get("function", {}).get("name")
    raw_args = call.get("function", {}).get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}
    return name, args


def _loose_equal(actual: Any, expected: Any) -> bool:
    """int/float compared numerically (a model returning 5 vs 5.0 for the
    same argument is not a real mistake); strings compared
    case/whitespace-insensitively (BFCL's own acceptable-value lists already
    include casing variants like "units"/"Units" for some params - this
    covers the ones they don't); everything else by equality."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected


def _params_match(expected: dict[str, list[Any]], actual: dict[str, Any]) -> bool:
    """Simplified BFCL-style AST match. For each parameter BFCL's ground
    truth lists, `actual`'s value must loosely-equal one of the acceptable
    alternatives; an empty string "" in that list is BFCL's own convention
    for "omitting this key entirely is also acceptable" (an optional param
    left at its default) - see possible_answer file examples like
    `"unit": ["units", ""]`.

    Deliberate simplification vs the official checker: extra keys in
    `actual` that aren't in `expected` are ignored here, where BFCL's real
    checker also penalizes hallucinated/unexpected parameters. Acceptable
    for this lab's relative-comparison purpose, not for a leaderboard-exact
    score - see this module's docstring."""
    for key, acceptable in expected.items():
        if key not in actual:
            if "" in acceptable:
                continue
            return False
        if not any(_loose_equal(actual[key], want) for want in acceptable if want != ""):
            return False
    return True


def _questions_to_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    # BFCL's "question" is a list of turns (multi-turn categories need
    # that); simple/irrelevance are always a single turn with one user
    # message, but flatten generally rather than assume that shape blindly.
    return [message for turn in case["question"] for message in turn]


def _run_simple(coordinator, cases, answers_by_id, max_tokens, temperature) -> list[dict[str, Any]]:
    outcomes = []
    for case in cases:
        tool = _bfcl_function_to_openai_tool(case["function"][0])
        messages = _questions_to_messages(case)
        message = call_llm(
            coordinator, messages, max_tokens=max_tokens, tools=[tool], tool_choice="auto", temperature=temperature,
        )
        actual_name, actual_args = _first_tool_call(message)

        ground_truth_row = answers_by_id.get(case["id"])
        if ground_truth_row is None:
            correct = False
        else:
            [expected] = ground_truth_row["ground_truth"]
            expected_name, expected_params = next(iter(expected.items()))
            correct = actual_name == expected_name and _params_match(expected_params, actual_args or {})

        outcomes.append({"id": case["id"], "category": "simple", "actual_tool": actual_name, "correct": correct})
    return outcomes


def _run_irrelevance(coordinator, cases, max_tokens, temperature) -> list[dict[str, Any]]:
    outcomes = []
    for case in cases:
        tool = _bfcl_function_to_openai_tool(case["function"][0])
        messages = _questions_to_messages(case)
        message = call_llm(
            coordinator, messages, max_tokens=max_tokens, tools=[tool], tool_choice="auto", temperature=temperature,
        )
        actual_name, _ = _first_tool_call(message)
        outcomes.append({
            "id": case["id"], "category": "irrelevance", "actual_tool": actual_name, "correct": actual_name is None,
        })
    return outcomes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="1.5b-awq-vllm-orin", help="key into configs/models.yaml")
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR, help="staged BFCL directory, see module docstring")
    p.add_argument("--limit", type=int, default=50, help="cases per category (simple/irrelevance each); BFCL has "
                   "hundreds per category, full runs are slow on-device - use a higher value for a final scorecard")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.1, help="pinned low by default - a real, measured "
                   "finding, not a guess: repeated identical calls at default sampling temperature (~0.7-0.8) "
                   "produced structured tool calls only 80%% of the time on llama.cpp/Qwen2.5-1.5B (12/15), "
                   "100%% (15/15) at 0.1 - vLLM was 100%% at both, so pinning this low costs nothing there and "
                   "fixes the llama.cpp gap. See docs/TODO.md Phase 1 for the full diagnostic.")
    p.add_argument("--ready-timeout", type=float, default=600.0)
    p.add_argument("--results-json", default=None)
    add_target_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    simple_cases = _load_jsonl(data_dir / "BFCL_v3_simple.json")[: args.limit]
    simple_answers = {row["id"]: row for row in _load_jsonl(data_dir / "possible_answer" / "BFCL_v3_simple.json")}
    irrelevance_cases = _load_jsonl(data_dir / "BFCL_v3_irrelevance.json")[: args.limit]

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")

    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)
    print("Connecting to remote server..." if args.target == "remote" else f"Starting {variant['backend']} server...")
    coordinator.start()
    try:
        if not coordinator.wait_ready():
            raise TimeoutError(f"{variant['backend']} server did not become ready")

        print(f"Running {len(simple_cases)} BFCL 'simple' cases...")
        outcomes = _run_simple(coordinator, simple_cases, simple_answers, args.max_tokens, args.temperature)
        print(f"Running {len(irrelevance_cases)} BFCL 'irrelevance' cases...")
        outcomes += _run_irrelevance(coordinator, irrelevance_cases, args.max_tokens, args.temperature)

        def _accuracy(category):
            subset = [o for o in outcomes if o["category"] == category]
            return (sum(1 for o in subset if o["correct"]) / len(subset)) if subset else None

        row = {
            "model_config": args.model_config,
            "model": variant["model"],
            "backend": variant["backend"],
            "platform": variant["platform"],
            "target": args.target,
            "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            "n_cases": len(outcomes),
            "temperature": args.temperature,
            "simple_accuracy": _accuracy("simple"),
            "irrelevance_accuracy": _accuracy("irrelevance"),
            "overall_accuracy": (sum(1 for o in outcomes if o["correct"]) / len(outcomes)) if outcomes else None,
            "note": (
                "simplified AST match, not the official bfcl-eval checker; "
                "tool_choice='auto' throughout - does NOT use orchestrator.py's "
                "forced tool_choice workaround, since the point is measuring "
                "unforced judgment - see module docstring. temperature pinned "
                "low (see --temperature's own help text for why - a measured "
                "fix for a real llama.cpp reliability gap, not an arbitrary "
                "default) so scores reflect tool-call judgment, not sampling "
                "noise"
            ),
            "outcomes": outcomes,
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
