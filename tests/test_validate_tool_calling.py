"""Unit tests for validate_tool_calling.py's pure scoring logic and eval-set
shape - no server/GPU needed. The end-to-end main() (which starts a real
coordinator) is exercised for real in docs/TODO.md Phase 1's pending smoke
test, not here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_tool_calling import EVAL_CASES, _first_tool_call_name  # noqa: E402


def test_first_tool_call_name_returns_none_when_no_tool_calls():
    assert _first_tool_call_name({"content": "hello", "tool_calls": None}) is None


def test_first_tool_call_name_returns_none_when_tool_calls_missing_key():
    assert _first_tool_call_name({"content": "hello"}) is None


def test_first_tool_call_name_extracts_the_function_name():
    message = {
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "ask_vlm", "arguments": "{}"}}],
    }
    assert _first_tool_call_name(message) == "ask_vlm"


def test_eval_cases_cover_all_three_expected_outcomes():
    expected_tools = {case["expected_tool"] for case in EVAL_CASES}
    assert expected_tools == {"ask_vlm", "read_scene_state", None}


def test_eval_cases_are_non_empty_per_category():
    for tool in ("ask_vlm", "read_scene_state", None):
        matching = [c for c in EVAL_CASES if c["expected_tool"] == tool]
        assert len(matching) >= 3, f"too few eval cases for expected_tool={tool!r}"
