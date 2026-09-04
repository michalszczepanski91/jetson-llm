"""Unit tests for validate_tool_calling.py's pure BFCL-format conversion and
scoring logic - no server/GPU/staged-dataset needed (small inline fixtures
mirror the real BFCL JSON shape). The end-to-end main() (which reads the
real staged dataset and starts a real coordinator) is exercised for real in
docs/TODO.md Phase 1's pending smoke test, not here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_tool_calling import (  # noqa: E402
    _bfcl_function_to_openai_tool,
    _first_tool_call,
    _loose_equal,
    _params_match,
)


def test_bfcl_function_to_openai_tool_renames_dict_to_object():
    func = {
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle.",
        "parameters": {
            "type": "dict",
            "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}},
            "required": ["base", "height"],
        },
    }
    tool = _bfcl_function_to_openai_tool(func)
    assert tool["type"] == "function"
    assert tool["function"]["parameters"]["type"] == "object"
    assert tool["function"]["name"] == "calculate_triangle_area"


def test_bfcl_function_to_openai_tool_renames_nested_dict_types_too():
    func = {
        "name": "f",
        "parameters": {
            "type": "dict",
            "properties": {"opts": {"type": "dict", "properties": {"x": {"type": "integer"}}}},
        },
    }
    tool = _bfcl_function_to_openai_tool(func)
    assert tool["function"]["parameters"]["properties"]["opts"]["type"] == "object"


def test_first_tool_call_returns_none_when_absent():
    assert _first_tool_call({"content": "hi", "tool_calls": None}) == (None, None)


def test_first_tool_call_extracts_name_and_parsed_arguments():
    message = {
        "tool_calls": [{"function": {"name": "math.factorial", "arguments": '{"number": 5}'}}],
    }
    name, args = _first_tool_call(message)
    assert name == "math.factorial"
    assert args == {"number": 5}


def test_first_tool_call_tolerates_malformed_arguments_json():
    message = {"tool_calls": [{"function": {"name": "f", "arguments": "not json"}}]}
    name, args = _first_tool_call(message)
    assert name == "f"
    assert args == {}


def test_loose_equal_numeric_int_vs_float():
    assert _loose_equal(5, 5.0) is True
    assert _loose_equal(5, 6) is False


def test_loose_equal_string_case_and_whitespace_insensitive():
    assert _loose_equal("Units", "units") is True
    assert _loose_equal(" units ", "units") is True
    assert _loose_equal("units", "meters") is False


def test_loose_equal_bool_compares_directly_not_via_numeric_branch():
    # bool is an int subclass in Python (True == 1) - routed through the
    # bool branch explicitly so a future numeric-branch change can't
    # accidentally start comparing True against, say, 1.0 via float().
    assert _loose_equal(True, 1) is True
    assert _loose_equal(True, False) is False


def test_params_match_exact_value():
    assert _params_match({"base": [10], "height": [5]}, {"base": 10, "height": 5}) is True


def test_params_match_empty_string_sentinel_allows_omission():
    expected = {"unit": ["units", ""]}
    assert _params_match(expected, {}) is True  # omitted - "" sentinel present
    assert _params_match(expected, {"unit": "units"}) is True
    assert _params_match(expected, {"unit": "meters"}) is False


def test_params_match_missing_required_key_without_sentinel_fails():
    assert _params_match({"base": [10]}, {}) is False


def test_params_match_ignores_extra_actual_keys():
    # Deliberate simplification vs the official checker - see
    # _params_match()'s docstring.
    assert _params_match({"base": [10]}, {"base": 10, "extra": "hallucinated"}) is True


def test_params_match_wrong_value_fails():
    assert _params_match({"base": [10]}, {"base": 99}) is False
