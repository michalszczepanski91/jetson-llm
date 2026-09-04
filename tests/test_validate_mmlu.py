"""Unit tests for validate_mmlu.py's pure prompt-formatting/answer-parsing
logic - no server/GPU/staged-dataset needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_mmlu import _format_prompt, _parse_answer_letter  # noqa: E402


def test_format_prompt_lists_all_four_choices_with_letters():
    row = {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"]}
    prompt = _format_prompt(row)
    assert "What is 2+2?" in prompt
    assert "A. 3" in prompt
    assert "B. 4" in prompt
    assert "C. 5" in prompt
    assert "D. 6" in prompt


def test_parse_answer_letter_bare_letter():
    assert _parse_answer_letter("B") == "B"


def test_parse_answer_letter_letter_with_punctuation():
    assert _parse_answer_letter("The answer is B.") == "B"
    assert _parse_answer_letter("(C)") == "C"


def test_parse_answer_letter_lowercase_is_normalized():
    assert _parse_answer_letter("b") == "B"


def test_parse_answer_letter_returns_none_for_empty_or_unparseable():
    assert _parse_answer_letter("") is None
    assert _parse_answer_letter("I'm not sure.") is None
