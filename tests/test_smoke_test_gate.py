"""The smoke-test gate must fail a BROKEN engine and pass a merely weak one.

Regression for 2026-09-10: scripts/smoke_test.py reported `"result": "PASS"`
for an engine that answered "What is the capital of Poland?" with
"0000000000000000..." - its only criterion was that the HTTP call had not
raised. A gate that passes garbage is worse than no gate, because it is
trusted; this file pins both directions of that behaviour.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from smoke_test import _degenerate_reason  # noqa: E402


# Real observed output from the broken Qwen3-8B-AWQ Edge-LLM engine, plus the
# shapes a damaged quantization has produced in this lab before.
BROKEN = [
    ("0" * 64, "digit spew - the actual Qwen3-8B-AWQ plain-completion output"),
    ("]\n\t" * 40, "punctuation/whitespace loop - its greedy-decode output"),
    (" again Floors\n\n" + "2" * 37, "its default-sampling output"),
    ("The past is a very large and " + "very large and " * 30, "short vocabulary looping"),
    ("", "empty"),
    ("   \n  ", "whitespace only"),
    (None, "no content field at all"),
]

# Must NOT be flagged: judging answer quality is BFCL/MMLU's job, not this gate's.
WORKING = [
    "The capital of Poland is Warsaw.",
    "Warsaw.",
    "I don't know.",
    # Wrong, but fluent - a weak model, not a broken engine.
    "The capital of Poland is Krakow, which has been the capital since 1596.",
    # Narrating a tool call in prose: a real, known llama.cpp/Qwen2.5 weakness.
    "I should call get_weather with city=Warsaw to find that out for you.",
    # Legitimately repetitive but varied enough to be language.
    "Warsaw is the capital. Warsaw is also the largest city in Poland.",
]


def test_broken_engine_output_is_flagged():
    for text, why in BROKEN:
        assert _degenerate_reason(text) is not None, f"should have been flagged ({why}): {text!r}"


def test_working_output_is_not_flagged():
    for text in WORKING:
        reason = _degenerate_reason(text)
        assert reason is None, f"false positive on legitimate output {text!r}: {reason}"
