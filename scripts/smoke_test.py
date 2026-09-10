#!/usr/bin/env python3
"""Tier 0 smoke test (docs/note.md §18) for one configs/models.yaml candidate:
does it serve, does a plain completion work, does a tool-calling completion
return a structured call. Not a benchmark - no schema-conformant document is
written, no percentiles, one rep of each check.

Every prior candidate in this registry (Qwen2.5-1.5B, Bielik-11B on both
backends, Apertus) was smoke-tested this way before any real benchmark ran
against it, but ad hoc - no script existed, so each smoke test was hand-typed
and only its *outcome* made it into docs/TODO.md, not a reusable tool. Written
now because Qwen2.5-3B/7B need the same treatment (docs/TODO.md Phase 5 Step 0)
and a fourth ad hoc round would be the same one-off cost a fourth time.

Fixed test shape, same as every prior smoke test in this lab, so results are
comparable: a plain completion ("What is the capital of Poland?") and a
tool-calling completion (get_weather/Warsaw, tool_choice="auto", unforced -
same reasoning as scripts/validate_tool_calling.py's docstring).

Usage:
    uv run python scripts/smoke_test.py --model-config 3b-awq-vllm-orin
    uv run python scripts/smoke_test.py --model-config 3b-q4-llamacpp-orin --temperature 0.1
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from manifest import gpu_holding_pids, running_containers  # noqa: E402

from llm_client import call_llm  # noqa: E402
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from model_config import load_model_config  # noqa: E402

_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}

_PLAIN_PROMPT = "What is the capital of Poland?"

_WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
_TOOL_PROMPT = "What's the weather like in Warsaw?"


def _degenerate_reason(text: str | None) -> str | None:
    """Does this response look like a BROKEN engine rather than a weak model?

    Exists because this gate reported **PASS** for an engine answering "What is
    the capital of Poland?" with `"0000000000000000..."` (Qwen3-8B-AWQ on
    Edge-LLM, 2026-09-10). The old criterion was only that `call_llm()` had not
    raised, so any HTTP 200 counted as a pass no matter what came back - the
    same hole that let a damaged self-quantized FP8 checkpoint through earlier
    (it emitted zero tool calls in 50 tries and scored a meaningless "100%
    irrelevance"). A gate that passes garbage is worse than no gate, because it
    is trusted.

    Deliberately conservative: this must flag BROKEN, never merely BAD. A weak
    model giving a wrong-but-fluent answer has to keep passing, because judging
    answer quality is what BFCL and MMLU are for. All three checks below fire
    only on output no working model produces.

    Returns a reason string, or None if the text looks like real language."""
    if text is None or not text.strip():
        return "empty response"
    stripped = text.strip()

    # 1. Almost no letters. Catches "000000...", "]\n\t\n\t...", digit spew.
    if len(stripped) >= 20:
        alpha_ratio = sum(c.isalpha() for c in stripped) / len(stripped)
        if alpha_ratio < 0.25:
            return f"only {alpha_ratio:.0%} alphabetic characters - not language"

    # 2. A long run of one repeated character.
    longest_run = run = 1
    for prev, cur in zip(stripped, stripped[1:]):
        run = run + 1 if cur == prev else 1
        longest_run = max(longest_run, run)
    if longest_run >= 20:
        return f"a single character repeats {longest_run} times consecutively"

    # 3. A short vocabulary looping. Catches "very large and very large and...",
    #    which has no single word repeated back-to-back and so slips past a
    #    naive consecutive-repeat check.
    words = stripped.split()
    if len(words) >= 15:
        variety = len(set(w.lower() for w in words)) / len(words)
        if variety < 0.20:
            return f"only {variety:.0%} distinct words over {len(words)} - looping"

    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", required=True, help="key into configs/models.yaml")
    p.add_argument("--temperature", type=float, default=None,
                   help="pin if the candidate's backend needs it for reliable tool-calling "
                        "(docs/TODO.md Phase 1: llama.cpp needed 0.1, vLLM did not)")
    p.add_argument("--ready-timeout", type=float, default=900.0)
    add_target_args(p)
    return p.parse_args()


def main():
    args = parse_args()

    if args.target == "local":
        others_c = running_containers(exclude={"vllm-llm-lab", "llamacpp-llm-lab"})
        others_g = gpu_holding_pids()
        if others_c or others_g:
            print("error: board is not quiet - a smoke test's numbers (esp. cold start) are "
                  "meaningless if something else is competing for the GPU:")
            for n in others_c:
                print(f"    container: {n}")
            for pid, cmd in others_g:
                print(f"    pid {pid}: {cmd}")
            sys.exit(1)

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")
    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)

    report = {"model_config": args.model_config, "model": variant["model"],
              "backend": variant["backend"], "platform": variant["platform"]}

    try:
        print(f"=== {args.model_config} ({variant['model']} on {variant['backend']}) ===")
        print(f"starting {variant['backend']}...")
        coordinator.start()
        if not coordinator.wait_ready():
            report["result"] = "FAILED - server did not become ready"
            print(json.dumps(report, indent=2))
            sys.exit(1)

        cold_start = coordinator.cold_start_breakdown()
        report["cold_start"] = cold_start
        print(f"cold start: {cold_start['total_s']}s (weights_cached={cold_start['weights_cached']})")

        print(f"plain completion: {_PLAIN_PROMPT!r}")
        try:
            msg = call_llm(coordinator, [{"role": "user", "content": _PLAIN_PROMPT}],
                            max_tokens=64, temperature=args.temperature)
            content = msg.get("content")
            degenerate = _degenerate_reason(content)
            report["plain_completion"] = {
                "ok": degenerate is None, "content": content,
            }
            if degenerate:
                # `ok: False` on a 200 response is deliberate: the server
                # answered, the ENGINE is broken, and that is a failed smoke
                # test. See _degenerate_reason's docstring.
                report["plain_completion"]["degenerate"] = degenerate
                print(f"  -> {content!r}")
                print(f"  FAILED: response is degenerate ({degenerate}) - "
                      f"the server answered, but this engine is not usable")
            else:
                print(f"  -> {content!r}")
        except Exception as exc:  # noqa: BLE001 - a failure here is the result
            report["plain_completion"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {type(exc).__name__}: {exc}")

        print(f"tool-calling completion: {_TOOL_PROMPT!r}, tool_choice=auto")
        try:
            msg = call_llm(coordinator, [{"role": "user", "content": _TOOL_PROMPT}],
                            max_tokens=128, tools=_WEATHER_TOOL, tool_choice="auto",
                            temperature=args.temperature)
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                call = tool_calls[0]["function"]
                report["tool_call"] = {"ok": True, "structured": True,
                                        "name": call["name"], "arguments": call["arguments"]}
                print(f"  -> structured call: {call['name']}({call['arguments']})")
            else:
                content = msg.get("content")
                degenerate = _degenerate_reason(content)
                report["tool_call"] = {"ok": degenerate is None, "structured": False,
                                       "content": content}
                if degenerate:
                    report["tool_call"]["degenerate"] = degenerate
                    print(f"  -> FAILED: degenerate response ({degenerate}): {content!r}")
                else:
                    # Narrating the call in prose is a real, known model
                    # weakness (llama.cpp/Qwen2.5 did exactly this at default
                    # temperature) - a weak result, NOT a broken engine, so it
                    # stays ok=True and is quantified by BFCL rather than here.
                    print(f"  -> NO structured call (formatting failure or narrated in prose): "
                          f"{content!r}")
        except Exception as exc:  # noqa: BLE001
            report["tool_call"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {type(exc).__name__}: {exc}")

        report["result"] = "PASS" if (
            report["plain_completion"].get("ok") and report["tool_call"].get("ok")
        ) else "PARTIAL/FAIL - see fields above"
    except Exception as exc:  # noqa: BLE001 - the server itself failed to start/serve
        report["result"] = f"FAILED - {type(exc).__name__}: {exc}"
        print(f"!! {type(exc).__name__}: {exc}")
    finally:
        if args.target != "remote":
            print(f"stopping {variant['backend']}...")
        coordinator.stop()

    print("\n--- summary ---")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
