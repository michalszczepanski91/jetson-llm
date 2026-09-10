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
            report["plain_completion"] = {"ok": True, "content": msg.get("content")}
            print(f"  -> {msg.get('content')!r}")
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
                report["tool_call"] = {"ok": True, "structured": False, "content": msg.get("content")}
                print(f"  -> NO structured call (formatting failure or narrated in prose): "
                      f"{msg.get('content')!r}")
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
