#!/usr/bin/env python3
"""MMLU accuracy - a quantization-regression sanity check, not a
model-ranking benchmark.

This is NOT meant to answer "is 7B smarter than 1.5B" (obviously yes, MMLU
tells you nothing new there). It exists to catch a *broken* quantization or
serving config - if an AWQ/GGUF variant scores dramatically below what that
model size normally gets on MMLU (public references: Qwen2.5-1.5B-Instruct
~50-60%, 3B ~65%, 7B ~70%+), something is misconfigured (wrong quant file,
broken chat template, truncated context), not just "the model is small."
Compare each backend/precision pair to its own same-size sibling, not across
sizes.

Generative multiple-choice scoring (not the logprob-based scoring the
official MMLU harness/lm-evaluation-harness uses): asks for a single
A/B/C/D letter, parses the first such letter out of the response text. Less
precise than logprob scoring, but works identically across vLLM and
llama.cpp through the same plain /v1/chat/completions call every other
script here uses - no backend-specific logprob API needed.

Dataset staging - not bundled (the "all" config's test split is ~14K
questions, several MB as parquet). Download the file this script actually
reads:

    huggingface-cli download cais/mmlu all/test-00000-of-00001.parquet \\
        --repo-type dataset --local-dir /opt/datasets/MMLU

Expected layout:
    /opt/datasets/MMLU/all/test-00000-of-00001.parquet

Fails fast with a clear message if missing, same convention as
jetson-vlm-lab's validate_textvqa.py.

Usage:
    uv run python scripts/validate_mmlu.py --model-config 1.5b-awq-vllm-orin --limit 200
    uv run python scripts/validate_mmlu.py --model-config 1.5b-q4-llamacpp-orin --limit 200 \\
        --results-json output/mmlu_1.5b_llamacpp.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_client import call_llm  # noqa: E402
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from model_config import load_model_config  # noqa: E402

_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}
_DEFAULT_DATA_PATH = "/opt/datasets/MMLU/all/test-00000-of-00001.parquet"
_LETTERS = "ABCD"
_ANSWER_RE = re.compile(r"\b([ABCD])\b")


def _load_mmlu_rows(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"missing {path} - MMLU isn't bundled with this repo, see this script's "
            "module docstring for the huggingface-cli download command to stage it"
        )
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = table.to_pylist()
    return rows[:limit] if limit else rows


def _format_prompt(row: dict) -> str:
    lines = [row["question"], ""]
    for letter, choice in zip(_LETTERS, row["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("\nAnswer with only the letter of the correct choice.")
    return "\n".join(lines)


def _parse_answer_letter(text: str) -> str | None:
    if not text:
        return None
    m = _ANSWER_RE.search(text.strip().upper())
    return m.group(1) if m else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="1.5b-awq-vllm-orin", help="key into configs/models.yaml")
    p.add_argument("--data-path", default=_DEFAULT_DATA_PATH, help="staged MMLU 'all/test' parquet file")
    p.add_argument("--limit", type=int, default=200, help="MMLU's 'all/test' split has 14K+ questions - a full run "
                   "is impractical on-device; 200 is a reasonable quantization-sanity sample size")
    p.add_argument("--max-tokens", type=int, default=8, help="a bare letter answer needs very few tokens")
    p.add_argument("--ready-timeout", type=float, default=600.0)
    p.add_argument("--results-json", default=None)
    add_target_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    rows = _load_mmlu_rows(Path(args.data_path), args.limit)

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")

    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)
    print("Connecting to remote server..." if args.target == "remote" else f"Starting {variant['backend']} server...")
    coordinator.start()
    try:
        if not coordinator.wait_ready():
            raise TimeoutError(f"{variant['backend']} server did not become ready")

        print(f"Running {len(rows)} MMLU questions...")
        outcomes = []
        for i, row in enumerate(rows):
            messages = [{"role": "user", "content": _format_prompt(row)}]
            message = call_llm(coordinator, messages, max_tokens=args.max_tokens)
            actual_letter = _parse_answer_letter(message.get("content") or "")
            expected_letter = _LETTERS[row["answer"]]
            correct = actual_letter == expected_letter
            outcomes.append({
                "subject": row["subject"], "expected": expected_letter, "actual": actual_letter, "correct": correct,
            })
            if (i + 1) % 50 == 0:
                running = sum(1 for o in outcomes if o["correct"]) / len(outcomes)
                print(f"  [{i + 1}/{len(rows)}] running accuracy={running:.1%}")

        accuracy = sum(1 for o in outcomes if o["correct"]) / len(outcomes) if outcomes else None
        unparsed = sum(1 for o in outcomes if o["actual"] is None)
        result = {
            "model_config": args.model_config,
            "model": variant["model"],
            "backend": variant["backend"],
            "platform": variant["platform"],
            "target": args.target,
            "dataset": "cais/mmlu (all/test)",
            "n_questions": len(outcomes),
            "accuracy": accuracy,
            "unparsed_responses": unparsed,
            "note": (
                "quantization-regression sanity check, not a model-ranking benchmark - "
                "compare same-size backend/precision pairs against each other, not across "
                "sizes (a bigger model scoring higher here is not news). Generative "
                "letter-parsing scoring, not logprob-based - see module docstring"
            ),
        }
        if args.target == "remote":
            result["remote_host"] = args.remote_host
            result["remote_port"] = args.remote_port
        print(json.dumps(result, indent=2))

        if args.results_json:
            out = Path(args.results_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(out.read_text()) if out.exists() else []
            existing.append(result)
            out.write_text(json.dumps(existing, indent=2))
    finally:
        if args.target != "remote":
            print(f"Stopping {variant['backend']} server...")
        coordinator.stop()


if __name__ == "__main__":
    main()
