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

Emits a document conforming to schemas/quality_result.schema.json into
results/raw/<result_id>/ (docs/TODO.md Phase 5's first task) - a per-case
failure (server error, timeout) is recorded rather than crashing a
potentially long run. Per-case outcomes are written to a sibling
outcomes.jsonl rather than embedded inline.

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
    uv run python scripts/validate_mmlu.py --model-config 1.5b-awq-vllm-orin \\
        --execution-condition standalone --limit 200
"""

import argparse
import datetime
import json
import re
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "src"))

sys.stdout.reconfigure(line_buffering=True)  # a redirected/backgrounded run fully
# buffers stdout otherwise, hiding a multi-hour campaign's progress from a live monitor
# - see scripts/run_experiment.py's own comment for the incident this fixes.

from manifest import SCHEMA_VERSION, assert_condition_matches_reality, quality_manifest, write_result  # noqa: E402

from llm_client import call_llm  # noqa: E402
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from model_config import load_model_config  # noqa: E402

_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}
_DEFAULT_DATA_PATH = "/opt/datasets/MMLU/all/test-00000-of-00001.parquet"
_DATASET_VERSION = "all-test"
_PROMPT_TEMPLATE_VERSION = "v1"
_LETTERS = "ABCD"
_ANSWER_RE = re.compile(r"\b([ABCD])\b")


def _load_mmlu_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"missing {path} - MMLU isn't bundled with this repo, see this script's "
            "module docstring for the huggingface-cli download command to stage it"
        )
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


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
    p.add_argument("--execution-condition", choices=["standalone", "co-resident"], required=True,
                   help="REQUIRED, no default - an accuracy score is as backend/board-state-dependent as a "
                        "latency number")
    p.add_argument("--co-resident", nargs="*", default=[],
                   help="which components ran alongside, e.g. --co-resident yolo stt tts")
    p.add_argument("--ready-timeout", type=float, default=600.0)
    p.add_argument("--results-root", default=None, help="default: <repo>/results")
    add_target_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    if args.execution_condition == "co-resident" and not args.co_resident:
        raise SystemExit(
            'error: --execution-condition co-resident requires --co-resident naming what ran alongside. '
            '"Under load" is not a reproducible condition.'
        )
    if args.target == "local":
        assert_condition_matches_reality(
            args.execution_condition,
            own_containers={"vllm-llm-lab", "llamacpp-llm-lab"},
            declared_co_resident=args.co_resident,
        )

    all_rows = _load_mmlu_rows(Path(args.data_path))
    n_available = len(all_rows)
    rows = all_rows[: args.limit] if args.limit else all_rows
    n_evaluated = len(rows)

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
            try:
                message = call_llm(coordinator, messages, max_tokens=args.max_tokens)
                error = None
            except Exception as exc:  # noqa: BLE001 - a per-case failure is a result, not a crash
                message, error = {}, f"{type(exc).__name__}: {exc}"
            actual_letter = None if error else _parse_answer_letter(message.get("content") or "")
            expected_letter = _LETTERS[row["answer"]]
            correct = (actual_letter == expected_letter) if not error else False
            outcomes.append({
                "subject": row["subject"], "expected": expected_letter, "actual": actual_letter,
                "correct": correct, "error": error,
            })
            if (i + 1) % 50 == 0:
                scored_so_far = [o for o in outcomes if not o["error"]]
                running = (sum(1 for o in scored_so_far if o["correct"]) / len(scored_so_far)) if scored_so_far else 0.0
                print(f"  [{i + 1}/{len(rows)}] running accuracy={running:.1%}")

        scored = [o for o in outcomes if not o["error"]]
        n_correct = sum(1 for o in scored if o["correct"])
        n_unparsed = sum(1 for o in scored if o["actual"] is None)
        n_errors = sum(1 for o in outcomes if o["error"])

        result_id = (
            f"{datetime.date.today().isoformat()}_{variant['platform']}_{args.model_config}"
            f"_mmlu-{_DATASET_VERSION}_n{n_evaluated}"
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "result_id": result_id,
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "manifest": quality_manifest(
                backend=variant["backend"], platform=variant["platform"], base_url=coordinator.base_url,
                image=getattr(coordinator, "image", None), command=shlex.join([sys.executable, *sys.argv]),
                model_config_key=args.model_config, target=args.target,
            ),
            "model": {
                "name": variant["model"], "precision": variant["precision"], "backend": variant["backend"],
            },
            "dataset": {
                "name": "MMLU", "version": _DATASET_VERSION, "source": "cais/mmlu",
                "split": "all/test", "n_available": n_available, "n_evaluated": n_evaluated,
                "sampling_method": "full" if (args.limit or n_available) >= n_available else "first-n",
                "sampling_seed": None, "prompt_template_version": _PROMPT_TEMPLATE_VERSION,
            },
            "protocol": {
                "scorer": "generative-letter-parse", "temperature": 0.0, "tool_choice": None,
                "max_tokens": args.max_tokens, "repetitions": 1,
            },
            "scores": {
                "overall_accuracy": (n_correct / len(scored)) if scored else None,
                "n": len(scored),
            },
            "notes": (
                "quantization-regression sanity check, not a model-ranking benchmark - compare same-size "
                "backend/precision pairs against each other, not across sizes. Generative letter-parsing "
                f"scoring, not logprob-based. {n_unparsed} response(s) had no parseable A/B/C/D letter; "
                f"{n_errors} case(s) failed at the server and are excluded from the accuracy figure."
                + (f" Co-resident with: {', '.join(args.co_resident)}." if args.co_resident else "")
            ),
        }

        print(json.dumps(result, indent=2))
        path = write_result(result, results_root=args.results_root or (REPO_ROOT / "results"))
        outcomes_path = path.parent / "outcomes.jsonl"
        outcomes_path.write_text("\n".join(json.dumps(o) for o in outcomes) + "\n")
        result["outcomes_file"] = str(outcomes_path.relative_to(REPO_ROOT))
        path.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {path}")
        _validate(result)
    finally:
        if args.target != "remote":
            print(f"Stopping {variant['backend']} server...")
        coordinator.stop()


def _validate(result: dict) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("! jsonschema not installed - result NOT validated against the schema")
        return
    schema_path = REPO_ROOT / "schemas" / "quality_result.schema.json"
    errors = sorted(
        Draft202012Validator(json.loads(schema_path.read_text())).iter_errors(result),
        key=lambda e: list(e.path),
    )
    if errors:
        print(f"! result does NOT conform to {schema_path.name}:")
        for e in errors[:10]:
            print(f"    {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}")
    else:
        print(f"OK - conforms to {schema_path.name}")


if __name__ == "__main__":
    main()
