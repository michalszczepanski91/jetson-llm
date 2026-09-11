#!/usr/bin/env python3
"""Re-score this lab's recorded BFCL `simple` results with the OFFICIAL
`bfcl-eval` AST checker, to measure how much of the `simple` score is a
false-negative artifact of the simplified checker rather than a model error.

**Why this exists.** `docs/thor-framework-comparison.md` reports that `simple`
is saturated: 7B and 14B fail the same four cases, and in every one the model
called exactly the right function - the failures are argument *formatting*
mismatches (`"x**2"` vs `"lambda x: x**2"`) that
`scripts/validate_tool_calling.py`'s hand-ported matcher does not normalize.
That was estimated as an "~8-point false-negative floor" but never measured,
which left `simple` unusable as a discriminator and left every `simple` number
in this repo carrying an unknown correction. This script measures it.

**Why it is offline and cheap.** It re-scores `outcomes.jsonl` documents that
already exist. `validate_tool_calling.py` records `actual_tool`,
`actual_arguments` and `acceptable_arguments` per case, which is exactly the
official checker's `model_output` / `possible_answer` input - so nothing has to
be re-generated, no server starts, and no GPU is touched. Re-scoring is a pure
function of data already committed.

**The torch hazard, and how this avoids it.** `validate_tool_calling.py`'s
docstring is right that `bfcl_eval` cannot simply be imported into this repo's
venv: `ast_checker` imports `MODEL_CONFIG_MAPPING`, which imports every model
handler it ships, which imports `anthropic`/`torch`/`transformers` - and
installing a generic PyPI `torch` into a Jetson venv is the wheel-shadowing
hazard `embedded-ai-chain/docs/environment.md` warns about. But that chain
exists for exactly ONE line of `ast_checker.py`:

    if MODEL_CONFIG_MAPPING[model_name_escaped].underscore_to_dot:

a model-registry lookup that is meaningless here, because the "model" being
scored is a local server, not one of BFCL's registered handlers. So this script
installs `bfcl-eval` with `--no-deps` into an ISOLATED venv (never this repo's)
and stubs that one module. Everything that actually decides a verdict is the
official code, unmodified.

`underscore_to_dot` is stubbed **False**, which is correct here and not a
convenience: it exists for handlers that rewrite BFCL's dotted function names
(`math.factorial`) into underscored ones because some provider APIs reject
dots. This lab passes tool names through untouched - the recorded outcomes
contain `math.factorial` and `math.sum` with their dots intact - so no
conversion back is wanted.

**That stub was verified, not assumed, and it is load-bearing.** Re-running
every set with it flipped to `True` moves the official score from 96% to 64% on
the Thor 7B set and changes all 10 re-scorable sets - because the checker then
underscores the expected name and every correctly-dotted call stops matching.
So `False` is not a harmless default: it is the value that makes the official
checker agree with reality on this pipeline's data, and the 34-point swing is
the evidence.

One real subtlety: the official checker expects BFCL's ORIGINAL JSON-Schema
type names (`dict`, `float`, `tuple`, `any` - its PYTHON_TYPE_MAPPING lists
them). `validate_tool_calling.py` renames those before sending them to a server
(llama.cpp's grammar builder crashes otherwise). So the function description is
re-loaded from the staged dataset rather than taken from the result document.

Setup (once):

    uv venv /tmp/bfclenv --python 3.12
    uv pip install --python /tmp/bfclenv/bin/python --no-deps bfcl-eval

Usage:

    /tmp/bfclenv/bin/python scripts/rescore_bfcl_official.py \\
        --results-root results/raw --out output/bfcl_official_rescore.json
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path("/opt/datasets/BFCL")


def _install_model_config_stub() -> None:
    """Replace the one module whose import chain pulls in torch. See docstring."""
    stub = types.ModuleType("bfcl_eval.constants.model_config")

    class _Cfg:
        underscore_to_dot = False  # see module docstring - deliberate, not a default

    class _Mapping(dict):
        def __getitem__(self, _key):
            return _Cfg()

        def get(self, _key, _default=None):
            return _Cfg()

    stub.MODEL_CONFIG_MAPPING = _Mapping()
    stub.ModelConfig = _Cfg
    sys.modules["bfcl_eval.constants.model_config"] = stub


def _load_official():
    _install_model_config_stub()
    try:
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    except ImportError as exc:
        raise SystemExit(
            f"cannot import bfcl_eval ({exc}).\n"
            "This script MUST run under an isolated venv, never this repo's:\n"
            "    uv venv /tmp/bfclenv --python 3.12\n"
            "    uv pip install --python /tmp/bfclenv/bin/python --no-deps bfcl-eval\n"
            "    /tmp/bfclenv/bin/python scripts/rescore_bfcl_official.py ..."
        ) from exc
    return ast_checker, Language


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing BFCL data: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rescore(outcomes_path: Path, funcs_by_id: dict, answers_by_id: dict,
            ast_checker, Language) -> dict | None:
    """Returns the comparison for one result directory, or None if it has no
    `simple` cases to re-score."""
    rows = []
    for line in outcomes_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("category") != "simple":
            continue
        case_id = o["id"]
        if case_id not in funcs_by_id:
            continue

        ours = bool(o.get("correct"))
        # A case where no tool was called at all is a genuine miss under either
        # checker - there is nothing to hand the AST matcher.
        if not o.get("tool_called") or not o.get("actual_tool"):
            rows.append({"id": case_id, "ours": ours, "official": False,
                         "official_error_type": "no_tool_call", "agree": ours is False})
            continue

        # Results written before 2026-09-09 record only the VERDICT
        # (correct_arguments: true/false), not the arguments themselves, so
        # there is nothing for an independent checker to re-examine. Skipped
        # explicitly rather than scored against empty dicts - doing that
        # silently reports a confident 0%, which is what the first run of this
        # script did before this guard existed.
        if "actual_arguments" not in o or "acceptable_arguments" not in o:
            return "NOT_RESCORABLE"

        model_output = [{o["actual_tool"]: o.get("actual_arguments") or {}}]
        possible_answer = [{o["actual_tool"]: (o.get("acceptable_arguments") or {})}]
        try:
            verdict = ast_checker(
                [funcs_by_id[case_id]], model_output, possible_answer,
                Language.PYTHON, "simple", "local-server",
            )
            official = bool(verdict.get("valid"))
            err = verdict.get("error_type")
        except Exception as exc:  # noqa: BLE001 - a checker crash is a result
            official, err = False, f"checker_exception:{type(exc).__name__}"
        rows.append({"id": case_id, "ours": ours, "official": official,
                     "official_error_type": err, "agree": ours == official})

    if not rows:
        return None
    n = len(rows)
    return {
        "n": n,
        "ours_accuracy": sum(r["ours"] for r in rows) / n,
        "official_accuracy": sum(r["official"] for r in rows) / n,
        "disagreements": [r for r in rows if not r["agree"]],
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=str(REPO / "results" / "raw"))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out", default=str(REPO / "output" / "bfcl_official_rescore.json"))
    args = p.parse_args()

    ast_checker, Language = _load_official()

    data = Path(args.data_dir)
    funcs_by_id = {c["id"]: c["function"][0] for c in _load_jsonl(data / "BFCL_v3_simple.json")}
    answers_by_id = {c["id"]: c for c in _load_jsonl(data / "possible_answer" / "BFCL_v3_simple.json")}

    report = {}
    skipped: list[str] = []
    for outcomes in sorted(Path(args.results_root).rglob("outcomes.jsonl")):
        res = rescore(outcomes, funcs_by_id, answers_by_id, ast_checker, Language)
        if res is None:
            continue
        key = outcomes.parent.name
        if res == "NOT_RESCORABLE":
            skipped.append(key)
            print(f"{key[:62]:64s} SKIPPED - pre-2026-09-09 format, arguments not retained")
            continue
        report[key] = res
        print(f"{key[:62]:64s} ours={res['ours_accuracy']:.0%} "
              f"official={res['official_accuracy']:.0%} "
              f"delta={(res['official_accuracy']-res['ours_accuracy'])*100:+.1f}pt "
              f"n={res['n']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rescored": report, "not_rescorable": skipped}, indent=2))
    print(f"\nwrote {out}: {len(report)} re-scored, {len(skipped)} not re-scorable")


if __name__ == "__main__":
    main()
