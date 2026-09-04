"""Schema tests - the thing that makes schemas/ load-bearing rather than
decorative.

The files in schemas/ started life (2026-09-04, alongside docs/note.md) as
example *instances*: JSON documents that looked like a result, with no
$schema/type/properties and therefore nothing a validator could check. They
are now real JSON Schema (draft 2020-12), and those original sketches live on
as schemas/examples/*.example.json.

Three things are checked here, in increasing order of usefulness:

  1. every schema is itself a valid JSON Schema (catches a typo'd keyword,
     which otherwise fails *open* - an unknown keyword is silently ignored by
     a validator, so a broken schema quietly validates everything);
  2. each schema's own `examples` validate against it (a schema whose own
     example doesn't pass is documentation that lies);
  3. **every row of configs/models.yaml validates against
     model.schema.json** - the only one of the three that guards live data
     today, and the reason model.schema.json's `required` list is exactly
     what every current row already satisfies rather than what
     docs/note.md §28 eventually wants (see schemas/README.md's conformance
     levels).

No Docker/GPU needed, same as the rest of tests/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "schemas"
SCHEMA_FILES = sorted(SCHEMA_DIR.glob("*.schema.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_schema_files_exist():
    """Guards against a rename silently emptying the parametrised tests below
    - pytest reports zero-parameter parametrisation as a pass, not a failure."""
    names = {p.name for p in SCHEMA_FILES}
    assert names == {
        "benchmark_result.schema.json",
        "experiment.schema.json",
        "model.schema.json",
        "quality_result.schema.json",
    }, names


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_is_valid_json_schema(path: Path):
    Draft202012Validator.check_schema(_load(path))


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_declares_identity(path: Path):
    """$schema/$id/title/description are what make these self-describing to
    anyone who opens one without this repo's context - and $id is what a
    future cross-file $ref would resolve against, if the no-resolver rule is
    ever relaxed."""
    schema = _load(path)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert schema.get("$id", "").endswith(path.name)
    assert schema.get("title")
    assert schema.get("description")


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_examples_validate(path: Path):
    schema = _load(path)
    validator = Draft202012Validator(schema)
    for i, example in enumerate(schema.get("examples", [])):
        errors = sorted(validator.iter_errors(example), key=lambda e: e.path)
        assert not errors, f"{path.name} example[{i}]: " + "; ".join(
            f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in errors
        )


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_has_no_bare_cross_file_refs(path: Path):
    """schemas/ is deliberately self-contained: every $ref is a local
    "#/$defs/..." pointer, never another file. Cross-file $refs would need a
    registry/resolver to validate, which breaks this repo's clone-and-run
    rule (the same reason benchmarks/harness.py is a hand-maintained copy
    rather than an import). Duplicating a few $defs is the cheaper trade."""
    text = path.read_text()
    for ref in [line for line in text.splitlines() if '"$ref"' in line]:
        assert '"#/' in ref, f"{path.name}: non-local $ref: {ref.strip()}"


# --- Example documents ---------------------------------------------------

TARGET_EXAMPLES = sorted((SCHEMA_DIR / "examples").glob("*.target.json"))


def test_target_examples_exist():
    assert {p.name for p in TARGET_EXAMPLES} == {
        "benchmark_result.target.json",
        "quality_result.target.json",
    }


@pytest.mark.parametrize("path", TARGET_EXAMPLES, ids=lambda p: p.name)
def test_target_example_validates(path: Path):
    """`*.target.json` are complete, conformant documents showing the shape the
    retrofit is aiming at - so they must validate, and this test is what keeps
    them honest as the schemas evolve.

    Their siblings `*.example.json` are deliberately NOT checked: those are the
    original 2026-09-04 sketches, kept verbatim as the historical record of
    what was proposed before the schemas existed. They do not validate against
    anything, which is precisely why the schemas were written."""
    schema_name = path.name.replace(".target.json", ".schema.json")
    validator = Draft202012Validator(_load(SCHEMA_DIR / schema_name))
    errors = sorted(validator.iter_errors(_load(path)), key=lambda e: list(e.path))
    assert not errors, f"{path.name}: " + "; ".join(
        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
    )


def test_original_sketches_are_preserved():
    """The pre-schema sketches must not be quietly deleted once superseded -
    same append-only reasoning docs/TODO.md and configs/models.yaml already
    apply to their own history."""
    kept = {p.name for p in (SCHEMA_DIR / "examples").glob("*.example.json")}
    assert kept == {
        "benchmark_result.example.json",
        "experiment.example.json",
        "model.example.json",
    }


# --- The one that guards live data --------------------------------------

MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"
MODEL_ROWS = sorted(yaml.safe_load(MODELS_YAML.read_text()).items())


@pytest.mark.parametrize("key,row", MODEL_ROWS, ids=[k for k, _ in MODEL_ROWS])
def test_models_yaml_row_matches_schema(key: str, row: dict):
    validator = Draft202012Validator(_load(SCHEMA_DIR / "model.schema.json"))
    errors = sorted(validator.iter_errors(row), key=lambda e: e.path)
    assert not errors, f"{key}: " + "; ".join(
        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
    )


def test_models_yaml_backend_specific_args_are_not_crossed():
    """Explicit regression test for the conditional in model.schema.json: a
    llama.cpp arg on a vLLM row (or vice versa) is accepted by the loader,
    ignored by the coordinator, and looks authoritative in the registry
    forever. Asserted here as a named behaviour rather than left implicit in
    the row-by-row test above, since it's the schema's only real logic."""
    validator = Draft202012Validator(_load(SCHEMA_DIR / "model.schema.json"))
    vllm_row = {
        "model": "x/y",
        "precision": "AWQ 4-bit",
        "backend": "vllm",
        "platform": "orin",
        "notes": "untested",
        "n_gpu_layers": -1,  # llama.cpp-only - must be rejected here
    }
    assert list(validator.iter_errors(vllm_row))

    llamacpp_row = {
        "model": "x/y",
        "precision": "GGUF Q4_K_M",
        "backend": "llama-cpp",
        "platform": "orin",
        "notes": "untested",
        "gpu_memory_utilization": 0.3,  # vLLM-only - must be rejected here
    }
    assert list(validator.iter_errors(llamacpp_row))
