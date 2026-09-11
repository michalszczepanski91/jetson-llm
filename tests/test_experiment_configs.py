"""Tests for the experiment-config layer - docs/TODO.md Phase 3.

The point of separating configs/benchmarks/ from configs/models.yaml is that
an experiment becomes re-runnable from (model config key, experiment config,
git SHA) alone, with no CLI flag carrying experimental meaning (GATE 3).
That only holds if the configs themselves are checked, so this module
validates every shipped config against experiment.schema.json and pins the
grid-expansion semantics that turn one into a set of cells.

No Docker/GPU needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from runner import Cell, build_prompt, cell_salt, expand_grid  # noqa: E402

import json  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "benchmarks"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
MODEL_KEYS = set(yaml.safe_load((REPO_ROOT / "configs" / "models.yaml").read_text()))


def test_configs_exist():
    """Zero-parameter parametrisation passes silently, so assert the set."""
    assert {p.stem for p in CONFIGS} == {
        "smoke", "output_sweep", "context_sweep", "scorecard", "p5_cross_platform",
        "precision_arm", "context_sweep_thor",
    }


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_matches_schema(path: Path):
    schema = json.loads((REPO_ROOT / "schemas" / "experiment.schema.json").read_text())
    cfg = yaml.safe_load(path.read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(cfg), key=lambda e: list(e.path))
    assert not errors, f"{path.name}: " + "; ".join(
        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_names_real_candidates(path: Path):
    """A typo'd model key would fail only after a campaign started - and on a
    grid runner that starts one server per model, potentially minutes in."""
    cfg = yaml.safe_load(path.read_text())
    unknown = set(cfg["models"]) - MODEL_KEYS
    assert not unknown, f"{path.name} references models not in configs/models.yaml: {unknown}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_filename_matches_its_name(path: Path):
    """`name` is a component of every experiment_id the config generates, so a
    mismatch would make results untraceable back to the file that produced
    them."""
    assert yaml.safe_load(path.read_text())["name"] == path.stem


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_input_lengths_fit_the_candidates_context(path: Path):
    """docs/note.md: an input longer than a candidate's context is an error,
    not a silent truncation. Every current row pins 2048, so a config that
    asked for 4096 would quietly measure a truncated prompt on some backends.
    Leaves headroom for the chat template and the per-run marker."""
    cfg = yaml.safe_load(path.read_text())
    rows = yaml.safe_load((REPO_ROOT / "configs" / "models.yaml").read_text())
    longest = max(cfg["workload"].get("input_tokens") or [0])
    longest_out = max(cfg["workload"]["output_tokens"])
    for key in cfg["models"]:
        row = rows[key]
        ctx = row.get("max_model_len") or row.get("ctx_size")
        assert ctx is not None, f"{key} declares no context length"
        assert longest + longest_out <= ctx, (
            f"{path.name}: input {longest} + output {longest_out} exceeds {key}'s context {ctx}"
        )


# --- grid expansion --------------------------------------------------------


def test_grid_is_the_cartesian_product():
    cells = expand_grid({"input_tokens": [128, 512], "output_tokens": [16, 32],
                         "batch_size": [1], "concurrency": [1]})
    assert len(cells) == 4
    assert (cells[0].input_tokens, cells[0].output_tokens) == (128, 16)
    assert (cells[-1].input_tokens, cells[-1].output_tokens) == (512, 32)


def test_grid_order_is_deterministic():
    """An interrupted campaign resumes by skipping results that already
    exist, which only works if cell order is stable across invocations."""
    workload = {"input_tokens": [512, 128], "output_tokens": [32, 16],
                "batch_size": [1], "concurrency": [1]}
    assert expand_grid(workload) == expand_grid(workload)
    assert [c.input_tokens for c in expand_grid(workload)] == [512, 512, 128, 128]


def test_absent_input_axis_yields_one_cell_with_no_target():
    """`input_tokens` is optional - omitting it means the fixed historical
    prompt, not zero-length input."""
    cells = expand_grid({"output_tokens": [128], "batch_size": [1], "concurrency": [1]})
    assert cells == [Cell(output_tokens=128, input_tokens=None)]


def test_batch_and_concurrency_are_separate_axes():
    """docs/note.md 13 forbids conflating them: they are different
    experimental conditions that happen to produce similar throughput curves."""
    cells = expand_grid({"output_tokens": [128], "batch_size": [1, 2], "concurrency": [1, 4]})
    assert {(c.batch_size, c.concurrency) for c in cells} == {(1, 1), (1, 4), (2, 1), (2, 4)}


# --- prompt construction ---------------------------------------------------


def test_uniqueness_marker_goes_at_the_front():
    """Both backends cache by prompt PREFIX, so a marker appended at the end
    would leave everything before it reusable and defeat nothing. Measured
    consequence of getting this wrong: TTFT p50 40.8ms vs 268.1ms."""
    prompt, _ = build_prompt(None, run_index=7)
    assert prompt.startswith("[run 7 ")


def test_prompts_differ_between_runs_but_are_stable_within_one():
    a, _ = build_prompt(128, run_index=1)
    b, _ = build_prompt(128, run_index=2)
    assert a != b
    assert a == build_prompt(128, run_index=1)[0]


def test_cells_sharing_an_input_length_do_not_share_prompts():
    """The regression guard for the 2x TTFT error of 2026-09-04.

    `build_prompt` is a function of input length and run index, so every cell
    of `output_sweep` (input held at 512, output varied) emitted a
    byte-identical prompt sequence. vLLM's automatic prefix cache served
    cells 1-5 out of cell 0's blocks: TTFT p50 74.9ms then ~30ms, a step at
    the cell boundary. The published `1.5b-awq-vllm-orin` TTFT of 31.1ms was
    that artifact; the honest number is ~66ms. Nothing about the marker being
    at the front prevents this - the marker was simply the same in both cells."""
    a = Cell(output_tokens=16, input_tokens=512)
    b = Cell(output_tokens=128, input_tokens=512)
    assert cell_salt(a) != cell_salt(b)
    for run_index in (0, 1, 29):
        pa, _ = build_prompt(512, run_index=run_index, salt=cell_salt(a))
        pb, _ = build_prompt(512, run_index=run_index, salt=cell_salt(b))
        assert pa != pb, f"cells share a prompt at run {run_index}"
        # fixed-width salt: the cells must still be the same workload point
        assert len(pa) == len(pb)


def test_prompt_regime_is_part_of_the_experiment_id():
    """Two materially different experiments must never share an identity.

    The prompt regime changes what is measured, not just how it is labelled -
    prompt_uniqueness alone was measured at 6.6x TTFT on this device, and the
    v1->v2 template change was made because v1 let cells share prefix-cache
    state. Before 2026-09-11 neither was in the ID, so a v1 and a v2 run of the
    same cell at the same replicate on the same day collided and write_result()
    would refuse the second as a duplicate - the immutability guard rejecting a
    new experiment rather than protecting an old one."""
    from datetime import date
    from manifest import experiment_id
    from runner import _prompt_regime

    def eid(regime):
        return experiment_id(
            platform="orin", model_config_key="k", experiment="e",
            input_tokens=512, output_tokens=128, replicate=1,
            prompt_regime=regime, on=date(2026, 9, 4),
        )

    # The default keeps its bare ID, so every result already on disk stays
    # reproducible - including results taken at template v2 before the ID
    # gained this component at all.
    assert eid(None) == "2026-09-04_orin_k_e_in512_out128_bs1_r01"
    assert eid(_prompt_regime("unique-per-run")) == eid(None)
    # The reachable axis is a different identity, not a relabelling.
    assert eid(_prompt_regime("identical-per-run")) != eid(None)

    # The template version must NOT reach the ID: it is a module constant, so
    # no pair of same-day runs can differ on it, and stamping it on every ID
    # made one ID shape mean two regimes. Recorded in the workload block
    # instead, which is what an analysis joins on.
    from runner import PROMPT_TEMPLATE_VERSION
    assert PROMPT_TEMPLATE_VERSION not in eid(_prompt_regime("identical-per-run"))


def test_cell_salt_is_deterministic():
    """A cell stays reproducible from its identity - the salt is a
    disambiguator, not a nonce."""
    c = Cell(output_tokens=128, input_tokens=512)
    assert cell_salt(c) == cell_salt(Cell(output_tokens=128, input_tokens=512))
    assert cell_salt(c, replicate=1) != cell_salt(c, replicate=2)


def test_no_run_index_means_no_marker():
    """identical-per-run must reproduce the historical prompt exactly, so the
    older runs on record stay comparable with a deliberate warm-cache cell."""
    prompt, source = build_prompt(None)
    assert "[run" not in prompt
    assert source.startswith("fixed_transcript_")


def test_prompt_source_distinguishes_fixed_from_built():
    assert build_prompt(None)[1].startswith("fixed_transcript_")
    assert build_prompt(512)[1].startswith("repeated_filler_")


def test_longer_targets_produce_longer_prompts():
    """Approximate by construction - there is no host tokenizer - but it must
    at least be monotonic, or a context sweep would not be sweeping anything."""
    assert len(build_prompt(128)[0]) < len(build_prompt(512)[0]) < len(build_prompt(2048)[0])
