#!/usr/bin/env python3
"""Run a whole experiment from configs/benchmarks/<name>.yaml - docs/TODO.md
Phase 3.

This is what GATE 3 asks for: a benchmark re-runnable from
`(configs/models.yaml key, configs/benchmarks/*.yaml, git SHA)` alone, with
no CLI flag carrying experimental meaning. Everything that defines the
experiment - which candidates, which token grids, sampling, warmup and
repetition floors, standalone vs co-resident - lives in the config file and
is recorded in every result's manifest.

One server session per model, not per cell. A vLLM cold start is ~136s on
this hardware, so a 12-cell grid that restarted per cell would spend 27
minutes starting servers. Cells sharing a session share one cold-start
measurement, and each such result carries
`cold_start_shared_across_cells_in_this_server_session` so an analysis can
never mistake N cells for N independent cold starts.

Failures are per-cell, not per-campaign: a candidate that OOMs at 2048
context is a real result (docs/note.md §54 - record the OOM, don't skip it),
and the remaining cells still run. A cell whose result file already exists
is skipped rather than overwritten, so an interrupted campaign resumes.

Usage:
    uv run python scripts/run_experiment.py configs/benchmarks/smoke.yaml
    uv run python scripts/run_experiment.py configs/benchmarks/output_sweep.yaml --dry-run
"""

import argparse
import json
import shlex
import sys
import traceback
from pathlib import Path

import yaml

sys.stdout.reconfigure(line_buffering=True)  # a redirected/backgrounded run fully
# buffers stdout otherwise, which silently hid a live campaign's progress from a
# `tail -f` monitor for 20+ minutes on 2026-09-04 - the process was fine, only the
# log looked idle.

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness import TegrastatsSampler  # noqa: E402
from llm_coordinator import add_target_args, build_coordinator  # noqa: E402
from manifest import assert_condition_matches_reality, write_result  # noqa: E402
from model_config import load_model_config  # noqa: E402
from runner import expand_grid, measure_cell  # noqa: E402

_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="path to a configs/benchmarks/*.yaml file")
    p.add_argument("--replicate", type=int, default=1,
                   help="replicate number for every cell in this run; re-running a campaign "
                        "means a new replicate, never an overwrite")
    p.add_argument("--results-root", default=None, help="default: <repo>/results")
    p.add_argument("--only-model", default=None, help="restrict to one models: entry, for debugging")
    p.add_argument("--dry-run", action="store_true",
                   help="expand and validate the grid, start nothing")
    p.add_argument("--ready-timeout", type=float, default=900.0)
    add_target_args(p)
    return p.parse_args()


def load_experiment(path: Path) -> dict:
    """Load and validate an experiment config. Validation is not optional
    politeness: an experiment config is the definition of what a campaign
    measured, and a typo'd axis name that silently expands to nothing would
    produce a campaign that looks complete and measured the wrong thing."""
    cfg = yaml.safe_load(path.read_text())
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("! jsonschema not installed - experiment config NOT validated")
        return cfg
    schema = json.loads((REPO_ROOT / "schemas" / "experiment.schema.json").read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(cfg), key=lambda e: list(e.path))
    if errors:
        detail = "\n".join(
            f"  {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise SystemExit(f"{path} does not conform to experiment.schema.json:\n{detail}")
    return cfg


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_experiment(cfg_path)

    cells = expand_grid(cfg["workload"])
    models = [m for m in cfg["models"] if not args.only_model or m == args.only_model]
    if not models:
        raise SystemExit(f"--only-model {args.only_model!r} matches nothing in {cfg_path}")

    condition = cfg["execution_condition"]
    co_resident = cfg.get("co_resident_workload", [])
    if condition == "co-resident" and not co_resident:
        raise SystemExit(
            f"{cfg_path}: execution_condition is co-resident but co_resident_workload is empty. "
            '"Under load" is not a reproducible condition.'
        )
    # Pre-flight, before anything expensive starts: a `standalone` claim on a
    # board with other containers running is a mislabel, and mislabelled
    # results contaminate every table they join.
    if args.target == "local":
        assert_condition_matches_reality(
            condition,
            own_containers={"vllm-llm-lab", "llamacpp-llm-lab"},
            declared_co_resident=co_resident,
        )
    reps = cfg["repetitions"]
    command = shlex.join([sys.executable, *sys.argv])
    results_root = args.results_root or (REPO_ROOT / "results")

    print(f"experiment : {cfg['name']} (tier={cfg.get('tier', 'unspecified')})")
    print(f"condition  : {condition}{' + ' + ', '.join(co_resident) if co_resident else ''}")
    print(f"grid       : {len(models)} model(s) x {len(cells)} cell(s) = {len(models) * len(cells)} results")
    for c in cells:
        print(f"    in={c.input_tokens if c.input_tokens is not None else 'fixed':<6} "
              f"out={c.output_tokens:<5} bs={c.batch_size} conc={c.concurrency}")
    if args.dry_run:
        print("\n--dry-run: nothing started.")
        return

    written, skipped, failed = [], [], []
    for model_key in models:
        variant = load_model_config(model_key)
        local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
        local_kwargs["extra_args"] = variant.get("extra_args")
        coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)

        print(f"\n=== {model_key} ===")
        sampler = TegrastatsSampler()
        sampler.start()
        try:
            print(f"starting {variant['backend']}...")
            coordinator.start()
            if not coordinator.wait_ready():
                raise TimeoutError(f"{variant['backend']} server did not become ready")
            cold_start = coordinator.cold_start_breakdown()
            print(f"cold start: {cold_start['total_s']}s (weights_cached={cold_start['weights_cached']})")

            for idx, cell in enumerate(cells):
                label = f"in={cell.input_tokens or 'fixed'} out={cell.output_tokens}"
                try:
                    result = measure_cell(
                        coordinator=coordinator,
                        variant=variant,
                        model_config_key=model_key,
                        cell=cell,
                        sampler=sampler,
                        cold_start=cold_start,
                        execution_condition=condition,
                        co_resident=co_resident,
                        experiment=cfg["name"],
                        command=command,
                        experiment_config=str(cfg_path),
                        temperature=cfg["sampling"]["temperature"],
                        seed=cfg["sampling"].get("seed"),
                        runs=reps["measurement_runs"],
                        warmup_min=reps.get("warmup_min", 3),
                        warmup_max=reps.get("warmup_max", 20),
                        min_measurement_s=reps.get("min_measurement_s", 10.0),
                        replicate=args.replicate,
                        cell_index=idx,
                        target=args.target,
                    )
                    path = write_result(result, results_root=results_root)
                    written.append(path)
                    _print_cell(label, result)
                except FileExistsError:
                    # An interrupted campaign resumes rather than either
                    # overwriting data or refusing to continue.
                    print(f"  {label:<28} SKIP (result already exists - bump --replicate to redo)")
                    skipped.append((model_key, label))
                except Exception as exc:  # noqa: BLE001
                    print(f"  {label:<28} FAIL {type(exc).__name__}: {exc}")
                    failed.append((model_key, label, f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # noqa: BLE001 - a model that won't serve is a result
            print(f"  !! {model_key} did not serve: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            failed.append((model_key, "<server>", f"{type(exc).__name__}: {exc}"))
        finally:
            sampler.stop()
            coordinator.stop()

    print(f"\n{'=' * 60}")
    print(f"wrote {len(written)} result(s), skipped {len(skipped)}, failed {len(failed)}")
    for m, label, err in failed:
        print(f"  FAILED {m} {label}: {err}")
    if failed:
        print("\nA failure here is a result, not a crash - record it in docs/TODO.md rather "
              "than re-running until it passes (docs/note.md §54).")


def _print_cell(label: str, result: dict) -> None:
    a, p = result["aggregates"], result["power"]
    ttft = a.get("ttft_ms", {}).get("p50")
    dec = a.get("decode_tok_s", {}).get("p50")
    jtok = p.get("energy_per_output_token_j")
    print(f"  {label:<28} ttft_p50={ttft:7.1f}ms  decode_p50={dec:6.1f}tok/s  "
          f"J/tok={jtok if jtok is not None else float('nan'):.4f}  "
          f"n={result['validity']['measurement_runs']}"
          + ("  [" + ",".join(result["validity"]["flags"]) + "]" if result["validity"]["flags"] else ""))


if __name__ == "__main__":
    main()
