# schemas/

Machine-readable contracts for everything this lab produces. Four schemas, JSON
Schema draft 2020-12, validated in `tests/test_schemas.py` (`make test` — no
Docker/GPU needed).

| File | Describes | Validated against live data today? |
|---|---|---|
| `model.schema.json` | one row of `configs/models.yaml` — a candidate (model × precision × backend × platform) | **Yes** — every one of the 9 current rows |
| `experiment.schema.json` | one file in `configs/benchmarks/` — the declarative "what to measure" | Not yet; `configs/benchmarks/` is created in `docs/TODO.md` Phase 3 |
| `benchmark_result.schema.json` | one performance cell: candidate × workload point, raw runs + aggregates | Not yet; it is the contract Phase 2 implements |
| `quality_result.schema.json` | one accuracy evaluation (BFCL, MMLU) | Not yet; Phase 5 brings the scripts into conformance |

## Why these exist

`docs/note.md` §3 is the governing rule: *a number without its experimental
context is not a result*. A schema is how that rule gets enforced by something
other than discipline. Two failure modes in particular are what the `required`
lists are aimed at:

- **Aggregates without raw data.** `benchmark_result` requires a `runs` array
  holding every measured repetition. Percentiles can always be recomputed from
  raw runs; raw runs cannot be recovered from percentiles. Today's
  `benchmarks/harness.py` computes `percentiles(latencies)` and then discards
  `latencies` — which is exactly the loss this requirement prevents.
- **Unlabelled execution conditions.** `execution_condition` (`standalone` /
  `co-resident`) has no default anywhere in these schemas. On a unified-memory
  board those are different physical quantities, and a table that silently mixes
  them is worse than no table.

## Conformance levels

The schemas are written as the **target**, not as a description of today's
output — with one deliberate exception.

- **`model.schema.json` is level-1 today.** Its `required` list is exactly what
  all 9 existing `configs/models.yaml` rows already satisfy (`model`,
  `precision`, `backend`, `platform`, `notes`), so the test suite guards live
  data from the first commit rather than failing until a backfill lands. The
  `docs/note.md` §28 additions — `family`, `parameters_b`, `revision`, `status`,
  `license`, `quantization{}` — are present, documented, and marked "Phase 3
  target" in their own `description`. They become `required` when the registry
  is backfilled, not before.
- **The other three are level-2.** They describe what `docs/TODO.md` Phases 2/3/5
  produce. Validating today's output against them would fail, which is the
  point: the gap is the work list.

Writing a schema that merely describes today's output would have made it a
report rather than a contract.

## Design rules

**Self-contained — no cross-file `$ref`.** Every `$ref` is a local
`#/$defs/...` pointer. Cross-file references need a registry/resolver to
validate, which breaks this repo's clone-and-run property (the same reasoning
that makes `benchmarks/harness.py` a hand-maintained copy rather than an
import). A little duplication between `percentiles` blocks is the cheaper
trade, and `tests/test_schemas.py` enforces it.

**`additionalProperties: false` everywhere.** A typo'd field name should fail
loudly at write time. The cost is that adding a field means editing the schema —
which is intended: a new field in a result *is* a change to the experimental
record.

**Quality and performance stay separate schemas.** `docs/note.md` §5 keeps the
measurement layers independently measurable, and §34 warns against collapsing
them into one score before the trade-off is characterised. The two document
types join at analysis time on `manifest.model_config_key`, not at write time.

**Descriptions carry the reasoning, not just the type.** Several fields exist
because of a specific incident in this lab, and the description says so — e.g.
`software.backend_version` must be read from the running server because
`dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04` is a tag rather than a llama.cpp
version, and builds 4579/5058/5283 had to be told apart by behaviour. Those
notes are the difference between a schema someone follows and one someone works
around.

**`_comment`** is an allowed string on both result schemas. JSON has no comment
syntax, and the example files need their "these numbers are illustrative"
caveat to travel with the file.

## `examples/`

| File | What it is |
|---|---|
| `benchmark_result.example.json`, `experiment.example.json`, `model.example.json` | The **original 2026-09-04 sketches**, kept verbatim. These were instance documents named `*.schema.json` — no `$schema`, `type`, or `properties`, so nothing could validate against them. Preserved rather than deleted, per the same append-only convention `docs/TODO.md` and `configs/models.yaml` apply to their own history. They do **not** validate against anything. |
| `benchmark_result.target.json` | A complete, conformant performance result. Numbers are **illustrative, not measured**. |
| `quality_result.target.json` | A complete, conformant quality result. Its **scores are real** (85% simple / 65% irrelevance / 75% overall, n=40, `1.5b-q4-llamacpp-orin`, 2026-09-04); the surrounding metadata is reconstructed to show the target shape. |

The `*.target.json` files are validated by the test suite; the `*.example.json`
sketches are deliberately not.

## Versioning

Every document carries `schema_version` (semver). Bump it when the shape
changes, and keep old campaigns readable rather than rewriting their files —
`docs/note.md` §29 makes historical results immutable. The schemas themselves
are versioned by the repo's git history; a result's `manifest.git_commit`
resolves which schema version it was written against.
