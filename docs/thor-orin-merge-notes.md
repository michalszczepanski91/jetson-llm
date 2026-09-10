# Merging thor-edge-llm and phase2-measurement-integrity

**Delete this file once the merge is done** — it's a one-time handoff note, not
permanent documentation. Written from the Thor side, diffing both branches against
their common base (`98c1aac`), so whoever does the merge doesn't start cold.

## The conflict map

| File | What `phase2-measurement-integrity` did | Risk |
|---|---|---|
| `configs/models.yaml` | Prepends an "EXCLUDED" note to two *existing* rows (`bielik-11b-awq-vllm-orin`, `apertus-8b-q4-llamacpp-orin`), both near the top of the file | **Near zero.** `thor-edge-llm`'s changes are all new rows appended lower down. Should merge cleanly on its own. |
| `src/llm_coordinator.py` | Adds a `_ColdStartMixin` and makes `VllmCoordinator`/`LlamaCppCoordinator` inherit from it, with new lines in their `__init__`/`start()` for cold-start telemetry | **Real, but mechanical.** `thor-edge-llm` extended the *same two classes'* `__init__`/`start()` independently — `extra_volumes` (a vLLM memory-profiling workaround) and `server_argv0` (llama.cpp image-family differences). Orthogonal features on the same lines: both sets of insertions need to end up coexisting, not chosen between. `EdgeLlmCoordinator` and `build_coordinator()`'s `edge-llm` dispatch are new code Orin's branch never touches — no risk there. |
| `docs/TODO.md` | Rewrote roughly two-thirds of the file (476 insertions / 520 deletions of ~757 lines) | **The real merge job.** Not a line-diff — take Orin's new structure as the base and manually re-insert `thor-edge-llm`'s Phase 6 summary (already condensed to ~12 lines) into wherever it belongs there. |
| `docs/thor-framework-comparison.md`, `docs/thor-precision-sweep.html` | Doesn't exist on that branch | **Zero** — new files. |

## Suggested sequence

```bash
git fetch origin
git checkout -b integrate-orin-thor origin/main
git merge origin/phase2-measurement-integrity   # bigger/structural — merge first
git merge origin/thor-edge-llm                  # resolve the two real conflicts here
uv run pytest tests/ -q                          # both platforms' tests must pass together
```

Then push `integrate-orin-thor`, review, and fast-forward `main` — at which point
`thor-edge-llm` and `phase2-measurement-integrity` are both done and can be deleted
(or kept as historical record; your call).

## What's on `thor-edge-llm`, if you need the full story

`docs/thor-framework-comparison.md` on that branch has it all: current recommendation
(TensorRT Edge-LLM + Qwen2.5-7B, GPTQ-Int4 quantized), the framework comparison, and
a real bug found and patched in Edge-LLM's own source (`patches/`). `README.md` and
`CLAUDE.md` on `main` already point here — this file exists only for the specific
merge mechanics those don't cover.
