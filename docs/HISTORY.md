# HISTORY — detailed incident and finding records

Companion to `docs/TODO.md`. That file is now a lean status tracker (objective, checklist,
key results, gate) — this file holds the full narrative behind each finding: root causes,
exact diagnostics, numbers, and what was tried before the fix that worked. Split out
2026-09-07 because `docs/TODO.md` had grown to ~980 lines of inline incident prose and was
no longer scannable. Nothing here is new; it was moved, not rewritten, from `docs/TODO.md`
(see that file's git history for the pre-split version if you need the exact original
wording). Organized by phase, matching `docs/TODO.md`'s numbering.

---

## Phase 1 — Foundations: smoke-test records

### Qwen2.5-1.5B on vLLM (`1.5b-awq-vllm-orin`), 2026-09-04

Real Docker/GPU smoke test (`VllmCoordinator` directly, not `make serve-1.5b-vllm`'s
foreground compose target): cold start 158-162s, `/health` 200, a plain completion and a
tool-calling completion (`get_weather({"city": "Warsaw"})`, correctly structured) both
succeeded. Ran alongside the real production `vllm-orchestrator` container (port 8001,
untouched) with no collision — confirmed `free -h`/`docker ps` before starting.

Two real bugs this test caught and fixed before it passed:
1. `configs/models.yaml`'s row claimed `max_model_len: 4096` as "known-good" —
   `docker inspect vllm-orchestrator` showed production actually runs `2048`. Fixed to
   match the real value; the 4096 guess was never sourced from `orchestrator_models.py`
   (which has no `max_model_len` field at all) and shouldn't have been asserted as
   "known-good" unverified.
2. `VllmCoordinator.start()`'s own command never included `--enable-auto-tool-choice
   --tool-call-parser hermes` — only `docker-compose.yml` had them. Every real
   tool-calling request failed with HTTP 400 until fixed (the class's own smoke test
   caught the mismatch between its two ways of starting the container).

Also found (unrelated to vLLM): `LlamaCppCoordinator`'s default port 8080 collides with
`embedded-ai-chain`'s own live dashboard on this device — changed the default to 8090
before this was ever run for real.

### Qwen2.5-1.5B on llama.cpp (`1.5b-q4-llamacpp-orin`), 2026-09-04

The guessed tag `dustynv/llama_cpp:r36.4.0` exists but ships an old llama.cpp build
(version 4579) whose server hard-errors on any request carrying `tool_choice`
(`HTTP 500: Unsupported param: tool_choice`) — useless for this lab. Switched to
`dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04` (version 5058) — confirmed to run correctly
on this device's actual driver despite its cu128/24.04 label (built against a newer
CUDA/Ubuntu base than this device's real CUDA 12.6/Ubuntu 22.04; `nvidia-container-runtime`'s
driver mount is backward-compatible here). Fixed `_DEFAULT_LLAMACPP_IMAGE`/
`docker-compose.llamacpp.yml` to this tag.

Cold start with this tag: plain completion worked. Cold start reported as 2.0s on a
*second* run only because the GGUF was already cached from the first attempt — not a real
cold-start number.

**Tool-calling reliability (RESOLVED).** The original single test failed once, but
repeating the *identical* call 15 times at default sampling temperature succeeded 12/15
(80%) — real stochastic variance, not a hard failure. Isolated cause: at the server's
default temperature (~0.8), the model sometimes narrates in prose instead of calling the
tool; at `temperature=0.1`, 15/15 (100%) succeeded. The same 30-call comparison against
vLLM: 15/15 at both default temperature AND 0.1 — vLLM's `--tool-call-parser hermes` is
temperature-robust (likely grammar-constrained), llama.cpp's Hermes-2-Pro detection is
not. Ruled out `max_tokens` as the cause first (raising 64→256 did NOT improve the rate).
Fix: `llm_client.call_llm()` gained an optional `temperature` parameter (`None` = omit,
server default), and `scripts/validate_tool_calling.py` now pins `--temperature 0.1` by
default (costs vLLM nothing, fixes llama.cpp).

**Second real bug**, found re-running the real script against actual BFCL data:
llama.cpp's server hard-errors (`HTTP 500: JSON schema conversion failed: Unrecognized
schema`) on several of BFCL's own non-standard JSON-schema type names — `"float"` (215
occurrences across the staged simple+irrelevance files), `"tuple"` (2), and `"any"` (1) —
vLLM tolerates these silently, which is why this went undetected until llama.cpp ran
against real BFCL data. Fixed: `_bfcl_function_to_openai_tool()` renames `float`→`number`
and `tuple`→`array`, drops `any` entirely (no JSON-schema equivalent), alongside the
existing `dict`→`object` rename.

**First real BFCL scorecard for any row**, after both fixes, `--limit 20` (40 cases):
simple 85% (17/20), irrelevance 65% (13/20), overall 75% (30/40). Small sample, not the
full ~400+239 corpus.

No GPU/port collision with production (`vllm-orchestrator`, `embedded-ai-chain`'s
dashboard) confirmed before and after.

Also done same day: `uv venv && uv pip install -r requirements.txt && make test` on real
hardware — 47/47 pass, clean install. BFCL + MMLU staged on-device: 399 `simple` cases +
399 ground-truth rows, 239 `irrelevance` cases, 14042 MMLU questions, all verified against
this repo's own loaders. Repo renamed `jetson-llm-qwen` → `jetson-llm`.

### Apertus-8B smoke test, 2026-09-04 — BLOCKED

llama.cpp does not recognize Apertus's GGUF architecture at all: `error loading model
architecture: unknown model architecture: 'apertus'`. Checked on two builds — this lab's
confirmed tag (build 5058) and the newest jetson-containers tag available at the time
(build 5283) — same failure on both, so this is genuinely missing upstream support, not a
stale-image problem fixable by bumping a tag. Not attempted: building llama.cpp from
source with Apertus support (unknown if/when that's landed upstream) — out of scope for a
smoke test. No vllm/orin row exists either (no trustworthy AWQ found) — **Apertus has no
working serving path in this lab at all**, a decisive negative result.

### Bielik-11B on llama.cpp, 2026-09-04 — WORKING

Cold start 338s (includes the ~6.7GB first-download, not just server startup), plain
completion succeeded, AND a tool-calling completion correctly returned a structured
`get_weather({"city":"Warsaw"})` call — where the *identical test* against Qwen2.5-1.5B on
llama.cpp (above) had the model describe wanting to call the tool in free text instead.
llama-server logged "Chat format: Generic" for Bielik (a JSON-schema-grammar tool-call
path) vs "Hermes 2 Pro" for Qwen — evidence the llama.cpp tool-calling gap may be
model/size-specific rather than backend-wide, though one data point, not a trend.

### Bielik-11B on vLLM, 2026-09-04 — serves, tool-calling broken

**Serving: works.** `gpu_memory_utilization=0.3` (an untested guess) needed no re-tuning;
cold start 234-274s across two runs. Plain completion and a Polish-language sanity check
("Jaka jest stolica Polski?" → correct, coherent Polish) both fine. vLLM auto-detected the
compressed-tensors int4 checkpoint with no `--quantization` flag.

**Tool calling: CONFIRMED NOT WORKING**, on two parsers — `hermes` (this lab's default)
and `llama3_json` (tried since Bielik-v3 is LlamaForCausalLM-architected). Both returned
an empty `tool_calls` list on the identical `get_weather`/Warsaw test that vLLM+Qwen2.5
and llama.cpp+Bielik each handle correctly. Two independent parsers, identical result —
not a fluke.

Root cause: Bielik's own GGUF README documents tool use only as a manual prompt-injection
convention (an Ollama Modfile asking the model to emit `{"name":...,"arguments":{...}}`
as plain text), not evidence of native training on any tagged format vLLM's per-family
regex parsers look for. Consistent with why llama.cpp's grammar-CONSTRAINED "Generic"
tool-call path works for this model (forces valid JSON structurally, regardless of what
the model was trained to emit) where vLLM's tag-DETECTION parsers don't. Not exhaustively
tried: vLLM has ~30 `--tool-call-parser` options; the other ~28 remain untried after two
cheap, plausible guesses failed — diminishing returns without a specific reason to expect
a particular one matches Bielik's training.

**The generalisable finding**: tool-calling reliability is a property of
(model × backend × parser × temperature), not of the model. Three of this lab's four
(family, backend) cells behave differently on that axis, in both directions — Qwen2.5
works on both backends, Bielik works only on llama.cpp. `schemas/quality_result.schema.json`
records all four fields for exactly this reason.

---

## Scope change, 2026-09-04 — `docs/note.md` merged into the plan

`docs/note.md` (added alongside `docs/project.diagram.md` and the `schemas/` sketches)
re-aimed this repo from "pick one orchestrator LLM" to "a reproducible benchmark suite
that also picks one." Phases 2+ were rewritten around it.

**What note.md asked for that this repo already had** (it was written without reading
`benchmarks/harness.py`): p50/p95/p99 with no means, cold start timed separately, "don't
report client RSS as model memory" (already flagged as `rss_mb_caveat`), system RAM vs
process RSS tracked separately, thermal steady-state detection, power sampled only in the
measurement window, power mode/clocks/L4T on every result, BFCL/MMLU already the two
quality axes, an append-only registry.

**Where note.md and reality disagreed, and how it was resolved:**

- **Fixed `warmup: 5, measurements: 30` would regress** the harness's existing
  temperature-driven steady-state policy. Kept as floors, not the policy.
- **Thor-as-primary-platform was unschedulable** — no Thor access existed. Orin is the
  platform of record; Thor is the cross-platform arm (also the stronger framing for P5:
  "does the ranking change with hardware" needs both boards).
- **The `"memory_gb": 128` example** doesn't match this Orin's ~30GB — every
  memory-budget figure is per-platform, never carried across.
- **The Qwen3-vLLM blocker was measured false.** Originally recorded as "vLLM ≥0.11 has
  no JetPack 6.2 wheels, may not serve Qwen3" — but the pinned container reports
  `vllm 0.19.0` when queried (`GET /version`), and serves fine. The wheel constraint is
  about *pip*; NVIDIA's container is a different distribution path. Second time in this
  repo a version was assumed from a name and was wrong — both times fixed by asking the
  running server.
- **note.md never mentions Bielik or Apertus** — this lab's actual 2nd/3rd families and
  its only decisive negative result. Kept in the matrix.
- **The full Cartesian matrix (model × size × precision × backend × platform × power ×
  context × output × batch) is unschedulable** at thousands of runs. Phases 4-9 are the
  real expansion order.
- **note.md implies a third paper.** Resolved: paper count is decided after the results
  exist (direct user decision), not pre-committed. This repo is the instrument for
  `paper.md`'s P3-P6 regardless.

The `schemas/` files added alongside note.md were example *instances*, not JSON Schema
(no `$schema`/`type`/`properties`). Phase 3 turned them into real ones.

---

## Phase 2 — Measurement-integrity retrofit: incident record

Verified by two real runs on `1.5b-q4-llamacpp-orin`, co-resident with the live
`vllm-orchestrator` (`results/raw/..._phase2-verify_...r01`/`r02`).

### Incident 1 — prefix-cache contamination (6.6× TTFT error)

Every TTFT/prefill number this lab had produced before this was measuring a KV-cache hit,
not a prefill. The benchmark scripts sent the *identical* prompt every repetition, and
both backends cache by prompt prefix. Caught reading llama-server's own log: for a
40-token prompt it reported `prompt eval time = ... / 1 tokens` on every run after the
first.

| | TTFT p50 | prefill |
|---|---|---|
| identical prompt (old behaviour) | 40.8 ms | ~1 token evaluated |
| unique per run (the fix) | 268.1 ms | 18 tokens evaluated |

Fixed: `--prompt-uniqueness` defaults to `unique-per-run`, marker at the *front* of the
prompt (a suffix would leave the prefix reusable). The chat template's own system prefix
stays cacheable, which is realistic — a real deployment also has a stable system prompt.

### Incident 2 — inert HF cache mount (weights re-downloaded every run)

`weights_cached` reported False twice in a row — not a detection bug: vLLM really was
re-downloading its weights every run. `VllmCoordinator` mounted
`/opt/hf-cache:/root/.cache/huggingface` and set `HF_HOME`, but the image **bakes in
`HUGGINGFACE_HUB_CACHE=/data/models/huggingface`**, which takes precedence — the mount
was inert, weights landed inside a `--rm` container and vanished on every stop. Found via
`docker inspect` on the production container's own environment.

Fixed by setting `HUGGINGFACE_HUB_CACHE`/`HF_HUB_CACHE` explicitly in both
`VllmCoordinator.start()` and `docker-compose.yml`.

| | cold start | container start | model load |
|---|---|---|---|
| cache mount inert (r01, r02) | 156.2s / 154.2s | 2.0s | 154.2s / 152.2s |
| after fix, weights cached (r03) | 136.2s | 2.0s | 134.2s |

Download was ~18s of a ~155s cold start for this 1.5GB model — real, and much smaller
than the raw number suggested; the other ~134s is vLLM's engine init. Download cost
scales with model size — Bielik-11B's AWQ is ~6.2GB.

### Incident 3 — energy figure backed by 9 power samples

The vLLM cell's energy figure rested on 9 tegrastats samples over 4.4s, because at
108 tok/s it finished 8 repetitions before the 500ms sampler had characterised anything.
`benchmarks/harness.py` has a `min_measurement_s` floor for exactly this reason and the
rewrite had dropped it. Restored as `--min-measurement-s` (default 10s), plus a
`power_window_thin_<n>_samples` validity flag.

### Both backends verified end-to-end

| | cold start | weights cached | backend version (queried) | TTFT p50 | decode p50 | J/output-token |
|---|---|---|---|---|---|---|
| `1.5b-q4-llamacpp-orin` | 2.0-4.0s | yes | `b5058-6bf28f01` | 268.1 ms | 21.1 tok/s | 0.996 |
| `1.5b-awq-vllm-orin` | 136.2s | yes | `vllm 0.19.0` | 30.2 ms | 106.6 tok/s | 0.308 |

**Not a backend comparison** — n=8, `warmup_reached_steady_state=false` on both, running
alongside production, different quantizations. Establishes only that the instrument works
on both backends. Also settled: vLLM's own prefix-cache hit rate was 32.6% even with
unique prompts (the chat template's system prefix is legitimately shared — realistic,
kept).

---

## Phase 3 — Experiment configuration: incident record

### Config bug: context sweep asked for more context than candidates have

`context_sweep.yaml`'s first draft used 128/512/1024/**2048** with `output_tokens: 128` —
but the context window holds input *and* output *and* the chat template, and every row
pins 2048. `tests/test_experiment_configs.py` checks `max(input) + max(output) <= ctx`
and failed before a server started. Ladder fixed to 128/512/1024/**1536**.

### Incident A — false `standalone` (container-level)

First real Phase 3 run declared `standalone` while `vllm-orchestrator` was up.
`assert_condition_matches_reality()` was added to refuse this, checking Docker
containers only — its own docstring admitted a bare-metal process would slip through.

### Incident B — false `standalone` (bare-metal, the container check missed)

That gap got exercised the same day. After the user approved stopping
`vllm-orchestrator` for a genuine standalone campaign, and mid-run the user asked
directly whether prior runs were contaminated: checking found `embedded-ai-chain`'s
**entire production pipeline** (YOLO + orchestrator + STT/TTS) was running as **one
bare-metal Python process** (`tts_consumer.py`, PID 93363, up since before this repo's
Phase 2 work even started), invisible to `docker ps`. Confirmed via `/proc/93363/fd` —
the process held open `nvhost-*.gpu-fd*`/`nvgpu-*-tsg*` file descriptors, a real GPU
device handle.

The campaign was killed (`pkill`; the orphaned `vllm-llm-lab` container needed manual
removal since `SIGTERM` doesn't run Python `finally` blocks, so the script's own
production-restore trap didn't fire; `vllm-orchestrator` was recreated and re-verified
healthy). The 6 results already written (all `standalone`) were **deleted, not
relabelled** — the co-resident workload was never declared or controlled, so nothing
honest could be written after the fact. Never committed, so no history needed rewriting.

Fixed: `assert_condition_matches_reality()` gained `gpu_holding_pids()` — scans every
process's `/proc/<pid>/fd` for `nvhost`/`nvgpu`/`nvmap` symlinks. The lab's own container
is excluded via `docker top <name> -eo pid` (host PIDs for containerized processes).
Verified: excluding `vllm-orchestrator`'s 3 host PIDs did not also exclude PID 93363.
Still not exhaustive (a GPU-idle process about to wake could still pass).

### Incident C — incomplete `co-resident` declaration (on already-committed data)

The user asked about one specific committed result by name
(`..._phase2-verify_out128_bs1_r01`). It correctly said `co-resident: [vllm-orchestrator]`
— but PID 93363 ran throughout its entire window too and was never declared. All 5
committed Phase 2 verification results had this gap.

**Corrected in place, not deleted**: unlike Incident B's data, this workload was known
(not uncontrolled), so each file's `co_resident_workload` was fixed with a `_comment`
explaining what changed and why; pre-correction content recoverable from git history
(`6f898f5`).

Fixed: `assert_condition_matches_reality()` gained a `declared_co_resident` parameter,
checked for **completeness** against the same two detectors — not just consulted for
the `standalone` case.

**Consequence**: a genuine `standalone` claim on this device now requires stopping
`tts_consumer.py`, not just the `vllm-orchestrator` container — a materially bigger
interruption (it owns the live dashboard and the real perception/dialogue loop) than the
container-only story assumed. Whether/when to stop it is the user's call each time, and
is exactly the standalone/co-resident tension Phase 9 exists to characterise on purpose.

---

## Phase 6 — Thor readiness: pre-emptive fix, 2026-09-07

Found auditing Thor-readiness in response to a direct user question, before any real
Thor row exists. `hardware_manifest()`/`software_manifest()` read `/proc`, `/sys`, and
the local Docker daemon unconditionally — all describe the CLIENT (this Orin), never the
actual inference host when `--target remote` points at a board like Thor. A real Thor run
would have silently reported the Orin's own `nvpmodel_mode`/`jetson_clocks_locked`/
`l4t_version`/`board`/Docker version *under a manifest whose `platform` field says
"thor"* — the same class of mislabeling as Phase 3's incidents, just a different field,
caught this time before a real campaign hit it.

Fixed: both functions take a `target` parameter; for `"remote"` they return honest
`"unknown (remote target...)"` / `null` values, mirroring the retired
`scripts/benchmark.py`'s old `tegrastats_caveat`. `backend_version` is unaffected — it
queries `base_url` itself (the actual remote server), genuinely remote-safe already.
5 new tests.

**Still open**: `assert_condition_matches_reality()` is skipped entirely for
`--target remote` in every caller, because it can only observe this client's own
containers/GPU-holding processes, not Thor's. The co-residency safety net Phase 3 built
does not extend to Thor at all yet — a `standalone` claim against Thor is currently
trusted, not verified. Needs either SSH-based remote checks or a manual on-Thor
companion check — a real design decision, deferred to when Phase 6 actually starts.
