# TODO — orchestrator-LLM selection lab

Action list for turning this repo into a working multi-candidate orchestrator-LLM
selection lab, per `docs/promotion-contract.md`, **and** — since 2026-09-04, per
`docs/note.md` — into a reproducible Jetson LLM benchmark suite of which that
selection scorecard is one subset. See "Scope change" below for how the two fit
together and what was reconciled. Phase-gated like `embedded-ai-chain`'s own
`docs/TODO.md` and `jetson-vlm-lab`'s own `docs/TODO.md` - don't start a phase until
the one before it is checked off.

## Phase 0 — Framework survey (research, done before any code was written)

- [x] Confirmed Qwen2.5 GGUF availability: `Qwen/Qwen2.5-{1.5B,3B,7B}-Instruct-GGUF`
      all exist as official, Apache-2.0, Qwen-published repos (HF Hub search,
      2026-09-04). `1.5B`'s repo lists `q4_k_m`/`q5_k_m`/`q6_k`/`q8_0`/`fp16` files
      directly - no proxy/third-party quant needed, same trustworthy-provenance bar
      `jetson-vlm-lab/docs/promotion-contract.md` §2a required.
- [x] **TensorRT excluded from v1 - confirmed, not guessed (2026-09-04):**
      - Standard TensorRT-LLM's Jetson support is a dedicated `v0.12.0-jetson` branch
        pinned to **JetPack 6.1**; Jetson isn't supported on the current main branch.
        This project's Orin is JetPack 6.2.x - unsupported/stale combination.
      - NVIDIA's current answer for Jetson Thor is a different product, **TensorRT
        Edge-LLM**, which requires **JetPack 7.x** on both Orin and Thor (this
        project's Orin is 6.2.x - can't run it at all) and has **no OpenAI-compatible
        HTTP server** (a C++ runtime meant for direct API integration, not a drop-in
        coordinator target). Qwen2.5 isn't confirmed in its supported-model table
        either (only Qwen3/3.5/3.6 listed).
      - Decision: build only vLLM and llama.cpp legs now. Revisit TensorRT-LLM/
        Edge-LLM only if/when Thor moves to JetPack 7.x, and treat it then as its own
        bespoke C++-integration project, not "add a coordinator class."
- [x] Confirmed llama.cpp's Jetson serving path: `dusty-nv/jetson-containers` is the
      standard prebuilt-image source (tagged per JetPack/L4T version); the only
      alternative found (`zhamm/llama-cpp-jetson`) targets Orin Nano 8GB specifically,
      not this AGX Orin, so it's not the right base. `llama-server` speaks the same
      OpenAI-compatible `/v1/chat/completions` + streaming shape vLLM does, so the
      existing coordinator contract needs no new integration pattern, just a new
      coordinator that launches a different image/command.

**GATE 0** — met, 2026-09-04. Two frameworks in scope (vllm, llama-cpp), TensorRT
explicitly deferred with reasons recorded above (not silently dropped).

## Phase 1 — Foundations (in progress)

- [x] Promotion contract written (`docs/promotion-contract.md`), adapted from
      `jetson-vlm-lab`'s
- [x] `configs/models.yaml` seeded: 3 vllm/orin rows (thresholds copied from
      `embedded-ai-chain/src/orchestrator_models.py`'s own already-tuned values, not
      re-derived) + 3 llama-cpp/orin rows (new backend, all untested starting points)
- [x] `src/llm_coordinator.py`: `VllmCoordinator` (renamed copy of jetson-vlm-lab's
      `VlmCoordinator` - nothing in it was ever VLM-specific), new
      `LlamaCppCoordinator`, `RemoteCoordinator` (renamed copy of
      `RemoteVlmCoordinator` - already backend-agnostic)
- [x] `src/llm_client.py`: text-only `call_llm()` returning the full message
      (content + tool_calls), not just text - needed by
      `scripts/validate_tool_calling.py`
- [x] `benchmarks/harness.py`: copied verbatim from `jetson-vlm-lab` (already fully
      generic, no image references)
- [x] `scripts/benchmark.py` / `scripts/benchmark_streaming.py`: text-only ports of
      jetson-vlm-lab's, no image loading
- [x] `scripts/validate_tool_calling.py`: **new accuracy axis**, replacing GQA/
      TextVQA (not applicable to a text orchestrator) - scores against real BFCL
      (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`) `simple`/`irrelevance`
      categories (a simplified AST-style checker, not the official `bfcl-eval`
      package - see the script's docstring), `tool_choice="auto"` only - deliberately
      not using orchestrator.py's forced-tool_choice workaround, since the point is
      measuring unforced judgment. Superseded an earlier version that hand-copied 15
      cases from `orchestrator.py`'s own patterns - real external dataset instead,
      per 2026-09-04 user request
- [x] `scripts/validate_mmlu.py`: **new axis**, a quantization-regression sanity
      check (not a ranking signal) using `cais/mmlu`'s `all/test` split - generative
      letter-parsing scoring (works identically across vLLM/llama.cpp, unlike
      logprob-based scoring which would need a backend-specific API)
- [x] Real Docker/GPU smoke test, 2026-09-04, `1.5b-awq-vllm-orin` on this Orin
      (`VllmCoordinator` directly, not through `make serve-1.5b-vllm`'s foreground
      compose target): cold start 158-162s, `/health` 200, a plain completion and a
      tool-calling completion (`get_weather({"city": "Warsaw"})`, correctly
      structured) both succeeded. Ran alongside the real production
      `vllm-orchestrator` container (port 8001, untouched) with no collision -
      confirmed `free -h`/`docker ps` before starting, per this file's own
      standing caution. Two real bugs this test caught and fixed before it passed:
      1. `configs/models.yaml`'s `1.5b-awq-vllm-orin` row claimed `max_model_len:
         4096` as "known-good" - `docker inspect vllm-orchestrator` showed
         production actually runs `2048`. Fixed to match the real value; the
         4096 guess was never sourced from `orchestrator_models.py` (that file has
         no `max_model_len` field at all) and shouldn't have been asserted as
         "known-good" unverified.
      2. `VllmCoordinator.start()`'s own command never included
         `--enable-auto-tool-choice --tool-call-parser hermes` - only
         `docker-compose.yml` had them. Every real tool-calling request failed
         with HTTP 400 until this was fixed (this class's own smoke test caught
         the mismatch between its two ways of starting the container).
      Also found (unrelated to vLLM): `LlamaCppCoordinator`'s default port 8080
      collides with `embedded-ai-chain`'s own live dashboard on this device -
      changed the default to 8090 before this was ever run for real.
- [x] Real Docker/GPU smoke test, 2026-09-04, `1.5b-q4-llamacpp-orin` on this Orin:
      - The guessed tag `dustynv/llama_cpp:r36.4.0` does exist but ships an old
        llama.cpp build (version 4579) whose server hard-errors on any request
        carrying `tool_choice` (`HTTP 500: Unsupported param: tool_choice`) -
        useless for this lab. Switched to `dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04`
        (version 5058) - confirmed to run correctly on this device's actual driver
        despite its cu128/24.04 label (built against a newer CUDA/Ubuntu base than
        this device's real CUDA 12.6/Ubuntu 22.04; `nvidia-container-runtime`'s
        driver mount is backward-compatible here, GPU detected fine). Fixed
        `_DEFAULT_LLAMACPP_IMAGE`/`docker-compose.llamacpp.yml` to this tag.
      - Cold start with this tag: plain completion worked (`-hf` downloader + cache
        mount both function as expected). Cold start reported as 2.0s on a *second*
        run only because the GGUF was already cached at `/opt/llama-cache` from the
        first attempt - not a real cold-start number, don't reuse it as one.
      - **RESOLVED, 2026-09-04** (was: "open finding, not yet resolved" - see below
        for the full diagnostic, kept rather than deleted per this file's own
        append-only convention). The original single test (`"What's the weather
        like in Warsaw?"` against a `get_weather` tool) failed once, but repeating
        the *identical* call 15 times at default sampling temperature succeeded
        12/15 (80%) - not a hard failure, real stochastic variance. Isolated the
        cause: at the server's default sampling temperature (~0.8), the model
        sometimes narrates in prose instead of calling the tool; **at
        `temperature=0.1`, 15/15 (100%) succeeded**. Ran the same 30-call
        comparison against `1.5b-awq-vllm-orin` (vLLM): **15/15 at both default
        temperature AND 0.1** - vLLM's `--tool-call-parser hermes` is
        temperature-robust (likely grammar-constrained), llama.cpp's Hermes-2-Pro
        detection is not, so llama.cpp genuinely needs a pinned low temperature
        where vLLM doesn't. Ruled out `max_tokens` as the cause first (raising it
        64->256 did NOT improve the rate - the failures weren't truncated
        responses, the model completed a full non-tool-call answer some fraction
        of the time regardless of budget). Fix applied: `llm_client.call_llm()`
        gained an optional `temperature` parameter (`None` = omit, server
        default - unaffected for `scripts/benchmark.py`/`benchmark_streaming.py`,
        which don't pass one), and `scripts/validate_tool_calling.py` now pins
        `--temperature 0.1` by default (costs vLLM nothing, fixes llama.cpp).
      - **Second real bug found while re-running the real script (not just the
        ad-hoc diagnostic) against actual BFCL data**: llama.cpp's server
        hard-errors (`HTTP 500: JSON schema conversion failed: Unrecognized
        schema`) on several of BFCL's own non-standard JSON-schema type names -
        `"float"` (215 occurrences across the staged simple+irrelevance files,
        common enough to silently wreck most of a real run), `"tuple"` (2), and
        `"any"` (1) - vLLM tolerates these silently, which is exactly why this
        went undetected until llama.cpp was actually run against real BFCL data
        rather than the ad-hoc `get_weather` schema (which only ever used
        `"string"`, a valid type, so it never triggered this). Counted every type
        actually present in the staged corpus rather than guessing which might
        appear. Fixed: `_bfcl_function_to_openai_tool()` now renames
        `float`->`number` and `tuple`->`array`, and drops `any` entirely (no
        JSON-schema equivalent - the idiomatic way to express "unconstrained" is
        omitting `type`, not inventing a fake name), alongside the existing
        `dict`->`object` rename.
      - **First real, working BFCL scorecard for any row in this lab**, after both
        fixes above, `1.5b-q4-llamacpp-orin`, `--limit 20` (40 total cases):
        simple 85% (17/20), irrelevance 65% (13/20), overall 75% (30/40). Small
        sample (`--limit 20`, not the full ~400+239 corpus) - a real number, not
        yet the final scorecard Phase 3 will run.
      - No GPU/port collision with the real production `vllm-orchestrator` container
        (confirmed `docker ps`/`free -h` before and after) or with
        `embedded-ai-chain`'s own dashboard (moved this lab's llama.cpp default port
        to 8090 before running anything, once the 8080 collision was found).
- [x] `uv venv && uv pip install -r requirements.txt && make test` on real hardware,
      2026-09-04: 47/47 pass (clean install from `requirements.txt`, not just the
      dev environment's pre-existing venv)
- [x] Staged BFCL + MMLU on-device, 2026-09-04, per README's "Dataset staging"
      section: 399 BFCL `simple` cases + 399 matching ground-truth rows, 239
      `irrelevance` cases, 14042 MMLU `all/test` questions - all loaded and verified
      against this repo's own loaders before being trusted
- [x] Repo renamed `jetson-llm-qwen` -> `jetson-llm` and Apertus-8B-Instruct-2509
      added as a second model family (`apertus-8b-q4-llamacpp-orin` in
      `configs/models.yaml`, llama.cpp/GGUF only - no trustworthy AWQ found, see that
      row's own notes), 2026-09-04, per direct user request following the same
      "second model family arrived, so did jetson-vlm-lab's own rename" reasoning
      that repo's `docs/TODO.md` already used. Not yet run - it needs its own smoke
      test before any of its numbers are trusted, same bar every Qwen row was held to
- [x] Bielik-11B-v3.0-Instruct added as a third model family
      (`bielik-11b-q4-llamacpp-orin`), 2026-09-04, per direct user request (Polish
      model, SpeakLeash/ACK Cyfronet AGH) - llama.cpp/GGUF via SpeakLeash's own
      official GGUF repo (not a third party's, unlike Apertus's bartowski quant).
- [x] Real smoke tests, 2026-09-04, for both new families:
      - **Apertus (`apertus-8b-q4-llamacpp-orin`) - BLOCKED.** llama.cpp does not
        recognize Apertus's GGUF architecture at all: `error loading model
        architecture: unknown model architecture: 'apertus'`. Checked on two
        builds - this lab's confirmed tag (`0.3.9-r36.4.0-cu128-24.04`, llama.cpp
        build 5058) and the newest jetson-containers tag available at the time
        (`b5283-r36.4-cu128-24.04`, build 5283) - same failure on both, so this is
        genuinely missing upstream support, not a stale-image problem this lab can
        fix by bumping a tag again. Not attempted: building llama.cpp from source
        with Apertus support (unknown whether/when that's landed upstream) - out
        of scope for a smoke test. No vllm/orin row exists either (no trustworthy
        AWQ, see the row's own notes), so **Apertus currently has no working
        serving path in this lab at all** - a real, decisive negative result, not
        a gap to paper over.
      - **Bielik (`bielik-11b-q4-llamacpp-orin`) - WORKING, and a genuinely useful
        result.** Cold start 338s (includes the ~6.7GB first-download, not just
        server startup), plain completion succeeded, AND a tool-calling completion
        correctly returned a structured `get_weather({"city":"Warsaw"})` call -
        where the *identical test* against Qwen2.5-1.5B on llama.cpp (this same
        Phase 1, above) had the model describe wanting to call the tool in free
        text instead of emitting one. llama-server logged "Chat format: Generic"
        for Bielik (a JSON-schema-grammar tool-call path) vs "Hermes 2 Pro" for
        Qwen - real evidence the earlier llama.cpp tool-calling gap may be
        model/size-specific rather than a backend-wide limitation, though this is
        one data point, not a trend - the real BFCL run is still what settles it.
- [x] Real Docker/GPU smoke test, 2026-09-04, `bielik-11b-awq-vllm-orin` - the
      vLLM half of the Bielik question, added after the llama.cpp row above worked.
      Recorded here as well as in README/`configs/models.yaml` because this file is
      the experimental record.
      - **Serving: works.** `gpu_memory_utilization=0.3` (an untested guess when the
        row was added) needed no re-tuning; cold start 234-274s across two runs.
        Plain completion and a Polish-language sanity check ("Jaka jest stolica
        Polski?" -> correct, coherent Polish) both fine. vLLM auto-detected the
        compressed-tensors int4 checkpoint with no `--quantization` flag, the same
        way it does for the Qwen AWQ rows.
      - **Tool calling: CONFIRMED NOT WORKING**, on two parsers - `hermes` (this
        lab's default) and `llama3_json` (tried because Bielik-v3 is
        LlamaForCausalLM-architected). Both returned an empty `tool_calls` list on
        the identical `get_weather`/Warsaw test that vLLM+Qwen2.5 and
        llama.cpp+Bielik each handle correctly. Two independent parsers, identical
        result - not a fluke.
      - Root cause, not just symptom: Bielik's own GGUF README documents tool use
        only as a manual prompt-injection convention (an Ollama Modfile asking the
        model to emit `{"name":...,"arguments":{...}}` as plain text), not training
        on any tagged format vLLM's per-family regex parsers look for. That is
        consistent with why llama.cpp's grammar-CONSTRAINED "Generic" path works for
        this model (it forces valid JSON structurally, regardless of what the model
        was trained to emit) where vLLM's tag-DETECTION parsers don't (nothing to
        detect if the tag is never emitted).
      - Not exhaustively tried: vLLM has ~30 `--tool-call-parser` options; the other
        ~28 remain untried after two cheap, plausible guesses failed -
        diminishing returns, not attempted further without a specific reason to
        expect a particular one matches Bielik's training.
      - **The generalisable finding, and the reason this matters beyond one row:**
        tool-calling reliability is a property of (model x backend x parser x
        temperature), not of the model. Three of this lab's four
        (family, backend) cells now behave differently on that axis, and in both
        directions - Qwen2.5 works on both backends, Bielik works only on llama.cpp.
        `schemas/quality_result.schema.json` records all four fields for exactly
        this reason.

**GATE 1** — met for Qwen2.5-1.5B on both backends, and for Bielik-11B on llama.cpp.
Bielik-11B on vLLM is a *partial* pass: it serves and answers correctly but cannot
tool-call, so it is benchmarkable for performance and excluded from the tool-calling
scorecard with a reason, not a blank.
Apertus is a closed question (blocked, documented) rather than an open one. Still
pending before the full matrix: the remaining Qwen sizes (3B/7B), and a real
BFCL/MMLU scorecard run for every row that's actually able to serve at all.

*(Written when that matrix was Phase 3; it is Phase 5 after the 2026-09-04
renumbering below. Left as written rather than edited, per this file's append-only
convention.)*

---

# Scope change, 2026-09-04 — `docs/note.md` merged into this plan

`docs/note.md` (added 2026-09-04 alongside `docs/project.diagram.md` - renamed
from `progect.diagram.md`, a typo in a never-committed file - and the
`schemas/` sketches) re-aims this repo from "pick one orchestrator LLM for
`embedded-ai-chain`" to "a reproducible, publication-quality LLM benchmark suite for
Jetson, which *also* picks that orchestrator LLM." Phases 2+ below are rewritten
around that. Phases 0-1 above are untouched — they are the experimental record, and
this file's own append-only convention applies to them.

**What note.md asks for that this repo already has** (do not rebuild these — note.md
§1 says so itself, and it was written without a full read of `benchmarks/harness.py`):

| note.md requirement | Already implemented |
|---|---|
| §9 p50/p95/p99, never means | `harness.percentiles()` — mean is deliberately absent |
| §9 cold start separate from inference | `run_benchmark()`'s `setup_fn` timing → `cold_start_ms` |
| §9 "do not report client-process RSS as model memory" | already the case, *and* already flagged in `scripts/benchmark.py`'s `rss_mb_caveat` |
| §9 system RAM vs process RSS on unified memory | `system_ram_mb` (tegrastats) vs `rss_mb`, separately |
| §38 thermal stability / steady-state window | `TegrastatsSampler.is_thermally_stable()` — real `tj` flatness, plus a `warmup_reached_steady_state=False` flag when it gives up |
| §14 power sampled *during* the measurement window only | `sampler.samples_between(t_meas_start, t_meas_end)` |
| §3 power mode / clock state on every result | `nvpmodel_mode()`, `jetson_clocks_locked_heuristic()`, `l4t_version()` on every row |
| §36 tool-calling eval, external dataset | `scripts/validate_tool_calling.py` against real BFCL |
| §17 a general-capability suite | `scripts/validate_mmlu.py` |
| §28 registry with platform/precision/backend/status-in-notes, never delete rows | `configs/models.yaml` + its own "confirmed vs untested" rule |
| §57 research discipline / traceable record | Phase 1 above *is* that record |

**Where note.md and this repo's reality disagree — resolved as follows:**

- **note.md §4/§10 "warmup: 5, measurements: 30" would be a regression.** The harness
  already warms up until junction temperature is flat across a rolling window, and
  says so when it couldn't. Keep that. Treat note.md's numbers as *floors*
  (`warmup_min`, `runs`), which is how `scripts/benchmark.py` already passes them.
- **note.md §26/§48 make Thor the primary platform; this lab has no Thor access at
  all** (old Phase 2, never started — no confirmed address, no `-thor` row in
  `configs/models.yaml`). A first publication experiment gated on unavailable hardware
  is not schedulable. **Decision: Orin is the platform of record; Thor is the
  cross-platform arm.** That is also the stronger framing — "does the preferred
  allocation change with the hardware?" is `embedded-ai-chain/docs/paper.md`'s P5,
  and it needs *both* platforms, not Thor alone.
- **note.md §7's example schema hardcodes `"memory_gb": 128`.** This Orin has ~30GB
  unified (verified, see `CLAUDE.md`). Every memory-budget figure is per-platform;
  never carry a Thor budget into an Orin analysis or vice versa.
- **note.md §45/§46 plan a Qwen2.5-vs-Qwen3.x campaign; this repo has no Qwen3 row.**
  Recorded here on 2026-09-04 as "may not be able to serve one — vLLM ≥0.11 has no
  prebuilt JetPack 6.2/CUDA 12.6 wheels". **Measured the same day: that premise does
  not apply to this lab's path.** The pinned container
  `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` reports **`vllm 0.19.0`** when
  asked (`GET /version`, during the Phase 2 verification run) and serves correctly on
  this JetPack 6.2 device. The wheel-availability constraint is about *pip* wheels;
  NVIDIA's container is a different distribution path and sidesteps it. So Qwen3 is
  very likely servable here, and its Phase 7 item stays gated on a real test only
  because *every* new row is — not because a known blocker stands in the way. This is
  the second time in this repo a version has been assumed from a name and been wrong;
  both times the fix was to ask the running server.
- **note.md never mentions Bielik or Apertus**, which are this lab's actual second and
  third families and its two most interesting existing results (Apertus blocked
  outright; Bielik the strongest llama.cpp row). They stay in the matrix. A benchmark
  paper that drops its only decisive negative result is a worse paper.
- **note.md's full Cartesian matrix is not schedulable.** model × size × precision ×
  backend × platform × power-mode × context × output × batch, at ~160s vLLM cold start
  and ≥30 runs per cell, is thousands of runs. note.md §4 already says "do not attempt
  the complete Cartesian product"; Phases 4-9 below are the actual expansion order.
- **note.md implies a third paper.** There are already two: `PAPER_PLAN.md`
  (Real-Time/Embedded Systems journal, 2027-01-25) and `docs/paper.md` (Metabolic
  Perception position paper). **Decision (revised 2026-09-04): the paper count is
  settled after the results exist, not now.** This repo is built as the instrument
  that produces `paper.md`'s P3/P4/P5/P6 evidence — see the mapping in Phase 4 — and
  that is true whether or not the benchmark data also supports a paper of its own.
  Deferring costs nothing, because no phase below changes based on the answer.

**The `schemas/` files added alongside note.md are examples, not schemas** — they are
instance documents (`"experiment_id": "thor-qwen3-8b-..."`), not JSON Schema (no
`$schema`, `type`, `properties`, `required`). Phase 3 turns them into real ones and
keeps the examples as fixtures.

---

## Phase 2 — Measurement-integrity retrofit (do this first)

The cheapest, highest-value work in the whole note.md plan: every task here is a small
change to code that already exists, and every one of them is *unrecoverable if
skipped* — a campaign run without these produces numbers that cannot be re-analysed
later. Nothing in Phases 4+ should run until this phase is closed.

**Done 2026-09-04**, verified by two real runs on this Orin (`1.5b-q4-llamacpp-orin`,
co-resident with the live `vllm-orchestrator`, `results/raw/..._phase2-verify_...r01`
and `r02`) — not by unit tests alone:

- [x] **Raw per-run measurements persisted** (note.md §10). Every repetition is
      written to `runs[]`; the aggregates are derived from it and from nothing else.
- [x] **Energy exists.** `scripts/benchmark_streaming.py` now runs the tegrastats
      sampler across its own measurement window, so the script that knows token
      counts is the script that knows power. First real figure: **0.996 J per output
      token** (rails VDD_GPU_SOC + VDD_CPU_CV; Qwen2.5-1.5B Q4_K_M on llama.cpp,
      co-resident). `benchmarks/manifest.py:energy_block()` records which rails were
      summed — VIN_SYS_5V0 is deliberately excluded, since a board-total and a
      GPU+CPU figure must never share an axis.
- [x] **Real token accounting.** `src/llm_client.py:stream_llm()` requests
      `stream_options={"include_usage": true}` and reads the server's own `usage`.
      **llama-server does honour it** (checked, not assumed — this lab has been
      wrong about vLLM/llama.cpp parity twice). When a server sends no usage block
      the delta count is used *and* flagged
      (`token_counts_are_sse_chunk_counts_not_tokens`), never silently mixed.
- [x] **Cold start decomposed** (note.md §37), via `_ColdStartMixin`:
      container_start / model_load / server_ready, plus `weights_cached` checked
      *before* the container runs. First real decomposition: 4.05s total = 2.04s
      container + 2.01s load, `weights_cached=true`. `download_s` is deliberately
      left null even when a download happened — it is not separable from load time
      without parsing each backend's logs, and `weights_cached=false` carries that
      information honestly instead. `RemoteCoordinator` returns
      `measured: false` with every component null, and the schema refuses a non-null
      total when `measured` is false.
- [x] **Environment manifest** (note.md §31). git commit + dirty flag, container
      image **and digest**, hardware read from the device, and the backend version
      **read from the running server**. That last one immediately paid for itself:
      the llama.cpp server reports `b5058-6bf28f01` — confirming by query the build
      number Phase 1 had to identify by behaviour.
- [x] **Experiment IDs and `results/raw/<experiment_id>/`** (note.md §29/§30).
      `write_result()` **refuses to overwrite**: re-running a cell means a new
      replicate, not a silently replaced file.
- [x] **`--execution-condition` is required with no default** (note.md §15), and a
      `co-resident` run without `--co-resident` naming the components is rejected by
      both the CLI and the schema.
- [x] Every emitted document is validated against
      `schemas/benchmark_result.schema.json` at write time, so a malformed row fails
      in seconds rather than at analysis time.

### Real finding from the verification runs: prefix-cache contamination

**Every TTFT and prefill number this lab could have produced before today was
measuring a KV-cache hit, not a prefill.** The benchmark scripts sent the *identical*
prompt on every repetition, and both backends cache by prompt prefix (vLLM's
`--enable-prefix-caching`, on by default in `VllmCoordinator`; llama-server's slot
cache). Caught by reading llama-server's own log during the first verification run:
for a 40-token prompt it reported `prompt eval time = ... / 1 tokens` on every run
after the first — it was not prefilling at all.

Measured impact, same config, same 8 runs, only the prompt varying:

| | TTFT p50 | prefill |
|---|---|---|
| identical prompt (`r01`, the old behaviour) | **40.8 ms** | ~1 token evaluated |
| unique per run (`r02`, the fix) | **268.1 ms** | 18 tokens evaluated |

A **6.6× error**, in the flattering direction. Fixed: `--prompt-uniqueness` defaults
to `unique-per-run`, prefixing a per-run marker at the *front* of the prompt (a
unique suffix would leave the prefix reusable and change nothing), and
`workload.prompt_uniqueness` is now a recorded schema field so a warm-cache cell can
never be compared against a cold-prefill one by accident. Note the chat template's
own system prefix stays cacheable even so — which is realistic, since a real
deployment also has a stable system prompt.

This is the clearest possible argument for Phase 2 preceding every campaign: the bug
was invisible in aggregate numbers and only surfaced by running the real thing and
reading the server's own log.

### Both backends verified end-to-end, 2026-09-04

Real runs on this Orin, both co-resident with the live `vllm-orchestrator`, both
schema-validated at write time:

| | cold start | weights cached | backend version (queried) | TTFT p50 | decode p50 | J/output-token |
|---|---|---|---|---|---|---|
| `1.5b-q4-llamacpp-orin` | 2.0-4.0s | yes | `b5058-6bf28f01` | 268.1 ms | 21.1 tok/s | 0.996 |
| `1.5b-awq-vllm-orin` | 136.2s | yes | `vllm 0.19.0` | 30.2 ms | 106.6 tok/s | 0.308 |

**These are tooling-verification runs, not benchmark results, and must not be quoted
as a backend comparison.** n=8, `warmup_reached_steady_state=false` on both, running
alongside a production container, and the two rows are different quantizations
(AWQ-4bit vs GGUF Q4_K_M) — note.md §41's fairness rules are violated on at least
three axes at once. What they legitimately establish is that the instrument works on
both backends and produces the fields the schema requires. The real comparison is
Phase 8's job.

Two things the runs settled that were previously assumed:

- **`vllm 0.19.0`** — read from the running server, not the tag. See the scope-change
  section: this retires the "vLLM ≥0.11 has no JetPack 6.2 wheels" concern for this
  lab's container path.
- **vLLM's own prefix-cache hit rate was 32.6%** even with `unique-per-run` prompts,
  because the chat template's system prefix is legitimately shared across requests.
  That is realistic (a real deployment has a stable system prompt) and is why the fix
  puts the uniqueness marker at the front of the *user* message rather than trying to
  defeat caching entirely — which would measure something no deployment ever sees.

**A third finding, from `weights_cached` reporting False twice in a row.** That
looked at first like a detection bug and was not: the vLLM container was genuinely
re-downloading its weights on every single run. `VllmCoordinator` mounted
`/opt/hf-cache:/root/.cache/huggingface` and set `HF_HOME` to match, but this image
**bakes in `HUGGINGFACE_HUB_CACHE=/data/models/huggingface`**, which takes precedence
— so the mount was inert, the weights landed inside a `--rm` container, and they went
away on every stop. Found by `docker inspect`-ing the production container's own
environment after the host cache stayed empty across two full runs.

Fixed by setting `HUGGINGFACE_HUB_CACHE`/`HF_HUB_CACHE` explicitly in both
`VllmCoordinator.start()` and `docker-compose.yml` — the same
coordinator-vs-compose drift Phase 1 already caught once in the other direction, so
both were changed together. Measured effect, same config, three runs:

| | cold start | of which container start | model load |
|---|---|---|---|
| r01, r02 — cache mount inert | 156.2s / 154.2s | 2.0s | 154.2s / 152.2s |
| r03 — after the fix, weights cached | **136.2s** | 2.0s | 134.2s |

So the download was **~18s of a ~155s cold start** for this 1.5GB model — real, worth
fixing, and considerably smaller than the raw number suggests. The other ~134s is
vLLM's own engine initialisation. That split is precisely what an undecomposed cold
start hides: Phase 1's recorded "158-162s" was 2s of container start, ~18s of
download nobody knew was happening, and ~140s of engine init, and nothing in the
number said so. Note the download cost scales with the model — Bielik-11B's AWQ is
~6.2GB, so a full Phase 5 matrix was on course to pay minutes of avoidable download
per run.

A third was caught and fixed by looking at the resulting document rather than the
summary line: the vLLM cell's energy figure rested on **9 tegrastats samples over
4.4s**, because at 108 tok/s it finished 8 repetitions before the 500ms sampler had
characterised anything. `benchmarks/harness.py` has a `min_measurement_s` floor for
exactly this reason and the rewrite had dropped it. Restored as
`--min-measurement-s` (default 10s), plus a `power_window_thin_<n>_samples` validity
flag so a thin figure can never read as authoritative as a full one.

### Still open in this phase

- [ ] `scripts/benchmark.py` (the non-streaming path through
      `benchmarks/harness.py`) has had none of the above applied: it still discards
      raw latencies, has no manifest or experiment ID, and sends an identical prompt
      every repetition, so its TTFT-equivalent figures carry the prefix-cache error
      above. Either retrofit it or retire it in favour of
      `benchmark_streaming.py` — ⟨DECIDE⟩, but do not run a campaign through it as
      it stands. `benchmarks/harness.py` itself is a hand-maintained copy shared with
      two sibling repos, so any change there has to be a deliberate divergence.
- [ ] Backfill the same treatment into `scripts/validate_tool_calling.py` /
      `validate_mmlu.py` so they emit `quality_result` documents (this is Phase 5's
      first task, listed there).

**GATE 2** — met for the streaming path, 2026-09-04: a real run on
`1.5b-q4-llamacpp-orin` produced raw per-run rows, an energy/token figure, a complete
manifest, an experiment ID, an explicit co-resident label, and passed schema
validation. **Not yet met for `scripts/benchmark.py`** — see "Still open" above.
Remaining before the gate closes fully: re-derive published aggregates from a raw
file as proof the raw data is sufficient.

## Phase 3 — Experiment configuration, separated from model configuration

**Done 2026-09-04.**

- [x] Promoted `schemas/*.json` from example instances to real JSON Schema
      (draft 2020-12), originals preserved as `schemas/examples/*.example.json`.
      Four schemas, not three: `quality_result` was added because note.md §5 keeps
      the quality and performance layers independently measurable, which is only
      enforceable if they are separate document types. `schemas/README.md` documents
      the two conformance levels — `model.schema.json` is written to what the
      registry satisfies today (so it guards live data now), the rest to what these
      phases produce.
- [x] `configs/benchmarks/{smoke,output_sweep,context_sweep}.yaml`, each validated
      against `experiment.schema.json` in the test suite. `configs/models.yaml`
      stays a candidate registry and absorbs none of this.
- [x] Sampling pinned explicitly in every config (`temperature: 0.0`). Never
      inherited — Phase 1 already proved llama.cpp's default ~0.8 changes
      *behaviour*, not just wording.
- [x] Prompt/template inputs versioned (note.md §21): `prompt_source`,
      `prompt_template_version` and `prompt_uniqueness` are recorded in every result.
- [x] `scripts/run_experiment.py` — the config-driven grid runner, and
      `benchmarks/runner.py` — the shared measurement core. Both entry points now go
      through one `measure_cell()`, so a second entry point cannot re-implement the
      measurement loop and drift; that drift has cost this repo a debugging session
      once already (`--enable-auto-tool-choice` in compose but not in the coordinator).
      One server session serves every cell of a model — a 12-cell vLLM grid
      restarting per cell would spend 27 minutes on cold starts — and cells after the
      first carry `cold_start_shared_across_cells_in_this_server_session` so N cells
      are never mistaken for N cold-start measurements.
- [x] Campaigns resume: a cell whose result already exists is skipped, not
      overwritten. A cell that fails is recorded and the rest continue — an OOM at
      2048 context is a result (note.md §54), not a reason to abort a campaign.
- [ ] Validate emitted `quality_result` documents in the test suite. Blocked on
      Phase 5 — `validate_tool_calling.py`/`validate_mmlu.py` don't emit them yet.

### Two bugs the tests and the runner caught before any campaign

**A config asked for more context than the candidates have.** `context_sweep.yaml`'s
first draft used the canonical 128/512/1024/**2048** ladder with `output_tokens: 128`
— but the context window has to hold input *and* output *and* the chat template, and
every current row pins 2048. `tests/test_experiment_configs.py` checks
`max(input) + max(output) <= ctx` per candidate and failed before a server was ever
started. Ladder is now 128/512/1024/**1536**, and the config says why. Note this is
also a real constraint on Phase 4: reaching 4096+ needs a per-row memory re-tune, not
a flag bump.

**A `standalone` result that wasn't.** The first real Phase 3 run declared
`execution_condition: standalone` while this device's production `vllm-orchestrator`
was up — because that is the natural thing to type, and nothing about the run looked
wrong. On unified memory those are co-resident numbers. A *mislabelled* result is
strictly worse than a missing one: it silently contaminates every table it joins.
`assert_condition_matches_reality()` now refuses a `standalone` claim when other
containers are running, naming them and printing the config lines that would declare
the truth. Deliberately an error with no override flag — the fix is one command
either way, and an override would get used. It detects containers only, so a
bare-metal process holding GPU memory still slips through; the message says what was
found rather than claiming the board is clean.

The result that triggered this was deleted rather than committed, which is why
`results/` holds no Phase 3 result: running `smoke.yaml` as written requires stopping
the production `vllm-orchestrator`, and that is not this plan's call to make.
⟨DECIDE⟩ — stop it for campaign runs (cleanest numbers, brief production downtime),
or run the whole campaign co-resident and label it honestly (no downtime, but every
figure carries the production container's memory footprint, and Phase 9's
standalone-vs-co-resident comparison loses its baseline).

**GATE 3** — met. A benchmark now runs from `(models.yaml key, configs/benchmarks/*.yaml,
git SHA)` with no CLI flag carrying experimental meaning; verified by a real two-cell
grid on `1.5b-q4-llamacpp-orin` (one server session, per-cell documents, shared
cold-start correctly flagged, thin-power windows correctly flagged, and a re-run
skipping both completed cells).

## Phase 4 — Canonical Orin campaign (the first complete vertical slice)

note.md §48's "minimum viable publication experiment", retargeted to the hardware that
actually exists. Deliberately one platform, one backend, one precision — every other
axis is held constant so the two swept axes mean something.

- [ ] **Context-length sweep** (note.md §11): input 128 / 512 / 1024 / 2048, batch=1,
      output=128. 4096+ only where `max_model_len` allows — every current `models.yaml`
      row is pinned to 2048, so raising it is itself an experimental change and needs
      its own memory re-tune, not a silent flag bump.
- [ ] **Output-length sweep** (note.md §12): output 16 / 32 / 64 / 128 / 256 / 512 at
      fixed input. This is *exactly* `embedded-ai-chain/docs/paper.md`'s **P3** core
      experiment ("generative extent can dominate the perception budget") — run it in
      the form P3 needs (total latency, decode latency, TTFT, p50/p95/p99, energy,
      generated tokens) so one campaign serves both documents.
- [ ] Per point measure: TTFT, prefill tok/s, decode tok/s, E2E latency, inter-token
      latency distribution, peak system RAM, power, J/output-token.
- [ ] ≥30 measured runs per cell, raw rows preserved, thermal state recorded, and any
      cell flagged invalid if throttling occurred (note.md §38).

**Mapping to `embedded-ai-chain/docs/paper.md`** — the reason this phase is worth its
cost beyond model selection:

| paper.md proposition | Evidence this repo produces |
|---|---|
| P3 — generative extent dominates | Phase 4's output-length sweep, verbatim |
| P4 — the decision to perceive has a metabolic cost | Phase 4 energy + Phase 5's BFCL "correct no-tool decision" rate: the cost of *deciding not to escalate* |
| P5 — hardware changes the optimal allocation | Phase 6 (Orin vs Thor, same models/inputs/methodology) |
| P6 — memory is an allocation resource | Phase 9 co-resident runs + every OOM recorded as a result |

**GATE 4** — both sweeps complete for at least one row, plots regenerable from raw
data, and the P3 table drafted into `paper.md` from real numbers.

## Phase 5 — Complete the scorecard matrix (was Phase 3)

Unchanged in intent from the original plan — this is what `docs/promotion-contract.md`
needs, and it is not superseded by the benchmark framing.

- [ ] Run `benchmark` / `benchmark-streaming` / `validate-tool-calling` /
      `validate-mmlu` for every row in `configs/models.yaml` that can serve at all:
      the remaining Qwen sizes (3B/7B) on both backends, and both Bielik rows.
      Record an OOM as a real result, not a skip. Note `bielik-11b-awq-vllm-orin`
      needs no `benchmark`/`benchmark-streaming` caveat (smoke-tested 2026-09-04,
      `gpu_memory_utilization: 0.3` confirmed working) but **cannot produce a
      tool-calling scorecard at all** — see Phase 1's record. That is a
      fail-with-reason entry, which GATE 5 accepts.
- [ ] Full-corpus BFCL, not `--limit 20`. The only real scorecard so far (75% overall,
      n=40, `1.5b-q4-llamacpp-orin`) is a sample, and Phase 1 says so.
- [ ] Report tool-calling as a confusion matrix (note.md §36), not one percentage:
      correct tool / correct arguments / correct no-tool / invalid call / hallucinated
      tool / formatting failure. The existing simple+irrelevance split already
      separates the two error directions — this makes that explicit.
- [ ] Write the comparison up (README "Current State" or a dedicated benchmarks doc).

**GATE 5** — every plausible row has a scorecard entry (pass, fail-with-reason, or
doesn't-fit-with-reason).

## Phase 6 — Thor access and the cross-platform arm (was Phase 2)

Demoted from "next" to "after the Orin slice is real" — it is a hardware-availability
dependency this repo does not control, and nothing above is blocked on it.

- [ ] Confirm SSH/LAN reachability to the user's Thor the same way `jetson-vlm-lab`'s
      Phase B did (`10.8.32.124` there — Thor's address here may differ, don't assume
      it's the same machine)
- [ ] Confirm what's already running on Thor (production VLM container, per
      `embedded-ai-chain`'s `RemoteVlmCoordinator` — avoid port/GPU-memory collision
      when standing up a temporary orchestrator-LLM test container alongside it, same
      caution `jetson-vlm-lab`'s `-thor` test rows already took)
- [ ] Record Thor's real power modes rather than assuming Orin's transfer (note.md
      §27), and its real unified-memory size rather than note.md §7's assumed 128GB
- [ ] Add `-thor` rows to `configs/models.yaml` for both backends once real serving
      args are confirmed there (don't guess `gpu_memory_utilization`/`n_gpu_layers`
      ahead of a real run, same rule Phase 1's orin rows already follow)
- [ ] Re-run Phase 4's two sweeps on Thor, same methodology, and answer P5's actual
      question: **does the ranking change**, not merely which board is faster
- [ ] `RemoteCoordinator` runs cannot measure the remote board's power or cold start —
      `benchmark.py` already flags this. Thor telemetry must be collected *on* Thor
      (note.md §56 step 7), not inferred from the client side.

**GATE 6** — one real run of each script against Thor succeeds end-to-end, and at
least one Phase 4 sweep is reproduced there with on-device telemetry.

## Phase 7 — Quantization study

- [ ] Same model, same backend, same hardware, different precision only (note.md §24).
      `7b-awq-vllm-orin` vs a 7B fp16 row is the natural first pair — **if** fp16 7B
      fits in ~30GB alongside nothing else; record the OOM as the result if not.
- [ ] Label quantization methods precisely, never as a generic "INT4" (note.md §23).
      `configs/models.yaml`'s `bielik-11b-awq-vllm-orin` row already does this right —
      it records `compressed-tensors int4, group_size=128` despite the repo being
      *named* `-awq`. Hold every future row to that bar.
- [ ] Per quantized candidate: smoke → MMLU regression → performance → memory → power,
      always against the same-model higher-precision baseline.
- [ ] Qwen3 row. **The expected blocker turned out not to exist**: the pinned
      `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` reports `vllm 0.19.0` (read from
      the running server 2026-09-04, not inferred from the tag), so the
      "vLLM ≥0.11 has no JetPack 6.2 wheels" constraint — which is about pip wheels —
      does not bind this container path. Qwen3 support is comfortably inside 0.19.0.
      Still add the row behind a real serving test, same bar as every other row, and
      note that `embedded-ai-chain/CLAUDE.md`'s "do not let a model choice force a
      runtime upgrade" rule is satisfied here without any upgrade at all: the runtime
      is already newer than that rule assumed. Worth propagating that correction back
      to `embedded-ai-chain/CLAUDE.md`, whose vLLM constraint is now misleading as
      written.

**GATE 7** — at least one same-model precision pair has quality *and* efficiency
numbers, with the quantization method named exactly.

## Phase 8 — Backend study

- [ ] Same model, same precision-class, same hardware, vLLM vs llama.cpp (note.md
      §50). The lab already has one qualitative finding to quantify: vLLM's
      `--tool-call-parser hermes` is temperature-robust where llama.cpp's Hermes-2-Pro
      detection is not (Phase 1). Turn that into a measured tool-call-reliability-vs-
      temperature curve for both backends.
- [ ] Note honestly that AWQ-vs-GGUF-Q4_K_M is not a clean precision match; the
      backend comparison is confounded by quantization format and must say so.

**GATE 8** — a backend comparison exists that states its own confounds.

## Phase 9 — Co-resident / concurrency (highest external value)

This is the phase that connects the lab to both papers, and to
`embedded-ai-chain/docs/TODO.md` Phase 4's still-unstarted concurrency-starvation
test — described in `PAPER_PLAN.md` as "the single most important unmeasured number in
the whole project."

- [ ] Condition A (standalone) vs Condition B (co-resident with the real pipeline:
      YOLO + STT + TTS + VLM), same model, same workload, same methodology.
- [ ] Report both directions, not just one: what the co-resident load does to LLM
      TTFT/decode/energy, **and** what the LLM does to `yolo_frame` p50/p95/p99. The
      second is the number `embedded-ai-chain` actually needs.
- [ ] Batch/concurrency sweep 1/2/4/8 (note.md §13) — but state plainly that batch=1
      is the deployment reality here and the rest is characterization, and never mix
      the two terms.
- [ ] Every OOM and allocation failure is a recorded result (paper.md P6), including
      the ones already on record: the VLM OOM at `gpu_memory_utilization=0.6`, the
      276MB-free thin-margin finding.

**GATE 9** — the concurrency-starvation number exists, whichever way it comes out.

## Phase 10 — Analysis pipeline

- [ ] `scripts/analyze.py` regenerating every table and plot from `results/raw/`
      (note.md §33). Never hand-edit a plot. The parent repo's `scripts/analyze_*.py`
      family is the pattern to follow.
- [ ] Pareto frontier over (quality, latency, memory, energy) — identify non-dominated
      candidates rather than inventing a weighted score (note.md §34).
- [ ] Keep the application scorecard separate from the raw benchmark (note.md §35):
      the promotion decision is allowed to weight axes; the benchmark is not.

**GATE 10** — every published figure regenerates from raw data with one command.

## Phase 11 — Declare (or defer) a winner (was Phase 4)

- [ ] Declare a winner per `docs/promotion-contract.md` §4, or explicitly record "no
      clean winner, deferred + tie-breaker condition"
- [ ] Add the pointer/decision-log row to `embedded-ai-chain/docs/TODO.md`
- [ ] If a winner changes the shipping backend/model, update
      `embedded-ai-chain/src/orchestrator_models.py`'s registry and
      `configs/orchestrator-vllm-compose.yml` (or add an equivalent llama.cpp compose
      file there) accordingly — that's `embedded-ai-chain`'s own change, made with a
      real number behind it, not this repo's to make directly

**GATE 11** — a decision (or an explicit, reasoned non-decision) is recorded in both
repos, not just left in this file.

## Deferred / explicitly not doing yet

- **TensorRT-LLM / TensorRT Edge-LLM** — see Phase 0's finding. Revisit only if Thor
  moves to JetPack 7.x; treat as a bespoke integration project even then, not a
  coordinator-class addition.
- **A declarative multi-machine launch config** — `jetson-vlm-lab`'s own TODO already
  judged this premature for a 2-machine reality; revisit together with
  `embedded-ai-chain`'s "embedded platform zoo" goal, not from this repo in isolation.
- **Deciding how many papers this work becomes.** Deferred to after the campaign, by
  direct decision 2026-09-04: the data decides. This lab feeds `paper.md` and
  `PAPER_PLAN.md` regardless; whether it *also* carries a standalone benchmark paper
  is a question the results answer better than a plan does. Nothing in Phases 2-10
  changes either way — which is exactly why it is safe to leave open.

### From note.md, deliberately not scheduled yet (not forgotten)

Each of these is a real note.md requirement, deferred with a reason rather than
silently dropped:

- **GSM8K / ARC-Challenge / IFEval / HumanEval / Belebele** (§17). MMLU + BFCL already
  cover the two axes this lab's decision turns on (quantization sanity, tool-call
  judgment). Add one at a time only when a specific claim needs it — a benchmark suite
  nobody reads is pure campaign cost.
- **Power-mode matrix** (§27). Every result already records its power mode; *sweeping*
  power modes multiplies the whole matrix. Defer until one campaign is complete at a
  single documented mode.
- **Benchmark tiers 0-3** (§18). Worth having; cheap to add once `configs/benchmarks/`
  exists (Phase 3) — it is essentially a sample-count field. Not a separate phase.
- **Custom application dataset** (§19). This is `embedded-ai-chain`'s orchestrator
  transcripts, which do not exist in reusable volume yet. Revisit after Phase 9
  produces real co-resident traffic. BFCL stays the external-comparability anchor
  regardless (§19 agrees).
- **Reasoning/thinking-token accounting** (§22). No reasoning model is in the registry.
  Becomes mandatory the moment a Qwen3 thinking-mode row is added (Phase 7) — do not
  compare a thinking model on visible output tokens alone.
- **Immutable `benchmark-v1.0` release** (§43). A packaging step, correct to do last.

---

## Decision log (append-only)

Rows are never deleted, even when superseded — same convention as
`embedded-ai-chain/docs/TODO.md`.

| Date | Decision | Reason |
|---|---|---|
| 2026-09-04 | Repo becomes a benchmark suite that also selects the orchestrator LLM, not one or the other | `docs/note.md`; the selection scorecard is a strict subset of the benchmark data |
| 2026-09-04 | Orin is the platform of record; Thor is the cross-platform arm | No Thor access confirmed yet; and "does the ranking change across hardware" needs both boards anyway (paper.md P5) |
| 2026-09-04 | Existing harness warmup/steady-state policy kept over note.md §4's fixed `warmup: 5` | Temperature-driven steady state with an honest `warmup_reached_steady_state` flag is strictly stronger than a guessed count |
| 2026-09-04 | Measurement-integrity retrofit (raw rows, energy, manifest, IDs, co-residency flag) precedes all campaigns | These are unrecoverable if skipped — a campaign run without them cannot be re-analysed |
| 2026-09-04 | ~~No third paper from this repo~~ **superseded same day** — see next row | Two papers already share one writing window to 2027-01-25 |
| 2026-09-04 | **How many papers is decided after the results exist, not now.** This repo is built as the instrument for `paper.md`'s P3/P4/P5/P6 either way; whether the benchmark data also carries a paper of its own is answered by the data | Direct user decision. The earlier row pre-committed to an answer that the campaign itself is better placed to give — and nothing in Phases 2-10 changes based on which way it goes, so there is no cost to deferring it |
| 2026-09-04 | Prompts vary per repetition by default (`--prompt-uniqueness unique-per-run`), and the choice is a recorded schema field | Identical prompts made both backends serve every repetition from a KV-cache hit; measured 6.6x TTFT error (40.8ms vs 268.1ms p50) in the flattering direction. See Phase 2's finding |
| 2026-09-04 | `HUGGINGFACE_HUB_CACHE`/`HF_HUB_CACHE` set explicitly in both the coordinator and compose | The image bakes in its own value that overrides `HF_HOME`, making the host cache mount inert; every vLLM run re-downloaded its weights into a `--rm` container. Measured: 156.2s -> 136.2s cold start for a 1.5GB model, and it scales with model size |
| 2026-09-04 | Measurement continues past `--runs` until `--min-measurement-s` elapses | A vLLM cell finished 8 repetitions in 4.4s and produced an energy figure backed by 9 power samples; the sampler needs wall-clock time, not repetitions |
| 2026-09-04 | `write_result()` refuses to overwrite an existing result | note.md §29 immutability, enforced rather than trusted — a re-run must be a new replicate, never a silent replacement of data a figure was drawn from |
| 2026-09-04 | Apertus and Bielik stay in the matrix despite note.md not mentioning them | They are this lab's actual second/third families and its only decisive negative result |
| 2026-09-04 | ~~Qwen3 gated because vLLM ≥0.11 has no JetPack 6.2 wheels~~ **superseded same day** — premise measured false for this path | The pip-wheel constraint does not apply to NVIDIA's container; see next row |
| 2026-09-04 | Qwen3 stays gated on a real serving test — but as routine practice, not because of a known blocker | `GET /version` on the pinned image returns `vllm 0.19.0`, running fine on this JetPack 6.2 board. Second time a version was assumed from a name and was wrong; both times the fix was asking the running server |
