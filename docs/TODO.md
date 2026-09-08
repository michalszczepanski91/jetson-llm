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
      - **SUPERSEDED for Thor only, 2026-09-08** — the revisit condition above fired,
        and two of the three blocking facts turned out not to hold any more. Checked
        live against `NVIDIA/TensorRT-Edge-LLM` (release 0.10.1, pushed 2026-09-03):
        (a) the user's Thor **is** JetPack 7.1 / CUDA 13.0 / TensorRT 10.13.3.9, and
        `Jetson Thor + JetPack 7.0/7.1` is an **Official** row in Edge-LLM's own
        support matrix — not merely "compatible"; (b) it now ships an **experimental
        OpenAI-compatible HTTP server** (`tensorrt-edgellm-serve <checkpoint>
        --port N`, redesigned in 0.10.1) *with* tool-calling support
        (`experimental/server/parsing/tool_calling.py` + `tool_chat_template.py`,
        covered by upstream unit tests) — i.e. it can meet
        `docs/promotion-contract.md` §1's contract as an ordinary coordinator target,
        which is exactly what the 2026-09-04 entry said it could not do; (c) Qwen2.5
        **is** in its supported-models list, including
        `Qwen2.5-{1.5B,3B,7B}-Instruct-AWQ` — the same checkpoints the `-orin` vllm
        rows already use, so the Thor Edge-LLM leg can hold the model constant
        instead of forcing a Qwen3.x switch. Orin stays excluded (JetPack 6.2.x;
        Edge-LLM's Orin row needs JetPack 7.2). See Phase 6 for the Thor plan.
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

**GATE 1** — met for Qwen2.5-1.5B on both backends, and for Bielik-11B on llama.cpp.
Apertus is a closed question (blocked, documented) rather than an open one. Still
pending before the full matrix: the remaining Qwen sizes (3B/7B), and a real
BFCL/MMLU scorecard run for every row that's actually able to serve at all.

*(Written when that matrix was Phase 3; it is Phase 5 after the 2026-09-04
renumbering below. Left as written rather than edited, per this file's append-only
convention.)*

---

# Scope change, 2026-09-04 — `docs/note.md` merged into this plan

`docs/note.md` (added 2026-09-04 alongside `docs/progect.diagram.md` and the
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
- **note.md §45/§46 plan a Qwen2.5-vs-Qwen3.x campaign; this repo has no Qwen3 row and
  may not be able to serve one.** vLLM ≥0.11 has no prebuilt JetPack 6.2/CUDA 12.6
  wheels and this lab is pinned to `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`;
  whether *that* build serves Qwen3 is unknown. Qwen3 is a Phase 7 task gated on a
  real check, not a planning assumption.
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
  Perception position paper). **Decision: this repo is not writing a third paper. It
  is the instrument that produces `paper.md`'s P3/P4/P5/P6 evidence** — see the
  mapping in Phase 4. Revisit only if the data turns out to justify a standalone
  benchmark paper on its own.

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

- [ ] **Persist raw per-run measurements** (note.md §10 — the single biggest gap).
      `run_benchmark()` currently computes `percentiles(latencies)` and then discards
      `latencies`; `benchmark_streaming.py` keeps its per-run list in memory and
      writes only percentiles. Both must write every run. Aggregates get regenerated
      from raw, never stored as the only copy.
- [ ] **Energy, and the split-brain that currently prevents it.** `benchmark.py` has
      power (tegrastats, via the harness) but does not know how many tokens were
      generated; `benchmark_streaming.py` knows the token count but never starts a
      sampler at all. So `J/token` — note.md §14's headline metric and `paper.md`
      §7.1's `E_decision` — is presently *uncomputable from either script*. Fix:
      give the streaming script the harness's sampler, then derive
      `energy_joules` (rail power integrated over the measured interval) and
      `energy_per_output_token_j`.
- [ ] **Real token accounting instead of counted SSE chunks.** `_stream_one()` counts
      content deltas, which is a chunk count, not a token count, and it has no input
      token count at all. Request `usage` from the server (`stream_options:
      {"include_usage": true}` on vLLM; confirm llama-server's equivalent rather than
      assuming parity — this lab has already been bitten twice by assumed
      vLLM/llama.cpp parity). Record `input_tokens`, `output_tokens` separately.
- [ ] **Decompose cold start** (note.md §37). Bielik's recorded 338s "cold start"
      is mostly a 6.7GB model *download*, and the 1.5B llama.cpp row's 2.0s is a
      warm-cache number — both already flagged in Phase 1, neither yet separated in
      code. Split into `download_s` / `container_start_s` / `model_load_s` /
      `server_ready_s`, and record which of them the run actually paid.
- [ ] **Environment manifest per run** (note.md §31). Currently absent: git commit of
      this repo, backend version, container image tag+digest, dataset version,
      Docker version, the exact command line. Add a `manifest` block to every result
      row. Backend version must be read from the running server, not from the image
      tag — the tag `0.3.9-r36.4.0-cu128-24.04` is not a llama.cpp version, and this
      lab has already had to identify builds 4579 vs 5058 vs 5283 by behaviour.
- [ ] **Experiment IDs and a result hierarchy** (note.md §29/§30). Replace the
      hand-passed `--results-json` path with `results/raw/<experiment_id>/`.
      Historical outputs under `output/` are not rewritten — they are moved, not
      reformatted, and the move is recorded here.
- [ ] **`standalone` vs `co-resident` on every row** (note.md §15). One required
      field, no default — a run must state which it was. This is the field that makes
      Phase 9 possible at all, and it is the distinction `embedded-ai-chain`'s own
      Phase 4 concurrency test depends on.

**GATE 2** — one full `benchmark` + `benchmark-streaming` pair on
`1.5b-awq-vllm-orin` produces: raw per-run rows, an energy/token figure, a complete
manifest, an experiment ID, and an explicit standalone/co-resident label. Re-derive
the published aggregates from the raw file as proof the raw data is sufficient.

## Phase 3 — Experiment configuration, separated from model configuration

- [ ] Promote `schemas/*.json` from example instances to real JSON Schema
      (`$schema`, `type`, `properties`, `required`), keeping the current files as
      `schemas/examples/` fixtures. Validate every written result against
      `benchmark_result.schema.json` in the test suite, so a malformed row fails in
      CI rather than three months into a campaign.
- [ ] Add `configs/benchmarks/{smoke,latency,streaming,context,output,quality}.yaml`
      (note.md §8) holding input/output token grids, warmup/measurement counts, and
      sampling params. `configs/models.yaml` stays a *candidate registry* and does not
      absorb these.
- [ ] Pin sampling explicitly in every benchmark config (`temperature: 0.0` for
      performance runs). Do not inherit server defaults — Phase 1 already proved
      llama.cpp's default ~0.8 changes *behaviour*, not just wording.
- [ ] Version the prompt/chat-template inputs (note.md §21). Record the tool schema
      and system prompt used, with a version string, in the manifest.

**GATE 3** — a benchmark can be re-run from `(models.yaml key, benchmarks/*.yaml,
git SHA)` alone, with no CLI flags carrying experimental meaning.

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
      the remaining Qwen sizes (3B/7B) on both backends, `bielik-11b-awq-vllm-orin`
      (added 2026-09-04, never run — its `gpu_memory_utilization: 0.3` is an untested
      guess), `bielik-11b-q4-llamacpp-orin`. Record an OOM as a real result, not a skip.
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

- [x] Confirm SSH/LAN reachability to the user's Thor — moot as of 2026-09-08: the
      user cloned this repo directly onto Thor and runs Claude Code there too, so
      this repo's own working copy on that device *is* the reachability path, no
      SSH hop needed the way `jetson-vlm-lab`'s Phase B required.
- [x] Confirm what's already running on Thor, 2026-09-08: `docker ps` shows
      `vllm-vlm-thor` (image `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` —
      **note the `-thor` image tag, distinct from this repo's own
      `_DEFAULT_VLLM_IMAGE`/`docker-compose.yml`, both still hardcoded to
      `-jetson-orin`**), serving `Qwen2.5-VL-7B-Instruct-AWQ` with
      `--gpu-memory-utilization 0.3 --max-model-len 4096` on `0.0.0.0:8000`
      (confirmed via `ss -tln`) — **this repo's default vLLM port (8000, hardcoded
      in both `docker-compose.yml` and `VllmCoordinator`'s default) collides
      directly** since both run with `network_mode: host`. Every `-thor` row below
      must pass an explicit non-8000 `port`. Two more containers
      (`tensorrt-edge-llm`, `edgellm-export`, both `nvcr.io/nvidia/pytorch:26.05-py3`)
      sit exited from an earlier TensorRT Edge-LLM experiment — consistent with
      Phase 0's finding that Edge-LLM needs JetPack 7.x (see next item), evidence
      someone already probed that path on this box; not investigated further here.
- [x] Recorded Thor's real power mode and unified-memory size, 2026-09-08:
      `nvpmodel -q` → **120W** (mode 1, no sudo needed to read it). `free -h` →
      **122GiB total** (vs Orin's ~30GB — a >4x difference, `gpu_memory_utilization`
      fractions do NOT transfer between platforms, confirm each `-thor` row's value
      independently). L4T `R38.4.0` / `nvidia-jetpack 7.1-b112` — genuinely JetPack
      7.x, unlike Orin's 6.2.x, which is the version Phase 0 flagged as required for
      TensorRT Edge-LLM (still not pursued here — out of scope per that decision,
      see the exited containers above).
- [x] Added `-thor` rows to `configs/models.yaml` for the vllm backend (mirroring
      every existing `-orin` vllm row: 1.5b/3b/7b Qwen2.5-AWQ + Bielik-11B-AWQ),
      2026-09-08 — `gpu_memory_utilization` values are conservative untested
      starting points (same convention Phase 1's orin rows used), chosen small
      enough to fit alongside `vllm-vlm-thor`'s own already-resident 0.3 share of
      this same memory pool; re-tune upward after each row's first real run same as
      any orin row. **llama-cpp on Thor needs a different image family than the
      Orin rows — and the first answer here was WRONG, worth recording as a
      method note.** Sequence of findings, all 2026-09-08:
      1. `dustynv/llama_cpp` on **Docker Hub** (this lab's Orin source) publishes
         no `r38` tag, only r35.x/r36.x. Concluded from that alone that "no Thor
         image exists" — **premature**: jetson-containers publishes its r38-era
         images to **GHCR under `nvidia-ai-iot/`**, not to that Docker Hub repo.
         Checking one registry is not checking the project.
      2. Tried upstream's own `ghcr.io/ggml-org/llama.cpp:server-cuda` (it does
         publish a linux/arm64 manifest). It gets impressively far —
         `--list-devices` reports `CUDA0: NVIDIA Thor (125771 MiB)`, the fp16 GGUF
         loads, and `llama_server: model loaded` — then dies on the **first
         inference request**: `CUDA error: an internal operation failed, in
         function cublas_handle (ggml-cuda/common.cuh:1537)`. So upstream's generic
         CUDA arm64 build is *enumeration*-compatible with Tegra but not
         *runtime*-compatible. Device detection is not proof of a working backend —
         this lab should not accept `--list-devices` as a smoke test again.
      3. The right image is **`ghcr.io/nvidia-ai-iot/llama_cpp:r38.2.arm64-sbsa-cu130-24.04`**
         (same naming pattern as the `nvidia-ai-iot/ollama` r38 image already on
         this box) — NVIDIA's own Thor build.
      Note the CPU-only `ghcr.io/ggml-org/llama.cpp:full` also sits on this box
      (pulled by the other user) — no CUDA libs at all, not a serving candidate.
- [ ] Re-run Phase 4's two sweeps on Thor, same methodology, and answer P5's actual
      question: **does the ranking change**, not merely which board is faster
- [ ] `RemoteCoordinator` runs cannot measure the remote board's power or cold start —
      `benchmark.py` already flags this. Thor telemetry must be collected *on* Thor
      (note.md §56 step 7), not inferred from the client side.
- [x] Staged BFCL/MMLU datasets on Thor, 2026-09-08, at the same `/opt/datasets/{BFCL,MMLU}`
      paths the scripts expect — `/opt/datasets` turned out to be `root:mlusers`
      `drwxrwsr-x`, so this needed **no sudo** and the files land group-readable
      (`michal:mlusers`), reusable by the box's other user rather than duplicated per
      home directory. Same convention as the shared `/opt/hf-cache`.
- [x] Thor host-side Python env, 2026-09-08: `uv` was missing (`python3` is 3.12.3
      via apt, and Ubuntu 24.04's PEP-668 marking makes a plain system `pip install`
      refuse anyway) — installed user-local via the official installer, then
      `uv venv && uv pip install -r requirements.txt`, so every README/Makefile
      command works verbatim here. `hf`/`huggingface-cli` 1.30.0 added to the same
      venv for staging. `pytest`/`pyyaml` also happen to exist system-wide
      (52/52 unit tests pass under both).

### Phase 6a — TensorRT Edge-LLM: Thor's third backend leg

Direct user direction, 2026-09-08: **Edge-LLM is the point of running on Thor**, not
an afterthought behind the vLLM rows. Phase 0's exclusion is superseded for Thor only
(see that entry) — Orin cannot run it at all, so this leg is inherently Thor-only and
its results are a *platform×backend* cell no Orin row can fill.

- [x] Cloned `NVIDIA/TensorRT-Edge-LLM` (0.10.1, `e8b2952`) to `~/dev/TensorRT-Edge-LLM`
      and **built it on-device, 2026-09-08 — it works**. Configure picked up CUDA
      13.0, the system TensorRT at `/usr`, and the **sm_110** CuTe DSL prebuilt
      (Thor's compute capability, reported as 11.0 at runtime), then built clean with
      `-j 12` (12 of 14 cores, deliberately leaving headroom for the box's other
      user): `libNvInfer_edgellm_plugin.so` plus the
      `_edgellm_runtime.cpython-312-aarch64-linux-gnu.so` binding. `pip install -e
      ".[server,server-tools,native-build]"` into a `--system-site-packages` venv
      then gave working `tensorrt-edgellm-serve`. Whole path is ~1h unattended,
      needs no sudo and no GPU. Its own venv lives in that tree, separate from this
      repo's.
- [x] **Its serve CLI is nearly flag-compatible with `VllmCoordinator`'s docker
      command**: `--host/--port/--enable-auto-tool-choice/--tool-call-parser` all
      exist, the parser choices being `{auto,generic,hermes,qwen3_xml,nemotron,openai}`
      — note `hermes` and `generic` are the *same two names* vLLM and llama.cpp use
      for the paths this lab already measured on Orin. An `EdgeLlmCoordinator` is
      therefore a near-copy of `VllmCoordinator`, not new architecture.
- [x] **Qwen2.5-1.5B-Instruct-AWQ does NOT build on Edge-LLM's server path,
      2026-09-08** — the first real negative result of this leg, and an upstream
      limitation rather than a config error on our side. The server always builds
      through the experimental *direct* (checkpoint→engine, no ONNX) builder with
      `--externalize-weights all`; for `quant=int4_awq` on a `qwen2` graph that
      combination reaches `backend.py::_add_bias` with an externalized bias whose
      `bias_recipe` is `None` and raises `ValueError: external FP16 bias has no
      checkpoint recipe`. Qwen2-family models carry attention QKV biases (Qwen3
      dropped them), so this is precisely the Qwen2.5 + AWQ + externalization corner.
      Consequence for this lab: **the Thor Edge-LLM leg cannot hold precision
      constant with the `-orin` AWQ rows.** It must vary either precision (FP16 same
      model — being tried next) or family (Qwen3.x, which is what upstream's own
      server examples use, and which Phase 7 wants a serving check for anyway).
      Not attempted: patching upstream's builder — out of scope for a serving
      evaluation, revisit only if Edge-LLM turns out to matter enough.
- [ ] ~~Clone and build~~ *(done above)* — remaining: pick the checkpoint this leg
      standardizes on, given the AWQ block above
      (JetPack 7.1 Thor command from upstream's installation guide:
      `-DEMBEDDED_TARGET=jetson-thor -DCUDA_CTK_VERSION=13.0 -DTRT_PACKAGE_DIR=/usr
      -DENABLE_CUTE_DSL=ALL`, plus `-DBUILD_PYTHON_BINDINGS=ON` — the bindings are
      **required** for the OpenAI server, not optional). Prerequisites verified
      present on the box, 2026-09-08: cmake 3.28.3, gcc 13.3, CUDA 13.0.48 at
      `/usr/local/cuda` (note `nvcc` is NOT on `PATH` by default), TensorRT 10.13.3.9
      under `/usr`, 14 cores, 199GB free disk (upstream wants 20-50GB for ONNX +
      engines, plus the server's default 50GiB LRU engine-bundle cache).
      **No prebuilt wheel exists** — not on PyPI, no release assets on any of the
      last five GitHub releases (checked live). Source build is the only path.
- [x] Served it — FP16, not AWQ (the AWQ attempt is the blocked item above). The
      `RemoteCoordinator`-against-localhost trick worked exactly as predicted: the
      first real Edge-LLM datapoint in this lab needed **zero repo code changes**.
- [x] **Ran the full accuracy matrix — all three Thor backends, one fixed model
      (`Qwen2.5-1.5B-Instruct` FP16), 2026-09-08.** Write-up in
      `docs/thor-framework-comparison.md`, raw JSON in `output/`. BFCL overall:
      **Edge-LLM 83% > vLLM 75% > llama.cpp 74%**, with the entire spread coming from
      `irrelevance` (78/68/60) rather than `simple` (88/82/88). MMLU control
      42.0/40.5/38.5 with zero unparsed responses on any leg — which is what makes
      the BFCL spread trustworthy rather than a broken-setup artifact. llama.cpp's
      shape (strong `simple`, weak `irrelevance`) reproduces the Orin result
      (85/65/75) on different hardware AND a different image family: the
      grammar-CONSTRAINED path forces a valid call where abstaining was correct.
- [x] Added `1.5b-fp16-edgellm-thor`, `1.5b-fp16-vllm-thor` and
      `1.5b-fp16-llamacpp-thor` — each only after it really served.
- [x] Built `EdgeLlmCoordinator` (process-owning, not container-owning) so the
      backend is a first-class `--target local` citizen and `cold_start_ms` becomes
      measurable for it; `LlamaCppCoordinator` gained `server_argv0` for the two
      llama.cpp image families' differing ENTRYPOINTs. 61 unit tests pass.
- [ ] **Blocked on an exclusive window**: `benchmark.py` and
      `benchmark_streaming.py` for all three legs. Two complications already visible
      that those runs must handle — **cold start is BIMODAL on two of the three
      backends** and must be reported as two numbers rather than averaged: vLLM's
      first start on Thor takes minutes (torch.compile/inductor) but 27.8s to init
      the engine once the compile cache is warm; Edge-LLM compiles TensorRT engines
      on a cache miss (minutes) and only loads them on a hit (seconds).
- [ ] Cheap follow-up that would sharpen the headline: re-run BFCL with
      `--tool-call-parser hermes` pinned on BOTH Edge-LLM and vLLM. The parsers
      differed (`auto` vs `hermes`), so today's honest claim is "Edge-LLM **as
      configured by default** judges best on Thor", not that its runtime is
      inherently better.

### Measurement integrity on Thor: it is a SHARED box

Unlike the Orin, this Thor has another user's work on it (`docker ps` shows a
long-running production `vllm-vlm-thor`; two exited Edge-LLM containers mount
another user's home directory). That user has offered to coordinate exclusive
windows, so spend
them only where exclusivity actually changes the number:

- **Needs an exclusive box** (ask first, then run): everything in `benchmark.py` and
  `benchmark_streaming.py` — wall latency, TTFT, tokens/sec, cold start — plus all
  `tegrastats` power/thermal readings (they are board-wide, so any other GPU work
  contaminates them), and any `gpu_memory_utilization`/memory-fit tuning (a fit
  measured next to someone else's resident model is not reproducible).
- **Fine to run on the shared box**: `validate_tool_calling.py`, `validate_mmlu.py`,
  `make test`, dataset staging, the Edge-LLM source build (CPU-bound). Accuracy is
  not perturbed by co-residency — only speed and memory headroom are.
- Record which regime each run used. The harness already carries a co-residency flag
  (Phase 3's measurement-integrity retrofit); a shared-box latency number that is not
  flagged as such is worse than no number.

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
- [ ] Qwen3 row, gated on a real check first: does the pinned
      `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` build actually serve a Qwen3
      checkpoint? If not, the Qwen2.5-vs-Qwen3 comparison is llama.cpp-only, and that
      constraint must be stated in the writeup rather than worked around by upgrading
      vLLM (which `embedded-ai-chain/CLAUDE.md` explicitly forbids: "do not let a model
      choice force a runtime upgrade").

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
- **A third paper from this repo.** See the scope-change section. This lab feeds
  `paper.md` and `PAPER_PLAN.md`; it does not compete with them for the same writing
  window.

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
| 2026-09-04 | No third paper from this repo; it is the instrument for `paper.md` P3/P4/P5/P6 | Two papers already share one writing window to 2027-01-25 |
| 2026-09-04 | Apertus and Bielik stay in the matrix despite note.md not mentioning them | They are this lab's actual second/third families and its only decisive negative result |
| 2026-09-04 | Qwen3 is gated on a real serving check, not assumed | vLLM ≥0.11 has no JetPack 6.2/CUDA 12.6 wheels; a model choice must not force a runtime upgrade |
| 2026-09-08 | Thor's vllm rows get their own image tag and a non-8000 port, not copies of the orin rows' values | `vllm-vlm-thor` is already live in production on Thor at port 8000 using the `-jetson-thor` image tag, not `-jetson-orin` - confirmed via `docker ps`, not assumed |
| 2026-09-08 | llama-cpp/thor stays row-less (blocked), not added with a guessed image tag | `dustynv/llama_cpp` publishes no r38/Thor tag yet (checked live) - same "no row until a real serving path exists" rule Apertus's llama.cpp block already set |
| 2026-09-08 | TensorRT Edge-LLM becomes a real backend leg, on Thor only | Direct user direction; and Phase 0's three blockers no longer hold on Thor - JetPack 7.1 is an Official row, Edge-LLM 0.10.1 ships an OpenAI-compatible server with tool-calling, and Qwen2.5-*-AWQ is in its supported models |
| 2026-09-08 | Thor timing/power/memory-fit runs require an exclusive box; accuracy runs do not | Thor is shared with another user running a production vLLM container - co-residency changes latency/thermal/fit numbers but not BFCL/MMLU correctness |
