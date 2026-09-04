# TODO — orchestrator-LLM selection lab

Action list for turning this repo into a working multi-candidate orchestrator-LLM
selection lab, per `docs/promotion-contract.md`. Phase-gated like
`embedded-ai-chain`'s own `docs/TODO.md` and `jetson-vlm-lab`'s own `docs/TODO.md` -
don't start a phase until the one before it is checked off.

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
- [ ] Real Docker/GPU smoke test: `make serve-1.5b-llamacpp` - **first real use of
      the unverified `dustynv/llama_cpp` image tag** (see
      `src/llm_coordinator.py`'s `_DEFAULT_LLAMACPP_IMAGE` comment) - confirm the tag,
      update the constant/compose file if it's wrong, and confirm `--jinja` actually
      produces structured `tool_calls` for Qwen2.5's chat template before trusting
      `validate_tool_calling.py`'s results against this backend at all
- [ ] `uv venv && uv pip install -r requirements.txt && make test` on real hardware
      (currently only written/self-consistent, not yet run)
- [ ] Stage BFCL + MMLU on-device per README's "Dataset staging" section (neither is
      bundled) before `validate-tool-calling`/`validate-mmlu` can run for real

**GATE 1** — met once both smoke tests pass and `make test` is green for real, not
just self-consistent on inspection.

## Phase 2 — Thor access

- [ ] Confirm SSH/LAN reachability to the user's Thor the same way
      `jetson-vlm-lab`'s Phase B did (`10.8.32.124` there - Thor's address here may
      differ, don't assume it's the same machine)
- [ ] Confirm what's already running on Thor (production VLM container, per
      `embedded-ai-chain`'s `RemoteVlmCoordinator` - avoid port/GPU-memory collision
      when standing up a temporary orchestrator-LLM test container alongside it, same
      caution `jetson-vlm-lab`'s `-thor` test rows already took)
- [ ] Add `-thor` rows to `configs/models.yaml` for both backends once real serving
      args are confirmed there (don't guess `gpu_memory_utilization`/`n_gpu_layers`
      ahead of a real run, same rule Phase 1's orin rows already follow)

**GATE 2** — one real run of each script against Thor succeeds end-to-end, mirroring
jetson-vlm-lab's own Gate B.

## Phase 3 — Full matrix + scorecard

- [ ] Run `benchmark` / `benchmark-streaming` / `validate-tool-calling` /
      `validate-mmlu` for every row in `configs/models.yaml` (2 backends x 3 sizes x
      2 platforms = up to 12 runs, fewer if a size doesn't fit on a given
      platform/backend - record the OOM as a real result, not a skip)
- [ ] Write up the comparison (README "Current State" or a dedicated benchmarks doc),
      same as jetson-vlm-lab's own Qwen-vs-Phi writeup

**GATE 3** — every plausible row has a scorecard entry (pass, fail-with-reason, or
doesn't-fit-with-reason).

## Phase 4 — Declare (or defer) a winner

- [ ] Declare a winner per `docs/promotion-contract.md` §4, or explicitly record "no
      clean winner, deferred + tie-breaker condition"
- [ ] Add the pointer/decision-log row to `embedded-ai-chain/docs/TODO.md`
- [ ] If a winner changes the shipping backend/model, update
      `embedded-ai-chain/src/orchestrator_models.py`'s registry and
      `configs/orchestrator-vllm-compose.yml` (or add an equivalent llama.cpp compose
      file there) accordingly - that's `embedded-ai-chain`'s own change, made with a
      real number behind it, not this repo's to make directly

**GATE 4** — a decision (or an explicit, reasoned non-decision) is recorded in both
repos, not just left in this file.

## Deferred / explicitly not doing yet

- **TensorRT-LLM / TensorRT Edge-LLM** — see Phase 0's finding. Revisit only if Thor
  moves to JetPack 7.x; treat as a bespoke integration project even then, not a
  coordinator-class addition.
- **A declarative multi-machine launch config** — `jetson-vlm-lab`'s own TODO already
  judged this premature for a 2-machine reality; revisit together with
  `embedded-ai-chain`'s "embedded platform zoo" goal, not from this repo in isolation.
