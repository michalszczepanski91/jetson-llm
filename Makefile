# ─────────────────────────────────────────────────────────────────────────────
#  Makefile — jetson-llm developer shortcuts
# ─────────────────────────────────────────────────────────────────────────────
# serve-*'s env vars are hand-kept in sync with configs/models.yaml (the
# actual source of truth the benchmark/validate scripts read) - these
# targets exist for manual interactive smoke tests (curl, scripts/*.py
# against an already-running container), not for the benchmark scripts
# themselves, which start/stop their own container via llm_coordinator.py.
.PHONY: help serve-1.5b-vllm serve-1.5b-llamacpp stop stop-llamacpp \
        benchmark experiment benchmark-streaming validate-tool-calling validate-mmlu test clean clean-docker

COMPOSE := docker compose

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

# ── Serving (manual smoke tests) ────────────────────────────────────────────
serve-1.5b-vllm:      ## Start vLLM serving Qwen2.5-1.5B-Instruct-AWQ in the foreground
	MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ GPU_MEMORY_UTILIZATION=0.15 MAX_MODEL_LEN=4096 $(COMPOSE) up vllm

serve-1.5b-llamacpp:  ## Start llama-server serving Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M) in the foreground
	MODEL=Qwen/Qwen2.5-1.5B-Instruct-GGUF QUANT=Q4_K_M $(COMPOSE) -f docker-compose.llamacpp.yml up llamacpp

stop:           ## Stop the vLLM container
	$(COMPOSE) down

stop-llamacpp:  ## Stop the llama.cpp container
	$(COMPOSE) -f docker-compose.llamacpp.yml down

# ── Benchmarking / validation (each starts+stops its own container) ────────
benchmark:      ## Latency/cold-start/thermal/power - pass CONFIG=<model-config> (default 1.5b-awq-vllm-orin)
	uv run python scripts/benchmark.py --model-config $(or $(CONFIG),1.5b-awq-vllm-orin) \
	  --results-json output/benchmark_$(or $(CONFIG),1.5b-awq-vllm-orin).json

experiment:     ## Run a full experiment from a config - EXP=configs/benchmarks/<name>.yaml [DRY=1]
	@test -n "$(EXP)" || { \
	  echo "error: EXP is required, e.g."; \
	  echo "  make experiment EXP=configs/benchmarks/smoke.yaml"; \
	  echo "  make experiment EXP=configs/benchmarks/output_sweep.yaml DRY=1"; \
	  echo "Available:"; ls configs/benchmarks/*.yaml | sed 's/^/  /'; \
	  exit 1; }
	uv run python scripts/run_experiment.py $(EXP) $(if $(DRY),--dry-run,)

benchmark-streaming: ## TTFT/tok-s/memory/power/ENERGY - CONFIG=<model-config> CONDITION=standalone|co-resident [CORESIDENT="yolo stt"]
	@test -n "$(CONDITION)" || { \
	  echo "error: CONDITION is required and has no default."; \
	  echo "  make benchmark-streaming CONFIG=1.5b-q4-llamacpp-orin CONDITION=standalone"; \
	  echo "  make benchmark-streaming CONFIG=... CONDITION=co-resident CORESIDENT=\"yolo stt tts\""; \
	  echo "On unified memory a standalone and a co-resident number are different physical"; \
	  echo "quantities (docs/note.md 15) - the run must say which it was."; \
	  exit 1; }
	uv run python scripts/benchmark_streaming.py --model-config $(or $(CONFIG),1.5b-awq-vllm-orin) \
	  --execution-condition $(CONDITION) $(if $(CORESIDENT),--co-resident $(CORESIDENT),)
	@echo "results written under results/raw/ - output/*.json is no longer used by this target"

validate-tool-calling: ## BFCL tool-call judgment accuracy - pass CONFIG=<model-config> (needs /opt/datasets/BFCL, see README)
	uv run python scripts/validate_tool_calling.py --model-config $(or $(CONFIG),1.5b-awq-vllm-orin) \
	  --results-json output/tool_calling_$(or $(CONFIG),1.5b-awq-vllm-orin).json

validate-mmlu:  ## MMLU quantization-sanity accuracy - pass CONFIG=<model-config> (needs /opt/datasets/MMLU, see README)
	uv run python scripts/validate_mmlu.py --model-config $(or $(CONFIG),1.5b-awq-vllm-orin) \
	  --results-json output/mmlu_$(or $(CONFIG),1.5b-awq-vllm-orin).json

# ── Dev ───────────────────────────────────────────────────────────────────
test:           ## Run the unit test suite (no Docker/GPU needed)
	uv run pytest tests/ -q

# ── Housekeeping ──────────────────────────────────────────────────────────
clean:          ## Remove __pycache__, .pyc
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true

clean-docker:   ## Stop and remove any lingering vllm/llamacpp containers from this lab
	docker stop vllm-llm-lab llamacpp-llm-lab 2>/dev/null; true
