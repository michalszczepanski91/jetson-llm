# Local patches to third-party sources

This lab clones and builds `NVIDIA/TensorRT-Edge-LLM` from source on-device
(`docs/thor-framework-comparison.md`, `docs/TODO.md` Phase 6a) rather than
vendoring it. Patches here are NOT upstreamed - they exist to make v0.10.1 build
checkpoints this lab's own campaign needs, applied by hand to the local clone at
`~/dev/TensorRT-Edge-LLM` (or wherever `EDGELLM_SERVE_BIN` points).

## edgellm-int4-bias-recipe.patch

**What it fixes**: `tensorrt-edgellm-serve` fails to build ANY int4-quantized
checkpoint (`int4_awq`, `int4_awq_modelopt`, `int4_gptq`) for an architecture
whose linear layers carry a bias - which includes Qwen2's `q_proj`/`k_proj`/
`v_proj` - with:

```
ValueError: external FP16 bias has no checkpoint recipe
```

**Root cause, confirmed 2026-09-09**: `experimental/builder/core/weights.py`'s
`linear_metadata()` computes `bias_recipe` correctly for every quant type (it's
shared code, evaluated before the quant-type branch), and the `QUANT_FP16`
branch passes it into the `LinearWeights` it returns. The three int4 branches
share one return statement further down the same function, and that return
statement simply **omits** `bias_recipe=bias_recipe` from its kwargs - `bias`
itself IS passed, correctly, as an externalized `ParameterSpec`, so
`backend.py`'s `_add_bias()` takes the "externalized bias with no recipe"
branch and raises. Not a checkpoint-format or provenance problem - every int4
checkpoint that reaches this return statement hits it, regardless of quant
tool or source.

**Confirmed fixed**: `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` (Qwen's own official
quant) builds, serves, answers a plain completion coherently, and returns a
correctly-structured unforced tool call after this one-line patch. Not yet
re-verified per-format for `int4_awq`/`int4_awq_modelopt` specifically, though
they share the identical return statement and should be expected to behave the
same way.

**Cleared of silently corrupting engines, 2026-09-10.** `Qwen3-8B-AWQ` builds
and serves under Edge-LLM and then emits degenerate output (`"0000000..."`), and
because this patch was applied when that engine was built, it was recorded as a
live suspect — an unexplained silent-corruption path in a patch the recommended
`7b-gptq-edgellm-thor` row depends on. It is not the cause, and the argument is
static rather than empirical, which makes it stronger than the rebuild test
originally proposed:

1. `bias_recipe` is initialised to `None` and only assigned inside
   `if self.has(bias_name):` (`weights.py`, `linear_metadata()`).
2. `LinearWeights.bias_recipe` already defaults to `None`, so the patched line
   passing `bias_recipe=None` is **identical in effect** to the unpatched code
   omitting the kwarg.
3. `Qwen3-8B-AWQ`'s checkpoint contains **zero bias tensors of any kind**
   (`model.safetensors.index.json` weight map: 0 entries ending in `.bias`) —
   Qwen3 dropped the attention QKV biases Qwen2 carried.

So for this checkpoint the patched line is provably a no-op, and reverting it
could not change the engine. Independently, the *checkpoint* is fine: the same
`Qwen/Qwen3-8B-AWQ` on vLLM/Thor answers "capital of Poland" coherently and
returns a correct unforced `get_weather` tool call (`qwen3-8b-awq-vllm-thor`,
smoke-tested 2026-09-10). Both halves of the original two-part test therefore
land the same way: the fault is somewhere else in Edge-LLM's AWQ path, and this
patch is not implicated in it.

**Apply it** (from `~/dev/TensorRT-Edge-LLM`, pure Python - no rebuild needed):

```bash
git apply /home/michal/dev/jetson-llm/patches/edgellm-int4-bias-recipe.patch
```

**Revisit**: check whether a newer Edge-LLM release fixes this upstream before
re-applying on an update; drop this patch once it does.
