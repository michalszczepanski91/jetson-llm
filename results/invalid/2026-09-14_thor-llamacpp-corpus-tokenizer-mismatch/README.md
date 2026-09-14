# 2026-09-14 — Thor llama.cpp p1-context r01: invalidated by a tokenizer contract mismatch

**What is here.** The first complete v2 run of `7b-fp16-llamacpp-thor-v2` — four
cells, 30/30 repetitions each, steady state reached, cache verified off — plus the
corpus file it built (`Qwen--Qwen2.5-7B-Instruct-GGUF.c1.json`). Nothing about the
*execution* went wrong. It is kept because it is the evidence for a defect that
silently produced a run which could not be compared with anything, and because the
corpus file is the defect made visible.

**What went wrong.** `benchmarks/prompts.py:TokenCounter` posted
`{"model": ..., "prompt": text}` to `/tokenize`. That is vLLM's contract.
**llama-server reads `content`** and ignores an unknown `prompt`, so it tokenised the
empty string and answered `{"tokens": []}` — HTTP 200, well-formed, meaning zero.

The counter accepted that zero, because zero is an `int` and the code only checked
the shape of the reply. Everything downstream followed correctly from a wrong number:

| | this run | the vLLM / Edge-LLM legs |
|---|---|---|
| corpus sha256 | `1bf63fe1f8fc…` | `8c36a604d0a5…` |
| corpus `reference_model` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | `Qwen/Qwen2.5-7B-Instruct` |
| `reference_tokens` per entry | **0, 0, 0, 0** (`exact: false`, `token_error: −128/−512/−2048/−8192`) | 128 / 512 / 2048 / 8192, all `exact: true` |
| probes to converge | 12 (the `max_iters` cap, i.e. never converged) | 3 |
| achieved prompt tokens | 158 / 437 / 1555 / 6026 | 173 / 555 / 2093 / 8239 |

**Two independent defects, and the second is the one that matters.**

1. *The counter believed a zero.* Bad, but self-contained: the corpus builder's own
   `exact: false` and `token_error: -512` recorded the failure faithfully. Nothing
   read them.
2. *The corpus was keyed by the row's own `model`.* The llama.cpp row names a
   `-GGUF` repo, so it keyed a **second corpus file** and built it from scratch
   instead of loading the one the other two legs send. **The prompts were therefore
   not byte-identical across backends** — which is the single property the whole
   three-backend comparison is constructed to guarantee (`benchmarks/prompts.py`'s
   module docstring explains why: if each backend tokenises to its own exact 2048,
   the bytes differ and the control is undone). Even with a perfect counter, this row
   would have sent different prompts.

So the run measured a real backend at prompt lengths of 158/437/1555/6026 against an
x axis the other legs do not share. Its numbers are internally consistent and
externally meaningless — a context-scaling comparison at "512" would have put 437
tokens beside 555.

**Collateral.** The max-context probe derives words-per-token from the corpus's
`reference_tokens`. With that at 0 (guarded by `max(1, …)`) the ratio became ~6000
words per token, so the probe's *floor* attempt — nominally ~128 tokens — sent
**764,533 tokens** and was refused at once. The run recorded
`max_context_measured: null` where the other legs measured a real limit
(Edge-LLM: 16,161). That null is the only symptom that appeared on the console, and
it is a downstream effect, not the disease.

**The fix**, in four parts (all under `tests/test_prompt_corpus.py`, which runs
against a real local HTTP server rather than a patched `urlopen` — the defect lived
in the request *body*, and a mock would have accepted whatever the code sent):

- `/tokenize` bodies now carry **both** `prompt` and `content`. Each server takes the
  key it knows and ignores the other.
- **A zero count for non-empty text is no longer a measurement.** The counter demotes
  itself to the `usage.prompt_tokens` fallback, which costs a round trip and cannot
  be silently wrong in this direction. A genuinely empty string may still count zero.
- **`build_corpus` raises** when a target misses by more than `max(4, target/4)`.
  An off-by-three prompt honestly labelled is the documented behaviour and still
  passes; a counter returning nothing now kills the run at second 30 instead of at
  minute 40.
- **`prompt_corpus_reference`** (new registry field, schema'd) lets a row name the
  corpus it shares. `7b-fp16-llamacpp-thor-v2` now names
  `Qwen/Qwen2.5-7B-Instruct`, so all three legs load one file.

**Why this is archived rather than deleted.** Same rule as the two 2026-09-11
entries: the evidence is worth more than the disk space. And like those, this is a
run that was *labelled truthfully and measured wrongly* — `execution_condition`,
warm-up and cache probe were all correct and all honest. What no condition guard
could catch is a prompt corpus that is internally consistent and externally
incomparable. The check that catches this class is the one that refuses to build a
corpus it could not converge, and it now runs before any board time is spent.

**Replaced by**: `results/v2/2026-09-14_thor_7b-fp16-llamacpp-thor-v2_p1-context_bs1_r01`,
run the same day after the fix, against the shared corpus `8c36a604d0a5…`. It reuses
the `r01` replicate number because this run left `results/v2/` entirely — the
replicate counter names surviving runs of a config, and a reader who finds both will
find this directory named in the other's stead.
