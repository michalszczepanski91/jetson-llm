# Chat-template probe — Orin leg, 2026-09-10

Two of this lab's three backends, one identical pre-rendered prompt each, via
`/v1/completions` with no `tools` parameter and no server-side template. Produced by
`scripts/chat_template_probe.py`; see that module's docstring for the design and for
why *identity* here would have been strong evidence while *difference* is weak.

Prompt bytes verified identical on all 30 cases (`prompt_sha` matches per case).
Sampling pinned on both: `temperature=0, top_p=1.0, top_k=1, seed=1234`.

| leg | precision | emitted a tool call | correct-abstain | for reference: same row via `/v1/chat/completions` |
|---|---|---:|---:|---:|
| `1.5b-awq-vllm-orin` | AWQ int4 | 20/30 | 33.3% | 36.7% |
| `1.5b-q4-llamacpp-orin` | GGUF Q4_K_M | 12/30 | 60.0% | 70.0% |

Byte-identical completions: **5/30**. Agreement on call-vs-abstain: 22/30, and all 8
disagreements run the same direction — vLLM calls where llama.cpp abstains.

## What this shows

**The gap survives removing the chat template.** Each backend's abstention rate moved
only a little when the template layer was taken out (36.7→33.3, 70.0→60.0) and the
ordering and most of the spread were preserved. On this board the template/parser layer
accounts for little of the difference; it lives in the model-plus-runtime.

## What it does not show

This leg is **confounded by precision** and cannot be read as a backend result. Orin's
vLLM row is AWQ int4 and its llama.cpp row is GGUF Q4_K_M, and the precision arm run the
same day measured precision *alone* moving BFCL `irrelevance` by 30 points on this board
(36.7% → 66.7%, AWQ vs fp16, model/backend/board held fixed). That is most of the spread
seen here, so quantization format is at least as good an explanation as backend runtime.

It also does **not** transfer to Thor's 40-point Edge-LLM finding, which held precision
at fp16 across all three backends and is therefore a different question. What it does do
is weaken "chat-template rendering is the leading candidate" as a general claim, which
makes the Edge-LLM leg on Thor more worth running, not less.

The precision-matched version of this test is the one worth running next: it needs an
fp16 GGUF row on Orin (none exists), or Thor, where all three backends already run fp16.
