"""Does the backend gap survive when the chat template is taken out of it?

`docs/TODO.md`'s "Not tested yet" has called this the cheap decisive test for
this lab's headline finding - the 40-point BFCL `irrelevance` spread between
Edge-LLM, llama.cpp and vLLM - and it kept not being run.

**The question.** Every score in this lab so far went through
`/v1/chat/completions`, so each backend rendered its own chat template and ran
its own tool-call parser before anything was compared. Two of the three
candidate explanations for the gap have already been excluded (decoding, in the
`top_k=1` control; parsing, in the `auto` vs `hermes` parity run), which leaves
template rendering. This bypasses it: one prompt string is rendered **here**,
once, and POSTed verbatim to `/v1/completions` on every backend, with no `tools`
parameter for a server to re-render and no parser between the model and the
comparison.

**Reading the result.** If the backends emit identical text, the runtimes are
exonerated and the gap has to come from what each server does *around* the
model - rendering or parsing. If they differ, the difference is in the runtime
or the weights, and the template hypothesis does not survive on its own.

**The Orin confound, stated not fixed.** The Thor comparison held precision at
fp16 across all three backends. This board cannot: its vLLM row is AWQ int4 and
its llama.cpp row is GGUF Q4_K_M, so a difference here is
backend-or-quantization, not backend alone. That is not a footnote - the
precision arm of 2026-09-10 measured precision alone moving BFCL `irrelevance`
by 30 points on this board (36.7% -> 66.7%, AWQ vs fp16, everything else held),
which is most of the gap this test is trying to explain. So a *difference*
here is weak evidence, while *identity* here is strong evidence, and the test
is worth running for the second case.

Sampling is pinned explicitly, including `top_p` and `top_k`: the backends ship
different defaults, so pinning temperature alone is not controlled sampling
(docs/thor-framework-comparison.md).

Not a benchmark - it writes no `benchmark_result` and measures no latency.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from model_config import load_model_config  # noqa: E402
from llm_coordinator import build_coordinator  # noqa: E402
from manifest import assert_condition_matches_reality  # noqa: E402

_META_KEYS = {"model", "backend", "platform", "precision", "notes", "extra_args"}

_TOOL_PREAMBLE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
)
_TOOL_POSTAMBLE = (
    "\n</tools>\n\nFor each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def render_prompt(case: dict) -> str:
    """Qwen2.5's ChatML tool-calling template, rendered once, here.

    Deliberately hand-rendered rather than fetched from either server: the whole
    point is that both backends receive the same bytes, which cannot be
    guaranteed by asking each of them to render it."""
    fn = json.dumps({"type": "function", "function": case["function"][0]}, ensure_ascii=False)
    user = case["question"][0][0]["content"]
    return (
        f"<|im_start|>system\n{_TOOL_PREAMBLE}{fn}{_TOOL_POSTAMBLE}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def complete(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 1234,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)["choices"][0]["text"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", required=True)
    p.add_argument("--data-dir", default="/opt/datasets/BFCL")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--out", required=True, help="where to write this leg's raw outputs")
    p.add_argument("--ready-timeout", type=float, default=900.0)
    p.add_argument("--target", default="local")
    p.add_argument("--base-url", default=None)
    args = p.parse_args()

    if args.target == "local":
        assert_condition_matches_reality(
            "standalone",
            own_containers={"vllm-llm-lab", "llamacpp-llm-lab"},
            declared_co_resident=[],
        )

    cases = [json.loads(l) for l in (Path(args.data_dir) / "BFCL_v3_irrelevance.json").read_text().splitlines() if l.strip()]
    cases = cases[: args.limit]

    variant = load_model_config(args.model_config)
    local_kwargs = {k: v for k, v in variant.items() if k not in _META_KEYS}
    local_kwargs["extra_args"] = variant.get("extra_args")
    coordinator = build_coordinator(args, variant, ready_timeout=args.ready_timeout, **local_kwargs)
    coordinator.start()
    try:
        if not coordinator.wait_ready():
            raise TimeoutError("server did not become ready")
        rows = []
        for i, case in enumerate(cases):
            prompt = render_prompt(case)
            text = complete(coordinator.base_url, coordinator.model, prompt, args.max_tokens)
            rows.append({
                "id": case["id"],
                "prompt_sha": __import__("hashlib").sha256(prompt.encode()).hexdigest()[:16],
                "text": text,
                "emits_tool_call": "<tool_call>" in text,
            })
            print(f"[{i+1}/{len(cases)}] {case['id']}: tool_call={'YES' if '<tool_call>' in text else 'no'}")
    finally:
        coordinator.stop()

    Path(args.out).write_text(json.dumps({
        "model_config_key": args.model_config,
        "model": variant["model"], "backend": variant["backend"],
        "precision": variant["precision"],
        "endpoint": "/v1/completions (no tools param, template rendered client-side)",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 1234},
        "n": len(rows),
        "n_emitting_tool_call": sum(r["emits_tool_call"] for r in rows),
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {args.out}: {sum(r['emits_tool_call'] for r in rows)}/{len(rows)} emitted a tool call")


if __name__ == "__main__":
    main()
