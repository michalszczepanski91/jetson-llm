#!/usr/bin/env python3
"""Chat-template CROSSFEED - why do three backends serving the SAME weights
differ by 40 points on BFCL `irrelevance`?

**Sibling to `scripts/chat_template_probe.py`, not a replacement for it.** That
script (written on the Orin arm) renders one prompt by hand and POSTs it to
every backend, which answers "do the runtimes agree given identical bytes?".
This one asks the question that has to be settled first: *what bytes does each
backend actually send?* It captures each backend's own rendering from the live
server, diffs them, and only then crossfeeds. The two differ in a way that
matters - `chat_template_probe.py` hand-renders `case["function"][0]` verbatim,
so its prompt carries BFCL's non-standard JSON-Schema type names (`dict`,
`float`), while every real chat path in this lab renames them first
(`_bfcl_function_to_openai_tool`, and llama.cpp's grammar builder crashes
without it). Its prompt is therefore identical across backends but not equal to
what any backend really sends.

`docs/thor-framework-comparison.md` reports Edge-LLM 94%, llama.cpp 62%, vLLM
54% on Qwen2.5-7B-Instruct FP16, same 100 cases, same weights. Two candidate
explanations have already been eliminated: decoding (`top_k=1` greedy
reproduced every score exactly) and Edge-LLM's tool-call parser (`auto` and
`hermes` gave 100/100 identical verdicts). The leading remaining candidate is
**prompt rendering** - the three backends each build the prompt string from
`messages` + `tools` with their own chat-template code, and nobody has ever
compared the strings.

This script separates rendering from runtime by making them independent
variables:

    stage 1 (render)     capture the EXACT prompt string each backend builds
                         for the same BFCL case
    stage 2 (crossfeed)  feed every captured rendering to every backend that
                         has /v1/completions, bypassing its own template

If abstention tracks the *rendering*, the cause is prompt construction and it
transfers - a vLLM at JetPack 6.2 (i.e. the Orin, which cannot run Edge-LLM at
all) could be handed Edge-LLM's rendering and get most of the win with no
reflash. If it tracks the *runtime*, rendering is exonerated and the cause is
further down.

**Edge-LLM has no `/v1/completions`** - `experimental/server/api/routes.py`
exposes only `/v1/chat/completions`, `/v1/messages`, and audio routes. So
Edge-LLM is a rendering *source* and a chat-endpoint reference point, but it
cannot be a crossfeed *target*. That asymmetry is a property of the backend,
not a shortcut taken here, and it is why the crossfeed matrix is 2 targets x 3
renderings rather than 3 x 3.

How each rendering is captured:

  vllm       POST /tokenize with {messages, tools, add_generation_prompt} and
             `return_token_strs`, then join the token strings. vLLM applies the
             same template here as on the chat route.
  llamacpp   POST /apply-template, which returns llama-server's rendered
             prompt directly.
  edgellm    in-process, through Edge-LLM's OWN
             `experimental.server.parsing.tool_chat_template.ToolChatTemplateFormatter`
             - the exact class `runtime/engine.py` calls - so this is not a
             reimplementation. Requires running under Edge-LLM's venv:
                 ~/dev/TensorRT-Edge-LLM/.venv/bin/python
             `--verify-against` re-checks the reconstruction by comparing its
             token count to the live server's own `usage.prompt_tokens`.

A rendering is only comparable across backends if all three TOKENIZE it the
same way - `<|im_start|>` has to become one special token, not literal text.
`--stage verify-tokenization` checks exactly that before any crossfeed number
is believed; a mismatch there means the crossfeed would be measuring a
tokenization difference, not a rendering one.

Sampling is pinned to the same greedy setting as the decoding control that
already ran (`scripts/greedy_framework_control.sh`): temperature=0, top_k=1,
top_p=1.0. Pinning temperature alone is NOT controlled sampling here - the
three backends apply three different top_p/top_k/min_p defaults.

Scoring is deliberately crude and identical for all cells: did the raw
completion contain a tool call at all? On `irrelevance` cases the correct
answer is no. Raw text is what /v1/completions returns, so there is no
tool-call parser in the loop - which is the point, since the parser has
already been controlled for separately.

Usage:
    # 1. rendering from each backend (server for that backend must be up)
    uv run python scripts/chat_template_probe.py --stage render --backend vllm \
        --base-url http://127.0.0.1:8001 --model Qwen/Qwen2.5-7B-Instruct
    ~/dev/TensorRT-Edge-LLM/.venv/bin/python scripts/chat_template_probe.py \
        --stage render --backend edgellm --checkpoint <hf snapshot dir> \
        --verify-against http://127.0.0.1:8002 --model Qwen/Qwen2.5-7B-Instruct

    # 2. tokenization parity gate
    uv run python scripts/chat_template_probe.py --stage verify-tokenization \
        --base-url http://127.0.0.1:8001 --backend vllm --model ... \
        --renderings output/chat_template_probe/rendering_*.json

    # 3. crossfeed (target must have /v1/completions: vllm or llamacpp)
    uv run python scripts/chat_template_probe.py --stage crossfeed --backend vllm \
        --base-url http://127.0.0.1:8001 --model Qwen/Qwen2.5-7B-Instruct \
        --renderings output/chat_template_probe/rendering_*.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

OUT_DIR = REPO / "output" / "chat_template_probe"
BFCL_DIR = Path("/opt/datasets/BFCL")

# Same renames scripts/validate_tool_calling.py applies - BFCL's type names
# ("dict"/"float"/"tuple") are not valid JSON Schema and llama.cpp's grammar
# builder crashes on them. Duplicated rather than imported because that module
# imports src/llm_client at module scope, which this script does not need and
# which would drag the whole coordinator stack into Edge-LLM's venv.
_BFCL_TYPE_RENAMES = {"dict": "object", "float": "number", "tuple": "array"}


def _bfcl_function_to_openai_tool(func: dict[str, Any]) -> dict[str, Any]:
    def _convert(node):
        if isinstance(node, dict):
            converted = {k: _convert(v) for k, v in node.items()}
            node_type = converted.get("type")
            if node_type in _BFCL_TYPE_RENAMES:
                converted["type"] = _BFCL_TYPE_RENAMES[node_type]
            elif node_type == "any":
                del converted["type"]
            return converted
        if isinstance(node, list):
            return [_convert(v) for v in node]
        return node

    return {"type": "function", "function": _convert(func)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing BFCL data: {path} (see validate_tool_calling.py's staging note)")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_cases(category: str, limit: int) -> list[dict[str, Any]]:
    cases = _load_jsonl(BFCL_DIR / f"BFCL_v3_{category}.json")[:limit]
    out = []
    for case in cases:
        messages = [m for turn in case["question"] for m in turn]
        tool = _bfcl_function_to_openai_tool(case["function"][0])
        out.append({"id": case["id"], "category": category, "messages": messages, "tool": tool})
    return out


def _post(url: str, body: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ── stage: render ────────────────────────────────────────────────────────────

def _byte_decoder() -> dict[str, int]:
    """Reverse of GPT-2's bytes_to_unicode(), which is the alphabet Qwen2.5's
    byte-level BPE reports token strings in. Naively substituting the two
    visible cases ('\u0120' for space, '\u010a' for newline) silently mangles
    every non-ASCII byte - it produced three bogus "rendering differences" on
    cases carrying '\u2019', '\u03c0' and '\u2212' before this was written.
    Decode the whole byte string at once, not per token: one UTF-8 codepoint
    can be split across two BPE tokens."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _byte_decoder()


def render_vllm(base_url: str, model: str, case: dict[str, Any]) -> str:
    """vLLM's /tokenize applies the same chat template as /v1/chat/completions.
    `return_token_strs` gives the pieces back in byte-level-BPE alphabet;
    mapping them back through _byte_decoder() reconstructs the rendered prompt
    exactly."""
    body = {
        "model": model,
        "messages": case["messages"],
        "tools": [case["tool"]],
        "add_generation_prompt": True,
        "return_token_strs": True,
    }
    data = _post(f"{base_url}/tokenize", body)
    strs = data.get("token_strs")
    if not strs:
        raise SystemExit(
            "vLLM /tokenize returned no token_strs - this vLLM is too old for "
            "the reconstruction path; capture the rendering another way rather "
            "than guessing at it."
        )
    joined = "".join(strs)
    missing = {ch for ch in joined if ch not in _BYTE_DECODER}
    if missing:
        raise SystemExit(
            f"token strings contain characters outside the byte-level BPE alphabet: {sorted(missing)!r} "
            "- the reconstruction would be lossy, so it is refused rather than reported"
        )
    return bytes(_BYTE_DECODER[ch] for ch in joined).decode("utf-8")


def render_llamacpp(base_url: str, model: str, case: dict[str, Any]) -> str:
    """llama-server's /apply-template hands back its rendered prompt directly."""
    body = {"model": model, "messages": case["messages"], "tools": [case["tool"]]}
    data = _post(f"{base_url}/apply-template", body)
    prompt = data.get("prompt")
    if not isinstance(prompt, str):
        raise SystemExit(f"llama.cpp /apply-template returned no prompt: {data!r}")
    return prompt


_EDGELLM_FORMATTER = None


def render_edgellm(checkpoint: str, case: dict[str, Any]) -> str:
    """Uses Edge-LLM's own ToolChatTemplateFormatter - the class its runtime/engine.py
    calls on every tools-bearing request - so this is the server's rendering,
    not a guess at it. tool_choice/parallel_tool_calls are passed the way
    engine.py passes them for tool_choice='auto'."""
    global _EDGELLM_FORMATTER
    if _EDGELLM_FORMATTER is None:
        tree = Path.home() / "dev" / "TensorRT-Edge-LLM"
        sys.path.insert(0, str(tree))
        try:
            from experimental.server.parsing.tool_chat_template import ToolChatTemplateFormatter
        except ImportError as exc:
            raise SystemExit(
                f"cannot import Edge-LLM's ToolChatTemplateFormatter ({exc}) - run this stage "
                f"under {tree}/.venv/bin/python, not the repo venv"
            ) from exc
        _EDGELLM_FORMATTER = ToolChatTemplateFormatter([checkpoint])
    return _EDGELLM_FORMATTER.format(
        case["messages"],
        tools=[case["tool"]],
        tool_choice="auto",
        parallel_tool_calls=True,
        add_generation_prompt=True,
    )


def stage_render(args) -> None:
    cases = load_cases(args.category, args.limit)
    renderings = []
    for case in cases:
        if args.backend == "vllm":
            prompt = render_vllm(args.base_url, args.model, case)
        elif args.backend == "llamacpp":
            prompt = render_llamacpp(args.base_url, args.model, case)
        elif args.backend == "edgellm":
            prompt = render_edgellm(args.checkpoint, case)
        else:
            raise SystemExit(f"unknown backend {args.backend}")
        renderings.append({"id": case["id"], "category": case["category"], "prompt": prompt})

    verified = None
    if args.verify_against:
        verified = _verify_render_against_server(args, cases, renderings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"rendering_{args.backend}_{args.tag}.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists - pass --overwrite deliberately, don't silently replace a rendering")
    path.write_text(json.dumps(
        {"backend": args.backend, "model": args.model, "category": args.category,
         "n": len(renderings), "verification": verified, "renderings": renderings},
        indent=2,
    ))
    print(f"wrote {path} ({len(renderings)} renderings)")
    print("--- first rendering ---")
    print(renderings[0]["prompt"])


def _verify_render_against_server(args, cases, renderings) -> dict[str, Any]:
    """A locally reconstructed rendering is a claim about what the server does.
    Check it: send the same case to the live chat endpoint and compare the
    server's own usage.prompt_tokens against the reconstruction's token count.
    Equal counts do not prove the strings are identical, but a mismatch proves
    they are not - and that is the failure this guards against."""
    case, rendered = cases[0], renderings[0]["prompt"]
    body = {
        "model": args.model, "messages": case["messages"], "tools": [case["tool"]],
        "tool_choice": "auto", "max_tokens": 1, "temperature": 0,
    }
    data = _post(f"{args.verify_against}/v1/chat/completions", body)
    server_tokens = data.get("usage", {}).get("prompt_tokens")

    local_tokens = None
    if _EDGELLM_FORMATTER is not None:
        local_tokens = _EDGELLM_FORMATTER.count_tokens(rendered)

    ok = server_tokens is not None and local_tokens is not None and server_tokens == local_tokens
    print(f"verification: server prompt_tokens={server_tokens} local={local_tokens} match={ok}")
    return {"case_id": case["id"], "server_prompt_tokens": server_tokens,
            "local_prompt_tokens": local_tokens, "match": ok}


# ── stage: verify-tokenization ───────────────────────────────────────────────

def stage_verify_tokenization(args) -> None:
    """Gate before any crossfeed number is trusted. Feeds each rendering to the
    target's own tokenizer and reports the token count. Compared across targets,
    equal counts for the same string mean every backend parses `<|im_start|>` as
    one special token; unequal counts mean a crossfeed would be measuring
    tokenization, not rendering."""
    results = {}
    for rpath in args.renderings:
        doc = json.loads(Path(rpath).read_text())
        prompt = doc["renderings"][0]["prompt"]
        if args.backend == "vllm":
            data = _post(f"{args.base_url}/tokenize", {"model": args.model, "prompt": prompt})
            count = data.get("count", len(data.get("tokens", [])))
        else:
            data = _post(f"{args.base_url}/tokenize", {"content": prompt})
            count = len(data.get("tokens", []))
        results[doc["backend"]] = count
        print(f"  rendering={doc['backend']:9s} -> {args.backend} tokenizes to {count} tokens")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"tokenization_{args.backend}_{args.tag}.json"
    path.write_text(json.dumps({"tokenizer_backend": args.backend, "counts": results}, indent=2))
    print(f"wrote {path}")


# ── stage: crossfeed ─────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>|\"name\"\s*:\s*\"|<function", re.IGNORECASE)


def _emitted_tool_call(text: str) -> bool:
    """Qwen2.5 emits tool calls as `<tool_call>{...}</tool_call>`. The looser
    alternatives catch a backend that starts the JSON without the tag, which
    would otherwise be scored as an abstention it is not."""
    return bool(_TOOL_CALL_RE.search(text))


def stage_crossfeed(args) -> None:
    if args.backend not in ("vllm", "llamacpp"):
        raise SystemExit(
            f"{args.backend} cannot be a crossfeed target - only vllm and llama.cpp expose "
            "/v1/completions (Edge-LLM's routes.py has no such route; see this file's docstring)"
        )
    all_results = {}
    for rpath in args.renderings:
        doc = json.loads(Path(rpath).read_text())
        source = doc["backend"]
        outcomes = []
        for item in doc["renderings"]:
            body = {
                "model": args.model, "prompt": item["prompt"], "max_tokens": args.max_tokens,
                "temperature": 0, "top_p": 1.0, "top_k": 1,
                # /v1/completions applies no chat template and therefore knows
                # nothing about where a turn ends. Without this the model runs
                # to max_tokens, sails past <|im_end|> and starts hallucinating
                # the NEXT turn - which can contain a tool call that the real
                # chat path would never have returned. Left unset, the first
                # pass had llama.cpp emitting a full 64 tokens on every case.
                "stop": ["<|im_end|>"],
            }
            try:
                data = _post(f"{args.base_url}/v1/completions", body)
                text = data["choices"][0].get("text", "")
                finish = data["choices"][0].get("finish_reason")
                err = None
            except Exception as exc:  # noqa: BLE001 - a per-case failure is a result
                text, finish, err = "", None, f"{type(exc).__name__}: {exc}"
            outcomes.append({
                "id": item["id"], "tool_call": _emitted_tool_call(text) if err is None else None,
                "finish_reason": finish, "error": err, "text": text,
            })
        scored = [o for o in outcomes if o["error"] is None]
        # Same truncation gate as stage_chat_reference - see its comment.
        truncated = [o for o in scored if o["finish_reason"] == "length" and not o["tool_call"]]
        abstained = sum(1 for o in scored if not o["tool_call"])
        rate = abstained / len(scored) if scored else None
        all_results[source] = {
            "n_scored": len(scored), "n_error": len(outcomes) - len(scored),
            "n_truncated_no_tool_call": len(truncated),
            "abstained": abstained, "abstention_rate": rate, "outcomes": outcomes,
        }
        pct = f"{rate:.0%}" if rate is not None else "n/a"
        print(f"  serving={args.backend:9s} rendering={source:9s} -> abstention {abstained}/{len(scored)} = {pct} "
              f"({len(truncated)} truncated)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"crossfeed_{args.backend}_{args.tag}.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists - pass --overwrite deliberately")
    path.write_text(json.dumps(
        {"serving_backend": args.backend, "model": args.model, "category": args.category,
         "max_tokens": args.max_tokens, "stop": ["<|im_end|>"],
         "sampling": {"temperature": 0, "top_p": 1.0, "top_k": 1},
         "by_rendering": all_results}, indent=2))
    print(f"wrote {path}")


# ── stage: chat-reference ────────────────────────────────────────────────────

def stage_chat_reference(args) -> None:
    """The matched baseline. The campaign's 94/62/54 were measured over 100
    BFCL cases through each backend's own /v1/chat/completions with a `tools`
    parameter; the crossfeed runs 50 cases through /v1/completions with the
    prompt pre-rendered. Comparing those two directly would confound the
    change of interest with a change of case count and response path, so this
    re-measures the chat path on exactly the crossfeed's 50 cases.

    Abstention here means the server returned no `tool_calls` - which, unlike
    the crossfeed, does run the backend's tool-call parser. That difference is
    the point: a backend whose raw text carries a tool call but whose parsed
    response does not (or vice versa) is localized by comparing these two
    numbers."""
    cases = load_cases(args.category, args.limit)
    outcomes = []
    for case in cases:
        body = {
            "model": args.model, "messages": case["messages"], "tools": [case["tool"]],
            "tool_choice": "auto", "max_tokens": args.max_tokens,
            "temperature": 0, "top_p": 1.0, "top_k": 1,
        }
        try:
            data = _post(f"{args.base_url}/v1/chat/completions", body)
            choice = data["choices"][0]
            msg = choice["message"]
            outcomes.append({"id": case["id"], "tool_call": bool(msg.get("tool_calls")),
                             "finish_reason": choice.get("finish_reason"),
                             "content": msg.get("content"), "error": None})
        except Exception as exc:  # noqa: BLE001 - a per-case failure is a result
            outcomes.append({"id": case["id"], "tool_call": None, "content": None,
                             "error": f"{type(exc).__name__}: {exc}"})
    scored = [o for o in outcomes if o["error"] is None]
    # A case that hit the token budget without emitting a tool call has not
    # abstained - it was cut off, and the tool call may have been the next
    # token. Counting those as abstentions is exactly the error that produced a
    # bogus 92% here at max_tokens=64. Reported separately rather than folded in.
    truncated = [o for o in scored if o["finish_reason"] == "length" and not o["tool_call"]]
    abstained = sum(1 for o in scored if not o["tool_call"])
    rate = abstained / len(scored) if scored else None
    pct = f"{rate:.0%}" if rate is not None else "n/a"
    print(f"  chat reference {args.backend:9s} -> abstention {abstained}/{len(scored)} = {pct} "
          f"({len(truncated)} of those were TRUNCATED, i.e. inconclusive)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"chatref_{args.backend}_{args.tag}.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists - pass --overwrite deliberately")
    path.write_text(json.dumps(
        {"backend": args.backend, "model": args.model, "category": args.category,
         "path": "/v1/chat/completions with tools", "max_tokens": args.max_tokens,
         "sampling": {"temperature": 0, "top_p": 1.0, "top_k": 1},
         "n_scored": len(scored), "n_error": len(outcomes) - len(scored),
         "n_truncated_no_tool_call": len(truncated),
         "abstained": abstained, "abstention_rate": rate, "outcomes": outcomes}, indent=2))
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["render", "verify-tokenization", "crossfeed", "chat-reference"])
    p.add_argument("--backend", required=True, choices=["vllm", "llamacpp", "edgellm"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--checkpoint", default=None, help="local HF snapshot dir (edgellm render only)")
    p.add_argument("--verify-against", default=None, help="live server base-url to check a reconstruction against")
    p.add_argument("--category", default="irrelevance", choices=["irrelevance", "simple"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=200, help="MUST match "
                   "scripts/validate_tool_calling.py's own default (200), or these numbers are not "
                   "comparable to the campaign's. Qwen2.5 on this prompt writes prose FIRST and emits "
                   "the tool call after it, so a short budget truncates the call and scores a "
                   "hallucinated tool as an abstention: at 64, vLLM measured 92%% irrelevance against "
                   "the campaign's 54%%, purely from finish_reason=length on every case.")
    p.add_argument("--renderings", nargs="+", default=[])
    p.add_argument("--tag", default="7b")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.stage == "render":
        if args.backend == "edgellm" and not args.checkpoint:
            raise SystemExit("--checkpoint is required for --backend edgellm")
        if args.backend != "edgellm" and not args.base_url:
            raise SystemExit("--base-url is required for a live-server rendering capture")
        stage_render(args)
    elif args.stage == "verify-tokenization":
        stage_verify_tokenization(args)
    elif args.stage == "chat-reference":
        stage_chat_reference(args)
    else:
        stage_crossfeed(args)


if __name__ == "__main__":
    main()
