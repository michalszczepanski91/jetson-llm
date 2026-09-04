"""Blocking HTTP client for an OpenAI-compatible chat-completions server
(vLLM or llama.cpp's llama-server, via llm_coordinator.py) serving the
orchestrator-LLM candidate under test. Text-only - the counterpart to
jetson-vlm-lab's vlm_client.py, minus JPEG/image handling (there is no image
in this pipeline's orchestrator turn, only a transcript + scene-state text
already rendered into the prompt - see embedded-ai-chain's
LlmOrchestrator._grounding_context()).

Returns the full assistant message (not just its text), unlike
jetson-vlm-lab's call_vlm(), because scripts/validate_tool_calling.py needs
to inspect message["tool_calls"], not just message["content"] -
scripts/benchmark.py (which only wants the text) simply reads
message["content"] itself.

Pure request/response, no lifecycle management (see llm_coordinator.py) and
no queue/threading concerns - kept dependency-free (stdlib only) so this
repo stays independently cloneable/runnable, same reasoning as
jetson-vlm-lab's vlm_client.py docstring.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_coordinator import LlamaCppCoordinator, RemoteCoordinator, VllmCoordinator

    Coordinator = VllmCoordinator | LlamaCppCoordinator | RemoteCoordinator


def call_llm(
    coordinator: "Coordinator",
    messages: list[dict[str, Any]],
    max_tokens: int = 128,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = "auto",
    temperature: float | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """One blocking chat-completion round trip against coordinator's
    already-running server. Returns the assistant message dict as-is
    (content, possibly tool_calls) - raises on a non-2xx response or
    malformed body, same "caller decides how to handle/retry" contract as
    vlm_client.py's call_vlm().

    `temperature=None` omits the field entirely (server default - vLLM
    ~0.7, llama.cpp ~0.8) - deliberately not defaulted to a fixed value
    here, since scripts/benchmark.py/benchmark_streaming.py want the
    server's own default (that's what a real deployment would see) while
    scripts/validate_tool_calling.py deliberately pins a low value on every
    call (see that script for why: a real, measured finding, not a
    guess - see docs/TODO.md Phase 1's temperature diagnostic)."""
    payload: dict[str, Any] = {
        "model": coordinator.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    req = urllib.request.Request(
        f"{coordinator.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]


def call_llm_full(
    coordinator: "Coordinator",
    messages: list[dict[str, Any]],
    max_tokens: int = 128,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = "auto",
    temperature: float | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Same round trip as call_llm(), returning the WHOLE response body
    instead of just the assistant message - specifically for its `usage`
    block (`prompt_tokens` / `completion_tokens`).

    Added as a second function rather than by changing call_llm()'s return
    type, because scripts/validate_tool_calling.py and validate_mmlu.py both
    depend on that contract and neither needs token counts.

    Why the server's own usage and not a local estimate: docs/note.md §3
    requires input and output token counts on every result, and the only
    authority on how a prompt tokenised is the thing that tokenised it. A
    client-side approximation would differ per model family and quietly
    corrupt every per-token figure derived from it - including J/token."""
    payload: dict[str, Any] = {
        "model": coordinator.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    req = urllib.request.Request(
        f"{coordinator.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def stream_llm(
    coordinator: "Coordinator",
    messages: list[dict[str, Any]],
    max_tokens: int = 128,
    temperature: float | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """One streaming round trip, returning the per-run timing and token
    record that benchmark_result.schema.json's `runs[]` entries are built
    from: ttft_ms, e2e_latency_ms, decode_ms, inter-token gaps, achieved
    input/output token counts, and finish_reason.

    Two things this gets right that the older inline version in
    scripts/benchmark_streaming.py did not:

    1. **Token counts are token counts.** That version counted SSE content
       deltas, which is a *chunk* count - a backend is free to coalesce
       several tokens into one delta or split one across two, and the two
       backends here do not agree on it. This requests
       `stream_options={"include_usage": true}`, which makes vLLM emit a
       final usage-bearing chunk, and falls back to the delta count only
       when the server sends no usage at all - recording which of the two
       happened in `token_source`, so a result can never silently mix them.

    2. **Inter-token latency survives.** The gaps between successive content
       deltas are kept as a distribution (docs/note.md §9), not averaged
       away: a mean hides exactly the stall a listener actually perceives.
       They are still *delta* gaps, not true per-token gaps, for the same
       coalescing reason - named `inter_delta` internally and reported under
       the schema's inter_token field with that caveat carried in
       token_source.

    All intervals come from time.perf_counter(); this device's NTP sync is
    known-broken, so nothing here may depend on wall-clock time."""
    payload: dict[str, Any] = {
        "model": coordinator.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    req = urllib.request.Request(
        f"{coordinator.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_token_at: float | None = None
    last_delta_at: float | None = None
    inter_delta_ms: list[float] = []
    n_deltas = 0
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            # The usage-bearing final chunk carries an empty choices list on
            # vLLM, so usage must be read before indexing into choices.
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                else:
                    inter_delta_ms.append((now - last_delta_at) * 1000)
                last_delta_at = now
                n_deltas += 1
    t_end = time.perf_counter()

    if usage and usage.get("completion_tokens") is not None:
        output_tokens = usage["completion_tokens"]
        input_tokens = usage.get("prompt_tokens")
        token_source = "server usage (stream_options.include_usage)"
    else:
        output_tokens = n_deltas
        input_tokens = None
        token_source = (
            "sse delta count - server sent no usage block, so this is a CHUNK count, "
            "not a token count; per-token figures derived from it are approximate"
        )

    ttft_ms = (first_token_at - t0) * 1000 if first_token_at else None
    e2e_ms = (t_end - t0) * 1000
    decode_ms = (t_end - first_token_at) * 1000 if first_token_at else None

    return {
        "ttft_ms": ttft_ms,
        "e2e_latency_ms": e2e_ms,
        "decode_ms": decode_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prefill_tok_s": (input_tokens / (ttft_ms / 1000)) if (input_tokens and ttft_ms) else None,
        "decode_tok_s": (output_tokens / (decode_ms / 1000)) if (output_tokens and decode_ms) else None,
        "inter_token_latency_ms": inter_delta_ms,
        "finish_reason": finish_reason,
        "error": None,
        "token_source": token_source,
    }
