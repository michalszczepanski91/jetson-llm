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
    top_p: float | None = None,
    top_k: int | None = None,
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
    guess - see docs/TODO.md Phase 1's temperature diagnostic).

    `top_p`/`top_k` are the same story one level deeper, and were added
    2026-09-09 after they turned out to be a real cross-backend confound:
    pinning only `temperature` still leaves each server applying its OWN
    truncation defaults, and they differ sharply - Edge-LLM's server defaults
    to top_p=0.9/top_k=50 (see its api/protocol.py) while vLLM defaults to
    top_p=1.0/top_k=-1, i.e. no truncation at all. On a tool-call ABSTENTION
    decision that is not cosmetic: if the model puts a few percent on starting
    a tool call for a question that needs none, a 0.9 nucleus can drop it
    where an untruncated sampler will occasionally emit it. Any comparison
    across backends that means to isolate the runtime must pin these too."""
    payload: dict[str, Any] = {
        "model": coordinator.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
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
