"""Unit tests for llm_client.call_llm()'s request/response shape. Mocks
urllib.request.urlopen - no real Docker/GPU needed."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import call_llm  # noqa: E402


def _fake_coordinator():
    coordinator = MagicMock()
    coordinator.model = "fake/model"
    coordinator.base_url = "http://127.0.0.1:8000"
    return coordinator


def _fake_response(message: dict):
    body = {"choices": [{"message": message}]}
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode("utf-8")
    resp.__enter__.return_value = resp
    return resp


def test_call_llm_returns_the_full_message():
    message = {"role": "assistant", "content": "hello", "tool_calls": None}
    with patch("llm_client.urllib.request.urlopen", return_value=_fake_response(message)):
        result = call_llm(_fake_coordinator(), [{"role": "user", "content": "hi"}])

    assert result == message


def test_call_llm_sends_model_and_messages_without_tools_by_default():
    coordinator = _fake_coordinator()
    with patch("llm_client.urllib.request.urlopen", return_value=_fake_response({"content": "ok"})) as mock_urlopen:
        call_llm(coordinator, [{"role": "user", "content": "hi"}], max_tokens=64)

    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://127.0.0.1:8000/v1/chat/completions"
    payload = json.loads(req.data)
    assert payload["model"] == "fake/model"
    assert payload["max_tokens"] == 64
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_call_llm_includes_tools_and_tool_choice_when_given():
    coordinator = _fake_coordinator()
    tools = [{"type": "function", "function": {"name": "read_scene_state"}}]
    with patch("llm_client.urllib.request.urlopen", return_value=_fake_response({"content": "ok"})) as mock_urlopen:
        call_llm(coordinator, [{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto")

    payload = json.loads(mock_urlopen.call_args[0][0].data)
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


def test_call_llm_returns_tool_calls_when_present():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "ask_vlm", "arguments": "{}"}}],
    }
    with patch("llm_client.urllib.request.urlopen", return_value=_fake_response(message)):
        result = call_llm(_fake_coordinator(), [{"role": "user", "content": "what color"}])

    assert result["tool_calls"][0]["function"]["name"] == "ask_vlm"
