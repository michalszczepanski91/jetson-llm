"""Unit tests for llm_coordinator's container lifecycle classes.

Mocks subprocess.Popen (docker run) and urllib.request.urlopen (the /health
poll) - no real Docker/GPU needed. Mirrors jetson-vlm-lab's
test_vlm_coordinator.py convention of mocking the process/network-touching
pieces specifically.
"""

import argparse
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_coordinator import (  # noqa: E402
    LlamaCppCoordinator,
    RemoteCoordinator,
    VllmCoordinator,
    add_target_args,
    build_coordinator,
)


def _fake_proc(returncode=None):
    proc = MagicMock()
    proc.poll.return_value = returncode
    proc.wait.return_value = 0
    return proc


def _healthy_resp():
    resp = MagicMock()
    resp.status = 200
    resp.__enter__.return_value = resp
    return resp


# ── VllmCoordinator ──────────────────────────────────────────────────────

def test_vllm_wait_ready_true_once_health_responds():
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc), \
         patch("llm_coordinator.urllib.request.urlopen", return_value=_healthy_resp()), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = VllmCoordinator(model="fake/model", ready_timeout=3.0)
        coordinator.start()
        assert coordinator.wait_ready(timeout=3.0) is True
        coordinator.stop()


def test_vllm_wait_ready_false_on_timeout():
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc), \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = VllmCoordinator(model="fake/model", ready_timeout=0.2)
        coordinator.start()
        assert coordinator.wait_ready() is False
        coordinator.stop()


def test_vllm_starting_twice_raises():
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc), \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = VllmCoordinator(model="fake/model")
        coordinator.start()
        try:
            coordinator.start()
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
        finally:
            coordinator.stop()


def test_vllm_base_url_and_model_properties():
    coordinator = VllmCoordinator(model="fake/model", port=9999)
    assert coordinator.base_url == "http://127.0.0.1:9999"
    assert coordinator.model == "fake/model"


def test_vllm_ignores_llamacpp_only_kwargs():
    # build_coordinator passes a unified kwargs dict regardless of backend;
    # VllmCoordinator must accept and drop ones meant for llama-cpp rows.
    coordinator = VllmCoordinator(model="fake/model", quant="Q4_K_M", n_gpu_layers=-1, ctx_size=4096)
    assert coordinator.model == "fake/model"


def test_vllm_command_includes_gpu_memory_utilization_and_model():
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = VllmCoordinator(model="Qwen/Qwen2.5-1.5B-Instruct-AWQ", gpu_memory_utilization=0.15)
        coordinator.start()
        coordinator.stop()

    cmd = mock_popen.call_args[0][0]
    assert "Qwen/Qwen2.5-1.5B-Instruct-AWQ" in cmd
    assert "0.15" in cmd


# ── LlamaCppCoordinator ──────────────────────────────────────────────────

def test_llamacpp_wait_ready_true_once_health_responds():
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc), \
         patch("llm_coordinator.urllib.request.urlopen", return_value=_healthy_resp()), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = LlamaCppCoordinator(model="Qwen/Qwen2.5-1.5B-Instruct-GGUF", ready_timeout=3.0)
        coordinator.start()
        assert coordinator.wait_ready(timeout=3.0) is True
        coordinator.stop()


def test_llamacpp_base_url_defaults_to_8080():
    coordinator = LlamaCppCoordinator(model="fake/model")
    assert coordinator.base_url == "http://127.0.0.1:8080"


def test_llamacpp_command_uses_hf_downloader_and_jinja():
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = LlamaCppCoordinator(model="Qwen/Qwen2.5-1.5B-Instruct-GGUF", quant="Q4_K_M", n_gpu_layers=-1)
        coordinator.start()
        coordinator.stop()

    cmd = mock_popen.call_args[0][0]
    assert "-hf" in cmd
    assert "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M" in cmd
    assert "--jinja" in cmd
    assert "-ngl" in cmd


def test_llamacpp_ignores_vllm_only_kwargs():
    coordinator = LlamaCppCoordinator(
        model="fake/model", gpu_memory_utilization=0.15, max_model_len=4096, enable_prefix_caching=True,
    )
    assert coordinator.model == "fake/model"


def test_llamacpp_container_exit_before_ready_raises():
    proc = _fake_proc(returncode=1)
    with patch("llm_coordinator.subprocess.Popen", return_value=proc), \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = LlamaCppCoordinator(model="fake/model", ready_timeout=3.0)
        coordinator.start()
        try:
            coordinator.wait_ready()
            assert False, "expected RuntimeError to propagate"
        except RuntimeError as exc:
            assert "exited unexpectedly" in str(exc)
        coordinator.stop()


# ── RemoteCoordinator ────────────────────────────────────────────────────

def test_remote_base_url_and_model_properties():
    coordinator = RemoteCoordinator(model="fake/model", host="10.0.0.5", port=9000)
    assert coordinator.base_url == "http://10.0.0.5:9000"
    assert coordinator.model == "fake/model"


def test_remote_start_and_stop_are_noops():
    coordinator = RemoteCoordinator(model="fake/model", host="10.0.0.5")
    coordinator.start()
    coordinator.stop()


def test_remote_wait_ready_true_once_health_responds():
    with patch("llm_coordinator.urllib.request.urlopen", return_value=_healthy_resp()):
        coordinator = RemoteCoordinator(model="fake/model", host="10.0.0.5", ready_timeout=3.0)
        assert coordinator.wait_ready() is True


def test_remote_wait_ready_false_on_timeout():
    with patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        coordinator = RemoteCoordinator(model="fake/model", host="10.0.0.5", ready_timeout=0.2, poll_interval=0.05)
        assert coordinator.wait_ready() is False


def test_remote_ignores_local_only_kwargs_from_either_backend():
    coordinator = RemoteCoordinator(
        model="fake/model", host="10.0.0.5",
        gpu_memory_utilization=0.6, max_model_len=4096, quant="Q4_K_M", n_gpu_layers=-1,
    )
    assert coordinator.model == "fake/model"


# ── build_coordinator dispatch ───────────────────────────────────────────

def test_build_coordinator_dispatches_vllm():
    p = argparse.ArgumentParser()
    add_target_args(p)
    args = p.parse_args([])
    coordinator = build_coordinator(args, {"model": "fake/model", "backend": "vllm"}, ready_timeout=1.0)
    assert isinstance(coordinator, VllmCoordinator)


def test_build_coordinator_dispatches_llama_cpp():
    p = argparse.ArgumentParser()
    add_target_args(p)
    args = p.parse_args([])
    coordinator = build_coordinator(args, {"model": "fake/model", "backend": "llama-cpp"}, ready_timeout=1.0)
    assert isinstance(coordinator, LlamaCppCoordinator)


def test_build_coordinator_unknown_backend_raises():
    p = argparse.ArgumentParser()
    add_target_args(p)
    args = p.parse_args([])
    try:
        build_coordinator(args, {"model": "fake/model", "backend": "tensorrt-llm"}, ready_timeout=1.0)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_build_coordinator_remote_requires_host():
    p = argparse.ArgumentParser()
    add_target_args(p)
    args = p.parse_args(["--target", "remote"])
    try:
        build_coordinator(args, {"model": "fake/model", "backend": "vllm"}, ready_timeout=1.0)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_build_coordinator_remote_ignores_backend_field():
    # --target remote always yields RemoteCoordinator regardless of backend -
    # both real backends speak the same wire format, so there is only one
    # remote target class.
    p = argparse.ArgumentParser()
    add_target_args(p)
    args = p.parse_args(["--target", "remote", "--remote-host", "10.0.0.5", "--remote-port", "9000"])
    coordinator = build_coordinator(args, {"model": "fake/model", "backend": "llama-cpp"}, ready_timeout=1.0)
    assert isinstance(coordinator, RemoteCoordinator)
    assert coordinator.base_url == "http://10.0.0.5:9000"
