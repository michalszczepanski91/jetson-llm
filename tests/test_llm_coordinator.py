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
    EdgeLlmCoordinator,
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


def test_vllm_extra_volumes_are_mounted_before_the_image():
    # The Thor rows mount a patched gpu_worker.py over vLLM's own, because
    # vLLM's memory profiling asserts free memory never INCREASES mid-profile
    # and dies on this unified-memory board even with the box idle. Ordering
    # matters: a -v after the image name would be read as a server argument.
    proc = _fake_proc()
    patch_mount = "/opt/hf-cache/gpu_worker_patched.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py:ro"
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = VllmCoordinator(model="fake/model", extra_volumes=[patch_mount])
        coordinator.start()
        coordinator.stop()

    cmd = mock_popen.call_args[0][0]
    assert patch_mount in cmd
    assert cmd.index(patch_mount) < cmd.index("fake/model")


def test_vllm_command_enables_structured_tool_calls():
    # Regression test: confirmed live 2026-09-04 that without these two
    # flags, vLLM returns HTTP 400 on any request carrying "tools" - an
    # earlier version of this file omitted them (they existed only in
    # docker-compose.yml), caught by this class's own real smoke test.
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = VllmCoordinator(model="fake/model")
        coordinator.start()
        coordinator.stop()

    cmd = mock_popen.call_args[0][0]
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"


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


def test_llamacpp_base_url_defaults_to_8090_not_8080():
    # Not llama.cpp's conventional 8080 - confirmed live on this device that
    # port is already bound by embedded-ai-chain's own dashboard.
    coordinator = LlamaCppCoordinator(model="fake/model")
    assert coordinator.base_url == "http://127.0.0.1:8090"


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


def test_llamacpp_omits_argv0_for_an_image_that_has_its_own_entrypoint():
    # ghcr.io/ggml-org/llama.cpp:server-cuda (the only image found that runs
    # on Thor) sets ENTRYPOINT [/app/llama-server]; repeating the binary name
    # would reach llama-server as a stray positional argument.
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.subprocess.run"):
        coordinator = LlamaCppCoordinator(
            model="fake/model", image="ghcr.io/ggml-org/llama.cpp:server-cuda", server_argv0=None,
        )
        coordinator.start()
        coordinator.stop()

    cmd = mock_popen.call_args[0][0]
    assert "llama-server" not in cmd
    # the flags still have to be there - only the argv0 is dropped
    assert cmd[cmd.index("ghcr.io/ggml-org/llama.cpp:server-cuda") + 1] == "-hf"
    assert "--jinja" in cmd


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

# ── EdgeLlmCoordinator ───────────────────────────────────────────────────
# Unlike the two above, this one owns a local PROCESS rather than a Docker
# container (Edge-LLM ships no image and no wheel), so these tests assert on
# the argv of `tensorrt-edgellm-serve` itself and on the process-GROUP kill.

def test_edgellm_wait_ready_true_once_health_responds():
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc), \
         patch("llm_coordinator.urllib.request.urlopen", return_value=_healthy_resp()), \
         patch("llm_coordinator.os.killpg"):
        coordinator = EdgeLlmCoordinator(model="fake/model", ready_timeout=3.0)
        coordinator.start()
        assert coordinator.wait_ready(timeout=3.0) is True
        coordinator.stop()


def test_edgellm_base_url_defaults_to_8002_clear_of_both_other_backends():
    # 8000 is VllmCoordinator's default AND the Thor's production
    # vllm-vlm-thor container; 8090 is LlamaCppCoordinator's. Three local
    # servers must be able to coexist on one box.
    coordinator = EdgeLlmCoordinator(model="fake/model")
    assert coordinator.base_url == "http://127.0.0.1:8002"


def test_edgellm_command_enables_structured_tool_calls():
    # Same reasoning as the vLLM equivalent: validate_tool_calling.py needs
    # message.tool_calls back, not tool-call-shaped prose.
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.os.killpg"):
        coordinator = EdgeLlmCoordinator(model="Qwen/Qwen2.5-1.5B-Instruct", ready_timeout=0.1)
        coordinator.start()
        coordinator.wait_ready()
        coordinator.stop()

    cmd = mock_popen.call_args[0][0]
    assert cmd[1] == "Qwen/Qwen2.5-1.5B-Instruct"  # positional checkpoint, not a flag
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser" in cmd
    assert "--cache-dir" in cmd


def test_edgellm_start_puts_cuda_on_path_for_the_engine_builder():
    # /usr/local/cuda/bin is NOT on a login shell's PATH on the Thor, and the
    # engine builder needs nvcc - a real failure mode, not a hypothetical.
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.os.killpg"):
        coordinator = EdgeLlmCoordinator(model="fake/model", ready_timeout=0.1)
        coordinator.start()
        coordinator.wait_ready()
        coordinator.stop()

    env = mock_popen.call_args[1]["env"]
    assert env["PATH"].startswith("/usr/local/cuda/bin:")
    assert mock_popen.call_args[1]["start_new_session"] is True


def test_edgellm_runs_from_the_source_tree_so_its_trt_plugin_resolves():
    # Regression test for a real failure, 2026-09-08: Edge-LLM registers its
    # plugin by the RELATIVE path "build/libNvInfer_edgellm_plugin.so", so a
    # server started from anywhere else dies deserializing its own cached
    # engine ("Cannot find plugin: AttentionPlugin"). cwd is load-bearing.
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.os.killpg"):
        coordinator = EdgeLlmCoordinator(
            model="fake/model",
            serve_bin="/opt/TensorRT-Edge-LLM/.venv/bin/tensorrt-edgellm-serve",
            ready_timeout=0.1,
        )
        coordinator.start()
        coordinator.wait_ready()
        coordinator.stop()

    assert mock_popen.call_args[1]["cwd"] == "/opt/TensorRT-Edge-LLM"


def test_edgellm_served_model_name_decouples_launch_arg_from_wire_name():
    # Regression test for a real failure, 2026-09-09: serving a local
    # checkpoint DIRECTORY makes Edge-LLM register the model under the
    # directory's basename, so a client sending the full path as `model` gets
    # HTTP 404 - which silently failed a whole 8-run benchmark suite. The
    # launch argument and the wire-format name must be independent.
    proc = _fake_proc()
    with patch("llm_coordinator.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.os.killpg"):
        coordinator = EdgeLlmCoordinator(
            model="/home/michal/dev/quantize-work/qwen2.5-7b-int8_sq",
            served_model_name="qwen2.5-7b-int8_sq",
            ready_timeout=0.1,
        )
        coordinator.start()
        coordinator.wait_ready()
        coordinator.stop()

    cmd = mock_popen.call_args[0][0]
    # launched with the PATH...
    assert cmd[1] == "/home/michal/dev/quantize-work/qwen2.5-7b-int8_sq"
    assert "--served-model-name" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "qwen2.5-7b-int8_sq"
    # ...but requests must carry the NAME
    assert coordinator.model == "qwen2.5-7b-int8_sq"


def test_edgellm_model_defaults_to_launch_arg_when_no_served_name():
    # HF repo ids are their own served name, so the common case is unchanged.
    coordinator = EdgeLlmCoordinator(model="Qwen/Qwen2.5-7B-Instruct")
    assert coordinator.model == "Qwen/Qwen2.5-7B-Instruct"


def test_edgellm_stop_signals_the_process_group_not_just_the_parent():
    proc = _fake_proc()
    proc.pid = 4321
    with patch("llm_coordinator.subprocess.Popen", return_value=proc), \
         patch("llm_coordinator.urllib.request.urlopen", return_value=_healthy_resp()), \
         patch("llm_coordinator.os.killpg") as mock_killpg:
        coordinator = EdgeLlmCoordinator(model="fake/model", ready_timeout=3.0)
        coordinator.start()
        coordinator.wait_ready(timeout=3.0)
        coordinator.stop()

    # uvicorn's workers and the C++ runtime thread outlive a bare terminate()
    # and would keep both the port and the GPU allocation held.
    assert mock_killpg.call_args[0][0] == 4321


def test_edgellm_ignores_other_backends_kwargs():
    coordinator = EdgeLlmCoordinator(
        model="fake/model", gpu_memory_utilization=0.15, quant="Q4_K_M", n_gpu_layers=-1,
    )
    assert coordinator.model == "fake/model"


def test_edgellm_process_exit_before_ready_raises():
    proc = _fake_proc(returncode=1)
    with patch("llm_coordinator.subprocess.Popen", return_value=proc), \
         patch("llm_coordinator.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
         patch("llm_coordinator.os.killpg"):
        coordinator = EdgeLlmCoordinator(model="fake/model", ready_timeout=3.0)
        coordinator.start()
        try:
            coordinator.wait_ready(timeout=3.0)
            assert False, "expected the exit to surface"
        except RuntimeError as exc:
            assert "exited unexpectedly" in str(exc)


def test_build_coordinator_dispatches_edge_llm():
    p = argparse.ArgumentParser()
    add_target_args(p)
    args = p.parse_args([])
    coordinator = build_coordinator(args, {"model": "fake/model", "backend": "edge-llm"}, ready_timeout=1.0)
    assert isinstance(coordinator, EdgeLlmCoordinator)


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
