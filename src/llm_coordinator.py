"""Lifecycle owners for the two serving backends this lab compares
(vLLM, llama.cpp), plus a target for an already-running remote server (e.g.
Thor) - the orchestrator-LLM counterpart to jetson-vlm-lab's
src/vlm_coordinator.py. Every class here shares one duck-typed interface
(`base_url`, `model`, `start()`, `wait_ready(timeout)`, `stop(timeout)`) so
every scripts/*.py caller works unchanged regardless of which backend or
target a configs/models.yaml row points at - see docs/promotion-contract.md
§1, copied from jetson-vlm-lab's own contract (already proven there when the
VLM moved from local Orin to a remote Thor with zero caller changes).

Both real backends speak an OpenAI-compatible `/v1/chat/completions` (+
`GET /health`) HTTP API - vLLM by design, llama.cpp's `llama-server` the
same way as of recent releases - so RemoteCoordinator below (which only
ever speaks that wire format, never a backend-specific one) needs no
per-backend variant at all.

Text-only: unlike jetson-vlm-lab's VlmCoordinator (silent about images -
image encoding lives in vlm_client.py instead), there is nothing
image-specific to differ here either. What differs from jetson-vlm-lab is
that TWO local coordinators exist now (one per backend) instead of one.
"""

from __future__ import annotations

import subprocess
import threading
import time
import urllib.error
import urllib.request

_DEFAULT_VLLM_IMAGE = "ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin"
_DEFAULT_HF_CACHE = "/opt/hf-cache"

# Confirmed live on this device, 2026-09-04. The bare "r36.4.0" tag exists
# but ships an old llama.cpp build (version 4579) whose server hard-errors
# on any request carrying tool_choice ({"code":500,"message":"Unsupported
# param: tool_choice"}) - useless for this lab's purpose. This tag (labeled
# cu128/24.04, i.e. built against a newer CUDA/Ubuntu base than this
# device's actual CUDA 12.6/Ubuntu 22.04) nonetheless runs correctly here -
# nvidia-container-runtime's driver mount is backward-compatible, GPU
# detected fine (`Orin, compute capability 8.7`) - and its much newer
# llama.cpp build (version 5058) accepts tool_choice without erroring and
# recognizes the model's Hermes-2-Pro tool format. Real, still-open finding
# from that first run: it accepted the param but the 1.5B model described
# wanting to call the tool in free text rather than emitting a structured
# tool_calls response - a reliability gap vs vLLM's grammar-constrained
# --tool-call-parser hermes, not a config error here. See docs/TODO.md
# Phase 1 for the full record; scripts/validate_tool_calling.py's real BFCL
# run is what actually quantifies this, not further one-off probing.
_DEFAULT_LLAMACPP_IMAGE = "dustynv/llama_cpp:0.3.9-r36.4.0-cu128-24.04"
_DEFAULT_LLAMA_CACHE = "/opt/llama-cache"  # separate from _DEFAULT_HF_CACHE -
# llama.cpp's `-hf` downloader (LLAMA_CACHE env var) uses its own on-disk
# layout, not the transformers/vLLM HF_HOME cache format - sharing one dir
# between the two would just be two incompatible layouts fighting over the
# same path, not a real reuse.


class VllmCoordinator:
    """Local Docker container running vLLM. Copied from jetson-vlm-lab's
    VlmCoordinator - nothing in that class was actually VLM-specific
    (no image handling lives here, only in vlm_client.py there / llm_client.py
    here), so this is a straight rename, not a rewrite."""

    def __init__(
        self,
        model: str,
        port: int = 8000,
        gpu_memory_utilization: float = 0.15,
        max_model_len: int = 4096,
        enable_prefix_caching: bool = True,
        image: str = _DEFAULT_VLLM_IMAGE,
        hf_cache_dir: str = _DEFAULT_HF_CACHE,
        container_name: str = "vllm-llm-lab",
        ready_timeout: float = 600.0,
        stop_timeout: float = 15.0,
        extra_args: list[str] | None = None,
        **_ignored,  # e.g. quant/n_gpu_layers/ctx_size from a llama-cpp row
                     # read generically by a caller that doesn't branch per backend
    ):
        self._model = model
        self._port = port
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._enable_prefix_caching = enable_prefix_caching
        self._image = image
        self._hf_cache_dir = hf_cache_dir
        self._container_name = container_name
        self._ready_timeout = ready_timeout
        self._stop_timeout = stop_timeout
        self._extra_args = list(extra_args) if extra_args else []

        self._proc: subprocess.Popen | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self.error: BaseException | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def model(self) -> str:
        return self._model

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError(f"{type(self).__name__} already started")
        self._stop_event.clear()
        cmd = [
            "docker", "run", "--name", self._container_name, "--rm",
            "--runtime", "nvidia", "--network", "host", "--ipc", "host",
            "-v", f"{self._hf_cache_dir}:/root/.cache/huggingface",
            "-e", "HF_HOME=/root/.cache/huggingface",
            self._image,
            "/opt/venv/bin/vllm", "serve", self._model,
            "--host", "0.0.0.0", "--port", str(self._port),
            "--gpu-memory-utilization", str(self._gpu_memory_utilization),
            "--max-model-len", str(self._max_model_len),
            "--trust-remote-code",
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
            # required for tools/tool_choice requests to work at all - confirmed
            # live 2026-09-04: without these, vLLM returns a 400 on any request
            # carrying "tools". Every real caller here (validate_tool_calling.py)
            # needs structured tool calls, so this isn't optional the way it might
            # be for a plain-chat-only server. Matches embedded-ai-chain's own
            # production orchestrator-vllm-compose.yml, which already has this -
            # this file previously did not (an earlier version had it only in
            # docker-compose.yml, not here, and this class's own smoke test is
            # what caught the mismatch).
        ]
        if not self._enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        cmd.extend(self._extra_args)
        self._proc = subprocess.Popen(cmd)
        self._poll_thread = threading.Thread(target=self._poll_health, daemon=True, name=f"{self._container_name}-health")
        self._poll_thread.start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        ready = self._ready_event.wait(timeout=timeout if timeout is not None else self._ready_timeout)
        if self.error is not None:
            raise self.error
        return ready

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout if timeout is not None else self._stop_timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        # Safety net regardless of terminate/kill outcome above - `docker run
        # --rm`'s container isn't reliably stopped by signaling the parent
        # process alone (same reasoning as jetson-vlm-lab's VlmCoordinator).
        subprocess.run(
            ["bash", "-c", f'docker ps -q --filter "name={self._container_name}" | xargs -r docker stop'],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self._proc = None
        self._ready_event.clear()

    def _poll_health(self) -> None:
        try:
            deadline = time.monotonic() + self._ready_timeout
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"{self._image} container exited unexpectedly "
                        f"(code {self._proc.returncode}) before becoming ready"
                    )
                try:
                    with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as resp:
                        if resp.status == 200:
                            self._ready_event.set()
                            return
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(2.0)
        except BaseException as exc:  # noqa: BLE001 - surfaced via .error / wait_ready()
            self.error = exc
            self._ready_event.set()


class LlamaCppCoordinator:
    """Local Docker container running llama.cpp's `llama-server`. New in
    this repo (jetson-vlm-lab never used this backend). Pulls the GGUF
    directly via llama.cpp's own `-hf <repo>:<quant>` downloader (same
    "point at an HF repo id, let the server fetch it" ergonomics
    VllmCoordinator gets from vLLM) rather than requiring a pre-staged local
    file - `quant` is a configs/models.yaml field (e.g. "Q4_K_M") selecting
    which file in that repo to pull.

    `--jinja` is required, not optional, for OpenAI-style `tool_calls` to
    come back structured (llama.cpp's chat-template/tool-call grammar path)
    - without it, the model can still emit tool-call-shaped text but
    llama-server won't parse it into `message.tool_calls`, which would
    silently break scripts/validate_tool_calling.py's scoring. Mirrors why
    embedded-ai-chain's own vLLM invocation passes
    `--enable-auto-tool-choice --tool-call-parser hermes` for the same
    reason on that backend.
    """

    def __init__(
        self,
        model: str,
        quant: str = "Q4_K_M",
        port: int = 8090,  # NOT llama.cpp's own conventional 8080 default -
        # confirmed live on this device (2026-09-04) that 8080 is already
        # bound by embedded-ai-chain's own live dashboard
        # (tts_consumer.py's dashboard.py, port 8080) - a real collision,
        # not a hypothetical one, so this lab picks a different default
        # rather than risk silently fighting the production pipeline for
        # that port.
        n_gpu_layers: int = -1,  # -1 = offload every layer to GPU
        ctx_size: int = 4096,
        image: str = _DEFAULT_LLAMACPP_IMAGE,
        llama_cache_dir: str = _DEFAULT_LLAMA_CACHE,
        container_name: str = "llamacpp-llm-lab",
        ready_timeout: float = 600.0,
        stop_timeout: float = 15.0,
        extra_args: list[str] | None = None,
        **_ignored,  # e.g. gpu_memory_utilization from a vLLM row read
                     # generically by a caller that doesn't branch per backend
    ):
        self._model = model
        self._quant = quant
        self._port = port
        self._n_gpu_layers = n_gpu_layers
        self._ctx_size = ctx_size
        self._image = image
        self._llama_cache_dir = llama_cache_dir
        self._container_name = container_name
        self._ready_timeout = ready_timeout
        self._stop_timeout = stop_timeout
        self._extra_args = list(extra_args) if extra_args else []

        self._proc: subprocess.Popen | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self.error: BaseException | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def model(self) -> str:
        return self._model

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError(f"{type(self).__name__} already started")
        self._stop_event.clear()
        cmd = [
            "docker", "run", "--name", self._container_name, "--rm",
            "--runtime", "nvidia", "--network", "host", "--ipc", "host",
            "-v", f"{self._llama_cache_dir}:/root/.cache/llama.cpp",
            "-e", "LLAMA_CACHE=/root/.cache/llama.cpp",
            self._image,
            "llama-server",
            "-hf", f"{self._model}:{self._quant}",
            "--host", "0.0.0.0", "--port", str(self._port),
            "-ngl", str(self._n_gpu_layers),
            "-c", str(self._ctx_size),
            "--jinja",
        ]
        cmd.extend(self._extra_args)
        self._proc = subprocess.Popen(cmd)
        self._poll_thread = threading.Thread(target=self._poll_health, daemon=True, name=f"{self._container_name}-health")
        self._poll_thread.start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        ready = self._ready_event.wait(timeout=timeout if timeout is not None else self._ready_timeout)
        if self.error is not None:
            raise self.error
        return ready

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout if timeout is not None else self._stop_timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        subprocess.run(
            ["bash", "-c", f'docker ps -q --filter "name={self._container_name}" | xargs -r docker stop'],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self._proc = None
        self._ready_event.clear()

    def _poll_health(self) -> None:
        try:
            deadline = time.monotonic() + self._ready_timeout
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"{self._image} container exited unexpectedly "
                        f"(code {self._proc.returncode}) before becoming ready"
                    )
                try:
                    with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as resp:
                        if resp.status == 200:
                            self._ready_event.set()
                            return
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(2.0)
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            self._ready_event.set()


class RemoteCoordinator:
    """Talks to an already-running OpenAI-compatible server on a separate,
    externally-managed machine (currently a Jetson Thor) instead of
    starting/stopping a local Docker container - backend-agnostic (vLLM or
    llama-server both qualify, since both speak the same wire format), so
    unlike the two classes above there is only one of these. Copied
    verbatim from jetson-vlm-lab's RemoteVlmCoordinator, renamed to drop the
    "Vlm" (nothing about it was ever VLM-specific either).

    start()/stop() are deliberately no-ops: this class doesn't own the
    remote server's lifecycle, so there is no cold start to measure -
    callers must not report `cold_start_ms` from a run against this class
    as real (see scripts' `--target remote` handling)."""

    def __init__(
        self,
        model: str,
        host: str,
        port: int = 8000,
        ready_timeout: float = 600.0,
        poll_interval: float = 2.0,
        **_ignored,  # e.g. gpu_memory_utilization/n_gpu_layers - accepted so
                     # callers can pass the same variant dict regardless of target
    ):
        self._model = model
        self._host = host
        self._port = port
        self._ready_timeout = ready_timeout
        self._poll_interval = poll_interval
        self.error: BaseException | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def model(self) -> str:
        return self._model

    def start(self) -> None:
        pass  # nothing to launch - see class docstring

    def wait_ready(self, timeout: float | None = None) -> bool:
        deadline = time.monotonic() + (timeout if timeout is not None else self._ready_timeout)
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(self._poll_interval)
        return False

    def stop(self, timeout: float | None = None) -> None:
        pass  # doesn't own the remote server's lifecycle - see class docstring


def add_target_args(parser) -> None:
    """Shared --target/--remote-host/--remote-port args for every
    scripts/*.py entry point, so callers don't each hand-duplicate the same
    flags/dispatch logic. Copied from jetson-vlm-lab's vlm_coordinator.py."""
    parser.add_argument(
        "--target", choices=["local", "remote"], default="local",
        help="'local' starts/stops a Docker container on this machine (default); "
        "'remote' talks to an already-running server elsewhere (e.g. Thor) - "
        "see --remote-host/--remote-port",
    )
    parser.add_argument(
        "--remote-host", default=None,
        help="required with --target remote - e.g. the Thor's LAN IP",
    )
    parser.add_argument(
        "--remote-port", type=int, default=8000,
        help="port the remote server listens on (default 8000)",
    )


def build_coordinator(args, variant: dict, ready_timeout: float, **local_kwargs):
    """Constructs the right coordinator for --target local/remote AND (new
    vs jetson-vlm-lab, which only ever had one local backend) the variant's
    own `backend` field for --target local. Callers pass whatever
    backend-specific kwargs the local coordinators need
    (gpu_memory_utilization/max_model_len/enable_prefix_caching for vllm;
    quant/n_gpu_layers/ctx_size for llama-cpp) - each local class ignores
    kwargs meant for the other backend via **_ignored, matching
    RemoteCoordinator's own **_ignored for local-only kwargs."""
    if args.target == "remote":
        if not args.remote_host:
            raise SystemExit("--target remote requires --remote-host")
        return RemoteCoordinator(
            model=variant["model"],
            host=args.remote_host,
            port=args.remote_port,
            ready_timeout=ready_timeout,
        )
    backend = variant["backend"]
    if backend == "vllm":
        return VllmCoordinator(model=variant["model"], ready_timeout=ready_timeout, **local_kwargs)
    if backend == "llama-cpp":
        return LlamaCppCoordinator(model=variant["model"], ready_timeout=ready_timeout, **local_kwargs)
    raise SystemExit(f"unknown backend {backend!r} in configs/models.yaml - expected 'vllm' or 'llama-cpp'")
