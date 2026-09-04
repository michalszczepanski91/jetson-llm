"""Unit tests for model_config.load_model_config() against the real
configs/models.yaml - no GPU/Docker needed, just confirms the registry
itself is well-formed and every row required by docs/promotion-contract.md
§2 is present."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_config import load_model_config  # noqa: E402

_REQUIRED_FIELDS = {"model", "precision", "backend", "platform"}
_KNOWN_BACKENDS = {"vllm", "llama-cpp"}


def test_known_key_loads():
    variant = load_model_config("1.5b-awq-vllm-orin")
    assert variant["model"] == "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    assert variant["backend"] == "vllm"


def test_unknown_key_raises_system_exit_listing_available():
    try:
        load_model_config("does-not-exist")
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "1.5b-awq-vllm-orin" in str(exc)


def test_every_row_has_required_fields_and_a_known_backend():
    import yaml

    all_configs = yaml.safe_load((Path(__file__).parent.parent / "configs" / "models.yaml").read_text())
    assert all_configs, "configs/models.yaml is empty"
    for key, variant in all_configs.items():
        missing = _REQUIRED_FIELDS - variant.keys()
        assert not missing, f"{key} is missing required fields: {missing}"
        assert variant["backend"] in _KNOWN_BACKENDS, f"{key} has unrecognized backend {variant['backend']!r}"
