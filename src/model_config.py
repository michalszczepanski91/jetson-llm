"""Loads a named model variant from configs/models.yaml - the shared lookup
every scripts/*.py uses for --model-config, so adding a new variant to
benchmark (a new size, quantization, or backend) only means adding a row to
that one file, not editing every script. Copied verbatim from
jetson-vlm-lab's src/model_config.py (this repo's sibling) - the loader
itself has nothing VLM-specific in it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


def load_model_config(name: str, path: Path = _CONFIG_PATH) -> dict[str, Any]:
    all_configs = yaml.safe_load(path.read_text())
    if name not in all_configs:
        available = ", ".join(sorted(all_configs))
        raise SystemExit(f"unknown --model-config {name!r} - available: {available} (see {path})")
    return all_configs[name]
