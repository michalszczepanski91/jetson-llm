"""Artifact readers and the decomposed representation block.

The property under test throughout is that these functions read the FILE,
not the label. A registry row that says "int4" and an artifact that stores
5.03 bits per weight must produce 5.03, because that number is the only one
comparable across the three runtimes."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import representation as R


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Minimal well-formed safetensors file: a uint64 header length, that
    many bytes of JSON, then the (here empty) data block."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        elems = 1
        for d in shape:
            elems *= d
        nbytes = int(elems * R.SAFETENSORS_DTYPE_BYTES[dtype])
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def test_effective_bits_is_exactly_16_for_an_fp16_checkpoint(tmp_path):
    """The calibration case for the whole metric: an unquantized 16-bit
    checkpoint must come out at 16.00 bits per weight, or every quantized
    number derived the same way is wrong by the same factor."""
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [512, 512]),
        "lm_head.weight": ("BF16", [1000, 512]),
    })
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 2,
        "hidden_size": 512, "torch_dtype": "bfloat16"}))
    art = R.describe_safetensors_model(tmp_path)
    assert art["param_count"] == 512 * 512 + 1000 * 512
    assert R.effective_bits_per_weight(art["artifact_bytes"], art["param_count"]) == pytest.approx(16.0, abs=0.01)


def test_packed_qweight_is_expanded_so_bits_per_weight_is_not_8x_wrong(tmp_path):
    """A GPTQ/AWQ `qweight` is int32 holding eight 4-bit weights. Counting
    its ELEMENTS as parameters would undercount the model eightfold and
    report a 4-bit checkpoint at ~32 bits per weight."""
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.layers.0.mlp.down_proj.qweight": ("I32", [64, 512]),
    })
    (tmp_path / "config.json").write_text(json.dumps({
        "quantization_config": {"bits": 4, "group_size": 128, "quant_method": "gptq", "sym": True}}))
    art = R.describe_safetensors_model(tmp_path)
    assert art["param_count"] == 64 * 512 * 8
    assert "expanded" in art["param_count_method"]


def test_kv_per_token_uses_kv_heads_not_attention_heads():
    """Grouped-query attention makes this a 7x error for Qwen2.5-7B - large
    enough to reverse a fits / does-not-fit verdict."""
    gqa = R.kv_cache_bytes_per_token(28, 4, 128, 2)
    mha = R.kv_cache_bytes_per_token(28, 28, 128, 2)
    assert gqa == 57344
    assert mha == gqa * 7


def test_kv_per_token_halves_for_an_8bit_cache():
    assert R.kv_cache_bytes_per_token(28, 4, 128, 1) == 57344 // 2


def test_gguf_k_quant_is_not_classified_as_plain_int4():
    """Q4_K_M mixes Q4_K and Q6_K tensors, so it is not the same object as a
    uniform grouped int4 even at the same nominal width. The comparability
    class must keep a figure from putting them side by side."""
    weights = {"dtype": "int4", "method": "gguf-k-quant"}
    mixed = {"Q4_K": 10, "Q6_K": 5, "F32": 1, "Q5_K": 2}
    uniform = {"I32": 10, "BF16": 1}
    assert R._comparability_class(weights, {}, mixed) == "w4_grouped_mixed"
    assert R._comparability_class(weights, {}, uniform) == "w4_grouped"


def test_comparability_separates_w8a8_from_w8a16():
    """Activation dtype is its own axis: INT8 weights with FP16 activations
    and INT8 weights with INT8 activations are different configurations and
    a table must not average them."""
    w = {"dtype": "int8"}
    assert R._comparability_class(w, {"dtype": "int8"}, {}) == "w8a8_int"
    assert R._comparability_class(w, {"dtype": "fp16"}, {}) == "w8a16_int"


def test_module_classification_reports_types_not_a_boolean():
    """'Was lm_head quantized?' has three answers on real artifacts - yes,
    no, and partly - so the block records the storage types it found."""
    out = R._classify_modules([
        ("output.weight", "Q6_K"), ("blk.0.attn_norm.weight", "F32"),
        ("token_embd.weight", "Q4_K"),
    ])
    assert out["lm_head"]["storage_types"] == ["Q6_K"]
    assert out["lm_head"]["kept_float"] is False
    assert out["norms"]["kept_float"] is True


def test_unknown_ggml_type_yields_none_not_a_guess():
    assert R._gguf_tensor_bytes({"dims": [256], "type": 999}) is None


def test_gguf_block_size_mismatch_is_refused():
    """A tensor whose element count is not a multiple of its block size
    cannot be stored in that type; guessing a byte count here would corrupt
    effective_bits_per_weight silently."""
    assert R._gguf_tensor_bytes({"dims": [100], "type": 12}) is None
    assert R._gguf_tensor_bytes({"dims": [512], "type": 12}) == 2 * 144
