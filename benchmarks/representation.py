"""Decompose the one-word precision label into the ~6 independent choices it
actually bundles (task brief, Priority 3).

The problem this solves is concrete and already live in this repo's own
data. `configs/models.yaml` labels one row `"AWQ int4 (weights), bf16
activations"` and another `"GGUF fp16"`, and a third `"compressed-tensors
int4, group_size=128, symmetric=false (NOT classic AWQ despite the repo
name)"` - that last label exists because a previous campaign nearly compared
two different things under one name. A figure that puts llama.cpp's Q4_K_M
next to Edge-LLM's INT4-GPTQ as if both were "INT4" is making a claim that
is simply false: Q4_K_M is a *mixed* scheme that leaves attention.v and the
feed-forward down-projection at 6 bits and `output.weight` at Q6_K, while
GPTQ-Int4 is uniform 4-bit grouped with an FP16 lm_head. Their effective
bits per weight differ by roughly a bit, and a bit at 7B is not a rounding
error.

Two design rules here:

  * **Read the artifact, do not trust the name.** Every field below is
    extracted from the file that will actually be loaded - the safetensors
    header, the GGUF tensor table, the Edge-LLM build config - not from the
    `precision:` string in the registry. The registry string becomes
    `declared_label`, kept only so a mismatch between what a row claims and
    what its artifact contains is visible rather than silent.
  * **`effective_bits_per_weight` is the one genuinely cross-framework
    number.** artifact bytes / parameter count x 8. It does not care whose
    quantizer produced the file, what the scheme is called, or which modules
    were spared; it is what the storage cost per weight actually came to. It
    is the only quantity in this block that may be compared across all three
    backends without a caveat, which is why every other field exists mainly
    to explain it.

Stdlib only - same constraint as benchmarks/envelope.py, and for the same
reason.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# GGML / GGUF tables
# --------------------------------------------------------------------------

#: ggml tensor type id -> (name, block_size, bytes_per_block). Straight from
#: ggml.h's type_traits. Needed because a GGUF tensor's byte size is NOT
#: n_elements x something - K-quants pack 256 elements into an irregular
#: block with its own scale/min fields, and getting this table wrong would
#: silently corrupt `effective_bits_per_weight`, the one number that is
#: supposed to be trustworthy across backends.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}

#: safetensors dtype -> bytes per element.
SAFETENSORS_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "U64": 8,
    "U32": 4, "U16": 2, "I4": 0.5, "U4": 0.5,
}

#: Tensor-name fragments that identify a module a quantizer commonly spares.
#: Matched case-insensitively against the tensor name as the artifact spells
#: it, so it works for both HF naming (`model.layers.0.input_layernorm`) and
#: GGUF naming (`blk.0.attn_norm.weight`).
_MODULE_PATTERNS = {
    "lm_head": ("lm_head", "output.weight"),
    "embeddings": ("embed_tokens", "token_embd", "wte"),
    "norms": ("norm", "layernorm", "ln_"),
    "biases": (".bias",),
}

_FLOAT_TYPES = {"F32", "F16", "BF16", "F64"}


# --------------------------------------------------------------------------
# artifact readers
# --------------------------------------------------------------------------

def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Tensor table of one .safetensors shard, without loading any weights.

    The format puts a uint64 little-endian header length at byte 0 followed
    by that many bytes of JSON, so this is a ~200KB read of a 4GB file."""
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header


def _read_gguf_string(fh) -> str:
    (n,) = struct.unpack("<Q", fh.read(8))
    return fh.read(n).decode("utf-8", errors="replace")


def _skip_gguf_value(fh, vtype: int) -> Any:
    """Read (and mostly discard) one GGUF metadata value. Values are only
    returned for the scalar/string types the caller might want; arrays are
    walked to keep the file position correct but summarised, because a GGUF
    token vocabulary is a 150k-element string array and materialising it
    here would cost more than everything else this module does."""
    simple = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
              6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if vtype in simple:
        fmt = simple[vtype]
        return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
    if vtype == 8:
        return _read_gguf_string(fh)
    if vtype == 9:
        (elem_type,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        for _ in range(count):
            _skip_gguf_value(fh, elem_type)
        return f"<array of {count} type-{elem_type}>"
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_tensors(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(metadata, tensors) from a GGUF file's header, no weights read.

    Only the FIRST shard of a split GGUF carries the full tensor table for
    that shard; callers wanting a whole split model must read every shard
    and concatenate, which `describe_gguf` does."""
    with path.open("rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path.name} is not a GGUF file (magic {magic!r})")
        version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
        meta: dict[str, Any] = {"gguf_version": version}
        for _ in range(n_kv):
            key = _read_gguf_string(fh)
            (vtype,) = struct.unpack("<I", fh.read(4))
            meta[key] = _skip_gguf_value(fh, vtype)
        tensors = []
        for _ in range(n_tensors):
            name = _read_gguf_string(fh)
            (n_dims,) = struct.unpack("<I", fh.read(4))
            dims = struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims))
            (ttype,) = struct.unpack("<I", fh.read(4))
            (offset,) = struct.unpack("<Q", fh.read(8))
            tensors.append({"name": name, "dims": list(dims), "type": ttype, "offset": offset})
    return meta, tensors


def _gguf_tensor_bytes(t: dict[str, Any]) -> int | None:
    info = GGML_TYPES.get(t["type"])
    if not info:
        return None
    _, block, per_block = info
    elems = 1
    for d in t["dims"]:
        elems *= d
    if elems % block:
        # A quantised tensor whose row length is not a multiple of the block
        # size cannot be stored in that type at all; seeing one means this
        # table is wrong for this file, and a wrong byte count here would
        # poison effective_bits_per_weight silently.
        return None
    return elems // block * per_block


def _classify_modules(names_and_types: list[tuple[str, str]]) -> dict[str, Any]:
    """Which of the commonly-spared modules are still at float precision.

    Returns, per module family, the set of distinct storage types its
    tensors actually use - so `{"lm_head": ["Q6_K"]}` and
    `{"lm_head": ["F16"]}` are both reportable facts rather than a boolean
    that would have to pick one of them to be wrong about."""
    out: dict[str, Any] = {}
    for family, patterns in _MODULE_PATTERNS.items():
        types = sorted({
            ty for name, ty in names_and_types
            if any(p.lower() in name.lower() for p in patterns)
        })
        if types:
            out[family] = {
                "storage_types": types,
                "kept_float": all(t in _FLOAT_TYPES for t in types),
            }
    return out


# --------------------------------------------------------------------------
# per-backend description
# --------------------------------------------------------------------------

def describe_safetensors_model(model_dir: Path) -> dict[str, Any]:
    """Representation of an HF-format checkpoint (what vLLM and Edge-LLM
    both consume)."""
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {model_dir}")

    artifact_bytes = sum(p.stat().st_size for p in shards)
    params = 0
    quantised_params = 0
    names_and_types: list[tuple[str, str]] = []
    dtypes: dict[str, int] = {}
    for shard in shards:
        for name, info in read_safetensors_header(shard).items():
            elems = 1
            for d in info["shape"]:
                elems *= d
            dtype = info["dtype"]
            names_and_types.append((name, dtype))
            dtypes[dtype] = dtypes.get(dtype, 0) + elems
            # An int32-packed qweight holds several logical weights per
            # element; counting its elements as parameters would undercount
            # the model by ~8x and make effective_bits_per_weight nonsense.
            if name.endswith("qweight"):
                quantised_params += elems
            else:
                params += elems

    qcfg = cfg.get("quantization_config") or {}
    bits = qcfg.get("bits") or qcfg.get("weight_bits")
    if quantised_params and bits:
        params += quantised_params * (32 // int(bits))

    return {
        "source": "safetensors header",
        "artifact_paths": [str(p) for p in shards],
        "artifact_bytes": artifact_bytes,
        "param_count": params,
        "param_count_method": (
            "summed from the safetensors tensor table; qweight tensors expanded by "
            "32/bits since they pack several weights per int32 element"
            if quantised_params else "summed from the safetensors tensor table"
        ),
        "storage_dtypes": dict(sorted(dtypes.items(), key=lambda kv: -kv[1])),
        "quantization_config": qcfg or None,
        "modules": _classify_modules(names_and_types),
        "hf_config": {
            k: cfg.get(k) for k in (
                "model_type", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "hidden_size", "vocab_size",
                "max_position_embeddings", "torch_dtype",
            ) if cfg.get(k) is not None
        },
    }


def describe_gguf(first_shard: Path) -> dict[str, Any]:
    """Representation of a GGUF model, following a split set if there is one.

    Every shard is read because a split GGUF distributes its tensors across
    files - reading only shard 1 would report roughly a quarter of the
    parameters at whatever mix of types happened to land there."""
    stem = first_shard.name
    if "-of-" in stem:
        shards = sorted(first_shard.parent.glob(stem.split("-0000")[0] + "-*.gguf"))
    else:
        shards = [first_shard]

    meta, _ = read_gguf_tensors(shards[0])
    params = 0
    tensor_bytes = 0
    unknown_types: set[int] = set()
    type_params: dict[str, int] = {}
    names_and_types: list[tuple[str, str]] = []
    for shard in shards:
        _, tensors = read_gguf_tensors(shard)
        for t in tensors:
            elems = 1
            for d in t["dims"]:
                elems *= d
            tname = GGML_TYPES.get(t["type"], (f"type-{t['type']}", 0, 0))[0]
            names_and_types.append((t["name"], tname))
            params += elems
            type_params[tname] = type_params.get(tname, 0) + elems
            nbytes = _gguf_tensor_bytes(t)
            if nbytes is None:
                unknown_types.add(t["type"])
            else:
                tensor_bytes += nbytes

    return {
        "source": "GGUF tensor table",
        "artifact_paths": [str(p) for p in shards],
        "artifact_bytes": sum(p.stat().st_size for p in shards),
        "tensor_bytes": tensor_bytes if not unknown_types else None,
        "param_count": params,
        "param_count_method": "summed from the GGUF tensor table across every shard",
        "storage_dtypes": dict(sorted(type_params.items(), key=lambda kv: -kv[1])),
        "unknown_ggml_types": sorted(unknown_types) or None,
        "gguf_file_type": meta.get("general.file_type"),
        "gguf_architecture": meta.get("general.architecture"),
        "modules": _classify_modules(names_and_types),
        "hf_config": {
            "num_hidden_layers": meta.get(f"{meta.get('general.architecture')}.block_count"),
            "num_attention_heads": meta.get(f"{meta.get('general.architecture')}.attention.head_count"),
            "num_key_value_heads": meta.get(f"{meta.get('general.architecture')}.attention.head_count_kv"),
            "hidden_size": meta.get(f"{meta.get('general.architecture')}.embedding_length"),
        },
    }


# --------------------------------------------------------------------------
# the decomposed representation block
# --------------------------------------------------------------------------

def effective_bits_per_weight(artifact_bytes: int, param_count: int) -> float | None:
    """artifact bytes / parameter count x 8.

    The only quantity in this module comparable across all three backends
    without a caveat. Deliberately computed from the WHOLE artifact, not
    from the weight tensors alone: the bytes a config costs on disk and in
    unified memory include its spared lm_head, its embedding table and its
    quantiser's scales and zero-points, and a scheme that shrinks the linear
    layers by paying for fat scales has not saved what a nominal "4-bit"
    would suggest. That is the comparison this number is for.

    The cost of that choice, stated so it is not mistaken for precision: for
    a GGUF it also includes the file's token vocabulary and metadata, a
    fixed few MB that matters at 1.5B and is noise at 7B."""
    if not param_count:
        return None
    return artifact_bytes / param_count * 8


def kv_cache_bytes_per_token(
    n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_element: float
) -> int:
    """2 x n_layers x n_kv_heads x head_dim x bytes_per_element.

    The 2 is key and value. Grouped-query attention is why `n_kv_heads` and
    not `n_attention_heads` belongs here, and it is a 7x difference for
    Qwen2.5-7B (4 KV heads against 28 attention heads) - using the wrong one
    would overstate the cache by enough to reverse a fits/does-not-fit
    verdict on this board."""
    return int(2 * n_layers * n_kv_heads * head_dim * bytes_per_element)


def _comparability_class(weights: dict, activations: dict, storage_dtypes: dict) -> str:
    """A coarse tag whose only job is to stop a figure from silently
    comparing two things that are not the same kind of object.

    Deliberately blunt. It is never used to claim two configs in the same
    class ARE equivalent - only to mark, on a heatmap or a table, the cells
    where a direct comparison is not defensible at all."""
    w_dtype = (weights.get("dtype") or "").lower()
    a_dtype = (activations.get("dtype") or "").lower()
    n_types = len([t for t, c in storage_dtypes.items() if c > 0])

    if w_dtype in ("fp16", "f16", "bf16", "float16", "bfloat16"):
        return "fp16_baseline"
    if "fp8" in w_dtype or "f8_e4m3" in w_dtype:
        return "fp8_e4m3"
    if w_dtype.startswith("int8") or w_dtype == "q8_0":
        return "w8a8_int" if a_dtype.startswith("int8") else "w8a16_int"
    if w_dtype.startswith("int4") or w_dtype.startswith("q4") or w_dtype.startswith("q5"):
        # A GGUF K-quant is a MIXED scheme - Q4_K_M leaves attn_v and
        # ffn_down at Q6_K - so it is not the same object as a uniform
        # grouped int4 even at the same nominal bit width. Separate class,
        # on purpose.
        if n_types > 3:
            return "w4_grouped_mixed"
        return "w4_grouped"
    return "unclassified"


def build_representation(
    *,
    artifact: dict[str, Any],
    declared_label: str,
    backend: str,
    kv_cache_dtype: str | None,
    kv_cache_dtype_source: str,
    activations_dtype: str | None,
    activations_dtype_source: str,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the decomposed representation block for one config.

    `artifact` is the output of `describe_safetensors_model` or
    `describe_gguf`. The three axes the brief insists on separating -
    weights, activations, KV cache - are separate arguments here precisely
    because only the first is readable from the artifact: activation and KV
    dtype are properties of how the *runtime* was launched, and each arrives
    with a `_source` saying how it was established, so a value read off a
    server's own log is never confused with one asserted by a config file."""
    storage = artifact.get("storage_dtypes") or {}
    qcfg = artifact.get("quantization_config") or {}
    dominant = max(storage.items(), key=lambda kv: kv[1])[0] if storage else None

    method = qcfg.get("quant_method")
    bits = qcfg.get("bits") or qcfg.get("weight_bits")
    group_size = qcfg.get("group_size")
    if not method and artifact["source"] == "GGUF tensor table":
        method = "gguf-k-quant" if any(t.endswith("_K") for t in storage) else "gguf"

    if bits:
        w_dtype = f"int{bits}"
    elif dominant:
        w_dtype = {"BF16": "bf16", "F16": "fp16", "F32": "fp32",
                   "F8_E4M3": "fp8_e4m3"}.get(dominant, dominant.lower())
    else:
        w_dtype = None

    weights = {
        "dtype": w_dtype,
        "method": method or "none (unquantized)",
        "granularity": (
            "per-group" if group_size else
            "per-block" if (method or "").startswith("gguf") else
            "per-tensor" if method else "n/a (unquantized)"
        ),
        "group_size": group_size or (256 if (method or "").startswith("gguf-k") else None),
        "symmetric": qcfg.get("sym"),
        "storage_dtype_histogram_params": storage,
        "_granularity_source": (
            "quantization_config in the checkpoint's config.json" if qcfg
            else "GGUF tensor type table" if artifact["source"] == "GGUF tensor table"
            else "safetensors tensor table - no quantization_config present, so this "
                 "checkpoint declares no quantizer and the storage dtype IS the scheme"
        ),
    }
    activations = {"dtype": activations_dtype, "_source": activations_dtype_source}
    kv = {"dtype": kv_cache_dtype, "_source": kv_cache_dtype_source}

    modules = artifact.get("modules") or {}
    fp_modules = sorted(k for k, v in modules.items() if v.get("kept_float"))
    quantised_modules = sorted(k for k, v in modules.items() if not v.get("kept_float"))

    ebw = effective_bits_per_weight(artifact["artifact_bytes"], artifact["param_count"])
    return {
        "declared_label": declared_label,
        "backend": backend,
        "weights": weights,
        "activations": activations,
        "kv_cache": kv,
        "modules_kept_fp": fp_modules,
        "modules_quantized": quantised_modules,
        "module_storage_types": {k: v["storage_types"] for k, v in modules.items()},
        "calibration": calibration or {
            "dataset": None,
            "n_samples": None,
            "matches_deployment_domain": None,
            "_not_collected": (
                "no calibration step in this config's provenance (unquantized, or a "
                "vendor checkpoint whose calibration set the publisher did not state)"
            ),
        },
        "artifact_bytes": artifact["artifact_bytes"],
        "param_count": artifact["param_count"],
        "param_count_method": artifact["param_count_method"],
        "effective_bits_per_weight": round(ebw, 4) if ebw else None,
        "comparability_class": _comparability_class(weights, activations, storage),
    }
