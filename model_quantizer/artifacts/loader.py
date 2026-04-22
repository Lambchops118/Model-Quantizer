"""Helpers for reloading quantized artifacts later."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM


def _normalize_remote_config(config) -> None:
    """Patch known remote-config incompatibilities before model construction."""

    if getattr(config, "model_type", None) != "phi3":
        return

    rope_scaling = getattr(config, "rope_scaling", None)
    if not rope_scaling:
        config.rope_scaling = None
        return

    if not isinstance(rope_scaling, dict):
        config.rope_scaling = None
        return

    normalized = copy.deepcopy(rope_scaling)
    rope_type = normalized.get("type") or normalized.get("rope_type")
    if rope_type:
        normalized["type"] = rope_type
        normalized["rope_type"] = rope_type

    has_longrope_factors = "short_factor" in normalized and "long_factor" in normalized
    if rope_type in {None, "", "default"} and not has_longrope_factors:
        config.rope_scaling = None
        return

    if not normalized.get("type"):
        if has_longrope_factors:
            normalized["type"] = "longrope"
            normalized["rope_type"] = "longrope"
        else:
            config.rope_scaling = None
            return

    config.rope_scaling = normalized


def load_normalized_model_config(
    model_dir: Path,
    trust_remote_code: bool,
    *,
    local_files_only: bool = True,
):
    """Load a model config and normalize known remote-code quirks."""

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    _normalize_remote_config(config)
    return config


def _iter_auto_map_module_files(config) -> Iterable[str]:
    """Yield Python module filenames referenced by a config auto_map."""

    auto_map = getattr(config, "auto_map", None) or {}
    for value in auto_map.values():
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple)):
            candidates = [item for item in value if isinstance(item, str)]
        else:
            continue

        for candidate in candidates:
            class_ref = candidate.split("--", 1)[-1]
            module_name = class_ref.split(".", 1)[0].strip()
            if module_name:
                yield f"{module_name}.py"


def _validate_artifact_model_files(artifact_dir: Path, config) -> None:
    """Require quantized artifacts to be self-contained for model construction."""

    required_files = set(_iter_auto_map_module_files(config))
    if not required_files:
        return

    if all((artifact_dir / filename).exists() for filename in required_files):
        return

    missing = ", ".join(sorted(required_files))
    raise RuntimeError(
        "Quantized artifact is not self-contained. Missing remote-code files: "
        f"{missing}. Expected them under {artifact_dir}. "
        "Re-quantize this model with the current code so the artifact contains "
        "all required support files."
    )


class QuantizedArtifactLoader:
    """Reconstructs a standard Transformers model from a saved artifact.

    The manual quantizers store compressed weights plus metadata. This loader
    dequantizes those weights back into dense tensors and then loads the state
    dict into a normal `AutoModelForCausalLM` instance so Part 2 can evaluate it.
    """

    @staticmethod
    def load_manifest(artifact_dir: Path) -> Dict[str, Any]:
        manifest_path = artifact_dir / "quantization_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def load_state_dict(cls, artifact_dir: Path) -> Dict[str, torch.Tensor]:
        """Reconstruct the dequantized state dict from a saved artifact."""

        manifest = cls.load_manifest(artifact_dir)
        tensor_store: Dict[str, torch.Tensor] = {}
        for shard in manifest["artifact_shards"]:
            shard_tensors = load_file(str(artifact_dir / shard["filename"]), device="cpu")
            tensor_store.update(shard_tensors)

        state_dict: Dict[str, torch.Tensor] = {}
        for record in manifest["tensor_records"]:
            kind = record["kind"]
            name = record["name"]
            if kind == "passthrough":
                state_dict[name] = tensor_store[record["storage_key"]]
            elif kind == "int8_weight":
                state_dict[name] = cls._restore_int8(record, tensor_store)
            elif kind == "int4_weight":
                state_dict[name] = cls._restore_int4(record, tensor_store)
            else:
                raise ValueError(f"Unsupported record kind: {kind}")
        return state_dict

    @classmethod
    def load_model(cls, artifact_dir: Path, device: str = "cpu") -> AutoModelForCausalLM:
        """Instantiate and populate a Transformers model from an artifact."""

        manifest = cls.load_manifest(artifact_dir)
        trust_remote_code = bool(manifest["source_model"]["trust_remote_code"])

        model_config = load_normalized_model_config(
            artifact_dir,
            trust_remote_code=trust_remote_code,
        )
        _validate_artifact_model_files(artifact_dir, model_config)
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if getattr(model_config, "model_type", None) == "phi3":
            model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_config(
            model_config,
            **model_kwargs,
        )
        incompatible = model.load_state_dict(cls.load_state_dict(artifact_dir), strict=False)
        allowed_missing = set()
        if getattr(model_config, "tie_word_embeddings", False):
            allowed_missing.add("lm_head.weight")

        unexpected_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Artifact state_dict did not match model structure. "
                f"missing={unexpected_missing} "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )

        if getattr(model_config, "tie_word_embeddings", False):
            model.tie_weights()
        return model.to(device)

    @staticmethod
    def _restore_int8(record: Dict[str, Any], tensor_store: Dict[str, torch.Tensor]) -> torch.Tensor:
        qweight = tensor_store[record["qweight_key"]].to(torch.float32)
        scales = tensor_store[record["scale_key"]].to(torch.float32)
        original_shape = tuple(record["original_shape"])
        if record["granularity"] == "per_tensor":
            restored = qweight * scales.item()
        else:
            view_shape = [1] * len(original_shape)
            view_shape[int(record["channel_axis"])] = original_shape[int(record["channel_axis"])]
            restored = qweight * scales.view(*view_shape)
        return restored.reshape(original_shape).to(getattr(torch, record["original_dtype"]))

    @staticmethod
    def _restore_int4(record: Dict[str, Any], tensor_store: Dict[str, torch.Tensor]) -> torch.Tensor:
        packed = tensor_store[record["packed_weight_key"]].flatten().to(torch.uint8)
        scales = tensor_store[record["scale_key"]].to(torch.float32)
        original_shape = tuple(record["original_shape"])
        original_numel = int(record["original_numel"])

        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unsigned = torch.stack((low, high), dim=1).reshape(-1)[:original_numel]
        signed = unsigned.to(torch.int16) - int(record["storage_zero_point"])
        restored = signed.to(torch.float32).view(record["rows"], record["padded_columns"])

        group_size = int(record["group_size"])
        dequantized = torch.empty_like(restored, dtype=torch.float32)
        for row_index in range(record["rows"]):
            column_start = 0
            for group_index, scale in enumerate(scales[row_index]):
                column_end = min(column_start + group_size, record["padded_columns"])
                dequantized[row_index, column_start:column_end] = (
                    restored[row_index, column_start:column_end] * scale
                )
                column_start = column_end

        trimmed = dequantized[:, : record["columns"]]
        return trimmed.reshape(original_shape).to(getattr(torch, record["original_dtype"]))
