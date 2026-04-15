"""Helpers for reloading quantized artifacts later."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM


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
        config = AutoConfig.from_pretrained(
            artifact_dir,
            trust_remote_code=bool(manifest["source_model"]["trust_remote_code"]),
        )
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=bool(manifest["source_model"]["trust_remote_code"]),
        )
        model.load_state_dict(cls.load_state_dict(artifact_dir))
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
