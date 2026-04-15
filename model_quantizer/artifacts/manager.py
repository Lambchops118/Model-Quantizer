"""Artifact layout and sharded tensor persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
from safetensors.torch import save_file

from model_quantizer.utils.filesystem import sanitize_name, write_json


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolved output locations for one quantized artifact."""

    quantized_dir: Path
    manifest_path: Path
    summary_metadata_path: Path


@dataclass
class ShardRecord:
    """Metadata about a single written safetensors shard."""

    filename: str
    tensor_keys: List[str] = field(default_factory=list)
    size_bytes: int = 0


class ArtifactManager:
    """Owns artifact path conventions."""

    def __init__(self, quantized_root: Path, metadata_root: Path) -> None:
        self.quantized_root = quantized_root
        self.metadata_root = metadata_root

    def prepare_layout(self, model_name: str, quantizer_name: str) -> ArtifactLayout:
        """Create directories for a (model, quantizer) output."""

        quantized_dir = self.quantized_root / sanitize_name(model_name) / sanitize_name(quantizer_name)
        quantized_dir.mkdir(parents=True, exist_ok=True)

        summary_metadata_path = (
            self.metadata_root
            / f"{sanitize_name(model_name)}__{sanitize_name(quantizer_name)}.json"
        )
        return ArtifactLayout(
            quantized_dir=quantized_dir,
            manifest_path=quantized_dir / "quantization_manifest.json",
            summary_metadata_path=summary_metadata_path,
        )


class ShardedTensorWriter:
    """Writes tensors into sequential safetensors shards.

    The writer keeps only one shard's worth of tensors in memory. This is useful
    when quantizing large models because we can flush intermediate results without
    waiting for the entire model artifact to be assembled in RAM.
    """

    def __init__(self, output_dir: Path, max_shard_size_bytes: int) -> None:
        self.output_dir = output_dir
        self.max_shard_size_bytes = max_shard_size_bytes
        self._current_tensors: Dict[str, torch.Tensor] = {}
        self._current_size_bytes = 0
        self._shards: List[ShardRecord] = []

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Queue a tensor for the current shard and flush if needed."""

        tensor = tensor.detach().cpu().contiguous()
        tensor_size = tensor.numel() * tensor.element_size()
        if (
            self._current_tensors
            and self._current_size_bytes + tensor_size > self.max_shard_size_bytes
        ):
            self._flush()
        self._current_tensors[name] = tensor
        self._current_size_bytes += tensor_size

    def finalize(self) -> List[Dict[str, Any]]:
        """Write any remaining tensors and return shard metadata."""

        if self._current_tensors:
            self._flush()
        return [
            {
                "filename": shard.filename,
                "tensor_keys": shard.tensor_keys,
                "size_bytes": shard.size_bytes,
            }
            for shard in self._shards
        ]

    def _flush(self) -> None:
        shard_index = len(self._shards) + 1
        filename = f"artifact-{shard_index:05d}.safetensors"
        output_path = self.output_dir / filename
        save_file(self._current_tensors, str(output_path))
        self._shards.append(
            ShardRecord(
                filename=filename,
                tensor_keys=list(self._current_tensors.keys()),
                size_bytes=self._current_size_bytes,
            )
        )
        self._current_tensors = {}
        self._current_size_bytes = 0


def write_artifact_metadata(manifest_path: Path, summary_path: Path, payload: Dict[str, Any]) -> None:
    """Write both the in-artifact manifest and the cross-run summary metadata."""

    write_json(manifest_path, payload)
    write_json(summary_path, payload)
