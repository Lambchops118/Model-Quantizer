"""Shared helpers for manual quantization backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class SourceTensor:
    """A single source tensor read from a model checkpoint."""

    name: str
    tensor: torch.Tensor
    source_file: str


def discover_checkpoint_files(model_dir: Path) -> List[Path]:
    """Return model checkpoint shard files in a stable order."""

    safetensor_index = model_dir / "model.safetensors.index.json"
    bin_index = model_dir / "pytorch_model.bin.index.json"

    if safetensor_index.exists():
        return _files_from_index(model_dir, safetensor_index)
    if bin_index.exists():
        return _files_from_index(model_dir, bin_index)

    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if safetensors_files:
        return safetensors_files

    bin_files = sorted(model_dir.glob("*.bin"))
    if bin_files:
        return bin_files

    raise FileNotFoundError(f"No supported checkpoint files found under {model_dir}")


def count_checkpoint_tensors(model_dir: Path) -> int:
    """Count tensors without materializing the entire checkpoint."""

    safetensor_index = model_dir / "model.safetensors.index.json"
    bin_index = model_dir / "pytorch_model.bin.index.json"
    if safetensor_index.exists():
        return _count_from_index(safetensor_index)
    if bin_index.exists():
        return _count_from_index(bin_index)

    total = 0
    for checkpoint_file in discover_checkpoint_files(model_dir):
        if checkpoint_file.suffix == ".safetensors":
            with safe_open(str(checkpoint_file), framework="pt", device="cpu") as handle:
                total += len(handle.keys())
        else:
            total += len(torch.load(checkpoint_file, map_location="cpu"))
    return total


def iter_source_tensors(model_dir: Path) -> Iterator[SourceTensor]:
    """Yield tensors from the raw checkpoint one by one."""

    for checkpoint_file in discover_checkpoint_files(model_dir):
        if checkpoint_file.suffix == ".safetensors":
            with safe_open(str(checkpoint_file), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    yield SourceTensor(
                        name=name,
                        tensor=handle.get_tensor(name),
                        source_file=checkpoint_file.name,
                    )
        else:
            state_dict = torch.load(checkpoint_file, map_location="cpu")
            for name, tensor in state_dict.items():
                yield SourceTensor(name=name, tensor=tensor, source_file=checkpoint_file.name)


def maybe_cast_passthrough_tensor(tensor: torch.Tensor, dtype_name: Optional[str]) -> torch.Tensor:
    """Optionally cast non-quantized float tensors before they are stored."""

    if dtype_name is None or not tensor.is_floating_point():
        return tensor
    return tensor.to(getattr(torch, dtype_name))


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Byte size of a tensor."""

    return tensor.numel() * tensor.element_size()


def _files_from_index(model_dir: Path, index_path: Path) -> List[Path]:
    with index_path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Dict[str, str]] = json.load(handle)
    filenames = sorted(set(payload["weight_map"].values()))
    return [model_dir / filename for filename in filenames]


def _count_from_index(index_path: Path) -> int:
    with index_path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Dict[str, str]] = json.load(handle)
    return len(payload["weight_map"])
