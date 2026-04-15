"""Device and dtype helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """Convert string config values into torch dtypes."""

    if name is None:
        return None
    try:
        return DTYPE_MAP[name]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"Unsupported torch dtype '{name}'. Supported values: {supported}") from exc


def resolve_compute_device(device_name: str) -> torch.device:
    """Resolve a CLI/config device name into a torch.device."""

    normalized = device_name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available.")
    return torch.device(device_name)


def collect_device_info(device_name: str) -> Dict[str, Any]:
    """Collect human-readable device information for logs and metadata."""

    resolved = resolve_compute_device(device_name)
    info: Dict[str, Any] = {
        "requested_device": device_name,
        "resolved_device": str(resolved),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        current_index = resolved.index if resolved.type == "cuda" else 0
        props = torch.cuda.get_device_properties(current_index)
        info.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "gpu_name": props.name,
                "gpu_total_memory_bytes": props.total_memory,
                "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "cuda_capability": f"{props.major}.{props.minor}",
            }
        )
    return info
