"""Filesystem helpers used across the project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

from model_quantizer.configuration import PathsConfig


def ensure_runtime_directories(paths: PathsConfig) -> None:
    """Ensure all configured runtime directories exist."""

    for directory in (
        paths.raw_models_dir,
        paths.quantized_models_dir,
        paths.logs_dir,
        paths.metadata_dir,
        paths.benchmark_results_dir,
        paths.benchmark_cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def sanitize_name(value: str) -> str:
    """Convert a user-facing identifier into a filesystem-safe name."""

    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def format_bytes(size_bytes: int) -> str:
    """Format byte counts for logs and metadata."""

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSON Lines with one object per row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")
