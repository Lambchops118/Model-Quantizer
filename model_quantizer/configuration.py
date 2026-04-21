"""Configuration loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the pipeline."""

    raw_models_dir: Path
    quantized_models_dir: Path
    logs_dir: Path
    metadata_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Project-wide runtime settings."""

    default_device: str = "auto"
    max_shard_size_mb: int = 2048
    prefer_safetensors: bool = True
    passthrough_float_dtype: Optional[str] = None


@dataclass(frozen=True)
class ModelConfig:
    """A single user-selectable model definition."""

    name: str
    hf_id: str
    revision: str = "main"
    torch_dtype: str = "float16"
    trust_remote_code: bool = False
    auth_token_env: Optional[str] = None
    enabled: bool = True


@dataclass(frozen=True)
class QuantizerConfig:
    """A single quantizer definition from config."""

    name: str
    type: str
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectConfig:
    """The fully parsed project configuration."""

    source_path: Path
    paths: PathsConfig
    runtime: RuntimeConfig
    models: Dict[str, ModelConfig]
    quantizers: Dict[str, QuantizerConfig]

    def resolve_model_names(
        self,
        requested: Optional[Iterable[str]],
        include_all: bool,
    ) -> List[str]:
        """Resolve CLI model selection into validated model names."""

        if include_all or not requested:
            return [name for name, item in self.models.items() if item.enabled]

        requested_names = list(dict.fromkeys(requested))
        unknown = [name for name in requested_names if name not in self.models]
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
        return requested_names

    def resolve_quantizer_names(
        self,
        requested: Optional[Iterable[str]],
        include_all: bool,
    ) -> List[str]:
        """Resolve CLI quantizer selection into validated quantizer names."""

        if include_all or not requested:
            return [name for name, item in self.quantizers.items() if item.enabled]

        requested_names = list(dict.fromkeys(requested))
        unknown = [name for name in requested_names if name not in self.quantizers]
        if unknown:
            raise ValueError(f"Unknown quantizer(s): {', '.join(unknown)}")
        return requested_names


def _coerce_paths(base_dir: Path, raw: Dict[str, Any]) -> PathsConfig:
    """Build path configuration relative to the config file directory."""

    return PathsConfig(
        raw_models_dir=(base_dir / raw["raw_models_dir"]).resolve(),
        quantized_models_dir=(base_dir / raw["quantized_models_dir"]).resolve(),
        logs_dir=(base_dir / raw["logs_dir"]).resolve(),
        metadata_dir=(base_dir / raw["metadata_dir"]).resolve(),
    )


def _coerce_runtime(raw: Dict[str, Any]) -> RuntimeConfig:
    """Build runtime configuration with defaults."""

    return RuntimeConfig(
        default_device=str(raw.get("default_device", "auto")),
        max_shard_size_mb=int(raw.get("max_shard_size_mb", 2048)),
        prefer_safetensors=bool(raw.get("prefer_safetensors", True)),
        passthrough_float_dtype=raw.get("passthrough_float_dtype"),
    )


def _coerce_models(raw: Dict[str, Dict[str, Any]]) -> Dict[str, ModelConfig]:
    """Build typed model definitions."""

    return {
        name: ModelConfig(
            name=name,
            hf_id=str(item["hf_id"]),
            revision=str(item.get("revision", "main")),
            torch_dtype=str(item.get("torch_dtype", "float16")),
            trust_remote_code=bool(item.get("trust_remote_code", False)),
            auth_token_env=item.get("auth_token_env"),
            enabled=bool(item.get("enabled", True)),
        )
        for name, item in raw.items()
    }


def _coerce_quantizers(raw: Dict[str, Dict[str, Any]]) -> Dict[str, QuantizerConfig]:
    """Build typed quantizer definitions."""

    quantizers: Dict[str, QuantizerConfig] = {}
    for name, item in raw.items():
        options = dict(item)
        quantizer_type = str(options.pop("type"))
        enabled = bool(options.pop("enabled", True))
        quantizers[name] = QuantizerConfig(
            name=name,
            type=quantizer_type,
            enabled=enabled,
            options=options,
        )
    return quantizers


def load_project_config(path: Path) -> ProjectConfig:
    """Load project configuration from YAML."""

    resolved = path.resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    base_dir = resolved.parent.parent if resolved.parent.name == "config" else resolved.parent
    return ProjectConfig(
        source_path=resolved,
        paths=_coerce_paths(base_dir, payload["paths"]),
        runtime=_coerce_runtime(payload.get("runtime", {})),
        models=_coerce_models(payload["models"]),
        quantizers=_coerce_quantizers(payload["quantizers"]),
    )
