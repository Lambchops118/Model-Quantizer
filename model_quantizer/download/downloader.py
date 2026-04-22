"""Model downloading and local caching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import snapshot_download

from model_quantizer.configuration import ModelConfig


@dataclass(frozen=True)
class DownloadedModel:
    """Information about a locally cached model snapshot."""

    name: str
    hf_id: str
    local_path: Path
    revision: str


class ModelDownloader:
    """Downloads configured models into the raw-model cache."""

    REQUIRED_MODEL_PATTERNS = (
        "config.json",
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
    )

    def __init__(self, raw_models_dir: Path, prefer_safetensors: bool = True) -> None:
        self.raw_models_dir = raw_models_dir
        self.prefer_safetensors = prefer_safetensors

    def ensure_downloaded(self, model_config: ModelConfig, logger) -> DownloadedModel:
        """Download a model snapshot if it is not already cached locally."""

        local_path = self.raw_models_dir / model_config.name
        local_path.mkdir(parents=True, exist_ok=True)
        token = self._read_token(model_config.auth_token_env)

        logger.info("Ensuring local model snapshot exists at %s", local_path)
        snapshot_download(
            repo_id=model_config.hf_id,
            revision=model_config.revision,
            local_dir=str(local_path),
            token=token,
            ignore_patterns=["*.onnx", "*.msgpack", "*.h5", "*.tflite"],
        )
        self._validate_snapshot_contents(local_path, model_config)

        return DownloadedModel(
            name=model_config.name,
            hf_id=model_config.hf_id,
            local_path=local_path,
            revision=model_config.revision,
        )

    def require_local_snapshot(self, model_config: ModelConfig) -> DownloadedModel:
        """Return a validated local snapshot without downloading anything."""

        local_path = self.raw_models_dir / model_config.name
        if not local_path.exists():
            raise FileNotFoundError(
                "Local raw model snapshot not found for "
                f"{model_config.name} at {local_path}. "
                "Run the quantization/download pipeline first."
            )

        self._validate_snapshot_contents(local_path, model_config)
        return DownloadedModel(
            name=model_config.name,
            hf_id=model_config.hf_id,
            local_path=local_path,
            revision=model_config.revision,
        )

    def _validate_snapshot_contents(self, local_path: Path, model_config: ModelConfig) -> None:
        """Reject partial local snapshots that do not contain model weights.

        Some gated repositories expose documentation files before the user has
        accepted the license or supplied a valid token. In that state
        `snapshot_download(...)` can leave behind a directory that looks real
        but does not actually contain a loadable model.
        """

        config_path = local_path / "config.json"
        weight_files = list(self._iter_matching_files(local_path, self.REQUIRED_MODEL_PATTERNS[1:]))
        if config_path.exists() and weight_files:
            return

        visible_files = sorted(
            str(path.relative_to(local_path))
            for path in local_path.rglob("*")
            if path.is_file()
        )
        preview = ", ".join(visible_files[:8]) if visible_files else "<empty>"
        raise RuntimeError(
            "Local snapshot for "
            f"{model_config.name} ({model_config.hf_id}) is incomplete: expected config.json and at least one "
            "weight file (*.safetensors, *.bin, *.pt, *.pth). "
            f"Found: {preview}. "
            "If this is a gated repository, confirm that your Hugging Face account has access and that "
            f"{model_config.auth_token_env or 'the configured token env var'} is set."
        )

    @staticmethod
    def _iter_matching_files(root: Path, patterns: Iterable[str]):
        for pattern in patterns:
            yield from root.rglob(pattern)

    @staticmethod
    def _read_token(env_name: Optional[str]) -> Optional[str]:
        if not env_name:
            return None
        return os.environ.get(env_name)
