"""Model downloading and local caching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
            resume_download=True,
            ignore_patterns=["*.onnx", "*.msgpack", "*.h5", "*.tflite"],
        )

        return DownloadedModel(
            name=model_config.name,
            hf_id=model_config.hf_id,
            local_path=local_path,
            revision=model_config.revision,
        )

    @staticmethod
    def _read_token(env_name: Optional[str]) -> Optional[str]:
        if not env_name:
            return None
        return os.environ.get(env_name)
