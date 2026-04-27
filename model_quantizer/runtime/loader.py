"""Load local raw snapshots or quantized artifacts for benchmark evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_quantizer.artifacts.loader import QuantizedArtifactLoader, load_normalized_model_config
from model_quantizer.configuration import ModelConfig, ProjectConfig
from model_quantizer.download.downloader import ModelDownloader
from model_quantizer.utils.device import resolve_compute_device, resolve_torch_dtype
from model_quantizer.utils.filesystem import sanitize_name


@dataclass(frozen=True)
class ModelLoadRequest:
    """Resolved inputs for loading one local model variant."""

    model_name: str
    source: str
    quantizer_name: Optional[str]
    device: str


@dataclass(frozen=True)
class LoadedModelBundle:
    """All runtime objects needed for one benchmark evaluation pass."""

    model_name: str
    source: str
    source_path: Path
    resolved_device: torch.device
    load_seconds: float
    tokenizer: Any
    model: AutoModelForCausalLM


class LocalModelLoader:
    """Loads local raw snapshots or quantized artifacts for evaluation."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.downloader = ModelDownloader(
            config.paths.raw_models_dir,
            prefer_safetensors=config.runtime.prefer_safetensors,
        )

    def load(self, request: ModelLoadRequest) -> LoadedModelBundle:
        """Load one local model variant from disk."""

        try:
            model_config = self.config.models[request.model_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.config.models))
            raise ValueError(
                f"Unknown model '{request.model_name}'. Available models: {available}"
            ) from exc

        resolved_device = resolve_compute_device(request.device)
        load_start = time.perf_counter()

        if request.source == "raw":
            source_path, tokenizer, model = self._load_raw_model(model_config, resolved_device)
        elif request.source == "quantized":
            if not request.quantizer_name:
                raise ValueError("quantizer_name is required when loading a quantized artifact.")
            source_path, tokenizer, model = self._load_quantized_model(
                request.model_name,
                request.quantizer_name,
                resolved_device,
            )
        else:
            raise ValueError(f"Unsupported model source '{request.source}'.")

        self._prepare_tokenizer(tokenizer)
        model.config.use_cache = False
        model.eval()

        return LoadedModelBundle(
            model_name=request.model_name,
            source=request.source,
            source_path=source_path,
            resolved_device=resolved_device,
            load_seconds=time.perf_counter() - load_start,
            tokenizer=tokenizer,
            model=model,
        )

    def _load_raw_model(
        self,
        model_config: ModelConfig,
        resolved_device: torch.device,
    ) -> tuple[Path, Any, AutoModelForCausalLM]:
        downloaded = self.downloader.require_local_snapshot(model_config)
        tokenizer = AutoTokenizer.from_pretrained(
            downloaded.local_path,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=True,
        )
        config = load_normalized_model_config(
            downloaded.local_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        torch_dtype = self._resolve_model_dtype(model_config, resolved_device)
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": model_config.trust_remote_code,
            "config": config,
            "dtype": torch_dtype,
            "local_files_only": True,
        }
        if getattr(config, "model_type", None) == "phi3":
            model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            downloaded.local_path,
            **model_kwargs,
        )
        return downloaded.local_path, tokenizer, model.to(resolved_device)

    def _load_quantized_model(
        self,
        model_name: str,
        quantizer_name: str,
        resolved_device: torch.device,
    ) -> tuple[Path, Any, AutoModelForCausalLM]:
        artifact_dir = (
            self.config.paths.quantized_models_dir
            / sanitize_name(model_name)
            / sanitize_name(quantizer_name)
        )
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Quantized artifact directory not found: {artifact_dir}. "
                "Run the quantization pipeline first."
            )

        manifest = QuantizedArtifactLoader.load_manifest(artifact_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            artifact_dir,
            trust_remote_code=bool(manifest["source_model"]["trust_remote_code"]),
            local_files_only=True,
        )
        model = QuantizedArtifactLoader.load_model(artifact_dir, device=str(resolved_device))
        if resolved_device.type == "cpu":
            first_param = next(model.parameters(), None)
            if first_param is not None and first_param.dtype == torch.float16:
                model = model.to(dtype=torch.float32)
        return artifact_dir, tokenizer, model

    @staticmethod
    def _prepare_tokenizer(tokenizer: Any) -> None:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    @staticmethod
    def _resolve_model_dtype(
        model_config: ModelConfig,
        resolved_device: torch.device,
    ) -> Optional[torch.dtype]:
        dtype = resolve_torch_dtype(model_config.torch_dtype)
        if resolved_device.type == "cpu" and dtype == torch.float16:
            return torch.float32
        return dtype


def build_prompt_text(
    tokenizer: Any,
    system_prompt: Optional[str],
    user_prompt: str,
) -> str:
    """Render a benchmark prompt through a chat template when available."""

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt})
    return _render_prompt(tokenizer, messages)


def _render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    lines: List[str] = []
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    for message in messages:
        role = role_names.get(message["role"], message["role"].capitalize())
        lines.append(f"{role}: {message['content'].strip()}")
    lines.append("Assistant:")
    return "\n\n".join(lines)
