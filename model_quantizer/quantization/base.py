"""Base abstractions shared by all quantizers."""

from __future__ import annotations

import copy
import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model_quantizer.configuration import ModelConfig, QuantizerConfig, RuntimeConfig
from model_quantizer.utils.device import resolve_compute_device


@dataclass(frozen=True)
class QuantizationContext:
    """Resolved inputs for a single quantization run."""

    model_config: ModelConfig
    quantizer_config: QuantizerConfig
    runtime_config: RuntimeConfig
    raw_model_dir: Path
    output_dir: Path
    requested_device: str
    logger: logging.Logger


@dataclass(frozen=True)
class QuantizationResult:
    """Structured output from a quantization run."""

    model_name: str
    quantizer_name: str
    output_dir: Path
    runtime_seconds: float
    original_size_bytes: int
    quantized_size_bytes: int
    manifest: Dict[str, Any]


class BaseQuantizer(ABC):
    """Contract implemented by every quantization backend."""

    def __init__(self, definition: QuantizerConfig) -> None:
        self.definition = definition

    def run(self, context: QuantizationContext) -> QuantizationResult:
        """Execute the quantizer and measure runtime."""

        start = time.perf_counter()
        manifest = self.quantize(context)
        runtime_seconds = time.perf_counter() - start
        return QuantizationResult(
            model_name=context.model_config.name,
            quantizer_name=context.quantizer_config.name,
            output_dir=context.output_dir,
            runtime_seconds=runtime_seconds,
            original_size_bytes=int(manifest["size_summary"]["original_size_bytes"]),
            quantized_size_bytes=int(manifest["size_summary"]["quantized_size_bytes"]),
            manifest=manifest,
        )

    @abstractmethod
    def quantize(self, context: QuantizationContext) -> Dict[str, Any]:
        """Perform quantization and return manifest payload."""

    def save_supporting_files(self, context: QuantizationContext) -> None:
        """Save config/tokenizer files next to the quantized artifact."""

        config = self.load_model_config(context)
        config.save_pretrained(context.output_dir)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                context.raw_model_dir,
                trust_remote_code=context.model_config.trust_remote_code,
            )
            tokenizer.save_pretrained(context.output_dir)
        except Exception as exc:  # pragma: no cover - best effort metadata copy
            context.logger.warning("Tokenizer save skipped: %s", exc)

        self._copy_optional_support_files(context)
        if context.model_config.trust_remote_code:
            self._copy_remote_code_files(context)

    def resolve_quantization_device(self, context: QuantizationContext) -> torch.device:
        """Resolve the compute device for tensor-by-tensor quantization math."""

        return resolve_compute_device(context.requested_device)

    def discover_linear_weight_names(self, context: QuantizationContext) -> List[str]:
        """Build an empty model so we can identify which tensors are Linear weights.

        We intentionally avoid loading the full checkpoint here. `init_empty_weights`
        creates the module structure on a meta device, which lets us inspect the
        architecture cheaply even for very large models.
        """

        config = self.load_model_config(context)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=context.model_config.trust_remote_code,
            )

        linear_names: List[str] = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prefix = f"{module_name}." if module_name else ""
                linear_names.append(f"{prefix}weight")
        return linear_names

    def load_model_config(self, context: QuantizationContext):
        """Load and normalize a model config before instantiating model code."""

        config = AutoConfig.from_pretrained(
            context.raw_model_dir,
            trust_remote_code=context.model_config.trust_remote_code,
        )
        self._normalize_remote_config(context, config)
        return config

    @staticmethod
    def _normalize_remote_config(context: QuantizationContext, config) -> None:
        """Patch known remote-config incompatibilities for local model construction.

        Phi-3 checkpoints can move between two rope-scaling conventions:
        older remote code expects `{"type": "longrope", ...}` while newer
        `transformers` utilities may expose `{"rope_type": "longrope", ...}`.
        Keep the config compatible with both conventions so
        `AutoModelForCausalLM.from_config(...)` can instantiate the module graph
        on a meta device for tensor discovery.
        """

        if getattr(config, "model_type", None) != "phi3":
            return

        rope_scaling = getattr(config, "rope_scaling", None)
        if not rope_scaling:
            config.rope_scaling = None
            context.logger.info(
                "Normalized Phi-3 config: using rope_scaling=None for local model construction."
            )
            return

        if not isinstance(rope_scaling, dict):
            config.rope_scaling = None
            context.logger.info(
                "Normalized Phi-3 config: cleared unsupported rope_scaling=%s.",
                rope_scaling,
            )
            return

        normalized = copy.deepcopy(rope_scaling)
        rope_type = normalized.get("type") or normalized.get("rope_type")
        if rope_type:
            normalized["type"] = rope_type
            normalized["rope_type"] = rope_type

        has_longrope_factors = "short_factor" in normalized and "long_factor" in normalized
        if rope_type in {None, "", "default"} and not has_longrope_factors:
            config.rope_scaling = None
            context.logger.info(
                "Normalized Phi-3 config: cleared default rope_scaling=%s.",
                rope_scaling,
            )
            return

        if not normalized.get("type"):
            if has_longrope_factors:
                normalized["type"] = "longrope"
                normalized["rope_type"] = "longrope"
            else:
                config.rope_scaling = None
                context.logger.info(
                    "Normalized Phi-3 config: cleared incomplete rope_scaling=%s.",
                    rope_scaling,
                )
                return

        config.rope_scaling = normalized
        context.logger.info(
            "Normalized Phi-3 config: canonicalized rope_scaling=%s.",
            normalized,
        )

    @staticmethod
    def _copy_remote_code_files(context: QuantizationContext) -> None:
        """Copy local remote-code Python files into the artifact directory."""

        for source_path in context.raw_model_dir.glob("*.py"):
            destination_path = context.output_dir / source_path.name
            shutil.copy2(source_path, destination_path)
            context.logger.info("Copied remote-code helper: %s", source_path.name)

    @staticmethod
    def _copy_optional_support_files(context: QuantizationContext) -> None:
        """Copy small model-side metadata files needed by some runtimes."""

        for filename in (
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "chat_template.jinja",
        ):
            source_path = context.raw_model_dir / filename
            if not source_path.exists():
                continue
            destination_path = context.output_dir / filename
            shutil.copy2(source_path, destination_path)
            context.logger.info("Copied support file: %s", filename)
