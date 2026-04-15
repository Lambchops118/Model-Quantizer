"""Base abstractions shared by all quantizers."""

from __future__ import annotations

import logging
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

        config = AutoConfig.from_pretrained(
            context.raw_model_dir,
            trust_remote_code=context.model_config.trust_remote_code,
        )
        config.save_pretrained(context.output_dir)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                context.raw_model_dir,
                trust_remote_code=context.model_config.trust_remote_code,
            )
            tokenizer.save_pretrained(context.output_dir)
        except Exception as exc:  # pragma: no cover - best effort metadata copy
            context.logger.warning("Tokenizer save skipped: %s", exc)

    def resolve_quantization_device(self, context: QuantizationContext) -> torch.device:
        """Resolve the compute device for tensor-by-tensor quantization math."""

        return resolve_compute_device(context.requested_device)

    def discover_linear_weight_names(self, context: QuantizationContext) -> List[str]:
        """Build an empty model so we can identify which tensors are Linear weights.

        We intentionally avoid loading the full checkpoint here. `init_empty_weights`
        creates the module structure on a meta device, which lets us inspect the
        architecture cheaply even for very large models.
        """

        config = AutoConfig.from_pretrained(
            context.raw_model_dir,
            trust_remote_code=context.model_config.trust_remote_code,
        )
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
