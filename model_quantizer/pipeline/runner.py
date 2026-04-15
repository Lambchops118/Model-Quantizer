"""Top-level pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from model_quantizer.artifacts.manager import ArtifactManager, write_artifact_metadata
from model_quantizer.configuration import ProjectConfig, QuantizerConfig
from model_quantizer.download.downloader import ModelDownloader
from model_quantizer.quantization.base import QuantizationContext, QuantizationResult
from model_quantizer.quantization.gptq import GPTQQuantizer
from model_quantizer.quantization.int4 import Int4GroupedQuantizer
from model_quantizer.quantization.int8 import Int8Quantizer
from model_quantizer.utils.device import collect_device_info
from model_quantizer.utils.filesystem import ensure_runtime_directories, format_bytes
from model_quantizer.utils.logging_utils import build_pair_logger


@dataclass(frozen=True)
class PipelineSelection:
    """Resolved CLI inputs."""

    model_names: List[str]
    quantizer_names: List[str]
    device: str


class PipelineRunner:
    """Runs the configured download + quantization pipeline."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.downloader = ModelDownloader(
            config.paths.raw_models_dir,
            prefer_safetensors=config.runtime.prefer_safetensors,
        )
        self.artifacts = ArtifactManager(
            quantized_root=config.paths.quantized_models_dir,
            metadata_root=config.paths.metadata_dir,
        )

    def run(self, selection: PipelineSelection) -> List[Dict[str, object]]:
        """Run all selected model/quantizer combinations."""

        ensure_runtime_directories(self.config.paths)
        results: List[Dict[str, object]] = []

        for model_name in selection.model_names:
            model_config = self.config.models[model_name]

            for quantizer_name in selection.quantizer_names:
                quantizer_config = self.config.quantizers[quantizer_name]
                layout = self.artifacts.prepare_layout(model_name, quantizer_name)
                logger, log_path = build_pair_logger(
                    self.config.paths.logs_dir,
                    model_name,
                    quantizer_name,
                )
                downloaded = self.downloader.ensure_downloaded(model_config, logger)
                logger.info("Preparing run for model=%s quantizer=%s", model_name, quantizer_name)
                logger.info("Raw model directory: %s", downloaded.local_path)
                logger.info("Quantized artifact directory: %s", layout.quantized_dir)
                logger.info("Device info: %s", collect_device_info(selection.device))
                logger.info("Quantizer options: %s", quantizer_config.options)

                try:
                    quantizer = build_quantizer(quantizer_config)
                    context = QuantizationContext(
                        model_config=model_config,
                        quantizer_config=quantizer_config,
                        runtime_config=self.config.runtime,
                        raw_model_dir=downloaded.local_path,
                        output_dir=layout.quantized_dir,
                        requested_device=selection.device,
                        logger=logger,
                    )
                    result = quantizer.run(context)
                    manifest = dict(result.manifest)
                    manifest["runtime_seconds"] = round(result.runtime_seconds, 3)
                    manifest["log_file"] = str(log_path)
                    manifest["size_summary"].update(
                        {
                            "original_size_human": format_bytes(result.original_size_bytes),
                            "quantized_size_human": format_bytes(result.quantized_size_bytes),
                        }
                    )
                    write_artifact_metadata(
                        layout.manifest_path,
                        layout.summary_metadata_path,
                        manifest,
                    )
                    logger.info(
                        "Completed quantization in %.2fs | original=%s | quantized=%s",
                        result.runtime_seconds,
                        format_bytes(result.original_size_bytes),
                        format_bytes(result.quantized_size_bytes),
                    )
                    results.append(
                        {
                            "model": model_name,
                            "quantizer": quantizer_name,
                            "status": "success",
                            "artifact_dir": str(layout.quantized_dir),
                            "log_file": str(log_path),
                            "runtime_seconds": round(result.runtime_seconds, 3),
                        }
                    )
                except Exception as exc:
                    logger.exception("Quantization failed: %s", exc)
                    failure_manifest = {
                        "model": model_name,
                        "quantizer": quantizer_name,
                        "status": "error",
                        "error": str(exc),
                        "log_file": str(log_path),
                    }
                    write_artifact_metadata(
                        layout.manifest_path,
                        layout.summary_metadata_path,
                        failure_manifest,
                    )
                    results.append(failure_manifest)

        return results


def build_quantizer(definition: QuantizerConfig):
    """Instantiate a quantizer implementation from config."""

    registry = {
        "int8": Int8Quantizer,
        "int4": Int4GroupedQuantizer,
        "gptq": GPTQQuantizer,
    }
    try:
        quantizer_cls = registry[definition.type]
    except KeyError as exc:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported quantizer type '{definition.type}'. Supported: {supported}") from exc
    return quantizer_cls(definition)
