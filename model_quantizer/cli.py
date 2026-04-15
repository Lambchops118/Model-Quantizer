"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from model_quantizer.configuration import load_project_config
from model_quantizer.pipeline.runner import PipelineRunner, PipelineSelection


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""

    parser = argparse.ArgumentParser(description="Download and quantize configured LLMs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model names to run.",
    )
    parser.add_argument(
        "--quantizers",
        nargs="+",
        default=None,
        help="Specific quantizer names to run.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all enabled models from config.",
    )
    parser.add_argument(
        "--all-quantizers",
        action="store_true",
        help="Run all enabled quantizers from config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Quantization compute device. Examples: auto, cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--list-quantizers",
        action="store_true",
        help="List available quantizers and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_project_config(args.config)
    if args.list_models:
        for name, model in config.models.items():
            print(f"{name}: {model.hf_id}")
        return 0

    if args.list_quantizers:
        for name, quantizer in config.quantizers.items():
            print(f"{name}: {quantizer.type}")
        return 0

    selection = PipelineSelection(
        model_names=config.resolve_model_names(args.models, args.all_models),
        quantizer_names=config.resolve_quantizer_names(args.quantizers, args.all_quantizers),
        device=args.device or config.runtime.default_device,
    )
    runner = PipelineRunner(config)
    results = runner.run(selection)

    success_count = sum(1 for item in results if item["status"] == "success")
    error_count = len(results) - success_count
    print(f"Completed {len(results)} runs: {success_count} succeeded, {error_count} failed.")
    for item in results:
        print(
            f"- {item['model']} | {item['quantizer']} | {item['status']} | "
            f"log={item['log_file']}"
        )
    return 0 if error_count == 0 else 1
