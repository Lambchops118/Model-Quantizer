"""Command-line interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from model_quantizer.artifacts.cleanup import QuantizedArtifactCleaner
from model_quantizer.configuration import load_project_config


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Download, quantize, benchmark, and clean up configured LLM artifacts."
    )
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
        help="Runtime device. Examples: auto, cpu, cuda, cuda:0",
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
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmark tasks and exit.",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run automated benchmark evaluation against local raw and quantized models.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Specific benchmark names to run. Defaults to all enabled benchmark tasks.",
    )
    parser.add_argument(
        "--no-raw-baseline",
        action="store_true",
        help="Skip raw-model baselines and evaluate only quantized artifacts.",
    )
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=None,
        help="Optional cap on the number of examples evaluated per benchmark.",
    )
    parser.add_argument(
        "--cleanup-benchmarked-quantized",
        action="store_true",
        help=(
            "Remove local quantized artifact directories that already have successful, "
            "full benchmark summaries for the selected benchmarks."
        ),
    )
    return parser


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from a local .env file.

    Existing environment variables win over values in `.env`.
    This keeps shell-provided secrets authoritative while allowing
    project-local configuration for users who cannot export variables.
    """

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        os.environ[key] = value


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    load_dotenv()
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

    if args.list_benchmarks:
        for name, benchmark in config.benchmarks.tasks.items():
            print(f"{name}: {benchmark.type}")
        return 0

    if args.cleanup_benchmarked_quantized:
        benchmark_names = config.resolve_benchmark_names(args.benchmarks)
        cleaner = QuantizedArtifactCleaner(
            quantized_root=config.paths.quantized_models_dir,
            benchmark_results_root=config.paths.benchmark_results_dir,
        )
        results = cleaner.cleanup_ready_artifacts(benchmark_names)
        removed_count = sum(1 for item in results if item["status"] == "removed")
        skipped_count = len(results) - removed_count
        print(
            f"Scanned {len(results)} quantized artifacts: "
            f"{removed_count} removed, {skipped_count} skipped."
        )
        for item in results:
            print(
                f"- {item['model_name']} | {item['quantizer_name']} | "
                f"{item['status']} | reason={item['reason']}"
            )
        return 0

    if args.run_benchmarks:
        from model_quantizer.benchmarks import BenchmarkRunner, BenchmarkSelection

        selection = BenchmarkSelection(
            model_names=config.resolve_model_names(args.models, args.all_models),
            quantizer_names=config.resolve_quantizer_names(args.quantizers, args.all_quantizers),
            benchmark_names=config.resolve_benchmark_names(args.benchmarks),
            device=args.device or config.runtime.default_device,
            include_raw_baseline=not args.no_raw_baseline,
            max_examples_per_benchmark=args.benchmark_limit,
        )
        runner = BenchmarkRunner(config)
        results = runner.run(selection)

        success_count = sum(1 for item in results if item["status"] == "success")
        error_count = len(results) - success_count
        print(
            f"Completed {len(results)} benchmark evaluations: "
            f"{success_count} succeeded, {error_count} failed."
        )
        for item in results:
            print(
                f"- {item['model_name']} | {item['variant_label']} | {item['benchmark_name']} | "
                f"{item['status']} | summary={item['summary_path']}"
            )
        return 0 if error_count == 0 else 1

    from model_quantizer.pipeline.runner import PipelineRunner, PipelineSelection

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
