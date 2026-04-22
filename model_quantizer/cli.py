"""Command-line interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from model_quantizer.configuration import load_project_config
from model_quantizer.inference.runner import InferenceRequest, InferenceRunner
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
        "--run-model",
        default=None,
        help="Load a configured model for inference instead of running quantization.",
    )
    parser.add_argument(
        "--model-source",
        choices=("raw", "quantized"),
        default="raw",
        help="Inference source for --run-model. Local files only.",
    )
    parser.add_argument(
        "--artifact-quantizer",
        default=None,
        help="Quantized artifact name to load when --model-source=quantized.",
    )
    parser.add_argument(
        "--chat-mode",
        choices=("one-shot", "conversational"),
        default=None,
        help="Inference mode. one-shot is stateless; conversational remembers history.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Initial user prompt for one-shot mode.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt injected before the first user query.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for inference.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling cutoff for inference.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty applied to repeated token patterns during inference.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling and use greedy decoding.",
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

    if args.run_model:
        inference_request = InferenceRequest(
            model_name=args.run_model,
            source=args.model_source,
            quantizer_name=args.artifact_quantizer,
            mode=args.chat_mode or config.inference.default_mode,
            device=args.device or config.runtime.default_device,
            prompt=args.prompt,
            system_prompt=(
                args.system_prompt
                if args.system_prompt is not None
                else config.inference.system_prompt
            ),
            max_new_tokens=(
                args.max_new_tokens
                if args.max_new_tokens is not None
                else config.inference.max_new_tokens
            ),
            temperature=(
                args.temperature if args.temperature is not None else config.inference.temperature
            ),
            top_p=args.top_p if args.top_p is not None else config.inference.top_p,
            do_sample=False if args.no_sample else config.inference.do_sample,
            repetition_penalty=(
                args.repetition_penalty
                if args.repetition_penalty is not None
                else config.inference.repetition_penalty
            ),
        )
        InferenceRunner(config).run(inference_request)
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
