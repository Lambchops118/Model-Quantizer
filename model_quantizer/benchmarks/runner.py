"""Batch benchmark runner for raw and quantized local models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from model_quantizer.artifacts.cleanup import QuantizedArtifactCleaner
from model_quantizer.artifacts.loader import QuantizedArtifactLoader
from model_quantizer.configuration import BenchmarkTaskConfig, ProjectConfig
from model_quantizer.runtime import (
    LoadedModelBundle,
    LocalModelLoader,
    ModelLoadRequest,
    build_prompt_text,
)
from model_quantizer.utils.filesystem import ensure_runtime_directories, sanitize_name, write_json, write_jsonl

try:  # pragma: no cover - import guard depends on optional environment state
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised only in environments without datasets
    load_dataset = None


@dataclass(frozen=True)
class BenchmarkSelection:
    """Resolved CLI inputs for a benchmark run."""

    model_names: List[str]
    quantizer_names: List[str]
    benchmark_names: List[str]
    device: str
    include_raw_baseline: bool
    max_examples_per_benchmark: Optional[int] = None


@dataclass(frozen=True)
class BenchmarkExample:
    """Canonical multiple-choice benchmark example."""

    example_id: str
    subject: Optional[str]
    prompt: str
    choices: List[str]
    answer_index: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class LoadedBenchmark:
    """Benchmark examples plus source metadata."""

    name: str
    task_type: str
    split: str
    dataset_name: Optional[str]
    dataset_config: Optional[str]
    revision: str
    examples: List[BenchmarkExample]
    metadata: Dict[str, Any]


class BenchmarkRunner:
    """Evaluates local raw snapshots and quantized artifacts on benchmark tasks."""

    DEFAULT_PROGRESS_INTERVAL = 5

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.model_loader = LocalModelLoader(config)
        self.artifact_cleaner = QuantizedArtifactCleaner(
            quantized_root=config.paths.quantized_models_dir,
            benchmark_results_root=config.paths.benchmark_results_dir,
        )

    def run(self, selection: BenchmarkSelection) -> List[Dict[str, Any]]:
        """Evaluate all selected models, variants, and benchmarks."""

        ensure_runtime_directories(self.config.paths)
        benchmarks = {
            name: self._load_benchmark(
                self.config.benchmarks.tasks[name],
                max_examples_override=selection.max_examples_per_benchmark,
            )
            for name in selection.benchmark_names
        }

        results: List[Dict[str, Any]] = []
        for model_name in selection.model_names:
            variant_requests = self._build_variant_requests(model_name, selection)
            for request in variant_requests:
                variant_label = request.quantizer_name or "raw"
                bundle: Optional[LoadedModelBundle] = None
                request_summaries: List[Dict[str, Any]] = []
                try:
                    bundle = self.model_loader.load(request)
                    source_metrics = self._collect_source_metrics(request, bundle.source_path)
                    for benchmark in benchmarks.values():
                        try:
                            summary = self._evaluate_benchmark(
                                bundle=bundle,
                                request=request,
                                benchmark=benchmark,
                                source_metrics=source_metrics,
                            )
                        except Exception as exc:
                            summary = self._build_error_summary(
                                model_name=model_name,
                                variant_label=variant_label,
                                request=request,
                                benchmark=benchmark,
                                error=exc,
                            )
                        self._write_benchmark_outputs(summary)
                        results.append(summary)
                        request_summaries.append(summary)
                except Exception as exc:
                    for benchmark in benchmarks.values():
                        summary = self._build_error_summary(
                            model_name=model_name,
                            variant_label=variant_label,
                            request=request,
                            benchmark=benchmark,
                            error=exc,
                        )
                        self._write_benchmark_outputs(summary)
                        results.append(summary)
                        request_summaries.append(summary)
                finally:
                    self._release_model(bundle)
                self._maybe_cleanup_quantized_artifact(
                    request=request,
                    selection=selection,
                    summaries=request_summaries,
                )
        return results

    def _build_variant_requests(
        self,
        model_name: str,
        selection: BenchmarkSelection,
    ) -> List[ModelLoadRequest]:
        requests: List[ModelLoadRequest] = []
        if selection.include_raw_baseline:
            requests.append(
                self._build_variant_request(
                    model_name,
                    source="raw",
                    device=selection.device,
                )
            )

        for quantizer_name in selection.quantizer_names:
            requests.append(
                self._build_variant_request(
                    model_name,
                    source="quantized",
                    quantizer_name=quantizer_name,
                    device=selection.device,
                )
            )
        return requests

    def _build_variant_request(
        self,
        model_name: str,
        source: str,
        device: str,
        quantizer_name: Optional[str] = None,
    ) -> ModelLoadRequest:
        return ModelLoadRequest(
            model_name=model_name,
            source=source,
            quantizer_name=quantizer_name,
            device=device,
        )

    def _load_benchmark(
        self,
        task: BenchmarkTaskConfig,
        *,
        max_examples_override: Optional[int],
    ) -> LoadedBenchmark:
        if load_dataset is None:
            raise RuntimeError(
                "Benchmark mode requires the 'datasets' package. "
                "Install project dependencies again to enable HellaSwag and MMLU evaluation."
            )

        examples: List[BenchmarkExample] = []
        if task.type == "hellaswag":
            rows = self._load_dataset_rows(task)
            for row_index, row in enumerate(rows):
                endings = [str(item) for item in row["endings"]]
                examples.append(
                    BenchmarkExample(
                        example_id=str(row.get("ind", row_index)),
                        subject=None,
                        prompt=self._format_hellaswag_prompt(
                            activity_label=str(row.get("activity_label", "")),
                            context=str(row["ctx"]),
                        ),
                        choices=endings,
                        answer_index=int(row["label"]),
                        metadata={
                            "activity_label": str(row.get("activity_label", "")),
                            "source_id": str(row.get("source_id", "")),
                            "split_type": str(row.get("split_type", "")),
                        },
                    )
                )
                if self._limit_reached(examples, task, max_examples_override):
                    break
        elif task.type == "mmlu":
            rows = self._load_dataset_rows(task)
            allowed_subjects = set(task.subjects)
            for row_index, row in enumerate(rows):
                subject = str(row["subject"])
                if allowed_subjects and subject not in allowed_subjects:
                    continue
                examples.append(
                    BenchmarkExample(
                        example_id=f"{subject}:{row_index}",
                        subject=subject,
                        prompt=self._format_mmlu_prompt(
                            question=str(row["question"]),
                            choices=[str(choice) for choice in row["choices"]],
                        ),
                        choices=[" A", " B", " C", " D"],
                        answer_index=self._coerce_mmlu_answer(row["answer"]),
                        metadata={"subject": subject},
                    )
                )
                if self._limit_reached(examples, task, max_examples_override):
                    break
        else:
            raise ValueError(f"Unsupported benchmark type '{task.type}'.")

        if not examples:
            raise RuntimeError(
                f"Benchmark '{task.name}' did not produce any examples. "
                "Check the configured split, subjects, and dataset source."
            )

        return LoadedBenchmark(
            name=task.name,
            task_type=task.type,
            split=task.split,
            dataset_name=task.dataset_name,
            dataset_config=task.dataset_config,
            revision=task.revision,
            examples=examples,
            metadata={
                "source": task.source,
                "subjects": task.subjects,
                "max_examples": max_examples_override or task.max_examples,
            },
        )

    def _load_dataset_rows(self, task: BenchmarkTaskConfig) -> Iterable[Dict[str, Any]]:
        if task.source == "hub":
            if not task.dataset_name:
                raise ValueError(f"Benchmark '{task.name}' is missing dataset_name.")
            return load_dataset(
                task.dataset_name,
                task.dataset_config,
                split=task.split,
                revision=task.revision,
                cache_dir=str(self.config.paths.benchmark_cache_dir),
            )

        if task.source == "local_jsonl":
            if task.local_path is None:
                raise ValueError(f"Benchmark '{task.name}' is missing local_path.")
            return load_dataset(
                "json",
                data_files=str(task.local_path),
                split="train",
                cache_dir=str(self.config.paths.benchmark_cache_dir),
            )

        raise ValueError(
            f"Unsupported benchmark source '{task.source}' for benchmark '{task.name}'."
        )

    @staticmethod
    def _format_hellaswag_prompt(activity_label: str, context: str) -> str:
        lines = [
            "Select the most plausible continuation for the scenario below.",
            "",
        ]
        if activity_label.strip():
            lines.append(f"Activity: {activity_label.strip()}")
        lines.append(f"Context: {context.strip()}")
        lines.append("")
        lines.append("Ending:")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_mmlu_prompt(question: str, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D"]
        lines = [
            "Answer the multiple-choice question by selecting the single best option.",
            "",
            f"Question: {question.strip()}",
        ]
        for label, choice in zip(labels, choices):
            lines.append(f"{label}. {choice.strip()}")
        lines.append("Answer:")
        return "\n".join(lines)

    def _evaluate_benchmark(
        self,
        *,
        bundle: LoadedModelBundle,
        request: ModelLoadRequest,
        benchmark: LoadedBenchmark,
        source_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        predictions: List[Dict[str, Any]] = []
        correct = 0
        total_prompt_tokens = 0
        total_choice_tokens = 0
        total_forward_passes = 0

        cuda_index = self._cuda_index(bundle.resolved_device)
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
            torch.cuda.reset_peak_memory_stats(cuda_index)

        start = time.perf_counter()
        total_examples = len(benchmark.examples)
        progress_interval = self._resolve_progress_interval(total_examples)
        self._print_benchmark_start(
            request=request,
            benchmark=benchmark,
            total_examples=total_examples,
        )
        for example_index, example in enumerate(benchmark.examples, start=1):
            scored = self._score_example(bundle, example)
            predictions.append(scored)
            correct += int(scored["is_correct"])
            total_prompt_tokens += int(scored["prompt_tokens"])
            total_choice_tokens += int(scored["predicted_choice_tokens"])
            total_forward_passes += int(scored["num_choices"])
            if example_index % progress_interval == 0 or example_index == total_examples:
                self._print_benchmark_progress(
                    request=request,
                    benchmark=benchmark,
                    completed=example_index,
                    total=total_examples,
                    start_time=start,
                )

        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
            max_allocated = round(torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 3), 3)
            max_reserved = round(torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 3), 3)
        else:
            max_allocated = None
            max_reserved = None
        elapsed = time.perf_counter() - start
        accuracy = correct / len(benchmark.examples)

        variant_label = request.quantizer_name or "raw"
        output_paths = self._build_output_paths(
            model_name=request.model_name,
            variant_label=variant_label,
            benchmark_name=benchmark.name,
        )
        summary = {
            "status": "success",
            "model_name": request.model_name,
            "source": request.source,
            "quantizer_name": request.quantizer_name,
            "variant_label": variant_label,
            "benchmark_name": benchmark.name,
            "benchmark_type": benchmark.task_type,
            "dataset_name": benchmark.dataset_name,
            "dataset_config": benchmark.dataset_config,
            "dataset_revision": benchmark.revision,
            "split": benchmark.split,
            "example_count": len(benchmark.examples),
            "correct_count": correct,
            "accuracy": round(accuracy, 6),
            "load_seconds": round(bundle.load_seconds, 3),
            "evaluation_seconds": round(elapsed, 3),
            "examples_per_second": round(len(benchmark.examples) / elapsed, 3) if elapsed > 0 else None,
            "avg_prompt_tokens": round(total_prompt_tokens / len(benchmark.examples), 3),
            "avg_predicted_choice_tokens": round(total_choice_tokens / len(benchmark.examples), 3),
            "total_forward_passes": total_forward_passes,
            "max_cuda_memory_allocated_gb": max_allocated,
            "max_cuda_memory_reserved_gb": max_reserved,
            "system_prompt": self.config.benchmarks.system_prompt,
            "source_metrics": source_metrics,
            "benchmark_metadata": benchmark.metadata,
            "summary_path": str(output_paths["summary"]),
            "predictions_path": str(output_paths["predictions"]),
            "predictions": predictions,
        }
        self._print_benchmark_complete(summary)
        return summary

    def _score_example(
        self,
        bundle: LoadedModelBundle,
        example: BenchmarkExample,
    ) -> Dict[str, Any]:
        prompt_text = build_prompt_text(
            bundle.tokenizer,
            self.config.benchmarks.system_prompt,
            example.prompt,
        )
        choice_scores = self._score_choices(bundle, prompt_text, example.choices)
        predicted_index = max(
            range(len(choice_scores)),
            key=lambda index: choice_scores[index]["normalized_logprob"],
        )
        sorted_scores = sorted(
            (item["normalized_logprob"] for item in choice_scores),
            reverse=True,
        )
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else None
        predicted = choice_scores[predicted_index]
        return {
            "example_id": example.example_id,
            "subject": example.subject,
            "answer_index": example.answer_index,
            "predicted_index": predicted_index,
            "is_correct": predicted_index == example.answer_index,
            "prompt_tokens": choice_scores[0]["prompt_tokens"],
            "predicted_choice_tokens": predicted["choice_tokens"],
            "num_choices": len(example.choices),
            "score_margin": round(margin, 6) if margin is not None else None,
            "choice_scores": [
                {
                    "choice_index": index,
                    "choice_tokens": item["choice_tokens"],
                    "sum_logprob": round(item["sum_logprob"], 6),
                    "normalized_logprob": round(item["normalized_logprob"], 6),
                }
                for index, item in enumerate(choice_scores)
            ],
            "metadata": example.metadata,
        }

    def _score_choices(
        self,
        bundle: LoadedModelBundle,
        prompt_text: str,
        choices: List[str],
    ) -> List[Dict[str, Any]]:
        tokenizer = bundle.tokenizer
        prompt_encoding = tokenizer(prompt_text, return_tensors="pt")
        prompt_token_count = int(prompt_encoding["input_ids"].shape[-1])
        combined_texts = [prompt_text + choice for choice in choices]

        original_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "right"
        try:
            encoded = tokenizer(
                combined_texts,
                return_tensors="pt",
                padding=True,
            )
        finally:
            tokenizer.padding_side = original_padding_side

        input_ids = encoded["input_ids"].to(bundle.resolved_device)
        attention_mask = encoded["attention_mask"].to(bundle.resolved_device)
        with torch.inference_mode():
            logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask).logits
        token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]

        scores: List[Dict[str, Any]] = []
        for row_index in range(input_ids.shape[0]):
            sequence_length = int(attention_mask[row_index].sum().item())
            choice_token_count = sequence_length - prompt_token_count
            if choice_token_count <= 0:
                raise RuntimeError(
                    "Encountered an empty benchmark continuation. "
                    "Prompt formatting must leave room for a choice completion."
                )

            start = prompt_token_count - 1
            stop = sequence_length - 1
            gathered = token_log_probs[row_index, start:stop].gather(
                -1,
                target_ids[row_index, start:stop].unsqueeze(-1),
            ).squeeze(-1)
            sum_logprob = float(gathered.sum().item())
            normalized_logprob = sum_logprob / choice_token_count
            scores.append(
                {
                    "prompt_tokens": prompt_token_count,
                    "choice_tokens": choice_token_count,
                    "sum_logprob": sum_logprob,
                    "normalized_logprob": normalized_logprob,
                }
            )
        return scores

    def _collect_source_metrics(
        self,
        request: ModelLoadRequest,
        source_path: Path,
    ) -> Dict[str, Any]:
        directory_bytes = self._directory_size_bytes(source_path)
        metrics: Dict[str, Any] = {
            "source_path": str(source_path),
            "source_disk_size_bytes": directory_bytes,
        }
        if request.source == "quantized":
            manifest = QuantizedArtifactLoader.load_manifest(source_path)
            size_summary = manifest.get("size_summary", {})
            original_size = size_summary.get("original_size_bytes")
            quantized_size = size_summary.get("quantized_size_bytes")
            metrics.update(
                {
                    "artifact_original_size_bytes": original_size,
                    "artifact_quantized_size_bytes": quantized_size,
                    "artifact_runtime_seconds": manifest.get("runtime_seconds"),
                }
            )
            if original_size and quantized_size:
                metrics["compression_ratio"] = round(original_size / quantized_size, 6)
        return metrics

    @staticmethod
    def _directory_size_bytes(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def _build_error_summary(
        self,
        *,
        model_name: str,
        variant_label: str,
        request: ModelLoadRequest,
        benchmark: LoadedBenchmark,
        error: Exception,
    ) -> Dict[str, Any]:
        output_paths = self._build_output_paths(
            model_name=model_name,
            variant_label=variant_label,
            benchmark_name=benchmark.name,
        )
        return {
            "status": "error",
            "model_name": model_name,
            "source": request.source,
            "quantizer_name": request.quantizer_name,
            "variant_label": variant_label,
            "benchmark_name": benchmark.name,
            "benchmark_type": benchmark.task_type,
            "dataset_name": benchmark.dataset_name,
            "dataset_config": benchmark.dataset_config,
            "dataset_revision": benchmark.revision,
            "split": benchmark.split,
            "error": str(error),
            "summary_path": str(output_paths["summary"]),
            "predictions_path": str(output_paths["predictions"]),
            "predictions": [],
        }

    @classmethod
    def _resolve_progress_interval(cls, total_examples: int) -> int:
        if total_examples <= 0:
            return 1
        return max(1, min(cls.DEFAULT_PROGRESS_INTERVAL, total_examples))

    @staticmethod
    def _print_benchmark_start(
        *,
        request: ModelLoadRequest,
        benchmark: LoadedBenchmark,
        total_examples: int,
    ) -> None:
        variant_label = request.quantizer_name or "raw"
        print(
            f"[benchmark] start model={request.model_name} variant={variant_label} "
            f"benchmark={benchmark.name} examples={total_examples}",
            flush=True,
        )

    @staticmethod
    def _print_benchmark_progress(
        *,
        request: ModelLoadRequest,
        benchmark: LoadedBenchmark,
        completed: int,
        total: int,
        start_time: float,
    ) -> None:
        elapsed = time.perf_counter() - start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        print(
            f"[benchmark] progress model={request.model_name} "
            f"variant={request.quantizer_name or 'raw'} benchmark={benchmark.name} "
            f"{completed}/{total} elapsed={elapsed:.1f}s rate={rate:.2f} ex/s",
            flush=True,
        )

    @staticmethod
    def _print_benchmark_complete(summary: Dict[str, Any]) -> None:
        print(
            f"[benchmark] complete model={summary['model_name']} "
            f"variant={summary['variant_label']} benchmark={summary['benchmark_name']} "
            f"accuracy={summary['accuracy']:.4f} elapsed={summary['evaluation_seconds']:.1f}s",
            flush=True,
        )

    def _maybe_cleanup_quantized_artifact(
        self,
        *,
        request: ModelLoadRequest,
        selection: BenchmarkSelection,
        summaries: List[Dict[str, Any]],
    ) -> None:
        if request.source != "quantized" or not request.quantizer_name:
            return
        if not self.config.benchmarks.delete_quantized_artifacts_after_success:
            return
        if selection.max_examples_per_benchmark is not None:
            return
        if not summaries or any(summary["status"] != "success" for summary in summaries):
            return

        enabled_benchmarks = {
            name for name, task in self.config.benchmarks.tasks.items() if task.enabled
        }
        if set(selection.benchmark_names) != enabled_benchmarks:
            return

        result = self.artifact_cleaner.cleanup_artifact(
            model_name=request.model_name,
            quantizer_name=request.quantizer_name,
            benchmark_names=selection.benchmark_names,
        )
        if result["status"] == "removed":
            print(
                f"[cleanup] removed benchmarked artifact model={request.model_name} "
                f"quantizer={request.quantizer_name} path={result['artifact_dir']}",
                flush=True,
            )

    def _write_benchmark_outputs(self, summary: Dict[str, Any]) -> None:
        summary_path = Path(summary["summary_path"])
        predictions_path = Path(summary["predictions_path"])
        predictions = summary.pop("predictions", [])
        write_json(summary_path, summary)
        write_jsonl(predictions_path, predictions)

    def _build_output_paths(
        self,
        *,
        model_name: str,
        variant_label: str,
        benchmark_name: str,
    ) -> Dict[str, Path]:
        root = (
            self.config.paths.benchmark_results_dir
            / sanitize_name(model_name)
            / sanitize_name(variant_label)
        )
        return {
            "summary": root / f"{sanitize_name(benchmark_name)}.summary.json",
            "predictions": root / f"{sanitize_name(benchmark_name)}.predictions.jsonl",
        }

    @staticmethod
    def _coerce_mmlu_answer(value: Any) -> int:
        if isinstance(value, int):
            return value
        text = str(value).strip().upper()
        if text in {"A", "B", "C", "D"}:
            return "ABCD".index(text)
        if text.isdigit():
            return int(text)
        raise ValueError(f"Unsupported MMLU answer format: {value}")

    @staticmethod
    def _limit_reached(
        examples: List[BenchmarkExample],
        task: BenchmarkTaskConfig,
        max_examples_override: Optional[int],
    ) -> bool:
        limit = max_examples_override if max_examples_override is not None else task.max_examples
        return limit is not None and len(examples) >= limit

    @staticmethod
    def _cuda_index(device: torch.device) -> Optional[int]:
        if device.type != "cuda":
            return None
        return device.index if device.index is not None else torch.cuda.current_device()

    @staticmethod
    def _release_model(bundle: Optional[LoadedModelBundle]) -> None:
        if bundle is None:
            return
        device = bundle.resolved_device
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
