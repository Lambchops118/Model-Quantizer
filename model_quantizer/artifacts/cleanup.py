"""Cleanup helpers for benchmarked quantized artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from model_quantizer.utils.filesystem import sanitize_name


class QuantizedArtifactCleaner:
    """Removes quantized artifacts once full benchmark results exist."""

    def __init__(self, quantized_root: Path, benchmark_results_root: Path) -> None:
        self.quantized_root = quantized_root
        self.benchmark_results_root = benchmark_results_root

    def cleanup_ready_artifacts(self, benchmark_names: Iterable[str]) -> List[Dict[str, Any]]:
        """Scan all quantized artifacts and remove the ones fully benchmarked."""

        results: List[Dict[str, Any]] = []
        if not self.quantized_root.exists():
            return results

        normalized_benchmarks = [sanitize_name(name) for name in benchmark_names]
        for model_dir in sorted(path for path in self.quantized_root.iterdir() if path.is_dir()):
            for artifact_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                results.append(
                    self.cleanup_artifact(
                        model_name=model_dir.name,
                        quantizer_name=artifact_dir.name,
                        benchmark_names=normalized_benchmarks,
                    )
                )
        return results

    def cleanup_artifact(
        self,
        *,
        model_name: str,
        quantizer_name: str,
        benchmark_names: Iterable[str],
    ) -> Dict[str, Any]:
        """Remove one artifact when all required benchmark summaries succeeded."""

        artifact_dir = (
            self.quantized_root / sanitize_name(model_name) / sanitize_name(quantizer_name)
        )
        if not artifact_dir.exists():
            return self._result(
                model_name=model_name,
                quantizer_name=quantizer_name,
                artifact_dir=artifact_dir,
                status="skipped",
                reason="artifact_missing",
            )

        ready, reason = self._is_ready_for_cleanup(
            model_name=model_name,
            quantizer_name=quantizer_name,
            benchmark_names=benchmark_names,
        )
        if not ready:
            return self._result(
                model_name=model_name,
                quantizer_name=quantizer_name,
                artifact_dir=artifact_dir,
                status="skipped",
                reason=reason,
            )

        shutil.rmtree(artifact_dir)
        model_dir = artifact_dir.parent
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()
        return self._result(
            model_name=model_name,
            quantizer_name=quantizer_name,
            artifact_dir=artifact_dir,
            status="removed",
            reason="fully_benchmarked",
        )

    def _is_ready_for_cleanup(
        self,
        *,
        model_name: str,
        quantizer_name: str,
        benchmark_names: Iterable[str],
    ) -> tuple[bool, str]:
        benchmark_root = (
            self.benchmark_results_root / sanitize_name(model_name) / sanitize_name(quantizer_name)
        )
        if not benchmark_root.exists():
            return False, "benchmark_results_missing"

        required_benchmarks = list(benchmark_names)
        if not required_benchmarks:
            return False, "no_benchmarks_requested"

        for benchmark_name in required_benchmarks:
            summary_path = benchmark_root / f"{sanitize_name(benchmark_name)}.summary.json"
            if not summary_path.exists():
                return False, f"missing_summary:{benchmark_name}"

            with summary_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if payload.get("status") != "success":
                return False, f"benchmark_not_successful:{benchmark_name}"

            max_examples = (payload.get("benchmark_metadata") or {}).get("max_examples")
            if max_examples is not None:
                return False, f"benchmark_was_limited:{benchmark_name}"

        return True, "fully_benchmarked"

    @staticmethod
    def _result(
        *,
        model_name: str,
        quantizer_name: str,
        artifact_dir: Path,
        status: str,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "model_name": model_name,
            "quantizer_name": quantizer_name,
            "artifact_dir": str(artifact_dir),
            "status": status,
            "reason": reason,
        }
