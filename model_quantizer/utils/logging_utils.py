"""Per-run logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from model_quantizer.utils.filesystem import sanitize_name


def build_pair_logger(logs_dir: Path, model_name: str, quantizer_name: str) -> tuple[logging.Logger, Path]:
    """Create a dedicated logfile for one (model, quantizer) pair."""

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{sanitize_name(model_name)}__{sanitize_name(quantizer_name)}.log"
    logger_name = f"model_quantizer.{sanitize_name(model_name)}.{sanitize_name(quantizer_name)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path
