from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .config import COLUMNS_PATH, METADATA_PATH, MODEL_PATH


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required model artifact was not found: {path}")
    return path


def load_model(path: Path = MODEL_PATH) -> Any:
    return joblib.load(_require_file(path))


def load_feature_columns(path: Path = COLUMNS_PATH) -> list[str]:
    columns = joblib.load(_require_file(path))
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        raise ValueError("Model feature-columns artifact must be a list of strings.")
    return columns


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    with _require_file(path).open(encoding="utf-8") as artifact:
        metadata = json.load(artifact)

    threshold = metadata.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError(
            "Model metadata must contain a numeric threshold between 0 and 1."
        )
    return metadata
