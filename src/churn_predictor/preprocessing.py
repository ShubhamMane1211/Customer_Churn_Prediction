from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "churn"
DROP_COLUMNS = ("customer_id", "credit_card")
CATEGORICAL_COLUMNS = ("country", "gender", "country_active")
REQUIRED_INPUT_COLUMNS = frozenset({"products_number", "country", "active_member"})


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_INPUT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required input columns: {', '.join(sorted(missing))}"
        )

    result = frame.copy()
    result["high_products"] = (result["products_number"] >= 3).astype("int8")
    result["country_active"] = (
        result["country"].astype(str) + "_" + result["active_member"].astype(str)
    )
    return result


def _encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    categories = [column for column in CATEGORICAL_COLUMNS if column in frame.columns]
    return pd.get_dummies(frame, columns=categories, drop_first=True, dtype="int8")


def preprocess_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Training data must contain '{TARGET_COLUMN}'.")

    target = frame[TARGET_COLUMN].astype("int8")
    features = frame.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS], errors="ignore")
    return _encode_features(engineer_features(features)), target


def preprocess_inference_data(
    frame: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    features = frame.drop(columns=DROP_COLUMNS, errors="ignore")
    encoded = _encode_features(engineer_features(features))
    return encoded.reindex(columns=feature_columns, fill_value=0)
