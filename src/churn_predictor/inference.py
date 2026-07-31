from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .artifacts import load_feature_columns, load_metadata, load_model
from .preprocessing import preprocess_inference_data


def risk_tier(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= threshold / 2:
        return "Medium"
    return "Low"


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    feature_columns = load_feature_columns()
    threshold = float(load_metadata()["threshold"])
    features = preprocess_inference_data(frame, feature_columns)
    probabilities = model.predict_proba(features)[:, 1]

    result = pd.DataFrame(
        {
            "churn_probability": probabilities.round(4),
            "churn_prediction": (probabilities >= threshold).astype("int8"),
        },
        index=frame.index,
    )
    if "customer_id" in frame:
        result.insert(0, "customer_id", frame["customer_id"])
    result["risk_tier"] = [
        risk_tier(probability, threshold) for probability in probabilities
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict customer churn from a CSV file."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Raw customer CSV path."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Predictions CSV path."
    )
    args = parser.parse_args()

    predictions = predict(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
