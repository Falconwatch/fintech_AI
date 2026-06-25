from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_bundle(model_path: str | Path = "models/fraud_model.joblib") -> dict:
    return joblib.load(model_path)


def predict_dataframe(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    feature_builder = bundle["feature_builder"]
    model = bundle["model"]
    threshold = bundle["threshold"]

    transformed = feature_builder.transform(df)
    scores = model.predict_proba(transformed)[:, 1]

    result = df.copy()
    result["fraud_probability"] = scores
    result["decision"] = pd.cut(
        scores,
        bins=[-1.0, threshold * 0.6, threshold, 1.0],
        labels=["approve", "review", "block"],
    ).astype(str)
    return result


def predict_payload(payload: dict, bundle: dict) -> dict:
    feature_builder = bundle["feature_builder"]
    threshold = bundle["threshold"]
    df = feature_builder.make_inference_frame(payload)
    prediction = predict_dataframe(df, bundle).iloc[0]
    score = float(prediction["fraud_probability"])
    if score >= threshold:
        decision = "block"
    elif score >= threshold * 0.6:
        decision = "review"
    else:
        decision = "approve"

    return {
        "fraud_probability": score,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fraud scores for a CSV file.")
    parser.add_argument("--model-path", default="models/fraud_model.joblib")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="outputs/predictions/predictions.csv")
    args = parser.parse_args()

    bundle = load_bundle(args.model_path)
    df = pd.read_csv(args.input_csv)
    predictions = predict_dataframe(df, bundle)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    summary = {
        "rows_scored": int(len(predictions)),
        "avg_score": float(predictions["fraud_probability"].mean()),
        "share_block": float((predictions["decision"] == "block").mean()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
