from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_raw_data
from src.features.build_features import FraudFeatureBuilder
from src.models.evaluate import (
    build_error_table,
    calculate_metrics,
    choose_best_f1_threshold,
    save_metrics,
)


TARGET_COLUMN = "isFraud"


def train_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[Pipeline, dict]:
    baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    baseline.fit(x_train, y_train)
    y_score = baseline.predict_proba(x_valid)[:, 1]
    threshold = choose_best_f1_threshold(y_valid, y_score)
    metrics = calculate_metrics(y_valid, y_score, threshold)
    return baseline, metrics


def train_main_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[LGBMClassifier, dict, float, pd.Series]:
    positive_rate = max(float(y_train.mean()), 1e-6)
    scale_pos_weight = max((1.0 - positive_rate) / positive_rate, 1.0)

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_valid)[:, 1]
    threshold = choose_best_f1_threshold(y_valid, y_score)
    metrics = calculate_metrics(y_valid, y_score, threshold)
    importance = pd.Series(model.feature_importances_, index=x_train.columns).sort_values(
        ascending=False
    )
    return model, metrics, threshold, importance


def save_bundle(
    model,
    feature_builder: FraudFeatureBuilder,
    metrics: dict,
    threshold: float,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_builder": feature_builder,
        "metrics": metrics,
        "threshold": threshold,
    }
    joblib.dump(payload, path)


def run_training(data_dir: str | Path, output_dir: str | Path, max_rows: int | None) -> dict:
    df = load_raw_data(data_dir=data_dir, nrows=max_rows)
    if TARGET_COLUMN not in df.columns:
        raise ValueError("В датасете отсутствует целевая колонка isFraud.")

    train_df, valid_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET_COLUMN]
    )

    feature_builder = FraudFeatureBuilder(max_v_features=20)
    x_train = feature_builder.fit_transform(train_df.drop(columns=[TARGET_COLUMN]))
    x_valid = feature_builder.transform(valid_df.drop(columns=[TARGET_COLUMN]))
    y_train = train_df[TARGET_COLUMN]
    y_valid = valid_df[TARGET_COLUMN]

    baseline_model, baseline_metrics = train_baseline(x_train, y_train, x_valid, y_valid)
    model, model_metrics, threshold, feature_importance = train_main_model(
        x_train, y_train, x_valid, y_valid
    )

    output_path = Path(output_dir)
    (output_path / "metrics").mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)
    (output_path / "predictions").mkdir(parents=True, exist_ok=True)

    combined_metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "positive_rate": float(df[TARGET_COLUMN].mean()),
        "baseline": baseline_metrics,
        "lightgbm": model_metrics,
    }
    save_metrics(combined_metrics, output_path / "metrics" / "metrics.json")

    validation_table = build_error_table(
        valid_df.reset_index(drop=True),
        y_valid.reset_index(drop=True),
        model.predict_proba(x_valid)[:, 1],
        threshold,
    )
    validation_table.to_csv(
        output_path / "predictions" / "validation_predictions.csv", index=False
    )

    feature_importance.rename_axis("feature").reset_index(name="importance").to_csv(
        output_path / "metrics" / "feature_importance.csv", index=False
    )

    baseline_path = Path("models/baseline_model.joblib")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": baseline_model, "feature_builder": feature_builder}, baseline_path)

    save_bundle(
        model=model,
        feature_builder=feature_builder,
        metrics=model_metrics,
        threshold=threshold,
        output_path="models/fraud_model.joblib",
    )

    return combined_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IEEE-CIS fraud model.")
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory with train_transaction.csv and train_identity.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for metrics, predictions, and figures",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick experiments",
    )
    args = parser.parse_args()

    metrics = run_training(args.data_dir, args.output_dir, args.max_rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
