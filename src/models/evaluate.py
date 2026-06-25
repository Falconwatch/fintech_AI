from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_best_f1_threshold(y_true: pd.Series, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def calculate_metrics(
    y_true: pd.Series, y_score: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    top_k = max(1, int(len(y_score) * 0.05))
    ranking = np.argsort(-y_score)
    top_true = np.asarray(y_true)[ranking[:top_k]]
    positives = max(1, int(np.sum(y_true)))

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_top_5pct": float(np.mean(top_true)),
        "recall_top_5pct": float(np.sum(top_true) / positives),
        "alert_rate": float(np.mean(y_pred)),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def build_error_table(
    source_df: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    result = source_df.copy()
    result["fraud_probability"] = y_score
    result["predicted_label"] = (y_score >= threshold).astype(int)
    result["actual_label"] = np.asarray(y_true)
    result["error_type"] = "correct"
    result.loc[
        (result["predicted_label"] == 1) & (result["actual_label"] == 0), "error_type"
    ] = "false_positive"
    result.loc[
        (result["predicted_label"] == 0) & (result["actual_label"] == 1), "error_type"
    ] = "false_negative"
    return result


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
