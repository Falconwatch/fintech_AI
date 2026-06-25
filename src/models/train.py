from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from src.data.load_data import load_raw_data
from src.features.build_features import FraudFeatureBuilder
from src.models.evaluate import (
    build_error_table,
    calculate_metrics,
    choose_best_f1_threshold,
    save_metrics,
)


TARGET_COLUMN = "isFraud"
REPORT_PATH = Path("docs/submission/one_pager.md")
TRAINING_REPORT_PATH = Path("docs/submission/model_training_report.md")
REPORT_START_MARKER = "<!-- METRICS:START -->"
REPORT_END_MARKER = "<!-- METRICS:END -->"
TRAINING_REPORT_START_MARKER = "<!-- TRAINING_METRICS:START -->"
TRAINING_REPORT_END_MARKER = "<!-- TRAINING_METRICS:END -->"
FEATURE_REPORT_START_MARKER = "<!-- FEATURE_IMPORTANCE:START -->"
FEATURE_REPORT_END_MARKER = "<!-- FEATURE_IMPORTANCE:END -->"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_input_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            "Папка с данными не найдена: "
            f"{path}\n"
            "Ожидаются стандартные пути:\n"
            "- data/raw/train_transaction.csv\n"
            "- data/raw/train_identity.csv"
        )

    transaction_path = path / "train_transaction.csv"
    identity_path = path / "train_identity.csv"
    missing_files = [str(file_path.name) for file_path in [transaction_path, identity_path] if not file_path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Не найдены обязательные файлы датасета: "
            f"{', '.join(missing_files)}\n"
            "Положите их в стандартную папку:\n"
            "- data/raw/train_transaction.csv\n"
            "- data/raw/train_identity.csv"
        )

    if transaction_path.stat().st_size == 0 or identity_path.stat().st_size == 0:
        raise ValueError(
            "Файлы датасета найдены, но один из них пустой.\n"
            "Проверьте содержимое data/raw/train_transaction.csv и data/raw/train_identity.csv."
        )

    return path


def train_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[Pipeline, dict]:
    logger.info("Обучаем baseline-модель LogisticRegression")
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
    logger.info("Обучаем основную модель LightGBM")
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


def format_metrics_summary(metrics: dict) -> str:
    baseline = metrics["baseline"]
    lightgbm = metrics["lightgbm"]
    lines = [
        "",
        "=" * 72,
        "ИТОГОВЫЕ МЕТРИКИ ОБУЧЕНИЯ",
        "=" * 72,
        f"Размер датасета: {metrics['dataset_rows']:,}".replace(",", " "),
        f"Train / validation: {metrics['train_rows']:,} / {metrics['validation_rows']:,}".replace(",", " "),
        f"Доля фрода: {metrics['positive_rate']:.4f}",
        "",
        "Baseline (LogisticRegression):",
        f"  ROC-AUC:               {baseline['roc_auc']:.6f}",
        f"  PR-AUC:                {baseline['pr_auc']:.6f}",
        f"  Precision@threshold:   {baseline['precision_at_threshold']:.6f}",
        f"  Recall@threshold:      {baseline['recall_at_threshold']:.6f}",
        f"  F1@threshold:          {baseline['f1_at_threshold']:.6f}",
        f"  Precision top 5%:      {baseline['precision_top_5pct']:.6f}",
        f"  Recall top 5%:         {baseline['recall_top_5pct']:.6f}",
        f"  Alert rate:            {baseline['alert_rate']:.6f}",
        f"  Threshold:             {baseline['threshold']:.6f}",
        "",
        "LightGBM:",
        f"  ROC-AUC:               {lightgbm['roc_auc']:.6f}",
        f"  PR-AUC:                {lightgbm['pr_auc']:.6f}",
        f"  Precision@threshold:   {lightgbm['precision_at_threshold']:.6f}",
        f"  Recall@threshold:      {lightgbm['recall_at_threshold']:.6f}",
        f"  F1@threshold:          {lightgbm['f1_at_threshold']:.6f}",
        f"  Precision top 5%:      {lightgbm['precision_top_5pct']:.6f}",
        f"  Recall top 5%:         {lightgbm['recall_top_5pct']:.6f}",
        f"  Alert rate:            {lightgbm['alert_rate']:.6f}",
        f"  Threshold:             {lightgbm['threshold']:.6f}",
        "",
        "Файлы с результатами:",
        "  - outputs/metrics/metrics.json",
        "  - outputs/metrics/feature_importance.csv",
        "  - outputs/predictions/validation_predictions.csv",
        "=" * 72,
    ]
    return "\n".join(lines)


def save_feature_importance_plot(
    feature_importance: pd.Series,
    output_path: str | Path,
    top_n: int = 10,
) -> None:
    if plt is None:
        logger.warning(
            "matplotlib не установлен, поэтому график feature importance сохранён не будет"
        )
        return

    top_features = feature_importance.head(top_n).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features.index, top_features.values, color="#2f6fed")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top feature importances for LightGBM")
    plt.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def build_report_metrics_block(metrics: dict) -> str:
    baseline = metrics["baseline"]
    lightgbm = metrics["lightgbm"]
    return "\n".join(
        [
            REPORT_START_MARKER,
            "## Текущие результаты обучения",
            "",
            f"- Размер выборки: {metrics['dataset_rows']:,}".replace(",", " "),
            f"- Train / validation: {metrics['train_rows']:,} / {metrics['validation_rows']:,}".replace(",", " "),
            f"- Доля fraud-класса: {metrics['positive_rate']:.4f}",
            "",
            "### Сравнение моделей",
            "",
            "| Модель | ROC-AUC | PR-AUC | Precision | Recall | F1 | Alert rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| LogisticRegression | {baseline['roc_auc']:.4f} | {baseline['pr_auc']:.4f} | {baseline['precision_at_threshold']:.4f} | {baseline['recall_at_threshold']:.4f} | {baseline['f1_at_threshold']:.4f} | {baseline['alert_rate']:.4f} |",
            f"| LightGBM | {lightgbm['roc_auc']:.4f} | {lightgbm['pr_auc']:.4f} | {lightgbm['precision_at_threshold']:.4f} | {lightgbm['recall_at_threshold']:.4f} | {lightgbm['f1_at_threshold']:.4f} | {lightgbm['alert_rate']:.4f} |",
            "",
            "### Интерпретация для отчёта",
            "",
            f"- Основная модель LightGBM достигла `PR-AUC = {lightgbm['pr_auc']:.4f}` и `ROC-AUC = {lightgbm['roc_auc']:.4f}`.",
            f"- На выбранном пороге модель показывает `precision = {lightgbm['precision_at_threshold']:.4f}` и `recall = {lightgbm['recall_at_threshold']:.4f}`.",
            f"- Доля транзакций, попадающих в alert queue, составляет `alert_rate = {lightgbm['alert_rate']:.4f}`.",
            "",
            "Эти значения нужно интерпретировать вместе с примерами false positive и false negative из `validation_predictions.csv`.",
            REPORT_END_MARKER,
        ]
    )


def update_report_with_metrics(metrics: dict, report_path: Path = REPORT_PATH) -> None:
    if not report_path.exists():
        raise FileNotFoundError(f"Не найден файл отчёта: {report_path}")

    content = report_path.read_text(encoding="utf-8")
    block = build_report_metrics_block(metrics)
    if REPORT_START_MARKER in content and REPORT_END_MARKER in content:
        before = content.split(REPORT_START_MARKER)[0].rstrip()
        after = content.split(REPORT_END_MARKER, maxsplit=1)[1].lstrip()
        new_content = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        new_content = content.rstrip() + "\n\n" + block + "\n"
    report_path.write_text(new_content, encoding="utf-8")


def build_training_report_metrics_block(metrics: dict) -> str:
    baseline = metrics["baseline"]
    lightgbm = metrics["lightgbm"]
    return "\n".join(
        [
            TRAINING_REPORT_START_MARKER,
            "## 6. Результаты обучения и метрики",
            "",
            "| Показатель | LogisticRegression | LightGBM |",
            "| --- | ---: | ---: |",
            f"| ROC-AUC | {baseline['roc_auc']:.4f} | {lightgbm['roc_auc']:.4f} |",
            f"| PR-AUC | {baseline['pr_auc']:.4f} | {lightgbm['pr_auc']:.4f} |",
            f"| Precision@threshold | {baseline['precision_at_threshold']:.4f} | {lightgbm['precision_at_threshold']:.4f} |",
            f"| Recall@threshold | {baseline['recall_at_threshold']:.4f} | {lightgbm['recall_at_threshold']:.4f} |",
            f"| F1@threshold | {baseline['f1_at_threshold']:.4f} | {lightgbm['f1_at_threshold']:.4f} |",
            f"| Precision top 5% | {baseline['precision_top_5pct']:.4f} | {lightgbm['precision_top_5pct']:.4f} |",
            f"| Recall top 5% | {baseline['recall_top_5pct']:.4f} | {lightgbm['recall_top_5pct']:.4f} |",
            f"| Alert rate | {baseline['alert_rate']:.4f} | {lightgbm['alert_rate']:.4f} |",
            f"| Рабочий порог | {baseline['threshold']:.4f} | {lightgbm['threshold']:.4f} |",
            "",
            "Дополнительные характеристики запуска:",
            "",
            f"- Размер полной выборки: {metrics['dataset_rows']:,}".replace(",", " "),
            f"- Train: {metrics['train_rows']:,}".replace(",", " "),
            f"- Validation: {metrics['validation_rows']:,}".replace(",", " "),
            f"- Доля fraud-класса: {metrics['positive_rate']:.4f}",
            "",
            "Операционная интерпретация LightGBM:",
            "",
            f"- Модель обнаружила `{lightgbm['true_positives']}` мошеннических транзакций на валидации и пропустила `{lightgbm['false_negatives']}`.",
            f"- Ложных срабатываний: `{lightgbm['false_positives']}`.",
            f"- Recall в топ-5% самых рискованных транзакций: `{lightgbm['recall_top_5pct']:.4f}`.",
            TRAINING_REPORT_END_MARKER,
        ]
    )


def build_feature_importance_block(metrics: dict) -> str:
    top_features = metrics.get("top_features", [])
    feature_rows = [
        FEATURE_REPORT_START_MARKER,
        "| Признак | Importance |",
        "| --- | ---: |",
    ]
    for item in top_features:
        feature_rows.append(f"| `{item['feature']}` | {item['importance']} |")
    feature_rows.extend(
        [
            "",
            "График важности признаков сохраняется в `outputs/figures/feature_importance.png`.",
            FEATURE_REPORT_END_MARKER,
        ]
    )
    return "\n".join(feature_rows)


def update_training_report(metrics: dict, report_path: Path = TRAINING_REPORT_PATH) -> None:
    if not report_path.exists():
        return

    content = report_path.read_text(encoding="utf-8")
    metrics_block = build_training_report_metrics_block(metrics)
    feature_block = build_feature_importance_block(metrics)
    if TRAINING_REPORT_START_MARKER in content and TRAINING_REPORT_END_MARKER in content:
        before = content.split(TRAINING_REPORT_START_MARKER)[0].rstrip()
        after = content.split(TRAINING_REPORT_END_MARKER, maxsplit=1)[1].lstrip()
        new_content = f"{before}\n\n{metrics_block}\n\n{after}".rstrip() + "\n"
    else:
        new_content = content.rstrip() + "\n\n" + metrics_block + "\n"

    if FEATURE_REPORT_START_MARKER in new_content and FEATURE_REPORT_END_MARKER in new_content:
        before = new_content.split(FEATURE_REPORT_START_MARKER)[0].rstrip()
        after = new_content.split(FEATURE_REPORT_END_MARKER, maxsplit=1)[1].lstrip()
        new_content = f"{before}\n\n{feature_block}\n\n{after}".rstrip() + "\n"
    else:
        new_content = new_content.rstrip() + "\n\n" + feature_block + "\n"
    report_path.write_text(new_content, encoding="utf-8")


def ask_to_update_report(metrics: dict) -> None:
    if not sys.stdin.isatty():
        logger.info("Интерактивный вопрос пропущен: stdin не является tty")
        return

    answer = input("Обновить метрики в отчёте? [y/N]: ").strip().lower()
    if answer in {"y", "yes", "д", "да"}:
        update_report_with_metrics(metrics)
        update_training_report(metrics)
        print(f"Отчёт обновлён: {REPORT_PATH}")
        if TRAINING_REPORT_PATH.exists():
            print(f"Подробный отчёт обновлён: {TRAINING_REPORT_PATH}")
    else:
        print("Отчёт не изменён.")


def run_training(data_dir: str | Path, output_dir: str | Path, max_rows: int | None) -> dict:
    validated_data_dir = validate_input_data_dir(data_dir)
    logger.info("Загружаем данные из %s", validated_data_dir)
    df = load_raw_data(data_dir=validated_data_dir, nrows=max_rows)
    if TARGET_COLUMN not in df.columns:
        raise ValueError("В датасете отсутствует целевая колонка isFraud.")

    logger.info("Данные загружены: %s строк, %s колонок", len(df), len(df.columns))
    logger.info("Делим данные на train и validation")
    train_df, valid_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET_COLUMN]
    )

    logger.info("Строим признаки")
    feature_builder = FraudFeatureBuilder(max_v_features=20)
    x_train = feature_builder.fit_transform(train_df.drop(columns=[TARGET_COLUMN]))
    x_valid = feature_builder.transform(valid_df.drop(columns=[TARGET_COLUMN]))
    y_train = train_df[TARGET_COLUMN]
    y_valid = valid_df[TARGET_COLUMN]
    logger.info(
        "Признаки готовы: train=%s x %s, validation=%s x %s",
        x_train.shape[0],
        x_train.shape[1],
        x_valid.shape[0],
        x_valid.shape[1],
    )

    baseline_model, baseline_metrics = train_baseline(x_train, y_train, x_valid, y_valid)
    model, model_metrics, threshold, feature_importance = train_main_model(
        x_train, y_train, x_valid, y_valid
    )

    output_path = Path(output_dir)
    logger.info("Создаём папки для артефактов в %s", output_path)
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
        "top_features": (
            feature_importance.head(10)
            .rename_axis("feature")
            .reset_index(name="importance")
            .to_dict(orient="records")
        ),
    }
    logger.info("Сохраняем метрики")
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
    logger.info("Сохраняем таблицу ошибок на валидации")

    feature_importance.rename_axis("feature").reset_index(name="importance").to_csv(
        output_path / "metrics" / "feature_importance.csv", index=False
    )
    logger.info("Сохраняем feature importance")
    save_feature_importance_plot(
        feature_importance, output_path / "figures" / "feature_importance.png"
    )
    logger.info("Сохраняем график важности признаков")

    baseline_path = Path("models/baseline_model.joblib")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": baseline_model, "feature_builder": feature_builder}, baseline_path)
    logger.info("Сохраняем baseline-модель: %s", baseline_path)

    save_bundle(
        model=model,
        feature_builder=feature_builder,
        metrics=model_metrics,
        threshold=threshold,
        output_path="models/fraud_model.joblib",
    )
    logger.info("Сохраняем основную модель: models/fraud_model.joblib")

    return combined_metrics


def main() -> None:
    configure_logging()
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

    logger.info("Старт обучения антифрод-модели")
    metrics = run_training(args.data_dir, args.output_dir, args.max_rows)
    logger.info("Обучение завершено")
    print(format_metrics_summary(metrics))
    ask_to_update_report(metrics)


if __name__ == "__main__":
    main()
