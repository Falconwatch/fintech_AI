#!/usr/bin/env bash

set -euo pipefail

if [ ! -x "./.venv/bin/python" ]; then
  echo "Локальное окружение .venv не найдено."
  echo "Сначала создайте и настройте его:"
  echo "python3 -m venv .venv"
  echo "source .venv/bin/activate"
  echo "pip install -r requirements.txt"
  exit 1
fi

./.venv/bin/python train_model.py

required_files=(
  "outputs/metrics/metrics.json"
  "outputs/metrics/feature_importance.csv"
  "outputs/figures/feature_importance.png"
  "outputs/predictions/validation_predictions.csv"
  "models/fraud_model.joblib"
  "models/baseline_model.joblib"
)

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Ошибка: после обучения не найден обязательный артефакт: $file"
    exit 1
  fi
done

if [ -f "docs/submission/model_training_report.md" ]; then
  if grep -q "Этот блок автоматически обновляется после запуска обучения." "docs/submission/model_training_report.md"; then
    echo "Ошибка: блок метрик в docs/submission/model_training_report.md не обновился."
    exit 1
  fi
fi

if [ -f "docs/submission/model_training_report.md" ]; then
  if grep -q "Таблица с наиболее важными признаками автоматически обновляется после запуска обучения." "docs/submission/model_training_report.md"; then
    echo "Ошибка: блок важности признаков в docs/submission/model_training_report.md не обновился."
    exit 1
  fi
fi

echo "Обучение завершено успешно, все ключевые артефакты найдены."
