# Антифрод-проект на IEEE-CIS Fraud Detection

## Что это за проект

Это командный проект под задание РЭШ по теме AI в финтехе.

Мы решаем задачу **выявления мошеннических онлайн-транзакций** для банка или платежного сервиса. Модель получает признаки транзакции и связанные identity/device-сигналы, после чего оценивает риск мошенничества и предлагает одно из решений:

- `approve`;
- `review`;
- `block`.

Проект специально собран так, чтобы закрывать требования задания:

- есть бизнес-постановка;
- есть воспроизводимый ML-пайплайн;
- есть baseline и основная модель;
- есть анализ ошибок и продакшн-риски;
- есть demo для видео;
- есть документы для защиты.

## Бизнес-постановка

Проблема антифрода в том, что простые правила часто работают слишком грубо:

- честные клиенты попадают в лишние блокировки;
- сложные мошеннические сценарии остаются незамеченными;
- команда ручной проверки получает слишком много слабых алертов.

Мы используем ML, чтобы комбинировать множество сигналов сразу и получать более точный fraud score по каждой транзакции.

## Данные

Основной датасет: **IEEE-CIS Fraud Detection**

- Страница соревнования: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
- Основные файлы:
  - `train_transaction.csv`
  - `train_identity.csv`

Таблицы объединяются по `TransactionID`.

## Что реализовано в коде

### Обучение

- загрузка и merge исходных таблиц;
- feature engineering;
- baseline на `LogisticRegression`;
- основная модель на `LightGBM`;
- выбор порога по валидации;
- расчёт метрик;
- сохранение артефактов модели.

### Артефакты после обучения

После запуска пайплайна сохраняются:

- `models/fraud_model.joblib` — основная модель;
- `models/baseline_model.joblib` — baseline;
- `outputs/metrics/metrics.json` — метрики;
- `outputs/metrics/feature_importance.csv` — важность признаков;
- `outputs/predictions/validation_predictions.csv` — предсказания и ошибки на валидации.

### Demo

`streamlit`-интерфейс подключен к сохранённой модели и позволяет показать в видео:

- ввод параметров транзакции;
- вероятность мошенничества;
- решение `approve / review / block`.

## Структура репозитория

```text
fintech_AI/
├── data/
│   ├── raw/
│   └── mock/
├── docs/
│   ├── one_pager.md
│   ├── slides_outline.md
│   ├── submission_checklist.md
│   └── video_script.md
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── app/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── requirements.txt
└── readme.md
```

## Пайплайн признаков

В проекте используются:

- транзакционные признаки: `TransactionAmt`, `TransactionDT`, `ProductCD`, `card*`, `addr*`, `dist*`;
- email-признаки: `P_emaildomain`, `R_emaildomain`;
- поведенческие и агрегатные признаки: `C*`, `D*`, `M*`;
- identity/device-признаки: `DeviceType`, `DeviceInfo`, `id_*`;
- часть `V*`-признаков с наименьшей долей пропусков.

### Feature engineering

Реализованы:

- `log(TransactionAmt)`;
- признаки часа, дня и недели из `TransactionDT`;
- `EmailDomainMatch`;
- `MissingValueCount`;
- frequency encoding для `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `DeviceInfo`;
- агрегаты по сущностям: число транзакций и средняя сумма для `card1`, `addr1`, `DeviceInfo`, `P_emaildomain`;
- индикаторы наличия device-информации.

## Метрики

Так как задача антифрода сильно несбалансирована, акцент сделан не на accuracy, а на:

- `ROC-AUC`;
- `PR-AUC`;
- `precision_at_threshold`;
- `recall_at_threshold`;
- `f1_at_threshold`;
- `precision_top_5pct`;
- `recall_top_5pct`;
- `alert_rate`.

Также формируется таблица ошибок с false positives и false negatives.

## Как запустить

### 1. Установить зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Положить реальные данные IEEE-CIS

Нужно скачать с Kaggle и поместить в `data/raw/`:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

### 3. Обучить модель на реальных данных

```bash
python -m src.models.train --data-dir data/raw --output-dir outputs
```

### 4. Быстрый smoke test без Kaggle-данных

Сначала можно сгенерировать совместимый mock-датасет:

```bash
python -m src.data.generate_mock_data --output-dir data/mock --rows 5000
python -m src.models.train --data-dir data/mock --output-dir outputs/mock_run
```

### 5. Запустить demo

```bash
streamlit run src/app/demo.py
```

## Что проверено

В этом репозитории пайплайн уже доведён до состояния, где его можно прогнать end-to-end на mock-данных. Это позволяет проверить:

- генерацию совместимого датасета;
- обучение baseline и LightGBM;
- сохранение модели;
- расчёт метрик;
- запуск demo с подключённой моделью.

Для финального результата под сдачу нужно отдельно прогнать тот же сценарий уже на настоящих Kaggle-данных IEEE-CIS.

## Документы для защиты

В папке [docs](/Users/igor/Repositories/fintech_AI/docs) уже лежат:

- [one_pager.md](/Users/igor/Repositories/fintech_AI/docs/one_pager.md) — краткое описание проекта;
- [video_script.md](/Users/igor/Repositories/fintech_AI/docs/video_script.md) — сценарий ролика на 4 участников;
- [slides_outline.md](/Users/igor/Repositories/fintech_AI/docs/slides_outline.md) — структура презентации;
- [submission_checklist.md](/Users/igor/Repositories/fintech_AI/docs/submission_checklist.md) — чеклист перед сдачей.

## Роли в команде

Рекомендуемое распределение:

- Участник 1: бизнес-постановка, экономика эффекта, вступление и вывод.
- Участник 2: данные, признаки, архитектура, риски данных.
- Участник 3: модели, метрики, анализ ошибок.
- Участник 4: demo, интерфейс, сборка презентации и видео.

## Ограничения проекта

- В репозитории нет самих Kaggle-данных, их нужно скачать отдельно.
- Итоговые метрики на реальном IEEE-CIS появятся только после локального запуска обучения на `data/raw/`.
- Demo сейчас показывает реальные предсказания только после того, как сохранена `models/fraud_model.joblib`.

## Что ещё нужно сделать перед финальной записью видео

1. Скачать реальные данные с Kaggle.
2. Прогнать обучение на `data/raw/`.
3. Зафиксировать итоговые метрики из `outputs/metrics/metrics.json`.
4. Выбрать 1 false positive и 1 false negative из `validation_predictions.csv`.
5. Снять demo-экран.
6. Собрать слайды по структуре из `docs/slides_outline.md`.
