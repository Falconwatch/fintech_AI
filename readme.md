# IEEE-CIS Fraud Detection Project

## Project Idea

This repository contains a team project for the NES course assignment on AI in fintech.

We solve a real fintech problem: **fraud detection in online transactions** using machine learning on the public **IEEE-CIS Fraud Detection** dataset.

The business goal is to help a bank or payment provider detect suspicious transactions earlier, reduce fraud losses, and lower the burden on manual review teams.

## Problem Statement

Traditional rule-based antifraud systems often produce too many false positives and miss more complex fraud patterns.

We build an ML pipeline that predicts whether a transaction is fraudulent using:

- transaction features;
- identity and device-related features;
- feature engineering based on behavior, frequency, and rarity signals.

## Dataset

Main dataset: **IEEE-CIS Fraud Detection**

- Competition page: https://www.kaggle.com/competitions/ieee-fraud-detection
- Main files:
  - `train_transaction.csv`
  - `train_identity.csv`

These files are joined by `TransactionID`.

## Repository Structure

```text
fintech_AI/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── one_pager.md
│   ├── slides_outline.md
│   └── video_script.md
├── models/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── app/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── requirements.txt
└── readme.md
```

## Planned Pipeline

1. Load `train_transaction.csv` and `train_identity.csv`.
2. Merge them on `TransactionID`.
3. Clean missing values and prepare categorical features.
4. Train a simple baseline model.
5. Add feature engineering.
6. Train an improved model.
7. Evaluate model quality and analyze errors.
8. Build a small demo for inference.

## Suggested Models

- Baseline: `LogisticRegression`
- Main model: `LightGBM`

This setup makes it easy to show improvement from simple linear modeling to a more practical tabular fraud detection model.

## Recommended Features

### Transaction Features

- `TransactionAmt`
- `ProductCD`
- `card1-card6`
- `addr1`, `addr2`
- `dist1`, `dist2`
- `P_emaildomain`, `R_emaildomain`
- `C1-C14`
- `D1-D15`
- `M1-M9`

### Identity Features

- `DeviceType`
- `DeviceInfo`
- `id_12-id_38`

## Feature Engineering Ideas

- `log(TransactionAmt)`
- frequency encoding for `card`, `addr`, `emaildomain`, `DeviceInfo`
- whether payer and recipient email domains match
- count of transactions per device or card
- average transaction amount by `card1` or `addr1`
- relative time features derived from `TransactionDT`
- missingness indicators for important columns

## Metrics

For fraud detection, we should not rely on accuracy because classes are highly imbalanced.

Primary metrics:

- ROC-AUC
- PR-AUC
- Recall at fixed precision
- Confusion matrix at selected threshold

## Error Analysis

We plan to examine:

- false positives: legitimate transactions incorrectly flagged as fraud;
- false negatives: fraudulent transactions missed by the model.

This is important for the assignment because the project must show model limitations and production risks.

## Risks and Monitoring

Production considerations to discuss in the final presentation:

- data drift;
- concept drift;
- increased false positive rate;
- missing identity data;
- explainability and regulatory constraints.

Suggested monitoring:

- flagged transaction rate;
- fraud capture rate;
- precision of manual review queue;
- drift in top features;
- score distribution stability.

## Demo Idea

A simple demo app can be built with `streamlit`.

Input:

- transaction amount;
- product code;
- card and device-related fields;
- email domains;
- address features.

Output:

- fraud probability;
- decision label: `approve`, `review`, or `block`;
- top factors affecting the prediction.

## Team Deliverables

According to the assignment, we should prepare:

- a video up to 3 minutes;
- repository with working code and launch instructions;
- one-page project description;
- pinned dependency versions;
- project authors and contribution split.

## Run Plan

After data download, the first implementation steps should be:

1. Place Kaggle files into `data/raw/`.
2. Create an EDA notebook.
3. Build the first merged training table.
4. Train a baseline model.
5. Add feature engineering and compare metrics.
6. Prepare charts and screenshots for the video.
