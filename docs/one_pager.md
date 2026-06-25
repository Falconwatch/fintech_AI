# One-Page Project Description

## Project Title

**Fraud detection in online transactions using machine learning on the IEEE-CIS dataset**

## Task

We solve the problem of detecting fraudulent online transactions for a bank or payment provider. The goal is to identify suspicious operations before they are approved, reduce direct fraud losses, and decrease the number of unnecessary manual reviews.

## Why This Matters

Traditional rule-based antifraud systems are often too rigid: they either miss complex fraud patterns or block too many legitimate users. Machine learning makes it possible to combine many weak signals at once and estimate fraud risk more accurately.

## Data

We use the public **IEEE-CIS Fraud Detection** dataset, which includes transaction-level and identity-related features. The dataset contains information about transaction amount, product category, card features, address fields, email domains, device-related signals, and anonymized behavioral variables.

The project uses two main files:

- `train_transaction.csv`
- `train_identity.csv`

They are merged by `TransactionID`.

## Model

We compare two approaches:

- a baseline model with limited preprocessing;
- an improved gradient boosting model with additional feature engineering.

Key feature engineering ideas:

- amount transformation;
- frequency encoding for high-cardinality categories;
- time-based features from `TransactionDT`;
- rarity and count statistics for cards, devices, and domains;
- missingness indicators and simple interaction signals.

## Metrics

Because fraud detection is a highly imbalanced classification problem, we focus on:

- PR-AUC;
- ROC-AUC;
- recall at a business-relevant operating threshold.

We also analyze false positives and false negatives to understand where the model fails.

## Risks and Production View

We discuss how the model would behave in production and what should be monitored:

- data drift and concept drift;
- changes in fraud tactics;
- rising false positive rate;
- incomplete identity information;
- explainability and regulatory concerns.

## Expected Result

We expect the improved model to outperform the baseline and show that transaction and identity signals together can provide a useful fraud risk score for prioritizing suspicious operations.

## Team Contributions

Recommended split:

- business framing and storytelling;
- EDA and feature engineering;
- modeling and evaluation;
- demo, slides, and video assembly.
