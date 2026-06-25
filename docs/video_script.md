# 3-Minute Video Script

## 0:00-0:20. Problem and Business Value

Hello. Our project is about fraud detection in online transactions for a bank or payment provider.

The problem is that traditional rule-based antifraud systems are too coarse. They often block normal users and still miss sophisticated fraud. Our goal is to predict suspicious transactions earlier, reduce fraud losses, and make manual review more efficient.

## 0:20-0:50. Data and Architecture

We use the public IEEE-CIS Fraud Detection dataset. It contains transaction features and identity-related information such as device and account signals.

Our pipeline consists of four stages: data merging, preprocessing, feature engineering, and model inference. We combine transaction amount, card and address information, email domains, device signals, and time-derived features into a unified training table.

## 0:50-1:50. Technical Implementation and Demo

First, we build a baseline model. Then we improve it with additional feature engineering and a gradient boosting model.

Important engineered features include transformed transaction amount, frequency-based encodings, time-related signals, and indicators of rare or missing identity attributes.

Here we show the prototype. A new transaction is passed to the model, and the system outputs a fraud probability together with an operational decision: approve, send to manual review, or block. We also display the main factors that influenced the prediction.

## 1:50-2:20. Quality, Errors, and Risks

Since fraud is a rare event, accuracy is not informative. We focus on PR-AUC, ROC-AUC, and recall at a selected threshold.

We also analyze model errors. False positives happen when legitimate user behavior looks unusual. False negatives happen when fraudulent transactions resemble normal ones.

In production, the main risks are data drift, concept drift, incomplete identity information, and excessive false alarms.

## 2:20-3:00. Conclusion

Our main result is that machine learning on transaction and identity features can provide a practical fraud risk score for fintech operations.

The limitations of the project are that we use a public dataset and a simplified offline setup. The next step would be online scoring, richer entity-level aggregation, and continuous monitoring in production.

Thank you. Our team members and their contributions are shown on the final slide.
