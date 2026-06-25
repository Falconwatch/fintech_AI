from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.build_features import add_basic_features


def train_model(df: pd.DataFrame, target_col: str = "isFraud") -> dict:
    """Train a simple starter LightGBM pipeline."""
    prepared = add_basic_features(df)
    features = prepared.drop(columns=[target_col])
    target = prepared[target_col]

    categorical_cols = features.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in features.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    model.fit(x_train, y_train)
    proba = model.predict_proba(x_valid)[:, 1]

    return {
        "model": model,
        "roc_auc": roc_auc_score(y_valid, proba),
        "pr_auc": average_precision_score(y_valid, proba),
    }


def save_model(model, output_path: str | Path = "models/fraud_model.joblib") -> None:
    """Persist trained model to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
