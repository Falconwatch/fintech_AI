from pathlib import Path

import pandas as pd


def load_raw_data(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load and merge IEEE-CIS transaction and identity tables."""
    data_path = Path(data_dir)
    transactions = pd.read_csv(data_path / "train_transaction.csv")
    identity = pd.read_csv(data_path / "train_identity.csv")
    return transactions.merge(identity, on="TransactionID", how="left")
