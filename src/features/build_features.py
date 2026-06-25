import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a small starter set of reusable features."""
    result = df.copy()

    if "TransactionAmt" in result.columns:
        result["TransactionAmtLog"] = np.log1p(result["TransactionAmt"])

    if "TransactionDT" in result.columns:
        result["TransactionHour"] = (result["TransactionDT"] // 3600) % 24
        result["TransactionDay"] = result["TransactionDT"] // (3600 * 24)

    if {"P_emaildomain", "R_emaildomain"}.issubset(result.columns):
        result["EmailDomainMatch"] = (
            result["P_emaildomain"].fillna("missing")
            == result["R_emaildomain"].fillna("missing")
        ).astype(int)

    return result
