from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_mock_ieee_data(
    output_dir: str | Path = "data/mock", rows: int = 5000, seed: int = 42
) -> None:
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    transaction_id = np.arange(100000, 100000 + rows)
    fraud = rng.binomial(1, 0.07, rows)

    device_pool = np.array(["Windows", "iOS", "Android", "MacOS", "Linux", "missing"])
    email_pool = np.array(["gmail.com", "yahoo.com", "outlook.com", "bankmail.com", "missing"])
    product_pool = np.array(["W", "C", "R", "H", "S"])
    card4_pool = np.array(["visa", "mastercard", "discover", "american express"])
    card6_pool = np.array(["credit", "debit", "charge"])
    device_type_pool = np.array(["desktop", "mobile"])

    suspicious_device = rng.choice(["Linux", "Android", "missing"], rows)
    chosen_device = np.where(fraud == 1, suspicious_device, rng.choice(device_pool, rows))

    transaction = pd.DataFrame(
        {
            "TransactionID": transaction_id,
            "TransactionDT": rng.integers(0, 60 * 60 * 24 * 180, rows),
            "TransactionAmt": np.where(
                fraud == 1,
                rng.gamma(5, 140, rows),
                rng.gamma(3, 55, rows),
            ),
            "ProductCD": rng.choice(product_pool, rows, p=[0.45, 0.15, 0.15, 0.15, 0.10]),
            "card1": np.where(fraud == 1, rng.integers(1000, 1020, rows), rng.integers(1000, 2000, rows)),
            "card2": rng.integers(100, 600, rows),
            "card3": rng.choice([150, 160, 180, 185], rows),
            "card4": rng.choice(card4_pool, rows),
            "card5": rng.integers(100, 300, rows),
            "card6": rng.choice(card6_pool, rows),
            "addr1": np.where(fraud == 1, rng.integers(100, 120, rows), rng.integers(100, 500, rows)),
            "addr2": rng.choice([50, 60, 70, 80], rows),
            "dist1": rng.gamma(2, 15, rows),
            "dist2": rng.gamma(1.5, 12, rows),
            "P_emaildomain": rng.choice(email_pool, rows, p=[0.35, 0.20, 0.20, 0.10, 0.15]),
            "R_emaildomain": rng.choice(email_pool, rows, p=[0.30, 0.20, 0.20, 0.10, 0.20]),
            "isFraud": fraud,
        }
    )

    for index in range(1, 15):
        transaction[f"C{index}"] = rng.gamma(2 + index * 0.05, 5, rows)
    for index in range(1, 16):
        transaction[f"D{index}"] = rng.normal(20 + index, 10, rows)
    for index in range(1, 10):
        transaction[f"M{index}"] = rng.choice(["T", "F", "missing"], rows, p=[0.45, 0.35, 0.20])
    for index in range(1, 21):
        transaction[f"V{index}"] = rng.normal(0, 1, rows) + fraud * rng.normal(1.5, 0.4, rows)

    identity = pd.DataFrame(
        {
            "TransactionID": transaction_id,
            "DeviceType": np.where(fraud == 1, "mobile", rng.choice(device_type_pool, rows)),
            "DeviceInfo": chosen_device,
        }
    )
    for index in range(12, 39):
        if index % 4 == 0:
            identity[f"id_{index}"] = rng.choice(["Found", "NotFound", "missing"], rows)
        else:
            identity[f"id_{index}"] = rng.normal(index, 3, rows) + fraud * 0.5

    for column in ["dist2", "DeviceInfo", "R_emaildomain", "id_18", "id_27"]:
        if column in transaction.columns:
            mask = rng.random(rows) < 0.25
            transaction.loc[mask, column] = np.nan
        if column in identity.columns:
            mask = rng.random(rows) < 0.25
            identity.loc[mask, column] = np.nan

    transaction.to_csv(output_path / "train_transaction.csv", index=False)
    identity.to_csv(output_path / "train_identity.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mock IEEE-CIS-like dataset.")
    parser.add_argument("--output-dir", default="data/mock")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_mock_ieee_data(args.output_dir, args.rows, args.seed)


if __name__ == "__main__":
    main()
