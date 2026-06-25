from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


TRANSACTION_CANDIDATES = [
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
]

C_COLUMNS = [f"C{i}" for i in range(1, 15)]
D_COLUMNS = [f"D{i}" for i in range(1, 16)]
M_COLUMNS = [f"M{i}" for i in range(1, 10)]
ID_COLUMNS = [f"id_{i}" for i in range(12, 39)]

FREQUENCY_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
]

AGGREGATION_COLUMNS = ["card1", "addr1", "DeviceInfo", "P_emaildomain"]


def _safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0))


@dataclass
class FraudFeatureBuilder:
    """Feature builder for the IEEE-CIS fraud dataset."""

    max_v_features: int = 20
    selected_columns: list[str] = field(default_factory=list)
    selected_v_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    frequency_maps: dict[str, dict] = field(default_factory=dict)
    aggregation_maps: dict[str, dict[str, dict]] = field(default_factory=dict)
    category_maps: dict[str, dict] = field(default_factory=dict)
    numeric_fill_values: dict[str, float] = field(default_factory=dict)
    input_defaults: dict[str, object] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "FraudFeatureBuilder":
        self.selected_columns = self._select_columns(df)
        self.selected_v_columns = self._select_v_columns(df)

        working = self._build_dataframe(df, fit_mode=True)
        self.categorical_columns = [
            col
            for col in working.columns
            if working[col].dtype == "object" or str(working[col].dtype) == "category"
        ]
        self.numeric_columns = [
            col for col in working.columns if col not in self.categorical_columns
        ]

        self.frequency_maps = {}
        for col in [c for c in FREQUENCY_COLUMNS if c in df.columns]:
            freq = df[col].fillna("missing").astype(str).value_counts(dropna=False)
            self.frequency_maps[col] = freq.to_dict()

        self.aggregation_maps = {}
        if "TransactionAmt" in df.columns:
            for col in [c for c in AGGREGATION_COLUMNS if c in df.columns]:
                grouped = df.groupby(col, dropna=False)["TransactionAmt"]
                self.aggregation_maps[col] = {
                    "count": grouped.count().to_dict(),
                    "mean": grouped.mean().to_dict(),
                }

        self.category_maps = {}
        for col in self.categorical_columns:
            categories = (
                working[col].fillna("missing").astype(str).value_counts().index.tolist()
            )
            self.category_maps[col] = {
                value: index for index, value in enumerate(categories, start=0)
            }

        self.numeric_fill_values = {}
        for col in self.numeric_columns:
            median = pd.to_numeric(working[col], errors="coerce").median()
            self.numeric_fill_values[col] = float(0.0 if pd.isna(median) else median)

        self.input_defaults = {}
        for col in self.selected_columns:
            if col in df.columns:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    median = series.median()
                    self.input_defaults[col] = (
                        float(0.0 if pd.isna(median) else median)
                    )
                else:
                    mode = series.mode(dropna=True)
                    self.input_defaults[col] = (
                        "missing" if mode.empty else str(mode.iloc[0])
                    )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        working = self._build_dataframe(df, fit_mode=False)

        for col in self.categorical_columns:
            mapped = (
                working[col]
                .fillna("missing")
                .astype(str)
                .map(self.category_maps.get(col, {}))
                .fillna(-1)
            )
            working[col] = mapped.astype(float)

        for col in self.numeric_columns:
            working[col] = pd.to_numeric(working[col], errors="coerce").fillna(
                self.numeric_fill_values.get(col, 0.0)
            )

        return working[self.numeric_columns + self.categorical_columns]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def make_inference_frame(self, payload: dict[str, object]) -> pd.DataFrame:
        row = {column: self.input_defaults.get(column, "missing") for column in self.selected_columns}
        row.update(payload)
        return pd.DataFrame([row])

    def _select_columns(self, df: pd.DataFrame) -> list[str]:
        candidates = TRANSACTION_CANDIDATES + C_COLUMNS + D_COLUMNS + M_COLUMNS
        candidates += [col for col in ID_COLUMNS if col in df.columns]
        if "DeviceType" in df.columns:
            candidates.append("DeviceType")
        if "DeviceInfo" in df.columns:
            candidates.append("DeviceInfo")
        columns = [col for col in candidates if col in df.columns]
        columns.extend(self._select_v_columns(df))
        return list(dict.fromkeys(columns))

    def _select_v_columns(self, df: pd.DataFrame) -> list[str]:
        v_columns = [col for col in df.columns if col.startswith("V")]
        if not v_columns:
            return []
        non_null_share = df[v_columns].notna().mean().sort_values(ascending=False)
        return non_null_share.head(self.max_v_features).index.tolist()

    def _build_dataframe(self, df: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        selected = self.selected_columns if self.selected_columns else self._select_columns(df)
        working = df.loc[:, [col for col in selected if col in df.columns]].copy()

        if "TransactionAmt" in working.columns:
            working["TransactionAmtLog"] = _safe_log1p(working["TransactionAmt"])

        if "TransactionDT" in working.columns:
            working["TransactionHour"] = ((working["TransactionDT"] // 3600) % 24).astype(
                "float64"
            )
            working["TransactionDay"] = (
                working["TransactionDT"] // (3600 * 24)
            ).astype("float64")
            working["TransactionWeek"] = (
                working["TransactionDT"] // (3600 * 24 * 7)
            ).astype("float64")

        if {"P_emaildomain", "R_emaildomain"}.issubset(working.columns):
            working["EmailDomainMatch"] = (
                working["P_emaildomain"].fillna("missing").astype(str)
                == working["R_emaildomain"].fillna("missing").astype(str)
            ).astype(int)

        working["MissingValueCount"] = working.isna().sum(axis=1).astype("float64")

        if "DeviceInfo" in working.columns:
            working["HasDeviceInfo"] = working["DeviceInfo"].notna().astype(int)
        if "DeviceType" in working.columns:
            working["HasDeviceType"] = working["DeviceType"].notna().astype(int)

        for col in [c for c in FREQUENCY_COLUMNS if c in df.columns]:
            freq_map = self.frequency_maps.get(col)
            if fit_mode and freq_map is None:
                freq_map = df[col].fillna("missing").astype(str).value_counts().to_dict()
            if freq_map is not None:
                working[f"{col}_freq"] = (
                    df[col].fillna("missing").astype(str).map(freq_map).fillna(0.0)
                )

        for col in [c for c in AGGREGATION_COLUMNS if c in df.columns]:
            stats = self.aggregation_maps.get(col)
            if fit_mode and not stats and "TransactionAmt" in df.columns:
                grouped = df.groupby(col, dropna=False)["TransactionAmt"]
                stats = {
                    "count": grouped.count().to_dict(),
                    "mean": grouped.mean().to_dict(),
                }
            if stats is not None:
                keys = df[col]
                working[f"{col}_tx_count"] = keys.map(stats["count"]).fillna(0.0)
                working[f"{col}_amt_mean"] = keys.map(stats["mean"]).fillna(0.0)

        for col in working.columns:
            if pd.api.types.is_object_dtype(working[col]) or str(working[col].dtype) == "category":
                working[col] = working[col].fillna("missing").astype(str)

        return working
