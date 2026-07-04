from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd


LEAKAGE_KEYWORDS = (
    "target",
    "label",
    "future",
    "actual",
    "tomorrow",
    "next_day",
    "predicted",
    "prediction",
)


@dataclass
class GuardrailResult:
    passed: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def audit_ohlcv_data(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    as_of: datetime | None = None,
    check_price_bounds: bool = True,
    price_tolerance: float = 1e-6,
) -> GuardrailResult:
    errors: list[str] = []
    warnings: list[str] = []
    required = ["timestamp", "open", "high", "low", "close", "volume"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"{ticker}: kolom wajib hilang: {', '.join(missing)}")
        return GuardrailResult(False, errors, warnings)

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    if data["timestamp"].isna().any():
        errors.append(f"{ticker}: ada timestamp yang tidak valid.")
    if not data["timestamp"].is_monotonic_increasing:
        errors.append(f"{ticker}: timestamp tidak berurutan naik.")
    if not data["timestamp"].is_unique:
        errors.append(f"{ticker}: timestamp duplikat ditemukan.")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[numeric_cols].isna().any().any():
        errors.append(f"{ticker}: ada nilai OHLCV yang kosong atau bukan angka.")
    if (data[numeric_cols] < 0).any().any():
        errors.append(f"{ticker}: ada nilai OHLCV negatif.")
    if check_price_bounds and (data["high"] + price_tolerance < data[["open", "close"]].max(axis=1)).any():
        errors.append(f"{ticker}: ada high yang lebih rendah dari open/close.")
    if check_price_bounds and (data["low"] - price_tolerance > data[["open", "close"]].min(axis=1)).any():
        errors.append(f"{ticker}: ada low yang lebih tinggi dari open/close.")

    if as_of is not None and not data.empty:
        latest_ts = data["timestamp"].max()
        if latest_ts > pd.Timestamp(as_of):
            errors.append(f"{ticker}: data berisi timestamp masa depan: {latest_ts.date()}.")

    zero_volume_ratio = (data["volume"] == 0).mean() if not data.empty else 0.0
    if zero_volume_ratio > 0.2:
        warnings.append(f"{ticker}: lebih dari 20% baris memiliki volume 0.")

    return GuardrailResult(not errors, errors, warnings)


def audit_feature_leakage(
    features_df: pd.DataFrame,
    allowed_non_feature_columns: Iterable[str] = ("timestamp", "open", "high", "low", "close", "volume"),
) -> GuardrailResult:
    errors: list[str] = []
    warnings: list[str] = []

    suspicious = [
        col
        for col in features_df.columns
        if any(keyword in col.lower() for keyword in LEAKAGE_KEYWORDS)
    ]
    if suspicious:
        errors.append("Kolom berpotensi leakage ditemukan: " + ", ".join(sorted(suspicious)))

    feature_cols = [col for col in features_df.columns if col.startswith("feat_")]
    non_feature_cols = set(allowed_non_feature_columns)
    unexpected_numeric = [
        col
        for col in _numeric_columns(features_df)
        if not col.startswith("feat_") and col not in non_feature_cols
    ]
    if unexpected_numeric:
        warnings.append("Kolom numerik non-fitur ikut terbawa: " + ", ".join(sorted(unexpected_numeric)))

    if features_df[feature_cols].isna().any().any() if feature_cols else False:
        errors.append("Fitur mengandung NaN.")
    if feature_cols and not features_df[feature_cols].apply(pd.api.types.is_numeric_dtype).all():
        errors.append("Ada kolom fitur yang bukan numerik.")

    return GuardrailResult(not errors, errors, warnings)


def assert_no_training_leakage(
    raw_df: pd.DataFrame,
    features_df: pd.DataFrame,
    ticker: str,
    prediction_date: str | datetime | None = None,
) -> GuardrailResult:
    errors: list[str] = []
    warnings: list[str] = []

    raw_audit = audit_ohlcv_data(raw_df, ticker=ticker)
    errors.extend(raw_audit.errors)
    warnings.extend(raw_audit.warnings)

    feature_audit = audit_feature_leakage(features_df)
    errors.extend(feature_audit.errors)
    warnings.extend(feature_audit.warnings)

    if prediction_date is not None and "timestamp" in features_df.columns and not features_df.empty:
        pred_ts = pd.to_datetime(prediction_date, errors="coerce")
        max_feature_ts = pd.to_datetime(features_df["timestamp"], errors="coerce").max()
        if pd.notna(pred_ts) and pd.notna(max_feature_ts) and max_feature_ts > pred_ts:
            errors.append(
                f"{ticker}: fitur training berisi data setelah tanggal prediksi "
                f"({max_feature_ts.date()} > {pred_ts.date()})."
            )

    return GuardrailResult(not errors, errors, warnings)
