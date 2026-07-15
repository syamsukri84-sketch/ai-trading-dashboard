from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from pandas.errors import EmptyDataError

from src.utils.atomic_io import atomic_write_csv


REQUIRED_PREDICTION_COLUMNS = [
    "timestamp_prediction",
    "ticker",
    "model_name",
    "current_date",
    "target_date",
    "current_price",
    "predicted_price",
]


@dataclass(frozen=True)
class PredictionCsvAudit:
    total_rows: int
    valid_rows: int
    invalid_rows: int
    missing_columns: list[str]
    invalid_reason_counts: dict[str, int]
    invalid_preview: pd.DataFrame

    @property
    def is_clean(self) -> bool:
        return self.invalid_rows == 0 and not self.missing_columns


def read_prediction_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _normal_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def prediction_validity_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if df.empty:
        return pd.Series(dtype=bool), pd.Series(dtype=object)

    reasons = pd.Series("", index=df.index, dtype=object)
    mask = pd.Series(True, index=df.index)

    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in df.columns]
    if missing_columns:
        reasons.loc[:] = "missing_required_columns"
        return pd.Series(False, index=df.index), reasons

    timestamp_ok = pd.to_datetime(df["timestamp_prediction"], errors="coerce").notna()
    current_date_ok = pd.to_datetime(df["current_date"], errors="coerce").notna()
    target_date_ok = pd.to_datetime(df["target_date"], errors="coerce").notna()
    current_price_ok = pd.to_numeric(df["current_price"], errors="coerce").gt(0)
    predicted_price_ok = pd.to_numeric(df["predicted_price"], errors="coerce").gt(0)

    ticker = _normal_text(df["ticker"]).str.upper()
    ticker_ok = ticker.str.match(r"^[A-Z0-9]{2,8}$", na=False)

    model_name = _normal_text(df["model_name"])
    known_bad_models = {"", "TRUE", "SKIP", "NEXT_DAY_DIRECTION", "FINAL", "EVALUATED", "PENDING"}
    model_ok = ~model_name.str.upper().isin(known_bad_models)

    checks = [
        ("invalid_timestamp_prediction", timestamp_ok),
        ("invalid_current_date", current_date_ok),
        ("invalid_target_date", target_date_ok),
        ("invalid_current_price", current_price_ok),
        ("invalid_predicted_price", predicted_price_ok),
        ("invalid_ticker", ticker_ok),
        ("invalid_model_name", model_ok),
    ]
    for reason, ok in checks:
        failed = ~ok
        reasons.loc[failed & (reasons == "")] = reason
        mask &= ok

    if "horizon_days" in df.columns:
        horizon_ok = pd.to_numeric(df["horizon_days"], errors="coerce").between(1, 30)
        failed = ~horizon_ok
        reasons.loc[failed & (reasons == "")] = "invalid_horizon_days"
        mask &= horizon_ok

    if "is_active" in df.columns:
        active_text = _normal_text(df["is_active"]).str.lower()
        active_ok = active_text.isin(["true", "false", "1", "0", "yes", "no"])
        failed = ~active_ok
        reasons.loc[failed & (reasons == "")] = "invalid_is_active"
        mask &= active_ok

    reasons.loc[mask] = "valid"
    return mask, reasons


def audit_prediction_csv(path: str) -> tuple[pd.DataFrame, PredictionCsvAudit]:
    df = read_prediction_csv(path)
    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in df.columns]
    if df.empty:
        audit = PredictionCsvAudit(0, 0, 0, missing_columns, {}, pd.DataFrame())
        return df, audit

    valid_mask, reasons = prediction_validity_mask(df)
    invalid_df = df.loc[~valid_mask].copy()
    if not invalid_df.empty:
        invalid_df.insert(0, "invalid_reason", reasons.loc[~valid_mask])

    audit = PredictionCsvAudit(
        total_rows=int(len(df)),
        valid_rows=int(valid_mask.sum()),
        invalid_rows=int((~valid_mask).sum()),
        missing_columns=missing_columns,
        invalid_reason_counts=reasons.loc[~valid_mask].value_counts().to_dict(),
        invalid_preview=invalid_df.head(50),
    )
    return df.loc[valid_mask].copy(), audit


def clean_prediction_csv(path: str, backup_dir: str | None = None) -> tuple[str, PredictionCsvAudit]:
    clean_df, audit = audit_prediction_csv(path)
    if audit.missing_columns:
        raise ValueError(f"Kolom wajib hilang: {', '.join(audit.missing_columns)}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    backup_dir = backup_dir or os.path.dirname(path)
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"predictions_log_backup_{timestamp}.csv")
    shutil.copy2(path, backup_path)
    atomic_write_csv(clean_df, path, index=False)
    return backup_path, audit
