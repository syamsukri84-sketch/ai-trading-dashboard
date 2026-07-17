"""Tracking harian metrik risiko (volatilitas GARCH + VaR multi-metodologi)
per ticker, ditulis via atomic_write_csv (Prinsip Desain #5).

Ini pelengkap `predictions_log.csv`/`accuracy_log.csv` (accuracy_tracker.py)
-- terpisah karena risk metrics bukan "prediksi" yang perlu dievaluasi
benar/salah, cuma potret risiko harian per ticker. Satu baris per
(ticker, analysis_date): run ulang di hari yang sama meng-upsert (replace),
bukan menumpuk baris duplikat.

Dipakai untuk mengisi `var95_lookup` di `streamlit_app.py::build_daily_decision_board`
supaya stop-loss memakai VaR 95% (metode direkomendasikan dari
src/models/var_analysis.py) alih-alih fallback default 3% flat -- lihat
CATATAN_SESI_VAR_DAN_GATING_2026-07-17.md bagian 3.1 untuk konteks kenapa
ini dibuat."""

import os

import pandas as pd
from pandas.errors import EmptyDataError

from src.utils.atomic_io import atomic_write_csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RISK_METRICS_FILE = os.path.join(PROJECT_ROOT, "data", "tracking", "risk_metrics_log.csv")

COLUMNS = [
    "timestamp_logged",
    "ticker",
    "analysis_date",
    "garch_volatility_pct",
    "garch_var95_pct",
    "var95_recommended_pct",
    "var95_recommended_method",
    "var99_recommended_pct",
    "var99_recommended_method",
    "var_n_obs",
    "var_data_terbatas",
]


def _ensure_dir(filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def log_risk_metrics(
    ticker: str,
    analysis_date: str,
    garch_volatility_pct: float | None,
    garch_var95_pct: float | None,
    var_suite: dict,
    path: str = RISK_METRICS_FILE,
) -> None:
    """Upsert satu baris risk metrics untuk (ticker, analysis_date).

    `var_suite` adalah hasil `compute_var_from_price_df` (lihat
    src/models/var_analysis.py) -- kalau berisi key "error" (data harga
    kurang), baris tetap ditulis dengan kolom VaR kosong (None) supaya
    riwayat "kenapa tidak ada VaR hari ini" tetap terlacak, bukan diam-diam
    dilewati.
    """
    ticker = str(ticker).replace(".JK", "").upper().strip()
    per_conf = var_suite.get("per_confidence", {}) if isinstance(var_suite, dict) else {}
    v95 = per_conf.get(0.95, {})
    v99 = per_conf.get(0.99, {})

    new_row = pd.DataFrame([{
        "timestamp_logged": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "analysis_date": analysis_date,
        "garch_volatility_pct": garch_volatility_pct,
        "garch_var95_pct": garch_var95_pct,
        "var95_recommended_pct": v95.get("recommended_pct"),
        "var95_recommended_method": v95.get("recommended_method"),
        "var99_recommended_pct": v99.get("recommended_pct"),
        "var99_recommended_method": v99.get("recommended_method"),
        "var_n_obs": var_suite.get("n_obs"),
        "var_data_terbatas": var_suite.get("data_terbatas"),
    }])

    _ensure_dir(path)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            existing = pd.read_csv(path)
        except EmptyDataError:
            existing = pd.DataFrame(columns=COLUMNS)
        existing = existing[
            ~((existing["ticker"] == ticker) & (existing["analysis_date"] == analysis_date))
        ]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    atomic_write_csv(combined[COLUMNS], path, index=False)


def load_risk_metrics_log(path: str = RISK_METRICS_FILE) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def get_latest_var95_lookup(path: str = RISK_METRICS_FILE) -> dict:
    """dict {ticker: {"var95_pct": float, "var95_method": str, "data_terbatas": bool}}
    dari baris TERBARU (analysis_date terbesar) per ticker. Ticker tanpa VaR
    valid (None/NaN, mis. data terlalu pendek) TIDAK disertakan -- caller
    treat sebagai "belum tersedia" dan fallback ke default, konsisten dengan
    perilaku volatility_lookup lama."""
    df = load_risk_metrics_log(path)
    if df.empty:
        return {}
    df = df.dropna(subset=["var95_recommended_pct"])
    if df.empty:
        return {}
    df = df.sort_values("analysis_date").drop_duplicates(subset=["ticker"], keep="last")
    return {
        row["ticker"]: {
            "var95_pct": float(row["var95_recommended_pct"]),
            "var95_method": str(row["var95_recommended_method"]),
            "data_terbatas": bool(row["var_data_terbatas"]),
        }
        for _, row in df.iterrows()
    }
