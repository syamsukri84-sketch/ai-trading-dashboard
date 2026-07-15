"""Deteksi & riwayat regime pasar (REBOUND/BEARISH/MIXED) dari market breadth.

Regime dihitung dari persentase ticker dengan return harian positif
(breadth) -- sinyal yang sama yang dipakai `build_daily_decision_board` di
streamlit_app.py untuk memfilter sinyal TURUN saat market rebound. Modul ini
menambahkan PERSISTENSI (riwayat harian) supaya dashboard bisa bilang "sudah
N hari di regime ini", bukan cuma snapshot hari ini -- lihat
ROADMAP_COGNITIVE_DASHBOARD.md Bagian B3.
"""

import os
from datetime import datetime

import pandas as pd
from pandas.errors import EmptyDataError

from src.utils.atomic_io import atomic_write_csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGIME_HISTORY_FILE = os.path.join(PROJECT_ROOT, "data", "regime_history.csv")
REGIME_HISTORY_COLUMNS = [
    "date", "market_regime", "breadth_up_pct", "avg_latest_return_pct", "sample_size", "logged_at",
]


def compute_market_breadth(tickers, raw_dir=None):
    """Menghitung breadth pasar (% ticker naik) dari harga penutupan terbaru
    di data/raw/*.csv. Logika identik dengan yang sebelumnya ada di dalam
    `build_daily_decision_board` (streamlit_app.py) -- disatukan di sini
    supaya bisa dipakai ulang dari CLI (scripts/daily_global_workflow_cli.py)
    maupun dashboard, tanpa duplikasi.
    """
    raw_dir = raw_dir or os.path.join(PROJECT_ROOT, "data", "raw")
    rows = []
    for ticker_code in tickers:
        raw_path = os.path.join(raw_dir, f"{ticker_code}_raw.csv")
        if not os.path.exists(raw_path):
            continue
        try:
            price_df = pd.read_csv(raw_path)
        except Exception:
            continue
        close_col = "close" if "close" in price_df.columns else "Close" if "Close" in price_df.columns else None
        if close_col is None or len(price_df) < 2:
            continue
        closes = pd.to_numeric(price_df[close_col], errors="coerce").dropna()
        if len(closes) < 2:
            continue
        prev_close = float(closes.iloc[-2])
        last_close = float(closes.iloc[-1])
        if prev_close <= 0:
            continue
        rows.append((last_close / prev_close - 1.0) * 100.0)

    if not rows:
        return {
            "breadth_up_pct": 0.0,
            "avg_latest_return_pct": 0.0,
            "market_regime": "UNKNOWN",
            "sample_size": 0,
        }
    returns = pd.Series(rows)
    breadth_up_pct = float((returns > 0).mean() * 100.0)
    avg_latest_return_pct = float(returns.mean())
    if breadth_up_pct >= 60.0 and avg_latest_return_pct > 0:
        market_regime = "REBOUND"
    elif breadth_up_pct <= 40.0 and avg_latest_return_pct < 0:
        market_regime = "BEARISH"
    else:
        market_regime = "MIXED"
    return {
        "breadth_up_pct": round(breadth_up_pct, 2),
        "avg_latest_return_pct": round(avg_latest_return_pct, 2),
        "market_regime": market_regime,
        "sample_size": int(len(returns)),
    }


def load_regime_history(path: str = REGIME_HISTORY_FILE) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=REGIME_HISTORY_COLUMNS)
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=REGIME_HISTORY_COLUMNS)


def log_regime_snapshot(breadth_result: dict, date: str | None = None, path: str = REGIME_HISTORY_FILE) -> pd.DataFrame:
    """Mencatat SATU baris regime per hari kalender (dedup by date).

    Aman dipanggil berkali-kali di hari yang sama (mis. dashboard dibuka
    berulang, atau workflow CLI + dashboard sama-sama memanggilnya) --
    baris lama untuk tanggal yang sama diganti, bukan ditumpuk duplikat.
    """
    if breadth_result.get("market_regime") == "UNKNOWN":
        return load_regime_history(path)

    date = date or datetime.now().strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_row = pd.DataFrame([{
        "date": date,
        "market_regime": breadth_result["market_regime"],
        "breadth_up_pct": breadth_result["breadth_up_pct"],
        "avg_latest_return_pct": breadth_result["avg_latest_return_pct"],
        "sample_size": breadth_result["sample_size"],
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    existing = load_regime_history(path)
    if not existing.empty:
        existing = existing[existing["date"] != date]
    updated = pd.concat([existing, new_row], ignore_index=True).sort_values("date").reset_index(drop=True)
    atomic_write_csv(updated, path, index=False)
    return updated


def summarize_regime_streaks(history_df: pd.DataFrame) -> dict:
    """Menghitung durasi streak regime saat ini dan rata-rata durasi historis
    per jenis regime -- dasar untuk kalimat "sudah N hari di regime ini,
    historisnya rata-rata bertahan ~M hari"."""
    if history_df.empty:
        return {"current_regime": None, "current_streak_days": 0, "avg_duration_by_regime": {}, "streaks": pd.DataFrame()}

    df = history_df.sort_values("date").reset_index(drop=True)
    streak_id = (df["market_regime"] != df["market_regime"].shift()).cumsum()
    streaks = df.groupby(streak_id).agg(
        market_regime=("market_regime", "first"),
        streak_days=("date", "count"),
        start_date=("date", "first"),
        end_date=("date", "last"),
    ).reset_index(drop=True)

    avg_duration_by_regime = streaks.groupby("market_regime")["streak_days"].mean().round(1).to_dict()
    current_regime = df["market_regime"].iloc[-1]
    current_streak_days = int(streaks.iloc[-1]["streak_days"])

    return {
        "current_regime": current_regime,
        "current_streak_days": current_streak_days,
        "avg_duration_by_regime": avg_duration_by_regime,
        "streaks": streaks,
    }
