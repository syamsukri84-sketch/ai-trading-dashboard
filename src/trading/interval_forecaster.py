"""Expected-range probabilistik berbasis EWMA volatility (prescriptive layer).

Latar belakang (validasi 2026-07-20, di luar repo ini lalu direplikasi ke sini):
- Interval +/- 1x ATR(14) x sqrt(H) yang umum dipakai ternyata BUKAN "kisaran
  tipikal" -- coverage empirisnya ~85-87% pada LPPF/ACES dan 86.4% pooled pada
  158 ticker likuid universe proyek ini (971 hari, walk-forward).
- EWMA volatility (lambda=0.94, RiskMetrics) memberi interval lebih tajam pada
  coverage yang sama: k=1.3 -> ~82% (interval 80%), k=2.0 -> ~93% (interval 95%).
- Setup arah (pullback/breakout) GAGAL pada uji lintas-universe (expectancy
  pooled -0.47%/trade setelah biaya, CI95 [-0.75, -0.18], 2,378 trade) --
  konsisten dengan temuan proyek: 0/265 ticker punya edge arah. Karena itu modul
  ini SENGAJA hanya memproyeksikan RENTANG (risiko), bukan arah.

Prinsip desain proyek yang dipatuhi:
- Tidak menyentuh bobot model prediksi (lapisan presentasi/risiko murni).
- File tracking ditulis via atomic_write_csv (prinsip #5).
- Saran position sizing nyata HANYA untuk ticker "Terverifikasi Ganda"
  (prinsip #4) -- selain itu berlabel SIMULASI/PAPER.
- Kalibrasi k dibekukan ke data/interval_calibration.csv; perubahan hanya lewat
  monitoring coverage (evaluate_interval_log), bukan re-fit harian.
"""

from __future__ import annotations

import math
import os
from datetime import datetime

import pandas as pd

from src.utils.atomic_io import atomic_write_csv

EWMA_LAMBDA = 0.94
DEFAULT_HORIZON_DAYS = 10
COVERAGE_TOLERANCE_PP = 5.0
MIN_OBS_FOR_MONITORING = 30

# Hasil kalibrasi universe 2026-07-21 via scripts/calibrate_intervals_cli.py
# (158 ticker likuid, 141.019 observasi walk-forward 10 hari):
# k=1.3 -> coverage 82.3% (target 80%), k=2.3 -> coverage 95.1% (target 95%).
# Fallback ini harus dijaga sama dengan data/interval_calibration.csv terbaru.
DEFAULT_K_TABLE = {0.80: 1.3, 0.95: 2.3}

CALIBRATION_FILENAME = os.path.join("data", "interval_calibration.csv")
INTERVAL_LOG_FILENAME = os.path.join("data", "interval_log.csv")


# ---------------------------------------------------------------------------
# Inti perhitungan
# ---------------------------------------------------------------------------

def compute_ewma_sigma_series(close: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """Deret sigma harian EWMA (RiskMetrics). Elemen pertama = NaN."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    ret = close.pct_change()
    var = ret.pow(2).ewm(alpha=1.0 - lam, adjust=False).mean()
    sigma = var.pow(0.5)
    sigma[ret.isna()] = float("nan")
    return sigma


def compute_ewma_sigma(close: pd.Series, lam: float = EWMA_LAMBDA) -> float:
    """Sigma harian EWMA terbaru (float, NaN kalau data < 2 baris)."""
    series = compute_ewma_sigma_series(close, lam=lam)
    if series.dropna().empty:
        return float("nan")
    return float(series.dropna().iloc[-1])


def expected_range(last_close: float, sigma_daily: float, horizon_days: int, k: float):
    """(bawah, atas) = last_close * (1 -/+ k * sigma * sqrt(H)).

    Ini RENTANG kewajaran statistik dengan coverage terkalibrasi -- bukan
    target harga dan bukan sinyal arah.
    """
    if not (last_close and last_close > 0) or not (sigma_daily and sigma_daily > 0):
        return (float("nan"), float("nan"))
    band = k * float(sigma_daily) * math.sqrt(max(int(horizon_days), 1))
    return (last_close * (1.0 - band), last_close * (1.0 + band))


# ---------------------------------------------------------------------------
# Kalibrasi (dibekukan ke CSV) & pemuatan k
# ---------------------------------------------------------------------------

def load_k_table(project_root: str = ".", horizon_days: int | None = None) -> dict:
    """Baca tabel k hasil kalibrasi; fallback DEFAULT_K_TABLE bila belum ada.

    Bila file kalibrasi memuat banyak horizon, `horizon_days` memilih baris
    horizon tersebut (k memang berbeda antar horizon -- scaling sqrt(H) tidak
    persis karena autokorelasi & ketebalan ekor berbeda per horizon). Kalau
    horizon yang diminta tidak ada di file, jatuh ke DEFAULT_K_TABLE.
    """
    path = os.path.join(project_root, CALIBRATION_FILENAME)
    if not os.path.exists(path):
        return dict(DEFAULT_K_TABLE)
    try:
        df = pd.read_csv(path)
        if horizon_days is not None and "horizon_days" in df.columns:
            sub = df[pd.to_numeric(df["horizon_days"], errors="coerce") == int(horizon_days)]
            if sub.empty:
                return dict(DEFAULT_K_TABLE)
            df = sub
        table = {
            float(r["coverage_target"]): float(r["k"])
            for _, r in df.iterrows()
            if pd.notna(r.get("coverage_target")) and pd.notna(r.get("k"))
        }
        return table or dict(DEFAULT_K_TABLE)
    except Exception:
        return dict(DEFAULT_K_TABLE)


def calibrate_k(
    raw_dir: str,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    coverage_targets=(0.80, 0.95),
    k_grid=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.3, 2.6, 3.0),
    min_rows: int = 400,
    min_median_value: float = 1e9,
    warmup_rows: int = 60,
    max_tickers: int | None = None,
    save_to: str | None = None,
    horizons=None,
) -> pd.DataFrame:
    """Kalibrasi k lintas-universe secara walk-forward (point-in-time).

    Untuk tiap hari t: interval dibentuk dari sigma EWMA yang HANYA memakai
    data sampai t, lalu dicek apakah |close[t+H]-close[t]| masuk interval.
    k terpilih = k terkecil pada grid dengan coverage >= target.
    `horizons` (tuple) mengkalibrasi banyak horizon sekaligus dalam satu pass
    data; bila None, hanya `horizon_days`.
    """
    import numpy as np

    horizons = tuple(int(h) for h in (horizons or (horizon_days,)))
    hits = {h: {k: 0 for k in k_grid} for h in horizons}
    total = {h: 0 for h in horizons}
    tickers_used = 0
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith("_raw.csv"))
    if max_tickers:
        files = files[:max_tickers]
    for fname in files:
        try:
            df = pd.read_csv(os.path.join(raw_dir, fname))
        except Exception:
            continue
        if len(df) < min_rows or "close" not in df.columns:
            continue
        close = pd.to_numeric(df["close"], errors="coerce")
        vol = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        if close.isna().any() or (close <= 0).any():
            continue
        if (vol * close).tail(60).median() < min_median_value:
            continue
        sigma = compute_ewma_sigma_series(close)
        tickers_used += 1
        c = close.to_numpy(dtype=float)
        s = sigma.to_numpy(dtype=float)
        n = len(c)
        for h in horizons:
            if n <= warmup_rows + h:
                continue
            idx = np.arange(warmup_rows, n - h)
            base = s[idx] * math.sqrt(h) * c[idx]
            valid = base > 0
            move = np.abs(c[idx + h] - c[idx])[valid]
            base = base[valid]
            total[h] += int(len(base))
            for k in k_grid:
                hits[h][k] += int(np.sum(move <= k * base))
    rows = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for h in horizons:
        for target in coverage_targets:
            chosen = None
            for k in sorted(k_grid):
                if total[h] and hits[h][k] / total[h] >= target:
                    chosen = k
                    break
            if chosen is None:
                chosen = max(k_grid)
            rows.append(
                {
                    "coverage_target": target,
                    "k": chosen,
                    "coverage_realized": round(hits[h][chosen] / total[h], 4) if total[h] else float("nan"),
                    "n_obs": total[h],
                    "tickers_used": tickers_used,
                    "horizon_days": h,
                    "ewma_lambda": EWMA_LAMBDA,
                    "calibrated_at": stamp,
                }
            )
    result = pd.DataFrame(rows)
    if save_to:
        atomic_write_csv(result, save_to, index=False)
    return result


# ---------------------------------------------------------------------------
# Forecast log + monitoring coverage (satu-satunya jalur rekalibrasi yang sah)
# ---------------------------------------------------------------------------

def log_issued_interval(
    ticker: str,
    issue_date: str,
    last_close: float,
    sigma_daily: float,
    horizon_days: int,
    k_table: dict,
    project_root: str = ".",
) -> pd.DataFrame:
    """Catat interval yang diterbitkan ke data/interval_log.csv (atomic).

    Duplikat (ticker+tanggal+horizon) tidak ditulis dua kali.
    """
    path = os.path.join(project_root, INTERVAL_LOG_FILENAME)
    row = {
        "ticker": str(ticker).upper().replace(".JK", ""),
        "issue_date": str(issue_date),
        "last_close": float(last_close),
        "sigma_daily": float(sigma_daily),
        "horizon_days": int(horizon_days),
    }
    for target, k in sorted(k_table.items()):
        low, high = expected_range(last_close, sigma_daily, horizon_days, k)
        pct = int(round(target * 100))
        row[f"k_{pct}"] = k
        row[f"low_{pct}"] = round(low, 4)
        row[f"high_{pct}"] = round(high, 4)
    if os.path.exists(path):
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame()
    if not existing.empty:
        dup = (
            (existing["ticker"] == row["ticker"])
            & (existing["issue_date"] == row["issue_date"])
            & (existing["horizon_days"] == row["horizon_days"])
        )
        if dup.any():
            return existing
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    atomic_write_csv(updated, path, index=False)
    return updated


def evaluate_interval_log(project_root: str = ".", raw_dir: str | None = None) -> pd.DataFrame:
    """Hitung coverage TEREALISASI dari interval yang pernah diterbitkan.

    Hanya baris yang horizonnya sudah lewat (ada harga H hari bursa setelah
    issue_date) yang dievaluasi. Output per level coverage: n, coverage
    realisasi, dan status monitoring.
    """
    log_path = os.path.join(project_root, INTERVAL_LOG_FILENAME)
    raw_dir = raw_dir or os.path.join(project_root, "data", "raw")
    if not os.path.exists(log_path):
        return pd.DataFrame()
    log_df = pd.read_csv(log_path)
    if log_df.empty:
        return pd.DataFrame()
    level_cols = sorted(
        int(c.split("_")[1]) for c in log_df.columns if c.startswith("low_")
    )
    counters = {lvl: [0, 0] for lvl in level_cols}  # lvl -> [hit, total]
    price_cache: dict[str, pd.DataFrame] = {}
    for _, r in log_df.iterrows():
        tk = str(r["ticker"])
        if tk not in price_cache:
            fp = os.path.join(raw_dir, f"{tk}_raw.csv")
            price_cache[tk] = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()
        px = price_cache[tk]
        if px.empty or "timestamp" not in px.columns:
            continue
        idx = px.index[px["timestamp"].astype(str) == str(r["issue_date"])]
        if len(idx) == 0:
            continue
        j = int(idx[0]) + int(r["horizon_days"])
        if j >= len(px):
            continue  # horizon belum lewat
        realized = float(pd.to_numeric(px["close"], errors="coerce").iloc[j])
        for lvl in level_cols:
            counters[lvl][1] += 1
            if float(r[f"low_{lvl}"]) <= realized <= float(r[f"high_{lvl}"]):
                counters[lvl][0] += 1
    rows = []
    for lvl in level_cols:
        hit, tot = counters[lvl]
        cov = hit / tot if tot else float("nan")
        rows.append(
            {
                "coverage_target": lvl / 100.0,
                "n_evaluated": tot,
                "coverage_realized": round(cov, 4) if tot else float("nan"),
                "status": interval_monitoring_status(cov, lvl / 100.0, tot),
            }
        )
    return pd.DataFrame(rows)


def interval_monitoring_status(
    coverage_realized: float,
    coverage_target: float,
    n_evaluated: int,
    tolerance_pp: float = COVERAGE_TOLERANCE_PP,
    min_n: int = MIN_OBS_FOR_MONITORING,
) -> str:
    """OK / RESTRICTED (kalibrasi ulang) / BELUM CUKUP DATA."""
    if n_evaluated < min_n:
        return "BELUM CUKUP DATA"
    if abs(coverage_realized - coverage_target) * 100.0 > tolerance_pp:
        return "RESTRICTED - kalibrasi ulang k"
    return "OK"


# ---------------------------------------------------------------------------
# Position sizing (prescriptive) -- gate "Terverifikasi Ganda" di sisi pemanggil
# ---------------------------------------------------------------------------

def position_size_from_range(
    equity_idr: float,
    risk_budget_pct: float,
    last_close: float,
    sigma_daily: float,
    horizon_days: int,
    k: float,
    is_verified: bool,
    lot_size: int = 100,
) -> dict:
    """Volatility-based sizing: risiko = jarak ke batas bawah interval.

    Size = (risk_budget% x ekuitas) / (last_close - batas_bawah).
    is_verified=False -> mode SIMULASI (paper); angka tetap dihitung supaya
    bisa dipelajari, tapi wajib ditampilkan sebagai simulasi, bukan saran.
    """
    low, _high = expected_range(last_close, sigma_daily, horizon_days, k)
    risk_per_share = last_close - low
    if not (risk_per_share and risk_per_share > 0):
        return {"mode": "TIDAK TERSEDIA", "lots": 0, "shares": 0, "risk_idr": 0.0}
    risk_idr = equity_idr * (risk_budget_pct / 100.0)
    shares = int(risk_idr // risk_per_share)
    lots = shares // lot_size
    return {
        "mode": "SARAN (Terverifikasi Ganda)" if is_verified else "SIMULASI / PAPER",
        "lots": lots,
        "shares": lots * lot_size,
        "risk_per_share": round(risk_per_share, 2),
        "risk_idr": round(min(risk_idr, lots * lot_size * risk_per_share), 2),
        "stop_reference": round(low, 2),
        "position_value_idr": round(lots * lot_size * last_close, 2),
    }
