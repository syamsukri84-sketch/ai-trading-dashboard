"""Test src/trading/interval_forecaster.py -- semua data sintetis, tanpa network."""

import math
import os
import random

import pandas as pd
import pytest

from src.trading.interval_forecaster import (
    DEFAULT_K_TABLE,
    calibrate_k,
    compute_ewma_sigma,
    compute_ewma_sigma_series,
    evaluate_interval_log,
    expected_range,
    interval_monitoring_status,
    load_k_table,
    log_issued_interval,
    position_size_from_range,
)


def _synthetic_prices(n=700, sigma=0.02, seed=7, start=1000.0):
    rng = random.Random(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(max(prices[-1] * (1.0 + rng.gauss(0, sigma)), 1.0))
    return pd.Series(prices)


def test_ewma_sigma_mendekati_sigma_sebenarnya():
    close = _synthetic_prices(n=900, sigma=0.02)
    est = compute_ewma_sigma(close)
    assert 0.012 < est < 0.03  # toleransi longgar, EWMA berbobot data terbaru


def test_ewma_sigma_series_nan_di_awal_dan_positif_setelahnya():
    close = _synthetic_prices(n=50)
    s = compute_ewma_sigma_series(close)
    assert math.isnan(s.iloc[0])
    assert (s.dropna() >= 0).all()


def test_expected_range_melebar_dengan_horizon_dan_k():
    lo5, hi5 = expected_range(1000, 0.02, 5, 1.3)
    lo10, hi10 = expected_range(1000, 0.02, 10, 1.3)
    lo10b, hi10b = expected_range(1000, 0.02, 10, 2.0)
    assert hi5 - lo5 < hi10 - lo10 < hi10b - lo10b
    assert lo5 < 1000 < hi5


def test_expected_range_input_tidak_valid_menghasilkan_nan():
    lo, hi = expected_range(0, 0.02, 10, 1.3)
    assert math.isnan(lo) and math.isnan(hi)
    lo, hi = expected_range(1000, float("nan"), 10, 1.3)
    assert math.isnan(lo) and math.isnan(hi)


def test_calibrate_k_pada_random_walk_mendekati_kuantil_normal(tmp_path):
    # Random walk normal murni: coverage 80% ~ k=1.28, 95% ~ k=1.96.
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        close = _synthetic_prices(n=700, sigma=0.02, seed=i)
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=len(close)).astype(str),
                "close": close,
                "volume": [10_000_000] * len(close),
            }
        ).to_csv(raw / f"TICK{i}_raw.csv", index=False)
    out = tmp_path / "calib.csv"
    res = calibrate_k(
        str(raw),
        horizon_days=10,
        min_rows=400,
        min_median_value=1e6,
        save_to=str(out),
    )
    assert out.exists()
    k80 = float(res[res["coverage_target"] == 0.80]["k"].iloc[0])
    k95 = float(res[res["coverage_target"] == 0.95]["k"].iloc[0])
    assert 1.0 <= k80 <= 1.7
    assert k80 < k95 <= 3.0
    # k terpilih memang memenuhi target pada data kalibrasi
    assert (res["coverage_realized"] >= res["coverage_target"]).all()


def test_load_k_table_fallback_dan_dari_file(tmp_path):
    assert load_k_table(str(tmp_path)) == DEFAULT_K_TABLE
    calib_dir = tmp_path / "data"
    calib_dir.mkdir()
    pd.DataFrame(
        [{"coverage_target": 0.80, "k": 1.4}, {"coverage_target": 0.95, "k": 2.1}]
    ).to_csv(calib_dir / "interval_calibration.csv", index=False)
    table = load_k_table(str(tmp_path))
    assert table == {0.80: 1.4, 0.95: 2.1}


def test_load_k_table_per_horizon(tmp_path):
    calib_dir = tmp_path / "data"
    calib_dir.mkdir()
    pd.DataFrame(
        [
            {"coverage_target": 0.80, "k": 1.2, "horizon_days": 1},
            {"coverage_target": 0.95, "k": 2.0, "horizon_days": 1},
            {"coverage_target": 0.80, "k": 1.3, "horizon_days": 10},
            {"coverage_target": 0.95, "k": 2.3, "horizon_days": 10},
        ]
    ).to_csv(calib_dir / "interval_calibration.csv", index=False)
    assert load_k_table(str(tmp_path), horizon_days=1) == {0.80: 1.2, 0.95: 2.0}
    assert load_k_table(str(tmp_path), horizon_days=10) == {0.80: 1.3, 0.95: 2.3}
    # horizon tak ada di file -> fallback default (bukan campuran antar horizon)
    assert load_k_table(str(tmp_path), horizon_days=7) == DEFAULT_K_TABLE


def test_calibrate_k_multi_horizon(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(2):
        close = _synthetic_prices(n=650, sigma=0.02, seed=20 + i)
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=len(close)).astype(str),
                "close": close,
                "volume": [10_000_000] * len(close),
            }
        ).to_csv(raw / f"MH{i}_raw.csv", index=False)
    res = calibrate_k(
        str(raw), horizons=(1, 5, 10), min_rows=400, min_median_value=1e6
    )
    assert sorted(res["horizon_days"].unique()) == [1, 5, 10]
    assert len(res) == 6  # 3 horizon x 2 target
    assert (res["coverage_realized"] >= res["coverage_target"]).all()


def test_log_issued_interval_append_dan_dedup(tmp_path):
    (tmp_path / "data").mkdir()
    df1 = log_issued_interval("BBCA.JK", "2026-07-20", 10000, 0.015, 10, DEFAULT_K_TABLE, str(tmp_path))
    assert len(df1) == 1 and df1.iloc[0]["ticker"] == "BBCA"
    df2 = log_issued_interval("BBCA", "2026-07-20", 10000, 0.015, 10, DEFAULT_K_TABLE, str(tmp_path))
    assert len(df2) == 1  # duplikat tidak ditulis dua kali
    df3 = log_issued_interval("BBCA", "2026-07-21", 10100, 0.015, 10, DEFAULT_K_TABLE, str(tmp_path))
    assert len(df3) == 2
    assert df3.iloc[0]["low_80"] < 10000 < df3.iloc[0]["high_80"]


def test_evaluate_interval_log_menghitung_coverage(tmp_path):
    (tmp_path / "data").mkdir()
    raw = tmp_path / "data" / "raw"
    raw.mkdir()
    close = _synthetic_prices(n=120, sigma=0.02, seed=3)
    dates = pd.date_range("2026-01-01", periods=len(close)).astype(str)
    pd.DataFrame({"timestamp": dates, "close": close, "volume": [1e7] * len(close)}).to_csv(
        raw / "SYN_raw.csv", index=False
    )
    sigma = compute_ewma_sigma(close.iloc[:60])
    for i in range(40, 70):
        log_issued_interval("SYN", dates[i], float(close.iloc[i]), sigma, 10, DEFAULT_K_TABLE, str(tmp_path))
    res = evaluate_interval_log(str(tmp_path))
    assert not res.empty
    row80 = res[res["coverage_target"] == 0.80].iloc[0]
    assert row80["n_evaluated"] == 30
    assert 0.0 <= row80["coverage_realized"] <= 1.0
    assert row80["status"] in ("OK", "RESTRICTED - kalibrasi ulang k")


def test_interval_monitoring_status_ambang():
    assert interval_monitoring_status(0.80, 0.80, 10) == "BELUM CUKUP DATA"
    assert interval_monitoring_status(0.82, 0.80, 50) == "OK"
    assert interval_monitoring_status(0.70, 0.80, 50).startswith("RESTRICTED")


def test_position_size_gate_dan_matematika():
    saran = position_size_from_range(20_000_000, 1.0, 1000, 0.02, 10, 1.3, is_verified=True)
    paper = position_size_from_range(20_000_000, 1.0, 1000, 0.02, 10, 1.3, is_verified=False)
    assert saran["mode"].startswith("SARAN")
    assert paper["mode"] == "SIMULASI / PAPER"
    assert saran["lots"] == paper["lots"] > 0
    # risiko total tidak melebihi budget 1%
    assert saran["risk_idr"] <= 200_000 + 1e-6
    bad = position_size_from_range(20_000_000, 1.0, 1000, float("nan"), 10, 1.3, True)
    assert bad["lots"] == 0 and bad["mode"] == "TIDAK TERSEDIA"
