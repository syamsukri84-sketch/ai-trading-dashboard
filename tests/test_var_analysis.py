"""Test untuk src/models/var_analysis.py -- VaR multi-metodologi (historical/
parametric/Cornish-Fisher/EWMA/Monte Carlo bootstrap) + pemilihan metode
direkomendasikan per confidence level.

Beberapa test memakai data harga sungguhan proyek ini (data/raw/LPPF_raw.csv,
bersumber yfinance lewat src/data_pipeline/auto_updater.py) sebagai regresi
terhadap angka yang sudah diverifikasi manual (lihat sesi analisis LPPF/MDS
Retailing, trailing 252 hari s.d. 2026-07-16) -- bukan data sintetis, supaya
test ini juga jadi bukti bahwa modul bekerja benar pada data pasar nyata."""

import os

import numpy as np
import pandas as pd
import pytest

from src.models.var_analysis import (
    MIN_OBS_FLOOR,
    _cornish_fisher_z,
    compute_var_from_price_df,
    compute_var_suite,
)

LPPF_RAW_PATH = os.path.join("data", "raw", "LPPF_raw.csv")


def _synthetic_normal_returns(n=500, mean=0.0005, std=0.015, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def test_cornish_fisher_z_zero_skew_kurtosis_returns_z_unchanged():
    z = -1.6448536269514722  # norm.ppf(0.05)
    assert _cornish_fisher_z(z, skew=0.0, excess_kurt=0.0) == pytest.approx(z, abs=1e-9)


def test_cornish_fisher_z_positive_kurtosis_widens_deep_tail_more_than_shallow_tail():
    # Suku koreksi kurtosis skala kubik terhadap z -- efeknya harus lebih besar
    # (secara absolut) di confidence 99% (z lebih negatif) daripada di 95%.
    from scipy import stats

    z95, z99 = stats.norm.ppf(0.05), stats.norm.ppf(0.01)
    delta95 = abs(_cornish_fisher_z(z95, 0.0, 2.8) - z95)
    delta99 = abs(_cornish_fisher_z(z99, 0.0, 2.8) - z99)
    assert delta99 > delta95


def test_compute_var_suite_insufficient_data_returns_error():
    result = compute_var_suite(pd.Series([0.01, -0.02, 0.005]))
    assert "error" in result
    assert result["n_obs"] == 3


def test_compute_var_suite_flat_returns_zero_var_no_crash():
    flat = pd.Series([0.0] * 20)
    result = compute_var_suite(flat)
    assert result["per_confidence"][0.95]["recommended_pct"] == 0.0
    assert "catatan_data" in result


def test_compute_var_suite_flags_data_terbatas_below_threshold():
    returns = _synthetic_normal_returns(n=MIN_OBS_FLOOR + 5)
    result = compute_var_suite(returns)
    assert result["data_terbatas"] is True
    assert "catatan_data" in result


def test_compute_var_suite_normal_data_parametric_matches_theory():
    from scipy import stats

    returns = _synthetic_normal_returns(n=2000, mean=0.0, std=0.02, seed=7)
    result = compute_var_suite(returns, confidence_levels=(0.95,), window=2000)
    z95 = stats.norm.ppf(0.05)
    expected_pct = -(returns.mean() + z95 * returns.std(ddof=1)) * 100
    assert result["per_confidence"][0.95]["parametric_pct"] == pytest.approx(expected_pct, rel=0.02)


def test_compute_var_suite_recommended_method_switches_at_99pct():
    returns = _synthetic_normal_returns(n=300, seed=3)
    result = compute_var_suite(returns, confidence_levels=(0.95, 0.99))
    assert result["per_confidence"][0.95]["recommended_method"] == "cornish_fisher"
    assert result["per_confidence"][0.99]["recommended_method"] == "historical_mc_bootstrap_avg"


def test_compute_var_suite_horizon_scaling_applies_sqrt_rule():
    returns = _synthetic_normal_returns(n=300, seed=5)
    r1 = compute_var_suite(returns, confidence_levels=(0.95,), horizon_days=1)
    r10 = compute_var_suite(returns, confidence_levels=(0.95,), horizon_days=10)
    ratio = r10["per_confidence"][0.95]["parametric_pct"] / r1["per_confidence"][0.95]["parametric_pct"]
    assert ratio == pytest.approx(np.sqrt(10), rel=1e-6)
    assert "catatan_horizon" in r10


def test_compute_var_from_price_df_uses_close_pct_change():
    df = pd.DataFrame({"close": [100.0, 101.0, 99.0, 102.0, 100.0] * 60})
    result = compute_var_from_price_df(df, window=250)
    expected_n = len(df) - 1
    assert result["n_obs"] == min(expected_n, 250)


@pytest.mark.skipif(not os.path.exists(LPPF_RAW_PATH), reason="data/raw/LPPF_raw.csv tidak ada di environment ini")
def test_compute_var_from_lppf_raw_matches_manually_verified_values():
    """Regresi terhadap angka yang sudah diverifikasi manual (trailing 252 hari
    s.d. 2026-07-16): mean~0.034%, std~1.413%, skew~0.013, excess_kurt~2.800,
    historical 99%~3.19%, parametric 99%~3.25%. Toleransi longgar karena file
    data ini terus diperbarui workflow harian -- tujuan test ini adalah
    memastikan PIPA PERHITUNGAN benar terhadap data real, bukan mem-freeze
    angka historis secara ketat."""
    df = pd.read_csv(LPPF_RAW_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    result = compute_var_from_price_df(df, confidence_levels=(0.95, 0.99), window=252)

    assert result["n_obs"] == 252
    assert result["skewness"] == pytest.approx(0.013, abs=0.05)
    assert result["excess_kurtosis"] == pytest.approx(2.80, abs=0.5)
    assert result["per_confidence"][0.99]["historical_pct"] == pytest.approx(3.19, abs=0.5)
    assert result["per_confidence"][0.99]["parametric_pct"] == pytest.approx(3.25, abs=0.5)
    # Cornish-Fisher di 99% seharusnya jadi outlier (lebih tinggi dari cluster
    # historical/parametric) -- ini justru temuan utama yang mendasari
    # pemilihan recommended_method di modul ini, bukan bug.
    cf99 = result["per_confidence"][0.99]["cornish_fisher_pct"]
    hist99 = result["per_confidence"][0.99]["historical_pct"]
    assert cf99 > hist99 * 1.2
    assert result["per_confidence"][0.99]["recommended_method"] == "historical_mc_bootstrap_avg"
