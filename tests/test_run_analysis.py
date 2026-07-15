"""Test untuk logika dedup/skip di run_analysis.py -- sebelum ini run_analysis.py
(orchestrator inti yang menghasilkan has_genuine_edge_h1/h3/h5/h10 dan menentukan
ticker mana yang di-skip harian) sama sekali tidak punya test, berbeda dari
modul lain yang sudah tercover (accuracy_tracker, personalization, market_regime).
Fungsi-fungsi di sini murni (operasi DataFrame, tanpa I/O/training model) jadi
murah diuji langsung. Lihat audit codebase 2026-07-12."""

import pandas as pd
import pytest

from run_analysis import (
    _available_prediction_dates,
    _has_completed_latest_analysis,
    _has_prediction_for_date,
    _prediction_exists,
)


def make_pred_row(**overrides):
    row = {
        "ticker": "BBCA",
        "model_name": "XGBoost",
        "current_date": "2026-07-10",
        "horizon_days": 3,
        "prediction_purpose": "THREE_DAY_FORECAST",
        "is_active": True,
    }
    row.update(overrides)
    return row


def test_prediction_exists_empty_df_returns_false():
    assert _prediction_exists(pd.DataFrame(), "BBCA", "XGBoost", "2026-07-10", 3, "THREE_DAY_FORECAST") is False


def test_prediction_exists_missing_required_columns_returns_false():
    df = pd.DataFrame([{"ticker": "BBCA"}])
    assert _prediction_exists(df, "BBCA", "XGBoost", "2026-07-10", 3, "THREE_DAY_FORECAST") is False


def test_prediction_exists_matching_active_row_returns_true():
    df = pd.DataFrame([make_pred_row()])
    assert _prediction_exists(df, "BBCA", "XGBoost", "2026-07-10", 3, "THREE_DAY_FORECAST") is True


def test_prediction_exists_ignores_inactive_row():
    df = pd.DataFrame([make_pred_row(is_active=False)])
    assert _prediction_exists(df, "BBCA", "XGBoost", "2026-07-10", 3, "THREE_DAY_FORECAST") is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("ticker", "BBRI"),
        ("model_name", "LightGBM"),
        ("current_date", "2026-07-11"),
        ("horizon_days", 5),
        ("prediction_purpose", "NEXT_DAY_DIRECTION"),
    ],
)
def test_prediction_exists_mismatched_field_returns_false(field, value):
    df = pd.DataFrame([make_pred_row()])
    kwargs = {
        "ticker": "BBCA",
        "model_name": "XGBoost",
        "current_date": "2026-07-10",
        "horizon_days": 3,
        "prediction_purpose": "THREE_DAY_FORECAST",
    }
    kwargs[field] = value
    assert _prediction_exists(df, **kwargs) is False


def test_prediction_exists_handles_string_horizon_days():
    """horizon_days di CSV bisa terbaca sebagai string -- pastikan koersi numerik
    tidak diam-diam gagal cocok (mis. '3' vs 3)."""
    df = pd.DataFrame([make_pred_row(horizon_days="3")])
    assert _prediction_exists(df, "BBCA", "XGBoost", "2026-07-10", 3, "THREE_DAY_FORECAST") is True


def test_has_completed_latest_analysis_true_when_all_required_present():
    rows = [
        make_pred_row(model_name="XGBoost", horizon_days=3, prediction_purpose="THREE_DAY_FORECAST"),
        make_pred_row(model_name="XGBoost", horizon_days=1, prediction_purpose="NEXT_DAY_DIRECTION"),
    ]
    df = pd.DataFrame(rows)
    assert _has_completed_latest_analysis(df, "BBCA", "2026-07-10", required_models=["XGBoost"]) is True


def test_has_completed_latest_analysis_false_when_h3_model_missing():
    df = pd.DataFrame([make_pred_row(model_name="XGBoost", horizon_days=1, prediction_purpose="NEXT_DAY_DIRECTION")])
    assert _has_completed_latest_analysis(df, "BBCA", "2026-07-10", required_models=["XGBoost"]) is False


def test_has_completed_latest_analysis_false_when_h1_direction_missing():
    df = pd.DataFrame([make_pred_row(model_name="XGBoost", horizon_days=3, prediction_purpose="THREE_DAY_FORECAST")])
    assert _has_completed_latest_analysis(df, "BBCA", "2026-07-10", required_models=["XGBoost"]) is False


def test_has_completed_latest_analysis_requires_every_required_model():
    """Kalau required_models = [XGBoost, LSTM], keduanya harus punya prediksi
    H+3 -- bukan cuma salah satu -- sebelum ticker dianggap 'sudah lengkap'
    dan dilewati di run harian berikutnya."""
    df = pd.DataFrame([make_pred_row(model_name="XGBoost", horizon_days=3, prediction_purpose="THREE_DAY_FORECAST"),
                        make_pred_row(model_name="XGBoost", horizon_days=1, prediction_purpose="NEXT_DAY_DIRECTION")])
    assert _has_completed_latest_analysis(df, "BBCA", "2026-07-10", required_models=["XGBoost", "LSTM"]) is False


def test_has_prediction_for_date_wraps_next_day_direction_xgboost():
    df = pd.DataFrame([make_pred_row(model_name="XGBoost", horizon_days=1, prediction_purpose="NEXT_DAY_DIRECTION")])
    assert _has_prediction_for_date(df, "BBCA", "2026-07-10") is True
    assert _has_prediction_for_date(df, "BBRI", "2026-07-10") is False


def test_available_prediction_dates_empty_inputs():
    assert _available_prediction_dates(pd.DataFrame()) == []
    assert _available_prediction_dates(pd.DataFrame({"close": [1, 2, 3]})) == []


def test_available_prediction_dates_below_min_rows_returns_empty():
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=100, freq="D")})
    assert _available_prediction_dates(df, min_rows=252) == []


def test_available_prediction_dates_returns_trailing_dates_after_min_rows():
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=260, freq="D")})
    dates = _available_prediction_dates(df, min_rows=252)
    assert len(dates) == 260 - 252 + 1
    assert dates[0] == pd.Timestamp("2026-01-01") + pd.Timedelta(days=251)
    assert dates[-1] == pd.Timestamp("2026-01-01") + pd.Timedelta(days=259)
