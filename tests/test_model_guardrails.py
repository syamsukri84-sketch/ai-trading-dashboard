import pandas as pd

from src.utils.model_guardrails import audit_feature_leakage, audit_ohlcv_data, assert_no_training_leakage


def _valid_ohlcv():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
        "open": [100, 101, 102, 103, 104],
        "high": [102, 103, 104, 105, 106],
        "low": [99, 100, 101, 102, 103],
        "close": [101, 102, 103, 104, 105],
        "volume": [1000, 1100, 1200, 1300, 1400],
    })


def test_ohlcv_guardrail_rejects_invalid_high_low():
    df = _valid_ohlcv()
    df.loc[2, "high"] = 100

    result = audit_ohlcv_data(df, ticker="BBRI")

    assert not result.passed
    assert any("high" in error for error in result.errors)


def test_feature_guardrail_rejects_future_or_target_columns():
    features = _valid_ohlcv()
    features["feat_rsi_14"] = 50.0
    features["future_return_target"] = 0.02

    result = audit_feature_leakage(features)

    assert not result.passed
    assert any("future_return_target" in error for error in result.errors)


def test_training_guardrail_rejects_features_after_prediction_date():
    raw = _valid_ohlcv()
    features = raw.copy()
    features["feat_rsi_14"] = 50.0

    result = assert_no_training_leakage(raw, features, ticker="BBRI", prediction_date="2026-01-03")

    assert not result.passed
    assert any("setelah tanggal prediksi" in error for error in result.errors)

