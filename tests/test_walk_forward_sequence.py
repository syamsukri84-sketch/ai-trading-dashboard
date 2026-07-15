"""Test untuk walk_forward_sequence_model_validation -- validasi walk-forward
untuk model sequence (LSTM) yang sebelumnya (temuan audit 2026-07-12) TIDAK
PERNAH divalidasi walk-forward sama sekali karena interface train(df)/predict(df)
tidak kompatibel dengan walk_forward_return_validation (yang menerima X, y datar).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.lstm_projector import LSTMPriceProjector
from src.models.walk_forward import walk_forward_sequence_model_validation


def make_feature_df(rows=200):
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, rows))
    open_ = close + rng.normal(0, 0.5, rows)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, rows)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, rows)
    volume = rng.integers(1_000_000, 5_000_000, rows)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df["feat_rsi_14"] = 50 + rng.normal(0, 10, rows)
    df["feat_atr_14"] = (df["high"] - df["low"]).rolling(14).mean().fillna(1.0)
    df["feat_momentum"] = df["close"].pct_change(5).fillna(0)
    return df


def small_lstm_factory():
    # lookback & epoch kecil supaya test cepat -- ini test korektnas mekanisme
    # walk-forward, bukan test kualitas prediksi LSTM itu sendiri.
    return LSTMPriceProjector(projection_horizon=3, lookback=5)


def test_walk_forward_sequence_runs_and_reports_folds():
    df = make_feature_df(rows=200)
    result = walk_forward_sequence_model_validation(
        df, small_lstm_factory, horizon_days=3, train_size=120, test_size=20, step_size=20, epochs=1,
    )
    assert result["samples"] > 0
    assert result["n_folds"] > 0
    assert 0.0 <= result["p_value_vs_baseline"] <= 1.0


def test_walk_forward_sequence_reports_baseline_edge():
    df = make_feature_df(rows=200)
    result = walk_forward_sequence_model_validation(
        df, small_lstm_factory, horizon_days=3, train_size=120, test_size=20, step_size=20, epochs=1,
    )
    assert result["edge_vs_zero_mae_pct"] == pytest.approx(
        result["baseline_zero_mae_pct"] - result["mae_pct"], abs=1e-6
    )
    assert result["edge_vs_mean_mae_pct"] == pytest.approx(
        result["baseline_mean_mae_pct"] - result["mae_pct"], abs=1e-6
    )


def test_walk_forward_sequence_uses_purge_gap():
    df = make_feature_df(rows=200)
    result = walk_forward_sequence_model_validation(
        df, small_lstm_factory, horizon_days=5, train_size=120, test_size=20, step_size=20, epochs=1,
    )
    assert result["purge_gap"] == 5


def test_walk_forward_sequence_insufficient_data_returns_neutral_result():
    """Kalau data jauh lebih pendek dari train_size+purge_gap+test_size, tidak
    boleh crash -- harus kembali hasil netral (0 sample, p-value 1.0)."""
    df = make_feature_df(rows=50)
    result = walk_forward_sequence_model_validation(
        df, small_lstm_factory, horizon_days=3, train_size=120, test_size=20, step_size=20, epochs=1,
    )
    assert result["samples"] == 0
    assert result["n_folds"] == 0
    assert result["p_value_vs_baseline"] == 1.0
