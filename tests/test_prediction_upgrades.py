import numpy as np
import pandas as pd

from src.models.baseline_strategies import evaluate_baseline_strategies
from src.models.direction_classifier import DirectionClassifier
from src.models.walk_forward import walk_forward_direction_validation


def make_feature_df(rows=360):
    rng = np.random.default_rng(42)
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


def test_direction_classifier_outputs_probabilities():
    df = make_feature_df()
    model = DirectionClassifier(horizon_days=1, model_type="logistic", calibrate=True)
    model.train(df)
    pred = model.predict(df)
    assert pred["direction"] in ["NAIK", "TURUN"]
    assert 0 <= pred["prob_up"] <= 1
    assert 50 <= pred["confidence_pct"] <= 100


def test_baseline_strategies_return_three_rows():
    df = make_feature_df()
    baselines = evaluate_baseline_strategies(df, horizon_days=3)
    assert len(baselines) == 3
    assert set(["strategy", "direction_accuracy_pct", "total_signals"]).issubset(baselines.columns)


def test_walk_forward_direction_validation_runs():
    df = make_feature_df()

    def factory():
        return DirectionClassifier(horizon_days=1, model_type="logistic").model

    result = walk_forward_direction_validation(df, factory, horizon_days=1, train_size=120, test_size=20, step_size=20)
    assert result["samples"] > 0
    assert 0 <= result["direction_accuracy_pct"] <= 100


def test_walk_forward_uses_purge_gap():
    df = make_feature_df()

    def factory():
        return DirectionClassifier(horizon_days=1, model_type="logistic", calibrate=False).model

    result = walk_forward_direction_validation(
        df,
        factory,
        horizon_days=5,
        train_size=120,
        test_size=20,
        step_size=20,
    )
    assert result["purge_gap"] == 5
    assert result["samples"] > 0
