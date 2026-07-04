from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_") or c in ["open", "high", "low", "close", "volume"]]


def walk_forward_direction_validation(
    features_df: pd.DataFrame,
    model_factory: Callable[[], object],
    horizon_days: int = 1,
    train_size: int = 252,
    test_size: int = 20,
    step_size: int = 20,
    purge_gap: int | None = None,
) -> Dict[str, float]:
    """Validasi walk-forward untuk prediksi arah dengan purge gap antar train-test."""
    purge_gap = horizon_days if purge_gap is None else max(int(purge_gap), 0)
    data = features_df.sort_values("timestamp").reset_index(drop=True).copy()
    feature_cols = _feature_columns(data)
    future_return = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
    data["target_direction"] = (future_return > 0).astype(int)
    data = data.loc[future_return.notna()].reset_index(drop=True)

    predictions = []
    actuals = []
    probabilities = []

    start = 0
    while start + train_size + purge_gap + test_size <= len(data):
        train = data.iloc[start:start + train_size]
        test_start = start + train_size + purge_gap
        test = data.iloc[test_start:test_start + test_size]
        if train["target_direction"].nunique() < 2:
            start += step_size
            continue

        model = model_factory()
        X_train = train[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        y_train = train["target_direction"]
        X_test = test[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        y_test = test["target_direction"]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            class_list = list(getattr(model, "classes_", [0, 1]))
            up_idx = class_list.index(1) if 1 in class_list else -1
            probabilities.extend(proba[:, up_idx].tolist())

        predictions.extend(pred.tolist())
        actuals.extend(y_test.tolist())
        start += step_size

    if not actuals:
        return {
            "samples": 0,
            "direction_accuracy_pct": 0.0,
            "avg_confidence_pct": 0.0,
            "purge_gap": int(purge_gap),
        }

    pred_series = pd.Series(predictions)
    actual_series = pd.Series(actuals)
    confidence = pd.Series(probabilities).apply(lambda p: max(p, 1 - p) * 100) if probabilities else pd.Series(dtype=float)
    return {
        "samples": int(len(actuals)),
        "direction_accuracy_pct": float((pred_series == actual_series).mean() * 100.0),
        "avg_confidence_pct": float(confidence.mean()) if not confidence.empty else 0.0,
        "purge_gap": int(purge_gap),
    }


def walk_forward_return_validation(
    features_df: pd.DataFrame,
    model_factory: Callable[[], object],
    horizon_days: int = 3,
    train_size: int = 252,
    test_size: int = 20,
    step_size: int = 20,
    purge_gap: int | None = None,
) -> Dict[str, float]:
    """Validasi walk-forward untuk prediksi return dengan purge gap antar train-test."""
    purge_gap = horizon_days if purge_gap is None else max(int(purge_gap), 0)
    data = features_df.sort_values("timestamp").reset_index(drop=True).copy()
    feature_cols = _feature_columns(data)
    data["target_return"] = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
    data = data.dropna(subset=["target_return"]).reset_index(drop=True)

    predictions = []
    actuals = []

    start = 0
    while start + train_size + purge_gap + test_size <= len(data):
        train = data.iloc[start:start + train_size]
        test_start = start + train_size + purge_gap
        test = data.iloc[test_start:test_start + test_size]
        model = model_factory()
        X_train = train[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        y_train = train["target_return"]
        X_test = test[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        y_test = test["target_return"]

        model.fit(X_train, y_train)
        predictions.extend(model.predict(X_test).tolist())
        actuals.extend(y_test.tolist())
        start += step_size

    if not actuals:
        return {
            "samples": 0,
            "direction_accuracy_pct": 0.0,
            "mae_pct": 0.0,
            "avg_predicted_return_pct": 0.0,
            "avg_actual_return_pct": 0.0,
            "purge_gap": int(purge_gap),
        }

    pred_series = pd.Series(predictions)
    actual_series = pd.Series(actuals)
    return {
        "samples": int(len(actuals)),
        "direction_accuracy_pct": float((np.sign(pred_series) == np.sign(actual_series)).mean() * 100.0),
        "mae_pct": float((pred_series - actual_series).abs().mean() * 100.0),
        "avg_predicted_return_pct": float(pred_series.mean() * 100.0),
        "avg_actual_return_pct": float(actual_series.mean() * 100.0),
        "purge_gap": int(purge_gap),
    }
