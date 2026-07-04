from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _evaluate_direction_signal(df: pd.DataFrame, signal: pd.Series, horizon_days: int) -> Dict[str, float]:
    data = df.copy()
    future_return = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
    valid = signal.notna() & future_return.notna() & signal.isin([-1, 1])
    if not valid.any():
        return {
            "total_signals": 0,
            "direction_accuracy_pct": 0.0,
            "avg_return_pct": 0.0,
            "win_rate_pct": 0.0,
        }

    signed_return = future_return[valid] * signal[valid]
    correct = signed_return > 0
    return {
        "total_signals": int(valid.sum()),
        "direction_accuracy_pct": float(correct.mean() * 100.0),
        "avg_return_pct": float(signed_return.mean() * 100.0),
        "win_rate_pct": float((signed_return > 0).mean() * 100.0),
    }


def evaluate_baseline_strategies(features_df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Baseline pembanding sederhana untuk memastikan AI mengalahkan rule-based strategy."""
    data = features_df.sort_values("timestamp").copy()
    baselines: List[Dict[str, float | str]] = []

    momentum_signal = np.sign(data["close"].pct_change(5)).replace(0, np.nan)
    ma_fast = data["close"].rolling(10).mean()
    ma_slow = data["close"].rolling(30).mean()
    ma_signal = pd.Series(np.where(ma_fast > ma_slow, 1, -1), index=data.index)
    rsi = data.get("feat_rsi_14", pd.Series(index=data.index, dtype=float))
    rsi_signal = pd.Series(np.nan, index=data.index)
    rsi_signal.loc[rsi < 30] = 1
    rsi_signal.loc[rsi > 70] = -1

    for name, signal in [
        ("Naive Momentum 5D", momentum_signal),
        ("MA Crossover 10/30", ma_signal),
        ("RSI Mean Reversion", rsi_signal),
    ]:
        metrics = _evaluate_direction_signal(data, signal, horizon_days)
        baselines.append({
            "strategy": name,
            "horizon_days": horizon_days,
            **metrics,
        })

    return pd.DataFrame(baselines)
