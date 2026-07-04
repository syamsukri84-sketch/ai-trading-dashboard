from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from src.utils.accuracy_tracker import get_model_accuracy_summary


def get_reliability_weights(
    ticker: str,
    model_names: Iterable[str],
    prediction_purpose: str = "NEXT_DAY_DIRECTION",
    min_evaluations: int = 3,
) -> Dict[str, float]:
    """Bobot model berbasis track record akurasi historis per ticker."""
    models = [str(model).strip() for model in model_names if str(model).strip()]
    if not models:
        return {}

    summary = get_model_accuracy_summary(prediction_purpose=prediction_purpose)
    if summary.empty:
        equal = 1.0 / len(models)
        return {model: equal for model in models}

    ticker_summary = summary[
        (summary["ticker"].astype(str).str.upper() == str(ticker).upper())
        & (summary["model_name"].astype(str).isin(models))
    ].copy()
    if ticker_summary.empty:
        equal = 1.0 / len(models)
        return {model: equal for model in models}

    scores = {}
    for model in models:
        row = ticker_summary[ticker_summary["model_name"] == model]
        if row.empty:
            scores[model] = 0.2
            continue
        record = row.iloc[0]
        sample_factor = min(float(record.get("total_evaluations", 0)) / max(min_evaluations, 1), 1.0)
        accuracy_component = float(record.get("direction_accuracy_pct", 0.0)) / 100.0
        error_component = 1.0 - min(max(float(record.get("avg_error_margin_pct", 100.0)), 0.0), 100.0) / 100.0
        scores[model] = max((accuracy_component * 0.75 + error_component * 0.25) * sample_factor, 0.05)

    total = sum(scores.values())
    if total <= 0:
        equal = 1.0 / len(models)
        return {model: equal for model in models}
    return {model: float(score / total) for model, score in scores.items()}


def weighted_direction_probability(predictions: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
    if not predictions:
        return {"prob_up": 0.5, "prob_down": 0.5, "confidence_pct": 50.0}

    total_weight = sum(weights.get(model, 0.0) for model in predictions)
    if total_weight <= 0:
        total_weight = len(predictions)
        weights = {model: 1.0 / total_weight for model in predictions}

    prob_up = 0.0
    for model, pred in predictions.items():
        prob_up += float(pred.get("prob_up", 0.5)) * weights.get(model, 0.0)

    prob_up = max(0.0, min(1.0, prob_up))
    prob_down = 1.0 - prob_up
    return {
        "prob_up": float(prob_up),
        "prob_down": float(prob_down),
        "direction": "NAIK" if prob_up >= 0.5 else "TURUN",
        "confidence_pct": float(max(prob_up, prob_down) * 100.0),
    }
