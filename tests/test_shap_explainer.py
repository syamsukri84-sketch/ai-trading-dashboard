import numpy as np
import pandas as pd

from src.explainability.shap_explainer import explain_direction_prediction, explain_return_prediction
from src.models.direction_classifier import DirectionClassifier
from src.models.price_projector import PriceProjector


def make_feature_df(rows=300):
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


def test_explain_direction_prediction_lightgbm_with_calibration():
    df = make_feature_df()
    clf = DirectionClassifier(horizon_days=1, model_type="lightgbm", calibrate=True)
    clf.train(df)
    row = df[clf.feature_names_].iloc[[-1]]

    result = explain_direction_prediction(clf.model, row, clf.feature_names_, top_n=5)

    assert result["available"] is True
    assert len(result["top_features"]) == 5
    assert result["total_features"] == len(clf.feature_names_)
    for feature_row in result["top_features"]:
        assert feature_row["direction"] in {"mendukung NAIK", "mendukung TURUN"}
    contributions = [abs(f["contribution"]) for f in result["top_features"]]
    assert contributions == sorted(contributions, reverse=True)


def test_explain_return_prediction_xgboost_regressor():
    df = make_feature_df()
    proj = PriceProjector(projection_horizon=3)
    proj.train(df)
    row = df[proj.feature_names_].iloc[[-1]]

    result = explain_return_prediction(proj.model, row, proj.feature_names_, top_n=4)

    assert result["available"] is True
    assert len(result["top_features"]) == 4
    for feature_row in result["top_features"]:
        assert feature_row["direction"] in {
            "mendorong proyeksi return lebih tinggi",
            "mendorong proyeksi return lebih rendah",
        }


def test_explain_direction_prediction_unsupported_for_non_tree_model():
    df = make_feature_df()
    clf = DirectionClassifier(horizon_days=1, model_type="logistic", calibrate=False)
    clf.train(df)
    row = df[clf.feature_names_].iloc[[-1]]

    result = explain_direction_prediction(clf.model, row, clf.feature_names_)

    assert result["available"] is False
    assert "reason" in result


def test_explain_prediction_empty_row_reports_unavailable():
    df = make_feature_df()
    clf = DirectionClassifier(horizon_days=1, model_type="lightgbm", calibrate=False)
    clf.train(df)
    empty_row = df[clf.feature_names_].iloc[0:0]

    result = explain_direction_prediction(clf.model, empty_row, clf.feature_names_)

    assert result["available"] is False
