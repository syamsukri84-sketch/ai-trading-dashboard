import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.models.baseline_strategies import evaluate_baseline_strategies
from src.models.direction_classifier import DirectionClassifier
from src.models.walk_forward import (
    apply_fdr_correction,
    walk_forward_direction_validation,
    walk_forward_return_validation,
)


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


def test_walk_forward_direction_reports_baseline_edge():
    """edge_vs_baseline_pct harus konsisten dengan direction_accuracy_pct -
    baseline_majority_accuracy_pct, dan baseline majority tidak boleh 100%
    collapse ke satu kelas (mode() dari data sintetis seimbang random walk)."""
    df = make_feature_df()

    def factory():
        return DirectionClassifier(horizon_days=1, model_type="logistic").model

    result = walk_forward_direction_validation(df, factory, horizon_days=1, train_size=120, test_size=20, step_size=20)
    assert "baseline_majority_accuracy_pct" in result
    assert "edge_vs_baseline_pct" in result
    assert result["edge_vs_baseline_pct"] == pytest.approx(
        result["direction_accuracy_pct"] - result["baseline_majority_accuracy_pct"], abs=1e-6
    )
    assert 0 <= result["pred_positive_rate_pct"] <= 100
    assert 0 <= result["actual_positive_rate_pct"] <= 100


def test_walk_forward_return_reports_baseline_edge():
    df = make_feature_df()

    result = walk_forward_return_validation(df, LinearRegression, horizon_days=3, train_size=120, test_size=20, step_size=20)
    assert result["samples"] > 0
    assert result["edge_vs_zero_mae_pct"] == pytest.approx(
        result["baseline_zero_mae_pct"] - result["mae_pct"], abs=1e-6
    )
    assert result["edge_vs_mean_mae_pct"] == pytest.approx(
        result["baseline_mean_mae_pct"] - result["mae_pct"], abs=1e-6
    )
    assert 0 <= result["baseline_mean_direction_accuracy_pct"] <= 100


def test_walk_forward_direction_reports_pvalue_and_n_folds():
    """Sejak audit codebase 2026-07-12: walk-forward harus melaporkan p-value
    berpasangan per-fold (model vs baseline) supaya screen_genuine_edge.py bisa
    menerapkan koreksi multiple-testing -- bukan cuma effect-size mentah."""
    df = make_feature_df()

    def factory():
        return DirectionClassifier(horizon_days=1, model_type="logistic").model

    result = walk_forward_direction_validation(df, factory, horizon_days=1, train_size=120, test_size=20, step_size=20)
    assert result["n_folds"] > 0
    assert 0.0 <= result["p_value_vs_baseline"] <= 1.0


def test_walk_forward_return_reports_pvalue_and_n_folds():
    df = make_feature_df()
    result = walk_forward_return_validation(df, LinearRegression, horizon_days=3, train_size=120, test_size=20, step_size=20)
    assert result["n_folds"] > 0
    assert 0.0 <= result["p_value_vs_baseline"] <= 1.0


def test_walk_forward_empty_result_has_neutral_pvalue():
    """Kalau data tidak cukup untuk membentuk satu fold pun, p-value harus
    default ke 1.0 (tidak signifikan) -- bukan crash atau NaN yang bisa lolos
    ambang FDR secara tidak sengaja."""
    df = make_feature_df(rows=50)

    def factory():
        return DirectionClassifier(horizon_days=1, model_type="logistic").model

    result = walk_forward_direction_validation(df, factory, horizon_days=1, train_size=120, test_size=20, step_size=20)
    assert result["samples"] == 0
    assert result["n_folds"] == 0
    assert result["p_value_vs_baseline"] == 1.0


def test_apply_fdr_correction_rejects_pure_noise():
    """Kalau semua p-value berasal dari null hypothesis benar (uniform acak),
    koreksi BH-FDR harus menolak SEBAGIAN BESAR -- ini yang membedakan dari
    ambang effect-size tetap yang meloloskan beberapa 'positif' murni karena
    varians sampel di antara ratusan uji (temuan inti audit 2026-07-12)."""
    rng = np.random.default_rng(7)
    p_values = rng.uniform(0.0, 1.0, size=200)  # null penuh: semua p-value seharusnya uniform
    significant = apply_fdr_correction(p_values, alpha=0.05)
    # Di bawah null murni, BH-FDR menjamin expected false-discovery rate <= alpha --
    # dengan 200 uji murni noise, jumlah yang lolos harus jauh lebih kecil dari
    # ambang naif "p < 0.05" tanpa koreksi (~10 dari 200).
    assert significant.sum() <= 15


def test_apply_fdr_correction_keeps_strong_signal():
    """Kalau ada p-value yang jauh lebih kecil dari sisanya (sinyal kuat),
    koreksi FDR tidak boleh menolak semuanya -- harus tetap sensitif."""
    p_values = [0.0001, 0.0005, 0.9, 0.85, 0.7, 0.6, 0.5, 0.4]
    significant = apply_fdr_correction(p_values, alpha=0.05)
    assert significant[0]
    assert significant[1]
    assert not significant[2]


def test_apply_fdr_correction_empty_input():
    result = apply_fdr_correction([])
    assert len(result) == 0


def test_direction_classifier_build_walk_forward_estimator_wraps_calibration():
    """Sejak audit codebase 2026-07-12: estimator yang dipakai walk-forward
    HARUS identik dengan yang dideploy ke prediksi live -- termasuk wrapping
    CalibratedClassifierCV kalau calibrate=True dan datanya cukup besar."""
    from sklearn.calibration import CalibratedClassifierCV

    clf = DirectionClassifier(horizon_days=1, model_type="logistic", calibrate=True)
    estimator = clf.build_walk_forward_estimator(n_train_rows=252)
    assert isinstance(estimator, CalibratedClassifierCV)

    clf_no_calibrate = DirectionClassifier(horizon_days=1, model_type="logistic", calibrate=False)
    estimator_raw = clf_no_calibrate.build_walk_forward_estimator(n_train_rows=252)
    assert not isinstance(estimator_raw, CalibratedClassifierCV)

    clf_small = DirectionClassifier(horizon_days=1, model_type="logistic", calibrate=True)
    estimator_small = clf_small.build_walk_forward_estimator(n_train_rows=50)
    assert not isinstance(estimator_small, CalibratedClassifierCV)
