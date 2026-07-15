from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Ambang "edge nyata" -- HEURISTIK praktis (bukan uji signifikansi statistik
# formal), dipilih supaya model harus mengalahkan tebakan naif dengan margin
# yang berarti, bukan cuma sedikit di atas 0 yang bisa jadi noise sampel.
# Dipakai konsisten di run_analysis.py, accuracy_tracker.py, dan
# scripts/screen_genuine_edge.py -- ubah di satu tempat ini saja kalau perlu
# menyesuaikan, supaya ketiga tempat itu tidak diam-diam berbeda ambang.
EDGE_THRESHOLD_PCT = 3.0


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_") or c in ["open", "high", "low", "close", "volume"]]


def _paired_one_sided_pvalue(better_when_higher: Sequence[float], worse_when_higher: Sequence[float]) -> float:
    """P-value satu-sisi (paired t-test) untuk H1: rata-rata `better_when_higher`
    > rata-rata `worse_when_higher`, dihitung per-FOLD (bukan per-sampel individual
    di dalam fold, yang saling berkorelasi karena berasal dari model & periode
    yang sama -- lihat catatan di `apply_fdr_correction` di bawah). Perlu >=2
    fold untuk uji berpasangan; kalau kurang, kembalikan 1.0 (tidak signifikan)
    alih-alih memaksakan uji yang tidak valid secara statistik."""
    a = np.asarray(list(better_when_higher), dtype=float)
    b = np.asarray(list(worse_when_higher), dtype=float)
    if len(a) < 2 or len(a) != len(b) or np.allclose(a, b):
        return 1.0
    try:
        t_stat, p_two_sided = stats.ttest_rel(a, b)
    except Exception:
        return 1.0
    if np.isnan(p_two_sided):
        return 1.0
    if t_stat > 0:
        return float(p_two_sided / 2.0)
    return float(1.0 - p_two_sided / 2.0)


def apply_fdr_correction(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Koreksi Benjamini-Hochberg (False Discovery Rate) untuk banyak uji
    sekaligus (mis. 265 ticker x 4 horizon = ~1060 uji independen).

    PENTING KENAPA INI PERLU: `EDGE_THRESHOLD_PCT` di atas adalah ambang effect-
    size TETAP, diuji ke ratusan ticker sekaligus TANPA koreksi ini sebelumnya.
    Kalau edge sungguhan = 0 di SEMUA ticker (konsisten dengan temuan sesi
    optimasi Juli 2026: rata-rata H+1 -2.2pp dari 265 ticker), beberapa ticker
    akan tetap lolos ambang 3.0pp murni karena varians sampel -- itu bukan
    edge, itu multiple-comparisons false positive yang diharapkan terjadi.
    Koreksi BH-FDR ini membatasi proporsi false positive di antara ticker yang
    "lolos" sampai `alpha` (default 5%), dipakai bersama gate effect-size yang
    sudah ada (`edge_vs_baseline_pct >= EDGE_THRESHOLD_PCT`) -- BUKAN pengganti,
    dua syarat ini harus lolos SEKALIGUS. Lihat audit codebase 2026-07-12.

    Return: array boolean sepanjang p_values, True = signifikan setelah koreksi.
    """
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    p = np.nan_to_num(p, nan=1.0)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresholds
    significant_sorted = np.zeros(n, dtype=bool)
    if below.any():
        max_rank = int(np.max(np.where(below)[0]))
        significant_sorted[: max_rank + 1] = True
    result = np.zeros(n, dtype=bool)
    result[order] = significant_sorted
    return result


def walk_forward_direction_validation(
    features_df: pd.DataFrame,
    model_factory: Callable[[], object],
    horizon_days: int = 1,
    train_size: int = 252,
    test_size: int = 20,
    step_size: int = 20,
    purge_gap: int | None = None,
) -> Dict[str, float]:
    """Validasi walk-forward untuk prediksi arah dengan purge gap antar train-test.

    PENTING: akurasi mentah ("direction_accuracy_pct") BISA MENYESATKAN kalau
    dilihat sendirian -- classifier yang cuma menebak kelas mayoritas terus-
    menerus bisa punya akurasi tinggi kalau kebetulan periode test didominasi
    kelas itu, padahal tidak belajar sinyal apa pun (lihat catatan di
    METODOLOGI_REVIEW_DAN_CLEANUP_KODE.md / laporan sesi optimasi). Fungsi ini
    SELALU menghitung baseline "tebak kelas mayoritas dari data training" pada
    fold yang SAMA, supaya "edge_vs_baseline_pct" jadi angka yang jujur
    menunjukkan apakah model benar-benar lebih baik dari tebakan naif -- bukan
    cuma kebetulan cocok dengan base rate periode test.
    """
    purge_gap = horizon_days if purge_gap is None else max(int(purge_gap), 0)
    data = features_df.sort_values("timestamp").reset_index(drop=True).copy()
    feature_cols = _feature_columns(data)
    future_return = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
    data["target_direction"] = (future_return > 0).astype(int)
    data = data.loc[future_return.notna()].reset_index(drop=True)

    predictions = []
    actuals = []
    probabilities = []
    baseline_predictions = []
    fold_model_accuracy = []
    fold_baseline_accuracy = []

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

        # Baseline naif: kelas mayoritas DARI DATA TRAINING fold ini, dipakai
        # sebagai tebakan konstan untuk seluruh test fold -- pembanding yang
        # dievaluasi pada fold & periode test PERSIS SAMA dengan model.
        majority_class = int(y_train.mode().iloc[0])
        baseline_predictions.extend([majority_class] * len(y_test))

        # Akurasi PER-FOLD (bukan per-sampel) dicatat terpisah untuk uji
        # signifikansi berpasangan -- sampel di dalam satu fold saling
        # berkorelasi (model & periode yang sama), jadi memperlakukan tiap
        # baris sebagai observasi independen akan meremehkan p-value. Tiap
        # fold, sebaliknya, adalah observasi yang cukup independen karena
        # test_size == step_size (tidak overlap secara kalender).
        fold_model_accuracy.append(float((pred == y_test.values).mean()))
        fold_baseline_accuracy.append(float((np.array([majority_class] * len(y_test)) == y_test.values).mean()))

        predictions.extend(pred.tolist())
        actuals.extend(y_test.tolist())
        start += step_size

    if not actuals:
        return {
            "samples": 0,
            "direction_accuracy_pct": 0.0,
            "baseline_majority_accuracy_pct": 0.0,
            "edge_vs_baseline_pct": 0.0,
            "pred_positive_rate_pct": 0.0,
            "actual_positive_rate_pct": 0.0,
            "avg_confidence_pct": 0.0,
            "purge_gap": int(purge_gap),
            "n_folds": 0,
            "p_value_vs_baseline": 1.0,
        }

    pred_series = pd.Series(predictions)
    actual_series = pd.Series(actuals)
    baseline_series = pd.Series(baseline_predictions)
    confidence = pd.Series(probabilities).apply(lambda p: max(p, 1 - p) * 100) if probabilities else pd.Series(dtype=float)
    model_accuracy = float((pred_series == actual_series).mean() * 100.0)
    baseline_accuracy = float((baseline_series == actual_series).mean() * 100.0)
    return {
        "samples": int(len(actuals)),
        "direction_accuracy_pct": model_accuracy,
        "baseline_majority_accuracy_pct": baseline_accuracy,
        "edge_vs_baseline_pct": model_accuracy - baseline_accuracy,
        "pred_positive_rate_pct": float((pred_series == 1).mean() * 100.0),
        "actual_positive_rate_pct": float((actual_series == 1).mean() * 100.0),
        "avg_confidence_pct": float(confidence.mean()) if not confidence.empty else 0.0,
        "purge_gap": int(purge_gap),
        "n_folds": len(fold_model_accuracy),
        "p_value_vs_baseline": _paired_one_sided_pvalue(fold_model_accuracy, fold_baseline_accuracy),
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
    """Validasi walk-forward untuk prediksi return dengan purge gap antar train-test.

    Sama seperti `walk_forward_direction_validation`, hasil di sini SELALU
    dibandingkan terhadap dua baseline naif yang dievaluasi pada fold & test
    period yang PERSIS SAMA dengan model: (1) "zero return" (tebak tidak ada
    perubahan harga -- baseline standar di keuangan/random-walk hypothesis),
    dan (2) "mean training return" (ekstrapolasi rata-rata drift dari data
    training). MAE model yang lebih kecil dari MAE baseline zero TIDAK
    otomatis berarti model bagus kalau se-episode itu memang volatil --
    lihat `edge_vs_zero_mae_pct`/`edge_vs_mean_mae_pct` (positif = model lebih
    baik dari baseline) untuk perbandingan yang adil.
    """
    purge_gap = horizon_days if purge_gap is None else max(int(purge_gap), 0)
    data = features_df.sort_values("timestamp").reset_index(drop=True).copy()
    feature_cols = _feature_columns(data)
    data["target_return"] = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
    data = data.dropna(subset=["target_return"]).reset_index(drop=True)

    predictions = []
    actuals = []
    baseline_zero_preds = []
    baseline_mean_preds = []
    fold_model_mae = []
    fold_baseline_mean_mae = []

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
        fold_pred = model.predict(X_test)
        fold_baseline_mean = float(y_train.mean())
        predictions.extend(fold_pred.tolist())
        actuals.extend(y_test.tolist())
        baseline_zero_preds.extend([0.0] * len(y_test))
        baseline_mean_preds.extend([fold_baseline_mean] * len(y_test))

        # MAE per-fold (bukan per-sampel) untuk uji signifikansi berpasangan --
        # sama alasannya seperti walk_forward_direction_validation di atas.
        fold_model_mae.append(float(np.abs(fold_pred - y_test.values).mean()))
        fold_baseline_mean_mae.append(float(np.abs(fold_baseline_mean - y_test.values).mean()))
        start += step_size

    if not actuals:
        return {
            "samples": 0,
            "direction_accuracy_pct": 0.0,
            "mae_pct": 0.0,
            "baseline_zero_mae_pct": 0.0,
            "baseline_mean_mae_pct": 0.0,
            "edge_vs_zero_mae_pct": 0.0,
            "edge_vs_mean_mae_pct": 0.0,
            "baseline_mean_direction_accuracy_pct": 0.0,
            "avg_predicted_return_pct": 0.0,
            "avg_actual_return_pct": 0.0,
            "purge_gap": int(purge_gap),
            "n_folds": 0,
            "p_value_vs_baseline": 1.0,
        }

    pred_series = pd.Series(predictions)
    actual_series = pd.Series(actuals)
    baseline_zero_series = pd.Series(baseline_zero_preds)
    baseline_mean_series = pd.Series(baseline_mean_preds)

    model_mae = float((pred_series - actual_series).abs().mean() * 100.0)
    baseline_zero_mae = float((baseline_zero_series - actual_series).abs().mean() * 100.0)
    baseline_mean_mae = float((baseline_mean_series - actual_series).abs().mean() * 100.0)
    return {
        "samples": int(len(actuals)),
        "direction_accuracy_pct": float((np.sign(pred_series) == np.sign(actual_series)).mean() * 100.0),
        "mae_pct": model_mae,
        "baseline_zero_mae_pct": baseline_zero_mae,
        "baseline_mean_mae_pct": baseline_mean_mae,
        "edge_vs_zero_mae_pct": baseline_zero_mae - model_mae,
        "edge_vs_mean_mae_pct": baseline_mean_mae - model_mae,
        "baseline_mean_direction_accuracy_pct": float(
            (np.sign(baseline_mean_series) == np.sign(actual_series)).mean() * 100.0
        ),
        "avg_predicted_return_pct": float(pred_series.mean() * 100.0),
        "avg_actual_return_pct": float(actual_series.mean() * 100.0),
        "purge_gap": int(purge_gap),
        "n_folds": len(fold_model_mae),
        # MAE lebih kecil = lebih baik, jadi dibalik tanda supaya konvensi
        # "better_when_higher" di _paired_one_sided_pvalue tetap konsisten.
        "p_value_vs_baseline": _paired_one_sided_pvalue(
            [-m for m in fold_model_mae], [-m for m in fold_baseline_mean_mae]
        ),
    }


def walk_forward_sequence_model_validation(
    features_df: pd.DataFrame,
    model_factory: Callable[[], object],
    horizon_days: int = 3,
    train_size: int = 252,
    test_size: int = 20,
    step_size: int = 20,
    purge_gap: int | None = None,
    epochs: int = 5,
) -> Dict[str, float]:
    """Walk-forward untuk model SEQUENCE (mis. LSTMPriceProjector) yang punya
    interface `train(df, epochs=...)` / `predict(df)` -- BEDA dari model
    sklearn-style (`walk_forward_return_validation` di atas) yang menerima
    `X, y` datar via `.fit(X, y)`/`.predict(X)`. Model sequence butuh
    DATAFRAME BERURUTAN PENUH karena membangun sequence lookback (mis. 20
    hari) sendiri secara internal.

    Metodologinya SAMA PERSIS dengan `walk_forward_return_validation` (purge
    gap, baseline zero/mean return dievaluasi di fold identik, p-value
    berpasangan per-fold via `_paired_one_sided_pvalue`) -- direplikasi di
    sini, bukan dipanggil ulang, karena bentuk pemanggilan model & prediksi
    per-hari-nya berbeda. `model_factory()` harus mengembalikan objek baru
    dengan atribut `.lookback` (int) dan method `.train(df, epochs=int)` /
    `.predict(df) -> {"projected_return_pct": float, ...}`.

    PERINGATAN BIAYA: melatih ulang model sequence dari nol di SETIAP fold
    jauh lebih mahal daripada model tree (LightGBM/XGBoost) -- untuk ~30 fold
    per ticker, ini bisa memakan waktu signifikan. Pakai `epochs` kecil untuk
    validasi (bukan `epochs` produksi) dan jangan jalankan ini di jalur
    otomatis rutin (lihat scripts/validate_lstm_walk_forward.py -- dijalankan
    manual, bukan bagian dari screen_genuine_edge.py mingguan). Lihat audit
    codebase 2026-07-12/13.
    """
    purge_gap = horizon_days if purge_gap is None else max(int(purge_gap), 0)
    data = features_df.sort_values("timestamp").reset_index(drop=True).copy()
    data["target_return"] = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
    data = data.dropna(subset=["target_return"]).reset_index(drop=True)

    predictions = []
    actuals = []
    baseline_zero_preds = []
    baseline_mean_preds = []
    fold_model_mae = []
    fold_baseline_mean_mae = []

    start = 0
    while start + train_size + purge_gap + test_size <= len(data):
        train = data.iloc[start:start + train_size]
        test_start = start + train_size + purge_gap
        test = data.iloc[test_start:test_start + test_size]

        model = model_factory()
        lookback = int(getattr(model, "lookback", 20))
        if len(train) <= lookback:
            start += step_size
            continue
        model.train(train, epochs=epochs)

        fold_preds = []
        fold_actuals = []
        context_start = max(test_start - lookback, 0)
        for offset in range(len(test)):
            window_end = test_start + offset + 1
            window = data.iloc[context_start:window_end]
            if len(window) <= lookback:
                continue
            pred = model.predict(window)
            fold_preds.append(float(pred["projected_return_pct"]) / 100.0)
            fold_actuals.append(float(data.iloc[window_end - 1]["target_return"]))

        if not fold_preds:
            start += step_size
            continue

        fold_pred_arr = np.array(fold_preds)
        fold_actual_arr = np.array(fold_actuals)
        fold_baseline_mean = float(train["target_return"].mean())

        predictions.extend(fold_pred_arr.tolist())
        actuals.extend(fold_actual_arr.tolist())
        baseline_zero_preds.extend([0.0] * len(fold_actual_arr))
        baseline_mean_preds.extend([fold_baseline_mean] * len(fold_actual_arr))

        fold_model_mae.append(float(np.abs(fold_pred_arr - fold_actual_arr).mean()))
        fold_baseline_mean_mae.append(float(np.abs(fold_baseline_mean - fold_actual_arr).mean()))

        start += step_size

    if not actuals:
        return {
            "samples": 0,
            "direction_accuracy_pct": 0.0,
            "mae_pct": 0.0,
            "baseline_zero_mae_pct": 0.0,
            "baseline_mean_mae_pct": 0.0,
            "edge_vs_zero_mae_pct": 0.0,
            "edge_vs_mean_mae_pct": 0.0,
            "baseline_mean_direction_accuracy_pct": 0.0,
            "avg_predicted_return_pct": 0.0,
            "avg_actual_return_pct": 0.0,
            "purge_gap": int(purge_gap),
            "n_folds": 0,
            "p_value_vs_baseline": 1.0,
        }

    pred_series = pd.Series(predictions)
    actual_series = pd.Series(actuals)
    baseline_zero_series = pd.Series(baseline_zero_preds)
    baseline_mean_series = pd.Series(baseline_mean_preds)

    model_mae = float((pred_series - actual_series).abs().mean() * 100.0)
    baseline_zero_mae = float((baseline_zero_series - actual_series).abs().mean() * 100.0)
    baseline_mean_mae = float((baseline_mean_series - actual_series).abs().mean() * 100.0)
    return {
        "samples": int(len(actuals)),
        "direction_accuracy_pct": float((np.sign(pred_series) == np.sign(actual_series)).mean() * 100.0),
        "mae_pct": model_mae,
        "baseline_zero_mae_pct": baseline_zero_mae,
        "baseline_mean_mae_pct": baseline_mean_mae,
        "edge_vs_zero_mae_pct": baseline_zero_mae - model_mae,
        "edge_vs_mean_mae_pct": baseline_mean_mae - model_mae,
        "baseline_mean_direction_accuracy_pct": float(
            (np.sign(baseline_mean_series) == np.sign(actual_series)).mean() * 100.0
        ),
        "avg_predicted_return_pct": float(pred_series.mean() * 100.0),
        "avg_actual_return_pct": float(actual_series.mean() * 100.0),
        "purge_gap": int(purge_gap),
        "n_folds": len(fold_model_mae),
        "p_value_vs_baseline": _paired_one_sided_pvalue(
            [-m for m in fold_model_mae], [-m for m in fold_baseline_mean_mae]
        ),
    }
