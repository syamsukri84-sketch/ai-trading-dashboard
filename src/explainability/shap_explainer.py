"""Penjelasan SHAP untuk model tree (LightGBM/XGBoost/RandomForest).

Dibangun supaya prediksi model bisa dijelaskan ke pengguna non-ML: untuk satu
baris (hari trading terakhir), fitur mana yang mendorong prediksi ke arah
tertentu dan seberapa besar. Pakai shap.TreeExplainer, yang bekerja langsung
di atas pohon-pohon terlatih tanpa perlu retraining atau sampling tambahan.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

_TREE_MODEL_MODULE_PREFIXES = ("lightgbm", "xgboost", "sklearn.ensemble", "sklearn.tree")

DIRECTION_LABELS = ("mendukung NAIK", "mendukung TURUN")
RETURN_LABELS = ("mendorong proyeksi return lebih tinggi", "mendorong proyeksi return lebih rendah")


def _unwrap_tree_estimator(model: Any) -> Any | None:
    """Mengambil estimator tree asli dari pembungkus (CalibratedClassifierCV,
    Pipeline) yang dipakai DirectionClassifier/GlobalDirectionModel.

    Mengembalikan None kalau model dasarnya bukan tree (mis. LogisticRegression
    di dalam Pipeline) -- shap.TreeExplainer tidak berlaku untuk kasus itu, dan
    pemanggil harus melaporkan "tidak didukung", bukan memaksakan hasil salah.
    """
    if isinstance(model, CalibratedClassifierCV):
        if not getattr(model, "calibrated_classifiers_", None):
            return None
        # Ambil estimator dari fold CV pertama sebagai wakil -- menjelaskan
        # rata-rata seluruh fold kalibrasi jauh lebih mahal dan tidak sepadan
        # untuk sekadar menunjukkan arah kontribusi fitur ke pengguna.
        return _unwrap_tree_estimator(model.calibrated_classifiers_[0].estimator)
    if isinstance(model, Pipeline):
        return _unwrap_tree_estimator(model.steps[-1][1])
    module_name = type(model).__module__
    if any(module_name.startswith(prefix) for prefix in _TREE_MODEL_MODULE_PREFIXES):
        return model
    return None


def _positive_class_shap_row(shap_values: Any, positive_class_index: int) -> np.ndarray:
    """Menormalkan bentuk output shap_values (beda antar library/versi) jadi
    satu larik 1D kontribusi per fitur, untuk baris tunggal & kelas positif."""
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # (n_samples, n_features, n_classes) -- RandomForestClassifier
        return values[0, :, positive_class_index]
    if values.ndim == 2:
        # (n_samples, n_features) -- LightGBM/XGBoost classifier biner atau regressor
        return values[0]
    raise ValueError(f"Bentuk shap_values tidak dikenali: {values.shape}")


def _base_value_scalar(base_value: Any, positive_class_index: int) -> float:
    values = np.atleast_1d(base_value)
    return float(values[positive_class_index] if values.size > 1 else values[0])


def explain_prediction(
    model: Any,
    row: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 8,
    positive_class_index: int = 1,
    direction_labels: tuple[str, str] = ("mendorong nilai lebih tinggi", "mendorong nilai lebih rendah"),
) -> dict:
    """Menjelaskan satu prediksi (baris terakhir/hari ini) lewat kontribusi
    SHAP per fitur, diurutkan dari yang paling berpengaruh.

    `positive_class_index` dipakai untuk classifier biner (1 = kelas "naik");
    diabaikan efeknya untuk regressor karena shap_values sudah 2D langsung.
    """
    tree_model = _unwrap_tree_estimator(model)
    if tree_model is None:
        return {
            "available": False,
            "reason": f"Tipe model '{type(model).__name__}' belum didukung penjelasan SHAP (bukan model tree).",
        }
    if row.empty:
        return {"available": False, "reason": "Tidak ada baris data untuk dijelaskan."}

    try:
        row_features = row[feature_names]
        explainer = shap.TreeExplainer(tree_model)
        raw_shap_values = explainer.shap_values(row_features)
        contributions = _positive_class_shap_row(raw_shap_values, positive_class_index)
        base_value = _base_value_scalar(explainer.expected_value, positive_class_index)
    except Exception as e:
        return {"available": False, "reason": f"Perhitungan SHAP gagal: {e}"}

    rows = [
        {"feature": name, "value": float(value), "contribution": float(contribution)}
        for name, value, contribution in zip(feature_names, row_features.iloc[0].tolist(), contributions)
    ]
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    top_rows = rows[:top_n]
    for r in top_rows:
        r["direction"] = direction_labels[0] if r["contribution"] >= 0 else direction_labels[1]

    return {
        "available": True,
        "base_value": base_value,
        "top_features": top_rows,
        "total_features": len(feature_names),
    }


def explain_direction_prediction(model: Any, row: pd.DataFrame, feature_names: list[str], top_n: int = 8) -> dict:
    """Jelaskan classifier arah (mis. DirectionClassifier.model) -- kontribusi
    positif berarti mendukung kelas "NAIK"."""
    return explain_prediction(
        model, row, feature_names, top_n=top_n, positive_class_index=1, direction_labels=DIRECTION_LABELS
    )


def explain_return_prediction(model: Any, row: pd.DataFrame, feature_names: list[str], top_n: int = 8) -> dict:
    """Jelaskan regressor return (mis. PriceProjector.model) -- kontribusi
    positif berarti mendorong proyeksi return lebih tinggi."""
    return explain_prediction(
        model, row, feature_names, top_n=top_n, positive_class_index=0, direction_labels=RETURN_LABELS
    )
