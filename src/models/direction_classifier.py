import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


logger = logging.getLogger(__name__)


class DirectionClassifier:
    """Classifier arah harga untuk horizon tertentu."""

    def __init__(
        self,
        horizon_days: int = 1,
        model_type: str = "lightgbm",
        min_return_threshold: float = 0.0,
        calibrate: bool = True,
        calibration_method: str = "sigmoid",
    ):
        self.horizon_days = int(horizon_days)
        self.model_type = str(model_type).lower().strip()
        self.min_return_threshold = float(min_return_threshold)
        self.calibrate = bool(calibrate)
        self.calibration_method = calibration_method
        self.model = self._build_model()
        self.feature_names_: list[str] = []

    def _build_model(self):
        # Hyperparameter LightGBM di bawah SUDAH DIVALIDASI lewat walk-forward
        # vs baseline naif pada 12 ticker acak (lihat sesi optimasi model,
        # Juli 2026): konfigurasi lama (n_estimators=200, num_leaves=31, tanpa
        # regularisasi) rata-rata KALAH -5.8pp dari baseline "tebak kelas
        # mayoritas"; konfigurasi teregulasi kuat di bawah mengurangi jarak
        # itu jadi -2.3pp (masih belum melampaui baseline, tapi perbaikan
        # nyata & terukur). Sinyal arah 1-hari memang sangat lemah di data
        # ini -- regularisasi mengurangi overfitting pada noise, tapi TIDAK
        # menciptakan sinyal yang tidak ada. Jalankan
        # `walk_forward_direction_validation` sebelum mengubah angka ini lagi.
        if self.model_type == "lightgbm" and LGBMClassifier is not None:
            return LGBMClassifier(
                n_estimators=50,
                learning_rate=0.03,
                max_depth=-1,
                num_leaves=7,
                min_child_samples=30,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1,
            )
        if self.model_type == "xgboost" and xgb is not None:
            # Belum divalidasi individual seperti LightGBM di atas -- angka
            # ini analog/konsisten (kompleksitas & regularisasi setara),
            # BUKAN hasil pengujian walk-forward terpisah untuk XGBoost.
            return xgb.XGBClassifier(
                n_estimators=50,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=10,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])

    def build_walk_forward_estimator(self, n_train_rows: int):
        """Kembalikan estimator UNFITTED yang identik dengan apa yang `train()`
        akan pasang ke `self.model` untuk fold berukuran `n_train_rows` --
        termasuk wrapping CalibratedClassifierCV kalau `self.calibrate=True`
        dan datanya cukup besar, persis logika di `train()` di bawah.

        Sebelum ini, walk-forward validation (run_analysis.py, scripts/
        screen_genuine_edge.py) memakai `DirectionClassifier(..., calibrate=False).model`
        -- estimator MENTAH, bukan objek yang sebenarnya dideploy ke prediksi
        live (yang default `calibrate=True`, dibungkus CalibratedClassifierCV
        di `train()`). Akibatnya kesimpulan "tidak ada edge nyata" belum
        pernah menguji model yang sesungguhnya dilihat pengguna. Lihat audit
        codebase 2026-07-12.
        """
        base_model = self._build_model()
        if self.calibrate and n_train_rows >= 120:
            n_splits = min(3, max(2, n_train_rows // 120))
            return CalibratedClassifierCV(
                estimator=base_model,
                method=self.calibration_method,
                cv=TimeSeriesSplit(n_splits=n_splits),
            )
        return base_model

    @staticmethod
    def _feature_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c.startswith("feat_") or c in ["open", "high", "low", "close", "volume"]]

    def _prepare_data(self, df: pd.DataFrame, is_training: bool = True):
        data = df.copy()
        feature_cols = self._feature_columns(data)
        X = data[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)

        if not is_training:
            return X

        future_return = (data["close"].shift(-self.horizon_days) / data["close"]) - 1.0
        y = (future_return > self.min_return_threshold).astype(int)
        valid_idx = future_return.dropna().index
        return X.loc[valid_idx], y.loc[valid_idx]

    def train(self, features_df: pd.DataFrame):
        X_train, y_train = self._prepare_data(features_df, is_training=True)
        if X_train.empty or y_train.nunique() < 2:
            raise ValueError("Data training classifier tidak cukup atau hanya punya satu kelas arah.")

        self.model = self.build_walk_forward_estimator(len(X_train))

        self.model.fit(X_train, y_train)
        self.feature_names_ = X_train.columns.tolist()
        return self

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        X_test = self._prepare_data(features_df, is_training=False)
        latest_features = X_test.iloc[[-1]]

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(latest_features)[0]
            classes = list(getattr(self.model, "classes_", [0, 1]))
            prob_up = float(proba[classes.index(1)]) if 1 in classes else 0.5
        else:
            # Model ini tidak mendukung predict_proba -- tidak ada probabilitas
            # terkalibrasi sungguhan untuk diambil. JANGAN pura-pura presisi
            # (mis. 0.55/0.45 tetap) karena angka itu akan mengalir ke
            # confidence_pct seolah hasil kalibrasi asli. Pakai nilai netral
            # (sedikit di atas/bawah 0.5 hanya untuk menentukan arah) dan catat
            # peringatan supaya kasus ini terlihat kalau pernah terjadi.
            logger.warning(
                "Model %s tidak punya predict_proba -- confidence_pct tidak "
                "terkalibrasi, hanya indikasi arah kasar.", self.model_type
            )
            pred = int(self.model.predict(latest_features)[0])
            prob_up = 0.51 if pred == 1 else 0.49

        prob_down = 1.0 - prob_up
        direction = "NAIK" if prob_up >= 0.5 else "TURUN"
        confidence = max(prob_up, prob_down) * 100.0

        return {
            "model_name": self.model_type.upper(),
            "horizon_days": self.horizon_days,
            "prob_up": float(prob_up),
            "prob_down": float(prob_down),
            "direction": direction,
            "confidence_pct": float(confidence),
        }
