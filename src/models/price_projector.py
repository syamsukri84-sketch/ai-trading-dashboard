import pandas as pd
import numpy as np
import logging
import joblib
import os
from typing import Dict, Any, Optional

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

logger = logging.getLogger(__name__)

class PriceProjector:
    """
    XGBoost-based model for short-term price projection.
    Projects the expected return for the next N days based on current technical features.
    """
    
    def __init__(self, projection_horizon: int = 3, model_type: str = "xgboost"):
        self.projection_horizon = projection_horizon # Proyeksi untuk N hari ke depan
        self.model_type = str(model_type).lower().strip()
        if self.model_type == "lightgbm" and LGBMRegressor is not None:
            self.model = LGBMRegressor(
                n_estimators=220,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=-1,
            )
        elif xgb is not None:
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
        else:
            self.model = None

    def _prepare_data(self, df: pd.DataFrame, is_training: bool = True):
        """Menyiapkan fitur dan target untuk model XGBoost."""
        data = df.copy()
        
        # Filter hanya kolom numerik (indikator teknikal dan harga)
        feature_cols = [c for c in data.columns if c.startswith('feat_') or c in ['open', 'high', 'low', 'close', 'volume']]
        X = data[feature_cols]
        
        if is_training:
            # Target: Persentase return N hari ke depan
            target = (data['close'].shift(-self.projection_horizon) / data['close']) - 1.0
            
            # Hapus baris terakhir yang targetnya NaN (karena masa depan belum terjadi)
            valid_idx = target.dropna().index
            return X.loc[valid_idx], target.loc[valid_idx]
            
        return X

    def train(self, features_df: pd.DataFrame, save_path: Optional[str] = None):
        """Melatih model regresi XGBoost pada data historis."""
        if self.model is None:
            logger.error("Library xgboost belum terinstal. Jalankan: pip install xgboost")
            return
            
        logger.info(f"Melatih model proyeksi harga untuk {self.projection_horizon} hari ke depan...")
        X_train, y_train = self._prepare_data(features_df, is_training=True)
        
        self.model.fit(X_train, y_train)
        self.feature_names_ = X_train.columns.tolist()
        logger.info("Pelatihan model proyeksi selesai.")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            joblib.dump(self, save_path)

    def predict(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Memproyeksikan harga berdasarkan baris data (hari) terakhir."""
        if self.model is None:
            return {"projected_return_pct": 0.0, "projected_price": 0.0}
            
        X_test = self._prepare_data(features_df, is_training=False)
        
        # Prediksi hanya untuk baris terakhir (kondisi hari ini)
        latest_features = X_test.iloc[[-1]]
        current_price = float(features_df['close'].iloc[-1])
        
        # Prediksi return
        projected_return = float(self.model.predict(latest_features)[0])
        
        # Konversi return ke proyeksi harga
        projected_price = current_price * (1 + projected_return)
        
        return {
            "projected_return_pct": projected_return * 100,  # Dalam bentuk persen
            "projected_price": projected_price
        }
