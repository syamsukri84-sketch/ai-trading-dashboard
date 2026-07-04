import pandas as pd
import numpy as np
import logging
import time
import os
import joblib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from pyod.models.copod import COPOD
except ImportError:
    logger.warning("Library 'pyod' not found. COPOD model will fallback to IF. To install: pip install pyod")
    COPOD = None

class COPODModel:
    """
    Copula-Based Outlier Detection (COPOD) Model.
    Used as a secondary validator in VOLATILE regimes.
    """
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = COPOD(contamination=self.contamination) if COPOD is not None else None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        timestamp_cols = [c for c in X.columns if c.lower() in ['timestamp', 'date', 'datetime', 'ticker', 'symbol']]
        X = X.drop(columns=timestamp_cols, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        if X.isnull().values.any():
            X = X.fillna(0) # Prevent failing in volatile edge cases
        return X

    def train(self, features_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        if self.model is None:
            raise ImportError("Cannot train COPOD: pyod library is missing.")
            
        X = self._prepare_features(features_df)
        self.feature_names_ = X.columns.tolist()
        
        start_time = time.time()
        self.model.fit(X)
        
        # Normalize bounds [0, 100]
        self.min_score_ = self.model.decision_scores_.min()
        self.max_score_ = self.model.decision_scores_.max()
        
        logger.info(f"COPOD Training completed in {time.time() - start_time:.3f}s")
        
        if save_path:
            self.save_model(save_path)

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
            raise ImportError("pyod is missing.")
            
        start_time = time.time()
        X = self._prepare_features(features_df)
        
        if hasattr(self, 'feature_names_'):
             missing = [col for col in self.feature_names_ if col not in X.columns]
             if not missing:
                 X = X[self.feature_names_]
                 
        raw_scores = self.model.decision_function(X)
        
        if hasattr(self, 'min_score_') and hasattr(self, 'max_score_'):
            range_score = self.max_score_ - self.min_score_
            scaled_scores = ((raw_scores - self.min_score_) / range_score) * 100 if range_score != 0 else np.zeros_like(raw_scores)
            scaled_scores = np.clip(scaled_scores, 0, 100)
        else:
            scaled_scores = np.zeros_like(raw_scores)
            
        return {
            "anomaly_score": scaled_scores.tolist() if len(scaled_scores) > 1 else float(scaled_scores[0]),
            "inference_time_ms": (time.time() - start_time) * 1000
        }

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"COPOD successfully saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> 'COPODModel':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        return joblib.load(path)


def soft_ensemble_predict(
    features_df: pd.DataFrame,
    regime: str,
    isolation_forest_model: Any,
    conformal_predictor: Any,
    copod_model: Optional[COPODModel] = None
) -> Dict[str, Any]:
    """
    Executes soft ensemble logic.
    Applies 75% IF + 25% COPOD strictly in VOLATILE regime.
    """
    start_time = time.time()
    
    # 1. Base IF Prediction
    if_result = isolation_forest_model.predict(features_df)
    if_score = if_result["anomaly_score"]
    if isinstance(if_score, list):
        if_score = if_score[-1] # Ambil nilai terbaru jika format list
        
    final_score = float(if_score)
    used_ensemble = False
    
    # 2. Apply COPOD if regime is VOLATILE
    if regime.upper() == 'VOLATILE' and copod_model is not None and copod_model.model is not None:
        try:
            copod_result = copod_model.predict(features_df)
            copod_score = copod_result["anomaly_score"]
            if isinstance(copod_score, list):
                copod_score = copod_score[-1]
                
            # Soft weighting
            final_score = (0.75 * final_score) + (0.25 * float(copod_score))
            used_ensemble = True
            logger.debug(f"Applied Soft Ensemble: IF({if_score:.2f}) + COPOD({copod_score:.2f}) -> {final_score:.2f}")
        except Exception as e:
            logger.warning(f"COPOD prediction failed, falling back to IF. Error: {e}")
    
    # 3. Apply Conformal Prediction to final score
    cp_result = conformal_predictor.predict(final_score)
        
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "anomaly_score": final_score,
        "if_score": float(if_score),
        "used_ensemble": used_ensemble,
        "p_value": cp_result.get("p_value", 1.0),
        "confidence_level": cp_result.get("confidence_level", 0.0),
        "signal_strength": cp_result.get("signal_strength", "LOW"),
        "is_significant": cp_result.get("is_significant", False),
        "inference_time_ms": latency_ms
    }