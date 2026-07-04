import pandas as pd
import numpy as np
import logging
import joblib
import time
import os
from typing import Dict, Optional, Any
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class IsolationForestModel:
    """
    Isolation Forest module for detecting anomalies in stock momentum, volatility, and volume.
    Expected to train on 3-year rolling windows (756 trading days).
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200, random_state: int = 42, n_jobs: int = -1):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and prepares the feature DataFrame for the model.
        """
        X = df.copy()
        
        # Drop non-feature identifiers if present
        timestamp_cols = [c for c in X.columns if c.lower() in ['timestamp', 'date', 'datetime', 'ticker', 'symbol']]
        X = X.drop(columns=timestamp_cols, errors='ignore')
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        if X.isnull().values.any():
            raise ValueError("Features contain NaN values. Data must be cleaned before modeling.")
            
        return X

    def train(self, features_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Fits the Isolation Forest model on historical features.
        """
        if len(features_df) < 252:
            logger.warning(f"Training data is relatively small: {len(features_df)} rows. Expected ~756 rows for optimal results.")
            
        X = self._prepare_features(features_df)
        self.feature_names_ = X.columns.tolist()
        
        logger.info(f"Training Isolation Forest with {len(X)} samples and {len(self.feature_names_)} features...")
        start_time = time.time()
        
        # Fit the model
        self.model.fit(X.to_numpy(dtype=float, copy=False))
        
        # Calculate raw scores to define normalization boundaries [0, 100]
        train_array = X.to_numpy(dtype=float, copy=False)
        raw_scores = -self.model.decision_function(train_array) # Negate so higher value = more anomalous
        self.min_score_ = raw_scores.min()
        self.max_score_ = raw_scores.max()
        
        # Training Metrics
        train_predictions = self.model.predict(train_array)
        anomaly_count = (train_predictions == -1).sum()
        
        logger.info(f"Training completed in {time.time() - start_time:.3f}s. "
                    f"Found {anomaly_count} anomalies ({(anomaly_count/len(X))*100:.2f}%).")
        
        if save_path:
            self.save_model(save_path)

    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predicts anomalies and calculates metrics for live inference.
        Returns a dict with scaled score [0-100], raw anomaly label, and latency.
        """
        start_time = time.time()
        X = self._prepare_features(features_df)
        
        if hasattr(self, 'feature_names_'):
             missing = [col for col in self.feature_names_ if col not in X.columns]
             if missing:
                 raise ValueError(f"Missing features required by model: {missing}")
             X = X[self.feature_names_] # Ensure identical column order
             
        predict_array = X.to_numpy(dtype=float, copy=False)
        decision_scores = self.model.decision_function(predict_array)
        is_anomaly = np.where(decision_scores < 0, -1, 1)
        raw_scores = -decision_scores
        
        # Scale to [0, 100]
        if hasattr(self, 'min_score_') and hasattr(self, 'max_score_'):
            range_score = self.max_score_ - self.min_score_
            scaled_scores = ((raw_scores - self.min_score_) / range_score) * 100 if range_score != 0 else np.zeros_like(raw_scores)
            scaled_scores = np.clip(scaled_scores, 0, 100)
        else:
            scaled_scores = np.zeros_like(raw_scores)
            
        inference_time_ms = (time.time() - start_time) * 1000
        
        return {
            "anomaly_score": scaled_scores.tolist() if len(scaled_scores) > 1 else float(scaled_scores[0]),
            "is_anomaly": is_anomaly.tolist() if len(is_anomaly) > 1 else int(is_anomaly[0]),
            "inference_time_ms": float(inference_time_ms),
            "features_used": len(X.columns)
        }

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model successfully saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> 'IsolationForestModel':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        logger.info(f"Loading model from {path}")
        return joblib.load(path)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Approximates feature importance by counting usage frequency 
        across all isolation trees.
        """
        if not hasattr(self, 'feature_names_'):
            raise ValueError("Model is not trained yet")
            
        feature_counts = np.zeros(len(self.feature_names_))
        for estimator in self.model.estimators_:
            for feature in estimator.tree_.feature:
                if feature >= 0:  # -2 indicates a leaf node in the tree structure
                    feature_counts[feature] += 1
                    
        importance = feature_counts / feature_counts.sum()
        feat_imp = {self.feature_names_[i]: float(importance[i]) for i in range(len(self.feature_names_))}
        
        # Return sorted by importance descending
        return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
