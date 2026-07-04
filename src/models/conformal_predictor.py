import numpy as np
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

class ConformalPredictor:
    """
    Conformal Prediction framework to calibrate Isolation Forest scores
    and reduce false positive trading signals using statistical guarantees.
    """
    
    def __init__(self, alpha: float = 0.05, target_fpr: float = 0.08):
        self.alpha = alpha
        self.target_fpr = target_fpr
        self.calibration_scores = np.array([])
        self.threshold_score = 0.0
        self.version = "1.0"
        self.last_calibrated = None

    def calibrate(self, historical_scores: Union[List[float], np.ndarray], target_fpr: float = 0.08) -> None:
        """
        Calibrates the predictor using historical anomaly scores (typically the last 252 days).
        """
        if len(historical_scores) < 100:
            logger.warning(f"Calibration data is small ({len(historical_scores)}). Conformal guarantees might be weak.")
            
        self.target_fpr = target_fpr
        self.calibration_scores = np.sort(np.array(historical_scores))
        
        # Calculate the threshold score where p-value roughly equals alpha.
        # This uses the (1 - alpha) empirical quantile of the calibration scores.
        idx = int(np.ceil((1 - self.alpha) * len(self.calibration_scores)))
        idx = min(idx, len(self.calibration_scores) - 1)
        self.threshold_score = float(self.calibration_scores[idx])
        
        self.last_calibrated = datetime.now().isoformat()
        logger.info(f"Calibrated Conformal Predictor. Threshold score for alpha={self.alpha} is {self.threshold_score:.2f}")

    def _calc_p_value(self, score: float) -> float:
        """
        Calculates the probability of observing an anomaly score this high or higher
        based on the calibration distribution.
        """
        if len(self.calibration_scores) == 0:
            return 1.0 # Max uncertainty if not calibrated
            
        count_greater_equal = np.sum(self.calibration_scores >= score)
        p_value = (count_greater_equal + 1) / (len(self.calibration_scores) + 1)
        return float(p_value)

    def predict(self, anomaly_score: float) -> Dict[str, Any]:
        """
        Converts a raw anomaly score into a statistically backed signal.
        Returns p-value, confidence level, and signal strength.
        """
        if len(self.calibration_scores) == 0:
            raise ValueError("ConformalPredictor is not calibrated. Call calibrate() first.")
            
        p_value = self._calc_p_value(anomaly_score)
        confidence_level = 1.0 - p_value
        
        # Signal strength based on statistical significance
        if p_value < (self.alpha / 2):
            signal_strength = 'HIGH'
        elif p_value < self.alpha:
            signal_strength = 'MODERATE'
        else:
            signal_strength = 'LOW'
            
        # Confidence bounds based on the calibration distribution context
        ci_lower = float(np.percentile(self.calibration_scores, 5))
        ci_upper = float(np.percentile(self.calibration_scores, 95))
        
        return {
            "p_value": p_value,
            "confidence_level": confidence_level,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "signal_strength": signal_strength,
            "false_positive_risk": p_value, # Empirical FPR risk estimate for this exact score
            "is_significant": p_value < self.alpha
        }

    def validate(self, test_scores: Union[List[float], np.ndarray]) -> Dict[str, Union[float, bool]]:
        """
        Validates the empirical FPR against a set of hold-out or ongoing test scores.
        """
        if len(self.calibration_scores) == 0:
            raise ValueError("Model not calibrated.")
            
        test_scores = np.array(test_scores)
        p_values = np.array([self._calc_p_value(s) for s in test_scores])
        empirical_fpr = np.mean(p_values < self.alpha)
        
        logger.info(f"Validation Empirical FPR: {empirical_fpr:.4f} (Target: {self.target_fpr})")
        return {
            "empirical_fpr": float(empirical_fpr),
            "target_fpr": self.target_fpr,
            "is_valid": bool(empirical_fpr <= self.target_fpr + 0.02) # Allowing small empirical variance
        }

    def save_thresholds(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            "alpha": self.alpha,
            "target_fpr": self.target_fpr,
            "threshold_score": self.threshold_score,
            "calibration_scores": self.calibration_scores.tolist(),
            "version": self.version,
            "last_calibrated": self.last_calibrated
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Conformal thresholds saved to {path}")

    @classmethod
    def load_thresholds(cls, path: str) -> 'ConformalPredictor':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Thresholds file not found: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        predictor = cls(alpha=data.get("alpha", 0.05), target_fpr=data.get("target_fpr", 0.08))
        predictor.threshold_score = data.get("threshold_score", 0.0)
        predictor.calibration_scores = np.array(data.get("calibration_scores", []))
        predictor.version = data.get("version", "1.0")
        predictor.last_calibrated = data.get("last_calibrated")
        
        logger.info(f"Loaded conformal thresholds from {path}")
        return predictor