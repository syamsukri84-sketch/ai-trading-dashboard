import pytest
import numpy as np
import os
from src.models.conformal_predictor import ConformalPredictor

@pytest.fixture
def historical_scores():
    """
    Generates 252 simulated IF anomaly scores (1 year of trading days).
    Mostly normal scores around 30, with a few clear anomalies around 80.
    """
    np.random.seed(42)
    normal_scores = np.random.normal(30, 10, 240)
    anomaly_scores = np.random.normal(80, 5, 12)
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    return np.clip(all_scores, 0, 100)

def test_calibration(historical_scores):
    cp = ConformalPredictor(alpha=0.05, target_fpr=0.08)
    cp.calibrate(historical_scores)
    
    assert len(cp.calibration_scores) == 252
    assert cp.threshold_score > 0
    assert cp.last_calibrated is not None

def test_p_values_valid(historical_scores):
    cp = ConformalPredictor(alpha=0.05)
    cp.calibrate(historical_scores)
    
    # Test a typical normal score
    result_normal = cp.predict(30.0)
    assert 0.0 <= result_normal["p_value"] <= 1.0
    assert result_normal["signal_strength"] == "LOW"
    assert not result_normal["is_significant"]
    
    # Test a high anomaly score
    result_anomaly = cp.predict(90.0)
    assert result_anomaly["p_value"] < 0.05
    assert result_anomaly["signal_strength"] in ["MODERATE", "HIGH"]
    assert result_anomaly["is_significant"]

def test_empirical_fpr(historical_scores):
    cp = ConformalPredictor(alpha=0.05)
    cp.calibrate(historical_scores)
    
    # Generate a hold-out test set of normal market data (no extreme anomalies)
    np.random.seed(99)
    test_scores = np.random.normal(30, 10, 1000)
    
    validation = cp.validate(test_scores)
    # Assuming test set has no true anomalies, the FPR should naturally be roughly alpha (~0.05)
    assert validation["empirical_fpr"] < 0.10  # Must strictly stay below target bounds
    assert validation["is_valid"] is True

def test_save_load_thresholds(historical_scores, tmp_path):
    cp = ConformalPredictor(alpha=0.05)
    cp.calibrate(historical_scores)
    
    save_path = tmp_path / "thresholds.json"
    cp.save_thresholds(str(save_path))
    assert os.path.exists(save_path)
    
    loaded_cp = ConformalPredictor.load_thresholds(str(save_path))
    assert loaded_cp.alpha == cp.alpha
    assert len(loaded_cp.calibration_scores) == len(cp.calibration_scores)
    assert loaded_cp.last_calibrated == cp.last_calibrated