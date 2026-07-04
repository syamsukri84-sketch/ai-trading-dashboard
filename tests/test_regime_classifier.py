import pytest
import pandas as pd
import numpy as np
import os
from src.models.regime_classifier import RegimeClassifier

@pytest.fixture
def mock_market_data():
    """Generates synthetic OHLCV data for regime feature calculation."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    
    # Create trending data to get a stable ADX and returns
    idx_close = np.linspace(100, 110, 100) + np.random.normal(0, 1, 100)
    idx_data = pd.DataFrame({'close': idx_close}, index=dates)
    
    lq45_close = np.linspace(50, 55, 100) + np.random.normal(0, 0.5, 100)
    lq45_data = pd.DataFrame({
        'open': lq45_close * np.random.uniform(0.99, 1.01, 100),
        'high': lq45_close * 1.02,
        'low': lq45_close * 0.98,
        'close': lq45_close,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    return idx_data, lq45_data

def test_regime_classification_logic():
    classifier = RegimeClassifier()
    
    # Test CALM conditions: VIX < 15, ADX > 25, corr_shift < 0.05
    res_calm = classifier.classify({"vix_proxy": 12.0, "adx": 30.0, "correlation_shift": 0.02, "volume_regime": 1.0})
    assert res_calm["regime"] == "CALM"
    assert res_calm["model_to_use"] == "isolation_forest_only"
    
    # Test CRASH conditions: VIX > 25, corr_shift > 0.10
    res_crash = classifier.classify({"vix_proxy": 30.0, "adx": 15.0, "correlation_shift": 0.15, "volume_regime": 1.5})
    assert res_crash["regime"] == "CRASH"
    assert res_crash["model_to_use"] == "robust_mahalanobis"
    
    # Test VOLATILE conditions: Anything else
    res_vol = classifier.classify({"vix_proxy": 18.0, "adx": 20.0, "correlation_shift": 0.08, "volume_regime": 1.2})
    assert res_vol["regime"] == "VOLATILE"
    assert res_vol["model_to_use"] == "isolation_forest_soft_ensemble"

def test_calculate_features(mock_market_data):
    idx_data, lq45_data = mock_market_data
    classifier = RegimeClassifier()
    
    features = classifier.calculate_regime_features(idx_data, lq45_data)
    
    assert "vix_proxy" in features
    assert "adx" in features
    assert "correlation_shift" in features
    assert "volume_regime" in features
    assert isinstance(features["vix_proxy"], float)

def test_regime_history_tracking(tmp_path):
    history_path = tmp_path / "test_regime_history.csv"
    classifier = RegimeClassifier(history_path=str(history_path))
    
    classifier.classify({"vix_proxy": 12.0, "adx": 30.0, "correlation_shift": 0.02})
    classifier.classify({"vix_proxy": 30.0, "adx": 15.0, "correlation_shift": 0.15})
    classifier.save_regime_history()
    
    assert os.path.exists(history_path)
    saved_df = pd.read_csv(history_path)
    assert len(saved_df) == 2
    assert list(saved_df['regime']) == ['CALM', 'CRASH']