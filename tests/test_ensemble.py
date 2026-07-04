import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.models.ensemble import soft_ensemble_predict

@pytest.fixture
def mock_if_model():
    model = MagicMock()
    # Kita set IF score menjadi 80.0
    model.predict.return_value = {"anomaly_score": 80.0, "inference_time_ms": 5.0}
    return model

@pytest.fixture
def mock_copod_model():
    model = MagicMock()
    model.model = MagicMock()  # Simulate that pyod is installed
    # Kita set COPOD score menjadi 40.0
    model.predict.return_value = {"anomaly_score": 40.0, "inference_time_ms": 5.0}
    return model

@pytest.fixture
def mock_cp():
    cp = MagicMock()
    cp.predict.return_value = {
        "p_value": 0.02,
        "confidence_level": 0.98,
        "signal_strength": "HIGH",
        "is_significant": True
    }
    return cp

def test_ensemble_volatile_regime(mock_if_model, mock_copod_model, mock_cp):
    df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    
    result = soft_ensemble_predict(df, "VOLATILE", mock_if_model, mock_cp, mock_copod_model)
    
    # Hitungan pembobotan: (0.75 * 80.0) + (0.25 * 40.0) = 60 + 10 = 70.0
    assert result["anomaly_score"] == 70.0
    assert result["used_ensemble"] is True
    assert result["is_significant"] is True
    assert result["if_score"] == 80.0

def test_ensemble_calm_regime(mock_if_model, mock_copod_model, mock_cp):
    df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    
    result = soft_ensemble_predict(df, "CALM", mock_if_model, mock_cp, mock_copod_model)
    
    # Karena CALM, model harus mengabaikan COPOD dan memakai IF saja (80.0)
    assert result["anomaly_score"] == 80.0
    assert result["used_ensemble"] is False