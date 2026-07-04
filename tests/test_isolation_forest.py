import pytest
import pandas as pd
import numpy as np
import os
from src.models.isolation_forest import IsolationForestModel

@pytest.fixture
def sample_features():
    """Generates 800 rows of random features simulating technical indicators."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=800, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'feat_rsi_14': np.random.uniform(20, 80, 800),
        'feat_macd': np.random.normal(0, 1, 800),
        'feat_atr_14': np.random.uniform(0.5, 2.0, 800)
    })
    # Inject clear anomalies
    df.loc[10, 'feat_rsi_14'] = 150
    df.loc[10, 'feat_macd'] = 15.0
    return df

def test_train_basic(sample_features):
    model = IsolationForestModel()
    model.train(sample_features)
    assert hasattr(model, 'model')
    assert hasattr(model, 'feature_names_')
    assert len(model.feature_names_) == 3  # timestamp should be dropped

def test_train_with_small_data(sample_features, caplog):
    model = IsolationForestModel()
    small_df = sample_features.head(100)
    model.train(small_df)
    assert "Training data is relatively small" in caplog.text

def test_train_with_nan_values(sample_features):
    model = IsolationForestModel()
    df_nan = sample_features.copy()
    df_nan.loc[5, 'feat_macd'] = np.nan
    with pytest.raises(ValueError, match="contain NaN"):
        model.train(df_nan)

def test_inference_returns_valid_scores(sample_features):
    model = IsolationForestModel()
    model.train(sample_features)
    
    test_df = sample_features.tail(10)
    result = model.predict(test_df)
    
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert "inference_time_ms" in result
    
    scores = result["anomaly_score"]
    # Ensure normalized outputs fall between 0 and 100
    assert all(0 <= s <= 100 for s in scores)

def test_inference_latency(sample_features):
    model = IsolationForestModel()
    model.train(sample_features)
    
    # Create 1000 samples for latency profiling
    large_test_df = pd.concat([sample_features] * 2).head(1000)
    
    result = model.predict(large_test_df)
    # Constraint dictates processing fast enough to handle stocks simultaneously. 
    # Typical Python threshold in CI < 50ms for 1000 batch inference of this size.
    assert result["inference_time_ms"] < 50.0 

def test_save_load_model(sample_features, tmp_path):
    model = IsolationForestModel()
    model.train(sample_features)
    
    save_path = tmp_path / "model.pkl"
    model.save_model(str(save_path))
    assert os.path.exists(save_path)
    
    loaded_model = IsolationForestModel.load_model(str(save_path))
    assert hasattr(loaded_model, 'feature_names_')
    
    # Check reproducibility after load
    test_df = sample_features.tail(5)
    np.testing.assert_array_almost_equal(model.predict(test_df)["anomaly_score"], 
                                         loaded_model.predict(test_df)["anomaly_score"])
                                         
def test_get_feature_importance(sample_features):
    model = IsolationForestModel()
    model.train(sample_features)
    importance = model.get_feature_importance()
    
    assert len(importance) == 3
    assert sum(importance.values()) == pytest.approx(1.0)
    assert all(k.startswith('feat_') for k in importance.keys())