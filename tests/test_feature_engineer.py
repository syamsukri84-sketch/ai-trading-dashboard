import pytest
import pandas as pd
import numpy as np
from src.data_pipeline.feature_engineer import FeatureEngineer

@pytest.fixture
def sample_ohlcv():
    """Creates a realistic OHLCV dataframe for feature generation."""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1h')
    np.random.seed(42)
    
    # Create realistic-looking continuous price data
    base_price = 100
    returns = np.random.normal(0, 0.01, 200)
    close = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close * np.random.uniform(0.99, 1.01, 200),
        'close': close,
        'volume': np.random.randint(1000, 10000, 200)
    })
    df['high'] = df[['open', 'close']].max(axis=1) * np.random.uniform(1.0, 1.02, 200)
    df['low'] = df[['open', 'close']].min(axis=1) * np.random.uniform(0.98, 1.0, 200)
    return df

def test_all_features_calculated(sample_ohlcv):
    engineer = FeatureEngineer(warmup_period=60, drop_correlated=False)
    result = engineer.generate_features(sample_ohlcv)
    feat_cols = [c for c in result.columns if c.startswith('feat_')]
    assert len(feat_cols) >= 20, "Should generate at least 20 features"

def test_no_nan_in_features(sample_ohlcv):
    engineer = FeatureEngineer(warmup_period=60, drop_correlated=False)
    result = engineer.generate_features(sample_ohlcv)
    assert result.isnull().sum().sum() == 0, "There should be no NaN values in the output"
    assert not np.isinf(result.select_dtypes(include=[np.number])).values.any(), "There should be no infinite values"

def test_minimum_lookback(sample_ohlcv):
    engineer = FeatureEngineer(warmup_period=60, drop_correlated=False)
    result = engineer.generate_features(sample_ohlcv)
    # Expected rows: Total (200) - Warmup (60) = 140
    assert len(result) == len(sample_ohlcv) - 60

def test_feature_correlation(sample_ohlcv):
    engineer = FeatureEngineer(warmup_period=60, drop_correlated=True, correlation_threshold=0.90)
    result = engineer.generate_features(sample_ohlcv)
    feat_cols = [c for c in result.columns if c.startswith('feat_')]
    corr_matrix = result[feat_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    assert not any(upper.max() > 0.90), "Highly correlated features should be dropped"
    
def test_feature_range_validation(sample_ohlcv):
    engineer = FeatureEngineer(warmup_period=60, drop_correlated=False)
    result = engineer.generate_features(sample_ohlcv)
    assert result['feat_rsi_14'].max() <= 100
    assert result['feat_rsi_14'].min() >= 0