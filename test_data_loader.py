import pytest
import pandas as pd
import numpy as np
from data_loader import DataLoader # Import directly from the root file
from unittest.mock import patch

@pytest.fixture
def sample_valid_data():
    """Creates a valid sample DataFrame of 800 rows (satisfies min_rows=756)"""
    dates = pd.date_range(start='2023-01-01', periods=800, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 800),
        'high': np.random.uniform(150, 250, 800),
        'low': np.random.uniform(50, 150, 800),
        'close': np.random.uniform(100, 200, 800),
        'volume': np.random.randint(1000, 10000, 800)
    })
    return df

@pytest.fixture
def data_loader():
    return DataLoader(min_rows=756)

@patch('data_loader.pd.read_csv')
@patch('data_loader.os.path.exists')
def test_successful_load(mock_exists, mock_read_csv, data_loader, sample_valid_data):
    # Setup mock to simulate file existing and reading from CSV
    mock_exists.return_value = True
    mock_read_csv.return_value = sample_valid_data.copy()
    
    # Test the load functionality
    result = data_loader.load_data('BBCA')
    
    assert result is not None
    assert len(result) == 800
    assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    mock_read_csv.assert_called()

@patch('data_loader.os.path.exists')
def test_missing_local_file_returns_none(mock_exists, data_loader):
    mock_exists.return_value = False # Simulate file not found
    result = data_loader.load_data('INVALID_TICKER')
    assert result is None

def test_validation_missing_values(data_loader, sample_valid_data):
    df_with_nan = sample_valid_data.copy()
    df_with_nan.loc[5, 'close'] = np.nan
    assert data_loader.validate_data(df_with_nan) is False

def test_validation_duplicate_timestamps(data_loader, sample_valid_data):
    df_duplicate_time = sample_valid_data.copy()
    df_duplicate_time.loc[5, 'timestamp'] = df_duplicate_time.loc[4, 'timestamp']
    assert data_loader.validate_data(df_duplicate_time) is False

def test_validation_insufficient_rows(data_loader, sample_valid_data):
    df_small = sample_valid_data.iloc[:500].copy()
    assert data_loader.validate_data(df_small) is False

def test_validation_negative_values(data_loader, sample_valid_data):
    df_negative = sample_valid_data.copy()
    df_negative.loc[10, 'volume'] = -100
    assert data_loader.validate_data(df_negative) is False

def test_parse_investing_csv(data_loader):
    """Test parsing logic for Investing.com Indonesian CSV format"""
    # Mock data format as exported from id.investing.com
    df_investing = pd.DataFrame({
        'Tanggal': ['16.06.2026', '15.06.2026'],  # Note descending order
        'Terakhir': ['5.000,50', '4.900,00'],     # Indonesian decimal/thousands
        'Buka': ['4.950,00', '4.850,00'],
        'Tinggi': ['5.050,00', '4.950,00'],
        'Rendah': ['4.900,00', '4.800,00'],
        'Vol.': ['12,5M', '800K'],                # M = million, K = thousand
        'Perubahan%': ['2,05%', '1,03%']
    })
    
    parsed_df = data_loader._parse_investing_csv(df_investing)
    
    # Assertions
    assert len(parsed_df) == 2
    assert list(parsed_df.columns) == ['timestamp', 'close', 'open', 'high', 'low', 'volume', 'Perubahan%']
    # Check chronological sorting (oldest first)
    assert parsed_df['close'].iloc[1] == 5000.50  # 16.06.2026 should be at index 1 now
    assert parsed_df['close'].iloc[0] == 4900.00
    assert parsed_df['volume'].iloc[1] == 12500000.0  # 12,5M parsed to float
    assert parsed_df['volume'].iloc[0] == 800000.0    # 800K parsed to float
