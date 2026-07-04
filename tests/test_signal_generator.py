import pytest
from src.trading.signal_generator import generate_signal

def test_signal_generation_long():
    """Test oversold + high anomaly = LONG signal"""
    result = generate_signal(
        ticker="BBCA",
        current_price=10000.0,
        anomaly_score=75.0,
        p_value=0.01,
        regime="CALM",
        rsi=25.0, # Oversold
        atr=100.0
    )
    
    assert result['action'] == 'TRADE'
    setup = result['setup']
    assert setup['direction'] == 'LONG'
    assert setup['entry'] == 10000.0
    assert setup['stop_loss'] == 9800.0 # 10000 - (2 * 100)
    assert setup['take_profit_1'] == 10200.0 # 10000 + (2 * 100)
    assert setup['take_profit_2'] == 10350.0 # 10000 + (3.5 * 100)
    assert setup['risk_reward_ratio'] == 1.0 # (10200-10000) / (10000-9800)

def test_signal_generation_short():
    """Test overbought + high anomaly = SHORT signal"""
    result = generate_signal(
        ticker="BBRI",
        current_price=5000.0,
        anomaly_score=68.0,
        p_value=0.03,
        regime="VOLATILE", # Threshold for VOLATILE is 65
        rsi=80.0, # Overbought
        atr=50.0
    )
    
    assert result['action'] == 'TRADE'
    setup = result['setup']
    assert setup['direction'] == 'SHORT'
    assert setup['stop_loss'] == 5100.0 # 5000 + (2 * 50)
    assert setup['take_profit_1'] == 4900.0 # 5000 - (2 * 50)

def test_signal_skipped_low_confidence():
    """Test skipping when p_value > 0.05"""
    result = generate_signal(
        ticker="BMRI", current_price=6000.0, anomaly_score=80.0, 
        p_value=0.10, # Not significant
        regime="CALM", rsi=20.0, atr=50.0
    )
    assert result['action'] == 'SKIP'
    assert result['reason'] == 'low_confidence'

def test_signal_skipped_low_anomaly_for_regime():
    """Test skipping when anomaly is below regime threshold"""
    # CALM regime requires >= 70
    result = generate_signal(
        ticker="TLKM", current_price=3000.0, anomaly_score=65.0, 
        p_value=0.01, regime="CALM", rsi=20.0, atr=30.0
    )
    assert result['action'] == 'SKIP'
    assert result['reason'] == 'below_threshold'

def test_signal_skipped_ambiguous_rsi():
    """Test skipping when RSI is not overbought/oversold"""
    result = generate_signal(
        ticker="ASII", current_price=5000.0, anomaly_score=85.0, 
        p_value=0.01, regime="CRASH", rsi=50.0, # Ambiguous momentum
        atr=100.0
    )
    assert result['action'] == 'SKIP'
    assert result['reason'] == 'ambiguous_rsi'