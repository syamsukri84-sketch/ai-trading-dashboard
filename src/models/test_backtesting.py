import pytest
import pandas as pd
from src.models.backtest_engine import BacktestEngine

def test_calculate_metrics_empty():
    engine = BacktestEngine()
    metrics = engine.calculate_metrics([])
    assert metrics['total_trades'] == 0
    assert metrics['win_rate'] == 0.0

def test_calculate_metrics_valid():
    engine = BacktestEngine()
    
    # Simulated trades (PNL values)
    mock_trades = [
        {'pnl': 100.0}, {'pnl': -50.0}, 
        {'pnl': 150.0}, {'pnl': -20.0}
    ]
    
    # Simulated equity curve corresponding to trade progression
    equity_curve = [10000.0, 10100.0, 10050.0, 10200.0, 10180.0]
    
    metrics = engine.calculate_metrics(mock_trades, equity_curve)
    
    assert metrics['total_trades'] == 4
    assert metrics['win_rate'] == 0.5  # 2 wins, 2 losses
    assert metrics['profit_factor'] == pytest.approx(250.0 / 70.0) # Gross Profit / Gross Loss
    assert metrics['net_profit'] == 180.0
    assert metrics['max_drawdown'] < 0.0 # Drawdown must be a negative value

def test_aggregate_results():
    engine = BacktestEngine()
    windows = [
        {"total_trades": 10, "win_rate": 0.6, "sharpe_ratio": 1.5, "net_profit": 500},
        {"total_trades": 10, "win_rate": 0.4, "sharpe_ratio": 0.5, "net_profit": -100}
    ]
    agg = engine.aggregate_results(windows)
    assert agg["avg_win_rate"] == 0.5
    assert agg["total_profit"] == 400.0

def test_simulate_signal_strategy_long_only():
    engine = BacktestEngine()
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6),
        "close": [100, 102, 104, 106, 108, 110],
        "low": [99, 101, 103, 105, 107, 109],
        "feat_rsi_14": [35, 40, 45, 50, 55, 60],
        "feat_atr_14": [2, 2, 2, 2, 2, 2],
        "projected_return_pct": [3, 3, 3, 3, 3, 3],
        "anomaly_score": [75, 75, 75, 75, 75, 75],
        "p_value": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "ai_confidence_score": [70, 70, 70, 70, 70, 70],
    })

    result = engine.simulate_signal_strategy(df, initial_capital=1_000_000, risk_pct=1.0, holding_period=3)

    assert result["metrics"]["total_trades"] > 0
    assert result["metrics"]["ending_capital"] > 1_000_000
