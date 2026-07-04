import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Walk-forward backtesting engine for trading strategies.
    Simulates trading over historical data to evaluate performance metrics.
    """
    
    def __init__(self, train_period: int = 504, test_period: int = 63, step: int = 1):
        self.train_period = train_period
        self.test_period = test_period
        self.step = step
        
    def calculate_metrics(self, trades: List[Dict[str, float]], equity_curve: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculates key trading metrics from a list of executed trades.
        Expected trade format: {'pnl': 150.0, 'duration': 3, ...}
        """
        if not trades:
            logger.warning("No trades executed in this period.")
            return {"total_trades": 0, "win_rate": 0.0, "sharpe_ratio": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}
            
        profits = [t.get('pnl', 0.0) for t in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Approximate Sharpe Ratio (assuming daily returns distribution)
        returns = pd.Series(profits)
        if len(returns) > 1 and returns.std() != 0:
            # Multiply by sqrt(252) assuming trade metrics reflect daily active trading
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0.0
            
        # Calculate Max Drawdown from equity curve
        max_drawdown = 0.0
        if equity_curve and len(equity_curve) > 0:
            eq_series = pd.Series(equity_curve)
            rolling_max = eq_series.cummax()
            drawdowns = (eq_series - rolling_max) / (rolling_max + 1e-8)
            max_drawdown = drawdowns.min()
            
        return {
            "total_trades": len(trades),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "net_profit": float(sum(profits))
        }

    def simulate_signal_strategy(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000_000.0,
        risk_pct: float = 1.0,
        holding_period: int = 3,
        min_confidence_score: float = 55.0,
    ) -> Dict[str, Any]:
        """
        Backtest sederhana untuk strategi long-only berbasis decision score.
        Didesain sebagai sanity check dashboard, bukan execution simulator penuh.
        """
        required_cols = [
            "timestamp",
            "close",
            "feat_rsi_14",
            "feat_atr_14",
            "projected_return_pct",
            "anomaly_score",
            "p_value",
            "ai_confidence_score",
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for backtest: {missing}")

        data = df.sort_values("timestamp").reset_index(drop=True)
        capital = float(initial_capital)
        equity_curve = [capital]
        trades: List[Dict[str, Any]] = []

        for idx in range(0, len(data) - holding_period):
            row = data.iloc[idx]
            if row["ai_confidence_score"] < min_confidence_score:
                continue
            if row["projected_return_pct"] <= 0:
                continue
            if row["p_value"] > 0.05:
                continue
            if row["feat_rsi_14"] > 70:
                continue

            entry = float(row["close"])
            atr = float(row["feat_atr_14"])
            if entry <= 0 or atr <= 0:
                continue

            stop_loss = entry - (2 * atr)
            risk_per_share = entry - stop_loss
            risk_amount = capital * risk_pct / 100.0
            shares = int(risk_amount // risk_per_share)
            shares = (shares // 100) * 100
            if shares <= 0:
                continue

            future = data.iloc[idx + 1: idx + holding_period + 1]
            exit_price = float(future["close"].iloc[-1])
            exit_reason = "horizon"

            stop_hits = future[future["low"] <= stop_loss] if "low" in future.columns else pd.DataFrame()
            if not stop_hits.empty:
                exit_price = float(stop_loss)
                exit_reason = "stop_loss"

            pnl = (exit_price - entry) * shares
            capital += pnl
            equity_curve.append(capital)

            trades.append({
                "entry_date": row["timestamp"],
                "exit_date": future["timestamp"].iloc[-1],
                "entry": entry,
                "exit": exit_price,
                "shares": shares,
                "pnl": float(pnl),
                "return_pct": float((exit_price / entry - 1.0) * 100.0),
                "exit_reason": exit_reason,
                "confidence_score": float(row["ai_confidence_score"]),
            })

        metrics = self.calculate_metrics(trades, equity_curve)
        metrics["ending_capital"] = float(capital)
        metrics["return_pct"] = float(((capital / initial_capital) - 1.0) * 100.0) if initial_capital else 0.0

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def aggregate_results(self, all_windows: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregates metrics across multiple walk-forward windows.
        """
        if not all_windows:
            return {}
            
        total_trades = sum([w.get("total_trades", 0) for w in all_windows])
        if total_trades == 0:
            return {"avg_win_rate": 0.0, "avg_sharpe": 0.0, "total_profit": 0.0}
            
        # Weight win rates by the number of trades in each window
        weighted_wins = sum([w.get("win_rate", 0) * w.get("total_trades", 0) for w in all_windows])
        avg_win_rate = weighted_wins / total_trades
        
        # Simple average for Sharpe
        valid_sharpes = [w.get("sharpe_ratio", 0) for w in all_windows if w.get("sharpe_ratio", 0) != 0]
        avg_sharpe = np.mean(valid_sharpes) if valid_sharpes else 0.0
        
        total_profit = sum([w.get("net_profit", 0) for w in all_windows])
        
        return {
            "total_trades": total_trades,
            "avg_win_rate": float(avg_win_rate),
            "avg_sharpe": float(avg_sharpe),
            "total_profit": float(total_profit)
        }
