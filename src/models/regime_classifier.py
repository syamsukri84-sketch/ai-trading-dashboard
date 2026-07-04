import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

class RegimeClassifier:
    """
    Market Regime Classifier to adapt model selection per market conditions.
    Regimes: CALM, VOLATILE, CRASH
    """
    
    def __init__(self, history_path: str = "data/training_metadata/regime_history.csv"):
        self.history_path = history_path
        self.regime_history = []

    def calculate_regime_features(self, idx_data: pd.DataFrame, lq45_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates features needed for market regime classification.
        idx_data: Dataframe representing the broader market index (e.g. IDX Composite)
        lq45_data: Dataframe representing the specific stock or sector index
        """
        try:
            # Need sufficient data for rolling calculations
            if len(lq45_data) < 60 or len(idx_data) < 60:
                logger.warning("Insufficient data to calculate full long-term regime features. Using shorter windows where necessary.")

            # 1. VIX Proxy (Annualized volatility of returns * 100)
            returns = lq45_data['close'].pct_change().dropna()
            vix_proxy = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
            
            # 2. ADX (Trend Strength)
            # Custom Pandas-based Trend Strength proxy (0-100) to replace TA-Lib ADX
            close_series = lq45_data['close']
            trend_proxy = 100 * abs(close_series - close_series.shift(14)) / (lq45_data['high'].rolling(14).max() - lq45_data['low'].rolling(14).min() + 1e-8)
            adx = float(trend_proxy.iloc[-1]) if not np.isnan(trend_proxy.iloc[-1]) else 20.0
            
            # 3. Correlation Shift
            idx_returns = idx_data['close'].pct_change().dropna()
            
            # Align indices to ensure correlation calculation is valid
            aligned_returns, aligned_idx = returns.align(idx_returns, join='inner')
            
            corr_20 = aligned_returns.rolling(20).corr(aligned_idx)
            corr_60 = aligned_returns.rolling(60).corr(aligned_idx)
            
            current_corr_20 = corr_20.iloc[-1] if not np.isnan(corr_20.iloc[-1]) else 0.5
            current_corr_60 = corr_60.iloc[-1] if len(corr_60) >= 60 and not np.isnan(corr_60.iloc[-1]) else current_corr_20
            correlation_shift = abs(current_corr_20 - current_corr_60)
            
            # 4. Volume Regime
            vol_sma_20 = lq45_data['volume'].rolling(20).mean().iloc[-1]
            vol_sma_60 = lq45_data['volume'].rolling(60).mean().iloc[-1] if len(lq45_data) >= 60 else vol_sma_20
            volume_regime = vol_sma_20 / (vol_sma_60 + 1e-8)
            
            return {
                "vix_proxy": float(vix_proxy),
                "adx": float(adx),
                "correlation_shift": float(correlation_shift),
                "volume_regime": float(volume_regime)
            }
        except Exception as e:
            logger.error(f"Error calculating regime features: {str(e)}. Fallback to default.")
            return {
                "vix_proxy": 20.0,
                "adx": 20.0,
                "correlation_shift": 0.08,
                "volume_regime": 1.0
            }

    def classify(self, features: Dict[str, float]) -> Dict[str, Union[str, float]]:
        """
        Classifies the current market regime based on calculated features.
        """
        vix = features.get("vix_proxy", 20.0)
        adx = features.get("adx", 20.0)
        corr_shift = features.get("correlation_shift", 0.08)
        
        # Classification Logic based on Constraints
        if vix < 15 and adx > 25 and corr_shift < 0.05:
            regime = 'CALM'
            model_to_use = 'isolation_forest_only'
            confidence = 0.85
        elif vix > 25 and corr_shift > 0.10:
            regime = 'CRASH'
            model_to_use = 'robust_mahalanobis'
            confidence = 0.90
        else:
            regime = 'VOLATILE'
            model_to_use = 'isolation_forest_soft_ensemble'
            confidence = 0.75
            
        result = {
            "regime": regime,
            "model_to_use": model_to_use,
            "confidence": confidence,
            "features": features,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        self.regime_history.append(result)
        logger.info(f"Market Regime classified as {regime} (Model Strategy: {model_to_use})")
        
        return result

    def save_regime_history(self) -> None:
        """Tracks and exports regime changes to a CSV file."""
        if not self.regime_history:
            return
            
        os.makedirs(os.path.dirname(self.history_path) or '.', exist_ok=True)
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'features'} | r['features'] for r in self.regime_history])
        df.to_csv(self.history_path, mode='a', header=not os.path.exists(self.history_path), index=False)
        self.regime_history.clear() # Clear memory after saving