import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering module for trading decision support.
    Generates 20+ technical indicators and ensures data quality 
    (no NaNs, infinite values, or highly correlated features).
    """
    
    def __init__(self, warmup_period: int = 60, drop_correlated: bool = True, correlation_threshold: float = 0.7):
        self.warmup_period = warmup_period
        self.drop_correlated = drop_correlated
        self.correlation_threshold = correlation_threshold

    def generate_features(self, df: pd.DataFrame, idx_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculates momentum, volatility, volume, correlation, and price action features.
        """
        if len(df) <= self.warmup_period:
            logger.error(f"Insufficient data. Minimum {self.warmup_period + 1} rows required.")
            return pd.DataFrame()
            
        out_df = df.copy()
        
        # Extract arrays for faster talib processing
        open_p = out_df['open'].values
        high = out_df['high'].values
        low = out_df['low'].values
        close = out_df['close'].values
        volume = out_df['volume'].astype(float).values
        
        # --- 1. MOMENTUM FEATURES ---
        delta = out_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        out_df['feat_rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        ema_12 = out_df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = out_df['close'].ewm(span=26, adjust=False).mean()
        out_df['feat_macd'] = ema_12 - ema_26
        out_df['feat_macd_signal'] = out_df['feat_macd'].ewm(span=9, adjust=False).mean()
        
        low_14 = out_df['low'].rolling(14).min()
        high_14 = out_df['high'].rolling(14).max()
        out_df['feat_stoch_k'] = 100 * ((out_df['close'] - low_14) / (high_14 - low_14 + 1e-8))
        out_df['feat_stoch_d'] = out_df['feat_stoch_k'].rolling(3).mean()
        
        ema_26 = out_df['close'].ewm(span=26, adjust=False).mean()
        out_df['feat_price_vs_ema'] = (out_df['close'] - ema_26) / (ema_26 + 1e-8)
        
        # Extra Momentum (to ensure robust variety)
        out_df['feat_roc_10'] = out_df['close'].pct_change(periods=10) * 100
        
        # --- 2. VOLATILITY FEATURES ---
        tr1 = out_df['high'] - out_df['low']
        tr2 = (out_df['high'] - out_df['close'].shift(1)).abs()
        tr3 = (out_df['low'] - out_df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        out_df['feat_true_range'] = tr
        
        out_df['feat_atr_14'] = tr.rolling(14).mean()
        atr_28 = tr.rolling(28).mean()
        out_df['feat_atr_ratio'] = out_df['feat_atr_14'] / (atr_28 + 1e-8)
        
        bb_middle = out_df['close'].rolling(20).mean()
        bb_std = out_df['close'].rolling(20).std()
        out_df['feat_bb_width'] = (4 * bb_std) / (bb_middle + 1e-8)
        
        hist_vol_20 = out_df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        out_df['feat_hist_vol_20'] = hist_vol_20
        hist_vol_60 = out_df['close'].pct_change().rolling(60).std() * np.sqrt(252)
        out_df['feat_vol_ratio'] = hist_vol_20 / (hist_vol_60 + 1e-8)
        
        # --- 3. VOLUME FEATURES ---
        volume_series = out_df['volume']
        out_df['feat_volume_ratio'] = volume_series / (volume_series.rolling(20).mean() + 1e-8)
        out_df['feat_obv'] = (np.sign(out_df['close'].diff()) * volume_series).fillna(0).cumsum()
        
        # Chaikin Money Flow (Custom logic approximation)
        mf_mult = ((out_df['close'] - out_df['low']) - (out_df['high'] - out_df['close'])) / (out_df['high'] - out_df['low'] + 1e-8)
        mf_vol = mf_mult * volume_series
        out_df['feat_cmf'] = mf_vol.rolling(20).sum() / (volume_series.rolling(20).sum() + 1e-8)
        
        out_df['feat_ad_line'] = mf_vol.cumsum()
        
        # Aggregated volume signal
        obv_pct = out_df['feat_obv'].pct_change().fillna(0)
        out_df['feat_agg_vol_signal'] = obv_pct + out_df['feat_volume_ratio']
        
        # --- 4. CORRELATION FEATURES ---
        if idx_df is not None and not idx_df.empty:
            idx_aligned = idx_df.copy()
            if "timestamp" in idx_aligned.columns and "timestamp" in out_df.columns:
                idx_aligned["timestamp"] = pd.to_datetime(idx_aligned["timestamp"], errors="coerce")
                idx_aligned = idx_aligned.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                idx_aligned = (
                    idx_aligned.set_index("timestamp")
                    .reindex(pd.to_datetime(out_df["timestamp"], errors="coerce"))
                    .ffill()
                    .reset_index(drop=True)
                )
            idx_ret = idx_aligned['close'].pct_change()
            stock_ret = out_df['close'].pct_change()
            out_df['feat_corr_60'] = stock_ret.rolling(60).corr(idx_ret)
            out_df['feat_beta_60'] = stock_ret.rolling(60).cov(idx_ret) / (idx_ret.rolling(60).var() + 1e-8)
            out_df['feat_ihsg_return_1d'] = idx_ret
            out_df['feat_ihsg_return_5d'] = idx_aligned['close'].pct_change(5)
            out_df['feat_ihsg_vol_20'] = idx_ret.rolling(20).std() * np.sqrt(252)
        else:
            logger.info("idx_df not provided, defaulting correlation features to 0 and beta to 1")
            out_df['feat_corr_60'] = 0.0
            out_df['feat_beta_60'] = 1.0
            out_df['feat_ihsg_return_1d'] = 0.0
            out_df['feat_ihsg_return_5d'] = 0.0
            out_df['feat_ihsg_vol_20'] = 0.0
            
        # --- 5. PRICE ACTION FEATURES ---
        prev_close = out_df['close'].shift(1)
        out_df['feat_gap'] = (out_df['open'] - prev_close) / (prev_close + 1e-8)
        
        # --- CLEANUP & VALIDATION ---
        # Remove warmup period
        out_df = out_df.iloc[self.warmup_period:].copy()
        
        out_df = out_df.ffill().fillna(0)
        out_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Drop highly correlated features
        feat_cols = [col for col in out_df.columns if col.startswith('feat_')]
        if self.drop_correlated:
            corr_matrix = out_df[feat_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
            if to_drop:
                logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
                out_df.drop(columns=to_drop, inplace=True)
                
        return out_df
