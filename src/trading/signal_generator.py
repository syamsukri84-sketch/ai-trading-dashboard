import logging
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

def generate_signal(
    ticker: str,
    current_price: float,
    anomaly_score: float,
    p_value: float,
    regime: str,
    rsi: float,
    atr: float
) -> Dict[str, Any]:
    """
    Converts an anomaly score into an actionable trading setup based on confidence,
    market regime, and momentum indicators.
    """
    try:
        # 1. FILTER by confidence (Statistical Significance)
        if p_value > 0.05:
            logger.debug(f"[{ticker}] SKIP: Low confidence (p_value={p_value:.4f} > 0.05)")
            return {'action': 'SKIP', 'reason': 'low_confidence'}
            
        # 2. FILTER by magnitude (Regime-Adjusted Anomaly Thresholds)
        regime_upper = regime.upper()
        thresholds = {
            'CALM': 70.0,
            'VOLATILE': 65.0,
            'CRASH': 60.0
        }
        # Default to most conservative if regime is unknown
        threshold = thresholds.get(regime_upper, 70.0) 
        
        if anomaly_score < threshold:
            logger.debug(f"[{ticker}] SKIP: Anomaly score too low for {regime_upper} regime ({anomaly_score:.2f} < {threshold})")
            return {'action': 'SKIP', 'reason': 'below_threshold'}
            
        # 3. DETERMINE direction (Momentum anomaly implies reversal)
        if rsi > 70:
            direction = 'SHORT'
        elif rsi < 30:
            direction = 'LONG'
        else:
            logger.debug(f"[{ticker}] SKIP: Ambiguous RSI ({rsi:.2f} is not overbought/oversold)")
            return {'action': 'SKIP', 'reason': 'ambiguous_rsi'}
            
        # 4. GENERATE setup (Risk-based position and targets)
        if atr <= 0:
             logger.warning(f"[{ticker}] SKIP: Invalid ATR ({atr})")
             return {'action': 'SKIP', 'reason': 'invalid_atr'}
             
        if direction == 'LONG':
            stop_loss = current_price - (2 * atr)
            take_profit_1 = current_price + (2 * atr)
            take_profit_2 = current_price + (3.5 * atr)
        else: # SHORT
            stop_loss = current_price + (2 * atr)
            take_profit_1 = current_price - (2 * atr)
            take_profit_2 = current_price - (3.5 * atr)
            
        risk_per_share = abs(current_price - stop_loss)
        reward_per_share = abs(take_profit_1 - current_price)
        
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
        
        setup = {
            'ticker': ticker,
            'direction': direction,
            'entry': float(current_price),
            'stop_loss': float(stop_loss),
            'take_profit_1': float(take_profit_1),
            'take_profit_2': float(take_profit_2),
            'position_size_pct': 2.0, # Target allocating 2% of portfolio per spec
            'risk_reward_ratio': float(risk_reward_ratio),
            'confidence': float(1.0 - p_value),
            'anomaly_score': float(anomaly_score),
            'regime': regime_upper
        }
        
        logger.info(f"[{ticker}] TRADE SIGNAL: {direction} at {current_price:.2f} "
                    f"(Score: {anomaly_score:.1f}, Regime: {regime_upper})")
                    
        return {
            'action': 'TRADE',
            'setup': setup
        }
        
    except Exception as e:
        logger.error(f"[{ticker}] Error generating signal: {str(e)}")
        return {'action': 'SKIP', 'reason': 'calculation_error'}