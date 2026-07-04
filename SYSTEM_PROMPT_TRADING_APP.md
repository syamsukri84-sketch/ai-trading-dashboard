# SYSTEM PROMPT: Trading Decision-Support Application
## Context untuk AI Coding Assistant (Copilot, Claude, Codeium)

**Last Updated**: June 2026  
**Project Status**: MVP Phase 1 - Isolation Forest + Streamlit  
**Developer**: Syamsukri (Data Science Master's Student)

---

## 🎯 PROJECT OVERVIEW

### Business Requirements
- **Purpose**: Real-time anomaly detection system untuk swing trading di pasar saham Indonesia
- **Target Market**: LQ45 stocks (~45 equities)
- **User Type**: Retail + Institutional traders
- **Trading Style**: Swing trading (1-hour bars, hold 1-5 days)
- **Deployment**: Streamlit (frontend) + FastAPI (backend) + SQLite/PostgreSQL (database)

### Technical Constraints (MUST FOLLOW)
```
Latency:          ≤100ms per inference (45 stocks parallel)
Data Frequency:   1-hour OHLCV bars
Retraining:       Daily (end of day)
Historical Data:  3 years minimum
False Positive:   "Better to miss signal" (target FPR < 8%)
Explainability:   Statistical rigor for institutions + simple for retail
```

### Architecture Decision
```
CHOSEN APPROACH: Isolation Forest (PRIMARY) + Conformal Prediction + Market Regime Classification

REJECTED: 4-model voting ensemble (IF + LOF + COPOD + XGBOD)
REASON: Latency 140-180ms exceeds 100ms SLA, high false positives, complex inference

PRIMARY MODEL:     Isolation Forest (8-12ms per stock inference)
UNCERTAINTY LAYER: Conformal Prediction (2-5ms overhead)
CONTEXT LAYER:     Market Regime Classification (VIX proxy, ADX, correlation)
SOFT ENSEMBLE:     IF + COPOD only in VOLATILE regime (18ms)
```

---

## 📋 PROJECT STRUCTURE

```
trading-decision-support/
├── config/
│   ├── config.yaml                 # App configuration (paths, API keys, thresholds)
│   ├── feature_config.yaml         # Technical indicators + parameters
│   └── model_config.yaml           # IF, conformal, regime classifier params
│
├── data/
│   ├── raw/                        # Downloaded OHLCV from yfinance
│   ├── processed/                  # Feature-engineered datasets
│   ├── models/                     # Saved pickle/joblib models
│   │   ├── isolation_forest_model.pkl
│   │   ├── regime_classifier.pkl
│   │   └── conformal_predictor.pkl
│   └── training_metadata/          # Regime stats, conformal thresholds
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Download OHLCV, validation
│   │   ├── feature_engineer.py     # 30+ technical indicators
│   │   └── data_validator.py       # Check NaN, outliers, data quality
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── isolation_forest.py     # IF training + inference
│   │   ├── regime_classifier.py    # Market regime classification
│   │   ├── conformal_predictor.py  # CP framework for uncertainty
│   │   └── ensemble.py             # Soft weighting untuk VOLATILE
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── backtest_engine.py      # Walk-forward validation
│   │   ├── metrics.py              # Sharpe, Win Rate, Drawdown, etc
│   │   └── strategy.py             # Trading logic
│   │
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── signal_generator.py     # Convert anomaly score → trading signal
│   │   ├── portfolio_manager.py    # Position sizing, Kelly Criterion
│   │   └── risk_manager.py         # Stop-loss, take-profit, exposure limits
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py       # SHAP values for feature importance
│   │   └── visualization.py        # SHAP plots + force plots
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── fastapi_app.py          # FastAPI server
│   │   ├── routes.py               # API endpoints
│   │   └── websocket.py            # Real-time streaming
│   │
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── streamlit_app.py        # Main dashboard
│   │   └── pages/
│   │       ├── market_overview.py
│   │       ├── signal_scanner.py
│   │       └── backtest_studio.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py               # SQLAlchemy ORM
│   │   └── crud.py                 # CRUD operations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Logging configuration
│       ├── config_loader.py        # Load YAML configs
│       └── helpers.py              # Utility functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # EDA, data validation
│   ├── 02_feature_engineering.ipynb       # Test indicators, optimize parameters
│   ├── 03_model_training.ipynb            # Train IF, regime classifier
│   ├── 04_backtesting.ipynb               # Walk-forward validation
│   └── 05_conformal_prediction.ipynb      # CP calibration
│
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_backtesting.py
│   └── test_api.py
│
├── scripts/
│   ├── download_data.py            # Fetch 3 years historical data
│   ├── train_models.py             # Train IF, regime, CP
│   ├── run_backtest.py             # Execute walk-forward backtest
│   └── evaluate_models.py          # Calculate metrics
│
├── requirements.txt                # Dependencies
├── setup.py                        # Package setup
├── README.md                       # Project documentation
├── .gitignore
├── .env.example                    # Environment variables template
└── Dockerfile                      # Containerization (optional)
```

---

## 🔧 CORE COMPONENTS SPECIFICATION

### 1. ISOLATION FOREST MODEL

**Purpose**: Primary anomaly detection for all market regimes

**Input Features**:
```python
# 30+ technical indicators (engineered from OHLCV)
momentum_features = [
    'rsi_14', 'rsi_divergence',
    'macd', 'macd_signal', 'macd_divergence',
    'stoch_k', 'stoch_d',
    'price_vs_ema_distance', 'ema_slope'
]

volatility_features = [
    'atr_14', 'atr_ratio',
    'bollinger_width', 'bollinger_squeeze_indicator',
    'historical_vol_20', 'vol_ratio'
]

volume_features = [
    'volume_ratio', 'obv', 'cmf',
    'ad_line', 'accumulation_indicator'
]

correlation_features = [
    'correlation_idx_60d', 'beta_60d'
]

price_action_features = [
    'gap_open', 'true_range', 'adr',
    'range_to_adr_ratio'
]
```

**Model Parameters**:
```python
{
    'n_estimators': 500,           # Number of trees
    'max_samples': 'auto',         # Sample size per tree
    'contamination': 0.05,         # Expected anomaly %
    'max_features': 1.0,           # Use all features
    'bootstrap': False,            # Subsampling without replacement
    'n_jobs': -1,                  # Parallel processing
    'random_state': 42,            # Reproducibility
}
```

**Training**:
```
Data Window:     3 years historical (252 trading days × 3)
Retrain Frequency: Daily (end of market)
Training Time:   ~20-30 seconds (batch process)
Inference Time:  ~12ms per stock (serial), ~1-2ms per stock (parallelized)

Rolling Window Strategy:
- Day N: Train on data [Day N-756 to Day N]  (3-year lookback)
- Output: Anomaly score untuk Day N closing bar
- Update: Save trained model to disk (joblib)
```

**Output**:
```python
{
    'anomaly_score': float,        # Raw IF score (-1 to 1, convert to 0-100)
    'is_anomaly': bool,            # Raw prediction (-1 or 1)
    'tree_paths': list,            # Tree path info (for SHAP)
    'timestamp': datetime,         # When calculated
}
```

---

### 2. CONFORMAL PREDICTION LAYER

**Purpose**: Quantify uncertainty, reduce false positives

**Framework**:
```python
# Calibration (done once, end of training)
1. Generate IF scores untuk semua historical data (3 years)
2. Calculate non-conformity scores: |score - median|
3. Determine quantile threshold berdasarkan desired FPR

# Inference (per bar)
1. Calculate IF score untuk current bar
2. Compute p-value: (# historical scores ≥ current) / (total + 1)
3. If p_value > alpha (e.g., 0.05): ACCEPT (low false positive risk)
   If p_value ≤ alpha: REJECT (high false positive risk)

# Output Confidence Interval
confidence_lower = percentile(historical_scores, alpha * 100)
confidence_upper = percentile(historical_scores, (1 - alpha) * 100)
```

**Key Hyperparameters**:
```python
{
    'alpha': 0.05,                 # Significance level (5%)
    'target_fpr': 0.08,            # Target false positive rate for trading
    'calibration_data': 'last_252_days',  # Recalibrate weekly
}
```

**Output**:
```python
{
    'anomaly_score': 78,           # Raw IF score
    'p_value': 0.02,               # Statistical significance
    'confidence_level': 0.98,      # 1 - p_value
    'ci_lower': 72,                # Confidence interval lower
    'ci_upper': 84,                # Confidence interval upper
    'signal_strength': 'HIGH',     # HIGH / MODERATE / LOW
    'false_positive_risk': 0.02,   # Estimated FPR for this signal
}
```

---

### 3. MARKET REGIME CLASSIFIER

**Purpose**: Adapt anomaly detection per market conditions (CALM → VOLATILE → CRASH)

**Features** (Calculated Daily at Market Close):
```python
# Index-level features
market_volatility = rolling_std(IDX_returns, 20)  # 20-day vol
market_trend = ADX(IDX_high, IDX_low, IDX_close)  # Trend strength
market_returns = (IDX_close[-1] - IDX_close[0]) / IDX_close[0]

# Correlation features
lq45_correlation_matrix = corr(LQ45_returns, 60)  # 60-day rolling
correlation_median = np.median(lq45_correlation_matrix)
correlation_shift = correlation_median - prev_day_median

# VIX-like proxy
vix_proxy = market_volatility * 100

# Classification Thresholds
if vix_proxy < 15 AND market_trend > 25 AND correlation_shift < 0.05:
    regime = "CALM"      # Trending, low vol, diversified
elif vix_proxy > 25 AND correlation_shift > 0.10:
    regime = "CRASH"     # High vol, correlations rising
else:
    regime = "VOLATILE"  # Choppy, range-bound

return {
    'regime': regime,
    'vix_proxy': vix_proxy,
    'adx': market_trend,
    'correlation_shift': correlation_shift,
    'model_to_use': {
        'CALM': 'isolation_forest_only',
        'VOLATILE': 'isolation_forest_soft_ensemble',
        'CRASH': 'robust_mahalanobis'
    }
}
```

**Training**:
```
Model: Random Forest Classifier OR Logistic Regression
Data: 3 years historical regime labels (generated from rules above)
Retrain: Weekly (if pattern changes)
Inference Time: ~5ms (negligible)
```

---

### 4. SOFT ENSEMBLE (VOLATILE REGIME ONLY)

**Purpose**: Add confirmation signal when market conditions ambiguous

**Only Used When**:
```python
if regime == "VOLATILE":
    # Soft weighting (NOT equal voting)
    if_score = isolation_forest.decision_function(features)
    copod_score = copod_model.decision_function(features)
    
    ensemble_score = 0.75 * if_score + 0.25 * copod_score
    
    # Apply conformal prediction on ensemble score
    p_value = conformal_predictor.p_value(ensemble_score)
    
else:
    # Use IF alone in CALM and CRASH
    ensemble_score = if_score
    p_value = conformal_predictor.p_value(if_score)
```

**COPOD Model**:
```python
{
    'model_type': 'COPOD (Contextual Outlier)',
    'contamination': 0.05,
    'training_time': ~10 seconds,
    'inference_time': ~8ms per stock,
    'only_trained_in_volatile_regime': True,
    'rationale': 'Provides local context in choppy markets'
}
```

---

## 📊 FEATURE ENGINEERING DETAIL

### Technical Indicators (30+ features)

```python
# ==================== MOMENTUM INDICATORS ====================
def calculate_momentum_features(df):
    """Returns list of momentum-based anomaly indicators"""
    
    df['rsi_14'] = talib.RSI(df['close'], 14)
    df['rsi_divergence'] = detect_divergence(df['high'], df['rsi_14'])
    
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['macd_divergence'] = detect_divergence(df['high'], df['macd'])
    
    df['stoch_k'], df['stoch_d'] = talib.STOCH(
        df['high'], df['low'], df['close']
    )
    
    df['price_vs_ema_distance'] = (df['close'] - talib.EMA(df['close'], 26)) / talib.EMA(df['close'], 26)
    df['ema_slope'] = calculate_slope(talib.EMA(df['close'], 12), window=5)
    
    return df[['rsi_14', 'rsi_divergence', 'macd', 'macd_signal', 
               'macd_divergence', 'stoch_k', 'stoch_d', 
               'price_vs_ema_distance', 'ema_slope']]

# ==================== VOLATILITY INDICATORS ====================
def calculate_volatility_features(df):
    """Returns list of volatility-based anomaly indicators"""
    
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
    df['bollinger_width'] = (bb_upper - bb_lower) / bb_middle
    df['bollinger_squeeze_indicator'] = (df['bollinger_width'] < 
                                         df['bollinger_width'].rolling(20).mean() * 0.5).astype(int)
    
    df['historical_vol_20'] = df['close'].pct_change().rolling(20).std()
    df['vol_ratio'] = df['historical_vol_20'] / df['historical_vol_20'].rolling(60).mean()
    
    return df[['atr_14', 'atr_ratio', 'bollinger_width', 
               'bollinger_squeeze_indicator', 'historical_vol_20', 'vol_ratio']]

# ==================== VOLUME INDICATORS ====================
def calculate_volume_features(df):
    """Returns list of volume-based anomaly indicators"""
    
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    df['obv'] = talib.OBV(df['close'], df['volume'])
    df['obv_ma'] = df['obv'].rolling(14).mean()
    df['obv_divergence'] = detect_divergence(df['high'], df['obv'])
    
    df['cmf'] = talib.CMF(df['high'], df['low'], df['close'], df['volume'], 20)
    
    df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['accumulation_indicator'] = (df['ad_line'] > df['ad_line'].rolling(20).mean()).astype(int)
    
    return df[['volume_ratio', 'obv', 'obv_ma', 'obv_divergence', 
               'cmf', 'ad_line', 'accumulation_indicator']]

# ==================== CORRELATION & BETA ====================
def calculate_correlation_features(df_stock, df_idx, window=60):
    """Returns correlation-based features"""
    
    stock_returns = df_stock['close'].pct_change()
    idx_returns = df_idx['close'].pct_change()
    
    correlation_60 = stock_returns.rolling(window).corr(idx_returns)
    beta_60 = calculate_beta(stock_returns, idx_returns, window)
    
    return {
        'correlation_idx_60d': correlation_60,
        'beta_60d': beta_60,
        'correlation_shift': correlation_60.diff()
    }

# ==================== PRICE ACTION ====================
def calculate_price_action_features(df):
    """Returns price action-based anomaly indicators"""
    
    df['gap_open'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
    df['adr'] = calculate_adr(df['high'], df['low'], window=20)
    df['range_to_adr_ratio'] = (df['high'] - df['low']) / df['adr']
    
    return df[['gap_open', 'true_range', 'adr', 'range_to_adr_ratio']]
```

**Feature Selection**:
```
Total Features Available: 35+
Features Used in IF Model: 25-30 (drop redundant/collinear)

Selection Criteria:
1. Correlation < 0.7 with other features
2. Non-zero variance (exclude constant features)
3. Non-NaN after first 60 rows (sufficient warmup)
4. Trading-relevant (domain knowledge)

Feature Importance (from backtest SHAP analysis):
Top 5 usually: RSI, ATR, Volume Ratio, Bollinger Width, MACD
```

---

## 🔄 TRAINING & RETRAINING PIPELINE

### Daily Retraining Schedule

```
END OF MARKET DAY (e.g., 16:00 WIB)
├─ Step 1: Download latest OHLCV data (45 LQ45 stocks)
│  └─ Time: ~30 seconds
│
├─ Step 2: Feature engineering on new data
│  └─ Time: ~10 seconds
│
├─ Step 3: Train Isolation Forest (3-year rolling window)
│  ├─ Data: Last 756 trading days
│  ├─ Time: ~20-30 seconds
│  └─ Save: data/models/isolation_forest_model.pkl
│
├─ Step 4: Recalibrate Conformal Prediction thresholds
│  ├─ Data: Last 252 days IF scores
│  ├─ Time: ~5 seconds
│  └─ Save: data/training_metadata/conformal_thresholds.json
│
├─ Step 5: Update Regime Classifier (if weekly trigger)
│  ├─ Time: ~10 seconds
│  └─ Save: data/models/regime_classifier_model.pkl
│
└─ Step 6: Run validation checks
   ├─ Model inference latency test
   ├─ Feature quality check
   └─ Log results to database

TOTAL TIME: ~90 seconds (can be parallelized to ~40 seconds)
```

### Continuous Inference (During Market Hours)

```
EVERY 1-HOUR BAR CLOSE (e.g., 11:00, 12:00, 13:00, ..., 16:00)
├─ For each of 45 stocks (PARALLELIZED):
│  ├─ Get latest OHLCV data
│  ├─ Calculate 30+ features (from last 60 bars minimum)
│  ├─ Run Isolation Forest inference (~12ms)
│  ├─ Apply Conformal Prediction (~3ms)
│  ├─ Determine Market Regime (~2ms if 1x daily, else cache)
│  ├─ Generate trading signal
│  └─ Store result to database
│
├─ PARALLELIZATION: 45 stocks × 15ms = 675ms serial
│                   Wall-time with parallelization: ~20-30ms ✅
│
└─ Output per stock:
   {
     'ticker': 'BBCA',
     'timestamp': '2025-06-16 11:00:00',
     'price': 9200,
     'anomaly_score': 78,
     'p_value': 0.02,
     'confidence_level': 0.98,
     'regime': 'CALM',
     'signal': 'LONG',
     'confidence': 'HIGH'
   }
```

---

## 📈 BACKTESTING FRAMEWORK

### Walk-Forward Validation Strategy

```python
"""
Rolling window backtesting (no look-ahead bias)
"""

training_period = 252 * 2      # 2 years
test_period = 63               # 3 months
step = 1                       # Daily steps

# Example loop
for i in range(training_start, data_length - test_period):
    # TRAIN PHASE
    train_data = data[i : i + training_period]
    test_data = data[i + training_period : i + training_period + test_period]
    
    # Train models on train_data
    isolation_forest.fit(train_data[features])
    regime_classifier.fit_and_calibrate(train_data)
    conformal_predictor.calibrate(train_data)
    
    # TEST PHASE
    # For each bar in test_data, generate signal, execute trade if meets criteria
    # Track P&L, calculate metrics
    
    results.append({
        'period': f'{train_data.index[0]} to {test_data.index[-1]}',
        'sharpe': calculate_sharpe(returns),
        'win_rate': wins / total_trades,
        'profit_factor': gross_profit / gross_loss,
        'max_drawdown': calculate_max_dd(cumsum_returns),
        'trades': total_trades
    })

# AGGREGATE RESULTS
aggregate = {
    'avg_sharpe': mean(results['sharpe']),
    'avg_win_rate': mean(results['win_rate']),
    'consistency': std(results['sharpe']),  # Measure stability across periods
    'per_period_breakdown': results
}
```

### Trade Execution Logic

```python
def generate_trade_signal(anomaly_score, p_value, regime, price, atr):
    """
    Convert anomaly score → trading action
    """
    
    # Step 1: Confidence filtering
    if p_value > 0.05:  # Not statistically significant
        return {'action': 'SKIP', 'reason': 'low_confidence'}
    
    # Step 2: Anomaly magnitude threshold (regime-adjusted)
    if regime == 'CALM':
        threshold = 70  # More conservative
    elif regime == 'VOLATILE':
        threshold = 65  # Slightly more aggressive
    else:  # CRASH
        threshold = 60  # Need lower anomaly score in crash to act
    
    if anomaly_score < threshold:
        return {'action': 'SKIP', 'reason': 'below_threshold'}
    
    # Step 3: Determine trade direction (momentum anomaly = reversal signal)
    # If high anomaly + overbought (RSI > 70) → SHORT
    # If high anomaly + oversold (RSI < 30) → LONG
    
    rsi_value = features['rsi_14'][-1]
    if anomaly_score > threshold and rsi_value > 70:
        signal = 'SHORT'
    elif anomaly_score > threshold and rsi_value < 30:
        signal = 'LONG'
    else:
        signal = None  # Ambiguous
    
    # Step 4: Generate setup
    if signal:
        atr_value = features['atr_14'][-1]
        setup = {
            'ticker': ticker,
            'timestamp': current_time,
            'signal': signal,
            'entry': price,
            'stop_loss': price - (2 * atr_value) if signal == 'LONG' else price + (2 * atr_value),
            'take_profit_1': price + (2 * atr_value) if signal == 'LONG' else price - (2 * atr_value),
            'take_profit_2': price + (3.5 * atr_value) if signal == 'LONG' else price - (3.5 * atr_value),
            'anomaly_score': anomaly_score,
            'confidence': p_value,
            'regime': regime,
        }
        
        return {'action': 'TRADE', 'setup': setup}
    else:
        return {'action': 'SKIP', 'reason': 'ambiguous_signal'}
```

### Evaluation Metrics

```python
METRICS = {
    'Total Trades': count of trades,
    'Win Rate': wins / total × 100,
    'Profit Factor': gross_profit / gross_loss,
    'Average Win': sum(winning_trades) / win_count,
    'Average Loss': sum(losing_trades) / loss_count,
    'Expectancy': (win_rate × avg_win) - ((1 - win_rate) × avg_loss),
    'Max Consecutive Losses': max(consecutive_loss_count),
    'Sharpe Ratio': (annual_return - risk_free_rate) / annual_volatility,
    'Sortino Ratio': (annual_return - rf) / downside_volatility,
    'Calmar Ratio': annual_return / max_drawdown,
    'Information Ratio': (strategy_return - benchmark_return) / tracking_error,
    'Maximum Drawdown': (peak_value - trough_value) / peak_value,
    'Recovery Factor': total_profit / max_drawdown,
}

TARGET VALUES:
├─ Win Rate: > 56%
├─ Profit Factor: > 1.8
├─ Sharpe Ratio: > 1.5
├─ Max Drawdown: < 12%
└─ Expectancy: > 0.5% per trade
```

---

## 🌐 API ENDPOINTS (FastAPI)

```python
BASE_URL: http://localhost:8000/api/v1

# ==================== REAL-TIME ANOMALY ====================
GET /stock/{ticker}/anomaly
"""
Get latest anomaly score untuk satu saham

Query Params:
  - timestamp: (optional) specific time, else latest
  
Response:
{
  "ticker": "BBCA",
  "timestamp": "2025-06-16 15:00:00",
  "price": 9200,
  "anomaly_score": 78,
  "p_value": 0.02,
  "confidence_level": 0.98,
  "confidence_interval": [72, 84],
  "regime": "CALM",
  "signal": "LONG",
  "false_positive_risk": 0.02,
  "model_info": {
    "model_used": "IsolationForest",
    "inference_latency_ms": 12,
    "features_used": 28,
    "training_date": "2025-06-15"
  }
}
"""

# ==================== MARKET OVERVIEW ====================
GET /market/overview
"""
Get anomaly summary untuk semua 45 LQ45 stocks

Response:
{
  "timestamp": "2025-06-16 15:00:00",
  "regime": "CALM",
  "regime_details": {
    "vix_proxy": 12,
    "adx": 28,
    "correlation_shift": 0.02
  },
  "anomalies": [
    {
      "ticker": "BBCA",
      "price": 9200,
      "anomaly_score": 82,
      "signal": "LONG",
      "confidence": 0.98
    },
    ...
  ],
  "signal_summary": {
    "strong_long": 3,
    "strong_short": 2,
    "moderate_signals": 5,
    "neutral": 35
  }
}
"""

# ==================== TRADING SETUP ====================
GET /stock/{ticker}/trade-setup
"""
Get recommended trade setup untuk entry

Response:
{
  "ticker": "BBCA",
  "signal": "LONG",
  "entry": 9200,
  "stop_loss": 9100,
  "take_profit_1": 9350,
  "take_profit_2": 9500,
  "position_size_pct": 2.0,
  "risk_reward_ratio": 2.5,
  "kelly_fraction": 0.25,
  "expected_value": 1.8,
  "confidence": "HIGH",
  "explanation": "High momentum anomaly (RSI 72) in calm market"
}
"""

# ==================== BACKTESTING ====================
POST /backtest/run
"""
Run walk-forward backtest

Request Body:
{
  "ticker": "BBCA",
  "test_period": "2024-01-01 to 2025-06-15",
  "test_name": "IF_CALM_REGIME"
}

Response:
{
  "sharpe_ratio": 1.45,
  "win_rate": 0.58,
  "profit_factor": 1.8,
  "max_drawdown": -0.082,
  "total_trades": 42,
  "monthly_returns": [...],
  "drawdown_chart": "...",
  "trade_log": [...]
}
"""

# ==================== REAL-TIME STREAM ====================
WebSocket /ws/stream/{ticker}
"""
Real-time streaming per 1-hour bar close

Message (every hour):
{
  "ticker": "BBCA",
  "timestamp": "2025-06-16 15:00:00",
  "price": 9200,
  "anomaly_score": 78,
  "signal": "LONG",
  "regime": "CALM"
}
"""
```

---

## 🛠 DEVELOPMENT PHASES (IMPLEMENTATION ORDER)

### PHASE 1: MVP (Week 1-2)
**Goal**: Working IF model + basic Streamlit dashboard

**Deliverables**:
- [ ] Data download & validation (3 years LQ45)
- [ ] Feature engineering (30+ indicators)
- [ ] Isolation Forest training
- [ ] Backtesting on 1 year test data
- [ ] Streamlit dashboard (basic layout)
- [ ] Evaluation: Sharpe, Win Rate, Max DD metrics

**Success Criteria**:
- Sharpe Ratio > 1.2
- Win Rate > 54%
- Inference latency < 50ms
- Dashboard responsive

---

### PHASE 2: Conformal Prediction (Week 3)
**Goal**: Add uncertainty quantification, reduce false positives

**Deliverables**:
- [ ] Conformal prediction framework
- [ ] Calibration on historical scores
- [ ] P-value + confidence interval outputs
- [ ] Signal filtering based on confidence
- [ ] Updated backtest with CP filtering
- [ ] API endpoints for p-values

**Success Criteria**:
- False positive rate < 8%
- Sharpe maintained or improved
- Win rate on high-confidence signals > 60%

---

### PHASE 3: Market Regime Classifier (Week 4)
**Goal**: Adaptive model selection per market conditions

**Deliverables**:
- [ ] Regime feature engineering (VIX proxy, ADX, correlation)
- [ ] Regime classification logic
- [ ] Per-regime model evaluation
- [ ] Dashboard regime indicator
- [ ] Backtest with regime switching

**Success Criteria**:
- CALM regime: Sharpe > 1.8
- VOLATILE regime: Sharpe > 1.2
- CRASH regime: Max DD < 8%
- Regime detection accuracy > 85%

---

### PHASE 4: Soft Ensemble + COPOD (Week 5)
**Goal**: Add validation layer untuk VOLATILE regime

**Deliverables**:
- [ ] COPOD model training
- [ ] Soft weighting logic (0.75 IF + 0.25 COPOD)
- [ ] Only enable in VOLATILE regime
- [ ] Latency verification (must stay < 100ms)
- [ ] Incremental backtest comparison

**Success Criteria**:
- VOLATILE regime improvement: +5-10% Sharpe
- Total latency still < 100ms
- Inference still parallelizable

---

### PHASE 5: FastAPI + Real-Time (Week 6-7)
**Goal**: Production-ready API + real-time inference

**Deliverables**:
- [ ] FastAPI server setup
- [ ] REST endpoints for anomaly, setup, backtest
- [ ] WebSocket streaming
- [ ] PostgreSQL database schema
- [ ] Redis caching layer
- [ ] Request/response logging

**Success Criteria**:
- API latency < 200ms (p95)
- Throughput > 100 req/s
- Uptime > 99%
- All endpoints documented (OpenAPI/Swagger)

---

### PHASE 6: Monitoring & Deployment (Week 8+)
**Goal**: Production deployment + monitoring

**Deliverables**:
- [ ] Docker containerization
- [ ] Model monitoring (drift detection)
- [ ] Performance tracking (Prometheus)
- [ ] Alerting system
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Production deployment

**Success Criteria**:
- Zero downtime
- Model retraining automated daily
- Alerts for performance degradation
- Full audit trail for trades

---

## 📝 CODING CONVENTIONS & GUIDELINES

### Python Style
```python
# Follow PEP 8
# Line length: 100 characters
# Type hints: Use for all function signatures
# Docstrings: Google-style docstrings for all functions

from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_anomaly_score(features: np.ndarray, model) -> Dict[str, float]:
    """
    Calculate anomaly score using trained Isolation Forest.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        model: Trained Isolation Forest model
        
    Returns:
        Dictionary with 'score', 'raw_prediction', 'timestamp'
        
    Raises:
        ValueError: If features contain NaN values
        
    Example:
        >>> result = calculate_anomaly_score(X, model)
        >>> print(result['score'])
        78.5
    """
    # Implementation
    pass
```

### Logging
```python
# Use logging, not print()
logger.info(f"Training IF model on {len(data)} samples")
logger.debug(f"Feature matrix shape: {features.shape}")
logger.warning(f"Missing values detected: {missing_count}")
logger.error(f"Failed to download data for {ticker}: {error}")
```

### Error Handling
```python
# Use custom exceptions
class DataValidationError(Exception):
    """Raised when data quality check fails"""
    pass

class ModelInferenceError(Exception):
    """Raised when model inference fails"""
    pass

# Handle gracefully
try:
    result = model.predict(features)
except Exception as e:
    logger.error(f"Inference failed: {e}")
    raise ModelInferenceError(f"Failed to generate prediction: {e}")
```

### Testing
```python
# Use pytest
# Test file naming: test_*.py
# Test function naming: test_*

import pytest
from src.models.isolation_forest import train_if_model

def test_train_if_model_basic():
    """Test IF training on synthetic data"""
    X = np.random.randn(100, 10)
    model = train_if_model(X)
    assert model is not None
    assert hasattr(model, 'decision_function')

def test_train_if_model_empty_data():
    """Test IF training with empty data"""
    X = np.array([]).reshape(0, 10)
    with pytest.raises(ValueError):
        train_if_model(X)
```

---

## ⚙️ CONFIGURATION (YAML Format)

### config.yaml
```yaml
# Application Configuration

app:
  name: "Trading Decision-Support System"
  version: "1.0.0"
  environment: "development"  # development, staging, production

data:
  symbols: 
    - BBCA
    - BBRI
    - BMRI
    # ... 42 more LQ45 stocks
  historical_years: 3
  timeframe: "1h"  # 1-hour bars
  data_source: "yfinance"
  
training:
  retrain_frequency: "daily"
  retrain_time: "16:30"  # After market close (WIB)
  train_window_days: 756  # 3 years
  validation_window_days: 252  # 1 year
  
model:
  isolation_forest:
    n_estimators: 500
    contamination: 0.05
    max_samples: auto
    n_jobs: -1
    
  conformal_prediction:
    alpha: 0.05
    target_fpr: 0.08
    recalibrate_frequency: "weekly"
    
  regime_classifier:
    retrain_frequency: "weekly"
    vix_threshold_calm: 15
    vix_threshold_crash: 25
    
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  
database:
  type: "sqlite"  # or "postgresql"
  sqlite_path: "./data/trading.db"
  # For PostgreSQL:
  # host: "localhost"
  # port: 5432
  # user: "trading_user"
  # password: "${DB_PASSWORD}"
  
redis:
  enabled: true
  host: "localhost"
  port: 6379
  ttl_seconds: 3600
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 🎯 SUCCESS CRITERIA & ACCEPTANCE

### MVP Phase Success (Week 2)
```
✅ Isolation Forest model trained on 3 years data
✅ Inference latency < 50ms per stock (serial), < 30ms (parallelized)
✅ Backtest Sharpe Ratio > 1.2
✅ Backtest Win Rate > 54%
✅ Streamlit dashboard working
✅ Zero crashes for 24-hour continuous run
✅ Feature engineering validated (no NaN, no infinite values)
```

### Phase 2 Success (Week 3)
```
✅ Conformal Prediction reduces false positives to < 8%
✅ High-confidence signals Win Rate > 60%
✅ P-values statistically valid (empirical FPR matches target)
✅ No latency increase > 5ms
✅ API endpoints documented
```

### Phase 3 Success (Week 4)
```
✅ Regime classifier accuracy > 85%
✅ Per-regime Sharpe: CALM > 1.8, VOLATILE > 1.2, CRASH stable
✅ Regime switching increases overall Sharpe by 5-10%
✅ Dashboard shows regime info
```

### Final Production Success (Week 8)
```
✅ API latency p95 < 200ms
✅ Uptime > 99.5%
✅ Daily retraining completes in < 2 minutes
✅ Model drift detection functional
✅ Full audit trail for all trades
✅ All unit tests pass
✅ Docker container runs without errors
```

---

## 📚 REFERENCES & EXTERNAL RESOURCES

### Academic Papers
1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest" - ICDM
2. Hariri, S., Kind, M. C., & Brunner, R. J. (2019). "Extended Isolation Forest" - arXiv
3. Barlow, M. (2013). "Conformal Prediction for Reliable Machine Learning"

### Python Libraries
- `scikit-learn`: Isolation Forest implementation
- `pyod`: COPOD, LOF, other anomaly detection
- `TA-Lib`: Technical analysis indicators
- `pandas_ta`: Alternative technical indicators
- `pandas`: Data manipulation
- `numpy`: Numerical computation
- `fastapi`: Web API framework
- `streamlit`: Frontend dashboard
- `sqlalchemy`: ORM for database
- `pytest`: Testing framework
- `shap`: Explainability

### Market Data
- `yfinance`: Historical OHLCV data
- Yahoo Finance API (rate limited)
- Alpha Vantage (requires API key)

---

## ❓ FREQUENTLY ASKED QUESTIONS FOR AI

### Q1: "Should I use more features?"
**A**: No. 25-30 uncorrelated features is optimal. More features → curse of dimensionality, harder to interpret, slower inference. Stick to domain-relevant indicators.

### Q2: "What if Isolation Forest performance drops?"
**A**: 
1. Check data quality (missing values, outliers)
2. Verify conformal prediction recalibration
3. Check market regime classification
4. If regime classification poor → retrain with more data

### Q3: "Should I ensemble all 4 models always?"
**A**: No. ONLY use soft ensemble (IF + COPOD) in VOLATILE regime. Use IF alone in CALM and CRASH. This keeps latency low and signal quality high.

### Q4: "What's the expected Sharpe Ratio?"
**A**: Conservative estimate: 1.2-1.5. This is good for momentum/mean-reversion strategies. If backtest shows > 2.0, likely overfitting → validate on out-of-sample data.

### Q5: "How often should I retrain?"
**A**: Daily (end of market, ~16:30 WIB). This ensures model stays fresh without overfitting to one regime. Weekly regime classifier update is also fine.

### Q6: "Is 100ms latency strict?"
**A**: Very strict. 100ms = parallelized 45 stocks taking ~2ms each. Don't add slow models. IF + conformal prediction gives you ~15ms margin.

---

**End of System Prompt**

---

# HOW TO USE THIS PROMPT WITH COPILOT

1. **Open file in VS Code**
   ```
   File → Open → SYSTEM_PROMPT_TRADING_APP.md
   ```

2. **Open Copilot Chat** (Ctrl+Shift+I on Windows, Cmd+Shift+I on Mac)

3. **Paste this prompt into Copilot**:
   ```
   I'm building a trading decision-support system. Here's the complete specification:
   
   [Paste entire content of this file]
   
   I need you to:
   1. Understand the full architecture
   2. Generate code for [specific component] following this spec
   3. Ask clarifying questions if anything is ambiguous
   4. Suggest optimizations while maintaining the constraints
   ```

4. **Then request specific components**:
   ```
   Generate the `src/models/isolation_forest.py` file that:
   - Implements Isolation Forest training on rolling 3-year window
   - Returns anomaly scores [0-100]
   - Saves trained model to disk
   - Includes logging and error handling
   ```

---

