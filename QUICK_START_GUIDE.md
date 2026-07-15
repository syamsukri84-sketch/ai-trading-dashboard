> ⚠️ **USANG (per 2026-07-12)** — panduan setup ini dari fase perencanaan awal (Juni 2026),
> merujuk ke `SYSTEM_PROMPT_TRADING_APP.md` yang arsitekturnya sudah berubah total.
> **Baca `STATUS_PROYEK_AI_TRADING.md` terlebih dahulu** untuk status & cara kerja terkini.

# 🚀 QUICK START: Menggunakan Prompt + AI Copilot untuk Development

**Tujuan**: Setup development environment + integrate AI Copilot untuk implement sesuai spec

**Total Setup Time**: ~5 menit  
**Development Time**: Tergantung jumlah components (estimate: 4-5 minggu untuk full system)

---

## 📋 3 FILES YANG SUDAH DIBUAT

```
1. SYSTEM_PROMPT_TRADING_APP.md
   └─ Comprehensive specification untuk seluruh project
   └─ 7000+ lines detailing: architecture, constraints, components, evaluation
   └─ GUNAKAN SEBAGAI: Context reference untuk Copilot

2. COPILOT_PROMPT_TEMPLATES.md
   └─ 13 ready-to-use prompt templates
   └─ Masing-masing untuk component spesifik
   └─ GUNAKAN SEBAGAI: Copy-paste prompts untuk Copilot

3. QUICK_START_GUIDE.md (file ini)
   └─ Step-by-step setup instructions
   └─ Workflow untuk daily development
```

---

## ⚙️ SETUP STEP (5 MENIT)

### 1. Clone/Create Project Folder

```bash
# Di Windows / macOS / Linux
mkdir trading-decision-support
cd trading-decision-support

# Create folder structure
mkdir -p src/{data_pipeline,models,backtesting,trading,api,frontend,database,utils}
mkdir data/{raw,processed,models,training_metadata}
mkdir notebooks tests scripts config
```

### 2. Copy Documentation Files

```
Download 3 files dari output folder:
- SYSTEM_PROMPT_TRADING_APP.md
- COPILOT_PROMPT_TEMPLATES.md
- QUICK_START_GUIDE.md

Place di root folder: /trading-decision-support/
```

### 3. Create Initial Files

```bash
# requirements.txt (dependencies)
python==3.11
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
pandas-ta==0.3.14
ta-lib==0.4.24
yfinance==0.2.28
fastapi==0.100.0
uvicorn==0.23.0
streamlit==1.28.0
pytest==7.4.0
sqlalchemy==2.0.0
pydantic==2.0.0

# Create it
cat > requirements.txt << 'EOF'
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
# ... (paste requirements above)
EOF
```

### 4. Open in VS Code

```bash
code .
# atau: code trading-decision-support
```

### 5. Install Python Environment (Optional but Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 DEVELOPMENT WORKFLOW

### DAILY WORKFLOW

```
Morning:
1. Open VS Code
2. Activate virtual environment (if first time)
3. Open Copilot Chat (Ctrl+Shift+I or Cmd+Shift+I)
4. Open COPILOT_PROMPT_TEMPLATES.md in split pane

Development:
1. Choose component from checklist (see below)
2. Copy relevant template from COPILOT_PROMPT_TEMPLATES.md
3. Paste into Copilot Chat
4. Wait for code generation
5. Review + save to file
6. Test with pytest
7. Commit to git

End of Day:
1. Run tests: pytest tests/
2. Check latency: python scripts/latency_check.py
3. Commit changes
```

---

## ✅ IMPLEMENTATION CHECKLIST

### Phase 1: MVP (Week 1-2)
```
WEEK 1:
├─ [ ] Data Ingestion
│  ├─ [ ] Download 3 years LQ45 data (yfinance)
│  └─ [ ] Data validation (no NaN, format check)
│
├─ [ ] Feature Engineering
│  ├─ [ ] Momentum indicators (RSI, MACD, Stoch)
│  ├─ [ ] Volatility indicators (ATR, Bollinger)
│  ├─ [ ] Volume indicators (OBV, CMF)
│  └─ [ ] Correlation features
│
└─ [ ] First Tests
   └─ [ ] Unit tests passing

WEEK 2:
├─ [ ] Isolation Forest Model
│  ├─ [ ] Training on 3-year data
│  ├─ [ ] Inference with latency check
│  └─ [ ] Model persistence (save/load)
│
├─ [ ] Backtesting
│  ├─ [ ] Walk-forward validation framework
│  ├─ [ ] Metrics calculation (Sharpe, Win Rate)
│  └─ [ ] Trade execution simulation
│
└─ [ ] Dashboard (Streamlit)
   ├─ [ ] Basic layout
   ├─ [ ] Display anomaly scores
   └─ [ ] Show backtest metrics

PHASE 1 SUCCESS: Sharpe > 1.2, Win Rate > 54%
```

### Phase 2: Conformal Prediction (Week 3)
```
├─ [ ] Conformal Prediction Framework
│  ├─ [ ] Calibration on historical scores
│  ├─ [ ] P-value calculation
│  └─ [ ] Confidence interval computation
│
├─ [ ] Integration
│  ├─ [ ] Filter signals by p-value
│  ├─ [ ] Updated backtest with CP filtering
│  └─ [ ] Dashboard showing confidence levels
│
└─ [ ] Validation
   └─ [ ] Empirical FPR < 8%

PHASE 2 SUCCESS: High-confidence signals Win Rate > 60%
```

### Phase 3: Market Regime (Week 4)
```
├─ [ ] Regime Feature Engineering
│  ├─ [ ] VIX proxy calculation
│  ├─ [ ] ADX calculation
│  └─ [ ] Correlation shift tracking
│
├─ [ ] Regime Classifier
│  ├─ [ ] Classification logic (CALM/VOLATILE/CRASH)
│  ├─ [ ] Per-regime model switching
│  └─ [ ] Historical regime tracking
│
└─ [ ] Testing
   └─ [ ] Per-regime Sharpe ratios

PHASE 3 SUCCESS: CALM Sharpe > 1.8, consistent across regimes
```

### Phase 4: Soft Ensemble (Week 5)
```
├─ [ ] COPOD Model Training
│  └─ [ ] Only for VOLATILE regime
│
├─ [ ] Soft Weighting Logic
│  ├─ [ ] 75% IF + 25% COPOD (VOLATILE only)
│  ├─ [ ] Latency validation
│  └─ [ ] Conformal prediction on ensemble
│
└─ [ ] Backtest Comparison
   └─ [ ] +5-10% improvement in VOLATILE regime

PHASE 4 SUCCESS: Total latency still < 100ms, +improvement
```

### Phase 5: FastAPI (Week 6-7)
```
├─ [ ] API Server Setup
│  ├─ [ ] FastAPI app initialization
│  ├─ [ ] CORS configuration
│  └─ [ ] Health check endpoint
│
├─ [ ] Endpoints Implementation
│  ├─ [ ] GET /api/v1/stock/{ticker}/anomaly
│  ├─ [ ] GET /api/v1/market/overview
│  ├─ [ ] GET /api/v1/stock/{ticker}/trade-setup
│  ├─ [ ] POST /api/v1/backtest/run
│  └─ [ ] WebSocket /ws/stream/{ticker}
│
├─ [ ] Database Setup
│  ├─ [ ] SQLAlchemy models
│  ├─ [ ] CRUD operations
│  └─ [ ] Data persistence
│
└─ [ ] Testing
   ├─ [ ] API latency p95 < 200ms
   └─ [ ] All endpoints working

PHASE 5 SUCCESS: Full API operational, Swagger docs available
```

### Phase 6: Monitoring (Week 8+)
```
├─ [ ] Docker Setup
│  ├─ [ ] Dockerfile
│  ├─ [ ] docker-compose.yml
│  └─ [ ] Container tested
│
├─ [ ] Monitoring
│  ├─ [ ] Model performance tracking
│  ├─ [ ] Data drift detection
│  └─ [ ] Alert system
│
└─ [ ] CI/CD
   ├─ [ ] GitHub Actions workflows
   ├─ [ ] Automated testing
   └─ [ ] Deployment pipeline

PHASE 6 SUCCESS: Production-ready system
```

---

## 🔧 HOW TO USE COPILOT EFFECTIVELY

### Before You Start
```
1. Open VS Code
2. Open Copilot Chat: Ctrl+Shift+I (Windows) atau Cmd+Shift+I (Mac)
3. Keep these files visible:
   - SYSTEM_PROMPT_TRADING_APP.md (for reference)
   - COPILOT_PROMPT_TEMPLATES.md (for templates)
   - Current .py file you're editing
```

### First Time Setup (Once)
```
Paste this into Copilot Chat:

---
I'm building a trading decision-support system for swing trading in Indonesian 
stock market (LQ45 stocks). Here's my full specification:

[COPY ENTIRE CONTENT OF SYSTEM_PROMPT_TRADING_APP.md]

I'll be working on implementing this system component by component. 
When I ask for code, please refer to this spec for context.

Confirm you understand by summarizing the key constraints.
---

Copilot will acknowledge and use this context for all subsequent prompts.
```

### For Each Component
```
Example: Implementing Isolation Forest Model

1. Open COPILOT_PROMPT_TEMPLATES.md
2. Find "TEMPLATE 3: Isolation Forest Training"
3. Copy the entire template
4. Paste into Copilot Chat
5. Hit Enter / Send

Copilot generates: src/models/isolation_forest.py

6. Review code
7. Copy to file: src/models/isolation_forest.py
8. Test: pytest tests/test_isolation_forest.py
9. Commit: git commit -m "Add isolation forest model"
```

### Tips for Better Code Generation
```
✅ DO:
- Include full context (paste system prompt once)
- Be specific about file paths
- Mention constraints (latency, data size, etc.)
- Ask for type hints + docstrings
- Request error handling
- Ask for logging

❌ DON'T:
- Ask vague questions ("make anomaly detector")
- Request multiple unrelated files at once
- Ignore warnings/errors in generated code
- Skip testing

Example BAD prompt:
"Generate the anomaly detection model"

Example GOOD prompt:
"Generate src/models/isolation_forest.py with:
- train() method on 756-day rolling window
- predict() returns anomaly_score [0-100] + latency
- Save/load from disk with joblib
- Latency < 20ms per stock
- Type hints on all functions
- Comprehensive docstrings
"
```

---

## 🧪 TESTING WORKFLOW

```bash
# Run specific test file
pytest tests/test_isolation_forest.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run single test function
pytest tests/test_isolation_forest.py::test_train_basic -v

# Show print statements (debugging)
pytest tests/ -s
```

---

## 📊 MONITORING DEVELOPMENT PROGRESS

### Metrics to Track
```
PHASE 1 (End of Week 2):
├─ Isolation Forest Sharpe Ratio: Target 1.2+
├─ Win Rate: Target 54%+
├─ Inference Latency: Target < 50ms serial
└─ Code Coverage: Target 60%+

PHASE 2 (End of Week 3):
├─ Conformal Prediction Calibration: FPR < 8%
├─ High-Confidence Win Rate: Target 60%+
└─ Code Coverage: Target 70%+

PHASE 3 (End of Week 4):
├─ Per-Regime Sharpe: CALM > 1.8, VOLATILE > 1.2
├─ Regime Detection Accuracy: > 85%
└─ Code Coverage: Target 75%+

PHASE 4 (End of Week 5):
├─ Total Latency: < 100ms (45 stocks parallel)
├─ VOLATILE Regime Improvement: +5-10%
└─ Code Coverage: Target 80%+

PHASE 5 (End of Week 7):
├─ API Latency p95: < 200ms
├─ Uptime: > 99%
└─ Code Coverage: Target 85%+
```

### Check Latency
```python
# Create scripts/latency_check.py
import time
import numpy as np
from src.models.isolation_forest import IsolationForestModel

model = IsolationForestModel()
X = np.random.randn(100, 30)  # 100 stocks, 30 features

start = time.time()
for _ in range(100):
    scores = model.predict(X)
elapsed = time.time() - start

print(f"Average latency: {elapsed/100*1000:.2f}ms")
print(f"Target: < 20ms per stock")
```

---

## 🐛 TROUBLESHOOTING

### Issue: "Copilot doesn't remember context"
**Solution**: Paste system prompt at start of each Copilot session

### Issue: "Generated code has bugs"
**Solution**: Ask Copilot to:
1. Add unit tests first
2. Include error handling
3. Add type checking: `mypy src/`
4. Show example usage

### Issue: "Latency exceeds 100ms"
**Solution**: 
1. Check if you're using all 4 models (remove LOF + COPOD + XGBOD)
2. Only use IF + soft ensemble in VOLATILE
3. Parallelize across 45 stocks
4. Profile with: `python -m cProfile script.py`

### Issue: "False positives too high"
**Solution**:
1. Check conformal prediction calibration (empirical FPR)
2. Increase p-value threshold (e.g., 0.05 → 0.03)
3. Increase anomaly score threshold
4. Verify regime classification

---

## 📁 FILE ORGANIZATION REFERENCE

```
trading-decision-support/
├── SYSTEM_PROMPT_TRADING_APP.md          ← Keep this visible during dev
├── COPILOT_PROMPT_TEMPLATES.md           ← Copy templates from here
├── QUICK_START_GUIDE.md                  ← You are here
├── requirements.txt
├── setup.py
├── README.md
├── .gitignore
├── .env.example
│
├── src/
│  ├── __init__.py
│  ├── data_pipeline/
│  │  ├── __init__.py
│  │  ├── data_loader.py
│  │  ├── feature_engineer.py
│  │  └── data_validator.py
│  │
│  ├── models/
│  │  ├── __init__.py
│  │  ├── isolation_forest.py
│  │  ├── regime_classifier.py
│  │  ├── conformal_predictor.py
│  │  └── ensemble.py
│  │
│  ├── backtesting/
│  │  ├── __init__.py
│  │  ├── backtest_engine.py
│  │  ├── metrics.py
│  │  └── strategy.py
│  │
│  ├── trading/
│  │  ├── __init__.py
│  │  ├── signal_generator.py
│  │  ├── portfolio_manager.py
│  │  └── risk_manager.py
│  │
│  ├── api/
│  │  ├── __init__.py
│  │  ├── fastapi_app.py
│  │  ├── routes.py
│  │  └── websocket.py
│  │
│  ├── frontend/
│  │  ├── __init__.py
│  │  ├── streamlit_app.py
│  │  └── pages/
│  │
│  ├── database/
│  │  ├── __init__.py
│  │  ├── models.py
│  │  └── crud.py
│  │
│  ├── explainability/
│  │  ├── __init__.py
│  │  └── shap_explainer.py
│  │
│  └── utils/
│     ├── __init__.py
│     ├── logger.py
│     ├── config_loader.py
│     └── helpers.py
│
├── data/
│  ├── raw/                    ← Downloaded OHLCV data
│  ├── processed/              ← Feature-engineered data
│  ├── models/                 ← Saved .pkl files
│  └── training_metadata/      ← Conformal thresholds, regime stats
│
├── notebooks/                 ← Jupyter for exploration
│  ├── 01_data_exploration.ipynb
│  ├── 02_feature_engineering.ipynb
│  ├── 03_model_training.ipynb
│  └── 04_backtesting.ipynb
│
├── tests/                     ← Unit tests
│  ├── __init__.py
│  ├── test_data_pipeline.py
│  ├── test_models.py
│  ├── test_backtesting.py
│  └── test_api.py
│
├── scripts/                   ← Standalone scripts
│  ├── download_data.py
│  ├── train_models.py
│  ├── run_backtest.py
│  └── latency_check.py
│
├── config/
│  ├── config.yaml             ← App config
│  ├── feature_config.yaml     ← Indicator params
│  └── model_config.yaml       ← Model hyperparams
│
└── docker/
   ├── Dockerfile
   └── docker-compose.yml
```

---

## 🎯 NEXT STEPS

### Immediate (Today)
```
1. ✅ Create project folder structure
2. ✅ Copy 3 documentation files
3. ✅ Setup Python environment
4. ✅ Install dependencies
5. ✅ Test Copilot Chat integration
```

### Week 1
```
1. Generate data_loader.py (TEMPLATE 1)
2. Generate feature_engineer.py (TEMPLATE 2)
3. Generate test files
4. Test data pipeline end-to-end
5. Commit to Git
```

### Week 2
```
1. Generate isolation_forest.py (TEMPLATE 3)
2. Generate backtest_engine.py (TEMPLATE 8)
3. Run full pipeline: data → features → training → backtest
4. Validate: Sharpe > 1.2, Win Rate > 54%
5. Generate Streamlit dashboard
```

### Ongoing
```
- Follow the implementation checklist
- Use templates from COPILOT_PROMPT_TEMPLATES.md
- Test each component
- Track metrics
- Commit regularly
```

---

## 📞 HELPFUL COMMANDS

```bash
# Initialize git repo
git init
git add .
git commit -m "Initial commit: project structure + docs"

# Run specific phase development
# Phase 1: MVP (week 1-2)
pytest tests/test_data_pipeline.py -v
pytest tests/test_models.py -v

# Phase 2: Conformal Prediction (week 3)
pytest tests/test_conformal_predictor.py -v

# Phase 3: Regime (week 4)
pytest tests/test_regime_classifier.py -v

# Phase 4: Ensemble (week 5)
pytest tests/test_ensemble.py -v

# Phase 5: API (week 6-7)
uvicorn src.api.fastapi_app:app --reload

# Phase 6: Monitoring (week 8+)
docker-compose up -d

# Check code quality
pylint src/
mypy src/
black src/

# Profile performance
python -m cProfile -s cumtime scripts/run_backtest.py
```

---

## 🚀 FINAL CHECKLIST BEFORE STARTING

```
Before opening Copilot:

✅ Project folder created
✅ Folder structure created (src/, data/, tests/, notebooks/)
✅ 3 documentation files copied and visible
✅ Python environment setup
✅ requirements.txt created
✅ Copilot Chat available (Ctrl+Shift+I works)
✅ Git initialized
✅ Editor configured for Python (syntax highlighting, formatter)

Then:

✅ Open COPILOT_PROMPT_TEMPLATES.md in split view
✅ Start with TEMPLATE 1: Data Loading
✅ Follow development checklist
✅ Test each component
✅ Move to next component
```

---

**You're now ready to start development with AI Copilot! 🚀**

**Questions?** Reference SYSTEM_PROMPT_TRADING_APP.md for detailed specifications.

