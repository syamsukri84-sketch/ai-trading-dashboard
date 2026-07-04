from fastapi import APIRouter, HTTPException, Depends
import logging
from typing import Dict, Any
from sqlalchemy.orm import Session

from data_loader import DataLoader
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.models.isolation_forest import IsolationForestModel
from src.models.price_projector import PriceProjector
from src.models.conformal_predictor import ConformalPredictor
from src.models.regime_classifier import RegimeClassifier
from src.models.ensemble import soft_ensemble_predict, COPODModel
from src.trading.signal_generator import generate_signal
from src.database.database import get_db
from src.database import crud

router = APIRouter()
logger = logging.getLogger(__name__)

# Inisialisasi modul (Global) agar tidak berulang kali dibuat saat API dipanggil
loader = DataLoader(min_rows=100)
engineer = FeatureEngineer(warmup_period=60)

@router.get("/health")
def health_check():
    """Mengecek apakah server API menyala dan sehat."""
    return {"status": "ok", "message": "API is running smoothly! 🚀"}

@router.get("/stock/{ticker}/anomaly")
def get_anomaly_and_projection(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Menghitung skor anomali dan memproyeksikan harga 3 hari ke depan.
    """
    try:
        ticker = ticker.upper()
        df = loader.load_data(ticker)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Data {ticker} tidak ditemukan. Pastikan CSV tersedia.")
            
        features_df = engineer.generate_features(df)
        
        # Melatih dan mendeteksi anomali
        if_model = IsolationForestModel(contamination=0.05)
        if_model.train(features_df)
        results = if_model.predict(features_df)
        
        scores_list = results['anomaly_score']
        latest_score = float(scores_list[-1] if isinstance(scores_list, list) else scores_list)

        # 2. Kalibrasi Conformal Predictor untuk mencari p_value
        cp = ConformalPredictor(alpha=0.05)
        if isinstance(scores_list, list) and len(scores_list) > 50:
            cp.calibrate(scores_list[:-1]) # Kalibrasi dengan data masa lalu
            cp_result = cp.predict(latest_score) # Prediksi skor hari ini
        else:
            cp_result = {"p_value": 0.05, "confidence_level": 0.95, "signal_strength": "MODERATE"}
            # Gunakan kalibrasi dummy jika data kurang agar ensemble tidak error
            cp.calibrate([50.0] * 100) 

        # 3. Klasifikasi Regime Pasar
        rc = RegimeClassifier()
        idx_df = loader.load_data('^JKSE') # Coba gunakan Indeks Harga Saham Gabungan (IHSG)
        if idx_df is None:
            idx_df = df # Fallback menggunakan data emiten itu sendiri jika IHSG tidak ditemukan
        regime_features = rc.calculate_regime_features(idx_df, df)
        regime_result = rc.classify(regime_features)
        regime_str = regime_result.get("regime", "VOLATILE")
        
        # 4. Terapkan Soft Ensemble (IF + COPOD jika VOLATILE)
        copod = None
        if regime_str == "VOLATILE":
            try:
                copod = COPODModel(contamination=0.05)
                copod.train(features_df)
            except Exception as e:
                logger.warning(f"Gagal melatih COPOD, fallback ke IF. Error: {e}")
                
        ensemble_result = soft_ensemble_predict(
            features_df=features_df,
            regime=regime_str,
            isolation_forest_model=if_model,
            conformal_predictor=cp,
            copod_model=copod
        )

        # Memproyeksikan harga
        projector = PriceProjector(projection_horizon=3)
        projector.train(features_df)
        projection = projector.predict(features_df)
        
        result_data = {
            "ticker": ticker,
            "timestamp": str(df['timestamp'].iloc[-1]),
            "current_price": float(df['close'].iloc[-1]),
            "anomaly_score": ensemble_result.get("anomaly_score", 0.0),
            "p_value": ensemble_result.get("p_value", 0.05),
            "regime": regime_str,
            "projection_3d": projection
        }
        
        # Simpan riwayat prediksi ke Database
        db_data = {
            "ticker": ticker,
            "timestamp": str(df['timestamp'].iloc[-1]),
            "current_price": float(df['close'].iloc[-1]),
            "anomaly_score": ensemble_result.get("anomaly_score", 0.0),
            "p_value": ensemble_result.get("p_value", 0.05),
            "regime": regime_str
        }
        crud.save_prediction(db, db_data)
        
        return result_data
    except Exception as e:
        logger.error(f"API Error pada /stock/{ticker}/anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/{ticker}/trade-setup")
def get_trade_setup(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Menghasilkan Setup Trading (Entry, SL, TP) berdasarkan AI.
    """
    try:
        # Panggil endpoint anomaly untuk mendapatkan data mentahnya
        anomaly_data = get_anomaly_and_projection(ticker, db)
        
        df = loader.load_data(ticker.upper())
        features_df = engineer.generate_features(df)
        
        rsi = float(features_df['feat_rsi_14'].iloc[-1])
        atr = float(features_df['feat_atr_14'].iloc[-1])
        
        setup_result = generate_signal(
            ticker.upper(), 
            anomaly_data["current_price"], 
            anomaly_data["anomaly_score"], 
            anomaly_data["p_value"], 
            anomaly_data["regime"], 
            rsi, 
            atr
        )
        
        # Simpan Setup Trading ke Database jika menghasilkan signal TRADE
        if setup_result.get("action") == "TRADE":
            db_setup = {
                "ticker": ticker.upper(),
                "timestamp": anomaly_data["timestamp"],
                "signal": setup_result["setup"]["signal"],
                "entry_price": setup_result["setup"]["entry"],
                "stop_loss": setup_result["setup"]["stop_loss"],
                "take_profit_1": setup_result["setup"]["take_profit_1"],
                "take_profit_2": setup_result["setup"]["take_profit_2"],
                "regime": anomaly_data["regime"],
                "anomaly_score": anomaly_data["anomaly_score"],
                "confidence": str(anomaly_data["p_value"])
            }
            crud.save_trade_setup(db, db_setup)
            
        return setup_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))