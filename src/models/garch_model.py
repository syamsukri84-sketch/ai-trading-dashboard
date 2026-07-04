import pandas as pd
import numpy as np
import logging
from typing import Dict

try:
    from arch import arch_model
    from scipy.stats import norm
except ImportError:
    arch_model = None
    norm = None

logger = logging.getLogger(__name__)

class GARCHModel:
    """
    Model GARCH/ARCH untuk memprediksi volatilitas pergerakan harga saham.
    Berbeda dengan XGBoost/LSTM yang memprediksi arah harga, GARCH memprediksi 
    seberapa besar rentang fluktuasi/risiko harga di masa depan.
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        # p: Lag error masa lalu (komponen ARCH)
        # q: Lag variance masa lalu (komponen GARCH)
        # Jika q=0 dan p>0, maka model ini secara otomatis menjadi model ARCH murni.
        self.p = p
        self.q = q
        self.model_result = None

    def _prepare_returns(self, df: pd.DataFrame) -> pd.Series:
        """Menyiapkan persentase return harian (dikali 100 agar konvergensi model GARCH stabil)."""
        # Menggunakan log return atau percentage change
        returns = df['close'].pct_change().dropna() * 100
        return returns

    def train(self, df: pd.DataFrame):
        """Melatih model GARCH berdasarkan histori volatilitas."""
        if arch_model is None:
            logger.error("Library 'arch' belum terinstal. Jalankan: pip install arch")
            return

        returns = self._prepare_returns(df)
        
        # Parameter rescale=False digunakan karena kita sudah mengalikan return dengan 100
        self.model = arch_model(returns, vol='Garch', p=self.p, q=self.q, rescale=False)
        
        # disp='off' untuk mematikan log output iterasi pengoptimalan dari library arch
        self.model_result = self.model.fit(disp='off')
        logger.info(f"Pelatihan model GARCH({self.p},{self.q}) selesai.")

    def predict(self, horizon: int = 3, confidence_level: float = 0.95) -> Dict[str, float]:
        """Memprediksi estimasi volatilitas dan Value at Risk (VaR) ke depan."""
        if self.model_result is None:
            return {
                "projected_volatility_pct": 0.0,
                "value_at_risk_95_pct": 0.0
            }
            
        # Mendapatkan ramalan varians ke depan
        forecasts = self.model_result.forecast(horizon=horizon)
        
        # --- Untuk volatilitas rata-rata (seperti sebelumnya) ---
        projected_variance_horizon = forecasts.variance.iloc[-1].values
        # Konversi Variance ke Standard Deviation (Volatilitas) dan kembalikan ke skala aslinya (/100)
        projected_volatility_horizon = np.sqrt(projected_variance_horizon) / 100.0
        # Rata-rata perkiraan volatilitas per hari
        avg_volatility_pct = float(np.mean(projected_volatility_horizon)) * 100

        # --- Perhitungan Value at Risk (VaR) ---
        var_95_pct = 0.0
        if norm is None:
            logger.warning("Library 'scipy' belum terinstal. VaR tidak dapat dihitung. Jalankan: pip install scipy")
        else:
            # Ambil forecast varians untuk 1 hari ke depan (h.1) untuk VaR
            projected_variance_next_day = forecasts.variance.iloc[-1]['h.1']
            
            # Konversi Variance ke Standard Deviation (Volatilitas) dan kembalikan ke skala aslinya (/100)
            projected_volatility_next_day = np.sqrt(projected_variance_next_day) / 100.0
            
            # Hitung Value at Risk (VaR) pada tingkat kepercayaan yang ditentukan
            # Asumsi return rata-rata = 0 untuk jangka pendek (worst-case loss)
            z_score = norm.ppf(confidence_level)
            var_95_pct = z_score * projected_volatility_next_day * 100
        
        return {
            "projected_volatility_pct": avg_volatility_pct,
            "value_at_risk_95_pct": float(var_95_pct)
        }