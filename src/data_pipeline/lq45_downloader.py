import os
import logging
import time
from datetime import datetime, timedelta
from typing import List

# Menyesuaikan dengan struktur import berdasarkan file data_loader
try:
    from src.data_pipeline.data_loader import DataLoader
except ImportError:
    from data_loader import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Daftar Emiten LQ45 (Representatif / Update terbaru)
LQ45_TICKERS = [
    "ACES", "ADRO", "AKRA", "AMRT", "ANTM", "ARTO", "ASII", "BBCA", "BBNI",
    "BBRI", "BBTN", "BMRI", "BRPT", "BUKA", "CPIN", "EMTK", "ESSA", "EXCL",
    "GOTO", "HRUM", "ICBP", "INCO", "INDF", "INKP", "INTP", "ITMG", "JPFA",
    "KLBF", "MDKA", "MEDC", "PGAS", "PTBA", "SCMA", "SIDO", "SMGR", "SRTG",
    "TBIG", "TINS", "TLKM", "TOWR", "TPIA", "UNTR", "UNVR"
]

def download_all_lq45(output_dir: str = "data/raw", interval: str = "1h", min_rows: int = 756) -> None:
    """
    Fungsi untuk menarik data maksimal 1-jam (730 hari ke belakang) dari yfinance.
    """
    # Memastikan folder output tersedia
    os.makedirs(output_dir, exist_ok=True)
    
    loader = DataLoader(min_rows=min_rows)
    success_count = 0
    
    logger.info(f"Memulai proses unduh data LQ45 (Interval: {interval}). Total: {len(LQ45_TICKERS)} emiten.")
    
    # Set start_date 729 hari ke belakang untuk mengakali batas 730 hari yfinance
    start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    for ticker in LQ45_TICKERS:
        logger.info(f"Memproses {ticker}...")
        
        # Panggil load_data dengan force_download=True untuk memastikan data baru diunduh
        df = loader.load_data(
            ticker=ticker, 
            interval=interval, 
            start_date=start_date, 
            end_date=end_date,
            force_download=True
        )
        
        if df is not None and not df.empty:
            # Simpan ke CSV
            save_path = os.path.join(output_dir, f"{ticker}_raw.csv")
            df.to_csv(save_path, index=False)
            logger.info(f"Berhasil menyimpan {ticker} ke {save_path} ({len(df)} baris)")
            success_count += 1
        else:
            logger.warning(f"Gagal menarik atau memvalidasi data untuk {ticker}. Melewati...")
            
        # Jeda sejenak untuk menghindari rate-limit dari API Yahoo Finance
        time.sleep(1)
        
    logger.info("=" * 40)
    logger.info(f"PROSES SELESAI. Berhasil: {success_count} / {len(LQ45_TICKERS)}")
    logger.info("=" * 40)

if __name__ == "__main__":
    # Jalankan penarikan data secara otomatis
    download_all_lq45()