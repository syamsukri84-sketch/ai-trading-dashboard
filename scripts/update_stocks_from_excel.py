import pandas as pd
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_yaml_from_excel(
    excel_path: str = "Daftar Saham  - Utama - 20260617.xlsx", 
    yaml_path: str = "config/stocks.yaml"
):
    # Gunakan path absolut untuk memastikan file ditemukan jika script dijalankan dari mana saja
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_excel_path = os.path.join(base_dir, excel_path)
    full_yaml_path = os.path.join(base_dir, yaml_path)

    if not os.path.exists(full_excel_path):
        logger.error(f"File Excel tidak ditemukan: {full_excel_path}")
        return
        
    try:
        logger.info(f"Membaca file Excel: {excel_path}...")
        df = pd.read_excel(full_excel_path)
        
        # Mencari kolom yang berisi kode saham (biasanya 'Kode', 'Ticker', 'Symbol', 'Code')
        ticker_col = None
        for col in df.columns:
            if str(col).strip().lower() in ['kode', 'ticker', 'symbol', 'code', 'kode saham']:
                ticker_col = col
                break
        
        # Jika tidak ada header standar, asumsikan kolom pertama adalah daftar ticker
        if not ticker_col:
            ticker_col = df.columns[0]
            logger.warning(f"Header standar tidak ditemukan. Menggunakan kolom pertama: '{ticker_col}'")
            
        # Ambil datanya, ubah ke string, hapus spasi kosong, dan buat jadi HURUF KAPITAL
        new_tickers = df[ticker_col].dropna().astype(str).str.strip().str.upper().tolist()
        
        # Baca file stocks.yaml yang sudah ada
        if os.path.exists(full_yaml_path):
            with open(full_yaml_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
            
        existing_tickers = config.get('tickers', [])
        
        # Gabungkan saham lama dengan yang baru (Gunakan set untuk menghilangkan duplikat)
        combined_tickers = list(set(existing_tickers + new_tickers))
        combined_tickers.sort() # Urutkan sesuai abjad agar rapi
        
        # Tulis kembali ke stocks.yaml
        config['tickers'] = combined_tickers
        with open(full_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"✅ Berhasil! {len(new_tickers)} saham dari Excel diproses.")
        logger.info(f"💾 File {yaml_path} sekarang memiliki total {len(combined_tickers)} saham unik.")
        
    except Exception as e:
        logger.error(f"Gagal memproses file: {e}")

if __name__ == "__main__":
    update_yaml_from_excel()