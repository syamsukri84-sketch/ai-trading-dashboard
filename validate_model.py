import argparse
import pandas as pd
from data_loader import DataLoader
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.models.isolation_forest import IsolationForestModel

def validate_pipeline(ticker):
    print(f"[1] Memuat Data {ticker} dari CSV lokal...")
    # Gunakan DataLoader yang sudah otomatis mendeteksi folder data/raw
    loader = DataLoader(min_rows=100) 
    df = loader.load_data(ticker)
    
    if df is None:
        print(f"[ERROR] Gagal memuat data {ticker}. Pastikan file {ticker}_raw.csv ada di data/raw/")
        return
        
    print(f"[OK] Data berhasil dimuat: {len(df)} baris.")
    print(f"     Rentang: {df['timestamp'].min()} s/d {df['timestamp'].max()}")
    
    print("\n[2] Melakukan Feature Engineering (Pembuatan Indikator)...")
    engineer = FeatureEngineer(warmup_period=60)
    features_df = engineer.generate_features(df)
    
    if features_df.empty:
        print("[ERROR] Gagal membuat fitur.")
        return
        
    print(f"[OK] Fitur berhasil dibuat: {len(features_df.columns)} indikator siap digunakan.")
    
    print("\n[3] Melatih Model Isolation Forest...")
    # Setting contamination 5% berarti kita meminta model mencari 5% hari paling anomali
    model = IsolationForestModel(contamination=0.05)
    model.train(features_df)
    print(f"[OK] Model AI berhasil mempelajari pola saham {ticker}!")
    
    print("\n[4] Menguji Kinerja Model (Mencari Anomali)...")
    results = model.predict(features_df)
    
    # Menggabungkan hasil skor anomali ke data harga asli (menyesuaikan baris yang terpotong warmup)
    aligned_df = df.iloc[engineer.warmup_period:].copy()
    aligned_df['anomaly_score'] = results['anomaly_score']
    
    print("\n" + "="*50)
    print(f"[*] TOP 5 HARI PALING ANOMALI {ticker} (MENURUT AI) [*]")
    print("="*50)
    # Urutkan berdasarkan skor anomali tertinggi
    top_anomalies = aligned_df.sort_values('anomaly_score', ascending=False).head(5)
    
    # Format tampilan agar mudah dibaca
    print(top_anomalies[['timestamp', 'close', 'volume', 'anomaly_score']].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validasi model Isolation Forest untuk ticker saham tertentu.")
    parser.add_argument("--ticker", type=str, required=True, help="Simbol ticker saham (contoh: BBCA, BBRI, dll.)")
    args = parser.parse_args()
    
    validate_pipeline(args.ticker.upper())
