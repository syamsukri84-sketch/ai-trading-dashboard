from data_loader import DataLoader
from datetime import datetime

def test_fetching():
    print("Mencoba memuat data BBRI dari 2023-01-01 hingga saat ini...")
    
    # Kita set min_rows sedikit lebih rendah untuk testing agar tidak langsung error jika hari libur
    loader = DataLoader(min_rows=100) 
    
    # Meminta data. Interval "1d" (harian) digunakan agar jika fallback ke yfinance, 
    # tidak terkena limit 730 hari untuk data 1-jam (1h).
    df = loader.download_data(ticker="BBRI", start_date="2023-01-01", interval="1d")
    
    if df is not None:
        print("\n✅ BERHASIL! Data siap digunakan.")
        print(f"Total baris data: {len(df)}")
        print(f"Data paling awal: {df['timestamp'].min()}")
        print(f"Data paling akhir: {df['timestamp'].max()}")
        print("\nCuplikan 5 data pertama:")
        print(df.head())
    else:
        print("\n❌ Gagal memuat data. Silakan periksa log di atas.")

if __name__ == "__main__":
    test_fetching()