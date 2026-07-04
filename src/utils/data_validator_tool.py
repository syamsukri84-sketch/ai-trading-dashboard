import pandas as pd
import logging

# Menggunakan try-except untuk fleksibilitas path
try:
    from src.data_pipeline.data_loader import DataLoader
except ImportError:
    from data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_data_sources(ticker: str, start_date: str = "2023-01-01"):
    """
    Membandingkan data dari CSV lokal (Investing.com) dengan yfinance
    untuk memvalidasi sinkronisasi.
    """
    loader = DataLoader(min_rows=10) # min_rows rendah untuk testing

    print(f"\n--- Memuat data {ticker} dari CSV Lokal (jika ada) ---")
    df_local = loader.load_data(ticker, start_date=start_date, interval="1d", force_download=False)
    if df_local is None:
        print(f"Tidak dapat memuat data lokal untuk {ticker}. Pastikan file '{ticker}_raw.csv' ada di 'data/raw/'.")
        return

    print(f"\n--- Memuat data {ticker} dari yfinance (paksa unduh) ---")
    df_yfinance = loader.load_data(ticker, start_date=start_date, interval="1d", force_download=True)
    if df_yfinance is None:
        print(f"Tidak dapat memuat data dari yfinance untuk {ticker}.")
        return

    # Gabungkan kedua dataframe berdasarkan tanggal
    comparison_df = pd.merge(
        df_local[['timestamp', 'close', 'volume']],
        df_yfinance[['timestamp', 'close', 'volume']],
        on='timestamp',
        suffixes=('_local', '_yfinance')
    )

    # Hitung perbedaan dan tampilkan hanya baris yang berbeda
    comparison_df['close_diff'] = (comparison_df['close_local'] - comparison_df['close_yfinance']).abs()
    comparison_df['volume_diff'] = (comparison_df['volume_local'] - comparison_df['volume_yfinance']).abs()

    differences = comparison_df[(comparison_df['close_diff'] > 0.01) | (comparison_df['volume_diff'] > 1)]

    if differences.empty:
        print("\n✅ SINKRON! Tidak ada perbedaan signifikan antara data lokal dan yfinance.")
    else:
        print("\n⚠️ DITEMUKAN PERBEDAAN! Berikut adalah baris data yang tidak sinkron:")
        print(differences)

if __name__ == "__main__":
    # Ganti 'BBRI' dengan ticker lain jika perlu
    compare_data_sources(ticker="BBRI")