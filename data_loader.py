import pandas as pd
import logging
import os
from datetime import datetime
from typing import Optional

from src.utils.model_guardrails import audit_ohlcv_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader for loading and validating trading data.
    Supports parsing local CSV files (like Investing.com exports) or downloading from yfinance.
    """
    
    def __init__(self, min_rows: int = 756, local_dir: str = "data/raw"):
        self.min_rows = min_rows
        self.local_dir = local_dir
        
        # Otomatis membuat folder jika belum ada
        os.makedirs(self.local_dir, exist_ok=True)

    def _parse_investing_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically formats CSV data exported from Investing.com (Indonesian).
        Fixes localized column names, date formats, and number formatting.
        """
        # 1. Rename columns to our standard
        col_map = {
            'Tanggal': 'timestamp', 
            'Terakhir': 'close', 
            'Buka': 'open', 'Pembukaan': 'open',
            'Tinggi': 'high', 'Tertinggi': 'high', 
            'Rendah': 'low', 'Terendah': 'low', 
            'Vol.': 'volume'
        }
        df = df.rename(columns=col_map)
        
        # 2. Fix DateTime format
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
            
        # 3. Fix Indonesian number format (e.g. 5.000,50 -> 5000.50)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
                
        # 4. Fix Volume string abbreviations (e.g. 12,5M -> 12500000)
        if 'volume' in df.columns and not pd.api.types.is_numeric_dtype(df['volume']):
            def parse_volume(val):
                if pd.isna(val) or val == '-': return 0.0
                val = str(val).replace(',', '.')
                if 'M' in val: return float(val.replace('M', '')) * 1_000_000
                if 'K' in val: return float(val.replace('K', '')) * 1_000
                if 'B' in val: return float(val.replace('B', '')) * 1_000_000_000
                return float(val)
            df['volume'] = df['volume'].apply(parse_volume)
            
        # 5. Investing.com typically sorts newest-first, we need oldest-first for timeseries
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def load_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Memuat dan mem-parsing data historis dari file CSV lokal.
        Nama file harus cocok dengan ticker (misal: 'BBRI_raw.csv').
        """
        clean_ticker = ticker.replace('.JK', '')
        file_path = os.path.join(self.local_dir, f"{clean_ticker}_raw.csv")

        if not os.path.exists(file_path):
            logger.warning(f"File data untuk {ticker} tidak ditemukan di: {file_path}")
            return None

        try:
            logger.info(f"Memuat data dari file: {file_path}")
            
            # Intip dulu baris pertama untuk melihat format kolomnya
            temp_df = pd.read_csv(file_path, nrows=0)

            if 'Tanggal' in temp_df.columns and 'Terakhir' in temp_df.columns:
                df = pd.read_csv(file_path, dtype=str)
                df = self._parse_investing_csv(df)
            else:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.lower()
                if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
                elif 'datetime' in df.columns: df = df.rename(columns={'datetime': 'timestamp'})
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            df = self._repair_latest_intraday_ohlc(df, ticker)

            if self.validate_data(df, ticker):
                return df
            return None
        except Exception as e:
            logger.error(f"Gagal memproses file {file_path}: {e}")
            return None

    def _repair_latest_intraday_ohlc(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Memperbaiki OHLC baris hari berjalan yang belum final.

        Saat pasar masih berjalan, sebagian provider dapat mengirim high/low sementara
        yang belum konsisten terhadap open/close. Perbaikan hanya dilakukan untuk
        tanggal terbaru jika tanggal tersebut sama dengan tanggal hari ini.
        """
        required_cols = {"timestamp", "open", "high", "low", "close"}
        if df.empty or not required_cols.issubset(df.columns):
            return df

        repaired_df = df.copy()
        repaired_df["timestamp"] = pd.to_datetime(repaired_df["timestamp"], errors="coerce")
        for col in ["open", "high", "low", "close"]:
            repaired_df[col] = pd.to_numeric(repaired_df[col], errors="coerce")

        latest_ts = repaired_df["timestamp"].max()
        if pd.isna(latest_ts) or latest_ts.date() != datetime.now().date():
            return repaired_df

        latest_mask = repaired_df["timestamp"].dt.date == latest_ts.date()
        latest_rows = repaired_df.loc[latest_mask, ["open", "high", "low", "close"]]
        if latest_rows.isna().any().any():
            return repaired_df

        repaired_high = latest_rows[["open", "high", "close"]].max(axis=1)
        repaired_low = latest_rows[["open", "low", "close"]].min(axis=1)
        changed = (
            (repaired_df.loc[latest_mask, "high"] != repaired_high)
            | (repaired_df.loc[latest_mask, "low"] != repaired_low)
        )

        if changed.any():
            logger.warning(
                "%s: OHLC hari berjalan belum final; high/low disesuaikan sementara untuk analisis.",
                ticker,
            )
            repaired_df.loc[latest_mask, "high"] = repaired_high
            repaired_df.loc[latest_mask, "low"] = repaired_low

        return repaired_df

    def validate_data(self, df: pd.DataFrame, ticker: str = "Unknown") -> bool:
        """
        Validates data quality based on trading system constraints.
        """
        guardrail = audit_ohlcv_data(df, ticker=ticker, check_price_bounds=False)
        if not guardrail.passed:
            for error in guardrail.errors:
                logger.error(f"Validation failed: {error}")
            return False
        for warning in guardrail.warnings:
            logger.warning(f"Data warning: {warning}")

        # 1. Minimum rows
        if len(df) < self.min_rows:
            logger.error(f"Validation failed for {ticker}: Insufficient rows. Expected >= {self.min_rows}, got {len(df)}")
            return False
            
        # 2. No NaN values
        if df.isnull().values.any():
            logger.error(f"Validation failed for {ticker}: Contains NaN values")
            return False
            
        # 3. Monotonic increasing timestamps
        if not df['timestamp'].is_monotonic_increasing or not df['timestamp'].is_unique:
            logger.error(f"Validation failed for {ticker}: Timestamps are not monotonic increasing or contain duplicates")
            return False
            
        # 4. Positive OHLCV values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        if (df[numeric_cols] < 0).values.any():
            logger.error(f"Validation failed for {ticker}: Contains negative OHLCV values")
            return False
            
        logger.info(f"Data validation passed for {ticker}. Total rows: {len(df)}")
        return True
