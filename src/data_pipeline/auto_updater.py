import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd
import yaml
import yfinance as yf

try:
    from yahooquery import Ticker as YahooQueryTicker
except ImportError:
    YahooQueryTicker = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")
YFINANCE_CACHE_DIR = os.path.join("data", "cache", "yfinance")

try:
    os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
    yf.set_tz_cache_location(YFINANCE_CACHE_DIR)
except Exception as e:
    logger.debug("Tidak dapat mengatur cache yfinance: %s", e)


def _load_tickers(config_path: str) -> List[str]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config.get("tickers", [])


def _normalize_existing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menormalkan CSV lokal ke format standar agar update selalu incremental.
    Mendukung format standar dan sebagian format ekspor Investing.com.
    """
    column_map = {
        "Tanggal": "timestamp",
        "Terakhir": "close",
        "Buka": "open",
        "Pembukaan": "open",
        "Tinggi": "high",
        "Tertinggi": "high",
        "Rendah": "low",
        "Terendah": "low",
        "Vol.": "volume",
        "date": "timestamp",
        "datetime": "timestamp",
    }
    normalized = df.rename(columns=column_map).copy()
    normalized.columns = [str(col).lower() if col != "Perubahan%" else col for col in normalized.columns]

    if "timestamp" not in normalized.columns:
        raise ValueError("Kolom timestamp/date/Tanggal tidak ditemukan.")

    timestamp_text = normalized["timestamp"].astype(str).str.strip()
    iso_mask = timestamp_text.str.match(ISO_DATE_PATTERN)
    parsed_timestamp = pd.Series(pd.NaT, index=normalized.index, dtype="datetime64[ns]")
    parsed_timestamp.loc[iso_mask] = pd.to_datetime(timestamp_text[iso_mask], errors="coerce")
    parsed_timestamp.loc[~iso_mask] = pd.to_datetime(timestamp_text[~iso_mask], format="mixed", dayfirst=True, errors="coerce")
    if parsed_timestamp.isna().any():
        raise ValueError("Sebagian tanggal tidak dapat dibaca.")
    normalized["timestamp"] = parsed_timestamp

    for col in ["open", "high", "low", "close"]:
        if col in normalized.columns and not pd.api.types.is_numeric_dtype(normalized[col]):
            normalized[col] = (
                normalized[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

    if "volume" in normalized.columns and not pd.api.types.is_numeric_dtype(normalized["volume"]):
        def parse_volume(value):
            if pd.isna(value) or value == "-":
                return 0.0
            text = str(value).replace(",", ".")
            if "M" in text:
                return float(text.replace("M", "")) * 1_000_000
            if "K" in text:
                return float(text.replace("K", "")) * 1_000
            if "B" in text:
                return float(text.replace("B", "")) * 1_000_000_000
            return float(text)

        normalized["volume"] = normalized["volume"].apply(parse_volume)

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in normalized.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak lengkap: {missing}")

    return normalized[required_cols].sort_values("timestamp").reset_index(drop=True)


def _normalize_downloaded_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    normalized = normalized.reset_index()
    normalized.columns = normalized.columns.str.lower()
    normalized = normalized.rename(columns={"date": "timestamp", "datetime": "timestamp"})
    normalized = normalized[["timestamp", "open", "high", "low", "close", "volume"]]
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"]).dt.tz_localize(None)
    return normalized


def _download_with_yfinance(ticker_yf: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(ticker_yf, start=start_date, end=end_date, interval="1d", progress=False)
    return _normalize_downloaded_data(data)


def _download_with_yahooquery(ticker_yf: str, start_date: str, end_date: str) -> pd.DataFrame:
    if YahooQueryTicker is None:
        raise ImportError("Paket yahooquery belum terinstal.")

    data = YahooQueryTicker(ticker_yf).history(start=start_date, end=end_date, interval="1d")
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
    else:
        data = data.reset_index()
        data["symbol"] = ticker_yf

    data.columns = data.columns.str.lower()
    data = data.rename(columns={"date": "timestamp"})
    if "timestamp" not in data.columns and "index" in data.columns:
        data = data.rename(columns={"index": "timestamp"})

    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.tz_localize(None)
    return data


def _download_incremental_data(ticker_yf: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    errors = []
    for provider_name, downloader in [
        ("yfinance", _download_with_yfinance),
        ("yahooquery", _download_with_yahooquery),
    ]:
        try:
            data = downloader(ticker_yf, start_date, end_date)
        except Exception as e:
            errors.append(f"{provider_name}: {e}")
            continue

        if not data.empty:
            return data, provider_name

        errors.append(f"{provider_name}: tidak ada data")

    raise RuntimeError("; ".join(errors))


def get_local_data_status(tickers: Optional[List[str]] = None, data_dir: str = "data/raw") -> pd.DataFrame:
    selected_tickers = tickers or []
    rows = []

    for ticker in selected_tickers:
        clean_ticker = str(ticker).replace(".JK", "").upper().strip()
        file_path = os.path.join(data_dir, f"{clean_ticker}_raw.csv")
        if not os.path.exists(file_path):
            rows.append({
                "ticker": clean_ticker,
                "last_date": None,
                "last_close": None,
                "rows": 0,
                "status": "FILE TIDAK ADA",
            })
            continue

        try:
            df = _normalize_existing_data(pd.read_csv(file_path))
            last_row = df.sort_values("timestamp").iloc[-1]
            rows.append({
                "ticker": clean_ticker,
                "last_date": last_row["timestamp"].strftime("%Y-%m-%d"),
                "last_close": float(last_row["close"]),
                "rows": len(df),
                "status": "OK",
            })
        except Exception as e:
            rows.append({
                "ticker": clean_ticker,
                "last_date": None,
                "last_close": None,
                "rows": 0,
                "status": f"ERROR: {e}",
            })

    return pd.DataFrame(rows)


def update_from_manual_dataframe(
    ticker: str,
    manual_df: pd.DataFrame,
    data_dir: str = "data/raw",
    source_name: str = "manual_csv",
) -> Dict[str, object]:
    clean_ticker = str(ticker).replace(".JK", "").upper().strip()
    if not clean_ticker:
        raise ValueError("Ticker wajib diisi.")

    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{clean_ticker}_raw.csv")
    new_data = _normalize_existing_data(manual_df)

    existing_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if os.path.exists(file_path):
        existing_df = _normalize_existing_data(pd.read_csv(file_path))

    before_rows = len(existing_df)
    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
    combined_df.to_csv(file_path, index=False)

    return {
        "ticker": clean_ticker,
        "provider": source_name,
        "rows_added": max(len(combined_df) - before_rows, 0),
        "rows_imported": len(new_data),
        "total_rows": len(combined_df),
        "last_date": combined_df["timestamp"].max().strftime("%Y-%m-%d"),
    }


def run_auto_updater(
    config_path: str = "config/stocks.yaml",
    data_dir: str = "data/raw",
    tickers: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    sleep_seconds: float = 0.2,
) -> Dict[str, object]:
    """
    Mengunduh dan memperbarui data saham ke CSV lokal.
    Hanya mengunduh hari-hari yang belum ada di file CSV.
    """
    os.makedirs(data_dir, exist_ok=True)
    summary: Dict[str, object] = {
        "updated": [],
        "skipped": [],
        "failed": [],
        "total": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
    }

    def notify(ticker: str, message: str) -> None:
        logger.info(message)
        if progress_callback:
            progress_callback(ticker, message)

    try:
        selected_tickers = tickers or _load_tickers(config_path)
    except FileNotFoundError:
        reason = f"File konfigurasi {config_path} tidak ditemukan."
        logger.error(reason)
        summary["failed"].append({"ticker": "CONFIG", "reason": reason})
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return summary

    summary["total"] = len(selected_tickers)
    today = datetime.now()
    today_date = today.date()
    # yfinance memakai end date eksklusif, jadi untuk mengambil data hari ini
    # end harus diarahkan ke besok.
    yf_end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in selected_tickers:
        ticker_yf = f"{ticker}.JK" if not ticker.endswith(".JK") else ticker
        clean_ticker = ticker.replace(".JK", "")
        file_path = os.path.join(data_dir, f"{clean_ticker}_raw.csv")

        existing_df = None
        start_date = (today - timedelta(days=365 * 4)).strftime("%Y-%m-%d")

        if os.path.exists(file_path):
            try:
                existing_df = _normalize_existing_data(pd.read_csv(file_path))
                last_date = existing_df["timestamp"].max()
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                notify(clean_ticker, f"[{clean_ticker}] Data terakhir: {last_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                existing_df = None
                notify(clean_ticker, f"[{clean_ticker}] File lokal tidak dapat dibaca, akan diunduh ulang. Error: {e}")

        if existing_df is not None and not existing_df.empty and existing_df["timestamp"].max().date() >= today_date:
            summary["skipped"].append({"ticker": clean_ticker, "reason": "Data sudah paling up-to-date."})
            notify(clean_ticker, f"[{clean_ticker}] Data sudah paling up-to-date.")
            continue

        notify(clean_ticker, f"[{clean_ticker}] Mengunduh data dari {start_date} hingga {today.strftime('%Y-%m-%d')}...")
        try:
            new_data, provider_name = _download_incremental_data(ticker_yf, start_date, yf_end_date)
        except Exception as e:
            summary["failed"].append({"ticker": clean_ticker, "reason": str(e)})
            notify(clean_ticker, f"[{clean_ticker}] Gagal mengunduh data: {e}")
            continue

        if new_data.empty:
            summary["skipped"].append({"ticker": clean_ticker, "reason": "Tidak ada data perdagangan baru."})
            notify(clean_ticker, f"[{clean_ticker}] Tidak ada data perdagangan baru.")
            continue

        before_rows = len(existing_df) if existing_df is not None and not existing_df.empty else 0
        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat([existing_df, new_data]).drop_duplicates(subset=["timestamp"], keep="last")
        else:
            combined_df = new_data

        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
        rows_added = max(len(combined_df) - before_rows, 0)
        if rows_added == 0 and before_rows > 0:
            summary["skipped"].append({"ticker": clean_ticker, "reason": "Tidak ada tanggal perdagangan unik baru."})
            notify(clean_ticker, f"[{clean_ticker}] Tidak ada tanggal perdagangan unik baru.")
            continue

        combined_df.to_csv(file_path, index=False)
        summary["updated"].append({
            "ticker": clean_ticker,
            "provider": provider_name,
            "rows_added": rows_added,
            "total_rows": len(combined_df),
        })
        notify(clean_ticker, f"[{clean_ticker}] Berhasil diperbarui via {provider_name}. Total baris: {len(combined_df)}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return summary


if __name__ == "__main__":
    print("Memulai auto-update data saham...")
    run_auto_updater()
