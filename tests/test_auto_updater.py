from datetime import datetime

import pandas as pd

from src.data_pipeline.auto_updater import (
    _download_incremental_data,
    _normalize_existing_data,
    get_local_data_status,
    update_from_manual_dataframe,
)


def test_normalize_existing_standard_csv():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-06-19", "2026-06-20"],
            "open": [100, 101],
            "high": [105, 106],
            "low": [99, 100],
            "close": [104, 105],
            "volume": [1000, 1200],
        }
    )

    normalized = _normalize_existing_data(df)

    assert list(normalized.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert normalized["timestamp"].max() == pd.Timestamp(datetime(2026, 6, 20))
    assert normalized["timestamp"].min() == pd.Timestamp(datetime(2026, 6, 19))


def test_normalize_existing_iso_date_does_not_flip_month_day():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-06-12"],
            "open": [100],
            "high": [105],
            "low": [99],
            "close": [104],
            "volume": [1000],
        }
    )

    normalized = _normalize_existing_data(df)

    assert normalized["timestamp"].iloc[0] == pd.Timestamp(datetime(2026, 6, 12))


def test_normalize_existing_investing_csv():
    df = pd.DataFrame(
        {
            "Tanggal": ["20.06.2026"],
            "Terakhir": ["1.250,50"],
            "Buka": ["1.200,00"],
            "Tinggi": ["1.260,00"],
            "Rendah": ["1.190,00"],
            "Vol.": ["2,5M"],
        }
    )

    normalized = _normalize_existing_data(df)

    assert normalized["close"].iloc[0] == 1250.50
    assert normalized["volume"].iloc[0] == 2_500_000


def test_download_incremental_uses_yahooquery_fallback(monkeypatch):
    def empty_yfinance(_ticker, _start, _end):
        return pd.DataFrame()

    def fallback_yahooquery(_ticker, _start, _end):
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-06-22")],
                "open": [100],
                "high": [110],
                "low": [95],
                "close": [105],
                "volume": [1000],
            }
        )

    monkeypatch.setattr("src.data_pipeline.auto_updater._download_with_yfinance", empty_yfinance)
    monkeypatch.setattr("src.data_pipeline.auto_updater._download_with_yahooquery", fallback_yahooquery)

    data, provider = _download_incremental_data("BBRI.JK", "2026-06-22", "2026-06-23")

    assert provider == "yahooquery"
    assert data["close"].iloc[0] == 105


def test_update_from_manual_dataframe_merges_local_data(tmp_path):
    data_dir = tmp_path / "raw"
    existing = pd.DataFrame(
        {
            "timestamp": ["2026-06-19"],
            "open": [100],
            "high": [105],
            "low": [99],
            "close": [104],
            "volume": [1000],
        }
    )
    update_from_manual_dataframe("BBRI", existing, data_dir=str(data_dir), source_name="seed")

    manual = pd.DataFrame(
        {
            "Tanggal": ["20.06.2026"],
            "Terakhir": ["110,00"],
            "Buka": ["105,00"],
            "Tinggi": ["111,00"],
            "Rendah": ["104,00"],
            "Vol.": ["1,5M"],
        }
    )
    summary = update_from_manual_dataframe("BBRI", manual, data_dir=str(data_dir), source_name="manual")
    status = get_local_data_status(["BBRI"], data_dir=str(data_dir))

    assert summary["rows_added"] == 1
    assert summary["last_date"] == "2026-06-20"
    assert status["last_date"].iloc[0] == "2026-06-20"
    assert status["last_close"].iloc[0] == 110.0
