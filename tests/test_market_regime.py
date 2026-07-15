import pandas as pd

from src.trading.market_regime import (
    compute_market_breadth,
    load_regime_history,
    log_regime_snapshot,
    summarize_regime_streaks,
)


def _write_raw(raw_dir, ticker, prev_close, last_close):
    pd.DataFrame({"close": [prev_close, last_close]}).to_csv(raw_dir / f"{ticker}_raw.csv", index=False)


def test_compute_market_breadth_classifies_rebound(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw(raw_dir, "AAA", 100, 105)
    _write_raw(raw_dir, "BBB", 100, 103)
    _write_raw(raw_dir, "CCC", 100, 102)

    result = compute_market_breadth(["AAA", "BBB", "CCC"], raw_dir=str(raw_dir))

    assert result["market_regime"] == "REBOUND"
    assert result["breadth_up_pct"] == 100.0
    assert result["sample_size"] == 3


def test_compute_market_breadth_classifies_bearish(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw(raw_dir, "AAA", 100, 95)
    _write_raw(raw_dir, "BBB", 100, 97)
    _write_raw(raw_dir, "CCC", 100, 98)

    result = compute_market_breadth(["AAA", "BBB", "CCC"], raw_dir=str(raw_dir))

    assert result["market_regime"] == "BEARISH"


def test_compute_market_breadth_handles_missing_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    result = compute_market_breadth(["ZZZ"], raw_dir=str(raw_dir))

    assert result["market_regime"] == "UNKNOWN"
    assert result["sample_size"] == 0


def test_log_regime_snapshot_dedups_by_date(tmp_path):
    path = tmp_path / "regime_history.csv"
    breadth = {"market_regime": "BEARISH", "breadth_up_pct": 20.0, "avg_latest_return_pct": -1.5, "sample_size": 10}

    log_regime_snapshot(breadth, date="2026-07-10", path=str(path))
    log_regime_snapshot(breadth, date="2026-07-10", path=str(path))
    updated = log_regime_snapshot({**breadth, "breadth_up_pct": 15.0}, date="2026-07-10", path=str(path))

    assert len(updated) == 1
    assert updated.iloc[0]["breadth_up_pct"] == 15.0


def test_log_regime_snapshot_skips_unknown_regime(tmp_path):
    path = tmp_path / "regime_history.csv"
    breadth = {"market_regime": "UNKNOWN", "breadth_up_pct": 0.0, "avg_latest_return_pct": 0.0, "sample_size": 0}

    result = log_regime_snapshot(breadth, date="2026-07-10", path=str(path))

    assert result.empty
    assert not path.exists()


def test_summarize_regime_streaks_computes_current_and_average_duration(tmp_path):
    path = tmp_path / "regime_history.csv"
    dates_regimes = [
        ("2026-07-01", "MIXED"),
        ("2026-07-02", "BEARISH"),
        ("2026-07-03", "BEARISH"),
        ("2026-07-04", "BEARISH"),
        ("2026-07-05", "REBOUND"),
        ("2026-07-06", "REBOUND"),
    ]
    for date, regime in dates_regimes:
        log_regime_snapshot(
            {"market_regime": regime, "breadth_up_pct": 50.0, "avg_latest_return_pct": 0.0, "sample_size": 5},
            date=date,
            path=str(path),
        )

    history = load_regime_history(str(path))
    summary = summarize_regime_streaks(history)

    assert summary["current_regime"] == "REBOUND"
    assert summary["current_streak_days"] == 2
    assert summary["avg_duration_by_regime"]["BEARISH"] == 3.0


def test_load_regime_history_returns_empty_when_missing(tmp_path):
    path = tmp_path / "missing.csv"

    df = load_regime_history(str(path))

    assert df.empty
    assert "market_regime" in df.columns
