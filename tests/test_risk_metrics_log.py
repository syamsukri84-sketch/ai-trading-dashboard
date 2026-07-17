import pandas as pd

from src.utils.risk_metrics_log import (
    get_latest_var95_lookup,
    load_risk_metrics_log,
    log_risk_metrics,
)


def _var_suite(var95_pct=4.0, var95_method="cornish_fisher", var99_pct=7.0, var99_method="historical_mc_bootstrap_avg", n_obs=252, data_terbatas=False):
    return {
        "n_obs": n_obs,
        "data_terbatas": data_terbatas,
        "per_confidence": {
            0.95: {"recommended_pct": var95_pct, "recommended_method": var95_method},
            0.99: {"recommended_pct": var99_pct, "recommended_method": var99_method},
        },
    }


def test_log_risk_metrics_writes_new_row(tmp_path):
    path = tmp_path / "risk_metrics_log.csv"
    log_risk_metrics("BBCA", "2026-07-16", 1.2, 2.5, _var_suite(), path=str(path))

    df = load_risk_metrics_log(str(path))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "BBCA"
    assert row["analysis_date"] == "2026-07-16"
    assert row["var95_recommended_pct"] == 4.0
    assert row["var95_recommended_method"] == "cornish_fisher"
    assert row["var99_recommended_pct"] == 7.0
    assert row["var99_recommended_method"] == "historical_mc_bootstrap_avg"


def test_log_risk_metrics_upserts_same_ticker_and_date(tmp_path):
    path = tmp_path / "risk_metrics_log.csv"
    log_risk_metrics("BBCA", "2026-07-16", 1.2, 2.5, _var_suite(var95_pct=4.0), path=str(path))
    log_risk_metrics("BBCA", "2026-07-16", 1.3, 2.6, _var_suite(var95_pct=4.5), path=str(path))

    df = load_risk_metrics_log(str(path))
    assert len(df) == 1
    assert df.iloc[0]["var95_recommended_pct"] == 4.5


def test_log_risk_metrics_keeps_separate_rows_per_date(tmp_path):
    path = tmp_path / "risk_metrics_log.csv"
    log_risk_metrics("BBCA", "2026-07-15", 1.2, 2.5, _var_suite(var95_pct=4.0), path=str(path))
    log_risk_metrics("BBCA", "2026-07-16", 1.3, 2.6, _var_suite(var95_pct=4.5), path=str(path))
    log_risk_metrics("BBRI", "2026-07-16", 1.0, 2.0, _var_suite(var95_pct=3.0), path=str(path))

    df = load_risk_metrics_log(str(path))
    assert len(df) == 3


def test_log_risk_metrics_handles_var_suite_error_gracefully(tmp_path):
    path = tmp_path / "risk_metrics_log.csv"
    log_risk_metrics("NEWIPO", "2026-07-16", None, None, {"error": "data terlalu sedikit", "n_obs": 3}, path=str(path))

    df = load_risk_metrics_log(str(path))
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["var95_recommended_pct"])


def test_load_risk_metrics_log_missing_file_returns_empty(tmp_path):
    path = tmp_path / "does_not_exist.csv"
    df = load_risk_metrics_log(str(path))
    assert df.empty


def test_get_latest_var95_lookup_returns_latest_per_ticker(tmp_path):
    path = tmp_path / "risk_metrics_log.csv"
    log_risk_metrics("BBCA", "2026-07-14", 1.0, 2.0, _var_suite(var95_pct=3.5), path=str(path))
    log_risk_metrics("BBCA", "2026-07-16", 1.2, 2.5, _var_suite(var95_pct=4.5, var95_method="ewma"), path=str(path))
    log_risk_metrics("BBRI", "2026-07-16", 1.0, 2.0, _var_suite(var95_pct=3.0), path=str(path))

    lookup = get_latest_var95_lookup(str(path))
    assert lookup["BBCA"]["var95_pct"] == 4.5
    assert lookup["BBCA"]["var95_method"] == "ewma"
    assert lookup["BBRI"]["var95_pct"] == 3.0


def test_get_latest_var95_lookup_excludes_tickers_without_valid_var(tmp_path):
    path = tmp_path / "risk_metrics_log.csv"
    log_risk_metrics("NEWIPO", "2026-07-16", None, None, {"error": "data terlalu sedikit", "n_obs": 3}, path=str(path))
    log_risk_metrics("BBCA", "2026-07-16", 1.2, 2.5, _var_suite(var95_pct=4.5), path=str(path))

    lookup = get_latest_var95_lookup(str(path))
    assert "NEWIPO" not in lookup
    assert "BBCA" in lookup


def test_get_latest_var95_lookup_empty_file_returns_empty_dict(tmp_path):
    path = tmp_path / "does_not_exist.csv"
    assert get_latest_var95_lookup(str(path)) == {}
