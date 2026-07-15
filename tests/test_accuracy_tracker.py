import pandas as pd

from src.utils.accuracy_tracker import (
    evaluate_pending_predictions,
    get_confidence_calibration_summary,
    get_best_model_recommendations,
    get_daily_accuracy_recap,
    get_model_accuracy_summary,
    get_model_trading_leaderboard,
    get_model_trust_audit,
    get_overall_daily_accuracy_recap,
    log_prediction,
)


def test_daily_accuracy_recap_and_best_model_recommendation(tmp_path):
    path = tmp_path / "accuracy_log.csv"
    pd.DataFrame([
        {
            "evaluation_date": "2026-06-22 09:00:00",
            "ticker": "BBRI",
            "model_name": "XGBoost",
            "direction_correct": True,
            "error_margin_pct": 2.0,
            "prediction_run_type": "FINAL",
        },
        {
            "evaluation_date": "2026-06-22 10:00:00",
            "ticker": "BBRI",
            "model_name": "XGBoost",
            "direction_correct": True,
            "error_margin_pct": 3.0,
            "prediction_run_type": "FINAL",
        },
        {
            "evaluation_date": "2026-06-22 11:00:00",
            "ticker": "BBRI",
            "model_name": "LSTM",
            "direction_correct": False,
            "error_margin_pct": 1.0,
            "prediction_run_type": "FINAL",
        },
        {
            "evaluation_date": "2026-06-23 09:00:00",
            "ticker": "BBCA",
            "model_name": "LSTM",
            "direction_correct": True,
            "error_margin_pct": 1.5,
            "prediction_run_type": "FINAL",
        },
    ]).to_csv(path, index=False)

    summary = get_model_accuracy_summary(str(path))
    daily = get_daily_accuracy_recap(str(path))
    overall_daily = get_overall_daily_accuracy_recap(str(path))
    recommendations = get_best_model_recommendations(str(path), min_evaluations=2)

    bbri_xgb = summary[(summary["ticker"] == "BBRI") & (summary["model_name"] == "XGBoost")].iloc[0]
    assert bbri_xgb["total_evaluations"] == 2
    assert bbri_xgb["direction_accuracy_pct"] == 100
    assert bbri_xgb["avg_error_margin_pct"] == 2.5

    daily_bbri = daily[(daily["evaluation_day"] == "2026-06-22") & (daily["ticker"] == "BBRI")]
    assert len(daily_bbri) == 2

    overall_xgb = overall_daily[
        (overall_daily["evaluation_day"] == "2026-06-22")
        & (overall_daily["model_name"] == "XGBoost")
    ].iloc[0]
    assert overall_xgb["total_predictions"] == 2
    assert overall_xgb["correct_predictions"] == 2
    assert overall_xgb["wrong_predictions"] == 0
    assert overall_xgb["direction_accuracy_pct"] == 100

    best_bbri = recommendations[recommendations["ticker"] == "BBRI"].iloc[0]
    assert best_bbri["model_name"] == "XGBoost"
    assert best_bbri["sample_status"] == "CUKUP"


def test_trading_leaderboard_and_calibration_summary(tmp_path):
    acc_path = tmp_path / "accuracy_log.csv"
    pred_path = tmp_path / "predictions_log.csv"
    pd.DataFrame([
        {
            "evaluation_date": "2026-06-22 09:00:00",
            "ticker": "BBRI",
            "model_name": "Direction-LightGBM",
            "predicted_date": "2026-06-21",
            "target_date_requested": "2026-06-22",
            "predicted_direction": "NAIK",
            "direction_correct": True,
            "predicted_return_pct": 1.0,
            "actual_return_pct": 2.0,
            "error_margin_pct": 1.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        },
        {
            "evaluation_date": "2026-06-23 09:00:00",
            "ticker": "BBRI",
            "model_name": "Direction-LightGBM",
            "predicted_date": "2026-06-22",
            "target_date_requested": "2026-06-23",
            "predicted_direction": "TURUN",
            "direction_correct": False,
            "predicted_return_pct": -1.0,
            "actual_return_pct": 1.0,
            "error_margin_pct": 2.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        },
    ]).to_csv(acc_path, index=False)
    pd.DataFrame([
        {
            "ticker": "BBRI",
            "model_name": "Direction-LightGBM",
            "current_date": "2026-06-21",
            "target_date": "2026-06-22",
            "confidence_pct": 72.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        },
        {
            "ticker": "BBRI",
            "model_name": "Direction-LightGBM",
            "current_date": "2026-06-22",
            "target_date": "2026-06-23",
            "confidence_pct": 68.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        },
    ]).to_csv(pred_path, index=False)

    leaderboard = get_model_trading_leaderboard(str(acc_path), prediction_purpose="NEXT_DAY_DIRECTION", min_evaluations=1)
    calibration = get_confidence_calibration_summary(str(pred_path), str(acc_path), prediction_purpose="NEXT_DAY_DIRECTION")

    assert not leaderboard.empty
    assert "trading_score" in leaderboard.columns
    assert int(leaderboard.iloc[0]["naik_signals"]) == 1
    assert leaderboard.iloc[0]["precision_naik_pct"] == 100
    assert leaderboard.iloc[0]["avg_return_after_naik_pct"] == 2.0
    assert not calibration.empty
    assert "calibration_gap_pct" in calibration.columns


def test_model_trust_audit_statuses(tmp_path):
    acc_path = tmp_path / "accuracy_log.csv"
    pred_path = tmp_path / "predictions_log.csv"
    records = []
    pred_records = []
    # BBRI: model benar-benar mendiskriminasi arah (7 NAIK, 5 TURUN pada data
    # aktual) dan selalu benar -- baseline tebak-mayoritas cuma ~58%, jadi ada
    # edge nyata ~42pp. Ini beda dari skenario "akurasi 100% tapi market
    # kebetulan naik terus" yang diuji terpisah di bawah.
    for i in range(12):
        target = f"2026-06-{i + 2:02d}"
        predicted = f"2026-06-{i + 1:02d}"
        actual_up = i < 7
        direction = "NAIK" if actual_up else "TURUN"
        records.append({
            "evaluation_date": f"{target} 09:00:00",
            "ticker": "BBRI",
            "model_name": "Direction-LightGBM",
            "predicted_date": predicted,
            "target_date_requested": target,
            "predicted_direction": direction,
            "direction_correct": True,
            "predicted_return_pct": 1.0 if actual_up else -1.0,
            "actual_return_pct": 1.2 if actual_up else -1.2,
            "error_margin_pct": 1.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        })
        pred_records.append({
            "ticker": "BBRI",
            "model_name": "Direction-LightGBM",
            "current_date": predicted,
            "target_date": target,
            "confidence_pct": 88.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        })
    for i in range(3):
        target = f"2026-07-{i + 2:02d}"
        predicted = f"2026-07-{i + 1:02d}"
        records.append({
            "evaluation_date": f"{target} 09:00:00",
            "ticker": "BMRI",
            "model_name": "Direction-XGBoost",
            "predicted_date": predicted,
            "target_date_requested": target,
            "predicted_direction": "NAIK",
            "direction_correct": True,
            "predicted_return_pct": 1.0,
            "actual_return_pct": 1.0,
            "error_margin_pct": 1.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        })
        pred_records.append({
            "ticker": "BMRI",
            "model_name": "Direction-XGBoost",
            "current_date": predicted,
            "target_date": target,
            "confidence_pct": 85.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        })

    pd.DataFrame(records).to_csv(acc_path, index=False)
    pd.DataFrame(pred_records).to_csv(pred_path, index=False)

    audit = get_model_trust_audit(
        str(pred_path),
        str(acc_path),
        prediction_purpose="NEXT_DAY_DIRECTION",
        min_evaluations=10,
        max_abs_calibration_gap_pct=20.0,
    )

    bbri = audit[audit["ticker"] == "BBRI"].iloc[0]
    bmri = audit[audit["ticker"] == "BMRI"].iloc[0]
    assert bbri["status_trust"] == "LAYAK DIPERCAYA"
    assert bbri["edge_vs_baseline_pct"] > 0
    assert bmri["status_trust"] == "PERLU DATA LAGI"


def test_model_trust_audit_flags_illusory_accuracy_without_baseline_edge(tmp_path):
    """Akurasi mentah 100% tapi TIDAK ada edge (market kebetulan naik terus
    sepanjang periode evaluasi) harus DITOLAK, bukan LAYAK DIPERCAYA -- ini
    persis pola bug yang ditemukan & diperbaiki sepanjang sesi optimasi model.
    """
    acc_path = tmp_path / "accuracy_log.csv"
    pred_path = tmp_path / "predictions_log.csv"
    records = []
    pred_records = []
    for i in range(12):
        target = f"2026-06-{i + 2:02d}"
        predicted = f"2026-06-{i + 1:02d}"
        records.append({
            "evaluation_date": f"{target} 09:00:00",
            "ticker": "ANEK",
            "model_name": "Direction-XGBoost",
            "predicted_date": predicted,
            "target_date_requested": target,
            "predicted_direction": "NAIK",
            "direction_correct": True,
            "predicted_return_pct": 1.0,
            "actual_return_pct": 1.2,
            "error_margin_pct": 1.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        })
        pred_records.append({
            "ticker": "ANEK",
            "model_name": "Direction-XGBoost",
            "current_date": predicted,
            "target_date": target,
            "confidence_pct": 88.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        })

    pd.DataFrame(records).to_csv(acc_path, index=False)
    pd.DataFrame(pred_records).to_csv(pred_path, index=False)

    audit = get_model_trust_audit(
        str(pred_path),
        str(acc_path),
        prediction_purpose="NEXT_DAY_DIRECTION",
        min_evaluations=10,
        max_abs_calibration_gap_pct=20.0,
    )

    anek = audit[audit["ticker"] == "ANEK"].iloc[0]
    assert anek["direction_accuracy_pct"] == 100.0
    assert anek["edge_vs_baseline_pct"] == 0.0
    assert anek["status_trust"] == "JANGAN DIIKUTI"
    assert "edge" in anek["alasan"].lower()


def test_overwrite_keeps_prediction_history_but_supersedes_old_pending(tmp_path, monkeypatch):
    pred_path = tmp_path / "predictions_log.csv"
    monkeypatch.setattr("src.utils.accuracy_tracker.PREDICTIONS_FILE", str(pred_path))

    log_prediction(
        "BBRI",
        "XGBoost",
        "2026-06-24",
        "2026-06-27",
        predicted_price=4100,
        current_price=4000,
        horizon_days=3,
        prediction_purpose="THREE_DAY_FORECAST",
        duplicate_policy="skip",
    )
    log_prediction(
        "BBRI",
        "XGBoost",
        "2026-06-24",
        "2026-06-27",
        predicted_price=3900,
        current_price=4000,
        horizon_days=3,
        prediction_purpose="THREE_DAY_FORECAST",
        duplicate_policy="overwrite",
    )

    pred_df = pd.read_csv(pred_path)

    assert len(pred_df) == 2
    assert pred_df["status"].tolist() == ["SUPERSEDED", "PENDING"]
    assert pred_df["is_active"].astype(str).str.lower().tolist() == ["false", "true"]
    assert pred_df.iloc[0]["superseded_by_policy"] == "OVERWRITE"


def test_backfill_is_excluded_from_trusted_accuracy_by_default(tmp_path):
    acc_path = tmp_path / "accuracy_log.csv"
    pd.DataFrame([
        {
            "evaluation_date": "2026-06-22 09:00:00",
            "ticker": "BBRI",
            "model_name": "XGBoost",
            "direction_correct": True,
            "error_margin_pct": 1.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
        },
        {
            "evaluation_date": "2026-06-22 09:00:00",
            "ticker": "BBRI",
            "model_name": "XGBoost",
            "direction_correct": False,
            "error_margin_pct": 9.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "BACKFILL",
        },
        {
            "evaluation_date": "2026-06-22 09:00:00",
            "ticker": "BBRI",
            "model_name": "XGBoost",
            "direction_correct": False,
            "error_margin_pct": 8.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "INTRADAY",
        },
        {
            "evaluation_date": "2026-06-22 09:00:00",
            "ticker": "BBRI",
            "model_name": "XGBoost",
            "direction_correct": False,
            "error_margin_pct": 7.0,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
            "is_active_at_evaluation": False,
        },
    ]).to_csv(acc_path, index=False)

    trusted = get_model_accuracy_summary(str(acc_path), prediction_purpose="NEXT_DAY_DIRECTION")
    with_backfill = get_model_accuracy_summary(
        str(acc_path),
        prediction_purpose="NEXT_DAY_DIRECTION",
        include_backfill=True,
    )

    assert trusted.iloc[0]["total_evaluations"] == 1
    assert trusted.iloc[0]["direction_accuracy_pct"] == 100
    assert with_backfill.iloc[0]["total_evaluations"] == 4
    assert with_backfill.iloc[0]["direction_accuracy_pct"] == 25


def test_evaluator_skips_inactive_pending_predictions(tmp_path, monkeypatch):
    pred_path = tmp_path / "predictions_log.csv"
    acc_path = tmp_path / "accuracy_log.csv"
    pd.DataFrame([
        {
            "timestamp_prediction": "2026-06-21 09:00:00",
            "ticker": "BBRI",
            "model_name": "XGBoost",
            "current_date": "2026-06-21",
            "target_date": "2026-06-22",
            "current_price": 4000,
            "predicted_price": 4100,
            "horizon_days": 1,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
            "is_active": False,
            "status": "PENDING",
            "superseded_at": "2026-06-21 10:00:00",
        },
        {
            "timestamp_prediction": "2026-06-21 10:00:00",
            "ticker": "BBRI",
            "model_name": "XGBoost",
            "current_date": "2026-06-21",
            "target_date": "2026-06-22",
            "current_price": 4000,
            "predicted_price": 4200,
            "horizon_days": 1,
            "prediction_purpose": "NEXT_DAY_DIRECTION",
            "prediction_run_type": "FINAL",
            "is_active": True,
            "status": "PENDING",
        },
    ]).to_csv(pred_path, index=False)

    class FakeLoader:
        def __init__(self, min_rows=50):
            self.min_rows = min_rows

        def load_data(self, ticker):
            return pd.DataFrame([
                {"timestamp": "2026-06-22", "open": 4000, "high": 4250, "low": 3990, "close": 4210, "volume": 1_000_000}
            ])

    monkeypatch.setattr("src.utils.accuracy_tracker.PREDICTIONS_FILE", str(pred_path))
    monkeypatch.setattr("src.utils.accuracy_tracker.ACCURACY_FILE", str(acc_path))
    monkeypatch.setattr("src.utils.accuracy_tracker.DataLoader", FakeLoader)

    evaluate_pending_predictions()

    pred_df = pd.read_csv(pred_path)
    acc_df = pd.read_csv(acc_path)

    assert pred_df["status"].tolist() == ["PENDING", "EVALUATED"]
    assert len(acc_df) == 1
    assert bool(acc_df.iloc[0]["is_active_at_evaluation"]) is True
    assert acc_df.iloc[0]["predicted_price"] == 4200
