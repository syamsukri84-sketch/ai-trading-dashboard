import pandas as pd

from src.utils.csv_audit import audit_prediction_csv, clean_prediction_csv


def test_audit_prediction_csv_flags_invalid_rows(tmp_path):
    path = tmp_path / "predictions_log.csv"
    pd.DataFrame(
        [
            {
                "timestamp_prediction": "2026-07-03 21:46:24",
                "ticker": "BBRI",
                "model_name": "Global-Direction-LIGHTGBM",
                "current_date": "2026-07-03",
                "target_date": "2026-07-04",
                "current_price": 4000,
                "predicted_price": 4040,
                "horizon_days": 1,
                "is_active": True,
            },
            {
                "timestamp_prediction": "True",
                "ticker": "",
                "model_name": "SKIP",
                "current_date": "FINAL",
                "target_date": "SKIP",
                "current_price": "bad",
                "predicted_price": "",
                "horizon_days": "NEXT_DAY_DIRECTION",
                "is_active": "maybe",
            },
        ]
    ).to_csv(path, index=False)

    clean_df, audit = audit_prediction_csv(str(path))

    assert audit.total_rows == 2
    assert audit.valid_rows == 1
    assert audit.invalid_rows == 1
    assert not audit.is_clean
    assert clean_df["ticker"].tolist() == ["BBRI"]
    assert "invalid_timestamp_prediction" in audit.invalid_reason_counts


def test_clean_prediction_csv_writes_clean_file_and_backup(tmp_path):
    path = tmp_path / "predictions_log.csv"
    backup_dir = tmp_path / "backups"
    pd.DataFrame(
        [
            {
                "timestamp_prediction": "2026-07-03 21:46:24",
                "ticker": "BBRI",
                "model_name": "Global-Direction-LIGHTGBM",
                "current_date": "2026-07-03",
                "target_date": "2026-07-04",
                "current_price": 4000,
                "predicted_price": 4040,
            },
            {
                "timestamp_prediction": "broken",
                "ticker": "TRUE",
                "model_name": "",
                "current_date": "PENDING",
                "target_date": "FINAL",
                "current_price": 0,
                "predicted_price": 0,
            },
        ]
    ).to_csv(path, index=False)

    backup_path, audit = clean_prediction_csv(str(path), str(backup_dir))
    cleaned = pd.read_csv(path)

    assert audit.invalid_rows == 1
    assert cleaned["ticker"].tolist() == ["BBRI"]
    assert backup_dir.exists()
    assert backup_path.endswith(".csv")
    assert pd.read_csv(backup_path).shape[0] == 2
