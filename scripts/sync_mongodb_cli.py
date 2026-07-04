from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.csv_audit import audit_prediction_csv
from src.utils.mongo_store import check_mongo_status, upload_json_files, upsert_dataframe


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def sync_predictions() -> dict:
    path = PROJECT_ROOT / "data" / "tracking" / "predictions_log.csv"
    clean_df, audit = audit_prediction_csv(str(path))
    if clean_df.empty:
        return {"collection": "predictions", "processed": 0, "invalid_rows": audit.invalid_rows}
    result = upsert_dataframe(
        "predictions",
        clean_df,
        [
            "timestamp_prediction",
            "ticker",
            "model_name",
            "current_date",
            "target_date",
            "horizon_days",
            "prediction_purpose",
        ],
    )
    result["invalid_rows"] = audit.invalid_rows
    result["collection"] = "predictions"
    return result


def sync_accuracy() -> dict:
    path = PROJECT_ROOT / "data" / "tracking" / "accuracy_log.csv"
    df = read_csv(path)
    if df.empty:
        return {"collection": "accuracy_logs", "processed": 0}
    result = upsert_dataframe(
        "accuracy_logs",
        df,
        [
            "evaluation_date",
            "ticker",
            "model_name",
            "predicted_date",
            "target_date_requested",
            "prediction_purpose",
        ],
    )
    result["collection"] = "accuracy_logs"
    return result


def sync_sentiment() -> dict:
    path = PROJECT_ROOT / "data" / "sentiment" / "market_issues.csv"
    df = read_csv(path)
    if df.empty:
        return {"collection": "sentiment_issues", "processed": 0}
    result = upsert_dataframe("sentiment_issues", df, ["date", "ticker", "source", "text"])
    result["collection"] = "sentiment_issues"
    return result


def sync_training_registry() -> dict:
    path = PROJECT_ROOT / "data" / "tracking" / "training_registry.json"
    if not path.exists():
        return {"collection": "training_registry", "processed": 0}
    result = upload_json_files("training_registry", [path])
    result["collection"] = "training_registry"
    return result


def sync_daily_workflows() -> dict:
    paths = sorted((PROJECT_ROOT / "data" / "daily_workflows").glob("daily_global_workflow_*.json"))
    if not paths:
        return {"collection": "daily_workflows", "processed": 0}
    result = upload_json_files("daily_workflows", paths)
    result["collection"] = "daily_workflows"
    return result


SYNC_TASKS = {
    "predictions": sync_predictions,
    "accuracy": sync_accuracy,
    "sentiment": sync_sentiment,
    "training": sync_training_registry,
    "workflows": sync_daily_workflows,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync data lokal AI Trading ke MongoDB Atlas.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(SYNC_TASKS),
        default=sorted(SYNC_TASKS),
        help="Pilih data yang disinkronkan. Default semua.",
    )
    args = parser.parse_args()

    status = check_mongo_status()
    print(f"MongoDB: {status['message']}")
    if not status["ok"]:
        return 1

    for name in args.only:
        result = SYNC_TASKS[name]()
        print(
            f"{result['collection']}: processed={result.get('processed', 0)} "
            f"upserted={result.get('upserted', 0)} modified={result.get('modified', 0)} "
            f"invalid={result.get('invalid_rows', 0)}"
        )
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
