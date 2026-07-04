import json
import os
from datetime import datetime
from typing import Iterable

import pandas as pd

from src.utils.accuracy_tracker import ACCURACY_FILE, load_accuracy_log


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAINING_REGISTRY_FILE = os.path.join(PROJECT_ROOT, "data", "tracking", "training_registry.json")


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _month_key(value=None) -> str:
    ts = pd.to_datetime(value if value is not None else datetime.now(), errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.now()
    return ts.strftime("%Y-%m")


def _week_key(value=None) -> str:
    ts = pd.to_datetime(value if value is not None else datetime.now(), errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.now()
    iso = ts.isocalendar()
    return f"{int(iso.year):04d}-W{int(iso.week):02d}"


def load_training_registry(path: str = TRAINING_REGISTRY_FILE) -> dict:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"runs": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"runs": []}
    if not isinstance(data, dict):
        return {"runs": []}
    if "runs" not in data or not isinstance(data["runs"], list):
        data["runs"] = []
    return data


def record_training_run(
    tickers: Iterable[str],
    run_type: str = "FINAL",
    trigger: str = "MANUAL",
    analyzed_count: int = 0,
    skipped_count: int = 0,
    failed_count: int = 0,
    path: str = TRAINING_REGISTRY_FILE,
) -> dict:
    registry = load_training_registry(path)
    now = datetime.now()
    run = {
        "training_run_id": now.strftime("%Y%m%d_%H%M%S"),
        "trained_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "training_month": now.strftime("%Y-%m"),
        "run_type": str(run_type or "FINAL").upper().strip(),
        "trigger": str(trigger or "MANUAL").upper().strip(),
        "tickers": sorted({str(t).replace(".JK", "").upper().strip() for t in tickers if str(t).strip()}),
        "analyzed_count": int(analyzed_count or 0),
        "skipped_count": int(skipped_count or 0),
        "failed_count": int(failed_count or 0),
    }
    registry.setdefault("runs", []).append(run)
    registry["last_run"] = run
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    return run


def latest_training_run(path: str = TRAINING_REGISTRY_FILE) -> dict:
    runs = load_training_registry(path).get("runs", [])
    if not runs:
        return {}
    return runs[-1]


def _recent_accuracy_window(
    accuracy_file: str,
    prediction_purpose: str,
    model_name: str,
    lookback_evaluations: int,
) -> pd.DataFrame:
    df = load_accuracy_log(accuracy_file)
    if df.empty:
        return df
    df = df[df["prediction_purpose"].astype(str).str.upper().str.strip() == prediction_purpose.upper().strip()]
    df = df[df["model_name"].astype(str).str.upper().str.strip() == model_name.upper().strip()]
    if "prediction_run_type" in df.columns:
        df = df[df["prediction_run_type"].astype(str).str.upper().str.strip() == "FINAL"]
    if df.empty:
        return df
    df["evaluation_dt"] = pd.to_datetime(df["evaluation_date"], errors="coerce")
    df = df.dropna(subset=["evaluation_dt"]).sort_values("evaluation_dt")
    return df.tail(max(int(lookback_evaluations or 20), 1))


def evaluate_training_policy(
    accuracy_file: str = ACCURACY_FILE,
    min_direction_accuracy_pct: float = 52.0,
    min_recent_evaluations: int = 20,
    lookback_evaluations: int = 50,
    prediction_purpose: str = "NEXT_DAY_DIRECTION",
    model_name: str = "XGBoost",
    monthly_training_enabled: bool = True,
    emergency_retrain_enabled: bool = True,
    routine_training_interval: str = "MONTHLY",
    registry_path: str = TRAINING_REGISTRY_FILE,
) -> dict:
    lookback_evaluations = max(int(lookback_evaluations or min_recent_evaluations), int(min_recent_evaluations or 1))
    latest_run = latest_training_run(registry_path)
    current_month = _month_key()
    current_week = _week_key()
    last_training_month = str(latest_run.get("training_month", "")) if latest_run else ""
    last_trained_at = latest_run.get("trained_at") if latest_run else None
    last_training_week = _week_key(last_trained_at) if last_trained_at else ""
    interval = str(routine_training_interval or "MONTHLY").upper().strip()
    if interval not in {"WEEKLY", "MONTHLY", "OFF"}:
        interval = "MONTHLY"
    weekly_due = bool(monthly_training_enabled and interval == "WEEKLY" and last_training_week != current_week)
    monthly_due = bool(monthly_training_enabled and interval == "MONTHLY" and last_training_month != current_month)
    routine_due = bool(weekly_due or monthly_due)

    recent_df = _recent_accuracy_window(
        accuracy_file=accuracy_file,
        prediction_purpose=prediction_purpose,
        model_name=model_name,
        lookback_evaluations=lookback_evaluations,
    )
    recent_count = int(len(recent_df))
    recent_accuracy = None
    accuracy_below_threshold = False
    if recent_count > 0:
        recent_accuracy = float(recent_df["direction_correct"].astype(bool).mean() * 100)
        accuracy_below_threshold = (
            recent_count >= int(min_recent_evaluations)
            and recent_accuracy < float(min_direction_accuracy_pct)
        )

    emergency_due = bool(emergency_retrain_enabled and accuracy_below_threshold)
    retrain_due = bool(routine_due or emergency_due)
    if retrain_due:
        status = "PERLU RETRAIN"
    elif recent_count < int(min_recent_evaluations):
        status = "PAKAI MODEL - SAMPEL AKURASI RENDAH"
    else:
        status = "PAKAI MODEL"

    reasons = []
    if weekly_due:
        reasons.append(f"Belum ada training untuk minggu {current_week}.")
    if monthly_due:
        reasons.append(f"Belum ada training untuk bulan {current_month}.")
    if emergency_due:
        reasons.append(
            f"Akurasi {model_name} {prediction_purpose} terbaru {recent_accuracy:.2f}% di bawah batas {float(min_direction_accuracy_pct):.2f}%."
        )
    if not reasons and recent_count < int(min_recent_evaluations):
        reasons.append(f"Evaluasi terbaru baru {recent_count}/{int(min_recent_evaluations)} sampel.")
    if not reasons:
        if interval == "OFF":
            reasons.append("Retrain rutin dimatikan dan akurasi belum melewati batas retrain.")
        elif interval == "WEEKLY":
            reasons.append("Training mingguan masih berlaku dan akurasi belum melewati batas retrain.")
        else:
            reasons.append("Training bulanan masih berlaku dan akurasi belum melewati batas retrain.")

    return {
        "status": status,
        "retrain_due": retrain_due,
        "monthly_due": monthly_due,
        "weekly_due": weekly_due,
        "routine_due": routine_due,
        "emergency_due": emergency_due,
        "current_month": current_month,
        "current_week": current_week,
        "last_training_month": last_training_month or "-",
        "last_training_week": last_training_week or "-",
        "last_trained_at": latest_run.get("trained_at", "-") if latest_run else "-",
        "last_training_run_id": latest_run.get("training_run_id", "-") if latest_run else "-",
        "recent_evaluations": recent_count,
        "recent_accuracy_pct": recent_accuracy,
        "min_direction_accuracy_pct": float(min_direction_accuracy_pct),
        "min_recent_evaluations": int(min_recent_evaluations),
        "lookback_evaluations": int(lookback_evaluations),
        "prediction_purpose": prediction_purpose,
        "model_name": model_name,
        "routine_training_interval": interval,
        "reason": " ".join(reasons),
    }


def evaluate_training_policy_by_model(
    model_names: list[str] | None = None,
    accuracy_file: str = ACCURACY_FILE,
    min_direction_accuracy_pct: float = 52.0,
    min_recent_evaluations: int = 20,
    lookback_evaluations: int = 50,
    prediction_purpose: str = "NEXT_DAY_DIRECTION",
) -> pd.DataFrame:
    if model_names is None:
        df = load_accuracy_log(accuracy_file)
        if df.empty or "model_name" not in df.columns:
            return pd.DataFrame()
        if "prediction_purpose" in df.columns:
            df = df[df["prediction_purpose"].astype(str).str.upper().str.strip() == prediction_purpose.upper().strip()]
        if "prediction_run_type" in df.columns:
            df = df[df["prediction_run_type"].astype(str).str.upper().str.strip() == "FINAL"]
        model_names = sorted(df["model_name"].dropna().astype(str).unique().tolist())

    rows = []
    for model_name in model_names:
        policy = evaluate_training_policy(
            accuracy_file=accuracy_file,
            min_direction_accuracy_pct=min_direction_accuracy_pct,
            min_recent_evaluations=min_recent_evaluations,
            lookback_evaluations=lookback_evaluations,
            prediction_purpose=prediction_purpose,
            model_name=model_name,
            monthly_training_enabled=False,
            emergency_retrain_enabled=True,
        )
        rows.append({
            "model_name": model_name,
            "status": policy["status"],
            "recent_evaluations": policy["recent_evaluations"],
            "recent_accuracy_pct": policy["recent_accuracy_pct"],
            "retrain_due": policy["emergency_due"],
            "reason": policy["reason"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["recent_accuracy_pct"] = pd.to_numeric(out["recent_accuracy_pct"], errors="coerce").round(2)
    return out.sort_values(["retrain_due", "recent_accuracy_pct", "recent_evaluations"], ascending=[False, True, False])
