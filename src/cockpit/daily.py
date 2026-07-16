from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRACKING_DIR = DATA_DIR / "tracking"
PREDICTIONS_FILE = TRACKING_DIR / "predictions_log.csv"
ACCURACY_FILE = TRACKING_DIR / "accuracy_log.csv"
EDGE_FILE = DATA_DIR / "edge_screening_status.json"
WORKFLOW_DIR = DATA_DIR / "daily_workflows"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _to_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.fillna(default).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if pd.isna(out):
            return default
        return out
    except Exception:
        return default


def _load_tickers(config_path: Path | None = None) -> list[str]:
    config_path = config_path or PROJECT_ROOT / "config" / "stocks.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    return [str(t).replace(".JK", "").upper().strip() for t in config.get("tickers", []) if str(t).strip()]


def _latest_workflow() -> dict[str, Any] | None:
    files = sorted(WORKFLOW_DIR.glob("daily_global_workflow_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_file"] = str(path.relative_to(PROJECT_ROOT))
            return data
        except Exception:
            continue
    return None


def _edge_lookup() -> dict[str, bool]:
    if not EDGE_FILE.exists():
        return {}
    try:
        payload = json.loads(EDGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("results") if isinstance(payload, dict) else payload
    lookup: dict[str, bool] = {}
    if not isinstance(rows, list):
        return lookup
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).replace(".JK", "").upper().strip()
        if not ticker:
            continue
        flags = [
            row.get("has_genuine_edge"),
            row.get("has_genuine_edge_h1"),
            row.get("has_genuine_edge_h3"),
            row.get("has_genuine_edge_h5"),
            row.get("has_genuine_edge_h10"),
        ]
        lookup[ticker] = any(flag is True or str(flag).lower() == "true" for flag in flags)
    return lookup


def _accuracy_summary() -> pd.DataFrame:
    acc = _read_csv(ACCURACY_FILE)
    if acc.empty:
        return pd.DataFrame()
    for col in ["ticker", "model_name", "prediction_purpose", "direction_correct"]:
        if col not in acc.columns:
            return pd.DataFrame()
    acc = acc.copy()
    acc["ticker"] = acc["ticker"].astype(str).str.upper().str.strip()
    acc["model_name"] = acc["model_name"].astype(str).str.strip()
    acc["prediction_purpose"] = acc["prediction_purpose"].fillna("").astype(str).str.upper().str.strip()
    if "prediction_run_type" in acc.columns:
        acc = acc[acc["prediction_run_type"].fillna("FINAL").astype(str).str.upper().str.strip() == "FINAL"]
    acc["direction_correct"] = _to_bool(acc["direction_correct"])
    grouped = acc.groupby(["ticker", "model_name", "prediction_purpose"], dropna=False).agg(
        total_evaluations=("direction_correct", "size"),
        accuracy_pct=("direction_correct", lambda s: round(float(s.mean() * 100), 2)),
    ).reset_index()
    return grouped


def _latest_predictions() -> pd.DataFrame:
    pred = _read_csv(PREDICTIONS_FILE)
    if pred.empty:
        return pd.DataFrame()
    required = {"ticker", "model_name", "prediction_purpose", "timestamp_prediction"}
    if not required.issubset(pred.columns):
        return pd.DataFrame()
    pred = pred.copy()
    pred["ticker"] = pred["ticker"].astype(str).str.upper().str.strip()
    pred["model_name"] = pred["model_name"].astype(str).str.strip()
    pred["prediction_purpose"] = pred["prediction_purpose"].fillna("").astype(str).str.upper().str.strip()
    pred["prediction_run_type"] = pred.get("prediction_run_type", "FINAL")
    pred["prediction_run_type"] = pred["prediction_run_type"].fillna("FINAL").astype(str).str.upper().str.strip()
    pred["is_active"] = _to_bool(pred.get("is_active", pd.Series([True] * len(pred))), default=True)
    pred = pred[(pred["is_active"]) & (pred["prediction_run_type"] == "FINAL")]
    pred["timestamp_prediction"] = pd.to_datetime(pred["timestamp_prediction"], errors="coerce")
    pred = pred.sort_values("timestamp_prediction", ascending=False)
    subset = ["ticker", "model_name", "prediction_purpose"]
    if "horizon_days" in pred.columns:
        subset.append("horizon_days")
    return pred.drop_duplicates(subset=subset, keep="first")


def build_status() -> dict[str, Any]:
    tickers = _load_tickers()
    pred = _read_csv(PREDICTIONS_FILE)
    acc = _read_csv(ACCURACY_FILE)
    workflow = _latest_workflow()
    raw_count = 0
    latest_data_date = "-"
    dates: list[str] = []
    for ticker in tickers:
        path = DATA_DIR / "raw" / f"{ticker}_raw.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        raw_count += 1
        try:
            sample = pd.read_csv(path, usecols=lambda c: str(c).lower() in {"tanggal", "timestamp", "date"})
            if not sample.empty:
                value = str(sample.iloc[-1, 0])
                if value and value != "nan":
                    dates.append(value[:10])
        except Exception:
            pass
    if dates:
        latest_data_date = max(dates)

    latest_prediction_date = "-"
    pending_count = 0
    active_count = 0
    if not pred.empty:
        if "current_date" in pred.columns:
            values = pd.to_datetime(pred["current_date"], errors="coerce").dropna()
            if not values.empty:
                latest_prediction_date = values.max().date().isoformat()
        if "status" in pred.columns:
            pending_count = int((pred["status"].astype(str).str.upper().str.strip() == "PENDING").sum())
        if "is_active" in pred.columns:
            active_count = int(_to_bool(pred["is_active"], default=True).sum())

    return {
        "ticker_count": len(tickers),
        "raw_count": raw_count,
        "latest_data_date": latest_data_date,
        "latest_prediction_date": latest_prediction_date,
        "prediction_rows": int(len(pred)),
        "accuracy_rows": int(len(acc)),
        "pending_count": pending_count,
        "active_prediction_count": active_count,
        "latest_workflow": workflow,
    }


def build_recommendations(limit: int = 30) -> list[dict[str, Any]]:
    latest = _latest_predictions()
    if latest.empty:
        return []

    h3 = latest[
        (latest["prediction_purpose"] == "THREE_DAY_FORECAST")
        & (latest["model_name"].astype(str).str.upper().str.contains("GLOBAL-PRICE|XGBOOST|LIGHTGBM", regex=True))
    ].copy()
    h1 = latest[
        (latest["prediction_purpose"] == "NEXT_DAY_DIRECTION")
        & (latest["model_name"].astype(str).str.upper().str.contains("DIRECTION", regex=False))
    ].copy()
    if h3.empty:
        return []

    for df in [h3, h1]:
        for col in ["current_price", "predicted_price", "predicted_return_pct", "confidence_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    h3 = h3.sort_values("timestamp_prediction", ascending=False).drop_duplicates("ticker", keep="first")
    h1 = h1.sort_values("timestamp_prediction", ascending=False).drop_duplicates("ticker", keep="first")
    merged = h3.merge(
        h1[["ticker", "predicted_direction", "confidence_pct"]].rename(
            columns={"predicted_direction": "direction_h1", "confidence_pct": "confidence_h1"}
        ),
        on="ticker",
        how="left",
    )

    acc = _accuracy_summary()
    acc_h3 = acc[acc["prediction_purpose"] == "THREE_DAY_FORECAST"].copy() if not acc.empty else pd.DataFrame()
    if not acc_h3.empty:
        acc_h3 = acc_h3.sort_values(["ticker", "total_evaluations", "accuracy_pct"], ascending=[True, False, False])
        acc_h3 = acc_h3.drop_duplicates("ticker", keep="first")
        merged = merged.merge(acc_h3[["ticker", "total_evaluations", "accuracy_pct"]], on="ticker", how="left")
    else:
        merged["total_evaluations"] = 0
        merged["accuracy_pct"] = 0.0

    edges = _edge_lookup()
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        ticker = str(row["ticker"])
        projected = _num(row.get("predicted_return_pct"))
        confidence = _num(row.get("confidence_h1"), _num(row.get("confidence_pct")))
        direction = str(row.get("direction_h1") or row.get("predicted_direction") or "-").upper()
        accuracy = _num(row.get("accuracy_pct"))
        evaluations = int(_num(row.get("total_evaluations")))
        has_edge = edges.get(ticker)
        signal = "WATCH"
        if has_edge is False:
            signal = "NO_EDGE"
        elif has_edge is True and projected >= 1.0 and direction == "NAIK" and confidence >= 60 and accuracy >= 55 and evaluations >= 10:
            signal = "BUY"
        elif projected <= -1.0 or direction == "TURUN":
            signal = "AVOID"

        rows.append({
            "ticker": ticker,
            "signal": signal,
            "current_price": round(_num(row.get("current_price")), 2),
            "target_price_h3": round(_num(row.get("predicted_price")), 2),
            "projected_return_pct": round(projected, 2),
            "direction_h1": direction,
            "confidence_pct": round(confidence, 2),
            "accuracy_pct": round(accuracy, 2),
            "evaluations": evaluations,
            "has_genuine_edge": has_edge,
            "prediction_date": str(row.get("current_date", ""))[:10],
            "target_date": str(row.get("target_date", ""))[:10],
        })

    order = {"BUY": 0, "WATCH": 1, "AVOID": 2, "NO_EDGE": 3}
    rows.sort(key=lambda item: (order.get(item["signal"], 9), -item["projected_return_pct"], -item["confidence_pct"]))
    return rows[: max(1, int(limit))]


def build_cockpit_payload(limit: int = 30) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": build_status(),
        "recommendations": build_recommendations(limit=limit),
    }

