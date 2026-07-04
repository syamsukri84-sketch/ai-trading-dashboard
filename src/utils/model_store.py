import json
import os
from datetime import datetime
from typing import Any

import joblib


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")
MODEL_REGISTRY_FILE = os.path.join(MODEL_DIR, "model_registry.json")


def normalize_ticker(ticker: str) -> str:
    return str(ticker).replace(".JK", "").upper().strip()


def model_key(model_name: str, horizon_days: int, prediction_purpose: str) -> str:
    clean_name = str(model_name).replace("/", "_").replace(" ", "_")
    clean_purpose = str(prediction_purpose).upper().strip()
    return f"{clean_name}_H{int(horizon_days)}_{clean_purpose}"


def load_model_registry(path: str = MODEL_REGISTRY_FILE) -> dict:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"models": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"models": {}}
    if not isinstance(data, dict):
        return {"models": {}}
    if "models" not in data or not isinstance(data["models"], dict):
        data["models"] = {}
    return data


def save_model_registry(registry: dict, path: str = MODEL_REGISTRY_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def save_model_artifact(
    ticker: str,
    model_name: str,
    horizon_days: int,
    prediction_purpose: str,
    model_object: Any,
    trained_until_date: str,
    training_run_id: str | None = None,
    run_type: str = "FINAL",
) -> dict:
    clean_ticker = normalize_ticker(ticker)
    key = model_key(model_name, horizon_days, prediction_purpose)
    trained_until = str(trained_until_date)
    version = f"{clean_ticker}_{key}_{trained_until.replace('-', '')}"
    ticker_dir = os.path.join(MODEL_DIR, clean_ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    artifact_path = os.path.join(ticker_dir, f"{key}.joblib")
    joblib.dump(model_object, artifact_path)

    record = {
        "ticker": clean_ticker,
        "model_name": str(model_name),
        "horizon_days": int(horizon_days),
        "prediction_purpose": str(prediction_purpose).upper().strip(),
        "trained_until_date": trained_until,
        "training_run_id": training_run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_version": version,
        "path": artifact_path,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_type": str(run_type or "FINAL").upper().strip(),
    }

    registry = load_model_registry()
    registry.setdefault("models", {}).setdefault(clean_ticker, {})[key] = record
    registry["last_updated_at"] = record["saved_at"]
    save_model_registry(registry)
    return record


def list_ticker_models(ticker: str, registry_path: str = MODEL_REGISTRY_FILE) -> list[dict]:
    registry = load_model_registry(registry_path)
    clean_ticker = normalize_ticker(ticker)
    models = registry.get("models", {}).get(clean_ticker, {})
    return list(models.values())


def load_model_artifact(record: dict) -> Any:
    path = record.get("path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Artifact model tidak ditemukan: {path}")
    return joblib.load(path)


def model_store_status(tickers: list[str] | None = None) -> dict:
    registry = load_model_registry()
    models_by_ticker = registry.get("models", {})
    selected = [normalize_ticker(t) for t in (tickers or models_by_ticker.keys()) if normalize_ticker(t)]
    rows = []
    for ticker in selected:
        ticker_models = list(models_by_ticker.get(ticker, {}).values())
        latest_trained = "-"
        if ticker_models:
            dates = sorted({str(row.get("trained_until_date", "")) for row in ticker_models if row.get("trained_until_date")})
            latest_trained = dates[-1] if dates else "-"
        rows.append({
            "ticker": ticker,
            "model_count": len(ticker_models),
            "latest_trained_until": latest_trained,
            "available_models": ", ".join(sorted({str(row.get("model_name")) for row in ticker_models})),
        })
    return {
        "total_tickers_with_models": len(models_by_ticker),
        "total_artifacts": sum(len(models) for models in models_by_ticker.values()),
        "last_updated_at": registry.get("last_updated_at", "-"),
        "rows": rows,
    }
