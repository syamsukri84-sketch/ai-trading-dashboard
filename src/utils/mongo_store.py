from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATABASE = "ai_trading"


@dataclass(frozen=True)
class MongoConfig:
    uri: str
    database: str = DEFAULT_DATABASE

    @property
    def enabled(self) -> bool:
        return bool(self.uri.strip())


def get_mongo_config() -> MongoConfig:
    uri = os.getenv("MONGODB_URI", "").strip()
    placeholder_tokens = ["USER:PASSWORD", "CLUSTER.mongodb.net", "<password>"]
    if any(token in uri for token in placeholder_tokens):
        uri = ""
    return MongoConfig(
        uri=uri,
        database=os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE).strip() or DEFAULT_DATABASE,
    )


def get_mongo_client(config: MongoConfig | None = None):
    config = config or get_mongo_config()
    if not config.enabled:
        return None
    from pymongo import MongoClient

    return MongoClient(config.uri, serverSelectionTimeoutMS=5000)


def check_mongo_status(config: MongoConfig | None = None) -> dict[str, Any]:
    config = config or get_mongo_config()
    if not config.enabled:
        return {
            "enabled": False,
            "ok": False,
            "database": config.database,
            "message": "MONGODB_URI belum diset.",
        }
    client = None
    try:
        client = get_mongo_client(config)
        client.admin.command("ping")
        return {
            "enabled": True,
            "ok": True,
            "database": config.database,
            "message": "MongoDB Atlas terhubung.",
        }
    except Exception as exc:
        return {
            "enabled": True,
            "ok": False,
            "database": config.database,
            "message": str(exc),
        }
    finally:
        if client is not None:
            client.close()


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    return [{key: _clean_value(value) for key, value in row.items()} for row in records]


def upsert_dataframe(
    collection_name: str,
    df: pd.DataFrame,
    unique_keys: list[str],
    config: MongoConfig | None = None,
) -> dict[str, int]:
    if df.empty:
        return {"matched": 0, "modified": 0, "upserted": 0, "processed": 0}
    missing_keys = [key for key in unique_keys if key not in df.columns]
    if missing_keys:
        raise ValueError(f"Kolom unique key tidak tersedia: {', '.join(missing_keys)}")

    config = config or get_mongo_config()
    if not config.enabled:
        raise RuntimeError("MONGODB_URI belum diset.")

    client = get_mongo_client(config)
    try:
        collection = client[config.database][collection_name]
        matched = modified = upserted = 0
        for record in dataframe_to_records(df):
            filter_doc = {key: record.get(key) for key in unique_keys}
            result = collection.update_one(filter_doc, {"$set": record}, upsert=True)
            matched += int(result.matched_count)
            modified += int(result.modified_count)
            upserted += 1 if result.upserted_id is not None else 0
        return {
            "matched": matched,
            "modified": modified,
            "upserted": upserted,
            "processed": int(len(df)),
        }
    finally:
        client.close()


def upload_json_files(
    collection_name: str,
    paths: list[Path],
    unique_key: str = "source_file",
    config: MongoConfig | None = None,
) -> dict[str, int]:
    rows = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload[unique_key] = path.name
        rows.append(payload)
    if not rows:
        return {"matched": 0, "modified": 0, "upserted": 0, "processed": 0}
    return upsert_dataframe(collection_name, pd.DataFrame(rows), [unique_key], config=config)


def fetch_collection_dataframe(collection_name: str, config: MongoConfig | None = None, limit: int = 0) -> pd.DataFrame:
    config = config or get_mongo_config()
    if not config.enabled:
        return pd.DataFrame()
    client = get_mongo_client(config)
    try:
        cursor = client[config.database][collection_name].find({}, {"_id": 0})
        if limit and limit > 0:
            cursor = cursor.limit(int(limit))
        return pd.DataFrame(list(cursor))
    finally:
        client.close()
