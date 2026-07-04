from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from data_loader import DataLoader
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.utils.accuracy_tracker import log_prediction
from src.utils.model_store import list_ticker_models, load_model_artifact, normalize_ticker, save_model_artifact

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None


GLOBAL_TICKER = "GLOBAL"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    base_cols = ["open", "high", "low", "close", "volume"]
    return [c for c in df.columns if (c.startswith("feat_") or c in base_cols) and c != "feat_ticker_id"]


def _purpose_for_horizon(horizon_days: int) -> str:
    if int(horizon_days) == 1:
        return "NEXT_DAY_DIRECTION"
    if int(horizon_days) == 3:
        return "THREE_DAY_FORECAST"
    return f"H{int(horizon_days)}_TREND_FORECAST"


def _target_date(current_ts: pd.Timestamp, horizon_days: int) -> str:
    return (current_ts + timedelta(days=int(horizon_days))).strftime("%Y-%m-%d")


@dataclass
class GlobalPriceModel:
    horizon_days: int = 3
    model_type: str = "xgboost"
    model: Any = None
    feature_names_: list[str] = field(default_factory=list)
    ticker_categories_: list[str] = field(default_factory=list)

    def _build_model(self):
        model_type = str(self.model_type).lower().strip()
        if model_type == "lightgbm" and LGBMRegressor is not None:
            return LGBMRegressor(
                n_estimators=260,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=-1,
            )
        if xgb is None:
            raise ImportError("xgboost belum terinstal.")
        return xgb.XGBRegressor(
            n_estimators=260,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
        )

    def _encode(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        out = df.copy()
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        if is_training:
            self.ticker_categories_ = sorted(out["ticker"].dropna().unique().tolist())
        cat = pd.Categorical(out["ticker"], categories=self.ticker_categories_)
        out["feat_ticker_id"] = pd.Series(cat.codes, index=out.index).replace(-1, 0).astype(float)
        X = out[_feature_columns(out) + ["feat_ticker_id"]].apply(pd.to_numeric, errors="coerce")
        return X.replace([np.inf, -np.inf], 0).fillna(0)

    def train(self, panel_df: pd.DataFrame):
        data = panel_df.copy()
        target = (data.groupby("ticker")["close"].shift(-int(self.horizon_days)) / data["close"]) - 1.0
        valid_idx = target.dropna().index
        X = self._encode(data.loc[valid_idx], is_training=True)
        y = target.loc[valid_idx].astype(float)
        if X.empty:
            raise ValueError("Dataset global kosong setelah target dibuat.")
        self.model = self._build_model()
        self.model.fit(X, y)
        self.feature_names_ = X.columns.tolist()
        return self

    def predict(self, features_df: pd.DataFrame) -> dict:
        if self.model is None:
            raise ValueError("Model global belum dilatih.")
        X = self._encode(features_df.copy(), is_training=False)
        latest = X.iloc[[-1]].reindex(columns=self.feature_names_, fill_value=0)
        current_price = float(features_df["close"].iloc[-1])
        projected_return = float(self.model.predict(latest)[0])
        return {
            "projected_return_pct": projected_return * 100.0,
            "projected_price": current_price * (1.0 + projected_return),
        }


@dataclass
class GlobalDirectionModel:
    horizon_days: int = 1
    model_type: str = "lightgbm"
    min_return_threshold: float = 0.0
    model: Any = None
    feature_names_: list[str] = field(default_factory=list)
    ticker_categories_: list[str] = field(default_factory=list)

    def _build_model(self):
        model_type = str(self.model_type).lower().strip()
        if model_type == "lightgbm" and LGBMClassifier is not None:
            return LGBMClassifier(
                n_estimators=260,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=-1,
            )
        if xgb is None:
            raise ImportError("xgboost belum terinstal.")
        return xgb.XGBClassifier(
            n_estimators=260,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )

    def _encode(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        out = df.copy()
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        if is_training:
            self.ticker_categories_ = sorted(out["ticker"].dropna().unique().tolist())
        cat = pd.Categorical(out["ticker"], categories=self.ticker_categories_)
        out["feat_ticker_id"] = pd.Series(cat.codes, index=out.index).replace(-1, 0).astype(float)
        X = out[_feature_columns(out) + ["feat_ticker_id"]].apply(pd.to_numeric, errors="coerce")
        return X.replace([np.inf, -np.inf], 0).fillna(0)

    def train(self, panel_df: pd.DataFrame):
        data = panel_df.copy()
        future_return = (data.groupby("ticker")["close"].shift(-int(self.horizon_days)) / data["close"]) - 1.0
        valid_idx = future_return.dropna().index
        y = (future_return.loc[valid_idx] > float(self.min_return_threshold)).astype(int)
        if y.nunique() < 2:
            raise ValueError("Dataset global hanya memiliki satu kelas arah.")
        X = self._encode(data.loc[valid_idx], is_training=True)
        self.model = self._build_model()
        self.model.fit(X, y)
        self.feature_names_ = X.columns.tolist()
        return self

    def predict(self, features_df: pd.DataFrame) -> dict:
        if self.model is None:
            raise ValueError("Model global belum dilatih.")
        X = self._encode(features_df.copy(), is_training=False)
        latest = X.iloc[[-1]].reindex(columns=self.feature_names_, fill_value=0)
        proba = self.model.predict_proba(latest)[0]
        classes = list(getattr(self.model, "classes_", [0, 1]))
        prob_up = float(proba[classes.index(1)]) if 1 in classes else 0.5
        prob_down = 1.0 - prob_up
        direction = "NAIK" if prob_up >= 0.5 else "TURUN"
        return {
            "model_name": f"GLOBAL-{str(self.model_type).upper()}",
            "horizon_days": int(self.horizon_days),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "direction": direction,
            "confidence_pct": max(prob_up, prob_down) * 100.0,
        }


def build_global_panel_dataset(tickers: list[str], min_rows: int = 252) -> tuple[pd.DataFrame, dict]:
    loader = DataLoader(min_rows=min_rows)
    engineer = FeatureEngineer(warmup_period=60, drop_correlated=False)
    idx_df = loader.load_data("^JKSE")
    frames = []
    summary = {"loaded": [], "failed": []}
    for ticker in [normalize_ticker(t) for t in tickers if normalize_ticker(t)]:
        try:
            df = loader.load_data(ticker)
            if df is None or df.empty:
                raise ValueError("Data lokal kosong/tidak tersedia.")
            features = engineer.generate_features(df, idx_df=idx_df if idx_df is not None else df)
            if features.empty:
                raise ValueError("Fitur teknikal kosong.")
            features = features.copy()
            features["ticker"] = ticker
            frames.append(features)
            summary["loaded"].append({"ticker": ticker, "rows": len(features)})
        except Exception as e:
            summary["failed"].append({"ticker": ticker, "reason": str(e)})
    if not frames:
        raise ValueError("Tidak ada data ticker yang berhasil dibuat menjadi dataset global.")
    panel = pd.concat(frames, ignore_index=True).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return panel, summary


def train_global_models(tickers: list[str], horizons: list[int] | None = None, run_type: str = "FINAL") -> dict:
    horizons = horizons or [1, 3, 5, 10]
    panel_df, data_summary = build_global_panel_dataset(tickers)
    trained_until = pd.to_datetime(panel_df["timestamp"], errors="coerce").max().strftime("%Y-%m-%d")
    summary = {"data": data_summary, "trained": [], "failed": []}

    for model_type in ["lightgbm", "xgboost"]:
        try:
            model = GlobalDirectionModel(horizon_days=1, model_type=model_type).train(panel_df)
            model_name = f"Global-Direction-{model_type.upper()}"
            record = save_model_artifact(
                GLOBAL_TICKER,
                model_name,
                1,
                "NEXT_DAY_DIRECTION",
                model,
                trained_until,
                run_type=run_type,
            )
            summary["trained"].append(record)
        except Exception as e:
            summary["failed"].append({"model_name": f"Global-Direction-{model_type.upper()}", "reason": str(e)})

    for horizon in [h for h in horizons if int(h) in {3, 5, 10}]:
        purpose = _purpose_for_horizon(int(horizon))
        for model_type in ["xgboost", "lightgbm"]:
            try:
                model = GlobalPriceModel(horizon_days=int(horizon), model_type=model_type).train(panel_df)
                model_name = f"Global-Price-{model_type.upper()}"
                record = save_model_artifact(
                    GLOBAL_TICKER,
                    model_name,
                    int(horizon),
                    purpose,
                    model,
                    trained_until,
                    run_type=run_type,
                )
                summary["trained"].append(record)
            except Exception as e:
                summary["failed"].append({"model_name": f"Global-Price-{model_type.upper()}-H{horizon}", "reason": str(e)})
    return summary


def predict_with_global_models(
    tickers: list[str],
    duplicate_policy: str = "skip",
    prediction_run_type: str = "FINAL",
    progress_callback=None,
) -> dict:
    loader = DataLoader(min_rows=252)
    engineer = FeatureEngineer(warmup_period=60, drop_correlated=False)
    idx_df = loader.load_data("^JKSE")
    model_records = list_ticker_models(GLOBAL_TICKER)
    summary = {"predicted": [], "skipped": [], "failed": []}
    if not model_records:
        return {"predicted": [], "skipped": [{"ticker": GLOBAL_TICKER, "reason": "Model global belum tersedia."}], "failed": []}

    selected_tickers = [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
    for index, ticker in enumerate(selected_tickers, start=1):
        if progress_callback:
            progress_callback({"stage": "global_prediction_started", "ticker": ticker, "completed": index - 1, "total": len(selected_tickers)})
        try:
            df = loader.load_data(ticker)
            if df is None or df.empty:
                raise ValueError("Data lokal kosong/tidak tersedia.")
            features = engineer.generate_features(df, idx_df=idx_df if idx_df is not None else df)
            if features.empty:
                raise ValueError("Fitur teknikal kosong.")
            features = features.copy()
            features["ticker"] = ticker
            latest_ts = pd.to_datetime(df["timestamp"].iloc[-1])
            current_date = latest_ts.strftime("%Y-%m-%d")
            current_price = float(df["close"].iloc[-1])
            used = 0
            for record in model_records:
                try:
                    model = load_model_artifact(record)
                    horizon_days = int(record.get("horizon_days") or 1)
                    purpose = str(record.get("prediction_purpose") or _purpose_for_horizon(horizon_days)).upper().strip()
                    model_name = str(record.get("model_name") or "Global-UNKNOWN")
                    prediction = model.predict(features)
                    target_date = _target_date(latest_ts, horizon_days)
                    predicted_direction = None
                    prob_up = prob_down = confidence_pct = None
                    if purpose == "NEXT_DAY_DIRECTION" and "direction" in prediction:
                        predicted_direction = prediction["direction"]
                        prob_up = prediction.get("prob_up")
                        prob_down = prediction.get("prob_down")
                        confidence_pct = prediction.get("confidence_pct")
                        predicted_price = current_price * (1.01 if str(predicted_direction).upper() == "NAIK" else 0.99)
                    else:
                        predicted_price = prediction.get("projected_price")
                    log_prediction(
                        ticker,
                        model_name,
                        current_date,
                        target_date,
                        float(predicted_price),
                        current_price,
                        horizon_days=horizon_days,
                        prediction_purpose=purpose,
                        predicted_direction=predicted_direction,
                        prob_up=prob_up,
                        prob_down=prob_down,
                        confidence_pct=confidence_pct,
                        duplicate_policy=duplicate_policy,
                        prediction_run_type=prediction_run_type,
                        model_version=record.get("model_version"),
                        training_run_id=record.get("training_run_id"),
                        trained_until_date=record.get("trained_until_date"),
                        prediction_mode="GLOBAL_MODEL",
                    )
                    used += 1
                except Exception as e:
                    summary["failed"].append({"ticker": ticker, "model_name": record.get("model_name", "-"), "reason": str(e)})
            if used:
                summary["predicted"].append({"ticker": ticker, "current_date": current_date, "models_used": used})
            else:
                summary["skipped"].append({"ticker": ticker, "reason": "Tidak ada model global yang berhasil dipakai."})
        except Exception as e:
            summary["failed"].append({"ticker": ticker, "reason": str(e)})
    return summary
