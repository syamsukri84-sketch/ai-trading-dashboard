from datetime import timedelta

import pandas as pd

from data_loader import DataLoader
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.utils.accuracy_tracker import log_prediction
from src.utils.model_store import list_ticker_models, load_model_artifact, normalize_ticker


def _target_date(current_ts: pd.Timestamp, horizon_days: int) -> str:
    return (current_ts + timedelta(days=int(horizon_days))).strftime("%Y-%m-%d")


def run_daily_prediction_from_saved_models(
    tickers: list[str],
    duplicate_policy: str = "skip",
    prediction_run_type: str = "FINAL",
    progress_callback=None,
) -> dict:
    selected_tickers = [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
    loader = DataLoader(min_rows=252)
    engineer = FeatureEngineer(warmup_period=60)
    summary = {"predicted": [], "skipped": [], "failed": []}

    for index, ticker in enumerate(selected_tickers, start=1):
        if progress_callback:
            progress_callback({
                "stage": "saved_model_prediction_started",
                "ticker": ticker,
                "message": f"Prediksi harian dari model tersimpan: {ticker} ({index}/{len(selected_tickers)})...",
                "total": len(selected_tickers),
                "completed": index - 1,
            })

        try:
            model_records = list_ticker_models(ticker)
            if not model_records:
                summary["skipped"].append({"ticker": ticker, "reason": "Belum ada model tersimpan."})
                continue

            df = loader.load_data(ticker)
            if df is None or df.empty:
                raise ValueError("Data harga lokal tidak dapat dimuat.")
            idx_df = loader.load_data("^JKSE")
            features_df = engineer.generate_features(df, idx_df=idx_df)
            if features_df.empty:
                raise ValueError("Fitur teknikal tidak dapat dibuat.")

            latest_ts = pd.to_datetime(df["timestamp"].iloc[-1])
            current_date = latest_ts.strftime("%Y-%m-%d")
            current_price = float(df["close"].iloc[-1])
            ticker_predicted = 0

            for record in model_records:
                try:
                    model = load_model_artifact(record)
                    horizon_days = int(record.get("horizon_days") or 1)
                    purpose = str(record.get("prediction_purpose") or "MODEL_ACCURACY").upper().strip()
                    model_name = str(record.get("model_name") or "UNKNOWN")
                    target_date = _target_date(latest_ts, horizon_days)

                    prediction = model.predict(features_df)
                    predicted_direction = None
                    prob_up = prob_down = confidence_pct = None
                    if purpose == "NEXT_DAY_DIRECTION" and "direction" in prediction:
                        predicted_direction = prediction.get("direction")
                        prob_up = prediction.get("prob_up")
                        prob_down = prediction.get("prob_down")
                        confidence_pct = prediction.get("confidence_pct")
                        predicted_price = current_price * (1.01 if str(predicted_direction).upper() == "NAIK" else 0.99)
                    else:
                        predicted_price = prediction.get("projected_price")

                    if predicted_price is None:
                        raise ValueError("Model tidak menghasilkan predicted_price.")

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
                        prediction_mode="SAVED_MODEL",
                    )
                    ticker_predicted += 1
                except Exception as model_error:
                    summary["failed"].append({
                        "ticker": ticker,
                        "model_name": record.get("model_name", "-"),
                        "reason": str(model_error),
                    })

            if ticker_predicted:
                summary["predicted"].append({
                    "ticker": ticker,
                    "current_date": current_date,
                    "models_used": ticker_predicted,
                })
            else:
                summary["skipped"].append({"ticker": ticker, "reason": "Tidak ada model tersimpan yang berhasil dipakai."})

            if progress_callback:
                progress_callback({
                    "stage": "saved_model_prediction_done",
                    "ticker": ticker,
                    "message": f"{ticker} selesai diprediksi dari model tersimpan.",
                    "total": len(selected_tickers),
                    "completed": index,
                    "predicted_count": len(summary["predicted"]),
                    "failed_count": len(summary["failed"]),
                    "skipped_count": len(summary["skipped"]),
                })
        except Exception as e:
            summary["failed"].append({"ticker": ticker, "reason": str(e)})

    return summary
