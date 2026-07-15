import sys
import os
from datetime import datetime
from datetime import timedelta

import pandas as pd
import yaml

from data_loader import DataLoader
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.models.garch_model import GARCHModel
from src.models.isolation_forest import IsolationForestModel
from src.models.lstm_projector import LSTMPriceProjector
from src.models.price_projector import PriceProjector
from src.models.direction_classifier import DirectionClassifier
from src.models.baseline_strategies import evaluate_baseline_strategies
from src.models.walk_forward import EDGE_THRESHOLD_PCT, walk_forward_direction_validation, walk_forward_return_validation
from src.explainability.shap_explainer import explain_direction_prediction, explain_return_prediction
from src.trading.reliability_ensemble import get_reliability_weights, weighted_direction_probability
from src.utils.accuracy_tracker import PREDICTIONS_FILE, evaluate_pending_predictions, log_prediction
from src.utils.model_guardrails import assert_no_training_leakage
from src.utils.model_store import list_ticker_models, load_model_artifact, save_model_artifact
from src.utils.training_policy import record_training_run


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _load_config_tickers(stock_list_path: str) -> list[str]:
    with open(stock_list_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config.get("tickers", [])


def _normalize_tickers(tickers: list[str]) -> list[str]:
    return [str(ticker).replace(".JK", "").upper().strip() for ticker in tickers if str(ticker).strip()]


def _prediction_exists(
    pred_df,
    ticker: str,
    model_name: str,
    current_date: str,
    horizon_days: int,
    prediction_purpose: str,
) -> bool:
    if pred_df.empty:
        return False

    required_columns = {"ticker", "model_name", "current_date", "horizon_days", "prediction_purpose"}
    if not required_columns.issubset(set(pred_df.columns)):
        return False

    active_mask = True
    if "is_active" in pred_df.columns:
        active_mask = pred_df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])

    return bool(
        (
            (pred_df["ticker"].astype(str).str.upper().str.strip() == ticker)
            & (pred_df["model_name"].astype(str).str.strip() == model_name)
            & (pred_df["current_date"].astype(str) == str(current_date))
            & (pd.to_numeric(pred_df["horizon_days"], errors="coerce").fillna(-1).astype(int) == int(horizon_days))
            & (pred_df["prediction_purpose"].astype(str).str.upper().str.strip() == prediction_purpose)
            & active_mask
        ).any()
    )


def _has_completed_latest_analysis(
    pred_df,
    ticker: str,
    current_date: str,
    required_models: list[str] | tuple[str, ...],
) -> bool:
    has_required_h3 = all(
        _prediction_exists(
            pred_df=pred_df,
            ticker=ticker,
            model_name=model_name,
            current_date=current_date,
            horizon_days=3,
            prediction_purpose="THREE_DAY_FORECAST",
        )
        for model_name in required_models
    )
    has_next_day_core = _prediction_exists(
        pred_df=pred_df,
        ticker=ticker,
        model_name="XGBoost",
        current_date=current_date,
        horizon_days=1,
        prediction_purpose="NEXT_DAY_DIRECTION",
    )
    return bool(has_required_h3 and has_next_day_core)


def _load_existing_predictions():
    if not os.path.exists(PREDICTIONS_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(PREDICTIONS_FILE)
    except Exception as e:
        print(f"Prediksi lama tidak bisa dibaca untuk pengecekan skip: {e}")
        return pd.DataFrame()


def _available_prediction_dates(df: pd.DataFrame, min_rows: int = 252) -> list[pd.Timestamp]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return []
    dated = df.copy()
    dated["timestamp"] = pd.to_datetime(dated["timestamp"], errors="coerce")
    dated = dated.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(dated) < min_rows:
        return []
    return dated.iloc[min_rows - 1 :]["timestamp"].dt.normalize().drop_duplicates().tolist()


def _has_prediction_for_date(pred_df: pd.DataFrame, ticker: str, current_date: str, model_name: str = "XGBoost") -> bool:
    return _prediction_exists(
        pred_df=pred_df,
        ticker=ticker,
        model_name=model_name,
        current_date=current_date,
        horizon_days=1,
        prediction_purpose="NEXT_DAY_DIRECTION",
    )


def _load_cached_lstm_if_trained_today(ticker: str, horizon_days: int, prediction_purpose: str, current_date_str: str):
    """Kalau sudah ada artifact LSTM tersimpan untuk ticker+horizon+purpose ini
    dengan trained_until_date == current_date_str, muat dan pakai ulang alih-alih
    melatih ulang dari nol. LSTM (PyTorch) jauh lebih mahal dilatih dibanding
    model lain di pipeline ini -- kalau data belum berubah sejak run terakhir
    hari ini (tanggal sama), melatih ulang cuma buang waktu tanpa hasil beda.
    Return (model, record) kalau ketemu & berhasil dimuat, selalu (None, None)
    kalau tidak ada atau gagal dimuat (fallback aman: caller akan melatih baru)."""
    try:
        records = list_ticker_models(ticker)
    except Exception:
        return None, None
    for record in records:
        if (
            record.get("model_name") == "LSTM"
            and int(record.get("horizon_days", -1)) == int(horizon_days)
            and str(record.get("prediction_purpose", "")).upper().strip() == str(prediction_purpose).upper().strip()
            and str(record.get("trained_until_date", "")) == str(current_date_str)
        ):
            try:
                model_obj = load_model_artifact(record)
            except Exception:
                return None, None
            if model_obj is not None:
                return model_obj, record
    return None, None


def run_backfill_analysis(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    max_days_per_ticker: int = 5,
    lstm_epochs: int = 1,
    include_lstm: bool = False,
    progress_callback=None,
) -> dict:
    """
    Replay prediksi untuk tanggal historis tanpa leakage.

    Untuk setiap tanggal backfill, data harga dipotong sampai tanggal tersebut
    sebelum feature engineering dan training. Hasil dicatat sebagai BACKFILL agar
    bisa dibedakan dari prediksi real-time harian.
    """
    tickers = _normalize_tickers(tickers)
    loader = DataLoader(min_rows=252)
    engineer = FeatureEngineer(warmup_period=60)
    existing_predictions_df = _load_existing_predictions()
    summary = {"analyzed": [], "failed": [], "skipped": []}

    start_ts = pd.to_datetime(start_date, errors="coerce").normalize() if start_date else None
    end_ts = pd.to_datetime(end_date, errors="coerce").normalize() if end_date else None
    max_days_per_ticker = max(int(max_days_per_ticker or 1), 1)

    jobs = []
    loaded_data = {}
    for ticker in tickers:
        df = loader.load_data(ticker)
        if df is None or df.empty:
            summary["failed"].append({"ticker": ticker, "reason": "Data tidak dapat dimuat."})
            continue
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        loaded_data[ticker] = df

        candidate_dates = _available_prediction_dates(df, min_rows=252)
        if start_ts is not None:
            candidate_dates = [date for date in candidate_dates if date >= start_ts]
        if end_ts is not None:
            candidate_dates = [date for date in candidate_dates if date <= end_ts]

        missing_dates = []
        for date in candidate_dates:
            current_date = pd.Timestamp(date).strftime("%Y-%m-%d")
            if _has_prediction_for_date(existing_predictions_df, ticker, current_date, model_name="XGBoost"):
                continue
            missing_dates.append(pd.Timestamp(date))

        for date in missing_dates[-max_days_per_ticker:]:
            jobs.append((ticker, date))

    total_jobs = len(jobs)
    if progress_callback:
        progress_callback({
            "stage": "backfill_started",
            "message": f"Memulai backfill {total_jobs} tanggal prediksi historis.",
            "total": total_jobs,
            "completed": 0,
            "analyzed_count": 0,
            "failed_count": len(summary["failed"]),
            "skipped_count": 0,
        })

    idx_full = loader.load_data("^JKSE")
    if idx_full is not None:
        idx_full = idx_full.copy()
        idx_full["timestamp"] = pd.to_datetime(idx_full["timestamp"], errors="coerce")
        idx_full = idx_full.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for index, (ticker, as_of_date) in enumerate(jobs, start=1):
        current_date_str = as_of_date.strftime("%Y-%m-%d")
        if progress_callback:
            progress_callback({
                "stage": "backfill_date_started",
                "ticker": ticker,
                "message": f"Backfill {ticker} untuk tanggal {current_date_str} ({index}/{total_jobs})...",
                "total": total_jobs,
                "completed": index - 1,
                "analyzed_count": len(summary["analyzed"]),
                "failed_count": len(summary["failed"]),
                "skipped_count": len(summary["skipped"]),
            })

        try:
            full_df = loaded_data[ticker]
            replay_df = full_df[full_df["timestamp"].dt.normalize() <= as_of_date].copy()
            if len(replay_df) < 252:
                raise ValueError(f"Data sampai {current_date_str} kurang dari 252 baris.")

            replay_idx_df = None
            if idx_full is not None and not idx_full.empty:
                replay_idx_df = idx_full[idx_full["timestamp"].dt.normalize() <= as_of_date].copy()
            if replay_idx_df is None or replay_idx_df.empty:
                replay_idx_df = replay_df

            features_df = engineer.generate_features(replay_df, idx_df=replay_idx_df)
            if features_df.empty:
                raise ValueError("Gagal membuat fitur backfill.")

            guardrail = assert_no_training_leakage(
                raw_df=replay_df,
                features_df=features_df,
                ticker=ticker,
                prediction_date=current_date_str,
            )
            if not guardrail.passed:
                raise ValueError("Guardrail backfill gagal: " + " | ".join(guardrail.errors))

            current_price = float(replay_df["close"].iloc[-1])
            next_day_target_date_str = (replay_df["timestamp"].iloc[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
            h3_target_date_str = (replay_df["timestamp"].iloc[-1] + timedelta(days=3)).strftime("%Y-%m-%d")
            prediction_log_kwargs = {
                "duplicate_policy": "skip",
                "prediction_run_type": "BACKFILL",
            }

            next_day_projector = PriceProjector(projection_horizon=1)
            next_day_projector.train(features_df)
            next_day_projection = next_day_projector.predict(features_df)
            log_prediction(
                ticker,
                "XGBoost",
                current_date_str,
                next_day_target_date_str,
                next_day_projection["projected_price"],
                current_price,
                horizon_days=1,
                prediction_purpose="NEXT_DAY_DIRECTION",
                **prediction_log_kwargs,
            )

            h3_projector = PriceProjector(projection_horizon=3)
            h3_projector.train(features_df)
            h3_projection = h3_projector.predict(features_df)
            log_prediction(
                ticker,
                "XGBoost",
                current_date_str,
                h3_target_date_str,
                h3_projection["projected_price"],
                current_price,
                horizon_days=3,
                prediction_purpose="THREE_DAY_FORECAST",
                **prediction_log_kwargs,
            )

            lstm_h1_return = None
            lstm_h3_return = None
            if include_lstm:
                lstm_h3 = LSTMPriceProjector(projection_horizon=3, lookback=20)
                lstm_h3.train(features_df, epochs=lstm_epochs)
                lstm_h3_projection = lstm_h3.predict(features_df)
                lstm_h3_return = lstm_h3_projection["projected_return_pct"]
                log_prediction(
                    ticker,
                    "LSTM",
                    current_date_str,
                    h3_target_date_str,
                    lstm_h3_projection["projected_price"],
                    current_price,
                    horizon_days=3,
                    prediction_purpose="THREE_DAY_FORECAST",
                    **prediction_log_kwargs,
                )

                lstm_h1 = LSTMPriceProjector(projection_horizon=1, lookback=20)
                lstm_h1.train(features_df, epochs=lstm_epochs)
                lstm_h1_projection = lstm_h1.predict(features_df)
                lstm_h1_return = lstm_h1_projection["projected_return_pct"]
                log_prediction(
                    ticker,
                    "LSTM",
                    current_date_str,
                    next_day_target_date_str,
                    lstm_h1_projection["projected_price"],
                    current_price,
                    horizon_days=1,
                    prediction_purpose="NEXT_DAY_DIRECTION",
                    **prediction_log_kwargs,
                )

            summary["analyzed"].append({
                "ticker": ticker,
                "current_date": current_date_str,
                "next_day_target_date": next_day_target_date_str,
                "target_date": h3_target_date_str,
                "xgboost_next_day_return_pct": next_day_projection["projected_return_pct"],
                "xgboost_h3_return_pct": h3_projection["projected_return_pct"],
                "lstm_next_day_return_pct": lstm_h1_return,
                "lstm_h3_return_pct": lstm_h3_return,
                "prediction_run_type": "BACKFILL",
            })
            if progress_callback:
                progress_callback({
                    "stage": "backfill_date_succeeded",
                    "ticker": ticker,
                    "message": f"Backfill {ticker} tanggal {current_date_str} selesai.",
                    "total": total_jobs,
                    "completed": index,
                    "analyzed_count": len(summary["analyzed"]),
                    "failed_count": len(summary["failed"]),
                    "skipped_count": len(summary["skipped"]),
                })
        except Exception as e:
            summary["failed"].append({"ticker": ticker, "current_date": current_date_str, "reason": str(e)})
            if progress_callback:
                progress_callback({
                    "stage": "backfill_date_failed",
                    "ticker": ticker,
                    "message": f"Backfill {ticker} tanggal {current_date_str} gagal: {e}",
                    "reason": str(e),
                    "total": total_jobs,
                    "completed": index,
                    "analyzed_count": len(summary["analyzed"]),
                    "failed_count": len(summary["failed"]),
                    "skipped_count": len(summary["skipped"]),
                })

    if progress_callback:
        progress_callback({
            "stage": "done",
            "message": "Backfill prediksi historis selesai.",
            "total": total_jobs,
            "completed": total_jobs,
            "analyzed_count": len(summary["analyzed"]),
            "failed_count": len(summary["failed"]),
            "skipped_count": len(summary["skipped"]),
        })

    return summary


def run_full_analysis(
    stock_list_path: str = "config/stocks.yaml",
    tickers: list[str] | None = None,
    lstm_epochs: int = 5,
    progress_callback=None,
    duplicate_policy: str | None = None,
    prediction_run_type: str | None = None,
    skip_completed: bool = True,
    completed_required_models: list[str] | tuple[str, ...] | None = None,
    include_lstm: bool = False,
) -> dict:
    """
    Menjalankan pipeline analisis end-to-end untuk emiten terpilih.
    Jika tickers tidak dikirim, daftar emiten dibaca dari config/stocks.yaml.

    include_lstm: default False -- LSTM (PyTorch) adalah model paling mahal
    dilatih di pipeline ini dan belum pernah divalidasi walk-forward (beda
    dari DirectionClassifier/PriceProjector yang sudah). Konsisten dengan
    parameter `include_lstm` yang sudah ada di run_backfill_analysis. Set True
    eksplisit kalau butuh proyeksi LSTM. Lihat audit codebase 2026-07-12.
    """
    if tickers is None:
        try:
            tickers = _load_config_tickers(stock_list_path)
        except FileNotFoundError:
            message = f"File konfigurasi tidak ditemukan di: {stock_list_path}"
            print(message)
            return {"analyzed": [], "failed": [{"ticker": "CONFIG", "reason": message}]}

    tickers = _normalize_tickers(tickers)
    if not tickers:
        message = "Tidak ada emiten yang dipilih untuk dianalisis."
        print(message)
        return {"analyzed": [], "failed": [{"ticker": "EMPTY", "reason": message}]}

    duplicate_policy = duplicate_policy or os.getenv("AI_TRADING_DUPLICATE_POLICY") or "skip"
    duplicate_policy = str(duplicate_policy).lower().strip().replace("-", "_")
    if duplicate_policy not in {"skip", "overwrite", "intraday"}:
        duplicate_policy = "skip"
    prediction_run_type = prediction_run_type or os.getenv("AI_TRADING_PREDICTION_RUN_TYPE")
    if not prediction_run_type:
        prediction_run_type = "INTRADAY" if duplicate_policy == "intraday" else "FINAL"
    prediction_run_type = str(prediction_run_type).upper().strip()
    prediction_log_kwargs = {
        "duplicate_policy": duplicate_policy,
        "prediction_run_type": prediction_run_type,
    }
    training_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    completed_required_models = completed_required_models or ["XGBoost"]
    skip_completed = bool(skip_completed and duplicate_policy == "skip")

    print(f"Memulai analisis untuk {len(tickers)} emiten: {', '.join(tickers)}")
    print(f"Kebijakan duplikasi prediksi: {duplicate_policy.upper()} | Tipe run: {prediction_run_type}")
    if skip_completed:
        print(f"Skip saham yang sudah lengkap untuk data terbaru: ON ({', '.join(completed_required_models)})")
    else:
        print("Skip saham yang sudah lengkap untuk data terbaru: OFF")
    print("=" * 60)

    print("Mengevaluasi akurasi prediksi hari-hari sebelumnya...")
    if progress_callback:
        progress_callback({
            "stage": "evaluating_accuracy",
            "message": "Mengevaluasi akurasi prediksi sebelumnya...",
            "total": len(tickers),
            "completed": 0,
        })
    evaluate_pending_predictions()
    print("Evaluasi selesai.\n")
    if progress_callback:
        progress_callback({
            "stage": "accuracy_done",
            "message": "Evaluasi akurasi selesai. Memulai analisis saham...",
            "total": len(tickers),
            "completed": 0,
        })

    loader = DataLoader(min_rows=252)
    engineer = FeatureEngineer(warmup_period=60)
    existing_predictions_df = _load_existing_predictions() if skip_completed else pd.DataFrame()
    summary = {"analyzed": [], "failed": [], "skipped": []}

    for index, ticker in enumerate(tickers, start=1):
        print(f"\nMenganalisis {ticker}...")
        if progress_callback:
            progress_callback({
                "stage": "ticker_started",
                "ticker": ticker,
                "message": f"Menganalisis {ticker} ({index}/{len(tickers)})...",
                "total": len(tickers),
                "completed": index - 1,
                "analyzed_count": len(summary["analyzed"]),
                "failed_count": len(summary["failed"]),
            })

        try:
            df = loader.load_data(ticker)
            if df is None:
                raise ValueError("Data tidak dapat dimuat.")

            current_date_str = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
            if skip_completed and _has_completed_latest_analysis(
                existing_predictions_df,
                ticker,
                current_date_str,
                completed_required_models,
            ):
                message = f"{ticker} dilewati karena data {current_date_str} sudah pernah dianalisis."
                print(f"   -> {message}")
                summary["skipped"].append({
                    "ticker": ticker,
                    "current_date": current_date_str,
                    "reason": f"Prediksi inti H+1 dan H+3 sudah ada untuk data {current_date_str}.",
                })
                if progress_callback:
                    progress_callback({
                        "stage": "ticker_skipped",
                        "ticker": ticker,
                        "message": message,
                        "reason": f"Prediksi inti H+1 dan H+3 sudah ada untuk data {current_date_str}.",
                        "total": len(tickers),
                        "completed": index,
                        "analyzed_count": len(summary["analyzed"]),
                        "failed_count": len(summary["failed"]),
                        "skipped_count": len(summary["skipped"]),
                    })
                continue

            idx_df = loader.load_data("^JKSE")
            if idx_df is None:
                idx_df = df
            features_df = engineer.generate_features(df, idx_df=idx_df)
            if features_df.empty:
                raise ValueError("Gagal membuat fitur.")

            guardrail = assert_no_training_leakage(
                raw_df=df,
                features_df=features_df,
                ticker=ticker,
                prediction_date=current_date_str,
            )
            if not guardrail.passed:
                raise ValueError("Guardrail data/model gagal: " + " | ".join(guardrail.errors))

            if_model = IsolationForestModel(contamination=0.05)
            if_model.train(features_df)
            results = if_model.predict(features_df)

            projection_horizons = [3, 5, 10]
            projections = {}
            projector_objects = {}
            lightgbm_projections = {}
            artifact_records = {}
            for project_horizon in projection_horizons:
                projector = PriceProjector(projection_horizon=project_horizon)
                projector.train(features_df)
                projector_objects[project_horizon] = projector
                purpose = "THREE_DAY_FORECAST" if project_horizon == 3 else f"H{project_horizon}_TREND_FORECAST"
                artifact_records[("XGBoost", project_horizon, purpose)] = save_model_artifact(
                    ticker,
                    "XGBoost",
                    project_horizon,
                    purpose,
                    projector,
                    current_date_str,
                    training_run_id=training_run_id,
                    run_type=prediction_run_type,
                )
                projections[project_horizon] = projector.predict(features_df)
                try:
                    lgbm_projector = PriceProjector(projection_horizon=project_horizon, model_type="lightgbm")
                    lgbm_projector.train(features_df)
                    artifact_records[("LightGBM", project_horizon, purpose)] = save_model_artifact(
                        ticker,
                        "LightGBM",
                        project_horizon,
                        purpose,
                        lgbm_projector,
                        current_date_str,
                        training_run_id=training_run_id,
                        run_type=prediction_run_type,
                    )
                    lightgbm_projections[project_horizon] = lgbm_projector.predict(features_df)
                except Exception as e:
                    print(f"   [LightGBMRegressor H+{project_horizon}] dilewati: {e}")

            next_day_projector = PriceProjector(projection_horizon=1)
            next_day_projector.train(features_df)
            artifact_records[("XGBoost", 1, "NEXT_DAY_DIRECTION")] = save_model_artifact(
                ticker,
                "XGBoost",
                1,
                "NEXT_DAY_DIRECTION",
                next_day_projector,
                current_date_str,
                training_run_id=training_run_id,
                run_type=prediction_run_type,
            )
            next_day_projection = next_day_projector.predict(features_df)

            classifiers = {}
            classifier_predictions = {}
            for model_type in ["lightgbm", "xgboost", "random_forest", "logistic"]:
                try:
                    clf = DirectionClassifier(horizon_days=1, model_type=model_type)
                    clf.train(features_df)
                    classifiers[model_type.upper()] = clf
                    direction_model_name = f"Direction-{model_type.upper()}"
                    artifact_records[(direction_model_name, 1, "NEXT_DAY_DIRECTION")] = save_model_artifact(
                        ticker,
                        direction_model_name,
                        1,
                        "NEXT_DAY_DIRECTION",
                        clf,
                        current_date_str,
                        training_run_id=training_run_id,
                        run_type=prediction_run_type,
                    )
                    classifier_predictions[model_type.upper()] = clf.predict(features_df)
                except Exception as e:
                    print(f"   [Classifier {model_type}] dilewati: {e}")

            reliability_weights = get_reliability_weights(
                ticker,
                classifier_predictions.keys(),
                prediction_purpose="NEXT_DAY_DIRECTION",
            )
            ensemble_direction = weighted_direction_probability(classifier_predictions, reliability_weights)

            # XAI: jelaskan prediksi arah H+1 lewat kontribusi SHAP per fitur,
            # pakai satu classifier tree yang tersedia sebagai wakil (bukan
            # rata-rata seluruh ensemble -- lihat catatan di shap_explainer.py).
            xai_direction_h1 = {"available": False, "reason": "Tidak ada classifier tree yang berhasil dilatih."}
            for model_key in ("LIGHTGBM", "XGBOOST", "RANDOM_FOREST"):
                explainer_classifier = classifiers.get(model_key)
                if explainer_classifier is None:
                    continue
                try:
                    latest_row = features_df[explainer_classifier.feature_names_].iloc[[-1]]
                    xai_direction_h1 = explain_direction_prediction(
                        explainer_classifier.model, latest_row, explainer_classifier.feature_names_
                    )
                except Exception as e:
                    xai_direction_h1 = {"available": False, "reason": f"Penjelasan SHAP gagal: {e}"}
                break

            project_horizon = 3
            projection = projections[project_horizon]

            # XAI: jelaskan proyeksi return H+3 (model XGBoost utama) lewat
            # kontribusi SHAP per fitur.
            xai_return_h3 = {"available": False, "reason": "Model proyeksi H+3 tidak tersedia."}
            h3_projector_obj = projector_objects.get(3)
            if h3_projector_obj is not None:
                try:
                    latest_row_h3 = features_df[h3_projector_obj.feature_names_].iloc[[-1]]
                    xai_return_h3 = explain_return_prediction(
                        h3_projector_obj.model, latest_row_h3, h3_projector_obj.feature_names_
                    )
                except Exception as e:
                    xai_return_h3 = {"available": False, "reason": f"Penjelasan SHAP gagal: {e}"}
            if include_lstm:
                cached_lstm_h3, cached_record_h3 = _load_cached_lstm_if_trained_today(
                    ticker, project_horizon, "THREE_DAY_FORECAST", current_date_str
                )
                if cached_lstm_h3 is not None:
                    lstm_projector = cached_lstm_h3
                    artifact_records[("LSTM", project_horizon, "THREE_DAY_FORECAST")] = cached_record_h3
                else:
                    lstm_projector = LSTMPriceProjector(projection_horizon=project_horizon, lookback=20)
                    lstm_projector.train(features_df, epochs=lstm_epochs)
                    artifact_records[("LSTM", project_horizon, "THREE_DAY_FORECAST")] = save_model_artifact(
                        ticker,
                        "LSTM",
                        project_horizon,
                        "THREE_DAY_FORECAST",
                        lstm_projector,
                        current_date_str,
                        training_run_id=training_run_id,
                        run_type=prediction_run_type,
                    )
                lstm_projection = lstm_projector.predict(features_df)

                cached_lstm_h1, cached_record_h1 = _load_cached_lstm_if_trained_today(
                    ticker, 1, "NEXT_DAY_DIRECTION", current_date_str
                )
                if cached_lstm_h1 is not None:
                    next_day_lstm_projector = cached_lstm_h1
                    artifact_records[("LSTM", 1, "NEXT_DAY_DIRECTION")] = cached_record_h1
                else:
                    next_day_lstm_projector = LSTMPriceProjector(projection_horizon=1, lookback=20)
                    next_day_lstm_projector.train(features_df, epochs=lstm_epochs)
                    artifact_records[("LSTM", 1, "NEXT_DAY_DIRECTION")] = save_model_artifact(
                        ticker,
                        "LSTM",
                        1,
                        "NEXT_DAY_DIRECTION",
                        next_day_lstm_projector,
                        current_date_str,
                        training_run_id=training_run_id,
                        run_type=prediction_run_type,
                    )
                next_day_lstm_projection = next_day_lstm_projector.predict(features_df)
            else:
                lstm_projection = None
                next_day_lstm_projection = None

            garch_model = GARCHModel(p=1, q=1)
            garch_model.train(df)
            garch_projection = garch_model.predict(horizon=project_horizon)

            aligned_df = df.iloc[engineer.warmup_period:].copy()
            aligned_df["anomaly_score"] = results["anomaly_score"]

            print(f"TOP 3 HARI PALING ANOMALI untuk {ticker}:")
            top_anomalies = aligned_df.sort_values("anomaly_score", ascending=False).head(3)
            print(top_anomalies[["timestamp", "close", "volume", "anomaly_score"]].to_string(index=False))

            current_price = float(df["close"].iloc[-1])
            next_day_target_date_str = (df["timestamp"].iloc[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

            target_date_str = (df["timestamp"].iloc[-1] + timedelta(days=project_horizon)).strftime("%Y-%m-%d")
            def artifact_log_kwargs(model_name, horizon_days, purpose):
                record = artifact_records.get((model_name, horizon_days, purpose), {})
                return {
                    **prediction_log_kwargs,
                    "model_version": record.get("model_version"),
                    "training_run_id": record.get("training_run_id"),
                    "trained_until_date": record.get("trained_until_date"),
                    "prediction_mode": "TRAIN_AND_PREDICT",
                }

            for horizon, horizon_projection in projections.items():
                horizon_target_date = (df["timestamp"].iloc[-1] + timedelta(days=horizon)).strftime("%Y-%m-%d")
                purpose = "THREE_DAY_FORECAST" if horizon == 3 else f"H{horizon}_TREND_FORECAST"
                log_prediction(ticker, "XGBoost", current_date_str, horizon_target_date, horizon_projection["projected_price"], current_price, horizon_days=horizon, prediction_purpose=purpose, **artifact_log_kwargs("XGBoost", horizon, purpose))
                if horizon in lightgbm_projections:
                    log_prediction(ticker, "LightGBM", current_date_str, horizon_target_date, lightgbm_projections[horizon]["projected_price"], current_price, horizon_days=horizon, prediction_purpose=purpose, **artifact_log_kwargs("LightGBM", horizon, purpose))
            if include_lstm:
                log_prediction(ticker, "LSTM", current_date_str, target_date_str, lstm_projection["projected_price"], current_price, horizon_days=project_horizon, prediction_purpose="THREE_DAY_FORECAST", **artifact_log_kwargs("LSTM", project_horizon, "THREE_DAY_FORECAST"))
            log_prediction(ticker, "XGBoost", current_date_str, next_day_target_date_str, next_day_projection["projected_price"], current_price, horizon_days=1, prediction_purpose="NEXT_DAY_DIRECTION", **artifact_log_kwargs("XGBoost", 1, "NEXT_DAY_DIRECTION"))
            if include_lstm:
                log_prediction(ticker, "LSTM", current_date_str, next_day_target_date_str, next_day_lstm_projection["projected_price"], current_price, horizon_days=1, prediction_purpose="NEXT_DAY_DIRECTION", **artifact_log_kwargs("LSTM", 1, "NEXT_DAY_DIRECTION"))
            for model_name, clf_prediction in classifier_predictions.items():
                direction_price = current_price * (1.01 if clf_prediction["direction"] == "NAIK" else 0.99)
                direction_model_name = f"Direction-{model_name}"
                log_prediction(
                    ticker,
                    direction_model_name,
                    current_date_str,
                    next_day_target_date_str,
                    direction_price,
                    current_price,
                    horizon_days=1,
                    prediction_purpose="NEXT_DAY_DIRECTION",
                    predicted_direction=clf_prediction["direction"],
                    prob_up=clf_prediction["prob_up"],
                    prob_down=clf_prediction["prob_down"],
                    confidence_pct=clf_prediction["confidence_pct"],
                    **artifact_log_kwargs(direction_model_name, 1, "NEXT_DAY_DIRECTION"),
                )
            ensemble_price = current_price * (1.01 if ensemble_direction["direction"] == "NAIK" else 0.99)
            log_prediction(
                ticker,
                "Direction-Ensemble",
                current_date_str,
                next_day_target_date_str,
                ensemble_price,
                current_price,
                horizon_days=1,
                prediction_purpose="NEXT_DAY_DIRECTION",
                predicted_direction=ensemble_direction["direction"],
                prob_up=ensemble_direction["prob_up"],
                prob_down=ensemble_direction["prob_down"],
                confidence_pct=ensemble_direction["confidence_pct"],
                **prediction_log_kwargs,
            )

            baseline_h1 = evaluate_baseline_strategies(features_df, horizon_days=1)
            baseline_h3 = evaluate_baseline_strategies(features_df, horizon_days=3)
            # PENTING: factory di sini HARUS membangun model dengan hyperparameter
            # yang SAMA PERSIS dengan DirectionClassifier/PriceProjector yang
            # sungguhan dipakai untuk prediksi live -- sebelumnya factory ini
            # mendefinisikan ulang LightGBM secara terpisah (n_estimators=120,
            # learning_rate=0.05, tanpa regularisasi apa pun), beda dari
            # hyperparameter teregulasi yang divalidasi & diterapkan ke
            # DirectionClassifier._build_model(). Akibatnya flag
            # has_genuine_edge_* bisa mengevaluasi model yang BUKAN model yang
            # sebenarnya dipakai untuk prediksi yang tampil ke pengguna. Pakai
            # instance asli supaya kalau hyperparameter produksi berubah,
            # validasi walk-forward ikut berubah otomatis (satu sumber
            # kebenaran, bukan dua definisi yang bisa diam-diam berbeda).
            #
            # Sebelumnya factory ini juga memakai calibrate=False -- padahal
            # prediksi live (baris ~565 di atas) memakai calibrate=True default
            # (dibungkus CalibratedClassifierCV di DirectionClassifier.train()).
            # build_walk_forward_estimator() mereplikasi wrapping kalibrasi
            # yang SAMA supaya walk-forward menguji objek yang benar-benar
            # identik dengan yang dideploy, bukan estimator mentah pra-kalibrasi.
            # Lihat audit codebase 2026-07-12.
            direction_factory = lambda: DirectionClassifier(horizon_days=1, model_type="lightgbm").build_walk_forward_estimator(n_train_rows=252)
            return_factory = lambda: PriceProjector(projection_horizon=3, model_type="xgboost").model
            wf_direction_h1 = walk_forward_direction_validation(features_df, direction_factory, horizon_days=1)
            wf_return_h3 = walk_forward_return_validation(features_df, return_factory, horizon_days=3)
            wf_return_h5 = walk_forward_return_validation(features_df, return_factory, horizon_days=5)
            wf_return_h10 = walk_forward_return_validation(features_df, return_factory, horizon_days=10)

            has_edge_h1 = wf_direction_h1["edge_vs_baseline_pct"] >= EDGE_THRESHOLD_PCT
            has_edge_h3 = wf_return_h3["edge_vs_mean_mae_pct"] > 0 and wf_return_h3["direction_accuracy_pct"] >= wf_return_h3["baseline_mean_direction_accuracy_pct"] + EDGE_THRESHOLD_PCT
            has_edge_h5 = wf_return_h5["edge_vs_mean_mae_pct"] > 0 and wf_return_h5["direction_accuracy_pct"] >= wf_return_h5["baseline_mean_direction_accuracy_pct"] + EDGE_THRESHOLD_PCT
            has_edge_h10 = wf_return_h10["edge_vs_mean_mae_pct"] > 0 and wf_return_h10["direction_accuracy_pct"] >= wf_return_h10["baseline_mean_direction_accuracy_pct"] + EDGE_THRESHOLD_PCT

            print(f"\nPROYEKSI HARGA {project_horizon} HARI KE DEPAN ({ticker}):")
            print(f"   Harga Terakhir   : Rp {current_price:,.2f}")
            print(f"   [XGBoost] Target : Rp {projection['projected_price']:,.2f} ({projection['projected_return_pct']:+.2f}%)")
            if lstm_projection is not None:
                print(f"   [LSTM] Target    : Rp {lstm_projection['projected_price']:,.2f} ({lstm_projection['projected_return_pct']:+.2f}%)")
            print(f"   [Direction Ensemble H+1] {ensemble_direction['direction']} | Confidence {ensemble_direction['confidence_pct']:.2f}%")
            print(f"   [GARCH] Volatilitas: {garch_projection.get('projected_volatility_pct', 0.0):,.2f}% per hari")
            print(f"   [GARCH] VaR 95%    : {garch_projection.get('value_at_risk_95_pct', 0.0):,.2f}%")
            print(f"   WALK-FORWARD vs BASELINE NAIF (edge asli, bukan akurasi mentah):")
            print(f"     H+1 arah : model={wf_direction_h1['direction_accuracy_pct']:.1f}% vs baseline={wf_direction_h1['baseline_majority_accuracy_pct']:.1f}% "
                  f"(edge={wf_direction_h1['edge_vs_baseline_pct']:+.1f}pp) -> {'ADA EDGE' if has_edge_h1 else 'TIDAK ADA EDGE NYATA'}")
            print(f"     H+3 return: MAE model={wf_return_h3['mae_pct']:.2f}% vs baseline-mean={wf_return_h3['baseline_mean_mae_pct']:.2f}% "
                  f"(edge={wf_return_h3['edge_vs_mean_mae_pct']:+.2f}pp) -> {'ADA EDGE' if has_edge_h3 else 'TIDAK ADA EDGE NYATA'}")
            print(f"     H+5 return: MAE model={wf_return_h5['mae_pct']:.2f}% vs baseline-mean={wf_return_h5['baseline_mean_mae_pct']:.2f}% "
                  f"(edge={wf_return_h5['edge_vs_mean_mae_pct']:+.2f}pp) -> {'ADA EDGE' if has_edge_h5 else 'TIDAK ADA EDGE NYATA'}")
            print(f"     H+10 return: MAE model={wf_return_h10['mae_pct']:.2f}% vs baseline-mean={wf_return_h10['baseline_mean_mae_pct']:.2f}% "
                  f"(edge={wf_return_h10['edge_vs_mean_mae_pct']:+.2f}pp) -> {'ADA EDGE' if has_edge_h10 else 'TIDAK ADA EDGE NYATA'}")
            print("   KENAPA MODEL MEMPREDIKSI INI (XAI/SHAP, fitur paling berpengaruh):")
            if xai_direction_h1["available"]:
                for f in xai_direction_h1["top_features"][:3]:
                    print(f"     [Arah H+1] {f['feature']}={f['value']:.2f} -> {f['direction']} (kontribusi {f['contribution']:+.4f})")
            else:
                print(f"     [Arah H+1] Penjelasan tidak tersedia: {xai_direction_h1['reason']}")
            if xai_return_h3["available"]:
                for f in xai_return_h3["top_features"][:3]:
                    print(f"     [Return H+3] {f['feature']}={f['value']:.2f} -> {f['direction']} (kontribusi {f['contribution']:+.4f})")
            else:
                print(f"     [Return H+3] Penjelasan tidak tersedia: {xai_return_h3['reason']}")
            print("-" * 40)

            summary["analyzed"].append({
                "ticker": ticker,
                "current_date": current_date_str,
                "target_date": target_date_str,
                "next_day_target_date": next_day_target_date_str,
                "xgboost_projected_return_pct": projection["projected_return_pct"],
                "lstm_projected_return_pct": lstm_projection["projected_return_pct"] if lstm_projection is not None else None,
                "xgboost_next_day_return_pct": next_day_projection["projected_return_pct"],
                "lstm_next_day_return_pct": next_day_lstm_projection["projected_return_pct"] if next_day_lstm_projection is not None else None,
                "xgboost_h5_return_pct": projections[5]["projected_return_pct"],
                "xgboost_h10_return_pct": projections[10]["projected_return_pct"],
                "lightgbm_h3_return_pct": lightgbm_projections.get(3, {}).get("projected_return_pct"),
                "lightgbm_h5_return_pct": lightgbm_projections.get(5, {}).get("projected_return_pct"),
                "lightgbm_h10_return_pct": lightgbm_projections.get(10, {}).get("projected_return_pct"),
                "direction_ensemble": ensemble_direction["direction"],
                "direction_confidence_pct": ensemble_direction["confidence_pct"],
                "walk_forward_h1_accuracy_pct": wf_direction_h1["direction_accuracy_pct"],
                "walk_forward_h3_accuracy_pct": wf_return_h3["direction_accuracy_pct"],
                "walk_forward_h5_accuracy_pct": wf_return_h5["direction_accuracy_pct"],
                "walk_forward_h10_accuracy_pct": wf_return_h10["direction_accuracy_pct"],
                "baseline_h1_best_accuracy_pct": float(baseline_h1["direction_accuracy_pct"].max()) if not baseline_h1.empty else 0.0,
                "baseline_h3_best_accuracy_pct": float(baseline_h3["direction_accuracy_pct"].max()) if not baseline_h3.empty else 0.0,
                "walk_forward_h1_baseline_majority_pct": wf_direction_h1["baseline_majority_accuracy_pct"],
                "walk_forward_h1_edge_pct": wf_direction_h1["edge_vs_baseline_pct"],
                "walk_forward_h1_pred_positive_rate_pct": wf_direction_h1["pred_positive_rate_pct"],
                "walk_forward_h3_edge_mae_pct": wf_return_h3["edge_vs_mean_mae_pct"],
                "walk_forward_h5_edge_mae_pct": wf_return_h5["edge_vs_mean_mae_pct"],
                "walk_forward_h10_edge_mae_pct": wf_return_h10["edge_vs_mean_mae_pct"],
                "has_genuine_edge_h1": bool(has_edge_h1),
                "has_genuine_edge_h3": bool(has_edge_h3),
                "has_genuine_edge_h5": bool(has_edge_h5),
                "has_genuine_edge_h10": bool(has_edge_h10),
                # PENTING: flag has_genuine_edge_* di atas HANYA gate effect-size
                # (edge_vs_baseline_pct >= EDGE_THRESHOLD_PCT) untuk SATU ticker
                # ini saja -- BELUM dikoreksi multiple-testing (FDR), karena
                # koreksi itu butuh p-value dari SELURUH universe ticker
                # sekaligus (lihat scripts/screen_genuine_edge.py +
                # data/edge_screening_status.json, yang MEMANG dikoreksi FDR
                # dan jadi acuan gating trust di dashboard). Angka di run
                # satu-ticker ini untuk observasi cepat, bukan keputusan trust.
                "walk_forward_h1_pvalue": wf_direction_h1.get("p_value_vs_baseline", 1.0),
                "walk_forward_h3_pvalue": wf_return_h3.get("p_value_vs_baseline", 1.0),
                "walk_forward_h5_pvalue": wf_return_h5.get("p_value_vs_baseline", 1.0),
                "walk_forward_h10_pvalue": wf_return_h10.get("p_value_vs_baseline", 1.0),
                "xai_direction_h1": xai_direction_h1,
                "xai_return_h3": xai_return_h3,
                "garch_volatility_pct": float(garch_projection.get("projected_volatility_pct", 0.0)),
                "garch_var95_pct": float(garch_projection.get("value_at_risk_95_pct", 0.0)),
            })
            if progress_callback:
                progress_callback({
                    "stage": "ticker_succeeded",
                    "ticker": ticker,
                    "message": f"{ticker} selesai dianalisis.",
                    "total": len(tickers),
                    "completed": index,
                    "analyzed_count": len(summary["analyzed"]),
                    "failed_count": len(summary["failed"]),
                    "skipped_count": len(summary["skipped"]),
                })
        except Exception as e:
            print(f"   -> Melewati {ticker}: {e}")
            summary["failed"].append({"ticker": ticker, "reason": str(e)})
            if progress_callback:
                progress_callback({
                    "stage": "ticker_failed",
                    "ticker": ticker,
                    "message": f"{ticker} gagal dianalisis: {e}",
                    "reason": str(e),
                    "total": len(tickers),
                    "completed": index,
                    "analyzed_count": len(summary["analyzed"]),
                    "failed_count": len(summary["failed"]),
                    "skipped_count": len(summary["skipped"]),
                })

    if progress_callback:
        progress_callback({
            "stage": "done",
            "message": "Analisis batch selesai.",
            "total": len(tickers),
            "completed": len(tickers),
            "analyzed_count": len(summary["analyzed"]),
            "failed_count": len(summary["failed"]),
            "skipped_count": len(summary["skipped"]),
        })

    try:
        if prediction_run_type == "FINAL":
            record_training_run(
                tickers=tickers,
                run_type=prediction_run_type,
                trigger="BATCH_ANALYSIS",
                analyzed_count=len(summary["analyzed"]),
                skipped_count=len(summary["skipped"]),
                failed_count=len(summary["failed"]),
            )
    except Exception as e:
        print(f"Registry training tidak dapat diperbarui: {e}")

    return summary


if __name__ == "__main__":
    run_full_analysis()
