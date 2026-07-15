import pandas as pd
import os
from datetime import datetime, timedelta
from pandas.errors import EmptyDataError
from data_loader import DataLoader
from src.models.walk_forward import EDGE_THRESHOLD_PCT
from src.utils.atomic_io import atomic_write_csv
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, "data", "tracking", "predictions_log.csv")
ACCURACY_FILE = os.path.join(PROJECT_ROOT, "data", "tracking", "accuracy_log.csv")
TRUSTED_ACCURACY_RUN_TYPES = {"FINAL"}
LEGACY_UNKNOWN_RUN_TYPE = "UNKNOWN_LEGACY"

def _ensure_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def load_accuracy_log(accuracy_file: str = ACCURACY_FILE) -> pd.DataFrame:
    if not os.path.exists(accuracy_file):
        return pd.DataFrame()
    if os.path.getsize(accuracy_file) == 0:
        return pd.DataFrame()

    try:
        df = pd.read_csv(accuracy_file)
    except EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["model_name"] = df["model_name"].astype(str).str.strip()
    df["direction_correct"] = df["direction_correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["error_margin_pct"] = pd.to_numeric(df["error_margin_pct"], errors="coerce")
    df["evaluation_day"] = pd.to_datetime(df["evaluation_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "prediction_purpose" not in df.columns:
        df["prediction_purpose"] = "MODEL_ACCURACY"
    df["prediction_purpose"] = (
        df["prediction_purpose"]
        .fillna("MODEL_ACCURACY")
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": "MODEL_ACCURACY", "NAN": "MODEL_ACCURACY", "NONE": "MODEL_ACCURACY"})
    )
    if "horizon_days" not in df.columns:
        df["horizon_days"] = None
    if "prediction_run_type" not in df.columns:
        df["prediction_run_type"] = LEGACY_UNKNOWN_RUN_TYPE
    df["prediction_run_type"] = (
        df["prediction_run_type"]
        .fillna(LEGACY_UNKNOWN_RUN_TYPE)
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": LEGACY_UNKNOWN_RUN_TYPE, "NAN": LEGACY_UNKNOWN_RUN_TYPE, "NONE": LEGACY_UNKNOWN_RUN_TYPE})
    )
    if "is_active_at_evaluation" not in df.columns:
        df["is_active_at_evaluation"] = True
    df["is_active_at_evaluation"] = df["is_active_at_evaluation"].fillna(True)
    df["is_active_at_evaluation"] = (
        df["is_active_at_evaluation"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    return df.dropna(subset=["evaluation_day"])

def log_prediction(
    ticker: str,
    model_name: str,
    current_date: str,
    target_date: str,
    predicted_price: float,
    current_price: float,
    horizon_days: int | None = None,
    prediction_purpose: str = "MODEL_ACCURACY",
    predicted_direction: str | None = None,
    prob_up: float | None = None,
    prob_down: float | None = None,
    confidence_pct: float | None = None,
    duplicate_policy: str | None = None,
    prediction_run_type: str | None = None,
    model_version: str | None = None,
    training_run_id: str | None = None,
    trained_until_date: str | None = None,
    prediction_mode: str | None = None,
):
    """
    Mencatat prediksi model ke file CSV.
    """
    _ensure_dir(PREDICTIONS_FILE)
    
    ticker = str(ticker).replace(".JK", "").upper().strip()
    model_name = str(model_name).strip()
    prediction_purpose = str(prediction_purpose).upper().strip()
    horizon_days = int(horizon_days) if horizon_days is not None else None
    duplicate_policy = (
        duplicate_policy
        or os.getenv("AI_TRADING_DUPLICATE_POLICY")
        or "skip"
    )
    duplicate_policy = str(duplicate_policy).lower().strip().replace("-", "_")
    if duplicate_policy not in {"skip", "overwrite", "intraday"}:
        duplicate_policy = "skip"
    prediction_run_type = (
        prediction_run_type
        or os.getenv("AI_TRADING_PREDICTION_RUN_TYPE")
        or ("INTRADAY" if duplicate_policy == "intraday" else "FINAL")
    )
    prediction_run_type = str(prediction_run_type).upper().strip()

    predicted_return_pct = (float(predicted_price) - float(current_price)) / float(current_price) * 100
    if predicted_direction is None:
        predicted_direction = "NAIK" if predicted_return_pct > 0 else "TURUN" if predicted_return_pct < 0 else "NETRAL"
    predicted_direction = str(predicted_direction).upper().strip()

    new_data = pd.DataFrame([{
        "timestamp_prediction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "model_name": model_name,
        "current_date": current_date,
        "target_date": target_date,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_return_pct": predicted_return_pct,
        "predicted_direction": predicted_direction,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "confidence_pct": confidence_pct,
        "horizon_days": horizon_days,
        "prediction_purpose": prediction_purpose,
        "prediction_run_type": prediction_run_type,
        "model_version": model_version,
        "training_run_id": training_run_id,
        "trained_until_date": trained_until_date,
        "prediction_mode": prediction_mode or "TRAIN_AND_PREDICT",
        "duplicate_policy": duplicate_policy.upper(),
        "is_active": True,
        "superseded_at": None,
        "superseded_by_policy": None,
        "status": "PENDING" # PENDING, EVALUATED
    }])
    
    if os.path.exists(PREDICTIONS_FILE) and os.path.getsize(PREDICTIONS_FILE) > 0:
        try:
            df = pd.read_csv(PREDICTIONS_FILE)
        except EmptyDataError:
            df = pd.DataFrame()
        for column, default_value in {
            "horizon_days": None,
            "prediction_purpose": "MODEL_ACCURACY",
            "predicted_return_pct": None,
            "predicted_direction": None,
            "prob_up": None,
            "prob_down": None,
            "confidence_pct": None,
            "prediction_run_type": "FINAL",
            "model_version": None,
            "training_run_id": None,
            "trained_until_date": None,
            "prediction_mode": "TRAIN_AND_PREDICT",
            "duplicate_policy": "SKIP",
            "is_active": True,
            "superseded_at": None,
            "superseded_by_policy": None,
        }.items():
            if column not in df.columns:
                df[column] = default_value
        df["superseded_at"] = df["superseded_at"].astype("object")
        df["superseded_by_policy"] = df["superseded_by_policy"].astype("object")
        duplicate_mask = (
            (df["ticker"].astype(str).str.upper().str.strip() == ticker)
            & (df["model_name"].astype(str).str.strip() == model_name)
            & (df["current_date"].astype(str) == str(current_date))
            & (df["target_date"].astype(str) == str(target_date))
            & (pd.to_numeric(df["horizon_days"], errors="coerce").fillna(-1).astype(int) == (horizon_days if horizon_days is not None else -1))
            & (df["prediction_purpose"].astype(str).str.upper().str.strip() == prediction_purpose)
        )
        active_duplicate_mask = duplicate_mask & df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])
        if active_duplicate_mask.any() and duplicate_policy == "skip":
            logger.info(f"Prediksi {model_name} untuk {ticker} pada target {target_date} sudah ada, tidak dicatat ulang.")
            return
        if duplicate_mask.any() and duplicate_policy == "overwrite":
            df.loc[duplicate_mask, "is_active"] = False
            if "status" in df.columns:
                pending_duplicate_mask = duplicate_mask & (df["status"].astype(str).str.upper().str.strip() == "PENDING")
                df.loc[pending_duplicate_mask, "status"] = "SUPERSEDED"
            df.loc[duplicate_mask, "superseded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.loc[duplicate_mask, "superseded_by_policy"] = "OVERWRITE"
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
        
    atomic_write_csv(df, PREDICTIONS_FILE, index=False)
    logger.info(f"Prediksi {model_name} untuk {ticker} berhasil dicatat.")

def evaluate_pending_predictions():
    """
    Mengevaluasi semua prediksi yang statusnya PENDING dengan membandingkan
    harga aktual di target_date (jika datanya sudah tersedia di lokal).
    """
    if not os.path.exists(PREDICTIONS_FILE):
        return
    if os.path.getsize(PREDICTIONS_FILE) == 0:
        return
        
    try:
        df_preds = pd.read_csv(PREDICTIONS_FILE)
    except EmptyDataError:
        return
    for column, default_value in {
        "horizon_days": None,
        "prediction_purpose": "MODEL_ACCURACY",
        "predicted_return_pct": None,
        "predicted_direction": None,
        "prob_up": None,
        "prob_down": None,
        "confidence_pct": None,
        "prediction_run_type": "FINAL",
        "duplicate_policy": "SKIP",
        "is_active": True,
        "timestamp_prediction": None,
        "status": "PENDING",
    }.items():
        if column not in df_preds.columns:
            df_preds[column] = default_value
    df_preds["status"] = df_preds["status"].astype(str).str.upper().str.strip()
    df_preds["is_active"] = df_preds["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])
    pending_mask = (df_preds["status"] == "PENDING") & df_preds["is_active"]
    
    if not pending_mask.any():
        return
        
    loader = DataLoader(min_rows=50) # Hanya butuh data terbaru
    evaluated_count = 0
    
    # Kelompokkan pending berdasarkan ticker agar kita hanya perlu memuat data per ticker sekali
    pending_tickers = df_preds.loc[pending_mask, 'ticker'].unique()
    
    accuracy_records = []
    
    for ticker in pending_tickers:
        # Muat data offline aktual
        actual_data = loader.load_data(ticker)
        if actual_data is None or actual_data.empty:
            continue
            
        # Konversi kolom timestamp ke datetime untuk perbandingan
        actual_data['timestamp'] = pd.to_datetime(actual_data['timestamp']).dt.normalize()
        
        # Ambil prediksi untuk ticker ini yang masih pending
        idx_pending = df_preds[
            (df_preds['ticker'] == ticker)
            & (df_preds['status'] == 'PENDING')
            & (df_preds["is_active"])
        ].index
        
        for idx in idx_pending:
            row = df_preds.loc[idx]
            try:
                target_date = pd.to_datetime(row['target_date']).normalize()
            except Exception:
                continue
            
            # Cek apakah target_date ada di data aktual (atau setelahnya)
            # Karena bisa saja hari libur bursa, kita ambil hari perdagangan pertama >= target_date
            future_data = actual_data[actual_data['timestamp'] >= target_date]
            
            if not future_data.empty:
                # Dapatkan baris data aktual untuk dievaluasi
                actual_row = future_data.iloc[0]
                actual_target_date = actual_row['timestamp'].strftime("%Y-%m-%d")
                actual_price = pd.to_numeric(actual_row['close'], errors="coerce")
                predicted_price = pd.to_numeric(row['predicted_price'], errors="coerce")
                current_price = pd.to_numeric(row['current_price'], errors="coerce")
                if pd.isna(actual_price) or pd.isna(predicted_price) or pd.isna(current_price):
                    continue
                if float(actual_price) <= 0 or float(current_price) <= 0:
                    continue
                
                # Hitung metrik
                # Return Aktual vs Return Prediksi
                actual_return = (actual_price - current_price) / current_price * 100
                predicted_return = (predicted_price - current_price) / current_price * 100
                logged_direction = str(row.get("predicted_direction", "")).upper().strip()
                predicted_direction = logged_direction if logged_direction in ["NAIK", "TURUN", "NETRAL"] else (
                    "NAIK" if predicted_return > 0 else "TURUN" if predicted_return < 0 else "NETRAL"
                )
                actual_direction = "NAIK" if actual_return > 0 else "TURUN" if actual_return < 0 else "NETRAL"
                
                # Arah (Naik/Turun)
                direction_correct = predicted_direction == actual_direction and actual_direction != "NETRAL"
                
                # Error Absolut
                absolute_error_pct = abs((predicted_price - actual_price) / actual_price) * 100
                
                accuracy_records.append({
                    "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "model_name": row['model_name'],
                    "predicted_date": row['current_date'],
                    "target_date_requested": row['target_date'],
                    "actual_evaluated_date": actual_target_date,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "actual_price": actual_price,
                    "predicted_return_pct": predicted_return,
                    "actual_return_pct": actual_return,
                    "predicted_direction": predicted_direction,
                    "actual_direction": actual_direction,
                    "direction_correct": direction_correct,
                    "error_margin_pct": absolute_error_pct,
                    "horizon_days": row.get("horizon_days"),
                    "prediction_purpose": row.get("prediction_purpose", "MODEL_ACCURACY"),
                    "prediction_run_type": str(row.get("prediction_run_type", "FINAL")).upper().strip(),
                    "model_version": row.get("model_version"),
                    "training_run_id": row.get("training_run_id"),
                    "trained_until_date": row.get("trained_until_date"),
                    "prediction_mode": row.get("prediction_mode", "TRAIN_AND_PREDICT"),
                    "duplicate_policy": str(row.get("duplicate_policy", "SKIP")).upper().strip(),
                    "timestamp_prediction": row.get("timestamp_prediction"),
                    "is_active_at_evaluation": bool(row.get("is_active", True)),
                    "evaluation_policy": "FIRST_AVAILABLE_TRADING_DAY_ON_OR_AFTER_TARGET",
                })
                
                # Tandai sebagai EVALUATED
                df_preds.at[idx, 'status'] = 'EVALUATED'
                evaluated_count += 1

    # Simpan kembali status prediksi
    if evaluated_count > 0:
        atomic_write_csv(df_preds, PREDICTIONS_FILE, index=False)

        # Simpan hasil evaluasi ke accuracy_log.csv
        _ensure_dir(ACCURACY_FILE)
        new_acc_df = pd.DataFrame(accuracy_records)
        if os.path.exists(ACCURACY_FILE) and os.path.getsize(ACCURACY_FILE) > 0:
            try:
                acc_df = pd.read_csv(ACCURACY_FILE)
            except EmptyDataError:
                acc_df = pd.DataFrame()
            acc_df = pd.concat([acc_df, new_acc_df], ignore_index=True)
        else:
            acc_df = new_acc_df
        atomic_write_csv(acc_df, ACCURACY_FILE, index=False)
        logger.info(f"Berhasil mengevaluasi {evaluated_count} prediksi.")

def _filter_accuracy_purpose(df: pd.DataFrame, prediction_purpose: str | None) -> pd.DataFrame:
    if prediction_purpose is None or df.empty:
        return df
    return df[df["prediction_purpose"].astype(str).str.upper().str.strip() == prediction_purpose.upper().strip()]


def _filter_trusted_accuracy_scope(df: pd.DataFrame, include_backfill: bool = False) -> pd.DataFrame:
    if include_backfill or df.empty:
        return df
    if "prediction_run_type" not in df.columns:
        return df
    run_type = df["prediction_run_type"].fillna("FINAL").astype(str).str.upper().str.strip()
    trusted = df[run_type.isin(TRUSTED_ACCURACY_RUN_TYPES)].copy()
    if "is_active_at_evaluation" in trusted.columns:
        trusted = trusted[trusted["is_active_at_evaluation"].astype(str).str.lower().isin(["true", "1", "yes"])]
    return trusted


def get_model_accuracy_summary(
    accuracy_file: str = ACCURACY_FILE,
    prediction_purpose: str | None = None,
    include_backfill: bool = False,
):
    """
    Menghasilkan ringkasan akurasi model untuk ditampilkan di dashboard.
    """
    df = load_accuracy_log(accuracy_file)
    df = _filter_accuracy_purpose(df, prediction_purpose)
    df = _filter_trusted_accuracy_scope(df, include_backfill=include_backfill)
    if df.empty:
        return pd.DataFrame()
        
    # Group by Model and Ticker
    summary = df.groupby(['model_name', 'ticker']).agg(
        total_evaluations=('direction_correct', 'count'),
        direction_accuracy_pct=('direction_correct', lambda x: x.mean() * 100),
        avg_error_margin_pct=('error_margin_pct', 'mean')
    ).reset_index()
    if {"predicted_direction", "actual_return_pct"}.issubset(df.columns):
        upside_df = df[df["predicted_direction"].astype(str).str.upper().str.strip() == "NAIK"].copy()
        if not upside_df.empty:
            upside_metrics = upside_df.groupby(["model_name", "ticker"]).agg(
                naik_signals=("direction_correct", "count"),
                precision_naik_pct=("direction_correct", lambda x: x.mean() * 100),
                avg_return_after_naik_pct=("actual_return_pct", "mean"),
            ).reset_index()
            summary = summary.merge(upside_metrics, on=["model_name", "ticker"], how="left")
    for column in ["naik_signals", "precision_naik_pct", "avg_return_after_naik_pct"]:
        if column not in summary.columns:
            summary[column] = 0.0
    summary["naik_signals"] = summary["naik_signals"].fillna(0).astype(int)
    summary["precision_naik_pct"] = summary["precision_naik_pct"].fillna(0.0).round(2)
    summary["avg_return_after_naik_pct"] = summary["avg_return_after_naik_pct"].fillna(0.0).round(2)
    
    # Sort berdasarkan akurasi tertinggi
    summary = summary.sort_values(by=['direction_accuracy_pct', 'avg_error_margin_pct'], ascending=[False, True])
    return summary


def get_daily_accuracy_recap(
    accuracy_file: str = ACCURACY_FILE,
    prediction_purpose: str | None = None,
    include_backfill: bool = False,
) -> pd.DataFrame:
    """Returns daily model accuracy recap by evaluation day, ticker, and model."""
    df = load_accuracy_log(accuracy_file)
    df = _filter_accuracy_purpose(df, prediction_purpose)
    df = _filter_trusted_accuracy_scope(df, include_backfill=include_backfill)
    if df.empty:
        return pd.DataFrame()

    recap = df.groupby(["evaluation_day", "ticker", "model_name"]).agg(
        total_evaluations=("direction_correct", "count"),
        correct_predictions=("direction_correct", "sum"),
        direction_accuracy_pct=("direction_correct", lambda x: x.mean() * 100),
        avg_error_margin_pct=("error_margin_pct", "mean"),
    ).reset_index()

    recap["wrong_predictions"] = recap["total_evaluations"] - recap["correct_predictions"]
    recap["direction_accuracy_pct"] = recap["direction_accuracy_pct"].round(2)
    recap["avg_error_margin_pct"] = recap["avg_error_margin_pct"].round(2)
    return recap.sort_values(["evaluation_day", "ticker", "direction_accuracy_pct"], ascending=[False, True, False])


def get_overall_daily_accuracy_recap(
    accuracy_file: str = ACCURACY_FILE,
    prediction_purpose: str | None = None,
    include_backfill: bool = False,
) -> pd.DataFrame:
    """Returns daily aggregate accuracy across all tickers, grouped by day and model."""
    df = load_accuracy_log(accuracy_file)
    df = _filter_accuracy_purpose(df, prediction_purpose)
    df = _filter_trusted_accuracy_scope(df, include_backfill=include_backfill)
    if df.empty:
        return pd.DataFrame()

    recap = df.groupby(["evaluation_day", "model_name"]).agg(
        total_predictions=("direction_correct", "count"),
        correct_predictions=("direction_correct", "sum"),
        avg_error_margin_pct=("error_margin_pct", "mean"),
        unique_tickers=("ticker", "nunique"),
    ).reset_index()

    recap["wrong_predictions"] = recap["total_predictions"] - recap["correct_predictions"]
    recap["direction_accuracy_pct"] = (
        recap["correct_predictions"] / recap["total_predictions"].clip(lower=1) * 100
    ).round(2)
    recap["avg_error_margin_pct"] = recap["avg_error_margin_pct"].round(2)
    return recap.sort_values(["evaluation_day", "direction_accuracy_pct"], ascending=[False, False])


def get_best_model_recommendations(
    accuracy_file: str = ACCURACY_FILE,
    min_evaluations: int = 3,
    prediction_purpose: str | None = None,
    include_backfill: bool = False,
) -> pd.DataFrame:
    """Recommends the most reliable model for each ticker."""
    summary = get_model_accuracy_summary(
        accuracy_file,
        prediction_purpose=prediction_purpose,
        include_backfill=include_backfill,
    )
    if summary.empty:
        return pd.DataFrame()

    recommendations = summary.copy()
    sample_factor = (recommendations["total_evaluations"] / max(min_evaluations, 1)).clip(upper=1.0)
    accuracy_component = recommendations["direction_accuracy_pct"] / 100
    error_component = (1 - recommendations["avg_error_margin_pct"].fillna(100).clip(lower=0, upper=100) / 100)
    recommendations["reliability_score"] = ((accuracy_component * 0.7 + error_component * 0.3) * sample_factor * 100).round(2)
    recommendations["sample_status"] = recommendations["total_evaluations"].apply(
        lambda count: "CUKUP" if count >= min_evaluations else "SAMPEL RENDAH"
    )

    recommendations = recommendations.sort_values(
        ["ticker", "reliability_score", "direction_accuracy_pct", "avg_error_margin_pct"],
        ascending=[True, False, False, True],
    )
    best = recommendations.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return best[[
        "ticker",
        "model_name",
        "reliability_score",
        "sample_status",
        "total_evaluations",
        "direction_accuracy_pct",
        "naik_signals",
        "precision_naik_pct",
        "avg_return_after_naik_pct",
        "avg_error_margin_pct",
    ]]


def get_model_trading_leaderboard(
    accuracy_file: str = ACCURACY_FILE,
    prediction_purpose: str | None = "NEXT_DAY_DIRECTION",
    min_evaluations: int = 3,
    include_backfill: bool = False,
) -> pd.DataFrame:
    """Leaderboard model per ticker berbasis akurasi dan proxy performa trading."""
    df = load_accuracy_log(accuracy_file)
    df = _filter_accuracy_purpose(df, prediction_purpose)
    df = _filter_trusted_accuracy_scope(df, include_backfill=include_backfill)
    if df.empty:
        return pd.DataFrame()

    for col in ["predicted_return_pct", "actual_return_pct", "error_margin_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    direction_sign = df["predicted_direction"].astype(str).str.upper().map({"NAIK": 1, "TURUN": -1}).fillna(0)
    df["is_naik_signal"] = df["predicted_direction"].astype(str).str.upper().str.strip() == "NAIK"
    df["strategy_return_pct"] = df["actual_return_pct"].fillna(0) * direction_sign
    df["win"] = df["strategy_return_pct"] > 0
    df["naik_win"] = df["is_naik_signal"] & (df["actual_return_pct"] > 0)
    # Arah aktual dari return yang benar-benar terjadi (bukan prediksi) -- dipakai
    # untuk baseline "tebak arah mayoritas" di bawah, sama seperti pendekatan
    # walk_forward.py: akurasi mentah TIDAK BERARTI apa-apa sendirian kalau
    # periode evaluasi kebetulan didominasi satu arah (mis. saat market
    # trending naik terus, tebak "NAIK" terus juga akan kelihatan akurat tanpa
    # skill apa pun).
    df["actual_direction_up"] = df["actual_return_pct"].fillna(0) > 0

    def profit_factor(series: pd.Series) -> float:
        gross_profit = series[series > 0].sum()
        gross_loss = abs(series[series <= 0].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def majority_baseline_pct(series: pd.Series) -> float:
        up_rate = series.mean()
        return max(up_rate, 1 - up_rate) * 100.0

    leaderboard = df.groupby(["ticker", "model_name"]).agg(
        total_evaluations=("direction_correct", "count"),
        direction_accuracy_pct=("direction_correct", lambda x: x.mean() * 100),
        avg_error_margin_pct=("error_margin_pct", "mean"),
        avg_strategy_return_pct=("strategy_return_pct", "mean"),
        total_strategy_return_pct=("strategy_return_pct", "sum"),
        win_rate_pct=("win", lambda x: x.mean() * 100),
        naik_signals=("is_naik_signal", "sum"),
        precision_naik_pct=("naik_win", lambda x: x.sum() / max(int(df.loc[x.index, "is_naik_signal"].sum()), 1) * 100),
        avg_return_after_naik_pct=("actual_return_pct", lambda x: x[df.loc[x.index, "is_naik_signal"]].mean()),
        profit_factor=("strategy_return_pct", profit_factor),
        baseline_majority_accuracy_pct=("actual_direction_up", majority_baseline_pct),
    ).reset_index()
    leaderboard["baseline_majority_accuracy_pct"] = leaderboard["baseline_majority_accuracy_pct"].round(2)
    leaderboard["edge_vs_baseline_pct"] = (
        leaderboard["direction_accuracy_pct"] - leaderboard["baseline_majority_accuracy_pct"]
    ).round(2)

    sample_factor = (leaderboard["total_evaluations"] / max(min_evaluations, 1)).clip(upper=1.0)
    accuracy_component = leaderboard["direction_accuracy_pct"].fillna(0) / 100.0
    error_component = 1 - leaderboard["avg_error_margin_pct"].fillna(100).clip(lower=0, upper=100) / 100.0
    return_component = ((leaderboard["avg_strategy_return_pct"].fillna(0).clip(lower=-5, upper=5) + 5) / 10)
    pf_component = (leaderboard["profit_factor"].replace(float("inf"), 5).fillna(0).clip(lower=0, upper=5) / 5)
    leaderboard["trading_score"] = (
        (accuracy_component * 0.35 + error_component * 0.20 + return_component * 0.25 + pf_component * 0.20)
        * sample_factor
        * 100
    ).round(2)
    leaderboard["sample_status"] = leaderboard["total_evaluations"].apply(
        lambda count: "CUKUP" if count >= min_evaluations else "SAMPEL RENDAH"
    )
    leaderboard["precision_naik_pct"] = leaderboard["precision_naik_pct"].fillna(0.0).round(2)
    leaderboard["avg_return_after_naik_pct"] = leaderboard["avg_return_after_naik_pct"].fillna(0.0).round(2)

    return leaderboard.sort_values(
        ["ticker", "trading_score", "direction_accuracy_pct", "profit_factor"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def get_confidence_calibration_summary(
    predictions_file: str = PREDICTIONS_FILE,
    accuracy_file: str = ACCURACY_FILE,
    prediction_purpose: str | None = "NEXT_DAY_DIRECTION",
    include_backfill: bool = False,
) -> pd.DataFrame:
    """Ringkasan apakah confidence prediksi sejalan dengan akurasi aktual."""
    if not os.path.exists(predictions_file):
        return pd.DataFrame()
    if os.path.getsize(predictions_file) == 0:
        return pd.DataFrame()
    try:
        pred_df = pd.read_csv(predictions_file)
    except EmptyDataError:
        return pd.DataFrame()
    acc_df = load_accuracy_log(accuracy_file)
    acc_df = _filter_accuracy_purpose(acc_df, prediction_purpose)
    acc_df = _filter_trusted_accuracy_scope(acc_df, include_backfill=include_backfill)
    if pred_df.empty or acc_df.empty or "confidence_pct" not in pred_df.columns:
        return pd.DataFrame()

    for col in ["ticker", "model_name"]:
        pred_df[col] = pred_df[col].astype(str).str.upper().str.strip()
        acc_df[col] = acc_df[col].astype(str).str.upper().str.strip()
    pred_df["current_date"] = pred_df["current_date"].astype(str)
    pred_df["target_date"] = pred_df["target_date"].astype(str)
    if "prediction_purpose" not in pred_df.columns:
        pred_df["prediction_purpose"] = "MODEL_ACCURACY"
    pred_df["prediction_purpose"] = pred_df["prediction_purpose"].astype(str).str.upper().str.strip()
    if "prediction_run_type" not in pred_df.columns:
        pred_df["prediction_run_type"] = LEGACY_UNKNOWN_RUN_TYPE
    pred_df["prediction_run_type"] = (
        pred_df["prediction_run_type"]
        .fillna(LEGACY_UNKNOWN_RUN_TYPE)
        .astype(str)
        .str.upper()
        .str.strip()
    )
    pred_df["confidence_pct"] = pd.to_numeric(pred_df["confidence_pct"], errors="coerce")
    for col in ["predicted_price", "current_price"]:
        if col in pred_df.columns:
            pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce").round(6)
        if col in acc_df.columns:
            acc_df[col] = pd.to_numeric(acc_df[col], errors="coerce").round(6)
    if prediction_purpose is not None:
        pred_df = pred_df[pred_df["prediction_purpose"] == prediction_purpose.upper()]
    if not include_backfill:
        pred_df = pred_df[pred_df["prediction_run_type"].isin(TRUSTED_ACCURACY_RUN_TYPES)]

    merge_columns = ["ticker", "model_name", "current_date", "target_date", "confidence_pct"]
    left_on = ["ticker", "model_name", "predicted_date", "target_date_requested"]
    right_on = ["ticker", "model_name", "current_date", "target_date"]
    if {"predicted_price", "current_price"}.issubset(pred_df.columns) and {"predicted_price", "current_price"}.issubset(acc_df.columns):
        merge_columns.extend(["predicted_price", "current_price"])
        left_on.extend(["predicted_price", "current_price"])
        right_on.extend(["predicted_price", "current_price"])

    merged = acc_df.merge(
        pred_df[merge_columns].drop_duplicates(),
        left_on=left_on,
        right_on=right_on,
        how="left",
    ).dropna(subset=["confidence_pct"])
    if merged.empty:
        return pd.DataFrame()

    bins = [0, 55, 60, 65, 70, 80, 90, 100]
    labels = ["<=55", "55-60", "60-65", "65-70", "70-80", "80-90", "90-100"]
    merged["confidence_bucket"] = pd.cut(merged["confidence_pct"], bins=bins, labels=labels, include_lowest=True)
    summary = merged.groupby(["model_name", "confidence_bucket"], observed=True).agg(
        total_predictions=("direction_correct", "count"),
        avg_confidence_pct=("confidence_pct", "mean"),
        actual_accuracy_pct=("direction_correct", lambda x: x.mean() * 100),
    ).reset_index()
    summary["calibration_gap_pct"] = (summary["avg_confidence_pct"] - summary["actual_accuracy_pct"]).round(2)
    summary["avg_confidence_pct"] = summary["avg_confidence_pct"].round(2)
    summary["actual_accuracy_pct"] = summary["actual_accuracy_pct"].round(2)
    return summary.sort_values(["model_name", "confidence_bucket"])


def get_model_trust_audit(
    predictions_file: str = PREDICTIONS_FILE,
    accuracy_file: str = ACCURACY_FILE,
    prediction_purpose: str | None = "NEXT_DAY_DIRECTION",
    min_evaluations: int = 10,
    min_accuracy_pct: float = 55.0,
    min_edge_vs_baseline_pct: float = EDGE_THRESHOLD_PCT,
    min_profit_factor: float = 1.2,
    max_abs_calibration_gap_pct: float = 15.0,
    include_backfill: bool = False,
) -> pd.DataFrame:
    """Audit kelayakan model per saham untuk dipakai sebagai acuan trading.

    `min_edge_vs_baseline_pct` default-nya sama persis dengan
    `EDGE_THRESHOLD_PCT` di walk_forward.py (satu konstanta dipakai bersama
    supaya tidak diam-diam berbeda ambang): akurasi arah
    mentah bisa tinggi murni karena base rate periode evaluasi (mis. saham
    kebetulan naik terus), jadi status "LAYAK DIPERCAYA" TIDAK diberikan
    kalau model belum mengalahkan tebakan mayoritas naif dengan margin yang
    berarti pada periode evaluasi yang sama -- lihat `edge_vs_baseline_pct`.
    """
    leaderboard = get_model_trading_leaderboard(
        accuracy_file=accuracy_file,
        prediction_purpose=prediction_purpose,
        min_evaluations=1,
        include_backfill=include_backfill,
    )
    if leaderboard.empty:
        return pd.DataFrame()

    calibration = get_confidence_calibration_summary(
        predictions_file=predictions_file,
        accuracy_file=accuracy_file,
        prediction_purpose=prediction_purpose,
        include_backfill=include_backfill,
    )
    if calibration.empty:
        calibration_by_model = pd.DataFrame(columns=["model_key", "calibration_gap_pct"])
    else:
        calibration["model_key"] = calibration["model_name"].astype(str).str.upper().str.strip()
        calibration_by_model = calibration.groupby("model_name", as_index=False).agg(
            calibration_gap_pct=("calibration_gap_pct", lambda x: x.abs().mean())
        )
        calibration_by_model["model_key"] = calibration_by_model["model_name"].astype(str).str.upper().str.strip()
        calibration_by_model = calibration_by_model[["model_key", "calibration_gap_pct"]]

    audit = leaderboard.copy()
    audit["model_key"] = audit["model_name"].astype(str).str.upper().str.strip()
    audit = audit.merge(calibration_by_model, on="model_key", how="left")
    audit["calibration_gap_pct"] = audit["calibration_gap_pct"].fillna(999.0)
    audit["baseline_majority_accuracy_pct"] = audit["baseline_majority_accuracy_pct"].fillna(0.0)
    audit["edge_vs_baseline_pct"] = audit["edge_vs_baseline_pct"].fillna(0.0)
    audit["beats_baseline"] = audit["edge_vs_baseline_pct"] >= min_edge_vs_baseline_pct

    def status_for(row):
        reasons = []
        if row["total_evaluations"] < min_evaluations:
            reasons.append("sample evaluasi belum cukup")
        if row["direction_accuracy_pct"] < min_accuracy_pct:
            reasons.append("akurasi arah belum cukup")
        if row["edge_vs_baseline_pct"] < min_edge_vs_baseline_pct:
            reasons.append(
                f"akurasi {row['direction_accuracy_pct']:.1f}% belum unggul cukup jauh dari baseline "
                f"tebak-mayoritas {row['baseline_majority_accuracy_pct']:.1f}% "
                f"(edge {row['edge_vs_baseline_pct']:+.1f}pp < batas {min_edge_vs_baseline_pct:.1f}pp) -- "
                f"akurasi tinggi bisa cuma kebetulan searah tren pasar, bukan skill model"
            )
        if row["profit_factor"] < min_profit_factor:
            reasons.append("profit factor belum mengalahkan baseline")
        if abs(row["calibration_gap_pct"]) > max_abs_calibration_gap_pct:
            reasons.append("confidence belum terkalibrasi")

        if row["total_evaluations"] < min_evaluations:
            status = "PERLU DATA LAGI"
        elif reasons:
            status = "JANGAN DIIKUTI"
        else:
            status = "LAYAK DIPERCAYA"
        return status, "; ".join(reasons) if reasons else "Semua kriteria utama terpenuhi."

    status_reason = audit.apply(status_for, axis=1)
    audit["status_trust"] = [item[0] for item in status_reason]
    audit["alasan"] = [item[1] for item in status_reason]

    status_rank = {"LAYAK DIPERCAYA": 0, "PERLU DATA LAGI": 1, "JANGAN DIIKUTI": 2}
    audit["_status_rank"] = audit["status_trust"].map(status_rank).fillna(9)
    audit = audit.sort_values(
        ["_status_rank", "trading_score", "direction_accuracy_pct", "profit_factor"],
        ascending=[True, False, False, False],
    ).drop(columns=["_status_rank"])

    return audit[[
        "ticker",
        "model_name",
        "status_trust",
        "alasan",
        "total_evaluations",
        "direction_accuracy_pct",
        "baseline_majority_accuracy_pct",
        "edge_vs_baseline_pct",
        "profit_factor",
        "calibration_gap_pct",
        "beats_baseline",
        "win_rate_pct",
        "naik_signals",
        "precision_naik_pct",
        "avg_return_after_naik_pct",
        "avg_strategy_return_pct",
        "trading_score",
        "sample_status",
    ]].reset_index(drop=True)
