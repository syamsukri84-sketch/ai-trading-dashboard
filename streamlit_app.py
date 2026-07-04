import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import logging
import yaml
import json
import subprocess
import sys
import uuid
import importlib
import re
from datetime import datetime, timedelta
from pandas.errors import EmptyDataError

from data_loader import DataLoader
import run_analysis as analysis_runner

analysis_runner = importlib.reload(analysis_runner)
run_backfill_analysis = analysis_runner.run_backfill_analysis
run_full_analysis = analysis_runner.run_full_analysis
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.data_pipeline.auto_updater import (
    _normalize_existing_data,
    get_local_data_status,
    run_auto_updater,
    update_from_manual_dataframe,
)
from src.models.isolation_forest import IsolationForestModel
from src.models.price_projector import PriceProjector
from src.models.conformal_predictor import ConformalPredictor
from src.models.regime_classifier import RegimeClassifier
from src.models.ensemble import soft_ensemble_predict, COPODModel
from src.models.garch_model import GARCHModel
from src.models.backtest_engine import BacktestEngine
from src.models.lstm_projector import LSTMPriceProjector
from src.models.direction_classifier import DirectionClassifier
from src.models.baseline_strategies import evaluate_baseline_strategies
from src.models.walk_forward import walk_forward_direction_validation, walk_forward_return_validation
from src.trading.reliability_ensemble import get_reliability_weights, weighted_direction_probability
from src.trading.decision_support import build_decision_support, build_trade_gate, calculate_ai_confidence_score, calculate_position_sizing
from src.trading.signal_generator import generate_signal
from src.utils.model_guardrails import assert_no_training_leakage
from src.utils.accuracy_tracker import (
    log_prediction,
    evaluate_pending_predictions,
    get_best_model_recommendations,
    get_daily_accuracy_recap,
    get_model_accuracy_summary,
    get_overall_daily_accuracy_recap,
    get_model_trading_leaderboard,
    get_confidence_calibration_summary,
    get_model_trust_audit,
)
from src.utils.csv_audit import audit_prediction_csv, clean_prediction_csv
from src.utils.mongo_store import check_mongo_status
from src.utils.training_policy import evaluate_training_policy, evaluate_training_policy_by_model
from src.utils.model_store import list_ticker_models, model_store_status
from src.models.global_models import predict_with_global_models
from src.nlp.news_fetcher import fetch_google_news_sentiment_items
from src.nlp.sentiment_analyzer import analyze_dataframe, analyze_text, append_issue, append_issues, build_local_sentiment_dataset, build_trading_sentiment_summary, get_local_sentiment_dataset_path, get_sentiment_engine_status, interpret_signal, load_issues, summarize_by_ticker

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Inisialisasi komponen pipeline untuk mode offline
loader = DataLoader(min_rows=100)
engineer = FeatureEngineer(warmup_period=60)
logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JOB_DIR = os.path.join(DATA_DIR, "jobs")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")
LEGACY_MODELS_ENABLED = False


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def get_dashboard_password():
    env_password = os.getenv("AI_TRADING_DASHBOARD_PASSWORD", "").strip()
    if env_password:
        return env_password
    try:
        return str(st.secrets.get("AI_TRADING_DASHBOARD_PASSWORD", "")).strip()
    except Exception:
        return ""


def get_runtime_setting(name, default=""):
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    try:
        value = st.secrets.get(name, default)
        return str(value).strip()
    except Exception:
        return str(default).strip()


def get_bool_runtime_setting(name, default=False):
    value = get_runtime_setting(name, str(default)).lower()
    return value in {"1", "true", "yes", "y", "on"}


def require_dashboard_password():
    expected_password = get_dashboard_password()
    if not expected_password:
        return
    if st.session_state.get("dashboard_authenticated"):
        return

    st.title("AI Trading Dashboard")
    st.caption("Masukkan password untuk membuka dashboard.")
    password = st.text_input("Password", type="password")
    if st.button("Masuk", type="primary"):
        if password == expected_password:
            st.session_state["dashboard_authenticated"] = True
            st.rerun()
        else:
            st.error("Password salah.")
    st.stop()


def parse_dashboard_dates(values):
    text_values = pd.Series(values).astype(str).str.strip()
    iso_mask = text_values.str.match(ISO_DATE_PATTERN)
    parsed = pd.Series(pd.NaT, index=text_values.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(text_values.loc[iso_mask], errors="coerce")
    parsed.loc[~iso_mask] = pd.to_datetime(
        text_values.loc[~iso_mask],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )
    return parsed


def apply_dashboard_theme():
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f6f8fb;
            --panel-bg: #ffffff;
            --panel-border: #dce5ee;
            --text-main: #17202a;
            --text-muted: #64748b;
            --accent-blue: #2f80ed;
            --accent-teal: #14a38b;
            --accent-green: #2f9e44;
            --accent-amber: #f2a93b;
            --accent-red: #dc4c4c;
        }
        .stApp {
            background: var(--app-bg);
            color: var(--text-main) !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stMarkdownContainer"],
        .main,
        .block-container {
            color: var(--text-main) !important;
        }
        h1, h2, h3, h4, h5, h6,
        p, li, label,
        .stMarkdown,
        .stMarkdown p,
        .stCaptionContainer,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] p {
            color: var(--text-main) !important;
        }
        small,
        [data-testid="stCaptionContainer"],
        div[data-testid="stMarkdownContainer"] small {
            color: var(--text-muted) !important;
        }
        [data-testid="stSidebar"] {
            background: #eef5f8;
            border-right: 1px solid var(--panel-border);
            color: var(--text-main) !important;
        }
        [data-testid="stSidebar"] * {
            color: var(--text-main) !important;
        }
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] .stCaptionContainer {
            color: var(--text-muted) !important;
        }
        input,
        textarea,
        [contenteditable="true"],
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            background: #ffffff !important;
            color: var(--text-main) !important;
            caret-color: var(--text-main) !important;
        }
        input::placeholder,
        textarea::placeholder {
            color: #8293a6 !important;
            opacity: 1 !important;
        }
        div[data-baseweb="select"],
        div[data-baseweb="select"] *,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] * {
            color: var(--text-main) !important;
        }
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border-color: #c8d6e2 !important;
        }
        div[role="radiogroup"] label,
        div[role="radiogroup"] label *,
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label * {
            color: var(--text-main) !important;
        }
        .app-header {
            background: linear-gradient(115deg, #ffffff 0%, #eef9f6 58%, #edf4ff 100%);
            border: 1px solid var(--panel-border);
            border-left: 5px solid var(--accent-teal);
            border-radius: 10px;
            padding: 18px 20px;
            margin: 0 0 16px 0;
            box-shadow: 0 6px 18px rgba(23, 32, 42, 0.06);
        }
        .app-header h1 {
            color: var(--text-main) !important;
            font-size: 1.75rem;
            line-height: 1.2;
            margin: 0;
            letter-spacing: 0;
        }
        .app-header p {
            color: var(--text-muted) !important;
            font-size: 0.98rem;
            margin: 6px 0 0 0;
        }
        .sync-strip {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 12px 14px;
            margin: 4px 0 14px 0;
        }
        .sync-status {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            background: #e7f6ef;
            color: #13795b !important;
            border: 1px solid #bce7d3;
        }
        .sync-status.warn {
            background: #fff4df;
            color: #996500 !important;
            border-color: #f5d28c;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 8px 10px;
            box-shadow: 0 3px 12px rgba(23, 32, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-weight: 650;
            font-size: 0.78rem !important;
            line-height: 1.1 !important;
        }
        div[data-testid="stMetricLabel"] * {
            color: var(--text-muted) !important;
            font-size: 0.78rem !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--text-main) !important;
            letter-spacing: 0;
            font-size: 1.05rem !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            word-break: break-word !important;
        }
        div[data-testid="stMetricValue"] * {
            color: var(--text-main) !important;
            font-size: 1.05rem !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            word-break: break-word !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--panel-border);
        }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            padding: 8px 14px;
            color: var(--text-main) !important;
        }
        .stTabs [data-baseweb="tab"] * {
            color: var(--text-main) !important;
        }
        .stTabs [aria-selected="true"] {
            color: #0f766e !important;
            border-top: 3px solid var(--accent-teal);
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] * {
            color: #0f766e !important;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #bfd3e2;
            font-weight: 650;
            background: #ffffff;
            color: var(--text-main) !important;
        }
        .stButton > button * {
            color: var(--text-main) !important;
        }
        .stButton > button[kind="primary"] {
            background: #147d73;
            border-color: #147d73;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"] * {
            color: #ffffff !important;
        }
        .stButton > button:disabled,
        .stButton > button:disabled * {
            background: #edf2f7 !important;
            color: #6b7c8f !important;
        }
        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            color: var(--text-main) !important;
        }
        div[data-testid="stAlert"],
        div[data-testid="stAlert"] *,
        [data-testid="stNotification"],
        [data-testid="stNotification"] * {
            color: var(--text-main) !important;
        }
        div[data-testid="stDataFrame"] {
            color: var(--text-main) !important;
        }
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary *,
        details,
        summary,
        summary * {
            background: #ffffff !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stExpander"] details,
        div[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
        div[data-testid="stSidebar"] div[data-testid="stExpander"] summary * {
            background: #ffffff !important;
            color: var(--text-main) !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] ul,
        div[data-baseweb="menu"] li,
        ul[role="listbox"],
        div[role="listbox"],
        div[role="menu"],
        div[role="option"],
        li[role="option"] {
            background: #ffffff !important;
            color: var(--text-main) !important;
            border-color: #c8d6e2 !important;
        }
        div[role="option"] *,
        li[role="option"] *,
        ul[role="listbox"] *,
        div[role="listbox"] *,
        div[data-baseweb="menu"] *,
        div[data-baseweb="popover"] * {
            color: var(--text-main) !important;
        }
        div[role="option"][aria-selected="true"],
        li[role="option"][aria-selected="true"],
        div[role="option"]:hover,
        li[role="option"]:hover,
        div[data-baseweb="menu"] li:hover {
            background: #dff4ef !important;
            color: #0f4f46 !important;
        }
        div[role="option"][aria-selected="true"] *,
        li[role="option"][aria-selected="true"] *,
        div[role="option"]:hover *,
        li[role="option"]:hover * {
            color: #0f4f46 !important;
        }
        [data-baseweb="tag"],
        [data-baseweb="tag"] * {
            background: #dff4ef !important;
            color: #0f4f46 !important;
        }
        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] *,
        table,
        table * {
            color: var(--text-main) !important;
        }
        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] thead *,
        div[data-testid="stTable"] th,
        table th {
            background: #eef5f8 !important;
            color: #17202a !important;
        }
        div[data-testid="stDataFrame"] td,
        div[data-testid="stTable"] td,
        table td {
            background: #ffffff !important;
            color: #17202a !important;
        }
        div[data-testid="stDataFrame"] canvas {
            filter: none !important;
        }
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] *,
        [data-testid="stNumberInput"],
        [data-testid="stNumberInput"] *,
        [data-testid="stTextInput"],
        [data-testid="stTextInput"] *,
        [data-testid="stTextArea"],
        [data-testid="stTextArea"] *,
        [data-testid="stSelectbox"],
        [data-testid="stSelectbox"] *,
        [data-testid="stMultiSelect"],
        [data-testid="stMultiSelect"] * {
            color: var(--text-main) !important;
        }
        [data-testid="stFileUploader"] section {
            background: #ffffff !important;
            border-color: #c8d6e2 !important;
        }
        .stApp a,
        .stApp a * {
            color: #0f766e !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stHorizontalBlock"],
        div[data-testid="column"] {
            color: var(--text-main) !important;
        }
        h1, h2, h3 {
            color: var(--text-main) !important;
            letter-spacing: 0;
        }
        /* Final contrast guard: keep every dashboard control readable even if Streamlit/BaseWeb injects dark theme styles. */
        .stApp :where(div, span, p, label, small, li, ol, ul, h1, h2, h3, h4, h5, h6, section, article, header, footer),
        [data-testid="stSidebar"] :where(div, span, p, label, small, li, ol, ul, h1, h2, h3, h4, h5, h6, section) {
            color: #17202a !important;
            -webkit-text-fill-color: #17202a !important;
            text-shadow: none !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        .caption,
        small {
            color: #64748b !important;
            -webkit-text-fill-color: #64748b !important;
        }
        div[data-baseweb="select"],
        div[data-baseweb="select"] *,
        div[data-baseweb="input"],
        div[data-baseweb="input"] *,
        div[data-baseweb="textarea"],
        div[data-baseweb="textarea"] *,
        [data-testid="stTextInput"] *,
        [data-testid="stTextArea"] *,
        [data-testid="stNumberInput"] *,
        [data-testid="stSelectbox"] *,
        [data-testid="stMultiSelect"] * {
            background-color: #ffffff !important;
            color: #17202a !important;
            -webkit-text-fill-color: #17202a !important;
        }
        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] svg,
        [data-testid="stSelectbox"] svg,
        [data-testid="stMultiSelect"] svg {
            fill: #17202a !important;
            color: #17202a !important;
        }
        body div[data-baseweb="popover"],
        body div[data-baseweb="popover"] *,
        body div[data-baseweb="menu"],
        body div[data-baseweb="menu"] *,
        body div[role="listbox"],
        body div[role="listbox"] *,
        body ul[role="listbox"],
        body ul[role="listbox"] *,
        body div[role="option"],
        body div[role="option"] *,
        body li[role="option"],
        body li[role="option"] * {
            background-color: #ffffff !important;
            color: #17202a !important;
            -webkit-text-fill-color: #17202a !important;
            text-shadow: none !important;
        }
        body div[role="option"]:hover,
        body li[role="option"]:hover,
        body div[role="option"][aria-selected="true"],
        body li[role="option"][aria-selected="true"],
        body div[data-baseweb="menu"] li:hover,
        body div[data-baseweb="menu"] [aria-selected="true"] {
            background-color: #dff4ef !important;
            color: #0f4f46 !important;
            -webkit-text-fill-color: #0f4f46 !important;
        }
        body div[role="option"]:hover *,
        body li[role="option"]:hover *,
        body div[role="option"][aria-selected="true"] *,
        body li[role="option"][aria-selected="true"] *,
        body div[data-baseweb="menu"] li:hover *,
        body div[data-baseweb="menu"] [aria-selected="true"] * {
            color: #0f4f46 !important;
            -webkit-text-fill-color: #0f4f46 !important;
        }
        [data-testid="stExpander"],
        [data-testid="stExpander"] *,
        details,
        details *,
        summary,
        summary * {
            background-color: #ffffff !important;
            color: #17202a !important;
            -webkit-text-fill-color: #17202a !important;
        }
        [data-testid="stMetric"],
        [data-testid="stMetric"] *,
        [data-testid="stAlert"],
        [data-testid="stAlert"] * {
            color: #17202a !important;
            -webkit-text-fill-color: #17202a !important;
        }
        .stButton > button[kind="primary"],
        .stButton > button[kind="primary"] *,
        button[kind="primary"],
        button[kind="primary"] * {
            background-color: #147d73 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .stButton > button:not([kind="primary"]),
        .stButton > button:not([kind="primary"]) * {
            color: #17202a !important;
            -webkit-text-fill-color: #17202a !important;
        }
        [data-baseweb="tag"],
        [data-baseweb="tag"] * {
            background-color: #dff4ef !important;
            color: #0f4f46 !important;
            -webkit-text-fill-color: #0f4f46 !important;
        }
        .stApp a,
        .stApp a * {
            color: #0f766e !important;
            -webkit-text-fill-color: #0f766e !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header():
    st.markdown(
        """
        <div class="app-header">
            <h1>AI Trading Decision Support Dashboard</h1>
            <p>Dashboard lokal untuk analisis saham, ranking prediksi, evaluasi akurasi, dan kontrol kualitas data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_launch_sync_snapshot(ticker_list):
    total_tickers = len(ticker_list)
    raw_dir = project_path("data", "raw")
    latest_data_date = "-"
    raw_count = sum(
        1
        for ticker_code in ticker_list
        if os.path.exists(os.path.join(raw_dir, f"{str(ticker_code).replace('.JK', '').upper().strip()}_raw.csv"))
    )
    latest_dates = []
    for ticker_code in ticker_list:
        file_path = os.path.join(raw_dir, f"{str(ticker_code).replace('.JK', '').upper().strip()}_raw.csv")
        if not os.path.exists(file_path):
            continue
        try:
            tail_df = pd.read_csv(
                file_path,
                usecols=lambda col: str(col).lower() in {"timestamp", "date", "datetime"} or col == "Tanggal",
            )
            date_col = next((col for col in ["timestamp", "date", "datetime", "Tanggal"] if col in tail_df.columns), None)
            if date_col is None:
                continue
            parsed_dates = parse_dashboard_dates(tail_df[date_col]).dropna()
            if not parsed_dates.empty:
                latest_dates.append(parsed_dates.max())
        except Exception:
            continue
    if latest_dates:
        latest_data_date = max(latest_dates).strftime("%Y-%m-%d")

    pred_path = project_path("data", "tracking", "predictions_log.csv")
    acc_path = project_path("data", "tracking", "accuracy_log.csv")
    pred_rows = pending_count = evaluated_count = active_count = superseded_count = 0
    latest_prediction_date = "-"
    if os.path.exists(pred_path) and os.path.getsize(pred_path) > 0:
        try:
            pred_df = pd.read_csv(pred_path)
            pred_rows = len(pred_df)
            status = pred_df.get("status", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
            is_active = pred_df.get("is_active", pd.Series([True] * len(pred_df))).astype(str).str.lower().isin(["true", "1", "yes"])
            pending_count = int(((status == "PENDING") & is_active).sum())
            evaluated_count = int((status == "EVALUATED").sum())
            active_count = int(is_active.sum())
            superseded_count = int((status == "SUPERSEDED").sum() + ((~is_active) & (status == "PENDING")).sum())
            if "current_date" in pred_df.columns:
                prediction_dates = pd.to_datetime(pred_df["current_date"], errors="coerce").dropna()
                if not prediction_dates.empty:
                    latest_prediction_date = prediction_dates.max().strftime("%Y-%m-%d")
        except Exception:
            pred_rows = 0

    acc_rows = 0
    trusted_ready = False
    if os.path.exists(acc_path) and os.path.getsize(acc_path) > 0:
        try:
            acc_df = pd.read_csv(acc_path, usecols=lambda col: col in {"prediction_run_type", "is_active_at_evaluation"})
            acc_rows = len(acc_df)
            trusted_ready = "prediction_run_type" in acc_df.columns
        except Exception:
            acc_rows = 0

    offline_ready = total_tickers > 0 and raw_count >= total_tickers
    tracking_ready = os.path.exists(pred_path) and os.path.exists(acc_path)
    status_label = "SINKRON" if offline_ready and tracking_ready else "PERLU CEK"
    return {
        "status_label": status_label,
        "latest_data_date": latest_data_date,
        "latest_prediction_date": latest_prediction_date,
        "raw_count": raw_count,
        "total_tickers": total_tickers,
        "pred_rows": pred_rows,
        "active_count": active_count,
        "pending_count": pending_count,
        "evaluated_count": evaluated_count,
        "superseded_count": superseded_count,
        "accuracy_rows": acc_rows,
        "trusted_ready": trusted_ready,
        "tracking_ready": tracking_ready,
    }


def render_launch_sync_snapshot(snapshot, operation_mode):
    status_class = "" if snapshot["status_label"] == "SINKRON" else "warn"
    st.markdown(
        f"""
        <div class="sync-strip">
            <span class="sync-status {status_class}">{snapshot["status_label"]}</span>
            <span style="margin-left: 10px; color: #64748b; font-weight: 600;">Mode: {operation_mode}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Tanggal Data", snapshot["latest_data_date"])
    s2.metric("Tanggal Prediksi", snapshot["latest_prediction_date"])
    s3.metric("Data Lokal", f"{snapshot['raw_count']}/{snapshot['total_tickers']}")
    s4.metric("Pending Aktif", f"{snapshot['pending_count']:,}")
    s5.metric("Akurasi Log", f"{snapshot['accuracy_rows']:,}")


st.set_page_config(page_title="AI Trading Dashboard", layout="wide", page_icon="📈")

require_dashboard_password()
apply_dashboard_theme()
render_app_header()


def load_config_tickers(config_path: str = "config/stocks.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return config.get("tickers", [])
    except Exception:
        return []


def parse_local_volume_value(value):
    if pd.isna(value) or str(value).strip() in {"", "-"}:
        return 0.0
    text = str(value).strip().replace(",", ".")
    multiplier = 1.0
    if text.endswith("B"):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("K"):
        multiplier = 1_000.0
        text = text[:-1]
    try:
        return float(text.replace(" ", "")) * multiplier
    except ValueError:
        return 0.0


@st.cache_data(show_spinner=False)
def build_liquidity_tier_table(ticker_list, raw_dir=None, lookback_rows=30):
    raw_dir = raw_dir or project_path("data", "raw")
    rows = []
    for ticker_code in ticker_list:
        ticker_code = normalize_ticker_code(ticker_code)
        file_path = os.path.join(raw_dir, f"{ticker_code}_raw.csv")
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            continue
        try:
            df = pd.read_csv(file_path)
            df = df.rename(columns={
                "Tanggal": "timestamp",
                "Terakhir": "close",
                "Buka": "open",
                "Pembukaan": "open",
                "Tinggi": "high",
                "Tertinggi": "high",
                "Rendah": "low",
                "Terendah": "low",
                "Vol.": "volume",
            })
            df.columns = [str(col).lower() for col in df.columns]
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df["close"]):
                close = pd.to_numeric(df["close"], errors="coerce")
            else:
                close = df["close"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                close = pd.to_numeric(close, errors="coerce")
            if pd.api.types.is_numeric_dtype(df["volume"]):
                volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            else:
                volume = df["volume"].apply(parse_local_volume_value)
            value = (close * volume).dropna().tail(int(lookback_rows))
            if value.empty:
                continue
            rows.append({
                "ticker": ticker_code,
                "avg_traded_value": float(value.mean()),
                "data_rows": int(len(df)),
            })
        except Exception:
            continue

    tier_df = pd.DataFrame(rows)
    if tier_df.empty:
        return pd.DataFrame(columns=["ticker", "liquidity_tier", "avg_traded_value", "data_rows"])

    tier_df = tier_df.sort_values("avg_traded_value", ascending=False).reset_index(drop=True)
    total = len(tier_df)
    tier1_cut = min(45, max(1, int(round(total * 0.20))))
    tier2_cut = min(total, max(tier1_cut + 1, int(round(total * 0.55)))) if total > 1 else total
    tier_df["liquidity_tier"] = "TIER 3 - LOW"
    tier_df.loc[: tier1_cut - 1, "liquidity_tier"] = "TIER 1 - HIGH"
    if tier2_cut > tier1_cut:
        tier_df.loc[tier1_cut: tier2_cut - 1, "liquidity_tier"] = "TIER 2 - MEDIUM"
    return tier_df[["ticker", "liquidity_tier", "avg_traded_value", "data_rows"]]


def add_liquidity_tier(df, tier_df):
    if df.empty or tier_df.empty or "ticker" not in df.columns:
        return df
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out.merge(tier_df[["ticker", "liquidity_tier", "avg_traded_value", "data_rows"]], on="ticker", how="left")


tickers = load_config_tickers()

offline_only = get_bool_runtime_setting("AI_TRADING_OFFLINE_ONLY", default=True)
if offline_only:
    operation_mode = "Offline"
    st.sidebar.info("Mode offline penuh aktif. Fitur internet dan MongoDB dinonaktifkan.")
else:
    operation_mode = st.sidebar.radio(
        "Mode Operasional",
        ["Offline", "Online opsional"],
        index=0,
        help="Offline memakai data lokal dan menonaktifkan aksi yang membutuhkan internet. Online opsional mengaktifkan update harga/berita otomatis.",
    )
offline_mode = operation_mode == "Offline"
launch_snapshot = build_launch_sync_snapshot(tickers)
render_launch_sync_snapshot(launch_snapshot, operation_mode)

with st.sidebar.expander("Status Sinkronisasi", expanded=True):
    st.metric("Data Lokal", f"{launch_snapshot['raw_count']}/{launch_snapshot['total_tickers']}")
    st.metric("Pending Aktif", f"{launch_snapshot['pending_count']:,}")
    st.metric("Prediksi Aktif", f"{launch_snapshot['active_count']:,}")
    if offline_mode:
        st.success("Mode offline aktif. Analisis lokal, ranking, dan evaluasi akurasi siap digunakan.")
        st.caption("Tombol update harga online dinonaktifkan. Ubah Mode Operasional ke Online opsional jika ingin update dari dashboard.")
    else:
        st.info("Mode online opsional aktif. Update data dan berita dapat memakai koneksi internet.")
    if offline_mode:
        st.caption("MongoDB tidak diperlukan pada mode offline. Semua fitur memakai arsip dan file lokal.")
    else:
        mongo_status = check_mongo_status()
        if mongo_status["ok"]:
            st.success(f"MongoDB Atlas aktif: {mongo_status['database']}")
        elif mongo_status["enabled"]:
            st.warning(f"MongoDB Atlas belum tersambung: {mongo_status['message']}")
        else:
            st.caption("MongoDB Atlas belum aktif. Isi MONGODB_URI untuk sinkronisasi online.")

st.sidebar.header("Ticker Aktif Global")
ticker_input = st.sidebar.text_input("Kode Saham", value="BBRI", help="Ticker ini otomatis dipakai sebagai default/filter di semua tab.")
ticker = ticker_input.strip().upper()
global_data_scope = st.sidebar.radio(
    "Cakupan data dashboard",
    ["Saham aktif", "Semua saham"],
    horizontal=True,
    help="Pilih Saham aktif untuk fokus ke satu emiten, atau Semua saham untuk proses dan tampilan gabungan.",
)
use_all_tickers = global_data_scope == "Semua saham"
active_tickers = tickers if use_all_tickers else ([ticker] if ticker else [])
portfolio_capital = st.sidebar.number_input("Modal Portfolio (Rp)", min_value=1_000_000, value=100_000_000, step=1_000_000)
risk_per_trade_pct = st.sidebar.number_input("Risiko per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

if use_all_tickers:
    st.sidebar.caption(f"Mode aktif: semua saham ({len(active_tickers)} ticker)")
elif ticker:
    st.sidebar.caption(f"Ticker aktif: {ticker}")

# --- TABS UTAMA: alur harian dulu, kontrol lanjutan setelahnya ---
tab_daily, tab_update, tab_ranking, tab_accuracy, tab_sentiment = st.tabs([
    "Ringkasan Harian",
    "Workflow Harian",
    "Ranking Prediksi",
    "Akurasi Model",
    "Sentimen Pasar",
])
tab_dashboard = None
tab_final = None


def render_update_summary(summary):
    updated = summary.get("updated", [])
    skipped = summary.get("skipped", [])
    failed = summary.get("failed", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Saham", summary.get("total", 0))
    col2.metric("Diperbarui", len(updated))
    col3.metric("Sudah Terbaru / Libur", len(skipped))
    col4.metric("Gagal", len(failed))

    if updated:
        st.subheader("Data yang Diperbarui")
        st.dataframe(pd.DataFrame(updated), width="stretch")
    if failed:
        st.subheader("Gagal Update")
        st.dataframe(pd.DataFrame(failed), width="stretch")


def classify_analysis_failure(reason):
    reason_text = str(reason or "").strip()
    reason_lower = reason_text.lower()

    if "data tidak dapat dimuat" in reason_lower:
        return "Data harga tidak tersedia / tidak valid"
    if "gagal membuat fitur" in reason_lower or "jumlah data belum cukup" in reason_lower:
        return "Data historis belum cukup untuk indikator"
    if "guardrail" in reason_lower or "leakage" in reason_lower:
        return "Pemeriksaan kualitas data/model gagal"
    if "nan" in reason_lower or "missing" in reason_lower:
        return "Data mengandung nilai kosong"
    if "lstm" in reason_lower:
        return "Model LSTM gagal dilatih/diprediksi"
    if "garch" in reason_lower:
        return "Model GARCH gagal dilatih/diprediksi"
    if "xgboost" in reason_lower or "projector" in reason_lower:
        return "Model prediksi harga gagal"
    return "Error teknis saat analisis"


def render_analysis_summary(summary, show_analyzed=True):
    analyzed = summary.get("analyzed", [])
    failed = summary.get("failed", [])
    skipped = summary.get("skipped", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Berhasil Dianalisis", len(analyzed))
    col2.metric("Gagal", len(failed))
    col3.metric("Dilewati", len(skipped))

    if show_analyzed and analyzed:
        with st.expander("Lihat saham yang berhasil dianalisis"):
            st.dataframe(pd.DataFrame(analyzed), width="stretch")

    if skipped:
        with st.expander("Lihat saham yang dilewati karena sudah dianalisis"):
            st.dataframe(pd.DataFrame(skipped), width="stretch")

    if not failed:
        st.success("Tidak ada saham yang gagal dianalisis.")
        return

    failed_df = pd.DataFrame(failed)
    if "reason" not in failed_df.columns:
        failed_df["reason"] = "Alasan gagal tidak tercatat."
    if "ticker" not in failed_df.columns:
        failed_df["ticker"] = "-"

    failed_df["kategori_penyebab"] = failed_df["reason"].apply(classify_analysis_failure)

    st.warning(
        "Sebagian saham gagal dianalisis. Penyebab paling umum biasanya data historis kurang lengkap, "
        "file harga tidak bisa dimuat, atau model tertentu gagal dilatih untuk pola data saham tersebut."
    )

    cause_df = (
        failed_df["kategori_penyebab"]
        .value_counts()
        .rename_axis("Penyebab")
        .reset_index(name="Jumlah Saham")
    )
    st.subheader("Ringkasan Penyebab Gagal")
    st.dataframe(cause_df, width="stretch")

    with st.expander("Detail saham yang gagal dan alasan teknis"):
        detail_df = failed_df[["ticker", "kategori_penyebab", "reason"]].rename(columns={
            "ticker": "Saham",
            "kategori_penyebab": "Penyebab yang Mudah Dipahami",
            "reason": "Alasan Teknis",
        })
        st.dataframe(detail_df, width="stretch")


def create_live_analysis_tracker(title="Live Progress Analisis"):
    st.caption(f"**{title}**")
    progress_bar = st.progress(0.0)
    summary_box = st.empty()
    status_box = st.empty()
    with st.expander("Log proses analisis", expanded=False):
        log_box = st.empty()
    with st.expander("Saham gagal sementara", expanded=False):
        failure_box = st.empty()
    events = []
    failures = []

    def update(event):
        total = max(int(event.get("total") or 0), 1)
        completed = int(event.get("completed") or 0)
        analyzed_count = int(event.get("analyzed_count") or 0)
        failed_count = int(event.get("failed_count") or 0)
        skipped_count = int(event.get("skipped_count") or 0)
        percent = min(max(completed / total, 0.0), 1.0)

        message = event.get("message", "Analisis sedang berjalan...")
        stage = event.get("stage", "")
        ticker = event.get("ticker", "-")

        progress_bar.progress(percent)
        current_ticker = ticker if stage == "ticker_started" else "-"
        summary_box.caption(
            f"Progress: **{completed}/{total}** | "
            f"Selesai: **{analyzed_count}** | "
            f"Gagal: **{failed_count}** | "
            f"Dilewati: **{skipped_count}** | "
            f"Saat ini: **{current_ticker}**"
        )
        status_box.caption(message)

        events.append({
            "Waktu": datetime.now().strftime("%H:%M:%S"),
            "Ticker": ticker,
            "Status": stage.replace("_", " ").title(),
            "Pesan": message,
        })
        log_box.dataframe(pd.DataFrame(events[-8:]), width="stretch", hide_index=True)

        if stage == "ticker_failed":
            failures.append({
                "Saham": ticker,
                "Penyebab": classify_analysis_failure(event.get("reason", "")),
                "Alasan Teknis": event.get("reason", ""),
            })
            failure_box.dataframe(pd.DataFrame(failures), width="stretch", hide_index=True)
        elif stage == "done":
            status_box.caption(f"**{message}**")

    return update


def fetch_store_latest_sentiment(sentiment_path, ticker_code, query, limit, include_article_body=True):
    news_rows = fetch_google_news_sentiment_items(
        ticker=ticker_code,
        query=query,
        limit=int(limit),
        include_article_body=include_article_body,
    )
    if news_rows:
        append_issues(sentiment_path, news_rows)
    return news_rows


def is_valid_projection(projection):
    return (
        isinstance(projection, dict)
        and float(projection.get("projected_price") or 0.0) > 0.0
        and pd.notna(projection.get("projected_return_pct"))
    )


def normalize_ticker_code(ticker_code):
    return str(ticker_code or "").replace(".JK", "").upper().strip()


def load_prediction_log(pred_file=None):
    pred_file = pred_file or project_path("data", "tracking", "predictions_log.csv")
    if not os.path.exists(pred_file):
        return pd.DataFrame()
    if os.path.getsize(pred_file) == 0:
        return pd.DataFrame()

    try:
        df, _audit = audit_prediction_csv(pred_file)
    except EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    for column, default_value in {
        "ticker": "",
        "model_name": "",
        "current_date": "",
        "prediction_purpose": "MODEL_ACCURACY",
        "horizon_days": None,
        "is_active": True,
        "prediction_run_type": "FINAL",
        "model_version": "",
        "training_run_id": "",
        "trained_until_date": "",
        "prediction_mode": "TRAIN_AND_PREDICT",
    }.items():
        if column not in df.columns:
            df[column] = default_value

    df["ticker"] = df["ticker"].apply(normalize_ticker_code)
    df["model_name"] = df["model_name"].astype(str).str.strip()
    df["current_date"] = pd.to_datetime(df["current_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["prediction_purpose"] = df["prediction_purpose"].astype(str).str.upper().str.strip()
    df["prediction_run_type"] = df["prediction_run_type"].fillna("FINAL").astype(str).str.upper().str.strip()
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce")
    df["is_active"] = df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "status" not in df.columns:
        df["status"] = "PENDING"
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    if "superseded_at" in df.columns:
        superseded_mask = (
            (~df["is_active"])
            & (df["status"] == "PENDING")
            & df["superseded_at"].notna()
            & (df["superseded_at"].astype(str).str.strip() != "")
        )
        df.loc[superseded_mask, "status"] = "SUPERSEDED"
    return df


def summarize_prediction_status(prediction_purpose=None):
    pred_df = load_prediction_log()
    if pred_df.empty:
        return pd.DataFrame()
    if prediction_purpose is not None:
        pred_df = pred_df[pred_df["prediction_purpose"] == str(prediction_purpose).upper().strip()]
    if pred_df.empty:
        return pd.DataFrame()

    group_columns = ["prediction_purpose", "status", "is_active"]
    return (
        pred_df.groupby(group_columns)
        .size()
        .reset_index(name="jumlah_prediksi")
        .sort_values(group_columns)
    )


def has_core_prediction_for_data_date(ticker_code, current_date, run_type="FINAL"):
    pred_df = load_prediction_log()
    if pred_df.empty:
        return False
    clean_ticker = normalize_ticker_code(ticker_code)
    current_date_str = pd.to_datetime(current_date, errors="coerce").strftime("%Y-%m-%d")
    if current_date_str == "NaT":
        current_date_str = str(current_date)
    active_df = pred_df[
        (pred_df["ticker"] == clean_ticker)
        & (pred_df["current_date"] == current_date_str)
        & (pred_df["is_active"])
        & (pred_df["prediction_run_type"] == str(run_type).upper().strip())
    ].copy()
    if active_df.empty:
        return False

    h1_exists = (
        (active_df["model_name"] == "XGBoost")
        & (active_df["prediction_purpose"] == "NEXT_DAY_DIRECTION")
        & (active_df["horizon_days"].fillna(-1).astype(int) == 1)
    ).any()
    h3_exists = (
        (active_df["model_name"] == "XGBoost")
        & (active_df["prediction_purpose"] == "THREE_DAY_FORECAST")
        & (active_df["horizon_days"].fillna(-1).astype(int) == 3)
    ).any()
    return bool(h1_exists and h3_exists)


def build_backfill_history_view(selected_tickers=None):
    pred_df = load_prediction_log()
    if pred_df.empty or "prediction_run_type" not in pred_df.columns:
        return pd.DataFrame()

    history_df = pred_df[
        pred_df["prediction_run_type"].fillna("").astype(str).str.upper().str.strip() == "BACKFILL"
    ].copy()
    if history_df.empty:
        return pd.DataFrame()

    normalized_filter = [normalize_ticker_code(t) for t in (selected_tickers or []) if normalize_ticker_code(t)]
    if normalized_filter:
        history_df = history_df[history_df["ticker"].isin(normalized_filter)].copy()
    if history_df.empty:
        return pd.DataFrame()

    history_df["timestamp_prediction"] = pd.to_datetime(history_df.get("timestamp_prediction"), errors="coerce")
    history_df["current_date"] = pd.to_datetime(history_df.get("current_date"), errors="coerce").dt.strftime("%Y-%m-%d")
    history_df["target_date"] = pd.to_datetime(history_df.get("target_date"), errors="coerce").dt.strftime("%Y-%m-%d")
    history_df["predicted_return_pct"] = pd.to_numeric(history_df.get("predicted_return_pct"), errors="coerce")
    history_df["confidence_pct"] = pd.to_numeric(history_df.get("confidence_pct"), errors="coerce")
    history_df["horizon_days"] = pd.to_numeric(history_df.get("horizon_days"), errors="coerce")
    history_df["is_active"] = history_df["is_active"].astype(bool)

    display_cols = [
        "timestamp_prediction",
        "ticker",
        "model_name",
        "current_date",
        "target_date",
        "horizon_days",
        "prediction_purpose",
        "predicted_direction",
        "predicted_return_pct",
        "confidence_pct",
        "status",
        "is_active",
    ]
    existing_cols = [col for col in display_cols if col in history_df.columns]
    return history_df[existing_cols].sort_values(
        ["current_date", "ticker", "model_name", "horizon_days"],
        ascending=[False, True, True, True],
    )


def build_high_accuracy_prediction_view(leaderboard_df, prediction_purpose=None, min_accuracy=60.0, min_samples=3):
    pred_df = load_prediction_log()
    if pred_df.empty or leaderboard_df.empty:
        return pd.DataFrame()

    if prediction_purpose is not None:
        pred_df = pred_df[pred_df["prediction_purpose"] == str(prediction_purpose).upper().strip()]
    if pred_df.empty:
        return pd.DataFrame()
    pred_df = pred_df[pred_df["is_active"]].copy()
    if pred_df.empty:
        return pd.DataFrame()

    pred_df["timestamp_prediction"] = pd.to_datetime(pred_df["timestamp_prediction"], errors="coerce")
    pred_df["horizon_days"] = pd.to_numeric(pred_df.get("horizon_days"), errors="coerce")
    pred_df["confidence_pct"] = pd.to_numeric(pred_df.get("confidence_pct"), errors="coerce")
    pred_df["predicted_return_pct"] = pd.to_numeric(pred_df.get("predicted_return_pct"), errors="coerce")
    latest_pred_df = pred_df.sort_values("timestamp_prediction", ascending=False).drop_duplicates(
        subset=["ticker", "model_name", "prediction_purpose", "horizon_days"],
        keep="first",
    )

    qualified = leaderboard_df[
        (leaderboard_df["direction_accuracy_pct"] >= float(min_accuracy))
        & (leaderboard_df["total_evaluations"] >= int(min_samples))
    ].copy()
    if qualified.empty:
        return pd.DataFrame()

    view_df = latest_pred_df.merge(
        qualified,
        on=["ticker", "model_name"],
        how="inner",
        suffixes=("_prediction", "_history"),
    )
    if view_df.empty:
        return pd.DataFrame()

    view_df["prediksi"] = view_df["predicted_direction"].fillna("-")
    view_df["confidence_pct"] = view_df["confidence_pct"].fillna(view_df["direction_accuracy_pct"])
    display_cols = [
        "ticker",
        "model_name",
        "prediction_purpose",
        "horizon_days",
        "current_date",
        "target_date",
        "prediksi",
        "predicted_return_pct",
        "confidence_pct",
        "direction_accuracy_pct",
        "total_evaluations",
        "win_rate_pct",
        "profit_factor",
        "trading_score",
        "status",
    ]
    existing_cols = [col for col in display_cols if col in view_df.columns]
    return view_df[existing_cols].sort_values(
        ["trading_score", "direction_accuracy_pct", "confidence_pct"],
        ascending=[False, False, False],
    )


def build_analysis_completion_status(selected_tickers, required_models=None):
    required_models = required_models or ["XGBoost"]
    normalized_tickers = [normalize_ticker_code(t) for t in selected_tickers if normalize_ticker_code(t)]
    status_df = get_local_data_status(normalized_tickers)
    pred_df = load_prediction_log()
    rows = []

    for _, status_row in status_df.iterrows():
        ticker_code = normalize_ticker_code(status_row.get("ticker"))
        last_date = status_row.get("last_date")
        local_status = status_row.get("status", "-")
        rows_count = int(status_row.get("rows") or 0)
        available_models = []

        if ticker_code and last_date and not pred_df.empty:
            ticker_preds = pred_df[
                (pred_df["ticker"] == ticker_code)
                & (pred_df["current_date"] == str(last_date))
                & (pred_df["prediction_purpose"] == "THREE_DAY_FORECAST")
                & (pred_df["horizon_days"].fillna(3).astype(int) == 3)
                & (pred_df["is_active"])
            ]
            available_models = sorted(ticker_preds["model_name"].dropna().unique().tolist())

        missing_models = [model for model in required_models if model not in available_models]
        if local_status != "OK":
            analysis_status = "DATA BELUM SIAP"
            action_needed = "Perbaiki/update data harga dulu"
        elif missing_models:
            analysis_status = "BELUM SELESAI"
            action_needed = "Jalankan ulang analisis"
        else:
            analysis_status = "LENGKAP"
            action_needed = "Tidak ada"

        rows.append({
            "ticker": ticker_code,
            "last_date": last_date,
            "rows": rows_count,
            "status_data": local_status,
            "model_tersedia": ", ".join(available_models) if available_models else "-",
            "model_wajib_belum_ada": ", ".join(missing_models) if missing_models else "-",
            "status_analisis": analysis_status,
            "aksi": action_needed,
        })

    return pd.DataFrame(rows)


def build_final_prediction_workflow_status(selected_tickers, required_models=None):
    required_models = required_models or ["XGBoost"]
    normalized_tickers = [normalize_ticker_code(t) for t in selected_tickers if normalize_ticker_code(t)]
    if not normalized_tickers:
        return pd.DataFrame()

    completion_df = build_analysis_completion_status(normalized_tickers, required_models=required_models)
    pred_df = load_prediction_log()
    if not pred_df.empty:
        pred_df["target_dt"] = pd.to_datetime(pred_df.get("target_date"), errors="coerce")

    rows = []
    for _, row in completion_df.iterrows():
        ticker_code = normalize_ticker_code(row.get("ticker"))
        last_date = row.get("last_date")
        last_dt = pd.to_datetime(last_date, errors="coerce")
        ticker_preds = pred_df[pred_df["ticker"] == ticker_code].copy() if not pred_df.empty else pd.DataFrame()
        final_preds = pd.DataFrame()
        due_pending = pd.DataFrame()

        if not ticker_preds.empty:
            final_preds = ticker_preds[
                (ticker_preds["prediction_run_type"] == "FINAL")
                & (ticker_preds["is_active"])
            ].copy()
            if pd.notna(last_dt):
                due_pending = final_preds[
                    (final_preds["status"] == "PENDING")
                    & (final_preds["target_dt"].notna())
                    & (final_preds["target_dt"] <= last_dt)
                ].copy()

        today_final = pd.DataFrame()
        if not final_preds.empty and last_date:
            today_final = final_preds[final_preds["current_date"] == str(last_date)].copy()

        h1_ready = bool(
            not today_final.empty
            and (
                (today_final["prediction_purpose"] == "NEXT_DAY_DIRECTION")
                & (today_final["horizon_days"].fillna(1).astype(int) == 1)
            ).any()
        )
        h3_ready = bool(
            not today_final.empty
            and (
                (today_final["prediction_purpose"] == "THREE_DAY_FORECAST")
                & (today_final["horizon_days"].fillna(3).astype(int) == 3)
            ).any()
        )

        if row.get("status_data") != "OK":
            readiness = "DATA BELUM SIAP"
            action = "Update/import data lokal dulu"
        elif h1_ready and h3_ready:
            readiness = "FINAL SUDAH ADA"
            action = "Tidak perlu prediksi ulang"
        else:
            readiness = "SIAP PREDIKSI FINAL"
            action = "Jalankan prediksi FINAL dengan skip"

        rows.append({
            "Saham": ticker_code,
            "Tanggal Data": last_date or "-",
            "Status Data": row.get("status_data", "-"),
            "Status Analisis H+3": row.get("status_analisis", "-"),
            "FINAL H+1 Ada": "YA" if h1_ready else "BELUM",
            "FINAL H+3 Ada": "YA" if h3_ready else "BELUM",
            "Pending FINAL Jatuh Tempo": int(len(due_pending)),
            "Prediksi FINAL Aktif": int(len(final_preds)),
            "Kesiapan": readiness,
            "Aksi": action,
        })

    return pd.DataFrame(rows)


def _decision_label(action):
    label_map = {
        "BUY": "🟢 BUY",
        "WATCH": "🟡 WATCH",
        "AVOID": "🔴 AVOID",
        "DATA BELUM SIAP": "⚪ DATA BELUM SIAP",
        "MODEL BELUM TERPERCAYA": "🟠 MODEL BELUM TERPERCAYA",
    }
    return label_map.get(action, action)


def build_daily_decision_board(
    selected_tickers,
    min_reliability=55.0,
    min_confidence=55.0,
    min_evaluations=20,
    portfolio_capital=100_000_000,
    risk_per_trade_pct=1.0,
):
    def latest_market_breadth(tickers):
        rows = []
        for ticker_code in tickers:
            raw_path = project_path("data", "raw", f"{ticker_code}_raw.csv")
            if not os.path.exists(raw_path):
                continue
            try:
                price_df = pd.read_csv(raw_path)
            except Exception:
                continue
            close_col = "close" if "close" in price_df.columns else "Close" if "Close" in price_df.columns else None
            if close_col is None or len(price_df) < 2:
                continue
            closes = pd.to_numeric(price_df[close_col], errors="coerce").dropna()
            if len(closes) < 2:
                continue
            prev_close = float(closes.iloc[-2])
            last_close = float(closes.iloc[-1])
            if prev_close <= 0:
                continue
            rows.append((last_close / prev_close - 1.0) * 100.0)
        if not rows:
            return {
                "breadth_up_pct": 0.0,
                "avg_latest_return_pct": 0.0,
                "market_regime": "UNKNOWN",
                "sample_size": 0,
            }
        returns = pd.Series(rows)
        breadth_up_pct = float((returns > 0).mean() * 100.0)
        avg_latest_return_pct = float(returns.mean())
        if breadth_up_pct >= 60.0 and avg_latest_return_pct > 0:
            market_regime = "REBOUND"
        elif breadth_up_pct <= 40.0 and avg_latest_return_pct < 0:
            market_regime = "BEARISH"
        else:
            market_regime = "MIXED"
        return {
            "breadth_up_pct": round(breadth_up_pct, 2),
            "avg_latest_return_pct": round(avg_latest_return_pct, 2),
            "market_regime": market_regime,
            "sample_size": int(len(returns)),
        }

    normalized_tickers = [normalize_ticker_code(t) for t in selected_tickers if normalize_ticker_code(t)]
    market_breadth = latest_market_breadth(normalized_tickers)
    market_rebound = market_breadth["market_regime"] == "REBOUND"
    completion_df = build_analysis_completion_status(normalized_tickers, required_models=["XGBoost"])
    pred_df = load_prediction_log()
    reliability_df = get_best_model_recommendations(
        min_evaluations=int(min_evaluations),
        prediction_purpose="THREE_DAY_FORECAST",
    )
    reliability_lookup = {}
    sample_lookup = {}
    accuracy_lookup = {}
    if not reliability_df.empty:
        reliability_lookup = reliability_df.set_index(["ticker", "model_name"])["reliability_score"].to_dict()
        sample_lookup = reliability_df.set_index(["ticker", "model_name"])["total_evaluations"].to_dict()
        accuracy_lookup = reliability_df.set_index(["ticker", "model_name"])["direction_accuracy_pct"].to_dict()

    trust_audit_df = get_model_trust_audit(
        prediction_purpose="NEXT_DAY_DIRECTION",
        min_evaluations=int(min_evaluations),
        min_accuracy_pct=float(min_reliability),
    )
    trust_lookup = {}
    trust_reason_lookup = {}
    if not trust_audit_df.empty:
        direction_trust = trust_audit_df[
            trust_audit_df["model_name"].astype(str).str.upper().str.strip() == "DIRECTION-ENSEMBLE"
        ].copy()
        if direction_trust.empty:
            direction_trust = trust_audit_df.copy()
        direction_trust = direction_trust.sort_values(
            ["ticker", "trading_score", "direction_accuracy_pct"],
            ascending=[True, False, False],
        ).drop_duplicates(subset=["ticker"], keep="first")
        trust_lookup = direction_trust.set_index("ticker")["status_trust"].to_dict()
        trust_reason_lookup = direction_trust.set_index("ticker")["alasan"].to_dict()

    rows = []
    if completion_df.empty:
        return pd.DataFrame()

    active_final = pred_df[
        (pred_df["is_active"])
        & (pred_df["prediction_run_type"] == "FINAL")
    ].copy() if not pred_df.empty else pd.DataFrame()
    if not active_final.empty:
        active_final["timestamp_prediction"] = pd.to_datetime(active_final["timestamp_prediction"], errors="coerce")
        active_final["predicted_return_pct"] = pd.to_numeric(active_final.get("predicted_return_pct"), errors="coerce")
        active_final["confidence_pct"] = pd.to_numeric(active_final.get("confidence_pct"), errors="coerce")

    for _, row in completion_df.iterrows():
        ticker_code = row["ticker"]
        status_data = row["status_data"]
        status_analisis = row["status_analisis"]
        last_date = row["last_date"]
        action = "WATCH"
        reason_parts = []

        if status_data != "OK":
            action = "DATA BELUM SIAP"
            reason_parts.append("Data harga lokal belum OK.")
        elif status_analisis != "LENGKAP":
            action = "DATA BELUM SIAP"
            reason_parts.append("Prediksi FINAL Global Model belum lengkap.")

        h3_pred = pd.DataFrame()
        h1_pred = pd.DataFrame()
        if not active_final.empty:
            ticker_preds = active_final[active_final["ticker"] == ticker_code].copy()
            h3_pred = ticker_preds[
                (ticker_preds["model_name"] == "XGBoost")
                & (ticker_preds["prediction_purpose"] == "THREE_DAY_FORECAST")
                & (ticker_preds["horizon_days"].fillna(3).astype(int) == 3)
            ].sort_values("timestamp_prediction", ascending=False).head(1)
            h1_pred = ticker_preds[
                (ticker_preds["model_name"] == "Direction-Ensemble")
                & (ticker_preds["prediction_purpose"] == "NEXT_DAY_DIRECTION")
                & (ticker_preds["horizon_days"].fillna(1).astype(int) == 1)
            ].sort_values("timestamp_prediction", ascending=False).head(1)

        current_price = 0.0
        if not h3_pred.empty and "current_price" in h3_pred.columns:
            current_price = float(pd.to_numeric(h3_pred["current_price"].iloc[0], errors="coerce") or 0.0)
        elif not h1_pred.empty and "current_price" in h1_pred.columns:
            current_price = float(pd.to_numeric(h1_pred["current_price"].iloc[0], errors="coerce") or 0.0)

        projected_return = float(h3_pred["predicted_return_pct"].iloc[0]) if not h3_pred.empty and pd.notna(h3_pred["predicted_return_pct"].iloc[0]) else 0.0
        raw_confidence = float(h1_pred["confidence_pct"].iloc[0]) if not h1_pred.empty and pd.notna(h1_pred["confidence_pct"].iloc[0]) else 0.0
        confidence = raw_confidence
        direction = str(h1_pred["predicted_direction"].iloc[0]) if not h1_pred.empty and "predicted_direction" in h1_pred.columns else "-"
        reliability = float(reliability_lookup.get((ticker_code, "XGBoost"), 50.0))
        total_evaluations = int(sample_lookup.get((ticker_code, "XGBoost"), 0) or 0)
        direction_accuracy = float(accuracy_lookup.get((ticker_code, "XGBoost"), 0.0) or 0.0)
        trust_status = str(trust_lookup.get(ticker_code, "PERLU DATA LAGI"))
        trust_reason = str(trust_reason_lookup.get(ticker_code, "Track record H+1 belum cukup untuk menjadi acuan utama."))

        target_h3 = current_price * (1 + projected_return / 100.0) if current_price > 0 else 0.0
        entry_low = current_price * 0.995 if current_price > 0 else 0.0
        entry_high = current_price * 1.005 if current_price > 0 else 0.0
        stop_loss = current_price * 0.97 if current_price > 0 else 0.0
        risk_reward = ((target_h3 - current_price) / max(current_price - stop_loss, 1e-9)) if current_price > 0 else 0.0
        sizing = calculate_position_sizing(
            capital=float(portfolio_capital),
            entry_price=current_price,
            stop_loss=stop_loss,
            risk_pct=float(risk_per_trade_pct),
        )

        if action == "WATCH":
            if trust_status != "LAYAK DIPERCAYA":
                action = "MODEL BELUM TERPERCAYA"
                reason_parts.append(f"Trust audit: {trust_status.lower()}.")
            elif reliability < min_reliability or total_evaluations < int(min_evaluations):
                action = "MODEL BELUM TERPERCAYA"
                reason_parts.append("Track record per ticker/model belum cukup.")
            elif confidence < min_confidence:
                action = "WATCH"
                reason_parts.append(f"Confidence H+1 {confidence:.1f}% di bawah batas {float(min_confidence):.1f}%; sinyal dianggap netral.")
            elif direction == "TURUN" and market_rebound:
                action = "WATCH"
                reason_parts.append(
                    f"Filter market regime: mayoritas saham rebound ({market_breadth['breadth_up_pct']:.1f}% naik, "
                    f"avg {market_breadth['avg_latest_return_pct']:+.2f}%), jadi sinyal TURUN H+1 tidak langsung dipercaya."
                )
            elif projected_return >= 1.0 and confidence >= min_confidence and direction == "NAIK" and risk_reward >= 1.0:
                action = "BUY"
                reason_parts.append("Prioritas H+3: potensi positif, H+1 mendukung, confidence tinggi, dan track record lolos batas.")
            elif projected_return <= -1.0 or direction == "TURUN":
                action = "AVOID"
                reason_parts.append("Arah/potensi return belum mendukung entry.")
            else:
                action = "WATCH"
                reason_parts.append("Sinyal belum cukup kuat untuk entry.")

        risk_label = "Rendah"
        if abs(projected_return) < 1.0 or confidence < min_confidence:
            risk_label = "Sedang"
        if reliability < min_reliability or action in {"AVOID", "DATA BELUM SIAP"}:
            risk_label = "Tinggi"

        rows.append({
            "Saham": ticker_code,
            "Sinyal": _decision_label(action),
            "Harga Terakhir": current_price,
            "Entry Area": f"{entry_low:,.0f} - {entry_high:,.0f}" if current_price > 0 else "-",
            "Stop Loss": stop_loss,
            "Target H+3": target_h3,
            "Risk/Reward": risk_reward,
            "Lot Maks": int(sizing.get("lots", 0)),
            "Nilai Posisi": float(sizing.get("position_value", 0.0)),
            "Potensi H+3": projected_return,
            "Arah H+1": direction,
            "Confidence": confidence,
            "Reliability": reliability,
            "Evaluasi": total_evaluations,
            "Akurasi Arah": direction_accuracy,
            "Regime Pasar": market_breadth["market_regime"],
            "Breadth Naik": market_breadth["breadth_up_pct"],
            "Trust Model": trust_status,
            "Risiko": risk_label,
            "Status Data": status_data,
            "Status Analisis": status_analisis,
            "Tanggal Data": last_date,
            "Alasan Utama": " ".join(reason_parts),
            "Catatan Trust": trust_reason,
        })

    board_df = pd.DataFrame(rows)
    if board_df.empty:
        return board_df
    action_order = {
        "🟢 BUY": 0,
        "🟡 WATCH": 1,
        "🟠 MODEL BELUM TERPERCAYA": 2,
        "⚪ DATA BELUM SIAP": 3,
        "🔴 AVOID": 4,
    }
    board_df["_sort"] = board_df["Sinyal"].map(action_order).fillna(9)
    return board_df.sort_values(["_sort", "Potensi H+3", "Risk/Reward", "Confidence"], ascending=[True, False, False, False]).drop(columns="_sort")


def render_daily_checklist(board_df, job_rows):
    if board_df.empty:
        st.info("Belum ada data untuk checklist harian.")
        return

    latest_date = board_df["Tanggal Data"].dropna().astype(str).max()
    unfinished_count = int((board_df["Status Analisis"] != "LENGKAP").sum())
    data_not_ready_count = int((board_df["Status Data"] != "OK").sum())
    buy_count = int(board_df["Sinyal"].astype(str).str.contains("BUY", regex=False).sum())
    stale_job_count = sum(1 for job in job_rows if job.get("status") in ["RUNNING", "STALE", "FAILED", "UNKNOWN"])

    checklist = pd.DataFrame([
        {"Checklist": "Data harga terbaru tersedia", "Status": "OK" if data_not_ready_count == 0 else "PERLU CEK", "Catatan": f"Tanggal data terbaru: {latest_date}"},
        {"Checklist": "Prediksi FINAL Global Model tersedia", "Status": "OK" if bool(list_ticker_models("GLOBAL")) else "PERLU CEK", "Catatan": "Gunakan Global Model untuk prediksi baru; log lama tetap untuk evaluasi."},
        {"Checklist": "Tidak ada job background bermasalah", "Status": "OK" if stale_job_count == 0 else "PERLU CEK", "Catatan": f"{stale_job_count} job perlu dicek"},
        {"Checklist": "Ada kandidat entry", "Status": "OK" if buy_count > 0 else "PERLU CEK", "Catatan": f"{buy_count} kandidat BUY"},
        {"Checklist": "Gunakan risk management", "Status": "WAJIB", "Catatan": "Validasi lot, stop loss, dan risk/reward sebelum order"},
    ])
    st.dataframe(checklist, width="stretch", hide_index=True)


def render_feature_status(title, status, detail="", action_hint=""):
    status_text = str(status or "INFO").upper()
    message = f"**{title}: {status_text}**"
    if detail:
        message += f" - {detail}"
    if action_hint:
        message += f"  \nLangkah berikutnya: {action_hint}"

    if status_text in ["SIAP", "OK", "SINKRON", "SIAP PREDIKSI HARIAN"]:
        st.success(message)
    elif status_text in ["PERLU CEK", "PERLU UPDATE", "BELUM LENGKAP", "PERLU RETRAIN", "DATA BELUM UPDATE", "MODEL BELUM TERSEDIA", "JOB MASIH BERJALAN"]:
        st.warning(message)
    elif status_text in ["ERROR", "GAGAL"]:
        st.error(message)
    else:
        st.info(message)


def build_user_friendly_readiness(selected_tickers, training_policy=None, store_status=None, job_rows=None):
    selected_tickers = [normalize_ticker_code(t) for t in (selected_tickers or []) if normalize_ticker_code(t)]
    job_rows = job_rows or []
    training_policy = training_policy or evaluate_training_policy()
    store_status = store_status or model_store_status(selected_tickers)

    status_df = get_local_data_status(selected_tickers) if selected_tickers else pd.DataFrame()
    data_ok = bool(not status_df.empty and int((status_df["status"] == "OK").sum()) == len(status_df))
    latest_data = "-"
    if not status_df.empty:
        latest_dates = status_df.loc[status_df["status"] == "OK", "last_date"].dropna().astype(str)
        if not latest_dates.empty:
            latest_data = latest_dates.max()

    running_jobs = [job for job in job_rows if job.get("status") == "RUNNING"]
    troubled_jobs = [job for job in job_rows if job.get("status") in ["STALE", "FAILED", "UNKNOWN"]]
    has_models = bool(list_ticker_models("GLOBAL")) if not LEGACY_MODELS_ENABLED else int(store_status.get("total_artifacts") or 0) > 0
    retrain_due = bool(training_policy.get("retrain_due"))

    if running_jobs:
        status = "JOB MASIH BERJALAN"
        action = "Pantau job background sampai selesai sebelum memulai proses baru."
        next_step = "Pantau Job"
    elif not data_ok:
        status = "DATA BELUM UPDATE"
        action = "Update harga atau import CSV lokal sebelum membuat prediksi."
        next_step = "Update Harga"
    elif not has_models:
        status = "MODEL BELUM TERSEDIA"
        action = "Jalankan training Global Model dari VS Code untuk membuat artifact GLOBAL."
        next_step = "Training Awal"
    else:
        status = "SIAP PREDIKSI HARIAN" if not retrain_due else "SIAP PREDIKSI - RETRAIN TERJADWAL"
        action = (
            "Gunakan prediksi harian dari Global Model dulu. "
            + (training_policy.get("reason", "Retrain dapat dijalankan setelah prediksi harian selesai.") if retrain_due else "Retrain belum diperlukan.")
        )
        next_step = "Prediksi Model Tersimpan"

    return {
        "status": status,
        "action": action,
        "next_step": next_step,
        "latest_data": latest_data,
        "data_ok": data_ok,
        "has_models": has_models,
        "retrain_due": retrain_due,
        "running_jobs": len(running_jobs),
        "troubled_jobs": len(troubled_jobs),
        "model_artifacts": int(store_status.get("total_artifacts") or 0),
    }


def render_readiness_panel(readiness):
    detail = (
        f"Tanggal data: {readiness.get('latest_data', '-')} | "
        f"Artifact model: {readiness.get('model_artifacts', 0)} | "
        f"Job berjalan: {readiness.get('running_jobs', 0)} | "
        f"Job perlu cek: {readiness.get('troubled_jobs', 0)}"
    )
    render_feature_status(
        "Kesiapan Trading",
        readiness.get("status", "INFO"),
        detail,
        readiness.get("action", ""),
    )


def has_running_analysis_job(job_rows=None):
    return any(job.get("status") == "RUNNING" for job in (job_rows or list_analysis_jobs(limit=10)))


def build_ideal_workflow_plan(training_policy, store_status, has_running_job=False):
    has_models = bool(list_ticker_models("GLOBAL")) if not LEGACY_MODELS_ENABLED else int(store_status.get("total_artifacts") or 0) > 0
    retrain_due = bool(training_policy.get("retrain_due"))
    if has_running_job:
        retrain_action = "TUNGGU_JOB"
        retrain_note = "Ada job background yang masih berjalan. Proses baru dikunci agar tidak tumpang tindih."
    elif not has_models:
        retrain_action = "TRAINING_AWAL"
        retrain_note = "Model tersimpan belum tersedia untuk cakupan ini."
    elif retrain_due:
        retrain_action = "RETRAIN_TERJADWAL"
        retrain_note = training_policy.get("reason", "Policy menandai retrain perlu dijalankan.")
    else:
        retrain_action = "LEWATI_RETRAIN"
        retrain_note = training_policy.get("reason", "Model tersimpan masih layak dipakai.")

    return [
        {"Urutan": 1, "Langkah": "Update data", "Aksi": "JALANKAN", "Catatan": "Ambil/rapikan OHLCV terbaru sebelum evaluasi."},
        {"Urutan": 2, "Langkah": "Evaluasi prediksi pending", "Aksi": "JALANKAN", "Catatan": "Prediksi lama tetap menjadi bahan akurasi."},
        {"Urutan": 3, "Langkah": "Prediksi saved model", "Aksi": "JALANKAN" if has_models else "MENUNGGU_MODEL", "Catatan": "Prediksi harian memakai artifact, bukan training ulang."},
        {"Urutan": 4, "Langkah": "Retrain", "Aksi": retrain_action, "Catatan": retrain_note},
    ]


def get_adaptive_min_evaluation_default(
    prediction_purpose="NEXT_DAY_DIRECTION",
    model_name=None,
    accuracy_file=None,
):
    accuracy_file = accuracy_file or project_path("data", "tracking", "accuracy_log.csv")
    if not os.path.exists(accuracy_file) or os.path.getsize(accuracy_file) == 0:
        return 20
    try:
        acc_df = pd.read_csv(accuracy_file)
    except Exception:
        return 20
    if acc_df.empty:
        return 20
    if "prediction_purpose" in acc_df.columns:
        acc_df = acc_df[acc_df["prediction_purpose"].astype(str).str.upper().str.strip() == prediction_purpose]
    if model_name and "model_name" in acc_df.columns:
        acc_df = acc_df[acc_df["model_name"].astype(str).str.upper().str.strip() == model_name.upper()]
    if "prediction_run_type" in acc_df.columns:
        acc_df = acc_df[acc_df["prediction_run_type"].fillna("FINAL").astype(str).str.upper().str.strip() == "FINAL"]
    if acc_df.empty:
        evaluated_count = 0
    elif model_name:
        evaluated_count = len(acc_df)
    elif "model_name" in acc_df.columns:
        model_counts = acc_df["model_name"].astype(str).str.strip().value_counts()
        evaluated_count = int(model_counts.median()) if not model_counts.empty else 0
    else:
        evaluated_count = len(acc_df)
    if evaluated_count >= 150:
        return 100
    if evaluated_count >= 75:
        return 50
    return 20


def build_model_evaluation_sample_status(
    prediction_purpose="NEXT_DAY_DIRECTION",
    accuracy_file=None,
):
    accuracy_file = accuracy_file or project_path("data", "tracking", "accuracy_log.csv")
    if not os.path.exists(accuracy_file) or os.path.getsize(accuracy_file) == 0:
        return pd.DataFrame()
    try:
        acc_df = pd.read_csv(accuracy_file)
    except Exception:
        return pd.DataFrame()
    if acc_df.empty:
        return pd.DataFrame()
    if "prediction_purpose" in acc_df.columns:
        acc_df = acc_df[acc_df["prediction_purpose"].astype(str).str.upper().str.strip() == prediction_purpose]
    if "prediction_run_type" in acc_df.columns:
        acc_df = acc_df[acc_df["prediction_run_type"].fillna("FINAL").astype(str).str.upper().str.strip() == "FINAL"]
    if acc_df.empty or "model_name" not in acc_df.columns or "direction_correct" not in acc_df.columns:
        return pd.DataFrame()
    acc_df["direction_correct"] = acc_df["direction_correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    summary = (
        acc_df.groupby("model_name")
        .agg(
            jumlah_evaluasi=("direction_correct", "count"),
            akurasi_arah_pct=("direction_correct", lambda x: x.mean() * 100),
        )
        .reset_index()
    )

    def sample_default(count):
        if count >= 150:
            return 100
        if count >= 75:
            return 50
        return 20

    def sample_status(count):
        if count >= 150:
            return "KUAT"
        if count >= 75:
            return "SEDANG"
        return "RENDAH"

    summary["default_min_sampel"] = summary["jumlah_evaluasi"].apply(sample_default)
    summary["status_sampel"] = summary["jumlah_evaluasi"].apply(sample_status)
    summary["akurasi_arah_pct"] = summary["akurasi_arah_pct"].round(2)
    return summary.sort_values(["jumlah_evaluasi", "akurasi_arah_pct"], ascending=[False, False]).reset_index(drop=True)


def build_system_sync_status(selected_tickers, required_models=None):
    required_models = required_models or ["XGBoost"]
    normalized_tickers = [normalize_ticker_code(t) for t in selected_tickers if normalize_ticker_code(t)]
    if not normalized_tickers:
        return pd.DataFrame(), pd.DataFrame()

    data_status_df = get_local_data_status(normalized_tickers)
    pred_df = load_prediction_log()
    accuracy_path = project_path("data", "tracking", "accuracy_log.csv")
    try:
        accuracy_df = (
            pd.read_csv(accuracy_path)
            if os.path.exists(accuracy_path) and os.path.getsize(accuracy_path) > 0
            else pd.DataFrame()
        )
    except EmptyDataError:
        accuracy_df = pd.DataFrame()
    if not accuracy_df.empty:
        accuracy_df["ticker"] = accuracy_df["ticker"].apply(normalize_ticker_code)
        accuracy_df["prediction_purpose"] = (
            accuracy_df.get("prediction_purpose", "MODEL_ACCURACY")
            .fillna("MODEL_ACCURACY")
            .astype(str)
            .str.upper()
            .str.strip()
        )

    rows = []
    for _, status_row in data_status_df.iterrows():
        ticker_code = normalize_ticker_code(status_row.get("ticker"))
        last_date = status_row.get("last_date")
        local_status = status_row.get("status", "-")
        ticker_preds = pred_df[pred_df["ticker"] == ticker_code].copy() if not pred_df.empty else pd.DataFrame()
        active_preds = ticker_preds[ticker_preds["is_active"]].copy() if not ticker_preds.empty else pd.DataFrame()

        latest_active_date = "-"
        if not active_preds.empty:
            active_dates = pd.to_datetime(active_preds["current_date"], errors="coerce").dropna()
            if not active_dates.empty:
                latest_active_date = active_dates.max().strftime("%Y-%m-%d")

        pending_count = int((ticker_preds["status"] == "PENDING").sum()) if not ticker_preds.empty else 0
        evaluated_count = int((ticker_preds["status"] == "EVALUATED").sum()) if not ticker_preds.empty else 0
        inactive_count = int((~ticker_preds["is_active"]).sum()) if not ticker_preds.empty else 0
        backfill_count = int((ticker_preds["prediction_run_type"] == "BACKFILL").sum()) if not ticker_preds.empty else 0
        backfill_pending_count = int(((ticker_preds["prediction_run_type"] == "BACKFILL") & (ticker_preds["status"] == "PENDING")).sum()) if not ticker_preds.empty else 0

        latest_horizons = "-"
        active_latest = pd.DataFrame()
        if not active_preds.empty and last_date:
            active_latest = active_preds[active_preds["current_date"] == str(last_date)].copy()
            if not active_latest.empty:
                latest_horizons = ", ".join(
                    f"H+{int(h)}"
                    for h in sorted(active_latest["horizon_days"].dropna().astype(int).unique().tolist())
                )

        missing_models = required_models.copy()
        if not active_latest.empty:
            h3_latest = active_latest[
                (active_latest["prediction_purpose"] == "THREE_DAY_FORECAST")
                & (active_latest["horizon_days"].fillna(3).astype(int) == 3)
            ]
            available_models = sorted(h3_latest["model_name"].dropna().unique().tolist())
            missing_models = [model for model in required_models if model not in available_models]

        ticker_acc = accuracy_df[accuracy_df["ticker"] == ticker_code] if not accuracy_df.empty else pd.DataFrame()
        evaluated_h1 = int((ticker_acc["prediction_purpose"] == "NEXT_DAY_DIRECTION").sum()) if not ticker_acc.empty else 0
        evaluated_h3 = int((ticker_acc["prediction_purpose"] == "THREE_DAY_FORECAST").sum()) if not ticker_acc.empty else 0

        issues = []
        if local_status != "OK":
            issues.append("Data harga belum OK")
        if local_status == "OK" and last_date and missing_models:
            issues.append("Prediksi aktif H+3 belum lengkap")
        if pending_count:
            issues.append("Ada prediksi pending")
        if backfill_pending_count:
            issues.append("Backfill belum dievaluasi")
        sync_status = "SINKRON" if not issues else "PERLU CEK"

        rows.append({
            "ticker": ticker_code,
            "status_sinkron": sync_status,
            "catatan": "; ".join(issues) if issues else "OK",
            "status_data": local_status,
            "tanggal_data_terakhir": last_date,
            "tanggal_prediksi_aktif_terbaru": latest_active_date,
            "horizon_ranking_aktif_tanggal_terakhir": latest_horizons,
            "model_h3_wajib_belum_ada": ", ".join(missing_models) if missing_models else "-",
            "prediksi_pending": pending_count,
            "prediksi_evaluated": evaluated_count,
            "prediksi_nonaktif": inactive_count,
            "backfill_total": backfill_count,
            "backfill_pending": backfill_pending_count,
            "evaluasi_h1": evaluated_h1,
            "evaluasi_h3": evaluated_h3,
        })

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = [
        {"metrik": "Total saham dicek", "nilai": len(detail_df)},
        {"metrik": "Data harga OK", "nilai": int((detail_df["status_data"] == "OK").sum())},
        {"metrik": "Saham sinkron", "nilai": int((detail_df["status_sinkron"] == "SINKRON").sum())},
        {"metrik": "Saham perlu cek", "nilai": int((detail_df["status_sinkron"] == "PERLU CEK").sum())},
        {"metrik": "Total prediksi pending", "nilai": int(detail_df["prediksi_pending"].sum())},
        {"metrik": "Total backfill pending", "nilai": int(detail_df["backfill_pending"].sum())},
        {"metrik": "Total evaluasi H+1", "nilai": int(detail_df["evaluasi_h1"].sum())},
        {"metrik": "Total evaluasi H+3", "nilai": int(detail_df["evaluasi_h3"].sum())},
    ]
    return pd.DataFrame(summary_rows), detail_df.sort_values(["status_sinkron", "ticker"], ascending=[False, True])


def audit_and_repair_local_ohlc(selected_tickers, data_dir="data/raw", apply_repair=False):
    rows = []
    for ticker_code in selected_tickers:
        clean_ticker = normalize_ticker_code(ticker_code)
        if not clean_ticker:
            continue

        file_path = os.path.join(data_dir, f"{clean_ticker}_raw.csv")
        if not os.path.exists(file_path):
            rows.append({
                "ticker": clean_ticker,
                "status": "FILE TIDAK ADA",
                "invalid_high_rows": 0,
                "invalid_low_rows": 0,
                "repaired": False,
                "note": "Jalankan update data atau upload CSV manual.",
            })
            continue

        try:
            df = _normalize_existing_data(pd.read_csv(file_path))
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
            invalid_low = df["low"] > df[["open", "close"]].min(axis=1)
            invalid_count = int(invalid_high.sum() + invalid_low.sum())

            if invalid_count and apply_repair:
                df.loc[invalid_high, "high"] = df.loc[invalid_high, ["open", "high", "close"]].max(axis=1)
                df.loc[invalid_low, "low"] = df.loc[invalid_low, ["open", "low", "close"]].min(axis=1)
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
                df.to_csv(file_path, index=False)

            rows.append({
                "ticker": clean_ticker,
                "status": "PERLU DIPERBAIKI" if invalid_count else "OK",
                "invalid_high_rows": int(invalid_high.sum()),
                "invalid_low_rows": int(invalid_low.sum()),
                "repaired": bool(invalid_count and apply_repair),
                "note": "OHLC diperbaiki dan CSV lokal disimpan." if invalid_count and apply_repair else "-",
            })
        except Exception as e:
            rows.append({
                "ticker": clean_ticker,
                "status": "ERROR",
                "invalid_high_rows": 0,
                "invalid_low_rows": 0,
                "repaired": False,
                "note": str(e),
            })

    return pd.DataFrame(rows)


def describe_feature_name(feature_name):
    labels = {
        "open": "Harga pembukaan",
        "high": "Harga tertinggi",
        "low": "Harga terendah",
        "close": "Harga penutupan",
        "volume": "Volume transaksi",
        "feat_rsi_14": "RSI 14 hari",
        "feat_atr_14": "ATR 14 hari",
        "feat_macd": "MACD",
        "feat_macd_signal": "Sinyal MACD",
        "feat_macd_diff": "Selisih MACD",
    }
    return labels.get(feature_name, feature_name.replace("feat_", "").replace("_", " ").title())


def build_xgboost_explanation(projector, features_df, max_display=12):
    if projector is None or getattr(projector, "model", None) is None:
        return pd.DataFrame(), "Model XGBoost belum tersedia."

    X_all = projector._prepare_data(features_df, is_training=False)
    if X_all.empty:
        return pd.DataFrame(), "Data fitur kosong."

    latest_features = X_all.iloc[[-1]]
    latest_values = latest_features.iloc[0]

    try:
        import shap

        explainer = shap.TreeExplainer(projector.model)
        shap_values = explainer.shap_values(latest_features)
        values = shap_values[0] if getattr(shap_values, "ndim", 1) > 1 else shap_values
        explanation_df = pd.DataFrame({
            "Fitur": latest_features.columns,
            "Nama Mudah": [describe_feature_name(col) for col in latest_features.columns],
            "Nilai Saat Ini": latest_values.values,
            "Kontribusi SHAP": values,
        })
        explanation_df["Arah Pengaruh"] = explanation_df["Kontribusi SHAP"].apply(
            lambda value: "Mendorong prediksi naik" if value > 0 else "Mendorong prediksi turun" if value < 0 else "Netral"
        )
        explanation_df["Abs Kontribusi"] = explanation_df["Kontribusi SHAP"].abs()
        explanation_df = explanation_df.sort_values("Abs Kontribusi", ascending=False).head(max_display)
        return explanation_df.drop(columns=["Abs Kontribusi"]), "SHAP aktif"
    except Exception as e:
        importances = getattr(projector.model, "feature_importances_", None)
        if importances is None:
            return pd.DataFrame(), f"SHAP belum dapat dihitung: {e}"

        explanation_df = pd.DataFrame({
            "Fitur": latest_features.columns,
            "Nama Mudah": [describe_feature_name(col) for col in latest_features.columns],
            "Nilai Saat Ini": latest_values.values,
            "Importance": importances,
        })
        explanation_df = explanation_df.sort_values("Importance", ascending=False).head(max_display)
        return explanation_df, f"SHAP belum tersedia, memakai feature importance XGBoost. Detail: {e}"


def get_job_status_path(job_id):
    return os.path.join(JOB_DIR, f"analysis_{job_id}.json")


def read_analysis_job_status(job_id):
    if not job_id:
        return None
    path = get_job_status_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            status = json.load(f)
    except Exception as e:
        return {"status": "UNKNOWN", "message": f"Gagal membaca status job: {e}", "job_id": job_id}

    if status.get("status") == "RUNNING":
        updated_at = pd.to_datetime(status.get("updated_at"), errors="coerce")
        if pd.notna(updated_at) and datetime.now() - updated_at.to_pydatetime() > timedelta(hours=2):
            status = dict(status)
            status["status"] = "STALE"
            status["message"] = (
                "Status job tampak macet: tidak ada update lebih dari 2 jam. "
                "Jalankan ulang analisis untuk ticker yang belum selesai."
            )
    return status


def list_analysis_jobs(limit=10):
    if not os.path.exists(JOB_DIR):
        return []

    jobs = []
    for filename in os.listdir(JOB_DIR):
        if not filename.startswith("analysis_") or not filename.endswith(".json"):
            continue
        job_id = filename.removeprefix("analysis_").removesuffix(".json")
        status = read_analysis_job_status(job_id) or {}
        jobs.append({
            "job_id": job_id,
            "status": status.get("status", "UNKNOWN"),
            "updated_at": status.get("updated_at", "-"),
            "message": status.get("message", "-"),
            "path": os.path.join(JOB_DIR, filename),
        })

    jobs.sort(key=lambda row: os.path.getmtime(row["path"]) if os.path.exists(row["path"]) else 0, reverse=True)
    return jobs[:limit]


def build_background_ticker_status_df(status):
    tickers_in_job = [normalize_ticker_code(t) for t in status.get("tickers", []) if normalize_ticker_code(t)]
    ticker_statuses = status.get("ticker_statuses") or {}
    rows_by_ticker = {}

    for index, ticker_code in enumerate(tickers_in_job, start=1):
        rows_by_ticker[ticker_code] = {
            "No": index,
            "Saham": ticker_code,
            "Status": "MENUNGGU",
            "Tahap": "pending",
            "Pesan": "Menunggu giliran analisis.",
            "Mulai": "-",
            "Selesai": "-",
            "Penyebab Gagal": "",
        }

    for ticker_code, ticker_status in ticker_statuses.items():
        clean_ticker = normalize_ticker_code(ticker_code)
        if not clean_ticker:
            continue
        rows_by_ticker.setdefault(clean_ticker, {
            "No": len(rows_by_ticker) + 1,
            "Saham": clean_ticker,
            "Status": "MENUNGGU",
            "Tahap": "pending",
            "Pesan": "Menunggu giliran analisis.",
            "Mulai": "-",
            "Selesai": "-",
            "Penyebab Gagal": "",
        })
        rows_by_ticker[clean_ticker].update({
            "Status": ticker_status.get("status") or rows_by_ticker[clean_ticker]["Status"],
            "Tahap": ticker_status.get("stage") or rows_by_ticker[clean_ticker]["Tahap"],
            "Pesan": ticker_status.get("message") or rows_by_ticker[clean_ticker]["Pesan"],
            "Mulai": ticker_status.get("started_at") or "-",
            "Selesai": ticker_status.get("finished_at") or "-",
            "Penyebab Gagal": ticker_status.get("reason") or "",
        })

    for event in status.get("events", []):
        ticker_code = normalize_ticker_code(event.get("ticker"))
        if not ticker_code or ticker_code == "-":
            continue
        rows_by_ticker.setdefault(ticker_code, {
            "No": len(rows_by_ticker) + 1,
            "Saham": ticker_code,
            "Status": "MENUNGGU",
            "Tahap": "pending",
            "Pesan": "Menunggu giliran analisis.",
            "Mulai": "-",
            "Selesai": "-",
            "Penyebab Gagal": "",
        })
        stage = event.get("stage", "")
        if stage == "ticker_started":
            rows_by_ticker[ticker_code]["Status"] = "BERJALAN"
            rows_by_ticker[ticker_code]["Mulai"] = event.get("time", "-")
        elif stage == "ticker_succeeded":
            rows_by_ticker[ticker_code]["Status"] = "SELESAI"
            rows_by_ticker[ticker_code]["Selesai"] = event.get("time", "-")
        elif stage == "ticker_skipped":
            rows_by_ticker[ticker_code]["Status"] = "DILEWATI"
            rows_by_ticker[ticker_code]["Selesai"] = event.get("time", "-")
        elif stage == "ticker_failed":
            rows_by_ticker[ticker_code]["Status"] = "GAGAL"
            rows_by_ticker[ticker_code]["Selesai"] = event.get("time", "-")
        rows_by_ticker[ticker_code]["Tahap"] = stage or rows_by_ticker[ticker_code]["Tahap"]
        rows_by_ticker[ticker_code]["Pesan"] = event.get("message") or rows_by_ticker[ticker_code]["Pesan"]

    summary = status.get("summary") or {}
    for row in summary.get("analyzed", []):
        ticker_code = normalize_ticker_code(row.get("ticker"))
        if ticker_code in rows_by_ticker:
            rows_by_ticker[ticker_code]["Status"] = "SELESAI"
            rows_by_ticker[ticker_code]["Tahap"] = "ticker_succeeded"
    for row in summary.get("failed", []):
        ticker_code = normalize_ticker_code(row.get("ticker"))
        if ticker_code in rows_by_ticker:
            rows_by_ticker[ticker_code]["Status"] = "GAGAL"
            rows_by_ticker[ticker_code]["Tahap"] = "ticker_failed"
            rows_by_ticker[ticker_code]["Penyebab Gagal"] = row.get("reason", "")
    for row in summary.get("skipped", []):
        ticker_code = normalize_ticker_code(row.get("ticker"))
        if ticker_code in rows_by_ticker:
            rows_by_ticker[ticker_code]["Status"] = "DILEWATI"
            rows_by_ticker[ticker_code]["Tahap"] = "ticker_skipped"
            rows_by_ticker[ticker_code]["Penyebab Gagal"] = row.get("reason", "")

    if not rows_by_ticker:
        return pd.DataFrame()
    return pd.DataFrame(sorted(rows_by_ticker.values(), key=lambda row: row["No"]))


def render_background_ticker_process(status, job_id, key_suffix=None):
    process_df = build_background_ticker_status_df(status)
    if process_df.empty:
        return

    status_counts = process_df["Status"].value_counts()
    c_waiting, c_running, c_done, c_skipped, c_failed = st.columns(5)
    c_waiting.metric("Menunggu", int(status_counts.get("MENUNGGU", 0)))
    c_running.metric("Berjalan", int(status_counts.get("BERJALAN", 0)))
    c_done.metric("Selesai", int(status_counts.get("SELESAI", 0)))
    c_skipped.metric("Dilewati", int(status_counts.get("DILEWATI", 0)))
    c_failed.metric("Gagal", int(status_counts.get("GAGAL", 0)))

    with st.expander("Proses per saham", expanded=status.get("status") == "RUNNING"):
        widget_key_suffix = key_suffix or job_id
        filter_options = ["Semua"] + sorted(process_df["Status"].dropna().unique().tolist())
        selected_status = st.selectbox(
            "Filter status saham",
            options=filter_options,
            key=f"bg_process_status_filter_{widget_key_suffix}",
        )
        search_text = st.text_input(
            "Cari kode saham",
            value="",
            key=f"bg_process_search_{widget_key_suffix}",
            placeholder="Contoh: BBRI",
        ).strip().upper()

        display_df = process_df.copy()
        if selected_status != "Semua":
            display_df = display_df[display_df["Status"] == selected_status]
        if search_text:
            display_df = display_df[display_df["Saham"].str.contains(search_text, case=False, na=False)]

        st.dataframe(display_df, width="stretch", hide_index=True)

        failed_df = process_df[process_df["Status"] == "GAGAL"].copy()
        if not failed_df.empty:
            failed_df["Kategori"] = failed_df["Penyebab Gagal"].apply(classify_analysis_failure)
            st.warning(f"{len(failed_df)} saham gagal dianalisis. Lihat detail penyebab di bawah.")
            st.dataframe(
                failed_df[["Saham", "Kategori", "Penyebab Gagal", "Pesan"]],
                width="stretch",
                hide_index=True,
            )


def start_background_analysis_job(job_tickers, lstm_epochs=3, duplicate_policy="skip", prediction_run_type=None, force_retrain=False):
    if not LEGACY_MODELS_ENABLED:
        raise RuntimeError("Model lama per-saham sedang dinonaktifkan. Gunakan Global Model untuk prediksi/training baru.")
    normalized = [normalize_ticker_code(t) for t in job_tickers if normalize_ticker_code(t)]
    if not normalized:
        raise ValueError("Tidak ada ticker valid untuk dianalisis.")

    os.makedirs(JOB_DIR, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    command = [
        sys.executable,
        os.path.join("scripts", "background_analysis_job.py"),
        "--job-id",
        job_id,
        "--tickers",
        ",".join(normalized),
        "--epochs",
        str(int(lstm_epochs)),
        "--duplicate-policy",
        str(duplicate_policy),
    ]
    if prediction_run_type:
        command.extend(["--run-type", str(prediction_run_type)])
    if force_retrain:
        command.append("--force-retrain")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen(command, cwd=os.getcwd(), creationflags=creationflags)
    st.session_state["active_analysis_job_id"] = job_id
    return job_id


def render_background_analysis_job(job_id):
    st.session_state["background_job_render_count"] = st.session_state.get("background_job_render_count", 0) + 1
    render_key_suffix = f"{job_id}_{st.session_state['background_job_render_count']}"
    status = read_analysis_job_status(job_id)
    if not status:
        st.info("Belum ada status job background yang bisa dibaca.")
        return

    job_status = status.get("status", "UNKNOWN")
    total = max(int(status.get("total") or 0), 1)
    completed = int(status.get("completed") or 0)
    progress = min(max(completed / total, 0.0), 1.0)

    st.caption(f"Job ID: `{job_id}` | Status: **{job_status}** | Update: {status.get('updated_at', '-')}")
    st.progress(progress)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Progress", f"{completed}/{total}")
    c2.metric("Berhasil", int(status.get("analyzed_count") or 0))
    c3.metric("Gagal", int(status.get("failed_count") or 0))
    c4.metric("Dilewati", int(status.get("skipped_count") or 0))
    c5.metric("Saham Saat Ini", status.get("current_ticker") or "-")
    st.info(status.get("message", "Job sedang berjalan..."))

    render_background_ticker_process(status, job_id, key_suffix=render_key_suffix)

    events = status.get("events", [])
    if events:
        with st.expander("Log job background", expanded=False):
            st.dataframe(pd.DataFrame(events[-15:]), width="stretch", hide_index=True)

    if job_status == "DONE" and status.get("summary"):
        render_analysis_summary(status["summary"], show_analyzed=False)
    elif job_status == "FAILED":
        st.error(status.get("error") or "Job background gagal tanpa detail error.")


with tab_daily:
    st.header("Dashboard Harian Global Model")
    st.write(
        "Gunakan alur sederhana setelah market close: update data, evaluasi prediksi pending, "
        "lalu buat prediksi baru dengan Global Model."
    )
    st.caption("Prediksi lama tetap tersimpan sebagai bahan evaluasi. Operasional baru hanya memakai model Global-*.")

    global_records = list_ticker_models("GLOBAL")
    global_dates = sorted({str(row.get("trained_until_date", "")) for row in global_records if row.get("trained_until_date")})
    g1, g2, g3 = st.columns(3)
    g1.metric("Mode Aktif", "Global Model")
    g2.metric("Artifact Global", len(global_records))
    g3.metric("Data Training", global_dates[-1] if global_dates else "-")
    if not global_records:
        st.warning("Global Model belum tersedia. Jalankan training global dari VS Code terlebih dahulu.")
        st.code("python scripts\\train_global_models_cli.py --config config\\stocks.yaml --run-type FINAL", language="powershell")

    workflow_steps = pd.DataFrame([
        {"Urutan": 1, "Langkah": "Update Data", "Tujuan": "Memastikan OHLCV lokal berisi data penutupan terbaru."},
        {"Urutan": 2, "Langkah": "Evaluasi Prediksi Lama", "Tujuan": "Mengubah prediksi jatuh tempo menjadi track record akurasi."},
        {"Urutan": 3, "Langkah": "Generate Prediksi FINAL", "Tujuan": "Membuat prediksi H+1, H+3, H+5, dan H+10 untuk tanggal data terbaru."},
        {"Urutan": 4, "Langkah": "Filter Trusted Signal", "Tujuan": "Memakai hanya model/ticker dengan sampel dan reliability cukup."},
        {"Urutan": 5, "Langkah": "Tampilkan Rencana Besok", "Tujuan": "Menjawab saham apa yang dipantau, kenapa, area entry, stop loss, dan trust model."},
    ])
    with st.expander("Urutan kerja yang dipakai dashboard", expanded=True):
        st.dataframe(workflow_steps, width="stretch", hide_index=True)

    daily_mode = st.radio(
        "Mode tampilan",
        ["Pemula", "Trader", "Audit"],
        horizontal=True,
        help="Pemula fokus ke keputusan. Trader menambah angka potensi/risk. Audit menampilkan status data dan job.",
    )
    min_daily_reliability = st.slider("Batas reliability model minimum", 0.0, 100.0, 55.0, 5.0)
    min_daily_confidence = st.slider("Batas confidence entry minimum", 0.0, 100.0, 60.0, 5.0)
    min_daily_evaluations = st.number_input(
        "Minimal track record per ticker/model",
        min_value=3,
        max_value=200,
        value=20,
        step=1,
        help="Sinyal tidak dianggap trusted jika evaluasi historisnya belum mencapai angka ini.",
    )

    daily_scope_tickers = active_tickers if active_tickers else tickers
    decision_board_df = build_daily_decision_board(
        daily_scope_tickers,
        min_reliability=float(min_daily_reliability),
        min_confidence=float(min_daily_confidence),
        min_evaluations=int(min_daily_evaluations),
        portfolio_capital=float(portfolio_capital),
        risk_per_trade_pct=float(risk_per_trade_pct),
    )
    recent_jobs = list_analysis_jobs(limit=5)
    daily_has_running_job = has_running_analysis_job(recent_jobs)
    daily_training_policy = evaluate_training_policy(
        min_direction_accuracy_pct=float(st.session_state.get("training_policy_min_accuracy", 52.0)),
        min_recent_evaluations=int(st.session_state.get("training_policy_min_samples", 20)),
        lookback_evaluations=int(st.session_state.get("training_policy_lookback", 50)),
        routine_training_interval={
            "Mingguan": "WEEKLY",
            "Bulanan": "MONTHLY",
            "Nonaktif": "OFF",
        }.get(st.session_state.get("training_policy_interval_label", "Bulanan"), "MONTHLY"),
    )
    daily_store_status = model_store_status(daily_scope_tickers)
    daily_global_models_available = bool(list_ticker_models("GLOBAL"))
    daily_readiness = build_user_friendly_readiness(
        daily_scope_tickers,
        training_policy=daily_training_policy,
        store_status=daily_store_status,
        job_rows=recent_jobs,
    )
    render_readiness_panel(daily_readiness)

    if decision_board_df.empty:
        st.warning("Belum ada data prediksi yang cukup untuk membuat ringkasan harian.")
    else:
        latest_data_date = decision_board_df["Tanggal Data"].dropna().astype(str).max()
        buy_count = int(decision_board_df["Sinyal"].astype(str).str.contains("BUY", regex=False).sum())
        trusted_count = int((decision_board_df["Trust Model"] == "LAYAK DIPERCAYA").sum()) if "Trust Model" in decision_board_df.columns else 0
        unfinished_df = decision_board_df[decision_board_df["Status Analisis"] != "LENGKAP"].copy()
        data_ready_count = int((decision_board_df["Status Data"] == "OK").sum())
        troubled_jobs = [job for job in recent_jobs if job.get("status") in ["RUNNING", "STALE", "FAILED", "UNKNOWN"]]
        daily_status = "SIAP" if len(unfinished_df) == 0 and not troubled_jobs else "PERLU CEK"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tanggal Data", latest_data_date)
        c2.metric("Saham Dicek", len(decision_board_df))
        c3.metric("Data OK", data_ready_count)
        c4.metric("Kandidat BUY", buy_count)
        c5.metric("Model Trusted", trusted_count)

        render_feature_status(
            "Status Harian",
            daily_status,
            f"{buy_count} kandidat BUY, {len(unfinished_df)} saham belum lengkap, {len(troubled_jobs)} job perlu dicek.",
            "Gunakan tombol cepat di bawah sebelum melihat detail ranking." if daily_status != "SIAP" else "Lanjut cek kandidat BUY dan risk/reward.",
        )

        st.subheader("Tombol Cepat After Market")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            quick_update_clicked = st.button(
                "Update Harga Saja",
                key="daily_quick_update",
                help="Hanya mengambil data harga terbaru. Ranking/prediksi baru berubah setelah analisis ulang dijalankan.",
                disabled=offline_mode or daily_has_running_job,
            )
        with q2:
            quick_accuracy_clicked = st.button(
                "Evaluasi Akurasi",
                key="daily_quick_accuracy",
                help="Mengevaluasi prediksi PENDING jika data aktual sudah tersedia.",
                disabled=daily_has_running_job or not daily_global_models_available,
            )
        with q3:
            quick_saved_model_clicked = st.button(
                "Prediksi Global Model",
                type="primary",
                key="daily_quick_saved_model",
                help="Prediksi harian tanpa training ulang memakai Global Model.",
                disabled=daily_has_running_job,
            )
        with q4:
            quick_retrain_clicked = st.button(
                "Retrain Lama Dinonaktifkan",
                key="daily_quick_retrain_due",
                help="Menjalankan training background hanya jika belum ada model atau policy retrain due.",
                disabled=True,
            )

        if quick_update_clicked:
            with st.spinner(f"Update data untuk {len(daily_scope_tickers)} saham..."):
                st.session_state["daily_quick_update_summary"] = run_auto_updater(
                    tickers=daily_scope_tickers,
                    sleep_seconds=0.05,
                )
            st.success("Update data selesai.")
            render_update_summary(st.session_state["daily_quick_update_summary"])

        if quick_accuracy_clicked:
            with st.spinner("Mengevaluasi prediksi pending..."):
                evaluate_pending_predictions()
            st.success("Evaluasi akurasi selesai.")

        if quick_saved_model_clicked:
            with st.spinner(f"Menjalankan prediksi harian Global Model untuk {len(daily_scope_tickers)} saham..."):
                progress_callback = create_live_analysis_tracker("Live Progress Prediksi Global Model Harian")
                saved_prediction_summary = predict_with_global_models(
                    tickers=daily_scope_tickers,
                    duplicate_policy="skip",
                    prediction_run_type="FINAL",
                    progress_callback=progress_callback,
                )
                st.session_state["last_global_model_prediction_summary"] = saved_prediction_summary
            st.success("Prediksi Global Model selesai.")

        if quick_retrain_clicked:
            st.info("Retrain model lama dinonaktifkan. Training Global Model dijalankan dari CLI agar tidak mengunci Streamlit.")

        st.subheader("Checklist Sebelum Sesi Berikutnya")
        render_daily_checklist(decision_board_df, recent_jobs)

        st.subheader("Rencana Trading Besok")
        if daily_mode == "Pemula":
            display_cols = ["Saham", "Sinyal", "Entry Area", "Stop Loss", "Target H+3", "Alasan Utama", "Trust Model", "Risiko"]
        elif daily_mode == "Trader":
            display_cols = [
                "Saham",
                "Sinyal",
                "Harga Terakhir",
                "Entry Area",
                "Stop Loss",
                "Target H+3",
                "Risk/Reward",
                "Lot Maks",
                "Potensi H+3",
                "Arah H+1",
                "Confidence",
                "Regime Pasar",
                "Breadth Naik",
                "Reliability",
                "Evaluasi",
                "Trust Model",
                "Alasan Utama",
            ]
        else:
            display_cols = [
                "Saham",
                "Sinyal",
                "Harga Terakhir",
                "Entry Area",
                "Stop Loss",
                "Target H+3",
                "Risk/Reward",
                "Lot Maks",
                "Nilai Posisi",
                "Potensi H+3",
                "Arah H+1",
                "Confidence",
                "Regime Pasar",
                "Breadth Naik",
                "Reliability",
                "Evaluasi",
                "Akurasi Arah",
                "Trust Model",
                "Risiko",
                "Status Data",
                "Status Analisis",
                "Tanggal Data",
                "Alasan Utama",
                "Catatan Trust",
            ]

        action_filter = st.multiselect(
            "Filter sinyal",
            options=decision_board_df["Sinyal"].dropna().unique().tolist(),
            default=st.session_state.get("daily_signal_filter", decision_board_df["Sinyal"].dropna().unique().tolist()),
            key="daily_signal_filter",
        )
        view_board_df = decision_board_df[decision_board_df["Sinyal"].isin(action_filter)].copy() if action_filter else decision_board_df
        st.dataframe(
            view_board_df[display_cols].style.format({
                "Harga Terakhir": "Rp {:,.0f}",
                "Stop Loss": "Rp {:,.0f}",
                "Target H+3": "Rp {:,.0f}",
                "Risk/Reward": "{:.2f}",
                "Nilai Posisi": "Rp {:,.0f}",
                "Potensi H+3": "{:+.2f}%",
                "Confidence": "{:.2f}%",
                "Breadth Naik": "{:.2f}%",
                "Reliability": "{:.2f}",
                "Akurasi Arah": "{:.2f}%",
            }),
            width="stretch",
            hide_index=True,
        )

        buy_plan_df = decision_board_df[decision_board_df["Sinyal"].astype(str).str.contains("BUY", regex=False)].copy()
        if buy_plan_df.empty:
            st.info("Belum ada kandidat BUY trusted. Fokus besok: pantau watchlist, evaluasi data, atau tunggu track record model bertambah.")
        else:
            st.subheader("Prioritas Pantauan Utama")
            st.dataframe(
                buy_plan_df[[
                    "Saham",
                    "Entry Area",
                    "Stop Loss",
                    "Target H+3",
                    "Risk/Reward",
                    "Lot Maks",
                    "Potensi H+3",
                    "Confidence",
                    "Reliability",
                    "Alasan Utama",
                ]].head(20).style.format({
                    "Stop Loss": "Rp {:,.0f}",
                    "Target H+3": "Rp {:,.0f}",
                    "Risk/Reward": "{:.2f}",
                    "Potensi H+3": "{:+.2f}%",
                    "Confidence": "{:.2f}%",
                    "Reliability": "{:.2f}",
                }),
                width="stretch",
                hide_index=True,
            )

        if not unfinished_df.empty:
            unfinished_tickers = unfinished_df["Saham"].dropna().astype(str).tolist()
            st.warning(f"{len(unfinished_tickers)} saham belum lengkap: {', '.join(unfinished_tickers[:20])}")
            st.caption("Gunakan tombol cepat `Analisis yang Belum Lengkap` di atas untuk menjalankan ulang hanya saham tersebut.")

        if recent_jobs:
            with st.expander("Status Job Background Terbaru", expanded=daily_mode == "Audit"):
                st.dataframe(pd.DataFrame(recent_jobs), width="stretch", hide_index=True)

        st.info(
            "Sinyal AI bukan rekomendasi final. Pakai hanya jika data FINAL tersedia, model lolos audit, "
            "risk/reward sesuai, dan ukuran posisi mengikuti batas risiko portfolio."
        )


def render_daily_workflow_summary(summary):
    if not summary:
        return

    st.subheader("Ringkasan Workflow Harian Terakhir")
    st.caption(f"Dijalankan pada: {summary.get('started_at', '-')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saham Diproses", int(summary.get("selected_count") or 0))
    c2.metric("Update Berhasil", int(summary.get("updated_count") or 0))
    c3.metric("Analisis Lengkap", int(summary.get("complete_count") or 0))
    c4.metric("Belum Selesai", int(summary.get("unfinished_count") or 0))

    status = summary.get("status", "UNKNOWN")
    if status == "OK":
        st.success(summary.get("message", "Workflow harian selesai."))
    elif status == "PARTIAL":
        st.warning(summary.get("message", "Workflow harian selesai dengan catatan."))
    else:
        st.info(summary.get("message", "Workflow harian sudah dijalankan."))

    job_id = summary.get("background_job_id")
    if job_id:
        st.info(f"Analisis saham yang belum selesai sedang berjalan di background. Job ID: `{job_id}`")

    details = summary.get("details") or []
    if details:
        with st.expander("Detail langkah workflow", expanded=False):
            st.dataframe(pd.DataFrame(details), width="stretch", hide_index=True)


if LEGACY_MODELS_ENABLED and tab_final is not None:
  with tab_final:
    st.header("FINAL Lama Arsip")
    st.write(
        "Tab ini dipertahankan sebagai arsip/evaluasi prediksi lama. Prediksi baru dibuat melalui Global Model di tab Workflow Harian."
    )
    st.info(
        "Model lama per-saham dinonaktifkan. Log FINAL lama tetap dipakai untuk evaluasi akurasi historis."
    )

    final_scope_col, final_model_col, final_epoch_col = st.columns([1.4, 1.2, 0.8])
    with final_scope_col:
        final_scope = st.radio(
            "Cakupan prediksi FINAL",
            ["Saham aktif", "Semua saham", "Saham tertentu"],
            index=1 if use_all_tickers else 0,
            horizontal=True,
            key="final_prediction_scope",
        )
    with final_model_col:
        final_required_models = st.multiselect(
            "Model wajib untuk status H+3",
            options=["XGBoost", "LSTM"],
            default=["XGBoost"],
            key="final_required_models",
        )
    with final_epoch_col:
        final_lstm_epochs = st.number_input(
            "Epoch LSTM",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key="final_lstm_epochs",
        )

    if final_scope == "Semua saham":
        final_scope_tickers = tickers
    elif final_scope == "Saham tertentu":
        default_final_selection = [ticker] if ticker in tickers else tickers[: min(5, len(tickers))]
        final_scope_tickers = st.multiselect(
            "Pilih saham untuk prediksi FINAL",
            options=tickers,
            default=default_final_selection,
            key="final_specific_tickers",
        )
    else:
        final_scope_tickers = active_tickers if active_tickers else ([ticker] if ticker else [])

    final_status_df = build_final_prediction_workflow_status(
        final_scope_tickers,
        required_models=final_required_models,
    )

    if final_status_df.empty:
        st.warning("Belum ada saham yang bisa dicek. Pastikan ticker aktif dan data lokal tersedia.")
    else:
        final_ready_df = final_status_df[final_status_df["Kesiapan"] == "SIAP PREDIKSI FINAL"].copy()
        final_done_df = final_status_df[final_status_df["Kesiapan"] == "FINAL SUDAH ADA"].copy()
        final_due_pending = int(final_status_df["Pending FINAL Jatuh Tempo"].sum())
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Saham Dicek", len(final_status_df))
        f2.metric("Siap FINAL", len(final_ready_df))
        f3.metric("FINAL Sudah Ada", len(final_done_df))
        f4.metric("Pending Jatuh Tempo", final_due_pending)

        st.subheader("Urutan Kerja Trusted")
        step1, step2, step3 = st.columns(3)
        with step1:
            st.markdown("**1. Update data lokal**")
            if offline_mode:
                st.caption("Mode offline aktif. Import/update CSV lokal terlebih dahulu jika data target H+1/H+3 sudah tersedia.")
                update_final_clicked = False
            else:
                update_final_clicked = st.button(
                "Update Harga Online",
                    key="final_online_update",
                    disabled=not bool(final_scope_tickers),
                )
            if update_final_clicked:
                with st.spinner(f"Update data untuk {len(final_scope_tickers)} saham..."):
                    st.session_state["final_update_summary"] = run_auto_updater(
                        tickers=final_scope_tickers,
                        sleep_seconds=0.05,
                    )
                st.success("Update data selesai.")
                render_update_summary(st.session_state["final_update_summary"])

        with step2:
            st.markdown("**2. Evaluasi prediksi lama**")
            st.caption("Mengubah pending FINAL menjadi evaluated jika harga aktual target sudah tersedia.")
            if st.button("Evaluasi Pending FINAL", key="final_evaluate_pending", type="primary"):
                with st.spinner("Mengevaluasi prediksi pending yang sudah jatuh tempo..."):
                    evaluate_pending_predictions()
                st.success("Evaluasi pending selesai. Refresh status untuk melihat perubahan.")

        with step3:
            st.markdown("**3. Jalankan FINAL hari ini**")
            final_data_confirmed = st.checkbox(
                "Saya konfirmasi data harga hari ini sudah final/bersih",
                value=False,
                key="final_data_confirmed",
            )
            final_to_predict = final_ready_df["Saham"].dropna().astype(str).tolist()
            run_final_clicked = st.button(
                "Jalankan Prediksi FINAL Lama",
                key="run_daily_final_prediction",
                type="primary",
                disabled=(not LEGACY_MODELS_ENABLED or not final_data_confirmed or not bool(final_to_predict)),
                help="Dinonaktifkan karena model lama per-saham tidak lagi dipakai. Gunakan Global Model.",
            )
            if run_final_clicked:
                job_id = start_background_analysis_job(
                    final_to_predict,
                    lstm_epochs=int(final_lstm_epochs),
                    duplicate_policy="skip",
                    prediction_run_type="FINAL",
                )
                st.session_state["final_prediction_job_id"] = job_id
                st.success(f"Job prediksi FINAL dimulai untuk {len(final_to_predict)} saham. Job ID: {job_id}")

        if "final_prediction_job_id" in st.session_state:
            with st.expander("Status job prediksi FINAL", expanded=True):
                render_background_analysis_job(st.session_state["final_prediction_job_id"])

        st.subheader("Audit Kesiapan Prediksi FINAL")
        status_filter = st.selectbox(
            "Filter kesiapan",
            options=["SEMUA"] + sorted(final_status_df["Kesiapan"].dropna().unique().tolist()),
            key="final_readiness_filter",
        )
        view_final_status_df = final_status_df.copy()
        if status_filter != "SEMUA":
            view_final_status_df = view_final_status_df[view_final_status_df["Kesiapan"] == status_filter]
        st.dataframe(view_final_status_df, width="stretch", hide_index=True)

        with st.expander("Aturan reliability trusted", expanded=False):
            trusted_rules = pd.DataFrame([
                {"Aturan": "Run type", "Nilai": "FINAL saja", "Dampak": "INTRADAY dan BACKFILL tidak dihitung sebagai reliability trusted."},
                {"Aturan": "Duplikasi", "Nilai": "Lewati duplikat", "Dampak": "Prediksi data/tanggal/model yang sama tidak menggandakan akurasi."},
                {"Aturan": "Evaluasi", "Nilai": "Setelah target tersedia", "Dampak": "PENDING berubah ke EVALUATED hanya jika harga aktual sudah ada di data lokal."},
                {"Aturan": "Data lama", "Nilai": "UNKNOWN_LEGACY", "Dampak": "Catatan lama tetap arsip, tetapi tidak dipakai sebagai trusted score baru."},
            ])
            st.dataframe(trusted_rules, width="stretch", hide_index=True)


with tab_update:
    st.header("Workflow Harian Trading")
    st.write("Jalankan proses harian secara berurutan agar data harga, prediksi, ranking, dan akurasi tetap sinkron.")
    st.caption("Gunakan workflow lengkap untuk operasi harian. Panel teknis di bawah tetap tersedia untuk update harga saja, import CSV, audit, dan backfill.")

    if not tickers:
        st.warning("Daftar saham kosong atau `config/stocks.yaml` tidak dapat dibaca.")
    else:
        st.caption(f"Daftar aktif: {len(tickers)} saham papan utama/konfigurasi lokal.")
        try:
            quick_status_df = get_local_data_status(active_tickers if active_tickers else tickers)
            ok_count = int((quick_status_df["status"] == "OK").sum()) if not quick_status_df.empty else 0
            status_label = "SIAP" if ok_count == len(quick_status_df) and ok_count > 0 else "PERLU CEK"
            render_feature_status(
                "Workflow Harian",
                status_label,
                f"{ok_count}/{len(quick_status_df)} file harga lokal OK.",
                "Jalankan update data jika tanggal data belum terbaru.",
            )
        except Exception as e:
            render_feature_status("Workflow Harian", "PERLU CEK", f"Status data belum bisa dibaca: {e}")

    st.subheader("1. Cek Sinkronisasi Sistem")
    st.caption(
        "Audit cepat untuk memastikan data harga, prediksi aktif, prediksi pending, backfill, ranking, dan akurasi sudah saling nyambung."
    )
    sync_col1, sync_col2 = st.columns([1.2, 1])
    with sync_col1:
        sync_scope = st.radio(
            "Cakupan cek sinkronisasi",
            ["Saham terpilih/aktif", "Semua saham"],
            horizontal=True,
            key="sync_system_scope",
        )
    with sync_col2:
        sync_required_models = st.multiselect(
            "Model H+3 wajib",
            options=["XGBoost", "LSTM"],
            default=["XGBoost"],
            key="sync_required_models",
        )
    sync_tickers = tickers if sync_scope == "Semua saham" else active_tickers
    if st.button("Cek Sinkronisasi Sistem", type="primary", disabled=not bool(sync_tickers), key="check_system_sync"):
        with st.spinner(f"Mengecek sinkronisasi {len(sync_tickers)} saham..."):
            sync_summary_df, sync_detail_df = build_system_sync_status(
                sync_tickers,
                required_models=sync_required_models,
            )
            st.session_state["system_sync_summary"] = sync_summary_df
            st.session_state["system_sync_detail"] = sync_detail_df

    if "system_sync_detail" in st.session_state:
        sync_summary_df = st.session_state.get("system_sync_summary", pd.DataFrame())
        sync_detail_df = st.session_state.get("system_sync_detail", pd.DataFrame())
        if sync_detail_df.empty:
            st.info("Belum ada data sinkronisasi yang bisa ditampilkan.")
        else:
            total_checked = len(sync_detail_df)
            total_ok = int((sync_detail_df["status_sinkron"] == "SINKRON").sum())
            total_need = int((sync_detail_df["status_sinkron"] == "PERLU CEK").sum())
            total_pending = int(sync_detail_df["prediksi_pending"].sum())
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Saham Dicek", total_checked)
            s2.metric("Sinkron", total_ok)
            s3.metric("Perlu Cek", total_need)
            s4.metric("Prediksi Pending", total_pending)

            with st.expander("Ringkasan Sinkronisasi", expanded=False):
                st.dataframe(sync_summary_df, width="stretch", hide_index=True)

            status_filter = st.selectbox(
                "Filter status sinkronisasi",
                options=["SEMUA"] + sorted(sync_detail_df["status_sinkron"].unique().tolist()),
                key="sync_status_filter",
            )
            view_sync_df = sync_detail_df.copy()
            if status_filter != "SEMUA":
                view_sync_df = view_sync_df[view_sync_df["status_sinkron"] == status_filter]
            st.dataframe(view_sync_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("2. Mode Operasi Model")
    st.caption("Pisahkan pemakaian harian dari training berat: cek status model, prediksi dari artifact, lalu retrain hanya jika perlu.")
    model_status_tab, daily_action_tab, training_setting_tab = st.tabs([
        "Status Model",
        "Aksi Harian",
        "Training",
    ])

    with training_setting_tab:
        st.caption("Pengaturan ini hanya membaca evaluasi historis. Training aktif sekarang memakai Global Model dari CLI.")
        adaptive_min_samples = get_adaptive_min_evaluation_default()
        model_sample_status_df = build_model_evaluation_sample_status()
        policy_col1, policy_col2, policy_col3, policy_col4 = st.columns(4)
        with policy_col1:
            min_policy_accuracy = st.number_input(
                "Batas akurasi retrain (%)",
                min_value=40.0,
                max_value=80.0,
                value=52.0,
                step=1.0,
                key="training_policy_min_accuracy",
                help="Jika akurasi arah model utama turun di bawah angka ini, dashboard menandai perlu retrain darurat.",
            )
        with policy_col2:
            min_policy_samples = st.number_input(
                "Minimal sampel evaluasi",
                min_value=5,
                max_value=200,
                value=int(st.session_state.get("training_policy_min_samples", adaptive_min_samples)),
                step=5,
                key="training_policy_min_samples",
                help="Default otomatis dihitung dari jumlah evaluasi historis FINAL XGBoost H+1 yang sudah tersedia.",
            )
            st.caption(f"Default adaptif dari data historis: {adaptive_min_samples}")
        with policy_col3:
            policy_lookback = st.number_input(
                "Lookback evaluasi",
                min_value=10,
                max_value=500,
                value=int(st.session_state.get("training_policy_lookback", max(50, adaptive_min_samples))),
                step=10,
                key="training_policy_lookback",
                help="Jumlah evaluasi terbaru yang dipakai untuk menghitung akurasi trigger retrain.",
            )
        with policy_col4:
            routine_training_interval = st.selectbox(
                "Jadwal retrain rutin",
                options=["Bulanan", "Mingguan", "Nonaktif"],
                index=0,
                key="training_policy_interval_label",
                help="Harian memakai Global Model. Pengaturan ini dipertahankan untuk membaca evaluasi historis.",
            )
        if not model_sample_status_df.empty:
            with st.expander("Status sampel evaluasi per model", expanded=False):
                st.dataframe(
                    model_sample_status_df.style.format({"akurasi_arah_pct": "{:.2f}%"}),
                    width="stretch",
                    hide_index=True,
                )

    training_policy = evaluate_training_policy(
        min_direction_accuracy_pct=float(st.session_state.get("training_policy_min_accuracy", 52.0)),
        min_recent_evaluations=int(st.session_state.get("training_policy_min_samples", 20)),
        lookback_evaluations=int(st.session_state.get("training_policy_lookback", 50)),
        prediction_purpose="NEXT_DAY_DIRECTION",
        model_name="XGBoost",
        routine_training_interval={
            "Mingguan": "WEEKLY",
            "Bulanan": "MONTHLY",
            "Nonaktif": "OFF",
        }.get(st.session_state.get("training_policy_interval_label", "Bulanan"), "MONTHLY"),
    )
    per_model_training_policy_df = evaluate_training_policy_by_model(
        min_direction_accuracy_pct=float(st.session_state.get("training_policy_min_accuracy", 52.0)),
        min_recent_evaluations=int(st.session_state.get("training_policy_min_samples", 20)),
        lookback_evaluations=int(st.session_state.get("training_policy_lookback", 50)),
        prediction_purpose="NEXT_DAY_DIRECTION",
    )
    st.session_state["training_policy_status"] = training_policy
    store_status = model_store_status(active_tickers)
    global_model_records = list_ticker_models("GLOBAL")
    global_trained_dates = sorted({str(row.get("trained_until_date", "")) for row in global_model_records if row.get("trained_until_date")})

    with model_status_tab:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Mode Model", "GLOBAL")
        p2.metric("Global Model", f"{len(global_model_records)} artifact")
        p3.metric(
            "Akurasi Lama Tersimpan",
            "-" if training_policy["recent_accuracy_pct"] is None else f"{training_policy['recent_accuracy_pct']:.2f}%",
        )
        p4.metric("Global Trained Until", global_trained_dates[-1] if global_trained_dates else "-")
        if global_model_records:
            st.success("Global Model tersedia. Prediksi baru akan memakai model Global-*.")
        else:
            st.warning("Global Model belum tersedia. Jalankan training global dari VS Code terlebih dahulu.")
            st.code("python scripts\\train_global_models_cli.py --config config\\stocks.yaml --run-type FINAL", language="powershell")
        with st.expander("Catatan evaluasi model lama", expanded=False):
            st.write(
                "Artifact dan prediksi model lama tidak dihapus. Log lama tetap dipakai untuk evaluasi akurasi historis, "
                "tetapi fitur operasional model lama sudah dinonaktifkan."
            )

    with daily_action_tab:
        st.caption("Gunakan ini untuk prediksi harian tanpa training ulang. Mode aktif: Global Model.")
        global_model_prediction_clicked = st.button(
            "Prediksi Harian Global Model",
            type="primary",
            disabled=(not bool(active_tickers) or has_running_analysis_job()),
            key="run_global_model_daily_prediction",
            help="Memakai model GLOBAL yang dilatih dari gabungan semua saham.",
        )
        if global_model_prediction_clicked:
            with st.spinner(f"Menjalankan prediksi harian Global Model untuk {len(active_tickers)} saham..."):
                progress_callback = create_live_analysis_tracker("Live Progress Prediksi Global Model")
                global_prediction_summary = predict_with_global_models(
                    tickers=active_tickers,
                    duplicate_policy="skip",
                    prediction_run_type="FINAL",
                    progress_callback=progress_callback,
                )
                st.session_state["last_global_model_prediction_summary"] = global_prediction_summary
            st.success("Prediksi harian Global Model selesai. Hasilnya masuk ke Ranking Prediksi dengan nama model Global-*.")

        if "last_global_model_prediction_summary" in st.session_state:
            global_summary = st.session_state["last_global_model_prediction_summary"]
            gs1, gs2, gs3 = st.columns(3)
            gs1.metric("Global Berhasil", len(global_summary.get("predicted", [])))
            gs2.metric("Global Dilewati", len(global_summary.get("skipped", [])))
            gs3.metric("Global Gagal", len(global_summary.get("failed", [])))
            if global_summary.get("failed"):
                with st.expander("Detail gagal prediksi Global Model", expanded=False):
                    st.dataframe(pd.DataFrame(global_summary["failed"]), width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("3. Cakupan Workflow")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        update_scope = st.radio(
            "Cakupan update",
            ["Saham aktif dari sidebar", "Semua saham di daftar", "Beberapa saham pertama", "Saham tertentu"],
            index=1 if use_all_tickers else 0,
            horizontal=True,
        )
    with col_b:
        sleep_seconds = st.number_input("Jeda antar saham (detik)", min_value=0.0, max_value=5.0, value=0.2, step=0.1)

    selected_tickers = active_tickers.copy()
    if update_scope == "Semua saham di daftar":
        selected_tickers = tickers
    elif update_scope == "Beberapa saham pertama":
        limit = st.number_input("Jumlah saham", min_value=1, max_value=max(len(tickers), 1), value=min(10, len(tickers)), step=1)
        selected_tickers = tickers[: int(limit)]
    elif update_scope == "Saham tertentu":
        default_selection = [ticker] if ticker in tickers else tickers[: min(5, len(tickers))]
        selected_tickers = st.multiselect("Pilih saham", options=tickers, default=default_selection)

    workflow_recent_jobs = list_analysis_jobs(limit=10)
    workflow_has_running_job = has_running_analysis_job(workflow_recent_jobs)
    workflow_store_status = model_store_status(selected_tickers)
    workflow_global_models_available = bool(list_ticker_models("GLOBAL"))
    workflow_readiness = build_user_friendly_readiness(
        selected_tickers,
        training_policy=training_policy,
        store_status=workflow_store_status,
        job_rows=workflow_recent_jobs,
    )
    render_readiness_panel(workflow_readiness)
    ideal_workflow_plan = build_ideal_workflow_plan(
        training_policy=training_policy,
        store_status=workflow_store_status,
        has_running_job=workflow_has_running_job,
    )
    with st.expander("Rencana workflow ideal", expanded=True):
        st.dataframe(pd.DataFrame(ideal_workflow_plan), width="stretch", hide_index=True)
        st.caption("Prediksi lama tidak ditimpa: workflow memakai `duplicate_policy=skip` untuk menjaga bahan evaluasi akurasi.")
    if workflow_has_running_job:
        st.warning("Ada job background yang masih berjalan. Tombol proses baru dinonaktifkan agar pekerjaan tidak saling tumpang tindih.")

    recommended_clicked = st.button(
        f"Jalankan Proses yang Disarankan: {workflow_readiness['next_step']}",
        type="primary",
        disabled=(not bool(selected_tickers) or workflow_has_running_job),
        key="run_recommended_daily_process",
        help="Dashboard memilih aksi otomatis: update data, evaluasi, lalu prediksi Global Model.",
    )
    if recommended_clicked:
        recommended_summary = {
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status_awal": workflow_readiness["status"],
            "aksi": workflow_readiness["next_step"],
            "details": [],
        }
        try:
            if workflow_readiness["next_step"] == "Update Harga":
                with st.spinner("Mengupdate harga sesuai rekomendasi dashboard..."):
                    update_summary = run_auto_updater(tickers=selected_tickers, sleep_seconds=float(sleep_seconds))
                    st.session_state["last_update_summary"] = update_summary
                    recommended_summary["details"].append({
                        "langkah": "Update harga",
                        "status": "SELESAI",
                        "ringkasan": f"{len(update_summary.get('updated', []))} saham diperbarui, {len(update_summary.get('failed', []))} gagal.",
                    })
            elif workflow_readiness["next_step"] in ["Training Awal", "Retrain Model"]:
                st.info("Training model lama dinonaktifkan. Jalankan `python scripts\\train_global_models_cli.py --config config\\stocks.yaml --run-type FINAL` untuk training Global Model.")
                recommended_summary["details"].append({
                    "langkah": workflow_readiness["next_step"],
                    "status": "DILEWATI",
                    "ringkasan": "Model lama per-saham dinonaktifkan.",
                })
            elif workflow_readiness["next_step"] == "Prediksi Model Tersimpan":
                with st.spinner("Menjalankan update, evaluasi, dan prediksi Global Model..."):
                    update_summary = run_auto_updater(tickers=selected_tickers, sleep_seconds=float(sleep_seconds)) if not offline_mode else {"updated": [], "failed": [], "skipped": []}
                    evaluate_pending_predictions()
                    progress_callback = create_live_analysis_tracker("Live Progress Proses Harian Disarankan")
                    saved_prediction_summary = predict_with_global_models(
                        tickers=selected_tickers,
                        duplicate_policy="skip",
                        prediction_run_type="FINAL",
                        progress_callback=progress_callback,
                    )
                    st.session_state["last_update_summary"] = update_summary
                    st.session_state["last_global_model_prediction_summary"] = saved_prediction_summary
                    recommended_summary["details"].append({
                        "langkah": "Prediksi Global Model",
                        "status": "SELESAI",
                        "ringkasan": f"{len(saved_prediction_summary.get('predicted', []))} saham berhasil diprediksi.",
                    })
                    st.success("Proses harian disarankan selesai. Cek Ranking Prediksi.")
            else:
                st.info(workflow_readiness["action"])
                recommended_summary["details"].append({
                    "langkah": workflow_readiness["next_step"],
                    "status": "INFO",
                    "ringkasan": workflow_readiness["action"],
                })
        except Exception as e:
            recommended_summary["details"].append({
                "langkah": workflow_readiness["next_step"],
                "status": "GAGAL",
                "ringkasan": str(e),
            })
            st.error(f"Proses yang disarankan gagal: {e}")
        st.session_state["last_recommended_process_summary"] = recommended_summary

    if "last_recommended_process_summary" in st.session_state:
        with st.expander("Ringkasan proses yang disarankan terakhir", expanded=False):
            st.json(st.session_state["last_recommended_process_summary"])

    st.markdown("---")
    st.subheader("4. Aksi Workflow")
    st.caption(
        "Panel ini menyatukan proses harian agar urut: update data, evaluasi prediksi pending, lalu prediksi memakai Global Model."
    )

    wf_col1, wf_col2, wf_col3 = st.columns(3)
    with wf_col1:
        workflow_run_update = st.checkbox(
            "1. Update data harga",
            value=not offline_mode,
            key="workflow_run_update",
            disabled=offline_mode,
        )
        if offline_mode:
            workflow_run_update = False
        workflow_repair_ohlc = st.checkbox("2. Audit & perbaiki OHLC", value=True, key="workflow_repair_ohlc")
    with wf_col2:
        workflow_evaluate_accuracy = st.checkbox("3. Evaluasi prediksi pending", value=True, key="workflow_evaluate_accuracy")
        workflow_predict_saved_models = st.checkbox(
            "4. Prediksi pakai Global Model",
            value=workflow_global_models_available,
            key="workflow_predict_saved_models",
            disabled=not workflow_global_models_available,
            help="Memakai artifact GLOBAL tanpa training ulang. Ini jalur utama harian.",
        )
        workflow_retrain_if_due = st.checkbox(
            "5. Retrain model lama",
            value=False,
            key="workflow_analyze_unfinished",
            disabled=True,
            help="Model lama per-saham dinonaktifkan. Training Global Model dijalankan via CLI.",
        )
    with wf_col3:
        workflow_required_models = ["Global-Direction-LIGHTGBM", "Global-Direction-XGBOOST"]
        st.caption("Model wajib aktif: Global-Direction-LIGHTGBM dan Global-Direction-XGBOOST.")
        workflow_lstm_epochs = st.number_input(
            "Epoch LSTM legacy",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            disabled=True,
            key="workflow_lstm_epochs",
        )

    with st.expander("Pengaturan prediksi dan duplikasi", expanded=False):
        duplicate_policy_label = st.radio(
            "Jika prediksi untuk saham/model/tanggal yang sama sudah ada",
            [
                "Lewati duplikat",
                "Timpa prediksi lama",
                "Simpan sebagai versi intraday",
            ],
            horizontal=True,
            key="daily_duplicate_policy_label",
            help=(
                "Lewati duplikat menjaga rekap akurasi tetap bersih. "
                "Timpa cocok jika analisis sebelumnya salah/ingin diganti. "
                "Intraday menyimpan snapshot sementara saat market belum tutup."
            ),
        )
    duplicate_policy_map = {
        "Lewati duplikat": "skip",
        "Timpa prediksi lama": "overwrite",
        "Simpan sebagai versi intraday": "intraday",
    }
    daily_duplicate_policy = duplicate_policy_map.get(duplicate_policy_label, "skip")
    daily_prediction_run_type = "INTRADAY" if daily_duplicate_policy == "intraday" else "FINAL"

    daily_workflow_clicked = st.button(
        "Jalankan Workflow Harian Lengkap",
        type="primary",
        disabled=(not bool(selected_tickers) or workflow_has_running_job),
        key="run_daily_trading_workflow",
    )

    if daily_workflow_clicked:
        workflow_summary = {
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "selected_count": len(selected_tickers),
            "updated_count": 0,
            "complete_count": 0,
            "unfinished_count": 0,
            "background_job_id": None,
            "status": "OK",
            "message": "Workflow harian selesai. Cek ringkasan dan status job background sebelum mengambil keputusan trading.",
            "details": [],
        }

        try:
            if workflow_run_update:
                progress = st.progress(0)
                status_box = st.empty()
                processed = {"count": 0}
                seen_tickers = set()
                total_selected = max(len(selected_tickers), 1)

                def show_daily_workflow_progress(ticker_name, message):
                    if ticker_name not in seen_tickers:
                        seen_tickers.add(ticker_name)
                        processed["count"] = min(processed["count"] + 1, total_selected)
                    progress.progress(processed["count"] / total_selected)
                    status_box.info(message)

                with st.spinner("Langkah 1/5: mengupdate data harga..."):
                    update_summary = run_auto_updater(
                        tickers=selected_tickers,
                        progress_callback=show_daily_workflow_progress,
                        sleep_seconds=float(sleep_seconds),
                    )
                    st.session_state["last_data_update_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["last_auto_update_dt"] = datetime.now()
                    st.session_state["last_update_summary"] = update_summary
                    st.session_state["local_data_status"] = get_local_data_status(selected_tickers)
                    workflow_summary["updated_count"] = len(update_summary.get("updated", []))
                    workflow_summary["details"].append({
                        "langkah": "Update data harga",
                        "status": "SELESAI",
                        "ringkasan": f"{workflow_summary['updated_count']} saham berhasil diperbarui.",
                    })
                    progress.progress(1.0)
                    status_box.success("Update data selesai.")
            else:
                workflow_summary["details"].append({
                    "langkah": "Update data harga",
                    "status": "DILEWATI",
                    "ringkasan": "Langkah update data tidak dipilih.",
                })

            if workflow_repair_ohlc:
                with st.spinner("Langkah 2/5: audit dan perbaikan kualitas OHLC..."):
                    ohlc_audit_df = audit_and_repair_local_ohlc(selected_tickers, apply_repair=True)
                    st.session_state["ohlc_quality_audit"] = ohlc_audit_df
                    st.session_state["local_data_status"] = get_local_data_status(selected_tickers)
                    repaired_count = 0
                    if not ohlc_audit_df.empty and "repaired_rows" in ohlc_audit_df.columns:
                        repaired_count = int(pd.to_numeric(ohlc_audit_df["repaired_rows"], errors="coerce").fillna(0).sum())
                    workflow_summary["details"].append({
                        "langkah": "Audit & perbaikan OHLC",
                        "status": "SELESAI",
                        "ringkasan": f"{repaired_count} baris OHLC diperbaiki.",
                    })
            else:
                workflow_summary["details"].append({
                    "langkah": "Audit & perbaikan OHLC",
                    "status": "DILEWATI",
                    "ringkasan": "Langkah perbaikan kualitas data tidak dipilih.",
                })

            if workflow_evaluate_accuracy:
                with st.spinner("Langkah 3/5: mengevaluasi prediksi yang sudah jatuh tempo..."):
                    evaluate_pending_predictions()
                    workflow_summary["details"].append({
                        "langkah": "Evaluasi prediksi pending",
                        "status": "SELESAI",
                        "ringkasan": "Prediksi yang sudah memiliki data aktual dievaluasi ulang.",
                    })
            else:
                workflow_summary["details"].append({
                    "langkah": "Evaluasi prediksi pending",
                    "status": "DILEWATI",
                    "ringkasan": "Langkah evaluasi akurasi tidak dipilih.",
                })

            if workflow_predict_saved_models:
                with st.spinner("Langkah 4/5: menjalankan prediksi Global Model..."):
                    progress_callback = create_live_analysis_tracker("Live Progress Prediksi Global Model")
                    saved_prediction_summary = predict_with_global_models(
                        tickers=selected_tickers,
                        duplicate_policy="skip",
                        prediction_run_type="FINAL",
                        progress_callback=progress_callback,
                    )
                    st.session_state["last_global_model_prediction_summary"] = saved_prediction_summary
                    workflow_summary["details"].append({
                        "langkah": "Prediksi Global Model",
                        "status": "SELESAI",
                        "ringkasan": (
                            f"{len(saved_prediction_summary.get('predicted', []))} saham berhasil, "
                            f"{len(saved_prediction_summary.get('skipped', []))} dilewati, "
                            f"{len(saved_prediction_summary.get('failed', []))} gagal."
                        ),
                    })
            else:
                workflow_summary["details"].append({
                    "langkah": "Prediksi Global Model",
                    "status": "DILEWATI",
                    "ringkasan": "Langkah prediksi harian Global Model tidak dipilih.",
                })

            with st.spinner("Langkah 5/5: audit kelengkapan analisis harian..."):
                completion_df = build_analysis_completion_status(
                    selected_tickers,
                    required_models=workflow_required_models,
                )
                st.session_state["analysis_completion_status"] = completion_df
                unfinished_tickers = []
                if not completion_df.empty:
                    workflow_summary["complete_count"] = int((completion_df["status_analisis"] == "LENGKAP").sum())
                    workflow_summary["unfinished_count"] = int((completion_df["status_analisis"] == "BELUM SELESAI").sum())
                    unfinished_tickers = completion_df.loc[
                        completion_df["status_analisis"] == "BELUM SELESAI",
                        "ticker",
                    ].dropna().astype(str).tolist()

                workflow_summary["details"].append({
                    "langkah": "Audit kelengkapan analisis",
                    "status": "SELESAI",
                    "ringkasan": f"{workflow_summary['complete_count']} lengkap, {workflow_summary['unfinished_count']} belum selesai.",
                })

            should_run_training = False
            if should_run_training:
                job_id = start_background_analysis_job(
                    selected_tickers,
                    lstm_epochs=int(workflow_lstm_epochs),
                    duplicate_policy="skip",
                    prediction_run_type="FINAL",
                    force_retrain=True,
                )
                workflow_summary["background_job_id"] = job_id
                workflow_summary["status"] = "PARTIAL"
                workflow_summary["message"] = (
                    "Workflow harian selesai dan retrain berjalan di background sesuai policy. "
                    "Prediksi lama tetap aman karena duplikat dilewati."
                )
                workflow_summary["details"].append({
                    "langkah": "Retrain policy background",
                    "status": "BERJALAN",
                    "ringkasan": f"{len(selected_tickers)} saham masuk job background {job_id}.",
                })
            elif workflow_retrain_if_due:
                workflow_summary["details"].append({
                    "langkah": "Retrain policy background",
                    "status": "TIDAK PERLU",
                    "ringkasan": "Model sudah tersedia dan policy belum menandai retrain.",
                })
            else:
                workflow_summary["details"].append({
                    "langkah": "Retrain policy background",
                    "status": "DILEWATI",
                    "ringkasan": "Retrain tidak dipilih untuk workflow ini.",
                })

        except Exception as e:
            workflow_summary["status"] = "FAILED"
            workflow_summary["message"] = f"Workflow harian berhenti karena error: {e}"
            workflow_summary["details"].append({
                "langkah": "Workflow harian",
                "status": "GAGAL",
                "ringkasan": str(e),
            })

        st.session_state["last_daily_workflow_summary"] = workflow_summary
        render_daily_workflow_summary(workflow_summary)
        if workflow_summary.get("background_job_id"):
            render_background_analysis_job(workflow_summary["background_job_id"])

    if "last_daily_workflow_summary" in st.session_state and not daily_workflow_clicked:
        render_daily_workflow_summary(st.session_state["last_daily_workflow_summary"])

    st.markdown("---")
    st.subheader("5. Backfill Prediksi Hari Terlewat")
    st.caption(
        "Gunakan ini jika Anda lupa menjalankan analisis beberapa hari. Backfill melakukan replay per tanggal historis "
        "dengan data dipotong sampai tanggal tersebut, lalu mencatat hasil sebagai `BACKFILL` agar tidak bocor data masa depan."
    )
    default_backfill_end = datetime.now().date()
    default_backfill_start = default_backfill_end - timedelta(days=7)
    bf_scope_col, bf_col1, bf_col2, bf_col3 = st.columns([1.3, 1, 1, 1])
    with bf_scope_col:
        backfill_scope = st.radio(
            "Cakupan backfill",
            ["Saham terpilih", "Semua saham dengan data OK"],
            horizontal=False,
            key="backfill_scope",
            help="Semua saham dengan data OK memakai daftar config/stocks.yaml dan hanya mengambil saham yang data lokalnya siap.",
        )
    with bf_col1:
        backfill_start_date = st.date_input(
            "Tanggal mulai backfill",
            value=default_backfill_start,
            key="backfill_start_date",
        )
    with bf_col2:
        backfill_end_date = st.date_input(
            "Tanggal akhir backfill",
            value=default_backfill_end,
            key="backfill_end_date",
        )
    with bf_col3:
        backfill_max_days = st.number_input(
            "Maks. tanggal per saham",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Batas pengaman agar replay tidak terlalu lama saat banyak saham dipilih.",
            key="backfill_max_days_per_ticker",
        )

    bf_opt1, bf_opt2 = st.columns(2)
    with bf_opt1:
        backfill_include_lstm = st.checkbox(
            "Sertakan LSTM saat backfill",
            value=False,
            help="Lebih lambat. Default mati agar backfill beberapa hari tetap ringan.",
            key="backfill_include_lstm",
        )
    with bf_opt2:
        backfill_evaluate_after = st.checkbox(
            "Evaluasi otomatis setelah backfill",
            value=True,
            help="Jika data aktual target sudah tersedia, hasil backfill langsung masuk tab Akurasi Model.",
            key="backfill_evaluate_after",
        )

    backfill_clicked = st.button(
        "Jalankan Backfill Prediksi Hari Terlewat",
        type="primary",
        disabled=not bool(selected_tickers if backfill_scope == "Saham terpilih" else tickers),
        key="run_prediction_backfill",
    )
    if backfill_clicked:
        if backfill_start_date > backfill_end_date:
            st.error("Tanggal mulai backfill tidak boleh lebih besar dari tanggal akhir.")
        else:
            if backfill_scope == "Semua saham dengan data OK":
                all_status_df = get_local_data_status(tickers)
                backfill_tickers = (
                    all_status_df.loc[all_status_df["status"] == "OK", "ticker"]
                    .dropna()
                    .astype(str)
                    .map(normalize_ticker_code)
                    .drop_duplicates()
                    .tolist()
                    if not all_status_df.empty
                    else []
                )
                st.info(f"Backfill semua saham: {len(backfill_tickers)} saham memiliki data lokal OK.")
            else:
                backfill_tickers = [normalize_ticker_code(t) for t in selected_tickers if normalize_ticker_code(t)]

            if not backfill_tickers:
                st.warning("Tidak ada saham dengan data lokal OK untuk backfill.")
                st.stop()

            with st.spinner(f"Menjalankan backfill untuk {len(backfill_tickers)} saham..."):
                progress_callback = create_live_analysis_tracker("Live Progress Backfill Prediksi")
                backfill_summary = run_backfill_analysis(
                    tickers=backfill_tickers,
                    start_date=backfill_start_date.strftime("%Y-%m-%d"),
                    end_date=backfill_end_date.strftime("%Y-%m-%d"),
                    max_days_per_ticker=int(backfill_max_days),
                    lstm_epochs=int(workflow_lstm_epochs) if "workflow_lstm_epochs" in locals() else 1,
                    include_lstm=bool(backfill_include_lstm),
                    progress_callback=progress_callback,
                )
                st.session_state["last_backfill_summary"] = backfill_summary
                if backfill_evaluate_after:
                    evaluate_pending_predictions()
            st.success("Backfill selesai. Cek ringkasan di bawah dan tab Akurasi Model.")
            render_analysis_summary(backfill_summary, show_analyzed=True)

    if "last_backfill_summary" in st.session_state and not backfill_clicked:
        with st.expander("Ringkasan Backfill Terakhir", expanded=False):
            render_analysis_summary(st.session_state["last_backfill_summary"], show_analyzed=True)

    st.subheader("Riwayat Backfill")
    history_scope = st.radio(
        "Tampilkan riwayat untuk",
        ["Saham terpilih", "Semua saham"],
        horizontal=True,
        key="backfill_history_scope",
    )
    history_filter_tickers = selected_tickers if history_scope == "Saham terpilih" else None
    backfill_history_df = build_backfill_history_view(history_filter_tickers)
    if backfill_history_df.empty:
        st.info("Belum ada riwayat backfill untuk cakupan ini.")
    else:
        h_col1, h_col2, h_col3 = st.columns(3)
        status_options = ["SEMUA"] + sorted(backfill_history_df["status"].dropna().astype(str).unique().tolist())
        purpose_options = ["SEMUA"] + sorted(backfill_history_df["prediction_purpose"].dropna().astype(str).unique().tolist())
        model_options = ["SEMUA"] + sorted(backfill_history_df["model_name"].dropna().astype(str).unique().tolist())
        selected_history_status = h_col1.selectbox("Filter status evaluasi", status_options, key="backfill_history_status")
        selected_history_purpose = h_col2.selectbox("Filter jenis prediksi", purpose_options, key="backfill_history_purpose")
        selected_history_model = h_col3.selectbox("Filter model", model_options, key="backfill_history_model")

        view_backfill_history_df = backfill_history_df.copy()
        if selected_history_status != "SEMUA":
            view_backfill_history_df = view_backfill_history_df[view_backfill_history_df["status"] == selected_history_status]
        if selected_history_purpose != "SEMUA":
            view_backfill_history_df = view_backfill_history_df[view_backfill_history_df["prediction_purpose"] == selected_history_purpose]
        if selected_history_model != "SEMUA":
            view_backfill_history_df = view_backfill_history_df[view_backfill_history_df["model_name"] == selected_history_model]

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Baris Backfill", len(view_backfill_history_df))
        b2.metric("Saham", view_backfill_history_df["ticker"].nunique() if not view_backfill_history_df.empty else 0)
        b3.metric("Tanggal Prediksi", view_backfill_history_df["current_date"].nunique() if not view_backfill_history_df.empty else 0)
        b4.metric("Pending", int((view_backfill_history_df["status"] == "PENDING").sum()) if not view_backfill_history_df.empty else 0)

        st.dataframe(
            view_backfill_history_df.style.format({
                "timestamp_prediction": lambda value: "" if pd.isna(value) else value.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_return_pct": "{:+.2f}%",
                "confidence_pct": "{:.2f}%",
                "horizon_days": "{:.0f}",
            }),
            width="stretch",
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("6. Status Data Lokal")
    if selected_tickers:
        status_df = get_local_data_status(selected_tickers)
        if not status_df.empty:
            latest_dates = status_df[status_df["status"] == "OK"]["last_date"].dropna()
            if not latest_dates.empty:
                st.info(f"Data lokal terakhir: {latest_dates.max()}")
            st.dataframe(status_df, width="stretch")

    col_status, col_force = st.columns(2)
    with col_status:
        if st.button("Cek Status Update Harga", disabled=not bool(selected_tickers)):
            st.session_state["local_data_status"] = get_local_data_status(selected_tickers)

    with col_force:
        force_analysis_clicked = st.button(
            "Paksa Analisis Ulang dari Data Lokal",
            disabled=(not bool(selected_tickers) or not LEGACY_MODELS_ENABLED),
            help="Dinonaktifkan karena model lama per-saham tidak lagi dipakai.",
        )

    if "local_data_status" in st.session_state:
        st.dataframe(st.session_state["local_data_status"], width="stretch")

    st.markdown("---")
    st.subheader("7. Audit & Perbaiki Kualitas OHLC")
    st.caption(
        "Gunakan ini jika analisis gagal karena high lebih rendah dari open/close atau low lebih tinggi dari open/close. "
        "Perbaikan hanya menormalkan batas OHLC lokal, bukan mengubah arah prediksi secara manual."
    )
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if st.button("Cek Anomali OHLC Lokal", disabled=not bool(selected_tickers)):
            st.session_state["ohlc_quality_audit"] = audit_and_repair_local_ohlc(
                selected_tickers,
                apply_repair=False,
            )
    with q_col2:
        if st.button("Perbaiki Anomali OHLC Lokal", disabled=not bool(selected_tickers)):
            st.session_state["ohlc_quality_audit"] = audit_and_repair_local_ohlc(
                selected_tickers,
                apply_repair=True,
            )
            st.session_state["local_data_status"] = get_local_data_status(selected_tickers)
            st.success("Perbaikan OHLC selesai. Jalankan audit kelengkapan analisis atau analisis ulang untuk saham terkait.")

    if "ohlc_quality_audit" in st.session_state:
        ohlc_audit_df = st.session_state["ohlc_quality_audit"]
        if not ohlc_audit_df.empty:
            st.dataframe(ohlc_audit_df, width="stretch", hide_index=True)
            needs_repair = ohlc_audit_df[ohlc_audit_df["status"] == "PERLU DIPERBAIKI"]
            missing_files = ohlc_audit_df[ohlc_audit_df["status"] == "FILE TIDAK ADA"]
            if not needs_repair.empty:
                st.warning(
                    "Ada saham dengan OHLC tidak konsisten. Klik Perbaiki Anomali OHLC Lokal, lalu jalankan analisis ulang untuk ticker tersebut."
                )
            if not missing_files.empty:
                st.info(
                    "Sebagian file data tidak ada. Untuk kasus ini, jalankan update data online atau upload CSV manual."
                )

    st.markdown("---")
    st.subheader("8. Job Background Analisis")
    st.caption(
        "Gunakan mode background untuk analisis banyak saham. Proses tetap berjalan sebagai proses terpisah, "
        "sementara dashboard hanya membaca progres dari file status."
    )
    bg_col1, bg_col2 = st.columns([1, 1])
    with bg_col1:
        background_lstm_epochs = st.number_input(
            "Epoch LSTM untuk job background",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="Untuk proses harian, 3 epoch biasanya cukup cepat. Naikkan jika ingin model LSTM belajar lebih lama.",
        )
    with bg_col2:
        execution_mode = st.radio(
            "Mode eksekusi analisis",
            ["Background", "Langsung di halaman"],
            horizontal=True,
            help="Background disarankan untuk banyak saham karena tetap berjalan walau dashboard rerun atau Anda berpindah tab.",
        )

    recent_jobs = list_analysis_jobs(limit=10)
    if recent_jobs:
        job_options = [job["job_id"] for job in recent_jobs]
        selected_job_id = st.selectbox(
            "Pantau job background",
            options=job_options,
            index=job_options.index(st.session_state["active_analysis_job_id"]) if st.session_state.get("active_analysis_job_id") in job_options else 0,
            format_func=lambda job_id: next(
                (
                    f"{job_id} | {job['status']} | {job['updated_at']}"
                    for job in recent_jobs
                    if job["job_id"] == job_id
                ),
                job_id,
            ),
        )
        if st.button("Gunakan Job Ini sebagai Job Aktif"):
            st.session_state["active_analysis_job_id"] = selected_job_id
            st.rerun()

    active_job_id = st.session_state.get("active_analysis_job_id")
    if active_job_id:
        render_background_analysis_job(active_job_id)
        if st.button("Refresh Status Job Background"):
            st.rerun()

    st.markdown("---")
    st.subheader("9. Audit Kelengkapan Analisis Setelah Update")
    st.caption(
        "Fitur ini mengecek apakah setiap saham sudah punya prediksi H+3 untuk tanggal data lokal terakhir. "
        "Gunakan sebelum trading harian untuk memastikan tidak ada saham yang tertinggal setelah update data."
    )

    required_analysis_models = st.multiselect(
        "Model wajib untuk dianggap lengkap",
        options=["XGBoost", "LSTM"],
        default=["XGBoost"],
        help="XGBoost menjadi default karena selalu dipakai oleh pipeline utama. Aktifkan LSTM jika PyTorch sudah tersedia dan Anda ingin mewajibkan prediksi LSTM juga.",
    )
    audit_clicked = st.button("Cek Saham yang Belum Selesai Dianalisis", disabled=not bool(selected_tickers))

    if audit_clicked:
        st.session_state["analysis_completion_status"] = build_analysis_completion_status(
            selected_tickers,
            required_models=required_analysis_models,
        )

    if "analysis_completion_status" in st.session_state:
        completion_df = st.session_state["analysis_completion_status"]
        if completion_df.empty:
            st.info("Belum ada saham yang bisa diaudit.")
        else:
            total_ready = int((completion_df["status_analisis"] == "LENGKAP").sum())
            total_unfinished = int((completion_df["status_analisis"] == "BELUM SELESAI").sum())
            total_data_issue = int((completion_df["status_analisis"] == "DATA BELUM SIAP").sum())
            c_ready, c_unfinished, c_data_issue = st.columns(3)
            c_ready.metric("Analisis Lengkap", total_ready)
            c_unfinished.metric("Belum Selesai", total_unfinished)
            c_data_issue.metric("Data Belum Siap", total_data_issue)
            st.dataframe(completion_df, width="stretch", hide_index=True)

            unfinished_tickers = completion_df.loc[
                completion_df["status_analisis"] == "BELUM SELESAI",
                "ticker",
            ].dropna().astype(str).tolist()

            rerun_unfinished_clicked = st.button(
                "Jalankan Ulang Analisis untuk Saham yang Belum Selesai",
                type="primary",
                disabled=(not bool(unfinished_tickers) or not LEGACY_MODELS_ENABLED),
            )

            if rerun_unfinished_clicked:
                if execution_mode == "Background":
                    job_id = start_background_analysis_job(
                        unfinished_tickers,
                        lstm_epochs=int(background_lstm_epochs),
                        duplicate_policy=daily_duplicate_policy,
                        prediction_run_type=daily_prediction_run_type,
                    )
                    st.success(f"Job background dimulai untuk {len(unfinished_tickers)} saham. Job ID: {job_id}")
                    render_background_analysis_job(job_id)
                else:
                    with st.spinner(f"Menjalankan ulang analisis untuk {len(unfinished_tickers)} saham yang belum selesai..."):
                        progress_callback = create_live_analysis_tracker("Live Progress Analisis Saham Tertinggal")
                        analysis_summary = run_full_analysis(
                            tickers=unfinished_tickers,
                            lstm_epochs=int(background_lstm_epochs),
                            progress_callback=progress_callback,
                            duplicate_policy=daily_duplicate_policy,
                            prediction_run_type=daily_prediction_run_type,
                        )
                        st.session_state["last_auto_analysis_summary"] = analysis_summary
                        st.session_state["analysis_completion_status"] = build_analysis_completion_status(
                            selected_tickers,
                            required_models=required_analysis_models,
                        )
                    st.success("Analisis ulang saham yang belum selesai sudah dijalankan.")
                    render_analysis_summary(analysis_summary)

    if force_analysis_clicked:
        force_tickers = [str(ticker).replace(".JK", "").upper().strip() for ticker in selected_tickers if str(ticker).strip()]
        if execution_mode == "Background":
            job_id = start_background_analysis_job(
                force_tickers,
                lstm_epochs=int(background_lstm_epochs),
                duplicate_policy=daily_duplicate_policy,
                prediction_run_type=daily_prediction_run_type,
            )
            st.success(f"Job background analisis ulang dimulai untuk {len(force_tickers)} saham. Job ID: {job_id}")
            render_background_analysis_job(job_id)
        else:
            with st.spinner(f"Menjalankan analisis ulang dari data lokal untuk {len(force_tickers)} saham..."):
                analysis_summary = run_full_analysis(
                    tickers=force_tickers,
                    lstm_epochs=int(background_lstm_epochs),
                    duplicate_policy=daily_duplicate_policy,
                    prediction_run_type=daily_prediction_run_type,
                )
                st.session_state["last_auto_analysis_summary"] = analysis_summary
            st.success("Analisis ulang dari data lokal selesai.")

    with st.expander("Fallback Manual: Upload CSV Investing.com / IDX"):
        manual_ticker = st.text_input(
            "Ticker untuk file manual",
            value=(selected_tickers[0] if selected_tickers else "BBRI"),
            key="manual_update_ticker",
        ).strip().upper()
        manual_file = st.file_uploader(
            "Upload CSV harga manual",
            type=["csv"],
            key="manual_price_csv",
            help="Mendukung format standar timestamp/open/high/low/close/volume dan sebagian format Investing.com: Tanggal, Terakhir, Buka, Tinggi, Rendah, Vol.",
        )
        if st.button("Import CSV Manual ke Data Lokal", disabled=manual_file is None or not manual_ticker):
            try:
                manual_df = pd.read_csv(manual_file)
                manual_summary = update_from_manual_dataframe(
                    manual_ticker,
                    manual_df,
                    source_name="manual_investing_idx_csv",
                )
                st.session_state["last_manual_update_summary"] = manual_summary
                st.success(
                    f"CSV manual berhasil diimport untuk {manual_summary['ticker']}. "
                    f"Data lokal terakhir: {manual_summary['last_date']}."
                )
            except Exception as e:
                st.error(f"Gagal import CSV manual: {e}")

        if "last_manual_update_summary" in st.session_state:
            st.json(st.session_state["last_manual_update_summary"])

    st.markdown("---")
    auto_enabled = st.checkbox(
        "Aktifkan update berkala saat dashboard terbuka",
        value=False,
        disabled=offline_mode,
    )
    if offline_mode:
        auto_enabled = False
    interval_minutes = st.number_input("Interval update berkala (menit)", min_value=5, max_value=1440, value=60, step=5)
    rerun_analysis_after_update = st.checkbox(
        "Jalankan analisis ulang otomatis setelah update data",
        value=False,
        disabled=not LEGACY_MODELS_ENABLED,
        help="Setelah data harga diperbarui, dashboard akan membuat ulang prediksi/ranking untuk ticker terkait.",
    )
    analysis_scope_after_update = st.radio(
        "Cakupan analisis ulang",
        ["Hanya saham yang berhasil diperbarui", "Semua saham yang dipilih"],
        horizontal=True,
        disabled=not rerun_analysis_after_update,
    )
    analysis_lstm_epochs = st.number_input(
        "Epoch LSTM untuk analisis otomatis",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        disabled=not rerun_analysis_after_update,
        help="Semakin besar epoch, analisis lebih lama. Untuk update otomatis, 3 epoch biasanya cukup.",
    )
    st.caption(
        "Epoch LSTM = jumlah putaran belajar model dari data historis. "
        "Nilai kecil lebih cepat dan cocok untuk analisis harian; nilai terlalu besar bisa lebih lama dan berisiko overfit pada pola lama."
    )

    if auto_enabled:
        refresh_seconds = int(interval_minutes * 60)
        st.caption(f"Dashboard akan refresh setiap {interval_minutes} menit selama tab browser ini terbuka.")
        st.markdown(f"<meta http-equiv='refresh' content='{refresh_seconds}'>", unsafe_allow_html=True)

    last_update = st.session_state.get("last_data_update_at")
    if last_update:
        st.info(f"Update terakhir: {last_update}")

    should_auto_update = False
    if auto_enabled and selected_tickers:
        last_auto_update = st.session_state.get("last_auto_update_dt")
        if last_auto_update is None or (datetime.now() - last_auto_update) >= timedelta(minutes=int(interval_minutes)):
            should_auto_update = True

    update_clicked = st.button(
        "Update Harga Saja",
        type="primary",
        disabled=offline_mode or not bool(selected_tickers),
    )

    if update_clicked or should_auto_update:
        progress = st.progress(0)
        status_box = st.empty()
        processed = {"count": 0}
        seen_tickers = set()
        total_selected = max(len(selected_tickers), 1)

        def show_progress(ticker_name, message):
            if ticker_name not in seen_tickers:
                seen_tickers.add(ticker_name)
                processed["count"] = min(processed["count"] + 1, total_selected)
            progress.progress(processed["count"] / total_selected)
            status_box.info(message)

        with st.spinner("Mengupdate data harga terbaru dari sumber online..."):
            summary = run_auto_updater(
                tickers=selected_tickers,
                progress_callback=show_progress,
                sleep_seconds=float(sleep_seconds),
            )
            st.session_state["last_data_update_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["last_auto_update_dt"] = datetime.now()
            st.session_state["last_update_summary"] = summary
            progress.progress(1.0)
            status_box.success("Update data selesai.")

        if rerun_analysis_after_update:
            updated_tickers = [row.get("ticker") for row in summary.get("updated", []) if row.get("ticker")]
            analysis_tickers = updated_tickers if analysis_scope_after_update == "Hanya saham yang berhasil diperbarui" else selected_tickers
            analysis_tickers = [str(ticker).replace(".JK", "").upper().strip() for ticker in analysis_tickers if str(ticker).strip()]

            if analysis_tickers:
                with st.spinner(f"Menjalankan analisis ulang untuk {len(analysis_tickers)} saham..."):
                    progress_callback = create_live_analysis_tracker("Live Progress Analisis Ulang")
                    analysis_summary = run_full_analysis(
                        tickers=analysis_tickers,
                        lstm_epochs=int(analysis_lstm_epochs),
                        progress_callback=progress_callback,
                        duplicate_policy=daily_duplicate_policy,
                        prediction_run_type=daily_prediction_run_type,
                    )
                    st.session_state["last_auto_analysis_summary"] = analysis_summary
                st.success("Analisis ulang selesai. Tab Ranking Prediksi dan Akurasi Model sudah memakai prediksi terbaru.")
            else:
                st.info("Tidak ada saham yang perlu dianalisis ulang dari hasil update ini.")

    if "last_update_summary" in st.session_state:
        render_update_summary(st.session_state["last_update_summary"])

    if "last_auto_analysis_summary" in st.session_state:
        analysis_summary = st.session_state["last_auto_analysis_summary"]
        st.subheader("Ringkasan Analisis Ulang Otomatis")
        render_analysis_summary(analysis_summary)

with tab_ranking:
    st.header("Ranking Prediksi")
    st.write("Menampilkan potensi kenaikan dan penurunan berdasarkan hasil prediksi model terbaru.")
    
    pred_file = project_path("data", "tracking", "predictions_log.csv")
    pred_status_df = load_prediction_log(pred_file)
    clean_preview_df, prediction_audit = audit_prediction_csv(pred_file)
    with st.expander("Audit & Pembersihan Log Prediksi", expanded=not prediction_audit.is_clean):
        audit_col1, audit_col2, audit_col3 = st.columns(3)
        audit_col1.metric("Total Baris", f"{prediction_audit.total_rows:,}".replace(",", "."))
        audit_col2.metric("Baris Valid", f"{prediction_audit.valid_rows:,}".replace(",", "."))
        audit_col3.metric("Baris Bermasalah", f"{prediction_audit.invalid_rows:,}".replace(",", "."))
        if prediction_audit.missing_columns:
            st.error(f"Kolom wajib hilang: {', '.join(prediction_audit.missing_columns)}")
        elif prediction_audit.invalid_rows:
            st.warning("Ada baris abnormal di log prediksi. Ranking otomatis mengabaikan baris ini.")
            st.write("Ringkasan penyebab:")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Penyebab": reason, "Jumlah": count}
                        for reason, count in prediction_audit.invalid_reason_counts.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.write("Preview baris bermasalah:")
            st.dataframe(prediction_audit.invalid_preview, width="stretch", hide_index=True)
            if st.button("Bersihkan predictions_log.csv", type="primary"):
                try:
                    backup_path, cleaned_audit = clean_prediction_csv(
                        pred_file,
                        backup_dir=project_path("data", "tracking", "backups"),
                    )
                    st.success(
                        f"Pembersihan selesai. {cleaned_audit.invalid_rows} baris abnormal dipindahkan dari file aktif. "
                        f"Backup dibuat di `{backup_path}`."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal membersihkan log prediksi: {e}")
        else:
            st.success("Log prediksi bersih. Ranking memakai semua baris valid.")

    if pred_status_df.empty:
        render_feature_status("Ranking Prediksi", "BELUM LENGKAP", "Belum ada prediksi aktif.", "Jalankan analisis saham terlebih dahulu.")
    else:
        latest_prediction_date = pred_status_df["current_date"].dropna().astype(str).max()
        active_prediction_count = int(pred_status_df["is_active"].sum())
        render_feature_status(
            "Ranking Prediksi",
            "SIAP" if active_prediction_count else "BELUM LENGKAP",
            f"{active_prediction_count} prediksi aktif. Tanggal terbaru: {latest_prediction_date}.",
            "Gunakan filter horizon/model untuk melihat kandidat utama.",
        )
    if os.path.exists(pred_file):
        try:
            df_preds = clean_preview_df.copy() if os.path.getsize(pred_file) > 0 else pd.DataFrame()
        except EmptyDataError:
            df_preds = pd.DataFrame()
        if not df_preds.empty:
            ranking_mode = st.radio(
                "Jenis Ranking",
                ["Proyeksi Swing H+3/H+5/H+10", "Arah Harian H+1"],
                horizontal=True,
                help="H+1 memakai prediksi arah harian. Swing memakai proyeksi harga H+3, H+5, dan H+10.",
            )
            if "prediction_purpose" in df_preds.columns:
                purpose_series = df_preds["prediction_purpose"].fillna("THREE_DAY_FORECAST").astype(str).str.upper()
                if ranking_mode == "Arah Harian H+1":
                    df_preds = df_preds[purpose_series == "NEXT_DAY_DIRECTION"]
                else:
                    df_preds = df_preds[purpose_series != "NEXT_DAY_DIRECTION"]
            elif ranking_mode == "Arah Harian H+1":
                df_preds = pd.DataFrame()
            if "is_active" in df_preds.columns:
                df_preds = df_preds[df_preds["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])]
            if df_preds.empty:
                st.info("Belum ada data ranking untuk mode ini. Jalankan analisis terlebih dahulu.")
                st.stop()
            # Ambil prediksi terbaru untuk setiap saham (berdasarkan timestamp_prediction)
            df_preds['timestamp_prediction'] = pd.to_datetime(df_preds['timestamp_prediction'])
            if "horizon_days" not in df_preds.columns:
                df_preds["horizon_days"] = 3
            if "prediction_purpose" not in df_preds.columns:
                df_preds["prediction_purpose"] = "THREE_DAY_FORECAST"
            df_preds["horizon_days"] = pd.to_numeric(df_preds["horizon_days"], errors="coerce").fillna(3)
            if ranking_mode == "Arah Harian H+1":
                df_preds = df_preds[df_preds["horizon_days"].astype(int) == 1].copy()
                if df_preds.empty:
                    st.info("Belum ada prediksi H+1 untuk ranking harian. Jalankan analisis terlebih dahulu.")
                    st.stop()
            latest_preds = df_preds.sort_values('timestamp_prediction', ascending=False).drop_duplicates(
                subset=['ticker', 'model_name', 'horizon_days', 'prediction_purpose']
            )
            
            # Hitung persentase potensi pergerakan
            for price_column in ["predicted_price", "current_price"]:
                latest_preds[price_column] = pd.to_numeric(latest_preds.get(price_column), errors="coerce")
            latest_preds = latest_preds.dropna(subset=["predicted_price", "current_price"]).copy()
            latest_preds = latest_preds[latest_preds["current_price"] > 0].copy()
            if latest_preds.empty:
                st.info("Belum ada data harga prediksi yang valid untuk ranking mode ini.")
                st.stop()
            latest_preds['potential_return_pct'] = ((latest_preds['predicted_price'] - latest_preds['current_price']) / latest_preds['current_price']) * 100
            latest_preds["prediction_purpose"] = latest_preds.get("prediction_purpose", "THREE_DAY_FORECAST")
            latest_preds["confidence_pct"] = pd.to_numeric(latest_preds.get("confidence_pct", 60.0), errors="coerce").fillna(60.0)
            latest_preds["horizon_days"] = pd.to_numeric(latest_preds.get("horizon_days", 3), errors="coerce").fillna(3)

            reliability_purpose = "NEXT_DAY_DIRECTION" if ranking_mode == "Arah Harian H+1" else "THREE_DAY_FORECAST"
            reliability_df = get_best_model_recommendations(
                min_evaluations=3,
                prediction_purpose=reliability_purpose,
            )
            if reliability_df.empty:
                latest_preds["historical_reliability"] = 50.0
            else:
                reliability_lookup = reliability_df.set_index(["ticker", "model_name"])["reliability_score"].to_dict()
                latest_preds["historical_reliability"] = latest_preds.apply(
                    lambda row: reliability_lookup.get((row["ticker"], row["model_name"]), 50.0),
                    axis=1,
                )
            latest_preds["ranking_score"] = (
                latest_preds["potential_return_pct"]
                * (latest_preds["confidence_pct"] / 100.0)
                * (latest_preds["historical_reliability"] / 100.0)
            )
            
            horizon_options = sorted(latest_preds["horizon_days"].dropna().astype(int).unique().tolist())
            selected_horizon = st.selectbox(
                "Horizon Ranking",
                options=horizon_options,
                index=horizon_options.index(3) if 3 in horizon_options else 0,
                format_func=lambda value: f"H+{value}",
                disabled=ranking_mode == "Arah Harian H+1",
            )
            latest_preds = latest_preds[latest_preds["horizon_days"].astype(int) == int(selected_horizon)]
            if latest_preds.empty:
                st.info(f"Belum ada prediksi untuk horizon H+{selected_horizon}. Jalankan analisis terlebih dahulu.")
                st.stop()

            # Filter model
            models_available = latest_preds['model_name'].unique()
            selected_model = st.selectbox("Pilih Model Prediksi", options=models_available, index=0)
            
            filtered_preds = latest_preds[latest_preds['model_name'] == selected_model].copy()
            sync_ranking_ticker = st.checkbox(
                f"Tampilkan hanya ticker aktif ({ticker})",
                value=bool(not use_all_tickers and ticker and ticker in filtered_preds["ticker"].astype(str).str.upper().values),
                disabled=use_all_tickers or not bool(ticker),
            )
            if sync_ranking_ticker and ticker:
                filtered_preds = filtered_preds[filtered_preds["ticker"].astype(str).str.upper() == ticker]
            
            if not filtered_preds.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Ranking Beli")
                    top_gainers = filtered_preds.sort_values(by='ranking_score', ascending=False).head(10)
                    display_gainers = top_gainers[['ticker', 'current_price', 'predicted_price', 'potential_return_pct', 'confidence_pct', 'historical_reliability', 'ranking_score', 'target_date']].copy()
                    display_gainers.rename(columns={
                        'ticker': 'Saham',
                        'current_price': 'Harga Saat Prediksi',
                        'predicted_price': 'Harga Prediksi',
                        'potential_return_pct': 'Potensi Kenaikan (%)',
                        'confidence_pct': 'Confidence (%)',
                        'historical_reliability': 'Reliability Historis',
                        'ranking_score': 'Ranking Score',
                        'target_date': 'Tanggal Target'
                    }, inplace=True)
                    st.dataframe(display_gainers.style.format({
                        "Harga Saat Prediksi": "{:,.0f}",
                        "Harga Prediksi": "{:,.0f}",
                        "Potensi Kenaikan (%)": "{:+.2f}%",
                        "Confidence (%)": "{:.2f}%",
                        "Reliability Historis": "{:.2f}",
                        "Ranking Score": "{:+.2f}",
                    }), width="stretch")
                
                with col2:
                    st.subheader("Top Potensi Penurunan")
                    top_losers = filtered_preds.sort_values(by='potential_return_pct', ascending=True).head(10)
                    display_losers = top_losers[['ticker', 'current_price', 'predicted_price', 'potential_return_pct', 'target_date']].copy()
                    display_losers.rename(columns={
                        'ticker': 'Saham',
                        'current_price': 'Harga Saat Prediksi',
                        'predicted_price': 'Harga Prediksi',
                        'potential_return_pct': 'Potensi Penurunan (%)',
                        'target_date': 'Tanggal Target'
                    }, inplace=True)
                    st.dataframe(display_losers.style.format({
                        "Harga Saat Prediksi": "{:,.0f}",
                        "Harga Prediksi": "{:,.0f}",
                        "Potensi Penurunan (%)": "{:+.2f}%"
                    }), width="stretch")
            else:
                st.info(f"Belum ada data prediksi terbaru untuk model {selected_model}.")
        else:
            st.info("File log prediksi kosong. Silakan jalankan analisis terlebih dahulu.")
    else:
        st.warning("Belum ada data prediksi yang disimpan. Jalankan prediksi Global Model dari tab Workflow Harian atau CLI global.")


with tab_sentiment:
    st.header("Analisis Sentimen Isu Pasar Modal")
    st.write("Menilai kecenderungan sentimen isu atau berita per emiten sebagai sinyal eksternal pendukung analisis teknikal.")

    sentiment_path = os.path.join("data", "sentiment", "market_issues.csv")
    engine_status = get_sentiment_engine_status()
    engine_state = "SIAP" if engine_status["model_available"] else "FALLBACK"
    engine_detail = engine_status["description"]
    if engine_status.get("training_rows"):
        engine_detail = f"{engine_detail} Data latih: {engine_status['training_rows']} headline."
    render_feature_status(
        "Engine NLP Sentimen",
        engine_state,
        f"Aktif: {engine_status['label']}.",
        engine_detail,
    )
    with st.expander("Detail integrasi tugas_nlp_ai_trading"):
        st.write(f"**Engine:** {engine_status['label']}")
        st.write(f"**Dataset:** `{engine_status['dataset_path']}`")
        st.write(f"**Dataset tersedia:** {'Ya' if engine_status['dataset_available'] else 'Tidak'}")
        st.caption("Jika dataset tugas NLP tersedia, dashboard memakai model ML. Jika tidak tersedia, sistem otomatis memakai kamus sentimen lokal.")
        local_dataset_path = get_local_sentiment_dataset_path()
        st.write(f"**Dataset lokal AI Trading:** `{local_dataset_path}`")
        if st.button("Bangun Dataset Sentimen Lokal", key="build_local_sentiment_dataset"):
            try:
                built_df = build_local_sentiment_dataset(sentiment_path, local_dataset_path)
                st.success(f"Dataset lokal berhasil dibuat: {len(built_df)} baris.")
                if not built_df.empty:
                    st.dataframe(
                        built_df[["date", "ticker", "source", "text", "label", "label_method", "label_confidence"]].head(100),
                        width="stretch",
                    )
                st.info("Engine sentimen akan memprioritaskan dataset lokal ini pada proses berikutnya.")
            except Exception as e:
                st.error(f"Gagal membangun dataset lokal: {e}")

    st.caption("Data tersimpan lokal di `data/sentiment/market_issues.csv`, jadi bisa dipakai dari dashboard tanpa membuka VS Code.")
    if os.path.exists(sentiment_path):
        try:
            sentiment_rows = len(pd.read_csv(sentiment_path))
            render_feature_status(
                "Sentimen Pasar",
                "SIAP" if sentiment_rows else "BELUM LENGKAP",
                f"{sentiment_rows} isu/berita tersimpan.",
                "Ambil berita terbaru jika ingin konteks pasar yang lebih segar.",
            )
        except Exception as e:
            render_feature_status("Sentimen Pasar", "PERLU CEK", f"File sentimen belum bisa dibaca: {e}")
    else:
        render_feature_status("Sentimen Pasar", "BELUM LENGKAP", "Belum ada file isu pasar.", "Tambahkan isu manual atau ambil berita terbaru.")

    st.subheader("Jalankan Analisis Sentimen")
    col_online_ticker, col_online_limit = st.columns([2, 1])
    online_ticker = col_online_ticker.text_input(
        "Ticker untuk berita terbaru",
        value=ticker if "ticker" in locals() else "BBRI",
        key="sentiment_online_ticker",
    ).strip().upper()
    online_limit = col_online_limit.number_input(
        "Jumlah berita online",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        key="sentiment_online_limit",
    )
    online_query = st.text_input(
        "Query berita online",
        value=f"{online_ticker} saham" if online_ticker else "saham Indonesia",
        key="sentiment_online_query",
    )
    online_first = st.checkbox(
        "Coba ambil berita terbaru dari internet sebelum analisis",
        value=not offline_mode,
        key="sentiment_online_first",
        disabled=offline_mode,
    )
    if offline_mode:
        online_first = False
    include_article_body = st.checkbox(
        "Analisis isi artikel jika halaman berita bisa dibaca",
        value=not offline_mode,
        key="sentiment_include_article_body",
        disabled=offline_mode,
    )

    if st.button("Jalankan Analisis Sentimen Sekarang", type="primary", key="run_online_sentiment"):
        st.session_state["sentiment_last_online_status"] = None
        if online_first and online_ticker:
            try:
                with st.spinner("Menghubungkan ke internet dan mengambil berita terbaru..."):
                    fetched_rows = fetch_store_latest_sentiment(
                        sentiment_path,
                        ticker_code=online_ticker,
                        query=online_query,
                        limit=int(online_limit),
                        include_article_body=include_article_body,
                    )
                if fetched_rows:
                    st.session_state["sentiment_last_online_status"] = (
                        "success",
                        f"{len(fetched_rows)} berita terbaru berhasil diambil dari internet, diperkaya dengan isi artikel jika tersedia, dan ditambahkan ke arsip lokal.",
                    )
                else:
                    st.session_state["sentiment_last_online_status"] = (
                        "warning",
                        "Internet terhubung, tetapi tidak ada berita baru yang ditemukan. Analisis memakai arsip lokal terakhir.",
                    )
            except Exception as e:
                st.session_state["sentiment_last_online_status"] = (
                    "error",
                    f"Gagal mengambil berita online: {e}. Analisis tetap memakai arsip lokal terakhir.",
                )
        elif online_first:
            st.session_state["sentiment_last_online_status"] = (
                "warning",
                "Ticker online kosong. Analisis memakai arsip lokal terakhir.",
            )
        else:
            st.session_state["sentiment_last_online_status"] = (
                "info",
                "Mode online dimatikan. Analisis memakai arsip lokal terakhir.",
            )

    last_online_status = st.session_state.get("sentiment_last_online_status")
    if last_online_status:
        status_type, status_message = last_online_status
        if status_type == "success":
            st.success(status_message)
        elif status_type == "warning":
            st.warning(status_message)
        elif status_type == "error":
            st.error(status_message)
        else:
            st.info(status_message)

    input_tab, auto_news_tab, upload_tab, quick_tab = st.tabs([
        "Tambah Berita/Isu",
        "Ambil Berita Otomatis",
        "Upload CSV",
        "Cek Cepat Teks",
    ])

    with input_tab:
        with st.form("sentiment_issue_form", clear_on_submit=True):
            col_date, col_ticker, col_source = st.columns([1, 1, 2])
            issue_date = col_date.date_input("Tanggal")
            issue_ticker = col_ticker.text_input("Ticker", value=ticker if "ticker" in locals() else "BBRI").strip().upper()
            issue_source = col_source.text_input("Sumber", value="manual")
            issue_text = st.text_area("Isi berita/isu", height=120, placeholder="Contoh: BBRI mencatat pertumbuhan kredit yang kuat dan laba meningkat signifikan.")
            submitted_issue = st.form_submit_button("Simpan dan Analisis", type="primary")

        if submitted_issue:
            if not issue_ticker or not issue_text.strip():
                st.warning("Ticker dan isi berita/isu wajib diisi.")
            else:
                append_issue(
                    sentiment_path,
                    date=issue_date.strftime("%Y-%m-%d"),
                    ticker=issue_ticker,
                    source=issue_source or "manual",
                    text=issue_text.strip(),
                )
                result = analyze_text(issue_text)
                st.success(f"Berita/isu disimpan. Hasil cepat: {result.label} ({result.sentiment_score:+.2f}) via {result.method}")

    with auto_news_tab:
        st.caption("Mengambil berita terbaru dari Google News RSS Indonesia, lalu mencoba membaca isi artikelnya untuk analisis sentimen.")
        col_ticker, col_limit = st.columns([2, 1])
        news_ticker = col_ticker.text_input(
            "Ticker untuk pencarian berita",
            value=ticker if "ticker" in locals() else "BBRI",
            key="auto_news_ticker",
        ).strip().upper()
        news_limit = col_limit.number_input("Jumlah headline", min_value=1, max_value=30, value=10, step=1)
        news_query = st.text_input(
            "Query pencarian",
            value=f"{news_ticker} saham" if news_ticker else "saham Indonesia",
            key="auto_news_query",
        )
        news_include_article_body = st.checkbox(
            "Ambil dan analisis isi artikel",
            value=True,
            key="auto_news_include_article_body",
        )

        if st.button("Ambil Berita Terbaru", key="fetch_latest_news", type="primary", disabled=offline_mode):
            if not news_ticker:
                st.warning("Ticker wajib diisi.")
            else:
                try:
                    with st.spinner("Mengambil berita terbaru dari browser/news RSS..."):
                        news_rows = fetch_store_latest_sentiment(
                            sentiment_path,
                            news_ticker,
                            news_query,
                            news_limit,
                            include_article_body=news_include_article_body,
                        )

                    if not news_rows:
                        st.info("Belum ada berita yang ditemukan untuk query tersebut. Data sentimen lokal sebelumnya tetap dipakai.")
                    else:
                        latest_scored = analyze_dataframe(pd.DataFrame(news_rows))
                        st.success(f"{len(news_rows)} berita ditemukan, disimpan, dan dianalisis.")
                        st.dataframe(
                            latest_scored[[col for col in ["date", "ticker", "source", "text", "label", "score", "sentiment_score", "method"] if col in latest_scored.columns]],
                            width="stretch",
                        )
                except Exception as e:
                    st.error(f"Gagal mengambil berita terbaru: {e}")
                    st.caption("Cek koneksi internet atau coba query yang lebih spesifik, misalnya `BBRI saham bank rakyat`.")

    with upload_tab:
        uploaded_sentiment = st.file_uploader("Upload CSV isu pasar modal", type=["csv"], key="sentiment_csv")

        if uploaded_sentiment is not None:
            os.makedirs(os.path.dirname(sentiment_path), exist_ok=True)
            with open(sentiment_path, "wb") as f:
                f.write(uploaded_sentiment.getbuffer())
            st.success("Data isu pasar modal berhasil diperbarui.")

        st.caption("Format CSV: `date,ticker,source,text`.")

    with quick_tab:
        quick_text = st.text_area("Teks untuk dicek tanpa disimpan", key="quick_sentiment_text", height=100)
        if st.button("Analisis Teks", key="quick_sentiment_button") and quick_text.strip():
            quick_result = analyze_text(quick_text)
            q1, q2, q3 = st.columns(3)
            q1.metric("Label", quick_result.label)
            q2.metric("Skor Sentimen", f"{quick_result.sentiment_score:+.2f}")
            q3.metric("Confidence", f"{quick_result.score:.2f}")
            st.caption(f"Metode analisis: {quick_result.method}")

    st.markdown("---")

    issues_df = load_issues(sentiment_path)
    if issues_df.empty:
        st.info("Belum ada data isu. Siapkan CSV dengan kolom: date, ticker, source, text.")
    else:
        try:
            scored_df = analyze_dataframe(issues_df)
            summary_df = summarize_by_ticker(scored_df)

            ticker_options = ["SEMUA"] + sorted(scored_df["ticker"].dropna().unique().tolist())
            preferred_ticker = "SEMUA" if use_all_tickers else (online_ticker or (ticker if "ticker" in locals() else "")).upper().strip()
            default_ticker_index = ticker_options.index(preferred_ticker) if preferred_ticker in ticker_options else 0

            selected_ticker = st.selectbox(
                "Pilih emiten",
                options=ticker_options,
                index=default_ticker_index,
                help="Pilih satu emiten untuk kesimpulan trading spesifik. Pilihan SEMUA berarti ringkasan sentimen pasar gabungan.",
            )

            view_df = scored_df if selected_ticker == "SEMUA" else scored_df[scored_df["ticker"] == selected_ticker]
            if selected_ticker == "SEMUA":
                st.warning("Mode SEMUA aktif: ringkasan dan pemicu utama akan mencampur berita dari seluruh saham.")

            col1, col2, col3 = st.columns(3)
            avg_score = float(view_df["sentiment_score"].mean()) if not view_df.empty else 0.0
            dominant = view_df["label"].value_counts().index[0] if not view_df.empty else "NEUTRAL"
            col1.metric("Rata-rata Skor Sentimen", f"{avg_score:+.2f}")
            col2.metric("Label Dominan", dominant)
            col3.metric("Jumlah Isu", len(view_df))
            st.caption(interpret_signal(avg_score))

            sentiment_summary = build_trading_sentiment_summary(scored_df, selected_ticker)
            st.subheader("Ringkasan Keputusan Sentimen")
            sum1, sum2, sum3 = st.columns(3)
            sum1.metric("Bias Sentimen", sentiment_summary["bias"])
            sum2.metric("Risiko Berita", sentiment_summary["risk_level"])
            sum3.metric("Confidence Ringkasan", f"{sentiment_summary['confidence']:.2f}")

            st.info(sentiment_summary["conclusion"])
            st.write(f"**Catatan trading:** {sentiment_summary['trading_note']}")
            st.caption(
                f"Komposisi berita: {sentiment_summary['positive_count']} positif, "
                f"{sentiment_summary['negative_count']} negatif, "
                f"{sentiment_summary['neutral_count']} netral."
            )

            if sentiment_summary["key_drivers"]:
                with st.expander("Pemicu utama dari berita"):
                    for driver in sentiment_summary["key_drivers"]:
                        st.write(f"- {driver}")

            st.subheader("Ringkasan per Emiten")
            st.dataframe(summary_df, width="stretch")

            st.subheader("Detail Isu dan Hasil Sentimen")
            display_columns = [col for col in ["date", "ticker", "source", "text", "label", "score", "sentiment_score", "method", "positive_hits", "negative_hits"] if col in view_df.columns]
            st.dataframe(view_df[display_columns], width="stretch")
        except Exception as e:
            st.error(f"Gagal memproses data sentimen: {e}")

with tab_accuracy:
    st.header("Track Record & Akurasi Model")
    st.write("Mengevaluasi prediksi sebelumnya berdasarkan pergerakan harga aktual.")
    st.caption("Default memakai prediksi arah H+1. Data evaluasi lama tetap bisa dilihat melalui pilihan jenis evaluasi.")
    accuracy_file = project_path("data", "tracking", "accuracy_log.csv")
    if os.path.exists(accuracy_file):
        try:
            accuracy_rows = len(pd.read_csv(accuracy_file)) if os.path.getsize(accuracy_file) > 0 else 0
            pending_status = summarize_prediction_status()
            pending_count = 0
            if not pending_status.empty:
                pending_count = int(pending_status[pending_status["status"] == "PENDING"]["jumlah_prediksi"].sum())
            render_feature_status(
                "Akurasi Model",
                "SIAP" if accuracy_rows else "BELUM LENGKAP",
                f"{accuracy_rows} evaluasi tersimpan, {pending_count} prediksi masih pending.",
                "Klik Evaluasi Ulang setelah data harga target tersedia.",
            )
        except Exception as e:
            render_feature_status("Akurasi Model", "PERLU CEK", f"File akurasi belum bisa dibaca: {e}")
    else:
        render_feature_status("Akurasi Model", "BELUM LENGKAP", "Belum ada log akurasi.", "Jalankan analisis, update data berikutnya, lalu evaluasi.")
    
    if st.button("Evaluasi Ulang (Update Akurasi)"):
        with st.spinner("Mengevaluasi prediksi dengan data terbaru..."):
            evaluate_pending_predictions()
            st.success("Evaluasi selesai!")
            
    min_samples = st.number_input(
        "Minimal evaluasi untuk model dianggap cukup reliabel",
        min_value=1,
        max_value=50,
        value=3,
        step=1,
    )

    accuracy_view_options = {
        "Akurasi Arah H+1": "NEXT_DAY_DIRECTION",
        "Akurasi Historis Lama": "MODEL_ACCURACY",
        "Akurasi Proyeksi 3 Hari": "THREE_DAY_FORECAST",
        "Akurasi Tren H+5": "H5_TREND_FORECAST",
        "Akurasi Tren H+10": "H10_TREND_FORECAST",
        "Semua Catatan Akurasi": None,
    }
    accuracy_view_label = st.selectbox(
        "Jenis evaluasi akurasi",
        options=list(accuracy_view_options.keys()),
        index=0,
        help="Pilih Akurasi Historis Lama untuk melihat catatan evaluasi yang dibuat sebelum fitur H+1 ditambahkan.",
    )
    accuracy_purpose = accuracy_view_options[accuracy_view_label]
    if accuracy_purpose == "NEXT_DAY_DIRECTION":
        st.info("Mode H+1: model hari ini memprediksi NAIK/TURUN untuk hari perdagangan berikutnya, lalu dicocokkan setelah data update tersedia.")
    elif accuracy_purpose == "MODEL_ACCURACY":
        st.info("Mode historis lama: menampilkan evaluasi prediksi yang dibuat sebelum label akurasi H+1/proyeksi 3 hari dipisahkan.")
    elif accuracy_purpose == "THREE_DAY_FORECAST":
        st.info("Mode proyeksi 3 hari: menampilkan evaluasi arah dari prediksi harga H+3.")
    elif accuracy_purpose in ["H5_TREND_FORECAST", "H10_TREND_FORECAST"]:
        st.info("Mode tren mingguan: evaluasi dipisahkan dari H+1 dan H+3 agar karakter horizon tidak tercampur.")
    else:
        st.warning("Mode semua catatan akan mencampur evaluasi lama, H+1, dan proyeksi 3 hari. Gunakan hanya untuk audit arsip.")

    summary_df = get_model_accuracy_summary(prediction_purpose=accuracy_purpose)
    daily_recap_df = get_daily_accuracy_recap(prediction_purpose=accuracy_purpose)
    overall_daily_recap_df = get_overall_daily_accuracy_recap(prediction_purpose=accuracy_purpose)
    recommendations_df = get_best_model_recommendations(min_evaluations=int(min_samples), prediction_purpose=accuracy_purpose)
    trading_leaderboard_df = get_model_trading_leaderboard(
        min_evaluations=int(min_samples),
        prediction_purpose=accuracy_purpose,
    )
    calibration_summary_df = get_confidence_calibration_summary(prediction_purpose=accuracy_purpose)
    trust_audit_df = get_model_trust_audit(
        prediction_purpose=accuracy_purpose,
        min_evaluations=int(min_samples),
    )
    prediction_status_df = summarize_prediction_status(prediction_purpose=accuracy_purpose)
    liquidity_tier_df = build_liquidity_tier_table(tuple(tickers))
    summary_df = add_liquidity_tier(summary_df, liquidity_tier_df)
    daily_recap_df = add_liquidity_tier(daily_recap_df, liquidity_tier_df)
    recommendations_df = add_liquidity_tier(recommendations_df, liquidity_tier_df)
    trading_leaderboard_df = add_liquidity_tier(trading_leaderboard_df, liquidity_tier_df)
    trust_audit_df = add_liquidity_tier(trust_audit_df, liquidity_tier_df)

    if not prediction_status_df.empty:
        with st.expander("Status Prediksi: Pending vs Evaluated", expanded=False):
            st.write(
                "Analisis baru menambah prediksi berstatus PENDING. Akurasi baru berubah setelah target tanggalnya memiliki data aktual dan prediksi dievaluasi."
            )
            st.dataframe(prediction_status_df, width="stretch", hide_index=True)

    if not overall_daily_recap_df.empty:
        latest_day = overall_daily_recap_df["evaluation_day"].max()
        latest_overall = overall_daily_recap_df[overall_daily_recap_df["evaluation_day"] == latest_day]
        latest_total = int(latest_overall["total_predictions"].sum())
        latest_correct = int(latest_overall["correct_predictions"].sum())
        latest_wrong = int(latest_overall["wrong_predictions"].sum())
        latest_accuracy = (latest_correct / latest_total * 100) if latest_total else 0.0

        st.subheader("Rekap Akurasi Harian Semua Saham")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Hari Evaluasi Terakhir", latest_day)
        o2.metric("Prediksi Benar", f"{latest_correct:,}")
        o3.metric("Prediksi Salah", f"{latest_wrong:,}")
        o4.metric("Akurasi Gabungan", f"{latest_accuracy:.2f}%")
        if accuracy_purpose == "NEXT_DAY_DIRECTION" and latest_accuracy < 50.0 and os.path.exists(accuracy_file):
            try:
                raw_accuracy_df = pd.read_csv(accuracy_file)
                raw_accuracy_df["evaluation_day"] = pd.to_datetime(raw_accuracy_df["evaluation_date"], errors="coerce").dt.strftime("%Y-%m-%d")
                latest_h1_df = raw_accuracy_df[
                    (raw_accuracy_df["evaluation_day"] == str(latest_day))
                    & (raw_accuracy_df["prediction_purpose"].astype(str).str.upper() == "NEXT_DAY_DIRECTION")
                ].copy()
                if not latest_h1_df.empty:
                    pred_counts = latest_h1_df["predicted_direction"].astype(str).str.upper().value_counts().to_dict()
                    actual_counts = latest_h1_df["actual_direction"].astype(str).str.upper().value_counts().to_dict()
                    model_diag = latest_h1_df.groupby("model_name").agg(
                        total=("direction_correct", "count"),
                        benar=("direction_correct", "sum"),
                        akurasi_pct=("direction_correct", lambda x: x.mean() * 100),
                    ).reset_index()
                    st.warning(
                        "Akurasi H+1 terakhir sedang lemah. Dashboard keputusan akan memperketat confidence, "
                        "memfilter sinyal TURUN saat market breadth rebound, dan memprioritaskan H+3 + reliability."
                    )
                    st.caption(f"Bias prediksi H+1: {pred_counts} | Realisasi: {actual_counts}")
                    st.dataframe(
                        model_diag.style.format({"akurasi_pct": "{:.2f}%"}),
                        width="stretch",
                        hide_index=True,
                    )
            except Exception as e:
                st.caption(f"Diagnosis akurasi harian belum bisa dibuat: {e}")

        with st.expander("Lihat rekap semua saham per hari dan model", expanded=True):
            st.dataframe(overall_daily_recap_df.style.format({
                "direction_accuracy_pct": "{:.2f}%",
                "avg_error_margin_pct": "{:.2f}%",
            }), width="stretch")

            overall_chart_df = overall_daily_recap_df.copy()
            if not overall_chart_df.empty:
                st.line_chart(overall_chart_df, x="evaluation_day", y="direction_accuracy_pct", color="model_name")

    if not summary_df.empty:
        sync_accuracy_ticker = st.checkbox(
            f"Filter akurasi ke ticker aktif ({ticker})",
            value=bool(not use_all_tickers and ticker and ticker in summary_df["ticker"].astype(str).str.upper().values),
            disabled=use_all_tickers or not bool(ticker),
        )
        view_summary_df = summary_df.copy()
        view_daily_recap_df = daily_recap_df.copy()
        view_recommendations_df = recommendations_df.copy()
        view_trading_leaderboard_df = trading_leaderboard_df.copy()
        view_trust_audit_df = trust_audit_df.copy()
        available_tiers = ["SEMUA"]
        if "liquidity_tier" in view_summary_df.columns:
            available_tiers += sorted(view_summary_df["liquidity_tier"].dropna().unique().tolist())
        selected_liquidity_tier = st.selectbox(
            "Filter tier likuiditas",
            options=available_tiers,
            index=0,
            help="Tier dihitung dari rata-rata nilai transaksi terbaru di data/raw. Gunakan untuk membandingkan performa model pada saham likuid vs kurang likuid.",
        )
        if selected_liquidity_tier != "SEMUA":
            if "liquidity_tier" in view_summary_df.columns:
                view_summary_df = view_summary_df[view_summary_df["liquidity_tier"] == selected_liquidity_tier]
            if "liquidity_tier" in view_daily_recap_df.columns:
                view_daily_recap_df = view_daily_recap_df[view_daily_recap_df["liquidity_tier"] == selected_liquidity_tier]
            if "liquidity_tier" in view_recommendations_df.columns:
                view_recommendations_df = view_recommendations_df[view_recommendations_df["liquidity_tier"] == selected_liquidity_tier]
            if "liquidity_tier" in view_trading_leaderboard_df.columns:
                view_trading_leaderboard_df = view_trading_leaderboard_df[view_trading_leaderboard_df["liquidity_tier"] == selected_liquidity_tier]
            if "liquidity_tier" in view_trust_audit_df.columns:
                view_trust_audit_df = view_trust_audit_df[view_trust_audit_df["liquidity_tier"] == selected_liquidity_tier]
        if sync_accuracy_ticker and ticker:
            view_summary_df = view_summary_df[view_summary_df["ticker"].astype(str).str.upper() == ticker]
            view_recommendations_df = view_recommendations_df[view_recommendations_df["ticker"].astype(str).str.upper() == ticker]
            view_trading_leaderboard_df = view_trading_leaderboard_df[view_trading_leaderboard_df["ticker"].astype(str).str.upper() == ticker]
            view_trust_audit_df = view_trust_audit_df[view_trust_audit_df["ticker"].astype(str).str.upper() == ticker]
            if not view_daily_recap_df.empty:
                view_daily_recap_df = view_daily_recap_df[view_daily_recap_df["ticker"].astype(str).str.upper() == ticker]

        rec_tab, trust_tab, high_pred_tab, leaderboard_tab, tier_tab, calibration_tab, summary_tab, daily_tab = st.tabs([
            "Rekomendasi Model per Saham",
            "Model Trust Audit",
            "Prediksi Akurasi Tinggi",
            "Leaderboard Trading Model",
            "Evaluasi per Tier",
            "Kalibrasi Confidence",
            "Ringkasan Akurasi Model",
            "Drill-down Rekap Harian per Saham",
        ])

        with rec_tab:
            st.write("Model terbaik dipilih berdasarkan akurasi arah, error margin, dan jumlah sample evaluasi.")
            st.dataframe(view_recommendations_df.style.format({
                "reliability_score": "{:.2f}",
                "direction_accuracy_pct": "{:.2f}%",
                "precision_naik_pct": "{:.2f}%",
                "avg_return_after_naik_pct": "{:+.2f}%",
                "avg_error_margin_pct": "{:.2f}%",
                "avg_traded_value": "{:,.0f}",
            }), width="stretch")

        with trust_tab:
            st.write(
                "Audit ini merangkum apakah model per saham sudah layak dijadikan acuan trading berdasarkan sample, akurasi, profit factor, kalibrasi, dan proxy walk-forward."
            )
            if view_trust_audit_df.empty:
                st.info("Belum ada data trust audit untuk jenis evaluasi ini.")
            else:
                trust_counts = view_trust_audit_df["status_trust"].value_counts()
                t1, t2, t3 = st.columns(3)
                t1.metric("Layak Dipercaya", int(trust_counts.get("LAYAK DIPERCAYA", 0)))
                t2.metric("Perlu Data Lagi", int(trust_counts.get("PERLU DATA LAGI", 0)))
                t3.metric("Jangan Diikuti", int(trust_counts.get("JANGAN DIIKUTI", 0)))
                st.dataframe(view_trust_audit_df.style.format({
                    "direction_accuracy_pct": "{:.2f}%",
                    "profit_factor": "{:.2f}",
                    "calibration_gap_pct": "{:+.2f}%",
                    "walk_forward_score": "{:.2f}%",
                    "win_rate_pct": "{:.2f}%",
                    "avg_strategy_return_pct": "{:+.2f}%",
                    "trading_score": "{:.2f}",
                    "precision_naik_pct": "{:.2f}%",
                    "avg_return_after_naik_pct": "{:+.2f}%",
                }), width="stretch", hide_index=True)

        with high_pred_tab:
            st.write("Menampilkan prediksi terbaru dari model/saham yang historisnya paling akurat pada jenis evaluasi yang dipilih.")
            hp_col1, hp_col2 = st.columns(2)
            min_high_accuracy = hp_col1.number_input(
                "Minimal akurasi historis (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=5.0,
            )
            min_high_samples = hp_col2.number_input(
                "Minimal jumlah evaluasi",
                min_value=1,
                max_value=100,
                value=int(min_samples),
                step=1,
            )
            high_accuracy_predictions_df = build_high_accuracy_prediction_view(
                view_trading_leaderboard_df,
                prediction_purpose=accuracy_purpose,
                min_accuracy=float(min_high_accuracy),
                min_samples=int(min_high_samples),
            )
            if high_accuracy_predictions_df.empty:
                st.info("Belum ada prediksi terbaru dari model yang memenuhi batas akurasi dan jumlah evaluasi.")
            else:
                st.dataframe(high_accuracy_predictions_df.style.format({
                    "predicted_return_pct": "{:+.2f}%",
                    "confidence_pct": "{:.2f}%",
                    "direction_accuracy_pct": "{:.2f}%",
                    "win_rate_pct": "{:.2f}%",
                    "profit_factor": "{:.2f}",
                    "trading_score": "{:.2f}",
                }), width="stretch", hide_index=True)

        with leaderboard_tab:
            st.write("Leaderboard ini memakai accuracy, error margin, proxy return, win rate, dan profit factor per ticker/model.")
            if view_trading_leaderboard_df.empty:
                st.info("Belum ada data leaderboard untuk jenis evaluasi ini.")
            else:
                st.dataframe(view_trading_leaderboard_df.style.format({
                    "direction_accuracy_pct": "{:.2f}%",
                    "avg_error_margin_pct": "{:.2f}%",
                    "avg_strategy_return_pct": "{:+.2f}%",
                    "total_strategy_return_pct": "{:+.2f}%",
                    "win_rate_pct": "{:.2f}%",
                    "precision_naik_pct": "{:.2f}%",
                    "avg_return_after_naik_pct": "{:+.2f}%",
                    "profit_factor": "{:.2f}",
                    "trading_score": "{:.2f}",
                }), width="stretch")

        with tier_tab:
            st.write("Ringkasan ini membantu melihat apakah model lebih dapat dipercaya pada saham likuid dibanding saham kurang likuid.")
            if view_trading_leaderboard_df.empty or "liquidity_tier" not in view_trading_leaderboard_df.columns:
                st.info("Belum ada data tier likuiditas yang bisa diringkas.")
            else:
                tier_eval_df = (
                    view_trading_leaderboard_df.dropna(subset=["liquidity_tier"])
                    .groupby("liquidity_tier", as_index=False)
                    .agg(
                        jumlah_model=("model_name", "count"),
                        total_evaluations=("total_evaluations", "sum"),
                        avg_direction_accuracy_pct=("direction_accuracy_pct", "mean"),
                        avg_precision_naik_pct=("precision_naik_pct", "mean"),
                        avg_return_after_naik_pct=("avg_return_after_naik_pct", "mean"),
                        avg_profit_factor=("profit_factor", "mean"),
                        avg_trading_score=("trading_score", "mean"),
                    )
                    .sort_values("liquidity_tier")
                )
                st.dataframe(tier_eval_df.style.format({
                    "avg_direction_accuracy_pct": "{:.2f}%",
                    "avg_precision_naik_pct": "{:.2f}%",
                    "avg_return_after_naik_pct": "{:+.2f}%",
                    "avg_profit_factor": "{:.2f}",
                    "avg_trading_score": "{:.2f}",
                }), width="stretch", hide_index=True)

        with calibration_tab:
            st.write("Kalibrasi membandingkan rata-rata confidence model dengan akurasi aktual pada bucket confidence yang sama.")
            if calibration_summary_df.empty:
                st.info("Belum cukup data confidence yang sudah dievaluasi untuk membuat ringkasan kalibrasi.")
            else:
                st.dataframe(calibration_summary_df.style.format({
                    "avg_confidence_pct": "{:.2f}%",
                    "actual_accuracy_pct": "{:.2f}%",
                    "calibration_gap_pct": "{:+.2f}%",
                }), width="stretch")

        with summary_tab:
            st.dataframe(view_summary_df.style.format({
                "direction_accuracy_pct": "{:.2f}%",
                "precision_naik_pct": "{:.2f}%",
                "avg_return_after_naik_pct": "{:+.2f}%",
                "avg_error_margin_pct": "{:.2f}%"
            }), width="stretch")

        with daily_tab:
            if view_daily_recap_df.empty:
                st.info("Belum ada rekap harian yang bisa ditampilkan.")
            else:
                filter_ticker = st.selectbox(
                    "Filter ticker rekap harian",
                    options=["SEMUA"] + sorted(view_daily_recap_df["ticker"].unique().tolist()),
                    index=0 if use_all_tickers else ((["SEMUA"] + sorted(view_daily_recap_df["ticker"].unique().tolist())).index(ticker) if ticker in view_daily_recap_df["ticker"].unique().tolist() else 0),
                    key="daily_accuracy_ticker_filter",
                )
                view_daily = view_daily_recap_df if filter_ticker == "SEMUA" else view_daily_recap_df[view_daily_recap_df["ticker"] == filter_ticker]
                st.dataframe(view_daily.style.format({
                    "direction_accuracy_pct": "{:.2f}%",
                    "avg_error_margin_pct": "{:.2f}%"
                }), width="stretch")

                chart_df = view_daily.groupby(["evaluation_day", "model_name"], as_index=False).agg(
                    total_predictions=("total_evaluations", "sum"),
                    correct_predictions=("correct_predictions", "sum"),
                )
                if not chart_df.empty:
                    chart_df["direction_accuracy_pct"] = (
                        chart_df["correct_predictions"] / chart_df["total_predictions"].clip(lower=1) * 100
                    )
                if not chart_df.empty:
                    st.line_chart(chart_df, x="evaluation_day", y="direction_accuracy_pct", color="model_name")
    else:
        st.info("Belum ada data akurasi yang dievaluasi. Lakukan prediksi hari ini dan update data besok untuk melihat hasilnya.")

if LEGACY_MODELS_ENABLED and tab_dashboard is not None:
  with tab_dashboard:
    st.header("Model Lama Nonaktif")
    if not LEGACY_MODELS_ENABLED:
        st.info(
            "Model lama per-saham sudah dinonaktifkan agar tidak membingungkan. "
            "Gunakan tab harian untuk prediksi Global Model, atau jalankan training global dari VS Code."
        )
        st.code("python scripts\\train_global_models_cli.py --config config\\stocks.yaml --run-type FINAL", language="powershell")
        st.code("python scripts\\predict_global_models_cli.py --config config\\stocks.yaml --duplicate-policy skip --run-type FINAL", language="powershell")
    st.write("Panel ini hanya arsip transisi. Jalur aktif sekarang adalah Global Model di Ringkasan/Workflow Harian.")
    dashboard_completion_df = build_analysis_completion_status(active_tickers if active_tickers else tickers, required_models=["XGBoost"])
    if dashboard_completion_df.empty:
        render_feature_status("Model Lama", "BELUM LENGKAP", "Belum ada ticker yang bisa dicek.", "Pastikan config/stocks.yaml terisi.")
    else:
        dashboard_unfinished = int((dashboard_completion_df["status_analisis"] != "LENGKAP").sum())
        render_feature_status(
            "Model Lama",
            "SIAP" if dashboard_unfinished == 0 else "BELUM LENGKAP",
            f"{len(dashboard_completion_df) - dashboard_unfinished}/{len(dashboard_completion_df)} saham punya prediksi lama.",
            "Prediksi baru tidak lagi memakai pipeline lama.",
        )

    # --- UPLOAD DATA MANUAL (OFFLINE MODE) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Update Data Offline")
    uploaded_file = st.sidebar.file_uploader(f"Upload file CSV historis untuk {ticker}", type=["csv"], disabled=use_all_tickers)
    if uploaded_file is not None:
        os.makedirs(os.path.join("data", "raw"), exist_ok=True)
        file_path = os.path.join("data", "raw", f"{ticker}_raw.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Data {ticker} berhasil diperbarui di lokal!")

    if use_all_tickers:
        st.info(f"Mode semua saham aktif: batch analysis akan memproses **{len(active_tickers)}** ticker dari konfigurasi.")
    else:
        st.info(f"Saham aktif untuk analisis: **{ticker}**. Ubah kode saham melalui sidebar jika ingin menganalisis emiten lain.")

    analyze_sidebar_btn = st.sidebar.button("Analisis Lama Nonaktif", type="primary", disabled=not LEGACY_MODELS_ENABLED)
    analyze_main_btn = st.button("Analisis Lama Nonaktif", type="primary", key="analysis_main_button", disabled=not LEGACY_MODELS_ENABLED)
    analyze_all_btn = st.button(
        "Analisis Semua Saham",
        key="analysis_all_button",
        disabled=(not bool(tickers) or not LEGACY_MODELS_ENABLED),
        help="Menjalankan pipeline batch untuk seluruh saham pada config/stocks.yaml.",
    )
    analyze_all_background_btn = st.button(
        "Analisis Semua Saham di Background",
        key="analysis_all_background_button",
        disabled=(not bool(tickers) or not LEGACY_MODELS_ENABLED),
        help="Menjalankan analisis sebagai job terpisah agar tetap berjalan saat Anda berpindah tab.",
    )
    analyze_btn = analyze_sidebar_btn or analyze_main_btn

    analyze_all_requested = analyze_all_btn or (use_all_tickers and analyze_btn)

    if st.session_state.get("active_analysis_job_id"):
        with st.expander("Status Job Background Analisis", expanded=True):
            render_background_analysis_job(st.session_state["active_analysis_job_id"])
            if st.button("Refresh Status Job", key="refresh_dashboard_bg_job"):
                st.rerun()

    if analyze_all_background_btn:
        job_tickers = [str(t).replace(".JK", "").upper().strip() for t in tickers]
        job_id = start_background_analysis_job(
            job_tickers,
            lstm_epochs=int(background_lstm_epochs) if "background_lstm_epochs" in locals() else 3,
            duplicate_policy=daily_duplicate_policy if "daily_duplicate_policy" in locals() else "skip",
            prediction_run_type=daily_prediction_run_type if "daily_prediction_run_type" in locals() else "FINAL",
        )
        st.success(f"Job background analisis semua saham dimulai. Job ID: {job_id}")
        render_background_analysis_job(job_id)

    if analyze_all_requested:
        with st.spinner(f"Menjalankan analisis batch untuk {len(tickers)} saham..."):
            progress_callback = create_live_analysis_tracker("Live Progress Analisis Semua Saham")
            analysis_summary = run_full_analysis(
                tickers=[str(t).replace(".JK", "").upper().strip() for t in tickers],
                lstm_epochs=3,
                progress_callback=progress_callback,
                duplicate_policy=daily_duplicate_policy if "daily_duplicate_policy" in locals() else "skip",
                prediction_run_type=daily_prediction_run_type if "daily_prediction_run_type" in locals() else "FINAL",
            )
            st.session_state["last_auto_analysis_summary"] = analysis_summary
        st.success("Analisis semua saham selesai. Hasil terbaru dapat dilihat di Ranking Prediksi dan Akurasi Model.")
        render_analysis_summary(analysis_summary, show_analyzed=False)

    if analyze_btn and not use_all_tickers and not ticker:
        st.warning("Masukkan kode saham terlebih dahulu, misalnya BBRI atau BBCA.")

    if analyze_btn and not use_all_tickers and ticker:
        with st.spinner(f"Menganalisis pergerakan saham {ticker} (Offline Mode)..."):
            try:
                # Mengevaluasi data prediksi yang pending jika data csv baru diupload
                evaluate_pending_predictions()
                
                # Memproses data secara offline (langsung mengeksekusi pipeline)
                df = loader.load_data(ticker)
                
                if df is not None:
                    latest_data_date = df["timestamp"].max().date()
                    current_date_str = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
                    active_duplicate_policy = daily_duplicate_policy if "daily_duplicate_policy" in locals() else "skip"
                    active_run_type = daily_prediction_run_type if "daily_prediction_run_type" in locals() else "FINAL"
                    if (
                        active_duplicate_policy == "skip"
                        and active_run_type == "FINAL"
                        and has_core_prediction_for_data_date(ticker, current_date_str, run_type="FINAL")
                    ):
                        st.info(
                            f"Analisis dilewati. Prediksi inti FINAL H+1 dan H+3 untuk {ticker} "
                            f"pada tanggal data {current_date_str} sudah pernah dibuat."
                        )
                        st.stop()
                    if latest_data_date == datetime.now().date():
                        st.warning(
                            "Data terakhir adalah data hari ini dan kemungkinan masih berjalan. "
                            "High/low intraday yang belum final akan disesuaikan sementara agar analisis tetap bisa berjalan."
                        )

                    idx_df = loader.load_data('^JKSE')
                    if idx_df is None:
                        idx_df = df
                    features_df = engineer.generate_features(df, idx_df=idx_df)
                    if features_df.empty:
                        raise ValueError("Gagal membuat fitur atau jumlah data belum cukup.")

                    guardrail = assert_no_training_leakage(
                        raw_df=df,
                        features_df=features_df,
                        ticker=ticker,
                        prediction_date=df["timestamp"].iloc[-1],
                    )
                    if not guardrail.passed:
                        st.error("Guardrail data/model gagal. Analisis dihentikan untuk mencegah bias atau data leakage.")
                        for error in guardrail.errors:
                            st.write(f"- {error}")
                        raise ValueError("Guardrail data/model gagal.")
                    with st.expander("Audit Guardrail Data & Model", expanded=False):
                        st.success("Data dan fitur lolos pemeriksaan dasar anti-leakage.")
                        if guardrail.warnings:
                            for warning in guardrail.warnings:
                                st.warning(warning)
                        else:
                            st.caption("Tidak ada warning kualitas data yang terdeteksi.")
                    
                    # 1. Melatih Model Isolation Forest
                    if_model = IsolationForestModel(contamination=0.05)
                    if_model.train(features_df)
                    results = if_model.predict(features_df)
                    
                    scores_list = results['anomaly_score']
                    latest_score = float(scores_list[-1] if isinstance(scores_list, list) else scores_list)

                    # 2. Kalibrasi Conformal Predictor
                    cp = ConformalPredictor(alpha=0.05)
                    if isinstance(scores_list, list) and len(scores_list) > 50:
                        cp.calibrate(scores_list[:-1])
                    else:
                        cp.calibrate([50.0] * 100)

                    # 3. Klasifikasi Regim
                    rc = RegimeClassifier()
                    regime_features = rc.calculate_regime_features(idx_df, df)
                    regime_result = rc.classify(regime_features)
                    regime_str = regime_result.get("regime", "VOLATILE")
                    
                    # 4. Soft Ensemble
                    copod = None
                    if regime_str == "VOLATILE":
                        try:
                            copod = COPODModel(contamination=0.05)
                            copod.train(features_df)
                        except Exception as e:
                            logger.warning(f"Gagal melatih COPOD: {e}")
                            
                    ensemble_result = soft_ensemble_predict(
                        features_df=features_df,
                        regime=regime_str,
                        isolation_forest_model=if_model,
                        conformal_predictor=cp,
                        copod_model=copod
                    )

                    # 5. Proyeksi Harga Multi-Horizon & Klasifikasi Arah
                    projection_horizons = [3, 5, 10]
                    projectors = {}
                    projections = {}
                    lightgbm_projections = {}
                    for horizon in projection_horizons:
                        horizon_projector = PriceProjector(projection_horizon=horizon)
                        horizon_projector.train(features_df)
                        projectors[horizon] = horizon_projector
                        projections[horizon] = horizon_projector.predict(features_df)
                        try:
                            lgbm_projector = PriceProjector(projection_horizon=horizon, model_type="lightgbm")
                            lgbm_projector.train(features_df)
                            lightgbm_projections[horizon] = lgbm_projector.predict(features_df)
                        except Exception as e:
                            logger.warning(f"Gagal melatih LightGBMRegressor H+{horizon}: {e}")

                    project_horizon = 3
                    projector = projectors[project_horizon]
                    projection = projections[project_horizon]
                    next_day_projector = PriceProjector(projection_horizon=1)
                    next_day_projector.train(features_df)
                    next_day_projection = next_day_projector.predict(features_df)

                    classifier_predictions = {}
                    for model_type in ["lightgbm", "xgboost", "random_forest", "logistic"]:
                        try:
                            clf = DirectionClassifier(horizon_days=1, model_type=model_type)
                            clf.train(features_df)
                            classifier_predictions[model_type.upper()] = clf.predict(features_df)
                        except Exception as e:
                            logger.warning(f"Gagal melatih classifier {model_type}: {e}")

                    reliability_weights = get_reliability_weights(
                        ticker,
                        classifier_predictions.keys(),
                        prediction_purpose="NEXT_DAY_DIRECTION",
                    )
                    direction_ensemble = weighted_direction_probability(classifier_predictions, reliability_weights)

                    lstm_projector = LSTMPriceProjector(projection_horizon=project_horizon, lookback=20)
                    lstm_projector.train(features_df, epochs=3)
                    lstm_projection = lstm_projector.predict(features_df)
                    next_day_lstm_projector = LSTMPriceProjector(projection_horizon=1, lookback=20)
                    next_day_lstm_projector.train(features_df, epochs=3)
                    next_day_lstm_projection = next_day_lstm_projector.predict(features_df)

                    # 6. Proyeksi Volatilitas (GARCH)
                    garch_model = GARCHModel(p=1, q=1)
                    garch_model.train(df)
                    garch_projection = garch_model.predict(horizon=project_horizon)

                    data = {
                        "current_price": float(df['close'].iloc[-1]),
                        "latest_data_date": df['timestamp'].iloc[-1].date(),
                        "anomaly_score": ensemble_result.get("anomaly_score", 0.0),
                        "p_value": ensemble_result.get("p_value", 0.05),
                        "regime": regime_str,
                        "projection_3d": projection,
                        "projection_5d": projections[5],
                        "projection_10d": projections[10],
                        "lightgbm_projections": lightgbm_projections,
                        "lstm_projection_3d": lstm_projection,
                        "direction_ensemble": direction_ensemble,
                        "volatility_pct": garch_projection.get("projected_volatility_pct", 0.0),
                        "var_95_pct": garch_projection.get("value_at_risk_95_pct", 0.0)
                    }
                    
                    # --- CATAT PREDIKSI UNTUK TRACKING ---
                    current_date_str = df['timestamp'].iloc[-1].strftime("%Y-%m-%d")
                    target_date_str = (df['timestamp'].iloc[-1] + timedelta(days=project_horizon)).strftime("%Y-%m-%d")
                    next_day_target_date_str = (df['timestamp'].iloc[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
                    prediction_log_kwargs = {
                        "duplicate_policy": daily_duplicate_policy if "daily_duplicate_policy" in locals() else "skip",
                        "prediction_run_type": daily_prediction_run_type if "daily_prediction_run_type" in locals() else "FINAL",
                    }
                    for horizon, horizon_projection in projections.items():
                        horizon_target_date_str = (df['timestamp'].iloc[-1] + timedelta(days=horizon)).strftime("%Y-%m-%d")
                        purpose = "THREE_DAY_FORECAST" if horizon == 3 else f"H{horizon}_TREND_FORECAST"
                        log_prediction(ticker, "XGBoost", current_date_str, horizon_target_date_str, horizon_projection['projected_price'], data['current_price'], horizon_days=horizon, prediction_purpose=purpose, **prediction_log_kwargs)
                        if horizon in lightgbm_projections:
                            log_prediction(ticker, "LightGBM", current_date_str, horizon_target_date_str, lightgbm_projections[horizon]['projected_price'], data['current_price'], horizon_days=horizon, prediction_purpose=purpose, **prediction_log_kwargs)
                    log_prediction(ticker, "XGBoost", current_date_str, next_day_target_date_str, next_day_projection['projected_price'], data['current_price'], horizon_days=1, prediction_purpose="NEXT_DAY_DIRECTION", **prediction_log_kwargs)
                    if is_valid_projection(lstm_projection):
                        log_prediction(ticker, "LSTM", current_date_str, target_date_str, lstm_projection['projected_price'], data['current_price'], horizon_days=project_horizon, prediction_purpose="THREE_DAY_FORECAST", **prediction_log_kwargs)
                    if is_valid_projection(next_day_lstm_projection):
                        log_prediction(ticker, "LSTM", current_date_str, next_day_target_date_str, next_day_lstm_projection['projected_price'], data['current_price'], horizon_days=1, prediction_purpose="NEXT_DAY_DIRECTION", **prediction_log_kwargs)
                    for model_name, clf_prediction in classifier_predictions.items():
                        direction_price = data['current_price'] * (1.01 if clf_prediction["direction"] == "NAIK" else 0.99)
                        log_prediction(
                            ticker,
                            f"Direction-{model_name}",
                            current_date_str,
                            next_day_target_date_str,
                            direction_price,
                            data['current_price'],
                            horizon_days=1,
                            prediction_purpose="NEXT_DAY_DIRECTION",
                            predicted_direction=clf_prediction["direction"],
                            prob_up=clf_prediction["prob_up"],
                            prob_down=clf_prediction["prob_down"],
                            confidence_pct=clf_prediction["confidence_pct"],
                            **prediction_log_kwargs,
                        )
                    ensemble_price = data['current_price'] * (1.01 if direction_ensemble["direction"] == "NAIK" else 0.99)
                    log_prediction(
                        ticker,
                        "Direction-Ensemble",
                        current_date_str,
                        next_day_target_date_str,
                        ensemble_price,
                        data['current_price'],
                        horizon_days=1,
                        prediction_purpose="NEXT_DAY_DIRECTION",
                        predicted_direction=direction_ensemble["direction"],
                        prob_up=direction_ensemble["prob_up"],
                        prob_down=direction_ensemble["prob_down"],
                        confidence_pct=direction_ensemble["confidence_pct"],
                        **prediction_log_kwargs,
                    )

                    st.markdown(f"#### Hasil Analisis AI: {ticker}")
                    if data["latest_data_date"] < datetime.now().date():
                        st.warning(
                            f"Data harga terakhir untuk {ticker} berasal dari {data['latest_data_date']}. "
                            "Jika berbeda dengan harga pasar hari ini, jalankan update data terlebih dahulu."
                        )
                    
                    # --- METRIK UTAMA (COMPACT) ---
                    anomaly_score = data['anomaly_score']
                    metrics_df = pd.DataFrame([
                        {"Indikator": "Harga Terakhir", "Nilai": f"Rp {data['current_price']:,.0f}"},
                        {"Indikator": "Tanggal Data", "Nilai": data["latest_data_date"].strftime("%Y-%m-%d")},
                        {"Indikator": "Skor Anomali", "Nilai": f"{anomaly_score:.2f}/100"},
                        {"Indikator": "P-Value", "Nilai": f"{data['p_value']:.3f}"},
                        {"Indikator": "Regim Pasar", "Nilai": data["regime"]},
                        {"Indikator": "Arah H+1 Ensemble", "Nilai": f"{direction_ensemble['direction']} ({direction_ensemble['confidence_pct']:.1f}%)"},
                        {"Indikator": "Volatilitas H+3", "Nilai": f"{data['volatility_pct']:.2f}%"},
                        {"Indikator": "VaR 95% 1-Hari", "Nilai": f"{data['var_95_pct']:.2f}%"},
                    ])
                    st.dataframe(metrics_df, width="stretch", hide_index=True, height=280)
                    
                    # --- PROYEKSI HARGA ---
                    if "projection_3d" in data:
                        proj = data["projection_3d"]
                        lstm_proj = data["lstm_projection_3d"]
                        projection_rows = [
                            {
                                "Model": "XGBoost",
                                "Target H+3": f"Rp {proj['projected_price']:,.0f}",
                                "Potensi H+3": f"{proj['projected_return_pct']:+.2f}%",
                                "Potensi H+5": f"{data['projection_5d']['projected_return_pct']:+.2f}%",
                                "Potensi H+10": f"{data['projection_10d']['projected_return_pct']:+.2f}%",
                            },
                        ]
                        if 3 in data["lightgbm_projections"]:
                            lgbm_proj = data["lightgbm_projections"][3]
                            projection_rows.append({
                                "Model": "LightGBM",
                                "Target H+3": f"Rp {lgbm_proj['projected_price']:,.0f}",
                                "Potensi H+3": f"{lgbm_proj['projected_return_pct']:+.2f}%",
                                "Potensi H+5": f"{data['lightgbm_projections'].get(5, {}).get('projected_return_pct', 0.0):+.2f}%",
                                "Potensi H+10": f"{data['lightgbm_projections'].get(10, {}).get('projected_return_pct', 0.0):+.2f}%",
                            })
                        if is_valid_projection(lstm_proj):
                            projection_rows.append({
                                "Model": "LSTM",
                                "Target H+3": f"Rp {lstm_proj['projected_price']:,.0f}",
                                "Potensi H+3": f"{lstm_proj['projected_return_pct']:+.2f}%",
                                "Potensi H+5": "-",
                                "Potensi H+10": "-",
                            })
                        else:
                            st.caption("Prediksi LSTM belum tersedia. Pastikan PyTorch terpasang jika ingin mengaktifkan model LSTM.")
                        st.dataframe(pd.DataFrame(projection_rows), width="stretch", hide_index=True)

                    st.markdown("### Direction Classifier H+1")
                    if classifier_predictions:
                        classifier_rows = []
                        for model_name, pred in classifier_predictions.items():
                            classifier_rows.append({
                                "Model": model_name,
                                "Peluang Naik": f"{pred['prob_up'] * 100:.2f}%",
                                "Peluang Turun": f"{pred['prob_down'] * 100:.2f}%",
                                "Arah": pred["direction"],
                                "Confidence": f"{pred['confidence_pct']:.2f}%",
                                "Bobot Reliability": f"{reliability_weights.get(model_name, 0.0) * 100:.2f}%",
                            })
                        classifier_rows.append({
                            "Model": "ENSEMBLE",
                            "Peluang Naik": f"{direction_ensemble['prob_up'] * 100:.2f}%",
                            "Peluang Turun": f"{direction_ensemble['prob_down'] * 100:.2f}%",
                            "Arah": direction_ensemble["direction"],
                            "Confidence": f"{direction_ensemble['confidence_pct']:.2f}%",
                            "Bobot Reliability": "100.00%",
                        })
                        st.dataframe(pd.DataFrame(classifier_rows), width="stretch", hide_index=True)
                    else:
                        st.warning("Classifier arah belum bisa dilatih untuk data ini.")

                    with st.expander("Baseline Strategy & Walk-Forward Validation", expanded=False):
                        baseline_h1 = evaluate_baseline_strategies(features_df, horizon_days=1)
                        baseline_h3 = evaluate_baseline_strategies(features_df, horizon_days=3)
                        btab1, btab2, btab3 = st.tabs(["Baseline H+1", "Baseline H+3", "Walk-Forward"])
                        with btab1:
                            st.dataframe(baseline_h1.style.format({
                                "direction_accuracy_pct": "{:.2f}%",
                                "avg_return_pct": "{:+.2f}%",
                                "win_rate_pct": "{:.2f}%",
                            }), width="stretch")
                        with btab2:
                            st.dataframe(baseline_h3.style.format({
                                "direction_accuracy_pct": "{:.2f}%",
                                "avg_return_pct": "{:+.2f}%",
                                "win_rate_pct": "{:.2f}%",
                            }), width="stretch")
                        with btab3:
                            try:
                                direction_factory = (
                                    lambda: LGBMClassifier(n_estimators=120, learning_rate=0.05, random_state=42, verbosity=-1)
                                    if LGBMClassifier is not None
                                    else xgb.XGBClassifier(n_estimators=120, learning_rate=0.05, random_state=42, eval_metric="logloss")
                                )
                                return_factory = (
                                    lambda: LGBMRegressor(n_estimators=120, learning_rate=0.05, random_state=42, verbosity=-1)
                                    if LGBMRegressor is not None
                                    else xgb.XGBRegressor(n_estimators=120, learning_rate=0.05, random_state=42, objective="reg:squarederror")
                                )
                                wf_h1 = walk_forward_direction_validation(features_df, direction_factory, horizon_days=1)
                                wf_h3 = walk_forward_return_validation(features_df, return_factory, horizon_days=3)
                                wf_h5 = walk_forward_return_validation(features_df, return_factory, horizon_days=5)
                                wf_h10 = walk_forward_return_validation(features_df, return_factory, horizon_days=10)
                                wf_df = pd.DataFrame([
                                    {"Validasi": "Direction H+1", **wf_h1},
                                    {"Validasi": "Return H+3", **wf_h3},
                                    {"Validasi": "Return H+5", **wf_h5},
                                    {"Validasi": "Return H+10", **wf_h10},
                                ])
                                st.dataframe(wf_df, width="stretch", hide_index=True)
                            except Exception as e:
                                st.warning(f"Walk-forward belum dapat dihitung: {e}")

                    with st.expander("AI Explainability: SHAP / Feature Importance XGBoost", expanded=False):
                        st.write(
                            "Bagian ini membantu membaca alasan model XGBoost membuat proyeksi. "
                            "Kontribusi positif mendorong prediksi return H+3 lebih tinggi, sedangkan kontribusi negatif menekan prediksi."
                        )
                        explanation_df, explanation_status = build_xgboost_explanation(projector, features_df)
                        if explanation_df.empty:
                            st.warning(explanation_status)
                        else:
                            st.caption(explanation_status)
                            if "Kontribusi SHAP" in explanation_df.columns:
                                chart_df = explanation_df.sort_values("Kontribusi SHAP")
                                colors = [
                                    "#16a34a" if value > 0 else "#dc2626"
                                    for value in chart_df["Kontribusi SHAP"]
                                ]
                                fig_shap = go.Figure(go.Bar(
                                    x=chart_df["Kontribusi SHAP"],
                                    y=chart_df["Nama Mudah"],
                                    orientation="h",
                                    marker_color=colors,
                                ))
                                fig_shap.update_layout(
                                    height=420,
                                    margin=dict(l=0, r=0, t=20, b=0),
                                    xaxis_title="Kontribusi terhadap prediksi return",
                                    yaxis_title="Fitur",
                                )
                                st.plotly_chart(fig_shap, width="stretch")
                                st.dataframe(
                                    explanation_df.style.format({
                                        "Nilai Saat Ini": "{:,.4f}",
                                        "Kontribusi SHAP": "{:+.6f}",
                                    }),
                                    width="stretch",
                                )
                            else:
                                chart_df = explanation_df.sort_values("Importance")
                                fig_importance = go.Figure(go.Bar(
                                    x=chart_df["Importance"],
                                    y=chart_df["Nama Mudah"],
                                    orientation="h",
                                    marker_color="#2563eb",
                                ))
                                fig_importance.update_layout(
                                    height=420,
                                    margin=dict(l=0, r=0, t=20, b=0),
                                    xaxis_title="Feature importance",
                                    yaxis_title="Fitur",
                                )
                                st.plotly_chart(fig_importance, width="stretch")
                                st.dataframe(
                                    explanation_df.style.format({
                                        "Nilai Saat Ini": "{:,.4f}",
                                        "Importance": "{:.6f}",
                                    }),
                                    width="stretch",
                                )
                    
                    # --- REKOMENDASI TRADING ---
                    st.markdown("---")
                    st.subheader("Rekomendasi Setup Trading")
                    
                    # Generate Setup Trading secara lokal
                    rsi = float(features_df['feat_rsi_14'].iloc[-1])
                    atr = float(features_df['feat_atr_14'].iloc[-1])
                    
                    setup = generate_signal(
                        ticker, 
                        data["current_price"], 
                        data["anomaly_score"], 
                        data["p_value"], 
                        data["regime"], 
                        rsi, 
                        atr
                    )

                    decision = build_decision_support(
                        ticker=ticker,
                        current_price=data["current_price"],
                        projected_return_pct=projection.get("projected_return_pct", 0.0),
                        anomaly_score=data["anomaly_score"],
                        p_value=data["p_value"],
                        regime=data["regime"],
                        rsi=rsi,
                        volatility_pct=data["volatility_pct"],
                        var_95_pct=data["var_95_pct"],
                        signal_result=setup,
                        capital=float(portfolio_capital),
                        risk_pct=float(risk_per_trade_pct),
                    )

                    st.markdown("### Decision Support")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("AI Confidence Score", f"{decision['ai_confidence_score']:.1f}/100", decision["confidence_label"])
                    d2.metric("Trade Readiness", decision["readiness"])
                    sizing = decision.get("position_sizing", {})
                    d3.metric("Rekomendasi Lot", f"{sizing.get('lots', 0):,.0f}")

                    if decision.get("reasons"):
                        st.caption("Catatan: " + " ".join(decision["reasons"]))

                    trade_gate = build_trade_gate(
                        projected_return_pct=projection.get("projected_return_pct", 0.0),
                        confidence_pct=direction_ensemble.get("confidence_pct", decision["ai_confidence_score"]),
                        volatility_pct=data["volatility_pct"],
                        min_confidence_pct=60.0,
                        min_projected_return_pct=1.0,
                        max_volatility_pct=8.0,
                    )
                    if trade_gate["passed"]:
                        st.success(f"Trade Gate: {trade_gate['status']}")
                    else:
                        st.warning(f"Trade Gate: {trade_gate['status']}")
                        st.caption(" ".join(trade_gate["reasons"]))
                    
                    if setup.get('action') == 'TRADE':
                        st.success(f"**SIGNAL: {setup['setup']['direction']}**")
                        s = setup['setup']
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Entry Price", f"Rp {s['entry']:,.0f}")
                        c2.metric("Stop Loss", f"Rp {s['stop_loss']:,.0f}")
                        c3.metric("Take Profit 1", f"Rp {s['take_profit_1']:,.0f}")
                        c4.metric("Take Profit 2", f"Rp {s['take_profit_2']:,.0f}")
                        if sizing:
                            st.caption(
                                f"Estimasi nilai posisi: Rp {sizing.get('position_value', 0.0):,.0f} "
                                f"({sizing.get('capital_used_pct', 0.0):.2f}% modal), "
                                f"risiko maksimum: Rp {sizing.get('risk_amount', 0.0):,.0f}."
                            )
                    else:
                        st.warning(f"**SIGNAL: SKIP (Tidak Ada Sinyal Entry)**")
                        st.write(f"**Alasan AI:** {setup.get('reason', 'Kondisi anomali tidak mencukupi standar setup yang aman.')}")

                    st.markdown("---")
                    st.subheader("Backtest Ringkas Decision Support")
                    try:
                        backtest_df = features_df.copy()
                        backtest_df["projected_return_pct"] = (
                            (backtest_df["close"].shift(-project_horizon) / backtest_df["close"]) - 1.0
                        ) * 100.0
                        backtest_df["anomaly_score"] = results["anomaly_score"]
                        backtest_df["p_value"] = 0.01
                        backtest_df["ai_confidence_score"] = backtest_df.apply(
                            lambda row: calculate_ai_confidence_score(
                                projected_return_pct=float(row["projected_return_pct"]) if pd.notna(row["projected_return_pct"]) else 0.0,
                                anomaly_score=float(row["anomaly_score"]),
                                p_value=float(row["p_value"]),
                                regime=regime_str,
                                rsi=float(row["feat_rsi_14"]),
                                volatility_pct=data["volatility_pct"],
                                var_95_pct=data["var_95_pct"],
                            )["ai_confidence_score"],
                            axis=1,
                        )
                        backtest_df = backtest_df.dropna(subset=["projected_return_pct"])
                        bt = BacktestEngine().simulate_signal_strategy(
                            backtest_df,
                            initial_capital=float(portfolio_capital),
                            risk_pct=float(risk_per_trade_pct),
                            holding_period=project_horizon,
                            min_confidence_score=55.0,
                        )
                        metrics = bt["metrics"]
                        b1, b2, b3, b4, b5 = st.columns(5)
                        b1.metric("Total Trade", f"{metrics.get('total_trades', 0):,.0f}")
                        b2.metric("Win Rate", f"{metrics.get('win_rate', 0.0) * 100:.1f}%")
                        b3.metric("Return", f"{metrics.get('return_pct', 0.0):+.2f}%")
                        b4.metric("Profit Factor", f"{metrics.get('profit_factor', 0.0):.2f}")
                        b5.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0) * 100:.2f}%")
                    except Exception as e:
                        st.warning(f"Backtest ringkas belum dapat dihitung: {e}")
                    
                    # --- VISUALISASI GRAFIK CANDLESTICK ---
                    st.markdown("---")
                    st.subheader(f"Pergerakan Harga Terakhir ({ticker})")
                    
                    # Kita menggunakan dataframe 'df' yang sudah diload di awal agar lebih cepat (tidak perlu baca CSV 2 kali)
                    df_plot = df.tail(100)
                    fig = go.Figure(data=[go.Candlestick(x=df_plot['timestamp'],
                                    open=df_plot['open'], high=df_plot['high'],
                                    low=df_plot['low'], close=df_plot['close'])])
                    fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, width="stretch")
                        
                else:
                    st.error(f"Data {ticker} tidak ditemukan. Pastikan file CSV tersedia di data/raw/.")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")

