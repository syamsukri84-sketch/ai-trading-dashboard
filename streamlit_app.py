import streamlit as st
import pandas as pd
import plotly.express as px
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

import run_analysis as analysis_runner

analysis_runner = importlib.reload(analysis_runner)
run_backfill_analysis = analysis_runner.run_backfill_analysis
run_full_analysis = analysis_runner.run_full_analysis
from src.data_pipeline.auto_updater import (
    _normalize_existing_data,
    get_local_data_status,
    run_auto_updater,
    update_from_manual_dataframe,
)
from src.trading.decision_support import calculate_position_sizing
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
from src.utils.user_feedback import load_user_feedback, log_user_feedback
from src.utils.risk_metrics_log import get_latest_var95_lookup
from src.trading.market_regime import compute_market_breadth, load_regime_history, log_regime_snapshot, summarize_regime_streaks
from src.trading.personalization import apply_personalization, load_user_profile, mute_ticker, unmute_ticker
from src.models.global_models import predict_with_global_models
from src.nlp.news_fetcher import fetch_google_news_sentiment_items
from src.nlp.sentiment_analyzer import analyze_dataframe, analyze_text, append_issue, append_issues, build_local_sentiment_dataset, build_trading_sentiment_summary, get_local_sentiment_dataset_path, get_sentiment_engine_status, interpret_signal, load_issues, summarize_by_ticker
from src.nlp.gemini_keyword_search import check_gemini_status, suggest_sentiment_keywords

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

with st.expander("👋 Pertama Kali di Sini? Mulai dari Sini", expanded=False):
    st.markdown(
        "1. **Pilih mode tampilan** di sidebar (Pemula/Trader/Audit) -- Pemula menyembunyikan kolom "
        "teknis dan cocok kalau baru pertama kali pakai dashboard ini.\n"
        "2. **Buka tab 'Ringkasan Harian'** -- tabel 'Rencana Trading Besok' menjawab pertanyaan utama: "
        "\"saham apa yang layak dipantau/dibeli hari ini, dan kenapa?\"\n"
        "3. **Lihat kolom Verifikasi** di tabel itu -- ✅ berarti sinyalnya sudah lolos dua jenis "
        "pengujian sekaligus (bukan cuma tebakan mentah model).\n"
        "4. Kalau ada istilah yang tidak dipahami, buka **'📖 Kamus Istilah'** di sidebar -- semua "
        "istilah teknis di dashboard ini dijelaskan di situ.\n"
        "5. Pengaturan lain di sidebar (batas reliability, confidence, dst.) **tidak wajib diubah** -- "
        "nilai defaultnya sudah masuk akal untuk mulai."
    )


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

price_update_online = get_bool_runtime_setting("AI_TRADING_PRICE_UPDATE_ONLINE", default=True)
external_services_enabled = get_bool_runtime_setting("AI_TRADING_EXTERNAL_SERVICES", default=False)
if price_update_online:
    operation_mode = "Hybrid offline"
    st.sidebar.info("Mode hybrid aktif. Internet hanya dipakai saat tombol update harga ditekan; AI Trading tetap memakai file lokal dan model offline.")
else:
    operation_mode = "Offline"
    st.sidebar.info("Mode offline penuh aktif. Semua fitur internet dinonaktifkan.")
offline_mode = not price_update_online
sentiment_online_enabled = external_services_enabled and price_update_online
launch_snapshot = build_launch_sync_snapshot(tickers)
render_launch_sync_snapshot(launch_snapshot, operation_mode)

with st.sidebar.expander("Status Sinkronisasi", expanded=True):
    st.metric("Data Lokal", f"{launch_snapshot['raw_count']}/{launch_snapshot['total_tickers']}")
    st.metric("Pending Aktif", f"{launch_snapshot['pending_count']:,}")
    st.metric("Prediksi Aktif", f"{launch_snapshot['active_count']:,}")
    if offline_mode:
        st.success("Mode offline aktif. Analisis lokal, ranking, dan evaluasi akurasi siap digunakan.")
        st.caption("Tombol update harga online dinonaktifkan oleh konfigurasi.")
    else:
        st.info("Mode hybrid aktif. Update harga boleh memakai internet; prediksi, ranking, dan evaluasi tetap offline.")
    if external_services_enabled:
        mongo_status = check_mongo_status()
        if mongo_status["ok"]:
            st.success(f"MongoDB Atlas aktif: {mongo_status['database']}")
        elif mongo_status["enabled"]:
            st.warning(f"MongoDB Atlas belum tersambung: {mongo_status['message']}")
        else:
            st.caption("MongoDB Atlas belum aktif. Isi MONGODB_URI jika ingin sinkronisasi eksternal.")
    else:
        st.caption("MongoDB dan layanan eksternal dinonaktifkan. Semua fitur AI Trading membaca/menulis file lokal.")

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

st.sidebar.header("Pengaturan Sensitivitas & Tampilan")
display_mode = st.sidebar.radio(
    "Mode tampilan",
    ["Pemula", "Trader", "Audit"],
    horizontal=True,
    help=(
        "Berlaku di semua tab. Pemula: sembunyikan kolom teknis, fokus ke keputusan. "
        "Trader: tambah angka potensi/risk. Audit: tampilkan semua detail teknis + status job."
    ),
)
with st.sidebar.expander("⚙️ Pengaturan Lanjutan (opsional -- nilai default sudah masuk akal)"):
    global_min_reliability = st.slider(
        "Batas reliability model minimum (%)", 0.0, 100.0, 55.0, 5.0,
        help="Dipakai di Ringkasan Harian untuk menyaring sinyal BUY/WATCH -- model dengan reliability_score di bawah ini dianggap belum layak jadi acuan.",
    )
    global_min_confidence = st.slider(
        "Batas confidence entry minimum (%)", 0.0, 100.0, 60.0, 5.0,
        help="Dipakai di Ringkasan Harian -- prediksi arah H+1 dengan confidence di bawah ini dianggap netral, bukan sinyal kuat.",
    )
    global_min_evaluations_trading = st.number_input(
        "Minimal track record untuk sinyal BUY/WATCH", min_value=3, max_value=200, value=20, step=1,
        help="Dipakai di Ringkasan Harian -- makin tinggi, makin ketat sebelum sinyal dianggap layak dieksekusi.",
    )
    global_min_evaluations_audit = st.number_input(
        "Minimal track record untuk tabel audit", min_value=1, max_value=50, value=3, step=1,
        help=(
            "Dipakai di tab Akurasi Model (bukan Ringkasan Harian) -- sengaja lebih rendah supaya model yang "
            "datanya belum banyak masih terlihat di tabel audit/telaah, bukan langsung hilang."
        ),
    )

with st.sidebar.expander("📖 Kamus Istilah"):
    st.markdown(
        "- **Walk-forward**: cara menguji model dengan train/test bergulir mengikuti waktu (bukan acak), "
        "supaya tidak ada kebocoran informasi masa depan ke masa lalu.\n"
        "- **Baseline naif**: pembanding sederhana (tebak arah mayoritas / tebak return nol). Model baru "
        "dianggap ada **edge** kalau mengalahkan ini, bukan sekadar akurasi tinggi.\n"
        "- **Edge vs baseline**: selisih akurasi model terhadap baseline naif pada periode uji yang sama. "
        "Positif = model benar-benar lebih baik dari tebakan naif.\n"
        "- **SHAP**: metode yang merinci fitur mana yang mendorong satu prediksi spesifik ke arah NAIK/TURUN, "
        "dan seberapa besar pengaruhnya.\n"
        "- **Reliability score**: skor 0-100 dari akurasi + error margin + jumlah sampel evaluasi historis.\n"
        "- **Trading score**: skor 0-100 yang juga memperhitungkan hasil simulasi return & profit factor, "
        "bukan cuma akurasi arah.\n"
        "- **Profit factor**: total untung dibagi total rugi dari sinyal historis. Di atas 1 berarti untung "
        "lebih besar dari rugi.\n"
        "- **Calibration gap**: selisih antara confidence yang diklaim model dan akurasi sungguhan -- "
        "makin dekat ke 0 makin bisa dipercaya angka confidence-nya.\n"
        "- **Trust audit**: status LAYAK DIPERCAYA/PERLU DATA LAGI/JANGAN DIIKUTI berdasarkan track record "
        "prediksi live yang sudah dievaluasi terhadap harga aktual.\n"
        "- **Genuine edge (walk-forward)**: sama seperti edge vs baseline, tapi dari backtest walk-forward "
        "penuh (bukan track record live) -- lihat tab Walk-Forward Genuine Edge.\n"
        "- **F1 per kelas**: satu angka yang menggabungkan presisi dan recall untuk SATU kelas (mis. NEGATIVE) "
        "saja. Berguna karena akurasi keseluruhan bisa menyembunyikan kelas yang datanya sedikit/model-nya "
        "lemah di situ -- F1 rendah di satu kelas berarti model itu masih sering salah khusus untuk kelas itu.\n"
        "- **Skor Personal**: angka -1 s.d. +1 dari riwayat umpan balik Anda sendiri (Ikuti/Berguna menambah, "
        "Lewati/Tidak Berguna mengurangi) -- HANYA memengaruhi urutan tampilan, TIDAK PERNAH mengubah "
        "Sinyal/Confidence/prediksi model.\n"
        "- **Dimute**: ticker yang Anda tandai untuk disembunyikan dari tabel Rencana Trading lewat tombol "
        "'Mute Ticker Ini' -- tetap dianalisis di belakang layar, cuma tidak ditampilkan di tabel utama.\n"
        "- **Regime pasar & streak**: kondisi pasar hari ini (REBOUND/BEARISH/MIXED) berdasarkan persentase "
        "saham yang naik. 'Streak' berarti sudah berapa hari berturut-turut regime itu bertahan -- dicatat "
        "otomatis tiap hari supaya dashboard punya 'memori' konteks pasar, bukan cuma snapshot hari ini."
    )

if st.sidebar.button("🔄 Refresh Data Sekarang", help="Bersihkan cache dan muat ulang semua data terbaru dari file lokal."):
    st.cache_data.clear()
    st.rerun()

# --- TABS UTAMA: alur harian dulu, kontrol lanjutan setelahnya ---
tab_beranda, tab_daily, tab_update, tab_ranking, tab_accuracy, tab_sentiment = st.tabs([
    "🏠 Beranda",
    "Ringkasan Harian",
    "Workflow Harian",
    "Ranking Mentah (Riset)",
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


def _decision_label(action):
    label_map = {
        "BUY": "🟢 BUY",
        "WATCH": "🟡 WATCH",
        "AVOID": "🔴 AVOID",
        "DATA BELUM SIAP": "⚪ DATA BELUM SIAP",
        "MODEL BELUM TERPERCAYA": "🟠 MODEL BELUM TERPERCAYA",
        "TIDAK ADA EDGE (WALK-FORWARD)": "⚫ TIDAK ADA EDGE (WALK-FORWARD)",
    }
    return label_map.get(action, action)


def compute_unified_trust_badge(status_trust, has_genuine_edge):
    """Menggabungkan dua sinyal kepercayaan yang sumbernya beda supaya pengguna
    tidak perlu membandingkan sendiri antar tab:
    - `status_trust`: track record LIVE (prediksi asli yang sudah dievaluasi
      terhadap harga aktual) -- dari tab Model Trust Audit.
    - `has_genuine_edge`: hasil BACKTEST walk-forward penuh (train/test
      bergulir) -- dari tab Walk-Forward Genuine Edge / screening.

    Keduanya bisa tidak sepakat (mis. live terlihat bagus tapi belum pernah
    lolos backtest, atau sebaliknya) -- itu justru informasi penting, bukan
    kontradiksi yang harus disembunyikan.
    """
    live_ok = status_trust == "LAYAK DIPERCAYA"
    if live_ok and has_genuine_edge is True:
        return "✅ Terverifikasi Ganda", "Live OK + lolos backtest walk-forward."
    if live_ok and has_genuine_edge is False:
        return "⚠️ Live OK, Backtest Belum Lolos", "Track record live bagus, tapi backtest walk-forward belum menunjukkan edge nyata."
    if live_ok and has_genuine_edge is None:
        return "⚠️ Live OK, Backtest Belum Diuji", "Track record live bagus, tapi belum discreening lewat walk-forward."
    if not live_ok and has_genuine_edge is True:
        return "⚠️ Backtest OK, Live Kurang", "Lolos backtest walk-forward, tapi track record live belum cukup/trusted."
    return "❌ Belum Lolos Verifikasi", "Belum lolos live trust audit maupun backtest walk-forward."


def render_interactive_accuracy_trend(df, x_col, y_col, color_col, title=""):
    """Chart interaktif (hover per titik, bisa zoom/pan) pengganti st.line_chart
    statis -- supaya pengguna bisa menelusuri sendiri tanggal/model mana yang
    melemah tanpa harus buka tabel mentah."""
    if df.empty:
        return
    fig = px.line(df, x=x_col, y=y_col, color=color_col, markers=True, title=title)
    fig.update_layout(hovermode="x unified", legend_title_text=color_col, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_interactive_contribution_bar(feat_df, title=""):
    """Chart kontribusi SHAP dengan warna hijau/merah sesuai arah dorongan
    prediksi, plus hover nilai fitur asli -- pengganti st.bar_chart statis
    yang cuma menunjukkan tinggi batang tanpa konteks."""
    if feat_df.empty:
        return
    chart_df = feat_df.sort_values("contribution")
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in chart_df["contribution"]]
    fig = go.Figure(go.Bar(
        x=chart_df["contribution"],
        y=chart_df["feature"],
        orientation="h",
        marker_color=colors,
        customdata=chart_df["value"],
        hovertemplate="%{y}: nilai=%{customdata:.4g}, kontribusi=%{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Kontribusi SHAP", yaxis_title="", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def build_daily_decision_board(
    selected_tickers,
    min_reliability=55.0,
    min_confidence=55.0,
    min_evaluations=20,
    portfolio_capital=100_000_000,
    risk_per_trade_pct=1.0,
    genuine_edge_lookup=None,
    edge_lookup_for_badge=None,
    var95_lookup=None,
    sentiment_bias_lookup=None,
    xai_reason_lookup=None,
):
    normalized_tickers = [normalize_ticker_code(t) for t in selected_tickers if normalize_ticker_code(t)]
    market_breadth = compute_market_breadth(normalized_tickers, raw_dir=project_path("data", "raw"))
    log_regime_snapshot(market_breadth)
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
            _cp = pd.to_numeric(h3_pred["current_price"].iloc[0], errors="coerce")
            current_price = float(_cp) if pd.notna(_cp) else 0.0
        elif not h1_pred.empty and "current_price" in h1_pred.columns:
            _cp = pd.to_numeric(h1_pred["current_price"].iloc[0], errors="coerce")
            current_price = float(_cp) if pd.notna(_cp) else 0.0

        projected_return = float(h3_pred["predicted_return_pct"].iloc[0]) if not h3_pred.empty and pd.notna(h3_pred["predicted_return_pct"].iloc[0]) else 0.0
        raw_confidence = float(h1_pred["confidence_pct"].iloc[0]) if not h1_pred.empty and pd.notna(h1_pred["confidence_pct"].iloc[0]) else 0.0
        confidence = raw_confidence
        direction = str(h1_pred["predicted_direction"].iloc[0]) if not h1_pred.empty and "predicted_direction" in h1_pred.columns else "-"
        reliability = float(reliability_lookup.get((ticker_code, "XGBoost"), 50.0))
        total_evaluations = int(sample_lookup.get((ticker_code, "XGBoost"), 0) or 0)
        direction_accuracy = float(accuracy_lookup.get((ticker_code, "XGBoost"), 0.0) or 0.0)
        trust_status = str(trust_lookup.get(ticker_code, "PERLU DATA LAGI"))
        trust_reason = str(trust_reason_lookup.get(ticker_code, "Track record H+1 belum cukup untuk menjadi acuan utama."))
        edge_for_badge = edge_lookup_for_badge.get(ticker_code) if edge_lookup_for_badge is not None else None
        unified_badge, unified_badge_reason = compute_unified_trust_badge(trust_status, edge_for_badge)
        var95_info = var95_lookup.get(ticker_code) if var95_lookup is not None else None
        sentiment_bias = sentiment_bias_lookup.get(ticker_code, "NEUTRAL") if sentiment_bias_lookup is not None else "NEUTRAL"
        xai_reason = xai_reason_lookup.get(ticker_code, "-") if xai_reason_lookup is not None else "-"

        # SL sekarang mengikuti VaR 95% ticker ini (metode "direkomendasikan"
        # dari src/models/var_analysis.py -- Cornish-Fisher atau rata-rata
        # historical+MC-bootstrap, tergantung skew/kurtosis data), bukan lagi
        # volatilitas GARCH mentah (fallback lama, tidak pernah kepakai --
        # lihat CATATAN_SESI_VAR_DAN_GATING_2026-07-17.md bagian 3.1/3.2).
        # VaR 95% sudah berupa kuantil kerugian (bukan std dev mentah), jadi
        # TIDAK dikalikan 1.5x lagi seperti formula GARCH lama -- cuma
        # disesuaikan bias sentimen (stop lebih ketat kalau BEARISH bertentangan
        # dengan sinyal). Batas 1.5%-8% tetap heuristik praktis (belum
        # dioptimasi empiris) supaya saham fluktuatif dapat ruang gerak wajar
        # dan saham tenang tidak kena stop longgar.
        if var95_info is not None and var95_info.get("var95_pct", 0) > 0:
            var95_pct = var95_info["var95_pct"]
            sentiment_multiplier = 0.85 if sentiment_bias == "BEARISH" else (1.1 if sentiment_bias == "BULLISH" else 1.0)
            sl_distance_pct = min(max(var95_pct * sentiment_multiplier, 1.5), 8.0)
            data_terbatas_flag = " [data terbatas]" if var95_info.get("data_terbatas") else ""
            sl_basis = (
                f"VaR 95% ({var95_info.get('var95_method', '-')}) {var95_pct:.2f}%/hari{data_terbatas_flag}, "
                f"sentimen {sentiment_bias.lower()}"
            )
        else:
            sl_distance_pct = 3.0
            sl_basis = "default 3% (data VaR belum tersedia -- jalankan analisis dulu)"

        target_h3 = current_price * (1 + projected_return / 100.0) if current_price > 0 else 0.0
        entry_low = current_price * 0.995 if current_price > 0 else 0.0
        entry_high = current_price * 1.005 if current_price > 0 else 0.0
        stop_loss = current_price * (1 - sl_distance_pct / 100.0) if current_price > 0 else 0.0
        risk_reward = ((target_h3 - current_price) / max(current_price - stop_loss, 1e-9)) if current_price > 0 else 0.0
        sizing = calculate_position_sizing(
            capital=float(portfolio_capital),
            entry_price=current_price,
            stop_loss=stop_loss,
            risk_pct=float(risk_per_trade_pct),
        )

        if action == "WATCH":
            if genuine_edge_lookup is not None and not genuine_edge_lookup.get(ticker_code, False):
                action = "TIDAK ADA EDGE (WALK-FORWARD)"
                reason_parts.append(
                    "Screening walk-forward penuh (backtest train/test bergulir) menunjukkan model TIDAK "
                    "mengalahkan baseline tebak-mayoritas untuk ticker ini di horizon manapun -- "
                    "lihat tab 'Walk-Forward Genuine Edge' untuk detail."
                )
            elif trust_status != "LAYAK DIPERCAYA":
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
                if unified_badge == "✅ Terverifikasi Ganda":
                    action = "BUY"
                    reason_parts.append("Prioritas H+3: potensi positif, H+1 mendukung, confidence tinggi, dan track record lolos batas.")
                else:
                    # PENTING: sebelum perbaikan ini, sinyal BUY (dengan entry/stop
                    # loss/lot konkret) bisa tampil untuk ticker apa pun yang lolos
                    # gate teknis di atas, TANPA syarat status "Verifikasi" ganda --
                    # genuine_edge_lookup (dipakai gate "TIDAK ADA EDGE" di atas)
                    # cuma aktif kalau user mencentang checkbox opt-in di tab ini,
                    # defaultnya OFF. Padahal unified_badge (dari
                    # compute_unified_trust_badge, dibangun dari edge_lookup_for_badge
                    # yang SELALU diisi) sudah tersedia independen dari checkbox itu.
                    # Ini melanggar Prinsip Desain #4 (STATUS_PROYEK_AI_TRADING.md):
                    # "Prescriptive analytics HANYA untuk ticker Terverifikasi Ganda"
                    # -- BUY adalah rekomendasi prescriptive paling actionable di
                    # dashboard ini (dilengkapi entry/stop loss/lot), jadi gate ini
                    # WAJIB berlaku selalu, bukan opsional lewat checkbox screening
                    # edge yang terpisah. Diperbaiki 2026-07-17.
                    action = "WATCH"
                    reason_parts.append(
                        f"Sinyal teknis mendukung BUY, tapi status verifikasi '{unified_badge}' belum "
                        "'✅ Terverifikasi Ganda' (butuh LOLOS live trust audit DAN backtest walk-forward "
                        "sekaligus). Sesuai Prinsip Desain #4 proyek ini, rekomendasi BUY dibatasi hanya "
                        "untuk ticker yang lolos verifikasi ganda -- entry/stop loss di atas jadi referensi "
                        "pemantauan, bukan sinyal siap eksekusi."
                    )
            elif projected_return <= -1.0 or direction == "TURUN":
                action = "AVOID"
                reason_parts.append("Arah/potensi return belum mendukung entry.")
            else:
                action = "WATCH"
                reason_parts.append("Sinyal belum cukup kuat untuk entry.")

        risk_label = "Rendah"
        if abs(projected_return) < 1.0 or confidence < min_confidence:
            risk_label = "Sedang"
        if reliability < min_reliability or action in {"AVOID", "DATA BELUM SIAP", "TIDAK ADA EDGE (WALK-FORWARD)"}:
            risk_label = "Tinggi"

        rows.append({
            "Saham": ticker_code,
            "Sinyal": _decision_label(action),
            "Harga Terakhir": current_price,
            "Entry Area": f"{entry_low:,.0f} - {entry_high:,.0f}" if current_price > 0 else "-",
            "Stop Loss": stop_loss,
            "Basis SL": sl_basis,
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
            "Verifikasi": unified_badge,
            "Risiko": risk_label,
            "Status Data": status_data,
            "Status Analisis": status_analisis,
            "Tanggal Data": last_date,
            "Alasan Utama": " ".join(reason_parts),
            "Catatan Trust": trust_reason,
            "Catatan Verifikasi": unified_badge_reason,
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
        "⚫ TIDAK ADA EDGE (WALK-FORWARD)": 5,
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


def get_latest_daily_workflow_run():
    """Baca ringkasan run TERAKHIR dari scripts/daily_global_workflow_cli.py
    (dijalankan Task Scheduler tanpa pengawasan tiap sore hari bursa). Sebelum
    ini, dashboard tidak punya visibilitas apa pun ke run otomatis itu --
    alert 'job background bermasalah' di tab Beranda cuma memantau job yang
    dipicu manual dari tombol UI (data/jobs/), bukan otomasi Task Scheduler
    (data/daily_workflows/). Dipakai bersama perbaikan try/except di
    daily_global_workflow_cli.py yang sekarang SELALU menulis status FAILED +
    alasan kalau ada step yang gagal, alih-alih crash senyap tanpa jejak."""
    workflow_dir = os.path.join(DATA_DIR, "daily_workflows")
    if not os.path.exists(workflow_dir):
        return None
    files = [f for f in os.listdir(workflow_dir) if f.startswith("daily_global_workflow_") and f.endswith(".json")]
    if not files:
        return None
    latest_path = max(
        (os.path.join(workflow_dir, f) for f in files),
        key=lambda p: os.path.getmtime(p),
    )
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            run = json.load(f)
    except Exception as e:
        return {"status": "UNKNOWN", "finished_at": None, "message": f"Gagal membaca ringkasan workflow harian: {e}"}
    run["_mtime"] = os.path.getmtime(latest_path)
    return run


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


@st.cache_data(show_spinner=False, ttl=60)
def load_latest_walk_forward_edge_status():
    """Membaca hasil walk-forward-vs-baseline (run_analysis.py) dari semua job
    JSON di data/jobs -- angka ini SEBELUMNYA cuma dicetak ke console dan tidak
    pernah tersimpan/tampil di dashboard, padahal ini validasi paling ketat
    (backtest walk-forward) yang ada di sistem. Untuk tiap ticker, ambil entry
    "analyzed" terbaru saja (job terbaru menang kalau ticker sama muncul di
    beberapa job)."""
    if not os.path.exists(JOB_DIR):
        return pd.DataFrame()

    latest_by_ticker = {}
    for filename in os.listdir(JOB_DIR):
        if not filename.startswith("analysis_") or not filename.endswith(".json"):
            continue
        path = os.path.join(JOB_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                status = json.load(f)
        except Exception:
            continue
        updated_at = str(status.get("updated_at", ""))
        for entry in (status.get("summary") or {}).get("analyzed", []):
            ticker_code = entry.get("ticker")
            if not ticker_code:
                continue
            existing = latest_by_ticker.get(ticker_code)
            if existing is not None and existing["_updated_at"] >= updated_at:
                continue
            enriched = dict(entry)
            enriched["_updated_at"] = updated_at
            latest_by_ticker[ticker_code] = enriched

    if not latest_by_ticker:
        return pd.DataFrame()
    return pd.DataFrame(latest_by_ticker.values()).rename(columns={"_updated_at": "computed_at"})


EDGE_SCREENING_PATH = os.path.join(DATA_DIR, "edge_screening_status.json")


@st.cache_data(show_spinner=False, ttl=60)
def load_genuine_edge_screening():
    """Membaca hasil scripts/screen_genuine_edge.py -- screening walk-forward
    penuh (bukan cuma ticker yang kebetulan pernah dianalisis lewat job biasa)
    untuk seluruh ticker di config/stocks.yaml. Dipakai untuk menggerbang
    dashboard supaya hanya bekerja dengan ticker yang benar-benar terbukti
    mengalahkan baseline naif, bukan seluruh ticker tanpa pandang bulu."""
    if not os.path.exists(EDGE_SCREENING_PATH):
        return None
    try:
        with open(EDGE_SCREENING_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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


def start_background_analysis_job(job_tickers, lstm_epochs=3, duplicate_policy="skip", prediction_run_type=None, force_retrain=False, include_lstm=False):
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
    if include_lstm:
        command.append("--include-lstm")
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

    daily_mode = display_mode
    st.caption(
        f"Mode tampilan **{display_mode}**, batas reliability **{global_min_reliability:.0f}%**, "
        f"batas confidence **{global_min_confidence:.0f}%**, minimal track record **{int(global_min_evaluations_trading)}** -- "
        "atur di sidebar 'Pengaturan Sensitivitas & Tampilan' (berlaku untuk semua tab)."
    )

    edge_screening = load_genuine_edge_screening()
    genuine_edge_lookup = None
    full_edge_lookup = None
    if edge_screening is None:
        st.caption(
            "Filter edge nyata (walk-forward) belum aktif -- jalankan `python scripts/screen_genuine_edge.py` "
            "untuk membuat data/edge_screening_status.json terlebih dahulu."
        )
    else:
        edge_basis = st.radio(
            "Definisi 'edge nyata' untuk gate ini",
            ["Horizon manapun (H+1/H+3/H+5/H+10)", "Hanya H+1"],
            horizontal=True,
            help="Screening walk-forward penuh terakhir: "
            f"{edge_screening.get('total_screened', 0)} ticker, selesai {edge_screening.get('finished_at', '-')}.",
        )
        edge_field = "has_any_genuine_edge" if edge_basis.startswith("Horizon manapun") else "has_genuine_edge_h1"
        full_edge_lookup = {r["ticker"]: bool(r.get(edge_field, False)) for r in edge_screening.get("results", [])}
        apply_edge_gate = st.checkbox(
            "Hanya proses ticker dengan edge nyata (walk-forward screening penuh)",
            value=False,
            help=f"{sum(full_edge_lookup.values())}/{len(full_edge_lookup)} ticker lolos berdasarkan definisi di atas.",
        )
        if apply_edge_gate:
            genuine_edge_lookup = full_edge_lookup

    daily_scope_tickers = active_tickers if active_tickers else tickers
    decision_board_df = build_daily_decision_board(
        daily_scope_tickers,
        min_reliability=float(global_min_reliability),
        min_confidence=float(global_min_confidence),
        min_evaluations=int(global_min_evaluations_trading),
        portfolio_capital=float(portfolio_capital),
        risk_per_trade_pct=float(risk_per_trade_pct),
        genuine_edge_lookup=genuine_edge_lookup,
        edge_lookup_for_badge=full_edge_lookup,
        var95_lookup=get_latest_var95_lookup(),
    )
    decision_board_df = apply_personalization(decision_board_df)

    regime_history_df = load_regime_history()
    regime_streak = summarize_regime_streaks(regime_history_df)
    if regime_streak["current_regime"]:
        avg_duration = regime_streak["avg_duration_by_regime"].get(regime_streak["current_regime"])
        avg_duration_text = f", historisnya rata-rata bertahan ~{avg_duration:.0f} hari" if avg_duration else " (belum cukup riwayat untuk rata-rata historis)"
        st.caption(
            f"📅 Konteks regime pasar: sudah **{regime_streak['current_streak_days']} hari** di regime "
            f"**{regime_streak['current_regime']}**{avg_duration_text}. Riwayat dicatat otomatis tiap hari dashboard "
            "ini dibuka atau workflow harian dijalankan."
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
        q1, q2, q3 = st.columns(3)
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

        st.subheader("Checklist Sebelum Sesi Berikutnya")
        render_daily_checklist(decision_board_df, recent_jobs)

        st.subheader("Rencana Trading Besok")
        st.caption(
            "**Kolom Verifikasi** -- satu badge, dua pengujian sekaligus: ✅ Terverifikasi Ganda = lolos "
            "riwayat prediksi nyata (live) DAN backtest walk-forward. ⚠️ = cuma lolos salah satu. "
            "❌ = belum lolos keduanya. Kalau cuma mau lihat satu angka, cukup lihat badge ini saja -- "
            "tab lain (Model Trust Audit, Walk-Forward Genuine Edge) berisi rincian yang mendasarinya."
        )
        if daily_mode == "Pemula":
            display_cols = ["Saham", "Sinyal", "Entry Area", "Stop Loss", "Target H+3", "Alasan Utama", "Verifikasi", "Risiko"]
        elif daily_mode == "Trader":
            display_cols = [
                "Saham",
                "Sinyal",
                "Harga Terakhir",
                "Entry Area",
                "Stop Loss",
                "Basis SL",
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
                "Verifikasi",
                "Skor Personal",
                "Alasan Utama",
            ]
        else:
            display_cols = [
                "Saham",
                "Sinyal",
                "Harga Terakhir",
                "Entry Area",
                "Stop Loss",
                "Basis SL",
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
                "Verifikasi",
                "Skor Personal",
                "Dimute",
                "Risiko",
                "Status Data",
                "Status Analisis",
                "Tanggal Data",
                "Alasan Utama",
                "Catatan Trust",
                "Catatan Verifikasi",
            ]

        action_filter = st.multiselect(
            "Filter sinyal",
            options=decision_board_df["Sinyal"].dropna().unique().tolist(),
            default=st.session_state.get("daily_signal_filter", decision_board_df["Sinyal"].dropna().unique().tolist()),
            key="daily_signal_filter",
        )
        hide_muted = st.checkbox(
            "Sembunyikan ticker yang saya mute",
            value=True,
            help="Ticker yang di-mute lewat panel 'Detail Per Saham' disembunyikan dari tabel ini, "
            "tapi tetap dianalisis di belakang layar (bukan dihapus datanya).",
        )
        view_board_df = decision_board_df[decision_board_df["Sinyal"].isin(action_filter)].copy() if action_filter else decision_board_df
        if hide_muted and "Dimute" in view_board_df.columns:
            view_board_df = view_board_df[~view_board_df["Dimute"]]
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
                "Skor Personal": "{:+.2f}",
            }),
            width="stretch",
            hide_index=True,
        )

        with st.expander("🔍 Detail Per Saham (Trust + Kenapa Model Memprediksi Ini)", expanded=False):
            detail_tickers = view_board_df["Saham"].dropna().unique().tolist()
            if not detail_tickers:
                st.caption("Tidak ada saham pada filter saat ini untuk ditampilkan detailnya.")
            else:
                detail_ticker_choice = st.selectbox("Pilih saham untuk detail", options=detail_tickers, key="daily_detail_ticker")
                detail_row = view_board_df[view_board_df["Saham"] == detail_ticker_choice].iloc[0]
                st.info(f"**{detail_row['Verifikasi']}** -- {detail_row['Catatan Verifikasi']}")

                def _render_xai_section():
                    wf_edge_detail_df = load_latest_walk_forward_edge_status()
                    wf_row = (
                        wf_edge_detail_df[wf_edge_detail_df["ticker"].astype(str).str.upper() == detail_ticker_choice]
                        if not wf_edge_detail_df.empty else pd.DataFrame()
                    )
                    if wf_row.empty:
                        st.caption(
                            "Belum ada data walk-forward/XAI untuk ticker ini -- jalankan analisis lewat tab Workflow Harian terlebih dahulu."
                        )
                        return
                    wf_row = wf_row.iloc[0]
                    dcol1, dcol2 = st.columns(2)
                    for col, xai_key, title in (
                        (dcol1, "xai_direction_h1", "Kenapa Arah H+1 Diprediksi Begini"),
                        (dcol2, "xai_return_h3", "Kenapa Proyeksi Return H+3 Begini"),
                    ):
                        with col:
                            st.markdown(f"**{title}**")
                            explanation = wf_row.get(xai_key)
                            if not isinstance(explanation, dict) or not explanation.get("available"):
                                reason = explanation.get("reason") if isinstance(explanation, dict) else "Data tidak tersedia -- jalankan analisis dulu."
                                st.caption(f"Belum tersedia: {reason}")
                                continue
                            feat_df = pd.DataFrame(explanation["top_features"])
                            render_interactive_contribution_bar(feat_df, title=title)
                            st.dataframe(
                                feat_df[["feature", "value", "contribution", "direction"]].head(5).style.format({
                                    "value": "{:.4g}",
                                    "contribution": "{:+.4f}",
                                }),
                                width="stretch",
                                hide_index=True,
                            )

                if daily_mode == "Pemula":
                    with st.expander("📊 Lihat Detail Teknis (Kenapa Model Memprediksi Ini -- opsional)", expanded=False):
                        _render_xai_section()
                else:
                    _render_xai_section()

                st.divider()
                st.markdown("**Beri Umpan Balik untuk Rekomendasi Ini**")
                st.caption(
                    "Umpan balik ini TIDAK mengubah prediksi model -- hanya tersimpan sebagai jurnal keputusan "
                    "Anda sendiri (dan dasar personalisasi tampilan di masa depan)."
                )
                fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)
                fb_actions = [
                    (fb_col1, "✅ Saya Ikuti", "IKUTI"),
                    (fb_col2, "⏭ Saya Lewati", "LEWATI"),
                    (fb_col3, "👍 Berguna", "BERGUNA"),
                    (fb_col4, "👎 Tidak Berguna", "TIDAK_BERGUNA"),
                ]
                for fb_col, fb_label, fb_action in fb_actions:
                    if fb_col.button(fb_label, key=f"fb_{fb_action}_{detail_ticker_choice}"):
                        log_user_feedback(detail_ticker_choice, detail_row["Sinyal"], fb_action)
                        st.success(f"Tersimpan: {fb_label}")

                ticker_feedback_history = load_user_feedback()
                if not ticker_feedback_history.empty:
                    ticker_feedback_history = ticker_feedback_history[
                        ticker_feedback_history["ticker"].astype(str).str.upper() == detail_ticker_choice
                    ]
                if not ticker_feedback_history.empty:
                    with st.expander(f"Riwayat umpan balik Anda untuk {detail_ticker_choice}", expanded=False):
                        st.dataframe(
                            ticker_feedback_history.sort_values("timestamp", ascending=False),
                            width="stretch",
                            hide_index=True,
                        )

                current_profile = load_user_profile()
                is_muted = detail_ticker_choice in current_profile["muted_tickers"]
                mute_col1, mute_col2 = st.columns([1, 3])
                with mute_col1:
                    if is_muted:
                        if st.button("🔊 Un-mute Ticker Ini", key=f"unmute_{detail_ticker_choice}"):
                            unmute_ticker(detail_ticker_choice)
                            st.success(f"{detail_ticker_choice} tidak lagi di-mute.")
                    else:
                        if st.button("🔇 Mute Ticker Ini", key=f"mute_{detail_ticker_choice}"):
                            mute_ticker(detail_ticker_choice)
                            st.success(f"{detail_ticker_choice} di-mute dari tampilan Rencana Trading.")
                with mute_col2:
                    st.caption(
                        "Mute menyembunyikan ticker ini dari tabel Rencana Trading (bukan dari analisis) -- "
                        "berguna untuk saham yang tidak relevan buat Anda, terlepas dari sinyal modelnya."
                    )

                st.markdown("---")
                st.markdown("**📏 Expected Range Probabilistik (EWMA terkalibrasi -- BUKAN target harga)**")
                try:
                    from src.trading.interval_forecaster import (
                        compute_ewma_sigma,
                        expected_range,
                        load_k_table,
                        log_issued_interval,
                        position_size_from_range,
                    )

                    _ivl_raw_path = project_path("data", "raw", f"{detail_ticker_choice}_raw.csv")
                    if not os.path.exists(_ivl_raw_path):
                        st.caption("Data harga lokal ticker ini belum ada di data/raw -- jalankan update harga dulu.")
                    else:
                        _ivl_px = pd.read_csv(_ivl_raw_path)
                        _ivl_close = pd.to_numeric(_ivl_px.get("close"), errors="coerce").dropna()
                        if len(_ivl_close) < 60:
                            st.caption("Riwayat harga < 60 hari -- sigma EWMA belum stabil, rentang tidak ditampilkan.")
                        else:
                            _ivl_sigma = compute_ewma_sigma(_ivl_close)
                            _ivl_last = float(_ivl_close.iloc[-1])
                            _ivl_date = str(_ivl_px["timestamp"].iloc[-1]) if "timestamp" in _ivl_px.columns else "?"
                            _ivl_rows = []
                            for _ivl_h in (1, 3, 5, 10):
                                _ivl_kh = load_k_table(BASE_DIR, horizon_days=_ivl_h)
                                _ivl_row = {"Horizon": f"H+{_ivl_h} hari bursa"}
                                for _ivl_t, _ivl_kv in sorted(_ivl_kh.items()):
                                    _lo, _hi = expected_range(_ivl_last, _ivl_sigma, _ivl_h, _ivl_kv)
                                    _ivl_row[f"Band {int(_ivl_t * 100)}% (k={_ivl_kv})"] = f"{_lo:,.0f} - {_hi:,.0f}"
                                _ivl_rows.append(_ivl_row)
                            _ivl_k = load_k_table(BASE_DIR, horizon_days=10)
                            st.dataframe(pd.DataFrame(_ivl_rows), hide_index=True, width="stretch")
                            st.caption(
                                f"Basis: close {_ivl_last:,.0f} ({_ivl_date}), sigma harian EWMA {_ivl_sigma * 100:.2f}%. "
                                "Rentang kewajaran statistik dengan coverage terkalibrasi lintas-universe -- bukan sinyal arah, "
                                "bukan target. Kalibrasi k dibekukan di data/interval_calibration.csv (fallback: default universe 2026-07-20)."
                            )
                            _ivl_verified = "Terverifikasi Ganda" in str(detail_row.get("Verifikasi", ""))
                            _ivl_size = position_size_from_range(
                                20_000_000, 1.0, _ivl_last, _ivl_sigma, 10, sorted(_ivl_k.items())[0][1], _ivl_verified
                            )
                            st.caption(
                                f"Sizing volatilitas ({_ivl_size['mode']}): risk budget 1% x Rp20 juta -> "
                                f"maks {_ivl_size['lots']} lot (referensi batas bawah band 80% H+10 = {_ivl_size.get('stop_reference', 0):,.0f}). "
                                + ("" if _ivl_verified else "Berstatus SIMULASI karena ticker belum '✅ Terverifikasi Ganda' (prinsip desain #4).")
                            )
                            if st.button(
                                "📝 Catat interval H+10 ke log monitoring",
                                key=f"ivl_log_{detail_ticker_choice}",
                                help="Menambah 1 baris ke data/interval_log.csv (atomic, dedup per ticker+tanggal). "
                                "Setelah >=30 interval melewati horizonnya, coverage aktual bisa diuji dari log ini.",
                            ):
                                log_issued_interval(
                                    detail_ticker_choice, _ivl_date, _ivl_last, _ivl_sigma, 10, _ivl_k, BASE_DIR
                                )
                                st.success("Interval tercatat ke data/interval_log.csv.")
                except Exception as _ivl_err:  # jangan pernah menjatuhkan dashboard karena fitur presentasi
                    st.caption(f"Expected range tidak tersedia: {_ivl_err}")

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


with tab_beranda:
    st.header("🏠 Beranda")
    st.caption(
        "Satu pandangan ringkas untuk semua yang penting hari ini -- kalau cuma sempat buka satu tab, "
        "buka ini. Detail lengkap tetap ada di tab lain (Ringkasan Harian, Akurasi Model, Sentimen Pasar)."
    )

    if decision_board_df.empty:
        st.info("Belum ada data prediksi untuk ditampilkan. Jalankan Workflow Harian terlebih dahulu.")
    else:
        has_verifikasi = "Verifikasi" in decision_board_df.columns
        buy_mask = decision_board_df["Sinyal"].astype(str).str.contains("BUY", regex=False)
        verified_mask = (
            decision_board_df["Verifikasi"] == "✅ Terverifikasi Ganda"
            if has_verifikasi else pd.Series(False, index=decision_board_df.index)
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sinyal BUY Hari Ini", int(buy_mask.sum()))
        k2.metric("Terverifikasi Ganda", int(verified_mask.sum()) if has_verifikasi else "-")
        k3.metric(
            "Regime Pasar",
            regime_streak["current_regime"] or "-",
            help=(
                f"Sudah {regime_streak['current_streak_days']} hari di regime ini"
                if regime_streak["current_regime"] else "Belum ada riwayat regime."
            ),
        )
        k4.metric("Data Terakhir", daily_readiness.get("latest_data", "-"))

        st.divider()
        st.subheader("🎯 Top Picks (Sinyal BUY + Terverifikasi Ganda)")
        st.caption(
            "Bukan daftar semua saham -- cuma yang lolos DUA syarat sekaligus: sinyal BUY dan sudah "
            "terverifikasi lewat live track record MAUPUN backtest walk-forward."
        )
        top_picks_df = decision_board_df[buy_mask & verified_mask].copy() if has_verifikasi else pd.DataFrame()
        if top_picks_df.empty:
            st.info(
                "Belum ada ticker yang memenuhi kedua syarat sekaligus saat ini -- ini bukan error, "
                "memang tidak selalu ada peluang berkualitas tinggi tiap hari."
            )
        else:
            top_pick_cols = [c for c in ["Saham", "Sinyal", "Entry Area", "Stop Loss", "Target H+3", "Verifikasi", "Alasan Utama"] if c in top_picks_df.columns]
            st.dataframe(top_picks_df[top_pick_cols], width="stretch", hide_index=True)

        st.divider()
        st.subheader("⚠️ Peringatan")
        alerts = []
        if daily_readiness.get("status") not in ("SIAP PREDIKSI HARIAN", "SIAP PREDIKSI - RETRAIN TERJADWAL"):
            alerts.append(f"Status kesiapan: **{daily_readiness.get('status')}** -- {daily_readiness.get('action', '')}")
        troubled_jobs = [job for job in recent_jobs if job.get("status") in ("STALE", "FAILED", "UNKNOWN")]
        if troubled_jobs:
            alerts.append(f"{len(troubled_jobs)} job background bermasalah -- cek tab Workflow Harian.")
        latest_daily_run = get_latest_daily_workflow_run()
        if latest_daily_run and latest_daily_run.get("status") == "FAILED":
            failed_steps = [s.get("step") for s in latest_daily_run.get("steps", []) if s.get("status") == "FAILED"]
            alerts.append(
                "Workflow harian otomatis (Task Scheduler) TERAKHIR gagal"
                + (f" di step: {', '.join(failed_steps)}" if failed_steps else "")
                + " -- prediksi/data mungkin tidak up-to-date. Cek `logs/daily_workflow.log`."
            )
        elif latest_daily_run and latest_daily_run.get("_mtime"):
            hours_since = (datetime.now().timestamp() - latest_daily_run["_mtime"]) / 3600
            if hours_since > 48 and datetime.now().weekday() < 5:
                alerts.append(
                    f"Workflow harian otomatis belum jalan lagi sejak >{int(hours_since // 24)} hari -- "
                    "cek apakah Task Scheduler masih aktif (`Get-ScheduledTaskInfo -TaskName 'AITrading_DailyWorkflow'`)."
                )
        if "Status Analisis" in decision_board_df.columns:
            unfinished_count = int((decision_board_df["Status Analisis"] != "LENGKAP").sum())
            if unfinished_count:
                alerts.append(f"{unfinished_count} saham belum lengkap analisisnya -- cek tab Workflow Harian.")
        if not alerts:
            st.success("Tidak ada peringatan aktif. Semua sistem berjalan normal.")
        else:
            for alert in alerts:
                st.warning(alert)

        st.caption(
            "Detail lengkap: tab **Ringkasan Harian** (semua sinyal), **Akurasi Model** (kepercayaan model), "
            "**Sentimen Pasar** (konteks berita)."
        )


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
            st.success("Prediksi harian Global Model selesai. Hasilnya masuk ke Ranking Mentah (Riset) dengan nama model Global-*.")

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
                    st.success("Proses harian disarankan selesai. Cek Ringkasan Harian untuk rencana trading yang sudah tergate.")
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

    if LEGACY_MODELS_ENABLED:
        col_status, col_force = st.columns(2)
        with col_status:
            if st.button("Cek Status Update Harga", disabled=not bool(selected_tickers)):
                st.session_state["local_data_status"] = get_local_data_status(selected_tickers)
        with col_force:
            force_analysis_clicked = st.button(
                "Paksa Analisis Ulang dari Data Lokal",
                disabled=not bool(selected_tickers),
            )
    else:
        force_analysis_clicked = False
        if st.button("Cek Status Update Harga", disabled=not bool(selected_tickers)):
            st.session_state["local_data_status"] = get_local_data_status(selected_tickers)

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

    if LEGACY_MODELS_ENABLED:
        st.markdown("---")
        st.subheader("8. Job Background Analisis")
        st.caption(
            "Gunakan mode background untuk analisis banyak saham. Proses tetap berjalan sebagai proses terpisah, "
            "sementara dashboard hanya membaca progres dari file status."
        )
        bg_col1, bg_col2 = st.columns([1, 1])
        with bg_col1:
            background_include_lstm = st.checkbox(
                "Sertakan LSTM",
                value=False,
                help=(
                    "Model paling mahal dilatih (PyTorch) dan belum divalidasi walk-forward apa pun -- "
                    "default mati supaya job background lebih ringan. Aktifkan hanya kalau butuh proyeksi LSTM."
                ),
                key="background_include_lstm",
            )
            background_lstm_epochs = st.number_input(
                "Epoch LSTM untuk job background",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                disabled=not background_include_lstm,
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

    if LEGACY_MODELS_ENABLED:
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
                    disabled=not bool(unfinished_tickers),
                )

                if rerun_unfinished_clicked:
                    if execution_mode == "Background":
                        job_id = start_background_analysis_job(
                            unfinished_tickers,
                            lstm_epochs=int(background_lstm_epochs),
                            duplicate_policy=daily_duplicate_policy,
                            prediction_run_type=daily_prediction_run_type,
                            include_lstm=bool(background_include_lstm),
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
                                include_lstm=bool(background_include_lstm),
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
                include_lstm=bool(background_include_lstm),
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
                    include_lstm=bool(background_include_lstm),
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
    if LEGACY_MODELS_ENABLED:
        rerun_analysis_after_update = st.checkbox(
            "Jalankan analisis ulang otomatis setelah update data",
            value=False,
            help="Setelah data harga diperbarui, dashboard akan membuat ulang prediksi/ranking untuk ticker terkait.",
        )
        analysis_scope_after_update = st.radio(
            "Cakupan analisis ulang",
            ["Hanya saham yang berhasil diperbarui", "Semua saham yang dipilih"],
            horizontal=True,
            disabled=not rerun_analysis_after_update,
        )
        analysis_include_lstm = st.checkbox(
            "Sertakan LSTM saat analisis ulang otomatis",
            value=False,
            disabled=not rerun_analysis_after_update,
            help=(
                "Model paling mahal dilatih (PyTorch) dan belum divalidasi walk-forward apa pun -- "
                "default mati supaya analisis otomatis tetap ringan."
            ),
            key="analysis_include_lstm",
        )
        analysis_lstm_epochs = st.number_input(
            "Epoch LSTM untuk analisis otomatis",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            disabled=not rerun_analysis_after_update or not analysis_include_lstm,
            help="Semakin besar epoch, analisis lebih lama. Untuk update otomatis, 3 epoch biasanya cukup.",
        )
        st.caption(
            "Epoch LSTM = jumlah putaran belajar model dari data historis. "
            "Nilai kecil lebih cepat dan cocok untuk analisis harian; nilai terlalu besar bisa lebih lama dan berisiko overfit pada pola lama."
        )
    else:
        rerun_analysis_after_update = False
        analysis_scope_after_update = "Hanya saham yang berhasil diperbarui"
        analysis_include_lstm = False
        analysis_lstm_epochs = 3

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
                        include_lstm=bool(analysis_include_lstm),
                    )
                    st.session_state["last_auto_analysis_summary"] = analysis_summary
                st.success("Analisis ulang selesai. Tab Ringkasan Harian dan Akurasi Model sudah memakai prediksi terbaru.")
            else:
                st.info("Tidak ada saham yang perlu dianalisis ulang dari hasil update ini.")

    if "last_update_summary" in st.session_state:
        render_update_summary(st.session_state["last_update_summary"])

    if "last_auto_analysis_summary" in st.session_state:
        analysis_summary = st.session_state["last_auto_analysis_summary"]
        st.subheader("Ringkasan Analisis Ulang Otomatis")
        render_analysis_summary(analysis_summary)

with tab_ranking:
    st.header("Ranking Mentah (Riset)")
    if display_mode == "Pemula":
        st.info(
            "Tab ini berisi ranking mentah untuk riset/analisis lanjutan -- **belum difilter status trust/edge**, "
            "jadi tidak cocok jadi acuan langsung untuk mode Pemula. Gunakan tab **Ringkasan Harian** yang sudah "
            "menyaring saham berdasarkan status kepercayaan (live track record + backtest walk-forward). "
            "Ganti mode tampilan ke Trader/Audit di sidebar kalau tetap ingin membuka tab ini."
        )
    else:
        st.write("Menampilkan potensi kenaikan dan penurunan berdasarkan hasil prediksi model terbaru.")
        st.warning(
            "⚠️ Tabel di tab ini diurutkan dari **akurasi mentah/potensi return**, BELUM difilter status "
            "trust/edge -- ticker berstatus 'JANGAN DIIKUTI' (tab Model Trust Audit) atau 'TIDAK ADA EDGE' "
            "(tab Walk-Forward Genuine Edge) tetap bisa muncul di ranking teratas. Cek kolom 'Status Trust' "
            "di bawah, atau pakai tab **Ringkasan Harian** kalau butuh daftar yang sudah tergate."
        )

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
            render_feature_status("Ranking Mentah (Riset)", "BELUM LENGKAP", "Belum ada prediksi aktif.", "Jalankan analisis saham terlebih dahulu.")
        else:
            latest_prediction_date = pred_status_df["current_date"].dropna().astype(str).max()
            active_prediction_count = int(pred_status_df["is_active"].sum())
            render_feature_status(
                "Ranking Mentah (Riset)",
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

                # Status Trust/Edge dihitung fresh di sini (bukan dipinjam dari tab
                # lain) supaya tab ini tetap benar berdiri sendiri kalau urutan tab
                # berubah. Dipakai untuk menutup gap: tab ini sebelumnya bisa
                # menampilkan ticker "JANGAN DIIKUTI" di posisi #1 tanpa peringatan.
                ranking_trust_audit_df = get_model_trust_audit(
                    prediction_purpose=reliability_purpose,
                    min_evaluations=3,
                )
                ranking_trust_lookup = {}
                if not ranking_trust_audit_df.empty:
                    ranking_trust_lookup = ranking_trust_audit_df.set_index(["ticker", "model_name"])["status_trust"].to_dict()
                ranking_edge_screening = load_genuine_edge_screening()
                ranking_edge_lookup = {}
                if ranking_edge_screening is not None:
                    edge_field = "has_genuine_edge_h1" if ranking_mode == "Arah Harian H+1" else "has_any_genuine_edge"
                    ranking_edge_lookup = {r["ticker"]: bool(r.get(edge_field, False)) for r in ranking_edge_screening.get("results", [])}
                latest_preds["Status Trust"] = latest_preds.apply(
                    lambda row: ranking_trust_lookup.get((row["ticker"], row["model_name"]), "PERLU DATA LAGI"),
                    axis=1,
                )
                latest_preds["Edge Nyata (Walk-Forward)"] = latest_preds["ticker"].map(
                    lambda t: ("Ya" if ranking_edge_lookup.get(t, False) else "Tidak") if ranking_edge_lookup else "Belum discreening"
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
                        display_gainers = top_gainers[['ticker', 'current_price', 'predicted_price', 'potential_return_pct', 'confidence_pct', 'historical_reliability', 'ranking_score', 'Status Trust', 'Edge Nyata (Walk-Forward)', 'target_date']].copy()
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
                        untrusted_in_top = int((display_gainers['Status Trust'] == 'JANGAN DIIKUTI').sum())
                        if untrusted_in_top:
                            st.error(
                                f"{untrusted_in_top} dari 10 ranking teratas berstatus 'JANGAN DIIKUTI' -- "
                                "jangan eksekusi tanpa cek ulang tab Model Trust Audit."
                            )
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
        holdout = engine_status.get("holdout_evaluation")
        if holdout:
            st.write(
                f"**Akurasi held-out (test {holdout['n_test']} baris, train {holdout['n_train']} baris):** "
                f"{holdout['accuracy_pct']}%"
            )
            f1_text = ", ".join(f"{label}: {score}" for label, score in holdout["f1_per_class"].items())
            st.caption(f"F1 per kelas (data test, bukan data latih): {f1_text}")
            st.caption(
                "Kelas dengan data test sedikit (mis. NEGATIVE) wajar punya F1 kurang stabil -- "
                "jangan diperlakukan sama percaya dirinya dengan kelas yang datanya banyak."
            )
        elif engine_status.get("holdout_evaluation_note"):
            st.caption(f"Validasi akurasi belum tersedia: {engine_status['holdout_evaluation_note']}")

        st.divider()
        st.caption(
            "Perbandingan robust engine sentimen (TF-IDF+SVM produksi vs 3 alternatif berbasis model bahasa "
            "pretrained, diuji 5 split acak) dipindahkan keluar dari dashboard -- alat riset, bukan kebutuhan "
            "pemantauan harian, dan butuh dependency berat (`transformers`+`torch`, unduhan model ~500MB-1GB). "
            "Jalankan manual dari terminal: `python scripts/compare_sentiment_engines_cli.py`."
        )

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
        value=sentiment_online_enabled,
        key="sentiment_online_first",
        disabled=not sentiment_online_enabled,
    )
    if not sentiment_online_enabled:
        online_first = False
    include_article_body = st.checkbox(
        "Analisis isi artikel jika halaman berita bisa dibaca",
        value=sentiment_online_enabled,
        key="sentiment_include_article_body",
        disabled=not sentiment_online_enabled,
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

        st.subheader("1. Cari Kata Kunci Sentimen Otomatis (Gemini AI)")
        st.caption(
            "Langkah opsional SEBELUM mengisi Ticker & Query di bawah -- minta Gemini AI mencari ticker mana "
            f"dari {len(tickers)} saham di config/stocks.yaml yang paling relevan untuk analisis sentimen pasar "
            "Indonesia saat ini, lengkap dengan saran query pencarian beritanya. Bukan pengganti langkah 2 di "
            "bawah, cuma bantu mengisinya otomatis."
        )
        gemini_status = check_gemini_status()
        if not gemini_status["enabled"] or not gemini_status["ok"]:
            st.caption(f"⚪ {gemini_status['message']} Isi `GEMINI_API_KEY` di file `.env` untuk mengaktifkan (lihat `.env.example`).")
        else:
            st.caption(f"🟢 {gemini_status['message']}")
            gemini_col1, gemini_col2 = st.columns([1, 2])
            gemini_top_n = gemini_col1.number_input(
                "Jumlah ticker disarankan",
                min_value=1,
                max_value=15,
                value=5,
                step=1,
                key="gemini_keyword_top_n",
            )
            gemini_search_clicked = gemini_col2.button(
                "🔍 Cari Kata Kunci dengan Gemini AI",
                key="gemini_search_keywords",
                type="primary",
                disabled=not sentiment_online_enabled,
            )
            if not sentiment_online_enabled:
                st.caption("Nonaktif -- aktifkan mode online (`AI_TRADING_EXTERNAL_SERVICES=true` di `.env` + update harga online tidak dimatikan) dulu.")

            if gemini_search_clicked:
                with st.spinner("Meminta Gemini AI mencari sentimen pasar saham Indonesia terkini..."):
                    st.session_state["gemini_keyword_suggestion"] = suggest_sentiment_keywords(tickers, top_n=int(gemini_top_n))

            gemini_suggestion = st.session_state.get("gemini_keyword_suggestion")
            if gemini_suggestion:
                if gemini_suggestion["warning"]:
                    st.warning(gemini_suggestion["warning"])
                if gemini_suggestion["error"]:
                    st.error(gemini_suggestion["error"])
                    if gemini_suggestion["raw_response"]:
                        with st.expander("Detail respons mentah (debug)"):
                            st.text(gemini_suggestion["raw_response"])
                elif gemini_suggestion["suggestions"]:
                    grounded_note = "hasil pencarian live" if gemini_suggestion["grounded"] else "dari pengetahuan model, bukan pencarian live"
                    st.success(f"{len(gemini_suggestion['suggestions'])} kandidat ticker ditemukan ({grounded_note}).")
                    for idx, row in enumerate(gemini_suggestion["suggestions"]):
                        sugg_col1, sugg_col2 = st.columns([5, 1])
                        sugg_col1.markdown(f"**{row['ticker']}** -- {row['reason']}  \nQuery: `{row['query']}`")
                        if sugg_col2.button("Pakai", key=f"use_gemini_suggestion_{idx}"):
                            st.session_state["auto_news_ticker"] = row["ticker"]
                            st.session_state["auto_news_query"] = row["query"]
                            st.rerun()

        st.divider()
        st.subheader("2. Ambil Berita untuk Ticker Terpilih")
        col_ticker, col_limit = st.columns([2, 1])
        # setdefault (bukan value= langsung) supaya tidak konflik dengan
        # session_state yang di-set tombol "Pakai" di langkah 1 (mengisi
        # auto_news_ticker/auto_news_query otomatis) -- Streamlit memperingatkan
        # kalau widget punya value= SEKALIGUS key yang sudah di-set manual.
        st.session_state.setdefault("auto_news_ticker", ticker if "ticker" in locals() else "BBRI")
        news_ticker = col_ticker.text_input(
            "Ticker untuk pencarian berita",
            key="auto_news_ticker",
        ).strip().upper()
        news_limit = col_limit.number_input("Jumlah headline", min_value=1, max_value=30, value=10, step=1)
        st.session_state.setdefault("auto_news_query", f"{news_ticker} saham" if news_ticker else "saham Indonesia")
        news_query = st.text_input(
            "Query pencarian",
            key="auto_news_query",
        )
        news_include_article_body = st.checkbox(
            "Ambil dan analisis isi artikel",
            value=True,
            key="auto_news_include_article_body",
        )

        if st.button("Ambil Berita Terbaru", key="fetch_latest_news", type="primary", disabled=not sentiment_online_enabled):
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
            
    min_samples = global_min_evaluations_audit
    st.caption(
        f"Minimal evaluasi untuk model dianggap cukup reliabel: **{int(min_samples)}** "
        "(atur di sidebar 'Pengaturan Sensitivitas & Tampilan')."
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
    wf_edge_df = load_latest_walk_forward_edge_status()
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
            render_interactive_accuracy_trend(
                overall_chart_df, "evaluation_day", "direction_accuracy_pct", "model_name",
                title="Tren Akurasi Arah per Model",
            )

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
        view_wf_edge_df = wf_edge_df.copy()
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
            if not view_wf_edge_df.empty:
                view_wf_edge_df = view_wf_edge_df[view_wf_edge_df["ticker"].astype(str).str.upper() == ticker]

        st.subheader("📋 Ringkasan Kepercayaan Model -- mulai di sini")
        st.caption(
            "Menggabungkan status trust audit (track record live, dari tab 'Model Trust Audit') dengan genuine "
            "edge walk-forward (backtest, dari tab 'Walk-Forward Genuine Edge') per ticker. Tab-tab di bawah "
            "berisi detail lengkapnya -- pakai ringkasan ini kalau cuma ingin tahu ticker mana yang paling bisa dipercaya."
        )
        if view_trust_audit_df.empty and view_wf_edge_df.empty:
            st.info("Belum ada data trust audit maupun walk-forward untuk ditampilkan. Jalankan analisis dan evaluasi akurasi terlebih dahulu.")
        else:
            best_trust_by_ticker = pd.DataFrame(columns=["ticker", "status_trust", "alasan"])
            if not view_trust_audit_df.empty:
                best_trust_by_ticker = view_trust_audit_df.drop_duplicates(subset=["ticker"], keep="first")[
                    ["ticker", "status_trust", "alasan"]
                ]
            trust_lookup_summary = best_trust_by_ticker.set_index("ticker")["status_trust"].to_dict()
            trust_reason_summary = best_trust_by_ticker.set_index("ticker")["alasan"].to_dict()
            edge_lookup_summary = {}
            if not view_wf_edge_df.empty and "has_genuine_edge_h1" in view_wf_edge_df.columns:
                edge_lookup_summary = view_wf_edge_df.set_index("ticker")["has_genuine_edge_h1"].to_dict()

            summary_tickers = sorted(set(trust_lookup_summary.keys()) | set(edge_lookup_summary.keys()))
            badge_rows = []
            for t in summary_tickers:
                status = trust_lookup_summary.get(t)
                has_edge = edge_lookup_summary.get(t) if t in edge_lookup_summary else None
                has_edge = bool(has_edge) if has_edge is not None and not pd.isna(has_edge) else None
                badge, reason = compute_unified_trust_badge(status, has_edge)
                badge_rows.append({
                    "Saham": t,
                    "Verifikasi": badge,
                    "Trust Live": status or "Belum ada data",
                    "Genuine Edge (Walk-Forward H+1)": "Ada" if has_edge is True else ("Tidak ada" if has_edge is False else "Belum discreening"),
                    "Catatan": reason or trust_reason_summary.get(t, ""),
                })
            badge_summary_df = pd.DataFrame(badge_rows)
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("✅ Terverifikasi Ganda", int((badge_summary_df["Verifikasi"] == "✅ Terverifikasi Ganda").sum()))
            b2.metric("⚠️ Live OK Saja", int(badge_summary_df["Verifikasi"].str.startswith("⚠️ Live OK").sum()))
            b3.metric("⚠️ Backtest OK Saja", int((badge_summary_df["Verifikasi"] == "⚠️ Backtest OK, Live Kurang").sum()))
            b4.metric("❌ Belum Lolos", int((badge_summary_df["Verifikasi"] == "❌ Belum Lolos Verifikasi").sum()))
            st.dataframe(badge_summary_df, width="stretch", hide_index=True)
        st.divider()
        st.caption(
            "3 tab pertama untuk keputusan sehari-hari. Tab '🔧 Detail Teknis Lainnya' berisi 6 sub-tab "
            "analisis mendalam (Prediksi Akurasi Tinggi, Leaderboard, Tier, Kalibrasi, Ringkasan, Drill-down) "
            "-- dikumpulkan di satu tab supaya tidak memenuhi bar tab utama."
        )

        trust_tab, rec_tab, wf_edge_tab, detail_tab = st.tabs([
            "Model Trust Audit",
            "Rekomendasi Model per Saham (Riset)",
            "Walk-Forward Genuine Edge",
            "🔧 Detail Teknis Lainnya",
        ])

        with rec_tab:
            st.write("Model terbaik dipilih berdasarkan akurasi arah, error margin, dan jumlah sample evaluasi.")
            rec_display_df = view_recommendations_df.copy()
            if not view_trust_audit_df.empty and "status_trust" in view_trust_audit_df.columns:
                rec_trust_lookup = view_trust_audit_df.set_index(["ticker", "model_name"])["status_trust"].to_dict()
                rec_display_df["status_trust"] = rec_display_df.apply(
                    lambda row: rec_trust_lookup.get((row["ticker"], row["model_name"]), "PERLU DATA LAGI"),
                    axis=1,
                )
                untrusted_rec_count = int((rec_display_df["status_trust"] == "JANGAN DIIKUTI").sum())
                if untrusted_rec_count:
                    st.warning(
                        f"{untrusted_rec_count} baris di tabel ini berstatus 'JANGAN DIIKUTI' pada tab Model Trust "
                        "Audit di sebelah (edge_vs_baseline_pct belum lolos ambang) -- urutan di sini murni akurasi "
                        "mentah, belum tergate trust/edge. Cek kolom 'status_trust' sebelum menindaklanjuti."
                    )
            else:
                st.caption(
                    "⚠️ Tabel ini belum tergate status trust/edge (lihat tab 'Model Trust Audit' di sebelah) -- "
                    "urutan di sini murni akurasi mentah historis."
                )
            st.dataframe(rec_display_df.style.format({
                "reliability_score": "{:.2f}",
                "direction_accuracy_pct": "{:.2f}%",
                "precision_naik_pct": "{:.2f}%",
                "avg_return_after_naik_pct": "{:+.2f}%",
                "avg_error_margin_pct": "{:.2f}%",
                "avg_traded_value": "{:,.0f}",
            }), width="stretch")

        with trust_tab:
            st.write(
                "Audit ini merangkum apakah model per saham sudah layak dijadikan acuan trading berdasarkan sample, "
                "akurasi, profit factor, kalibrasi, dan edge nyata terhadap baseline tebak-mayoritas naif "
                "(bukan sekadar akurasi mentah, yang bisa menyesatkan kalau periode evaluasi kebetulan searah tren pasar)."
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
                    "baseline_majority_accuracy_pct": "{:.2f}%",
                    "edge_vs_baseline_pct": "{:+.2f}pp",
                    "profit_factor": "{:.2f}",
                    "calibration_gap_pct": "{:+.2f}%",
                    "win_rate_pct": "{:.2f}%",
                    "avg_strategy_return_pct": "{:+.2f}%",
                    "trading_score": "{:.2f}",
                    "precision_naik_pct": "{:.2f}%",
                    "avg_return_after_naik_pct": "{:+.2f}%",
                }), width="stretch", hide_index=True)

        with wf_edge_tab:
            st.write(
                "Ini hasil backtest walk-forward (bukan track record live seperti tab Model Trust Audit): tiap ticker "
                "diuji ulang lewat train/test bergulir dengan purge gap, lalu akurasinya dibandingkan ke baseline "
                "tebak-mayoritas pada fold yang sama. Sebelumnya angka ini cuma tercetak di console saat menjalankan "
                "analisis dan tidak pernah tersimpan/tampil di dashboard."
            )
            st.warning(
                "⚠️ Angka di tab ini berasal dari run analisis SATU-TICKER (data/jobs/) dan belum dikoreksi "
                "multiple-testing (FDR) -- menguji ambang effect-size tetap ke banyak ticker tanpa koreksi ini bisa "
                "meloloskan beberapa ticker murni karena varians sampel, bukan edge sungguhan. Verdict yang SUDAH "
                "dikoreksi FDR lintas seluruh universe ticker ada di gate 'Ringkasan Harian'/badge 'Terverifikasi "
                "Ganda' (bersumber dari `data/edge_screening_status.json`, dibuat lewat "
                "`python scripts/screen_genuine_edge.py`) -- itu acuan untuk keputusan trading, bukan tabel di bawah ini."
            )
            if view_wf_edge_df.empty:
                st.info(
                    "Belum ada data. Jalankan analisis (tab Workflow Harian / Analisis Manual) minimal sekali -- "
                    "hasilnya otomatis tersimpan di data/jobs/analysis_<job_id>.json dan akan muncul di sini."
                )
            else:
                only_genuine_edge_h1 = st.checkbox(
                    "Tampilkan hanya ticker dengan edge nyata H+1 (has_genuine_edge_h1)",
                    value=False,
                )
                edge_display_df = view_wf_edge_df.copy()
                if only_genuine_edge_h1 and "has_genuine_edge_h1" in edge_display_df.columns:
                    edge_display_df = edge_display_df[edge_display_df["has_genuine_edge_h1"] == True]  # noqa: E712

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Edge nyata H+1", int(view_wf_edge_df.get("has_genuine_edge_h1", pd.Series(dtype=bool)).sum()))
                e2.metric("Edge nyata H+3", int(view_wf_edge_df.get("has_genuine_edge_h3", pd.Series(dtype=bool)).sum()))
                e3.metric("Edge nyata H+5", int(view_wf_edge_df.get("has_genuine_edge_h5", pd.Series(dtype=bool)).sum()))
                e4.metric("Edge nyata H+10", int(view_wf_edge_df.get("has_genuine_edge_h10", pd.Series(dtype=bool)).sum()))

                edge_table_columns = [
                    "ticker", "computed_at",
                    "walk_forward_h1_accuracy_pct", "walk_forward_h1_baseline_majority_pct",
                    "walk_forward_h1_edge_pct", "has_genuine_edge_h1",
                    "walk_forward_h3_edge_mae_pct", "has_genuine_edge_h3",
                    "walk_forward_h5_edge_mae_pct", "has_genuine_edge_h5",
                    "walk_forward_h10_edge_mae_pct", "has_genuine_edge_h10",
                ]
                available_columns = [c for c in edge_table_columns if c in edge_display_df.columns]
                st.dataframe(
                    edge_display_df[available_columns].sort_values("walk_forward_h1_edge_pct", ascending=False).style.format({
                        "walk_forward_h1_accuracy_pct": "{:.1f}%",
                        "walk_forward_h1_baseline_majority_pct": "{:.1f}%",
                        "walk_forward_h1_edge_pct": "{:+.1f}pp",
                        "walk_forward_h3_edge_mae_pct": "{:+.2f}pp",
                        "walk_forward_h5_edge_mae_pct": "{:+.2f}pp",
                        "walk_forward_h10_edge_mae_pct": "{:+.2f}pp",
                    }, na_rep="-"),
                    width="stretch",
                    hide_index=True,
                )

                st.divider()
                st.subheader("Kenapa Model Memprediksi Ini? (XAI / SHAP)")
                st.caption(
                    "Kontribusi tiap fitur terhadap prediksi arah H+1 dan proyeksi return H+3 untuk hari data terakhir "
                    "yang dianalisis -- bukan rata-rata umum, tapi alasan spesifik untuk prediksi ticker ini."
                )
                xai_ticker_options = sorted(view_wf_edge_df["ticker"].astype(str).str.upper().unique().tolist())
                if xai_ticker_options:
                    default_index = xai_ticker_options.index(ticker) if ticker in xai_ticker_options else 0
                    xai_selected_ticker = st.selectbox(
                        "Pilih ticker untuk dijelaskan", options=xai_ticker_options, index=default_index, key="xai_ticker_select"
                    )
                    xai_row = view_wf_edge_df[view_wf_edge_df["ticker"].astype(str).str.upper() == xai_selected_ticker]
                    if not xai_row.empty:
                        xai_row = xai_row.iloc[0]
                        xai_col1, xai_col2 = st.columns(2)
                        for col, key, title in (
                            (xai_col1, "xai_direction_h1", "Arah H+1 (NAIK/TURUN)"),
                            (xai_col2, "xai_return_h3", "Proyeksi Return H+3"),
                        ):
                            with col:
                                st.markdown(f"**{title}**")
                                explanation = xai_row.get(key)
                                if not isinstance(explanation, dict) or not explanation.get("available"):
                                    reason = explanation.get("reason") if isinstance(explanation, dict) else "Data tidak tersedia."
                                    st.caption(f"Penjelasan tidak tersedia: {reason}")
                                    continue
                                feat_df = pd.DataFrame(explanation["top_features"])
                                render_interactive_contribution_bar(feat_df, title=title)
                                st.dataframe(
                                    feat_df[["feature", "value", "contribution", "direction"]].style.format({
                                        "value": "{:.4g}",
                                        "contribution": "{:+.4f}",
                                    }),
                                    width="stretch",
                                    hide_index=True,
                                )

        with detail_tab:
            if display_mode == "Pemula":
                st.info(
                    "6 sub-tab analisis mendalam di sini untuk audit/riset (kalibrasi confidence, evaluasi per "
                    "tier, drill-down harian, dll) -- bukan kebutuhan pemantauan harian untuk mode Pemula. "
                    "Ganti mode tampilan ke Trader/Audit di sidebar kalau tetap ingin membukanya."
                )
            else:
                st.caption("6 sub-tab analisis mendalam -- untuk audit/riset, bukan kebutuhan pemantauan harian.")
                high_pred_tab, leaderboard_tab, tier_tab, calibration_tab, summary_tab, daily_tab = st.tabs([
                    "Prediksi Akurasi Tinggi",
                    "Leaderboard Trading Model",
                    "Evaluasi per Tier",
                    "Kalibrasi Confidence",
                    "Ringkasan Akurasi Model",
                    "Drill-down Rekap Harian per Saham",
                ])

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
                        render_interactive_accuracy_trend(
                            chart_df, "evaluation_day", "direction_accuracy_pct", "model_name",
                            title="Tren Akurasi Arah per Model (Drill-down)",
                        )
    else:
        st.info("Belum ada data akurasi yang dievaluasi. Lakukan prediksi hari ini dan update data besok untuk melihat hasilnya.")

