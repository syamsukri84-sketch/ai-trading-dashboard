"""Lapisan personalisasi tampilan berbasis umpan balik pengguna.

PENTING (lihat ROADMAP_COGNITIVE_DASHBOARD.md, prinsip desain wajib): modul
ini HANYA memengaruhi urutan/prioritas TAMPILAN (skor personal untuk ranking,
ticker yang di-mute dari tampilan). Modul ini TIDAK PERNAH mengubah nilai
Sinyal, Confidence, edge_vs_baseline_pct, atau input apa pun ke model
prediksi -- "suka/tidak suka" pengguna bukan indikator kebenaran statistik
prediksi. Kalau ada dorongan untuk membuat modul ini memengaruhi model
prediksi, itu tandanya salah tempat -- baca lagi prinsip desain di roadmap.
"""

import json
import os

from src.utils import user_feedback as user_feedback_module
from src.utils.user_feedback import get_feedback_summary_by_ticker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USER_PROFILE_FILE = os.path.join(PROJECT_ROOT, "data", "user_profile.json")

DEFAULT_PROFILE = {
    "muted_tickers": [],
    "personal_risk_tolerance": None,
}


def load_user_profile(path: str = USER_PROFILE_FILE) -> dict:
    if not os.path.exists(path):
        return dict(DEFAULT_PROFILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception:
        return dict(DEFAULT_PROFILE)
    merged = dict(DEFAULT_PROFILE)
    merged.update(profile)
    return merged


def save_user_profile(profile: dict, path: str = USER_PROFILE_FILE) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def mute_ticker(ticker: str, path: str = USER_PROFILE_FILE) -> dict:
    profile = load_user_profile(path)
    ticker = str(ticker).replace(".JK", "").upper().strip()
    if ticker not in profile["muted_tickers"]:
        profile["muted_tickers"].append(ticker)
    save_user_profile(profile, path)
    return profile


def unmute_ticker(ticker: str, path: str = USER_PROFILE_FILE) -> dict:
    profile = load_user_profile(path)
    ticker = str(ticker).replace(".JK", "").upper().strip()
    profile["muted_tickers"] = [t for t in profile["muted_tickers"] if t != ticker]
    save_user_profile(profile, path)
    return profile


def compute_personal_scores(feedback_path: str | None = None) -> dict:
    """Skor personal per ticker dari akumulasi umpan balik pengguna.

    Skor makin TINGGI kalau ticker sering ditandai IKUTI/BERGUNA, makin
    RENDAH kalau sering LEWATI/TIDAK_BERGUNA -- rentang -1.0 s.d. 1.0, 0.0
    kalau belum ada feedback sama sekali. Skor ini HANYA dipakai untuk
    ranking/urutan tampilan, bukan nilai prediksi apa pun.
    """
    path = feedback_path or user_feedback_module.USER_FEEDBACK_FILE
    summary = get_feedback_summary_by_ticker(path)
    if summary.empty:
        return {}

    scores = {}
    for _, row in summary.iterrows():
        positive = row.get("IKUTI", 0) + row.get("BERGUNA", 0)
        negative = row.get("LEWATI", 0) + row.get("TIDAK_BERGUNA", 0)
        total = positive + negative
        scores[row["ticker"]] = round((positive - negative) / total, 3) if total else 0.0
    return scores


def apply_personalization(board_df, feedback_path: str | None = None, profile_path: str = USER_PROFILE_FILE):
    """Menambah kolom 'Skor Personal' dan 'Dimute' ke DataFrame papan
    keputusan berdasarkan riwayat umpan balik & profil pengguna.

    TIDAK mengubah kolom Sinyal/Confidence/edge yang sudah ada -- baris yang
    di-mute TETAP ada di data (bukan dihapus) supaya transparan; caller (UI)
    yang memilih untuk menyembunyikannya lewat kolom 'Dimute'.
    """
    if board_df.empty:
        return board_df

    profile = load_user_profile(profile_path)
    scores = compute_personal_scores(feedback_path)

    result = board_df.copy()
    result["Skor Personal"] = result["Saham"].map(scores).fillna(0.0)
    result["Dimute"] = result["Saham"].isin(profile["muted_tickers"])
    return result
