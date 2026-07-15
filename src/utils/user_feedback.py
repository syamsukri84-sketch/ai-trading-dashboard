"""Log umpan balik eksplisit pengguna terhadap rekomendasi dashboard.

PENTING (lihat ROADMAP_COGNITIVE_DASHBOARD.md): file ini HANYA mencatat
preferensi subjektif pengguna. Data di sini TIDAK BOLEH dipakai untuk melatih
ulang atau mengubah bobot DirectionClassifier/PriceProjector/model prediksi
apa pun -- "suka/tidak suka" pengguna bukan indikator kebenaran statistik
prediksi. Dipakai nanti HANYA untuk lapisan personalisasi/ranking tampilan
(Bagian A3 di roadmap), yang berdiri terpisah dari model prediksi.
"""

import os
from datetime import datetime

import pandas as pd
from pandas.errors import EmptyDataError

from src.utils.atomic_io import atomic_write_csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USER_FEEDBACK_FILE = os.path.join(PROJECT_ROOT, "data", "tracking", "user_feedback_log.csv")

VALID_ACTIONS = {"IKUTI", "LEWATI", "BERGUNA", "TIDAK_BERGUNA"}


def _ensure_dir(filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def load_user_feedback(path: str = USER_FEEDBACK_FILE) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["timestamp", "ticker", "signal_shown", "action", "note"])
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=["timestamp", "ticker", "signal_shown", "action", "note"])


def log_user_feedback(ticker: str, signal_shown: str, action: str, note: str = "") -> pd.DataFrame:
    """Mencatat satu umpan balik pengguna terhadap sinyal yang ditampilkan.

    `action` salah satu dari IKUTI/LEWATI/BERGUNA/TIDAK_BERGUNA. Setiap klik
    adalah event independen (append-only) -- tidak ada konsep "supersede"
    seperti pada log prediksi model, karena tiap umpan balik adalah keputusan
    pengguna pada satu titik waktu, bukan sesuatu yang perlu ditimpa.
    """
    action = str(action).upper().strip()
    if action not in VALID_ACTIONS:
        raise ValueError(f"action harus salah satu dari {VALID_ACTIONS}, dapat: {action}")

    _ensure_dir(USER_FEEDBACK_FILE)
    new_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": str(ticker).replace(".JK", "").upper().strip(),
        "signal_shown": str(signal_shown),
        "action": action,
        "note": str(note or ""),
    }])
    existing = load_user_feedback(USER_FEEDBACK_FILE)
    updated = pd.concat([existing, new_row], ignore_index=True)
    atomic_write_csv(updated, USER_FEEDBACK_FILE, index=False)
    return updated


def get_feedback_summary_by_ticker(path: str = USER_FEEDBACK_FILE) -> pd.DataFrame:
    """Ringkasan jumlah tiap jenis aksi per ticker -- dasar untuk lapisan
    personalisasi di masa depan (Bagian A3), belum dipakai untuk apa pun
    selain ditampilkan sebagai jurnal keputusan pengguna saat ini."""
    df = load_user_feedback(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "IKUTI", "LEWATI", "BERGUNA", "TIDAK_BERGUNA", "total"])

    pivot = df.pivot_table(index="ticker", columns="action", values="timestamp", aggfunc="count", fill_value=0)
    for action in VALID_ACTIONS:
        if action not in pivot.columns:
            pivot[action] = 0
    pivot = pivot[sorted(VALID_ACTIONS)]
    pivot["total"] = pivot.sum(axis=1)
    return pivot.reset_index().sort_values("total", ascending=False)
