import pandas as pd

from src.trading.personalization import (
    apply_personalization,
    compute_personal_scores,
    load_user_profile,
    mute_ticker,
    save_user_profile,
    unmute_ticker,
)
from src.utils.user_feedback import log_user_feedback


def test_load_user_profile_returns_default_when_missing(tmp_path):
    path = tmp_path / "user_profile.json"

    profile = load_user_profile(str(path))

    assert profile["muted_tickers"] == []
    assert profile["personal_risk_tolerance"] is None


def test_mute_and_unmute_ticker_roundtrip(tmp_path):
    path = tmp_path / "user_profile.json"

    profile = mute_ticker("bbri", path=str(path))
    assert profile["muted_tickers"] == ["BBRI"]

    profile = mute_ticker("bbri", path=str(path))
    assert profile["muted_tickers"] == ["BBRI"], "muting twice should not duplicate"

    profile = unmute_ticker("BBRI", path=str(path))
    assert profile["muted_tickers"] == []

    reloaded = load_user_profile(str(path))
    assert reloaded["muted_tickers"] == []


def test_compute_personal_scores_with_monkeypatched_feedback_file(tmp_path, monkeypatch):
    feedback_path = tmp_path / "user_feedback_log.csv"
    monkeypatch.setattr("src.utils.user_feedback.USER_FEEDBACK_FILE", str(feedback_path))

    log_user_feedback("BBRI", "🟢 BUY", "ikuti")
    log_user_feedback("BBRI", "🟢 BUY", "berguna")
    log_user_feedback("BBRI", "🟡 WATCH", "lewati")
    log_user_feedback("BBCA", "🔴 AVOID", "tidak_berguna")
    log_user_feedback("TLKM", "🟡 WATCH", "tidak_berguna")
    log_user_feedback("TLKM", "🟡 WATCH", "lewati")

    scores = compute_personal_scores(str(feedback_path))

    assert scores["BBRI"] == round(1 / 3, 3)
    assert scores["BBCA"] == -1.0
    assert scores["TLKM"] == -1.0


def test_apply_personalization_adds_columns_without_changing_existing_ones(tmp_path, monkeypatch):
    feedback_path = tmp_path / "user_feedback_log.csv"
    profile_path = tmp_path / "user_profile.json"
    monkeypatch.setattr("src.utils.user_feedback.USER_FEEDBACK_FILE", str(feedback_path))

    log_user_feedback("BBRI", "🟢 BUY", "tidak_berguna")
    log_user_feedback("BBRI", "🟢 BUY", "tidak_berguna")
    save_user_profile({"muted_tickers": ["BBCA"], "personal_risk_tolerance": None}, path=str(profile_path))

    board_df = pd.DataFrame([
        {"Saham": "BBRI", "Sinyal": "🟢 BUY", "Confidence": 80.0},
        {"Saham": "BBCA", "Sinyal": "🟡 WATCH", "Confidence": 60.0},
        {"Saham": "TLKM", "Sinyal": "🔴 AVOID", "Confidence": 40.0},
    ])

    result = apply_personalization(board_df, feedback_path=str(feedback_path), profile_path=str(profile_path))

    assert list(board_df.columns) == ["Saham", "Sinyal", "Confidence"], "original df must not be mutated"
    assert result.loc[result["Saham"] == "BBRI", "Sinyal"].iloc[0] == "🟢 BUY"
    assert result.loc[result["Saham"] == "BBRI", "Confidence"].iloc[0] == 80.0
    assert result.loc[result["Saham"] == "BBRI", "Skor Personal"].iloc[0] == -1.0
    assert bool(result.loc[result["Saham"] == "BBCA", "Dimute"].iloc[0]) is True
    assert bool(result.loc[result["Saham"] == "TLKM", "Dimute"].iloc[0]) is False
    assert result.loc[result["Saham"] == "TLKM", "Skor Personal"].iloc[0] == 0.0


def test_apply_personalization_handles_empty_board(tmp_path):
    result = apply_personalization(pd.DataFrame())
    assert result.empty
