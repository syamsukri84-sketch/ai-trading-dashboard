import pytest

from src.utils.user_feedback import (
    get_feedback_summary_by_ticker,
    load_user_feedback,
    log_user_feedback,
)


def test_log_user_feedback_appends_rows(tmp_path, monkeypatch):
    path = tmp_path / "user_feedback_log.csv"
    monkeypatch.setattr("src.utils.user_feedback.USER_FEEDBACK_FILE", str(path))

    log_user_feedback("bbri", "🟢 BUY", "ikuti", note="entry area masuk akal")
    updated = log_user_feedback("BBCA", "🟡 WATCH", "TIDAK_BERGUNA")

    assert len(updated) == 2
    assert updated.iloc[0]["ticker"] == "BBRI"
    assert updated.iloc[0]["action"] == "IKUTI"
    assert updated.iloc[1]["ticker"] == "BBCA"
    assert updated.iloc[1]["action"] == "TIDAK_BERGUNA"
    assert path.exists()


def test_log_user_feedback_rejects_invalid_action(tmp_path, monkeypatch):
    path = tmp_path / "user_feedback_log.csv"
    monkeypatch.setattr("src.utils.user_feedback.USER_FEEDBACK_FILE", str(path))

    with pytest.raises(ValueError):
        log_user_feedback("BBRI", "🟢 BUY", "SUKA_BANGET")


def test_load_user_feedback_returns_empty_when_missing(tmp_path, monkeypatch):
    path = tmp_path / "user_feedback_log.csv"
    monkeypatch.setattr("src.utils.user_feedback.USER_FEEDBACK_FILE", str(path))

    df = load_user_feedback(str(path))

    assert df.empty
    assert list(df.columns) == ["timestamp", "ticker", "signal_shown", "action", "note"]


def test_feedback_summary_by_ticker_counts_actions(tmp_path, monkeypatch):
    path = tmp_path / "user_feedback_log.csv"
    monkeypatch.setattr("src.utils.user_feedback.USER_FEEDBACK_FILE", str(path))

    log_user_feedback("BBRI", "🟢 BUY", "ikuti")
    log_user_feedback("BBRI", "🟢 BUY", "berguna")
    log_user_feedback("BBRI", "🟡 WATCH", "lewati")
    log_user_feedback("BBCA", "🔴 AVOID", "tidak_berguna")

    summary = get_feedback_summary_by_ticker(str(path))
    bbri_row = summary[summary["ticker"] == "BBRI"].iloc[0]

    assert bbri_row["total"] == 3
    assert bbri_row["IKUTI"] == 1
    assert bbri_row["BERGUNA"] == 1
    assert bbri_row["LEWATI"] == 1
