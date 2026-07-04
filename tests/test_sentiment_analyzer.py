import pandas as pd

from src.nlp.sentiment_analyzer import (
    analyze_dataframe,
    append_issue,
    append_issues,
    build_local_sentiment_dataset,
    build_trading_sentiment_summary,
    get_sentiment_engine_status,
    load_issues,
)


def test_append_issue_and_analyze(tmp_path):
    path = tmp_path / "market_issues.csv"

    append_issue(
        path,
        date="2026-06-22",
        ticker="bbri",
        source="manual",
        text="BBRI mencatat laba meningkat dan pertumbuhan kredit kuat.",
    )

    issues = load_issues(path)
    scored = analyze_dataframe(issues)

    assert len(scored) == 1
    assert scored["ticker"].iloc[0] == "BBRI"
    assert scored["label"].iloc[0] == "POSITIVE"


def test_append_issues_removes_duplicate_rows(tmp_path):
    path = tmp_path / "market_issues.csv"
    rows = [
        {
            "date": "2026-06-22",
            "ticker": "bbri",
            "source": "news",
            "text": "BBRI laba meningkat.",
        },
        {
            "date": "2026-06-22",
            "ticker": "BBRI",
            "source": "news",
            "text": "BBRI laba meningkat.",
        },
    ]

    updated = append_issues(path, rows)

    assert len(updated) == 1
    assert updated["ticker"].iloc[0] == "BBRI"


def test_build_trading_sentiment_summary_positive_news_is_contrarian_bearish():
    scored = analyze_dataframe(
        pd.DataFrame([
            {
                "date": "2026-06-22",
                "ticker": "BBRI",
                "source": "news",
                "text": "BBRI mencatat laba naik kuat dan pertumbuhan kredit stabil.",
            },
            {
                "date": "2026-06-22",
                "ticker": "BBRI",
                "source": "news",
                "text": "Dividen BBRI meningkat dan prospek bisnis positif.",
            },
        ])
    )

    summary = build_trading_sentiment_summary(scored, "BBRI")

    assert summary["bias"] == "BEARISH"
    assert summary["risk_level"] == "HIGH"
    assert summary["interpretation_mode"] == "contrarian_indonesia"
    assert summary["market_signal_score"] < 0
    assert summary["positive_count"] == 2
    assert summary["news_count"] == 2
    assert summary["key_drivers"]


def test_build_trading_sentiment_summary_empty():
    summary = build_trading_sentiment_summary(analyze_dataframe(load_issues("missing.csv")), "BBRI")

    assert summary["bias"] == "NEUTRAL"
    assert summary["news_count"] == 0


def test_sentiment_engine_status_has_required_fields():
    status = get_sentiment_engine_status()

    assert status["engine"] in {"ml_tfidf_linear_svm", "lexicon"}
    assert status["label"]
    assert status["dataset_path"]
    assert "description" in status


def test_build_local_sentiment_dataset(tmp_path):
    source = tmp_path / "market_issues.csv"
    output = tmp_path / "processed" / "financial_news_clean.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-22",
                "ticker": "BBRI",
                "source": "manual",
                "text": "BBRI mencatat laba meningkat dan kredit tumbuh kuat.",
            },
            {
                "date": "2026-06-23",
                "ticker": "TLKM",
                "source": "manual",
                "text": "TLKM menghadapi tekanan margin dan risiko penurunan laba.",
            },
        ]
    ).to_csv(source, index=False)

    dataset = build_local_sentiment_dataset(source, output, include_seed_examples=False)

    assert output.exists()
    assert {"text", "label", "clean_text", "source"}.issubset(dataset.columns)
    assert {"ticker", "date", "label_method", "label_confidence"}.issubset(dataset.columns)
    assert len(dataset) == 2
    assert set(dataset["label"]).issubset({"positive", "neutral", "negative"})


def test_build_local_sentiment_dataset_keeps_manual_seed_labels(tmp_path, monkeypatch):
    source = tmp_path / "market_issues.csv"
    output = tmp_path / "processed" / "financial_news_clean.csv"
    seed = tmp_path / "seed_examples.csv"
    pd.DataFrame(
        [{"date": "2026-06-22", "ticker": "BBRI", "source": "manual", "text": "BBRI laba meningkat."}]
    ).to_csv(source, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-07-03",
                "ticker": "IHSG",
                "source": "seed_negative",
                "text": "IHSG melemah karena tekanan jual asing.",
                "manual_label": "negative",
            }
        ]
    ).to_csv(seed, index=False)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.get_seed_sentiment_examples_path", lambda: seed)

    dataset = build_local_sentiment_dataset(source, output)

    seed_row = dataset[dataset["dataset_source"] == "seed_examples"].iloc[0]
    assert seed_row["label"] == "negative"
    assert seed_row["label_method"] == "manual_seed_label"
