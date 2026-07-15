import pandas as pd
import pytest

from src.nlp.sentiment_analyzer import (
    SENTIMENT_LABELS,
    _evaluate_embedding_classifier_model,
    _evaluate_indobert_model,
    _evaluate_sentiment_model,
    _load_embedding_model,
    _load_indobert_sentiment_model,
    _mean_pool_embeddings,
    _predict_with_indobert,
    analyze_dataframe,
    append_issue,
    append_issues,
    build_local_sentiment_dataset,
    build_trading_sentiment_summary,
    compare_sentiment_engines,
    compare_sentiment_engines_repeated,
    DRIFT_Z_THRESHOLD,
    compute_sentiment_drift,
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


def _make_labeled_dataset(rows_per_label: int = 15) -> pd.DataFrame:
    positive_texts = [
        "laba perusahaan meningkat tajam dan kinerja bisnis menguat",
        "pendapatan tumbuh kuat didukung ekspansi pasar baru",
        "manajemen optimis pertumbuhan berlanjut tahun depan",
    ]
    neutral_texts = [
        "perusahaan mengumumkan jadwal rapat pemegang saham tahunan",
        "manajemen memberikan keterangan rutin kepada otoritas bursa",
        "laporan keuangan kuartalan dipublikasikan sesuai jadwal",
    ]
    negative_texts = [
        "laba perusahaan anjlok akibat tekanan biaya operasional",
        "kinerja bisnis melemah dan risiko gagal bayar meningkat",
        "penurunan penjualan tajam membebani prospek perusahaan",
    ]
    rows = []
    for i in range(rows_per_label):
        rows.append({"clean_text": positive_texts[i % len(positive_texts)] + f" edisi {i}", "label": "positive"})
        rows.append({"clean_text": neutral_texts[i % len(neutral_texts)] + f" edisi {i}", "label": "neutral"})
        rows.append({"clean_text": negative_texts[i % len(negative_texts)] + f" edisi {i}", "label": "negative"})
    return pd.DataFrame(rows)


def test_evaluate_sentiment_model_reports_holdout_metrics(tmp_path):
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=15).to_csv(dataset_path, index=False)

    evaluation = _evaluate_sentiment_model(dataset_path)

    assert evaluation is not None
    assert evaluation["n_train"] + evaluation["n_test"] == 45
    assert 0.0 <= evaluation["accuracy_pct"] <= 100.0
    assert set(evaluation["f1_per_class"].keys()) == {"POSITIVE", "NEUTRAL", "NEGATIVE"}
    assert sum(evaluation["test_label_counts"].values()) == evaluation["n_test"]


def test_evaluate_sentiment_model_returns_none_for_tiny_dataset(tmp_path):
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=1).to_csv(dataset_path, index=False)

    assert _evaluate_sentiment_model(dataset_path) is None


def test_load_indobert_returns_none_when_transformers_not_installed(monkeypatch):
    _load_indobert_sentiment_model.cache_clear()
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoTokenizer", None)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModelForSequenceClassification", None)

    assert _load_indobert_sentiment_model() is None
    _load_indobert_sentiment_model.cache_clear()


def test_load_indobert_returns_none_when_download_fails(monkeypatch):
    _load_indobert_sentiment_model.cache_clear()

    class _RaisingLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("simulated: tidak ada koneksi internet / model tidak ter-cache")

    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoTokenizer", _RaisingLoader)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModelForSequenceClassification", _RaisingLoader)

    assert _load_indobert_sentiment_model() is None
    _load_indobert_sentiment_model.cache_clear()


def test_predict_with_indobert_returns_none_when_model_unavailable(monkeypatch):
    _load_indobert_sentiment_model.cache_clear()
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoTokenizer", None)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModelForSequenceClassification", None)

    assert _predict_with_indobert("BBRI mencatat laba meningkat.") is None
    _load_indobert_sentiment_model.cache_clear()


def test_predict_with_indobert_returns_none_for_empty_text():
    assert _predict_with_indobert("   ") is None


@pytest.mark.slow
def test_predict_with_indobert_real_inference_returns_valid_result():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    _load_indobert_sentiment_model.cache_clear()

    result = _predict_with_indobert("BBRI mencatat laba meningkat dan pertumbuhan kredit kuat.")

    if result is None:
        pytest.skip("Model IndoBERT tidak bisa dimuat (kemungkinan tidak ada koneksi internet/cache).")
    assert result.label in SENTIMENT_LABELS
    assert result.method == "indobert_pretrained"
    assert 0.0 <= result.score <= 1.0


def test_evaluate_indobert_model_returns_none_when_unavailable(tmp_path, monkeypatch):
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=15).to_csv(dataset_path, index=False)
    _load_indobert_sentiment_model.cache_clear()
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoTokenizer", None)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModelForSequenceClassification", None)

    assert _evaluate_indobert_model(dataset_path) is None
    _load_indobert_sentiment_model.cache_clear()


def test_evaluate_indobert_model_returns_none_for_tiny_dataset(tmp_path):
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=1).to_csv(dataset_path, index=False)

    assert _evaluate_indobert_model(dataset_path) is None


def test_compare_sentiment_engines_handles_indobert_unavailable(tmp_path, monkeypatch):
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=15).to_csv(dataset_path, index=False)
    _load_indobert_sentiment_model.cache_clear()
    _load_embedding_model.cache_clear()
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoTokenizer", None)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModelForSequenceClassification", None)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModel", None)

    comparison = compare_sentiment_engines(dataset_path)

    assert comparison["tfidf_svm"] is not None
    assert comparison["indobert_pretrained"] is None
    assert comparison["indo_e5_embedding_svm"] is None
    assert comparison["indobert_embedding_svm"] is None
    assert comparison["verdict"] == "TIDAK BISA DIBANDINGKAN"
    _load_indobert_sentiment_model.cache_clear()
    _load_embedding_model.cache_clear()


def test_mean_pool_embeddings_respects_attention_mask():
    torch = pytest.importorskip("torch")

    class _FakeTokenizer:
        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=256):
            input_ids = torch.zeros((len(texts), 3), dtype=torch.long)
            attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]][: len(texts)], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _FakeOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    class _FakeModel:
        def __call__(self, input_ids, attention_mask):
            batch, seq_len = input_ids.shape
            hidden = (
                torch.arange(1, seq_len + 1, dtype=torch.float32)
                .view(1, seq_len, 1)
                .expand(batch, seq_len, 2)
                .clone()
            )
            return _FakeOutput(hidden)

    embeddings = _mean_pool_embeddings(_FakeTokenizer(), _FakeModel(), ["teks a", "teks b"])

    assert embeddings.shape == (2, 2)
    assert embeddings[0][0] == pytest.approx(2.0)
    assert embeddings[1][0] == pytest.approx(1.5)


def test_evaluate_embedding_classifier_model_returns_none_when_unavailable(tmp_path, monkeypatch):
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=15).to_csv(dataset_path, index=False)
    _load_embedding_model.cache_clear()
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoTokenizer", None)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModel", None)

    assert _evaluate_embedding_classifier_model(dataset_path, "some/model-id") is None
    _load_embedding_model.cache_clear()


def test_evaluate_embedding_classifier_model_returns_none_for_tiny_dataset(tmp_path):
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=1).to_csv(dataset_path, index=False)

    assert _evaluate_embedding_classifier_model(dataset_path, "some/model-id") is None


@pytest.mark.slow
def test_evaluate_embedding_classifier_model_real_inference_matches_tfidf_structure(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=15).to_csv(dataset_path, index=False)
    _load_embedding_model.cache_clear()

    tfidf_result = _evaluate_sentiment_model(dataset_path)
    embedding_result = _evaluate_embedding_classifier_model(dataset_path, "LazarusNLP/all-indo-e5-small-v4")

    if embedding_result is None:
        pytest.skip("Model embedding tidak bisa dimuat (kemungkinan tidak ada koneksi internet/cache).")
    assert embedding_result["n_train"] == tfidf_result["n_train"]
    assert embedding_result["n_test"] == tfidf_result["n_test"]
    assert 0.0 <= embedding_result["accuracy_pct"] <= 100.0
    assert set(embedding_result["f1_per_class"].keys()) == set(SENTIMENT_LABELS)
    _load_embedding_model.cache_clear()


@pytest.mark.slow
def test_evaluate_indobert_model_real_inference_matches_tfidf_structure(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=15).to_csv(dataset_path, index=False)
    _load_indobert_sentiment_model.cache_clear()

    tfidf_result = _evaluate_sentiment_model(dataset_path)
    indobert_result = _evaluate_indobert_model(dataset_path)

    if indobert_result is None:
        pytest.skip("Model IndoBERT tidak bisa dimuat (kemungkinan tidak ada koneksi internet/cache).")
    assert indobert_result["n_train"] == tfidf_result["n_train"]
    assert indobert_result["n_test"] == tfidf_result["n_test"]
    assert indobert_result["test_label_counts"] == tfidf_result["test_label_counts"]
    assert 0.0 <= indobert_result["accuracy_pct"] <= 100.0
    assert set(indobert_result["f1_per_class"].keys()) == set(SENTIMENT_LABELS)


@pytest.mark.slow
def test_compare_sentiment_engines_real_verdict_on_project_dataset():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from src.nlp.sentiment_analyzer import get_local_sentiment_dataset_path
    _load_indobert_sentiment_model.cache_clear()
    _load_embedding_model.cache_clear()

    dataset_path = get_local_sentiment_dataset_path()
    if not dataset_path.exists():
        pytest.skip("Dataset sentimen lokal proyek belum ada.")

    comparison = compare_sentiment_engines(dataset_path)

    assert comparison["verdict"] == "TIDAK BISA DIBANDINGKAN" or "LEBIH BAIK" in comparison["verdict"] or comparison["verdict"] == "SERI"
    for engine_key in ("tfidf_svm", "indobert_pretrained", "indo_e5_embedding_svm", "indobert_embedding_svm"):
        engine_result = comparison.get(engine_key)
        if engine_result is not None:
            print(f"\n[{engine_key}] akurasi={engine_result['accuracy_pct']}% f1_per_class={engine_result['f1_per_class']}")
    print(f"\n[VERDICT] {comparison['verdict']}: {comparison['verdict_reason']}")


def test_compare_sentiment_engines_repeated_aggregates_when_alternatives_unavailable(tmp_path, monkeypatch):
    dataset_path = tmp_path / "financial_news_clean.csv"
    _make_labeled_dataset(rows_per_label=15).to_csv(dataset_path, index=False)
    _load_indobert_sentiment_model.cache_clear()
    _load_embedding_model.cache_clear()
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoTokenizer", None)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModelForSequenceClassification", None)
    monkeypatch.setattr("src.nlp.sentiment_analyzer.AutoModel", None)

    result = compare_sentiment_engines_repeated(dataset_path, seeds=(1, 2, 3))

    assert result["seeds"] == [1, 2, 3]
    assert len(result["per_seed_results"]) == 3
    tfidf_summary = result["summary_by_engine"]["tfidf_svm"]
    assert tfidf_summary["n_seeds_evaluated"] == 3
    assert tfidf_summary["mean_accuracy_pct"] == pytest.approx(
        sum(tfidf_summary["accuracies_per_seed"]) / 3, abs=0.05
    )
    assert result["summary_by_engine"]["indobert_pretrained"] is None
    assert result["verdict"] == "TIDAK BISA DIBANDINGKAN"
    _load_indobert_sentiment_model.cache_clear()
    _load_embedding_model.cache_clear()


@pytest.mark.slow
def test_compare_sentiment_engines_repeated_real_verdict_on_project_dataset():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from src.nlp.sentiment_analyzer import get_local_sentiment_dataset_path
    _load_indobert_sentiment_model.cache_clear()
    _load_embedding_model.cache_clear()

    dataset_path = get_local_sentiment_dataset_path()
    if not dataset_path.exists():
        pytest.skip("Dataset sentimen lokal proyek belum ada.")

    result = compare_sentiment_engines_repeated(dataset_path, seeds=(1, 7, 42, 99, 123))

    for engine_key, summary in result["summary_by_engine"].items():
        if summary is not None:
            print(
                f"\n[{engine_key}] rata-rata={summary['mean_accuracy_pct']}% "
                f"stdev={summary['stdev_accuracy_pct']} per_seed={summary['accuracies_per_seed']}"
            )
    print(f"\n[VERDICT ROBUST] {result['verdict']}: {result['verdict_reason']}")

    assert "ROBUST" in result["verdict"] or result["verdict"] == "TIDAK BISA DIBANDINGKAN"


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


def test_compute_sentiment_drift_returns_none_when_missing_file(tmp_path):
    assert compute_sentiment_drift(tmp_path / "missing.csv") is None


def test_compute_sentiment_drift_returns_none_without_required_columns(tmp_path):
    path = tmp_path / "no_score.csv"
    pd.DataFrame([{"date": "2026-01-01", "text": "sesuatu"}]).to_csv(path, index=False)

    assert compute_sentiment_drift(path) is None


def test_compute_sentiment_drift_detects_no_drift_for_stable_sentiment(tmp_path):
    path = tmp_path / "financial_news_clean.csv"
    rows = []
    for i in range(20):
        rows.append({"date": f"2026-01-{i + 1:02d}", "sentiment_score": 0.1 + (0.01 if i % 2 == 0 else -0.01)})
    pd.DataFrame(rows).to_csv(path, index=False)

    result = compute_sentiment_drift(path, window_days=5)

    assert result is not None
    assert result["is_drifting"] is False
    assert abs(result["drift_z_score"]) < DRIFT_Z_THRESHOLD


def test_compute_sentiment_drift_detects_drift_for_sudden_shift(tmp_path):
    path = tmp_path / "financial_news_clean.csv"
    rows = []
    for i in range(20):
        rows.append({"date": f"2026-01-{i + 1:02d}", "sentiment_score": 0.05})
    for i in range(20, 25):
        rows.append({"date": f"2026-01-{i + 1:02d}", "sentiment_score": -0.9})
    pd.DataFrame(rows).to_csv(path, index=False)

    result = compute_sentiment_drift(path, window_days=5)

    assert result is not None
    assert result["is_drifting"] is True
    assert result["drift_z_score"] < 0
    assert len(result["daily_series"]) == 25
