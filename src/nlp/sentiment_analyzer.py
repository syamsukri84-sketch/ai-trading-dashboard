"""Utilities for capital-market sentiment analysis.

The first version is intentionally offline and lightweight. It scores issue/news
texts with a finance-oriented lexicon so the trading dashboard can use sentiment
without depending on an external API or a large transformer model.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


POSITIVE_TERMS = {
    "akumulasi", "bertumbuh", "bullish", "buyback", "dividen", "ekspansi",
    "kenaikan", "laba", "menguat", "naik", "optimis", "pemulihan",
    "peningkatan", "positif", "profit", "rebound", "rekor", "stabil",
    "surplus", "tumbuh", "upgrade", "growth", "strong", "positive",
}

NEGATIVE_TERMS = {
    "anjlok", "bearish", "beban", "defisit", "gagal", "koreksi", "krisis",
    "lemah", "melemah", "merugi", "negatif", "penurunan", "rugi", "risiko",
    "sanksi", "tekanan", "turun", "volatil", "downgrade", "loss", "negative",
    "weak", "risk",
}

NEGATION_TERMS = {"tidak", "bukan", "belum", "kurang", "tanpa", "not", "no"}
INTENSIFIER_TERMS = {"sangat", "signifikan", "tajam", "kuat", "besar", "very", "strong"}


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    sentiment_score: float
    positive_hits: int
    negative_hits: int
    method: str = "lexicon"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", str(text).lower())


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_local_sentiment_dataset_path() -> Path:
    return _project_root() / "data" / "sentiment" / "processed" / "financial_news_clean.csv"


def get_seed_sentiment_examples_path() -> Path:
    return _project_root() / "data" / "sentiment" / "seed_examples.csv"


def get_practicum_sentiment_dataset_path() -> Path:
    return _project_root().parent / "tugas_nlp_ai_trading" / "data" / "processed" / "financial_news_clean.csv"


def _default_external_dataset_path() -> Path:
    configured = os.environ.get("AI_TRADING_SENTIMENT_DATASET", "").strip()
    if configured:
        return Path(configured)
    local_dataset = get_local_sentiment_dataset_path()
    if local_dataset.exists():
        return local_dataset
    return get_practicum_sentiment_dataset_path()


def _train_sentiment_model_from_dataset(dataset_path: Path) -> Pipeline | None:
    if not dataset_path.exists():
        return None

    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return None

    if "clean_text" not in df.columns or "label" not in df.columns:
        return None

    train_df = df[["clean_text", "label"]].dropna().copy()
    train_df["clean_text"] = train_df["clean_text"].astype(str).map(clean_text)
    train_df["label"] = train_df["label"].astype(str).str.upper().str.strip()
    train_df = train_df[
        train_df["clean_text"].str.len().gt(0)
        & train_df["label"].isin(["POSITIVE", "NEUTRAL", "NEGATIVE"])
    ]
    if len(train_df) < 30 or train_df["label"].nunique() < 2:
        return None

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]
    )
    model.fit(train_df["clean_text"], train_df["label"])
    return model


@lru_cache(maxsize=1)
def _load_financial_sentiment_model() -> Pipeline | None:
    """Loads the selected dataset and trains a lightweight local classifier.

    The AI Trading app keeps this optional: if the selected dataset is missing
    or unreadable, the lexicon analyzer remains the fallback so the dashboard
    still works offline.
    """
    return _train_sentiment_model_from_dataset(_default_external_dataset_path())


def get_sentiment_engine_status() -> dict:
    """Returns dashboard-friendly metadata for the active sentiment engine."""
    dataset_path = _default_external_dataset_path()
    model = _load_financial_sentiment_model()
    if model is None:
        return {
            "engine": "lexicon",
            "label": "Lexicon fallback",
            "dataset_path": str(dataset_path),
            "dataset_available": dataset_path.exists(),
            "model_available": False,
            "description": "Memakai kamus sentimen finansial lokal karena dataset/model NLP tugas belum tersedia.",
        }

    try:
        df = pd.read_csv(dataset_path, usecols=["clean_text", "label"])
        usable_rows = int(df.dropna(subset=["clean_text", "label"]).shape[0])
    except Exception:
        usable_rows = None

    return {
        "engine": "ml_tfidf_linear_svm",
        "label": "TF-IDF + Linear SVM",
        "dataset_path": str(dataset_path),
        "dataset_available": True,
        "model_available": True,
        "training_rows": usable_rows,
        "description": "Memakai pipeline tugas_nlp_ai_trading: preprocessing teks, TF-IDF ngram 1-2, dan Linear SVM.",
    }


def _term_weight(tokens: list[str], index: int) -> float:
    window = tokens[max(0, index - 2):index]
    weight = 1.0
    if any(term in window for term in INTENSIFIER_TERMS):
        weight += 0.5
    if any(term in window for term in NEGATION_TERMS):
        weight *= -1
    return weight


def _analyze_text_lexicon(text: str) -> SentimentResult:
    tokens = tokenize(text)
    raw_score = 0.0
    positive_hits = 0
    negative_hits = 0

    for index, token in enumerate(tokens):
        weight = _term_weight(tokens, index)
        if token in POSITIVE_TERMS:
            raw_score += weight
            positive_hits += 1
        elif token in NEGATIVE_TERMS:
            raw_score -= weight
            negative_hits += 1

    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        return SentimentResult("NEUTRAL", 0.5, 0.0, 0, 0)

    normalized = max(-1.0, min(1.0, raw_score / total_hits))
    confidence = min(0.99, 0.5 + abs(normalized) / 2)

    if normalized > 0.15:
        label = "POSITIVE"
    elif normalized < -0.15:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return SentimentResult(label, round(confidence, 4), round(normalized, 4), positive_hits, negative_hits)


def _analyze_text_ml(text: str) -> SentimentResult | None:
    model = _load_financial_sentiment_model()
    cleaned = clean_text(text)
    if model is None or not cleaned:
        return None

    label = str(model.predict([cleaned])[0]).upper()
    sentiment_score = {
        "POSITIVE": 0.65,
        "NEUTRAL": 0.0,
        "NEGATIVE": -0.65,
    }.get(label, 0.0)

    confidence = 0.75
    try:
        classifier = model.named_steps["classifier"]
        classes = list(classifier.classes_)
        distances = model.decision_function([cleaned])[0]
        if hasattr(distances, "__iter__"):
            distance = float(distances[classes.index(label)])
        else:
            distance = abs(float(distances))
        confidence = min(0.99, 0.55 + min(abs(distance), 3.0) / 6.0)
    except Exception:
        pass

    return SentimentResult(label, round(confidence, 4), sentiment_score, 0, 0, "ml_tfidf_linear_svm")


def _predict_with_model(model: Pipeline, text: str, method: str) -> SentimentResult:
    cleaned = clean_text(text)
    label = str(model.predict([cleaned])[0]).upper()
    sentiment_score = {
        "POSITIVE": 0.65,
        "NEUTRAL": 0.0,
        "NEGATIVE": -0.65,
    }.get(label, 0.0)
    confidence = 0.75
    try:
        classifier = model.named_steps["classifier"]
        classes = list(classifier.classes_)
        distances = model.decision_function([cleaned])[0]
        if hasattr(distances, "__iter__"):
            distance = float(distances[classes.index(label)])
        else:
            distance = abs(float(distances))
        confidence = min(0.99, 0.55 + min(abs(distance), 3.0) / 6.0)
    except Exception:
        pass
    return SentimentResult(label, round(confidence, 4), sentiment_score, 0, 0, method)


def analyze_text(text: str) -> SentimentResult:
    ml_result = _analyze_text_ml(text)
    if ml_result is not None:
        return ml_result
    return _analyze_text_lexicon(text)


def build_local_sentiment_dataset(
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
    text_column: str = "text",
    include_seed_examples: bool = True,
) -> pd.DataFrame:
    """Builds a local AI Trading sentiment dataset from archived issue/news rows.

    The output follows the practicum dataset shape while retaining useful
    trading context columns. Labels are bootstrapped from the active analyzer,
    so users can later correct the CSV manually if needed.
    """
    source = Path(source_path) if source_path is not None else _project_root() / "data" / "sentiment" / "market_issues.csv"
    output = Path(output_path) if output_path is not None else get_local_sentiment_dataset_path()

    if not source.exists():
        raise FileNotFoundError(f"File sumber dataset sentimen tidak ditemukan: {source}")

    raw_df = pd.read_csv(source)
    if text_column not in raw_df.columns:
        raise ValueError(f"Kolom '{text_column}' tidak ditemukan pada file sumber.")
    raw_df["dataset_source"] = "market_issues"

    seed_path = get_seed_sentiment_examples_path()
    if include_seed_examples and seed_path.exists():
        seed_df = pd.read_csv(seed_path)
        if text_column not in seed_df.columns:
            raise ValueError(f"Kolom '{text_column}' tidak ditemukan pada seed examples.")
        seed_df["dataset_source"] = "seed_examples"
        raw_df = pd.concat([raw_df, seed_df], ignore_index=True, sort=False)

    bootstrap_model = _train_sentiment_model_from_dataset(get_practicum_sentiment_dataset_path())
    valid_labels = {"positive", "neutral", "negative"}

    rows = []
    for _, row in raw_df.fillna("").iterrows():
        text = str(row.get(text_column, "")).strip()
        cleaned = clean_text(text)
        if not cleaned:
            continue
        manual_label = str(row.get("manual_label", "")).lower().strip()
        if manual_label in valid_labels:
            result = SentimentResult(
                manual_label.upper(),
                1.0,
                {"positive": 0.65, "neutral": 0.0, "negative": -0.65}[manual_label],
                0,
                0,
                "manual_seed_label",
            )
        elif bootstrap_model is not None:
            result = _predict_with_model(bootstrap_model, text, "bootstrap_tugas_nlp_ai_trading")
        else:
            result = _analyze_text_lexicon(text)
        rows.append(
            {
                "text": text,
                "label": result.label.lower(),
                "clean_text": cleaned,
                "source": str(row.get("source", "ai_trading_local")).strip() or "ai_trading_local",
                "ticker": str(row.get("ticker", "")).upper().strip(),
                "date": str(row.get("date", "")).strip(),
                "label_method": result.method,
                "label_confidence": result.score,
                "sentiment_score": result.sentiment_score,
                "dataset_source": str(row.get("dataset_source", "market_issues")).strip(),
            }
        )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        dataset = pd.DataFrame(
            columns=[
                "text",
                "label",
                "clean_text",
                "source",
                "ticker",
                "date",
                "label_method",
                "label_confidence",
                "sentiment_score",
                "dataset_source",
            ]
        )
    else:
        dataset = dataset.drop_duplicates(subset=["clean_text", "label"], keep="last")
        label_order = {"negative": 0, "neutral": 1, "positive": 2}
        dataset["_label_order"] = dataset["label"].map(label_order).fillna(9)
        dataset = dataset.sort_values(["_label_order", "ticker", "date", "clean_text"]).drop(columns=["_label_order"])

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    _load_financial_sentiment_model.cache_clear()
    return dataset


def analyze_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Kolom '{text_column}' tidak ditemukan pada data sentimen.")

    rows = []
    for text in df[text_column].fillna(""):
        result = analyze_text(text)
        rows.append(result.__dict__)

    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if "ticker" in scored.columns:
        scored["ticker"] = scored["ticker"].astype(str).str.upper().str.strip()
    return scored


def load_issues(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker", "source", "text"])
    return pd.read_csv(path)


def append_issue(path: str | Path, date: str, ticker: str, source: str, text: str) -> pd.DataFrame:
    """Adds one issue/news item to the local sentiment CSV and returns the updated data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([{
        "date": date,
        "ticker": str(ticker).upper().strip(),
        "source": source,
        "text": text,
    }])
    existing = load_issues(path)
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(path, index=False)
    return updated


def append_issues(path: str | Path, rows: Iterable[dict]) -> pd.DataFrame:
    """Adds multiple issue/news rows and removes exact duplicates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = pd.DataFrame(list(rows), columns=["date", "ticker", "source", "text"])
    if new_rows.empty:
        return load_issues(path)

    new_rows["ticker"] = new_rows["ticker"].astype(str).str.upper().str.strip()
    existing = load_issues(path)
    updated = pd.concat([existing, new_rows], ignore_index=True)
    updated = updated.drop_duplicates(subset=["date", "ticker", "source", "text"], keep="last")
    updated.to_csv(path, index=False)
    return updated


def summarize_by_ticker(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty or "ticker" not in scored_df.columns:
        return pd.DataFrame(columns=["ticker", "avg_sentiment_score", "news_count", "dominant_label"])

    summary = scored_df.groupby("ticker").agg(
        avg_sentiment_score=("sentiment_score", "mean"),
        news_count=("text", "count"),
        dominant_label=("label", lambda values: values.value_counts().index[0]),
    ).reset_index()
    summary["avg_sentiment_score"] = summary["avg_sentiment_score"].round(4)
    return summary.sort_values("avg_sentiment_score", ascending=False)


def build_trading_sentiment_summary(scored_df: pd.DataFrame, ticker: str | None = None) -> dict:
    """Builds a concise trading-oriented summary from scored sentiment rows.

    The text classifier keeps the literal news tone: positive news gets a
    positive sentiment score and negative news gets a negative score. For the
    Indonesia trading summary, the market signal is interpreted contrarian:
    very positive news can become a sell-the-news warning, while very negative
    news can mark capitulation/rebound potential.
    """
    if scored_df.empty:
        return {
            "ticker": ticker or "SEMUA",
            "bias": "NEUTRAL",
            "risk_level": "MEDIUM",
            "confidence": 0.0,
            "avg_sentiment_score": 0.0,
            "market_signal_score": 0.0,
            "interpretation_mode": "contrarian_indonesia",
            "news_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "conclusion": "Belum ada berita/isu yang bisa diringkas.",
            "trading_note": "Tunggu data sentimen tersedia sebelum memakai sentimen sebagai faktor keputusan.",
            "key_drivers": [],
        }

    view = scored_df.copy()
    selected_ticker = (ticker or "SEMUA").upper().strip()
    if ticker and selected_ticker != "SEMUA" and "ticker" in view.columns:
        view = view[view["ticker"].astype(str).str.upper().str.strip() == selected_ticker]

    if view.empty:
        return build_trading_sentiment_summary(pd.DataFrame(), selected_ticker)

    avg_score = float(view["sentiment_score"].mean())
    market_signal_score = -avg_score
    positive_count = int((view["label"] == "POSITIVE").sum())
    negative_count = int((view["label"] == "NEGATIVE").sum())
    neutral_count = int((view["label"] == "NEUTRAL").sum())
    news_count = int(len(view))
    confidence = min(0.99, abs(avg_score) * 0.7 + min(news_count, 10) / 30)

    if avg_score >= 0.35 and positive_count >= negative_count:
        bias = "BEARISH"
        risk_level = "HIGH" if negative_count == 0 else "MEDIUM"
        conclusion = "Berita sangat positif dibaca kontrarian: risiko sell-the-news atau distribusi meningkat."
        trading_note = "Jangan entry agresif hanya karena berita positif; tunggu pullback, volume sehat, dan konfirmasi harga."
    elif avg_score <= -0.35 and negative_count >= positive_count:
        bias = "BULLISH"
        risk_level = "MEDIUM"
        conclusion = "Berita negatif dibaca kontrarian: potensi capitulation dan rebound mulai menarik."
        trading_note = "Cari konfirmasi rebound dan akumulasi; tetap batasi risiko karena berita negatif bisa berlanjut."
    elif negative_count > positive_count and avg_score < 0:
        bias = "MILDLY BULLISH"
        risk_level = "MEDIUM"
        conclusion = "Tekanan berita negatif mulai dominan, tetapi belum cukup ekstrem untuk sinyal bullish kuat."
        trading_note = "Gunakan sentimen sebagai watchlist rebound, bukan alasan entry tanpa konfirmasi teknikal."
    elif positive_count > negative_count and avg_score > 0:
        bias = "CAUTIOUS"
        risk_level = "MEDIUM"
        conclusion = "Berita agak positif, tetapi secara kontrarian belum memberi margin of safety yang menarik."
        trading_note = "Waspadai euforia berita; prioritaskan saham yang masih punya risk-reward dan reliability kuat."
    else:
        bias = "NEUTRAL"
        risk_level = "MEDIUM"
        conclusion = "Sentimen relatif netral atau belum menunjukkan arah yang jelas."
        trading_note = "Prioritaskan sinyal teknikal, likuiditas, dan manajemen risiko."

    key_drivers = []
    if "text" in view.columns:
        ranked = view.assign(abs_score=view["sentiment_score"].abs()).sort_values("abs_score", ascending=False)
        for _, row in ranked.head(3).iterrows():
            text = re.sub(r"\s+", " ", str(row.get("text", ""))).strip()
            if len(text) > 180:
                text = text[:177].rstrip() + "..."
            row_ticker = str(row.get("ticker", selected_ticker)).upper().strip()
            key_drivers.append(f"{row_ticker} | {row.get('label', 'NEUTRAL')}: {text}")

    return {
        "ticker": selected_ticker,
        "bias": bias,
        "risk_level": risk_level,
        "confidence": round(confidence, 4),
        "avg_sentiment_score": round(avg_score, 4),
        "market_signal_score": round(market_signal_score, 4),
        "interpretation_mode": "contrarian_indonesia",
        "news_count": news_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "conclusion": conclusion,
        "trading_note": trading_note,
        "key_drivers": key_drivers,
    }


def interpret_signal(sentiment_score: float) -> str:
    if sentiment_score >= 0.35:
        return "Mode kontrarian Indonesia: berita positif menjadi peringatan risiko sell-the-news/bearish."
    if sentiment_score <= -0.35:
        return "Mode kontrarian Indonesia: berita negatif membuka peluang capitulation/rebound bullish."
    return "Sentimen relatif netral atau campuran."
