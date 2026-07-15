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

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.utils.atomic_io import atomic_write_csv

try:
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    torch = None
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

SENTIMENT_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

# Model embedding beku (frozen feature extractor, encoder BIASA lewat
# AutoModel -- BUKAN classifier siap pakai) untuk pendekatan "embedding +
# classifier linear dilatih ulang di data lokal". Dipilih berdasarkan riset
# 2026-07 (lihat memori/roadmap terkait): kandidat #1 adalah model
# sentence-embedding Indonesia paling banyak diadopsi yang ditemukan; kandidat
# #2 adalah encoder dasar IndoBERT (Indo4B corpus mencakup berita, bukan cuma
# medsos) sebagai pembanding. KEDUANYA BUKAN classifier siap pakai -- classifier
# tetap dilatih dari nol di 215 baris dataset lokal, cuma representasi fiturnya
# yang beda dari TF-IDF.
EMBEDDING_MODEL_IDS = {
    "indo_e5_embedding_svm": "LazarusNLP/all-indo-e5-small-v4",
    "indobert_embedding_svm": "indobenchmark/indobert-base-p1",
}
ENGINE_DISPLAY_NAMES = {
    "tfidf_svm": "TF-IDF+SVM",
    "indobert_pretrained": "IndoBERT Classifier Pretrained",
    "indo_e5_embedding_svm": "Embedding Indo-E5+SVM",
    "indobert_embedding_svm": "Embedding IndoBERT+SVM",
}

# Model IndoBERT PRETRAINED (sudah di-fine-tune untuk sentimen oleh pihak lain
# di dataset publik "Prosa" dari benchmark IndoNLU -- BUKAN dilatih ulang dari
# nol di sini, karena dataset lokal AI Trading cuma ~215 baris, terlalu kecil
# untuk fine-tune transformer tanpa overfitting parah). Pemetaan label
# diverifikasi manual dari model card HuggingFace (2026-07): config.json model
# ini cuma expose "LABEL_0"/"LABEL_1"/"LABEL_2" generik, makna semantiknya TIDAK
# ada di config, jadi peta di bawah ini WAJIB dicocokkan ulang kalau model_id
# diganti ke model lain.
INDOBERT_SENTIMENT_MODEL_ID = "mdhugol/indonesia-bert-sentiment-classification"
INDOBERT_LABEL_MAP = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}


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
    """Dataset bootstrap (1320 baris, dari repo terpisah `tugas_nlp_ai_trading`)
    dipakai untuk melatih model sentiment ketika dataset lokal proyek ini
    (`get_local_sentiment_dataset_path`, ~216 baris) belum cukup.

    Sebelum ini path menunjuk LANGSUNG ke `../tugas_nlp_ai_trading/...` di
    luar proyek -- dependency ke repo sibling yang membuat `build_local_
    sentiment_dataset()` (dipanggil OTOMATIS tiap hari lewat Task Scheduler)
    diam-diam turun kualitas ke lexicon sederhana tanpa peringatan kalau
    dijalankan di mesin lain / deploy cloud yang tidak punya folder itu.
    Sekarang divendor sebagai snapshot ke dalam proyek ini supaya workflow
    harian self-contained. Snapshot ini TIDAK otomatis sinkron dengan
    perubahan di repo sibling -- kalau dataset practicum diperbarui, copy
    ulang manual ke `data/sentiment/external/financial_news_clean.csv`.
    Lihat audit codebase 2026-07-12.
    """
    return _project_root() / "data" / "sentiment" / "external" / "financial_news_clean.csv"


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
        & train_df["label"].isin(SENTIMENT_LABELS)
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


def _clean_labeled_dataset(dataset_path: Path) -> pd.DataFrame | None:
    if not dataset_path.exists():
        return None
    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return None
    if "clean_text" not in df.columns or "label" not in df.columns:
        return None

    data = df[["clean_text", "label"]].dropna().copy()
    data["clean_text"] = data["clean_text"].astype(str).map(clean_text)
    data["label"] = data["label"].astype(str).str.upper().str.strip()
    data = data[data["clean_text"].str.len().gt(0) & data["label"].isin(SENTIMENT_LABELS)]
    return data


def _split_labeled_dataset(
    dataset_path: Path, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Split 75/25 (stratified) dipakai bersama oleh SEMUA evaluator engine
    (TF-IDF+SVM, IndoBERT, embedding+SVM, dst.) supaya perbandingan antar
    engine adil -- diuji di test set yang identik, bukan sampel berbeda yang
    bisa membuat satu engine kelihatan lebih baik cuma karena kebetulan.

    `random_state` bisa divariasikan lewat `compare_sentiment_engines_repeated`
    -- PENTING: test set held-out proyek ini cuma ~54 baris, jadi verdict dari
    SATU split (random_state tetap) bisa menyesatkan murni karena noise
    statistik. Dibuktikan langsung 2026-07: pada random_state=42, embedding
    IndoBERT+SVM sempat 'menang' (68.5% vs 66.7%), tapi di 4 dari 5 seed lain
    TF-IDF+SVM menang dengan margin lebih besar (rata-rata 73.3% vs 70.7%).
    Jangan mengambil keputusan produksi dari satu split saja.
    """
    data = _clean_labeled_dataset(dataset_path)
    if data is None:
        return None, None
    label_counts = data["label"].value_counts()
    if len(data) < 30 or data["label"].nunique() < 2 or label_counts.min() < 2:
        return None, None

    try:
        train_df, test_df = train_test_split(
            data, test_size=0.25, stratify=data["label"], random_state=random_state
        )
    except ValueError:
        return None, None
    if test_df.empty or train_df["label"].nunique() < 2:
        return None, None
    return train_df, test_df


def _classification_metrics(y_true, y_pred) -> dict:
    f1_scores = f1_score(y_true, y_pred, labels=SENTIMENT_LABELS, average=None, zero_division=0)
    return {
        "accuracy_pct": round(float(accuracy_score(y_true, y_pred)) * 100.0, 1),
        "f1_per_class": {
            label: round(float(score), 3) for label, score in zip(SENTIMENT_LABELS, f1_scores)
        },
    }


def _evaluate_sentiment_model(dataset_path: Path, random_state: int = 42) -> dict | None:
    """Held-out train/test evaluation model TF-IDF+SVM produksi.

    Dilatih ulang di split 75/25 (terpisah dari model produksi yang dilatih
    di 100% data) supaya ini tetap pengujian out-of-sample yang jujur, bukan
    akurasi latihan yang dilaporkan seolah performa asli. Mengembalikan None
    kalau dataset terlalu kecil/timpang untuk split yang andal (mis. satu
    kelas cuma <2 baris), bukan memaksakan angka.
    """
    train_df, test_df = _split_labeled_dataset(dataset_path, random_state=random_state)
    if train_df is None:
        return None

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]
    )
    model.fit(train_df["clean_text"], train_df["label"])
    y_pred = model.predict(test_df["clean_text"])
    y_true = test_df["label"]

    metrics = _classification_metrics(y_true, y_pred)
    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **metrics,
        "test_label_counts": {str(k): int(v) for k, v in test_df["label"].value_counts().to_dict().items()},
    }


def _evaluate_indobert_model(dataset_path: Path, random_state: int = 42) -> dict | None:
    """Held-out evaluation IndoBERT PRETRAINED pada test set yang SAMA PERSIS
    dengan `_evaluate_sentiment_model` (split sama, random_state sama) --
    supaya perbandingan adil.

    IndoBERT TIDAK dilatih ulang di sini (sengaja -- lihat catatan di
    INDOBERT_SENTIMENT_MODEL_ID) -- train_df dipakai HANYA untuk menyamakan
    proporsi split dengan evaluator TF-IDF+SVM, prediksi dijalankan langsung
    dari bobot pretrained di test_df. Mengembalikan None kalau dataset tidak
    cukup untuk split, ATAU kalau model IndoBERT sendiri tidak tersedia
    (transformers belum terinstal / gagal dimuat/diunduh).
    """
    train_df, test_df = _split_labeled_dataset(dataset_path, random_state=random_state)
    if train_df is None:
        return None
    if _load_indobert_sentiment_model() is None:
        return None

    predictions = []
    skipped = 0
    for text in test_df["clean_text"]:
        result = _predict_with_indobert(text)
        if result is None:
            skipped += 1
            predictions.append("NEUTRAL")
            continue
        predictions.append(result.label)

    y_true = test_df["label"]
    metrics = _classification_metrics(y_true, predictions)
    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **metrics,
        "test_label_counts": {str(k): int(v) for k, v in test_df["label"].value_counts().to_dict().items()},
        "skipped_predictions": skipped,
    }


@lru_cache(maxsize=4)
def _load_embedding_model(model_id: str):
    """Memuat encoder BIASA (AutoModel, BUKAN ForSequenceClassification) untuk
    dipakai sebagai ekstraktor embedding beku -- classifier tetap dilatih dari
    nol di data lokal, bukan memakai classifier head siap pakai (yang sudah
    terbukti gagal karena domain-mismatch di kasus IndoBERT pretrained
    classifier). Mengembalikan None kalau library tidak ada atau model gagal
    dimuat/diunduh."""
    if AutoTokenizer is None or AutoModel is None:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()
    except Exception:
        return None
    return tokenizer, model


def _mean_pool_embeddings(tokenizer, model, texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Attention-mask-aware mean pooling atas last_hidden_state -- terbukti
    lebih baik dari [CLS] pooling untuk representasi kalimat pendek (Reimers &
    Gurevych, Sentence-BERT 2019; diverifikasi ulang lewat riset 2026-07 untuk
    proyek ini). Diproses per-batch supaya tidak membebani memori kalau
    dataset bertambah besar di kemudian hari."""
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            token_embeddings = model(**inputs).last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        all_embeddings.append((summed / counts).numpy())
    return np.concatenate(all_embeddings, axis=0)


def _evaluate_embedding_classifier_model(dataset_path: Path, model_id: str, random_state: int = 42) -> dict | None:
    """Held-out evaluation: ekstrak embedding BEKU dari `model_id` (encoder
    tidak dilatih ulang), lalu latih LinearSVC dari NOL di embedding itu --
    di test set yang SAMA PERSIS dengan evaluator lain. Kekuatan regularisasi
    (C) dituning lewat StratifiedKFold HANYA di train set (tidak pernah
    melihat test set), sesuai rekomendasi riset supaya rezim dimensi-tinggi
    (768) vs sampel sedikit (~160) tidak overfit begitu saja.
    """
    train_df, test_df = _split_labeled_dataset(dataset_path, random_state=random_state)
    if train_df is None:
        return None
    loaded = _load_embedding_model(model_id)
    if loaded is None:
        return None
    tokenizer, model = loaded

    try:
        X_train_raw = _mean_pool_embeddings(tokenizer, model, train_df["clean_text"].tolist())
        X_test_raw = _mean_pool_embeddings(tokenizer, model, test_df["clean_text"].tolist())
    except Exception:
        return None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    y_train = train_df["label"]
    y_test = test_df["label"]

    n_splits = min(5, int(y_train.value_counts().min()))
    if n_splits < 2:
        classifier = LinearSVC(class_weight="balanced", C=1.0, max_iter=20000)
        classifier.fit(X_train, y_train)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        search = GridSearchCV(
            LinearSVC(class_weight="balanced", max_iter=20000),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
            cv=cv,
            scoring="f1_macro",
        )
        search.fit(X_train, y_train)
        classifier = search.best_estimator_

    y_pred = classifier.predict(X_test)
    metrics = _classification_metrics(y_test, y_pred)
    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **metrics,
        "test_label_counts": {str(k): int(v) for k, v in test_df["label"].value_counts().to_dict().items()},
        "model_id": model_id,
    }


def compare_sentiment_engines(dataset_path: Path, random_state: int = 42) -> dict:
    """Membandingkan TF-IDF+SVM (produksi) vs 3 alternatif berbasis model
    bahasa pretrained di SATU split held-out (random_state tetap) -- cepat,
    tapi lihat peringatan di `_split_labeled_dataset`: test set cuma ~54
    baris, jadi hasil SATU split ini BISA menyesatkan murni karena noise
    statistik. Untuk keputusan produksi, WAJIB pakai
    `compare_sentiment_engines_repeated` (multi-seed), bukan fungsi ini saja.
    """
    tfidf_result = _evaluate_sentiment_model(dataset_path, random_state=random_state)
    alternatives = {
        "indobert_pretrained": _evaluate_indobert_model(dataset_path, random_state=random_state),
        "indo_e5_embedding_svm": _evaluate_embedding_classifier_model(dataset_path, EMBEDDING_MODEL_IDS["indo_e5_embedding_svm"], random_state=random_state),
        "indobert_embedding_svm": _evaluate_embedding_classifier_model(dataset_path, EMBEDDING_MODEL_IDS["indobert_embedding_svm"], random_state=random_state),
    }
    available_alternatives = {name: r for name, r in alternatives.items() if r is not None}

    if tfidf_result is None:
        verdict = "TIDAK BISA DIBANDINGKAN"
        reason = "Baseline TF-IDF+SVM sendiri gagal dievaluasi (dataset terlalu kecil/tidak seimbang)."
    elif not available_alternatives:
        verdict = "TIDAK BISA DIBANDINGKAN"
        reason = "Tidak ada engine alternatif yang berhasil dievaluasi (dependency belum terinstal, atau model gagal dimuat/diunduh)."
    else:
        tfidf_acc = tfidf_result["accuracy_pct"]
        best_name = max(available_alternatives, key=lambda name: available_alternatives[name]["accuracy_pct"])
        best_acc = available_alternatives[best_name]["accuracy_pct"]
        best_label = ENGINE_DISPLAY_NAMES.get(best_name, best_name)
        if best_acc > tfidf_acc:
            verdict = f"{best_label.upper()} LEBIH BAIK"
            reason = f"{best_label} ({best_acc:.1f}%) mengalahkan TF-IDF+SVM ({tfidf_acc:.1f}%) di test set yang sama."
        elif best_acc == tfidf_acc:
            verdict = "SERI"
            reason = f"Akurasi terbaik alternatif ({best_label}) sama dengan TF-IDF+SVM ({tfidf_acc:.1f}%)."
        else:
            verdict = "TF-IDF+SVM LEBIH BAIK"
            reason = f"TF-IDF+SVM ({tfidf_acc:.1f}%) masih mengungguli semua alternatif yang diuji (terbaik: {best_label} {best_acc:.1f}%)."

    return {
        "tfidf_svm": tfidf_result,
        "indobert_pretrained": alternatives["indobert_pretrained"],
        "indo_e5_embedding_svm": alternatives["indo_e5_embedding_svm"],
        "indobert_embedding_svm": alternatives["indobert_embedding_svm"],
        "verdict": verdict,
        "verdict_reason": reason,
    }


def compare_sentiment_engines_repeated(dataset_path: Path, seeds: tuple[int, ...] = (1, 7, 42, 99, 123)) -> dict:
    """Perbandingan ROBUST: ulangi `compare_sentiment_engines` di beberapa
    random_state berbeda (bukan satu split tetap), lalu laporkan rata-rata +
    win-rate per engine.

    WAJIB dipakai sebelum memutuskan mengganti engine produksi -- BUKAN
    `compare_sentiment_engines` dengan satu split saja. Dibuktikan langsung
    2026-07: pada random_state=42 saja, embedding IndoBERT+SVM sempat
    "menang" (68.5% vs 66.7%) dan sepintas terlihat seperti temuan positif
    genuine -- tapi setelah diuji ulang di 5 seed berbeda, TF-IDF+SVM
    ternyata menang di 4/5 split dengan rata-rata akurasi lebih tinggi
    (73.3% vs 70.7%). Verdict dari satu split kecil (~54 baris test) TIDAK
    cukup andal untuk keputusan produksi.
    """
    per_seed_results = [compare_sentiment_engines(dataset_path, random_state=seed) for seed in seeds]
    engine_keys = ["tfidf_svm", "indobert_pretrained", "indo_e5_embedding_svm", "indobert_embedding_svm"]

    per_engine_accuracies: dict[str, list[float]] = {key: [] for key in engine_keys}
    for result in per_seed_results:
        for key in engine_keys:
            if result[key] is not None:
                per_engine_accuracies[key].append(result[key]["accuracy_pct"])

    summary = {}
    for key, accs in per_engine_accuracies.items():
        if not accs:
            summary[key] = None
            continue
        mean_acc = sum(accs) / len(accs)
        variance = sum((a - mean_acc) ** 2 for a in accs) / len(accs)
        summary[key] = {
            "mean_accuracy_pct": round(mean_acc, 1),
            "stdev_accuracy_pct": round(variance ** 0.5, 1),
            "n_seeds_evaluated": len(accs),
            "accuracies_per_seed": accs,
        }

    tfidf_summary = summary.get("tfidf_svm")
    alt_summaries = {k: v for k, v in summary.items() if k != "tfidf_svm" and v is not None}

    if tfidf_summary is None:
        verdict = "TIDAK BISA DIBANDINGKAN"
        reason = "Baseline TF-IDF+SVM gagal dievaluasi di seed manapun."
    elif not alt_summaries:
        verdict = "TIDAK BISA DIBANDINGKAN"
        reason = "Tidak ada engine alternatif yang berhasil dievaluasi di seed manapun."
    else:
        best_name = max(alt_summaries, key=lambda name: alt_summaries[name]["mean_accuracy_pct"])
        best_mean = alt_summaries[best_name]["mean_accuracy_pct"]
        best_label = ENGINE_DISPLAY_NAMES.get(best_name, best_name)
        tfidf_mean = tfidf_summary["mean_accuracy_pct"]
        wins = sum(
            1
            for result in per_seed_results
            if result[best_name] is not None
            and result["tfidf_svm"] is not None
            and result[best_name]["accuracy_pct"] > result["tfidf_svm"]["accuracy_pct"]
        )
        n_seeds = len(seeds)
        if best_mean > tfidf_mean and wins > n_seeds / 2:
            verdict = f"{best_label.upper()} LEBIH BAIK (ROBUST)"
            reason = (
                f"{best_label} menang di {wins}/{n_seeds} split dengan rata-rata akurasi lebih tinggi "
                f"({best_mean:.1f}% vs {tfidf_mean:.1f}%) -- konsisten di berbagai split, bukan kebetulan satu seed."
            )
        else:
            verdict = "TF-IDF+SVM TETAP TERBAIK (ROBUST)"
            reason = (
                f"TF-IDF+SVM rata-rata {tfidf_mean:.1f}% vs alternatif terbaik ({best_label}) {best_mean:.1f}% "
                f"-- {best_label} cuma menang di {wins}/{n_seeds} split, tidak cukup konsisten untuk mengganti produksi."
            )

    return {
        "seeds": list(seeds),
        "per_seed_results": per_seed_results,
        "summary_by_engine": summary,
        "verdict": verdict,
        "verdict_reason": reason,
    }


@lru_cache(maxsize=1)
def _load_financial_sentiment_model() -> Pipeline | None:
    """Loads the selected dataset and trains a lightweight local classifier.

    The AI Trading app keeps this optional: if the selected dataset is missing
    or unreadable, the lexicon analyzer remains the fallback so the dashboard
    still works offline.
    """
    return _train_sentiment_model_from_dataset(_default_external_dataset_path())


@lru_cache(maxsize=1)
def _load_sentiment_model_evaluation() -> dict | None:
    return _evaluate_sentiment_model(_default_external_dataset_path())


@lru_cache(maxsize=1)
def _load_indobert_sentiment_model():
    """Memuat model IndoBERT PRETRAINED (lihat catatan di INDOBERT_SENTIMENT_MODEL_ID
    -- sudah di-fine-tune untuk sentimen oleh pihak lain, tidak dilatih ulang
    di sini). Dimuat sekali & di-cache karena model transformer relatif berat
    untuk dimuat berulang di tiap panggilan.

    Mengembalikan None (bukan raise) kalau `torch`/`transformers` belum
    terinstal, atau model gagal diunduh/dimuat (mis. offline tanpa cache HF
    lokal) -- caller WAJIB fallback ke TF-IDF+SVM/lexicon, sama seperti pola
    fallback _load_financial_sentiment_model yang sudah ada. Belum divalidasi
    lewat held-out test apakah model ini genuinely lebih baik untuk teks
    berita finansial -- lihat catatan di ROADMAP terkait sebelum menjadikan
    ini engine utama.
    """
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(INDOBERT_SENTIMENT_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(INDOBERT_SENTIMENT_MODEL_ID)
        model.eval()
    except Exception:
        return None
    return tokenizer, model


def _predict_with_indobert(text: str) -> SentimentResult | None:
    """Prediksi sentimen pakai IndoBERT pretrained. Mengembalikan None (bukan
    hasil netral palsu) kalau model tidak tersedia atau teks kosong setelah
    dibersihkan -- caller yang memutuskan fallback ke engine lain."""
    loaded = _load_indobert_sentiment_model()
    cleaned = clean_text(text)
    if loaded is None or not cleaned:
        return None
    tokenizer, model = loaded
    try:
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax())
        label = INDOBERT_LABEL_MAP.get(pred_idx, "NEUTRAL")
        confidence = float(probs[pred_idx])
    except Exception:
        return None

    sentiment_score = {"POSITIVE": 0.65, "NEUTRAL": 0.0, "NEGATIVE": -0.65}.get(label, 0.0)
    return SentimentResult(label, round(confidence, 4), sentiment_score, 0, 0, "indobert_pretrained")


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

    evaluation = _load_sentiment_model_evaluation()
    status = {
        "engine": "ml_tfidf_linear_svm",
        "label": "TF-IDF + Linear SVM",
        "dataset_path": str(dataset_path),
        "dataset_available": True,
        "model_available": True,
        "training_rows": usable_rows,
        "description": "Memakai pipeline tugas_nlp_ai_trading: preprocessing teks, TF-IDF ngram 1-2, dan Linear SVM.",
        "holdout_evaluation": evaluation,
    }
    if evaluation is None:
        status["holdout_evaluation_note"] = (
            "Dataset terlalu kecil/tidak seimbang untuk validasi train/test yang andal "
            "(minimal 30 baris berlabel dan tiap kelas >=2 baris)."
        )
    return status


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


def _analyze_text_ml(text: str) -> SentimentResult | None:
    model = _load_financial_sentiment_model()
    cleaned = clean_text(text)
    if model is None or not cleaned:
        return None
    return _predict_with_model(model, text, "ml_tfidf_linear_svm")


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
    atomic_write_csv(dataset, output, index=False)
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
    atomic_write_csv(updated, path, index=False)
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
    atomic_write_csv(updated, path, index=False)
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


DRIFT_Z_THRESHOLD = 2.0


def compute_sentiment_drift(
    dataset_path: str | Path, window_days: int = 7, date_column: str = "date"
) -> dict | None:
    """Mendeteksi pergeseran (drift) distribusi sentimen dibanding baseline historis.

    Membandingkan rata-rata `sentiment_score` pada window TERBARU
    (`window_days` hari terakhir dari data) terhadap baseline (seluruh data
    SEBELUM window itu) lewat z-score. |z| > 2 dianggap pergeseran signifikan
    -- ambang praktis/heuristik (rule-of-thumb deteksi outlier sederhana),
    BUKAN uji signifikansi statistik formal. Drift terdeteksi bisa berarti
    kondisi pasar/berita memang berubah, atau tanda dataset training sudah
    usang (waktunya retrain) -- lihat ROADMAP_COGNITIVE_DASHBOARD.md Bagian B1.

    Mengembalikan None kalau dataset tidak ada, tidak punya kolom
    tanggal/sentiment_score, atau tidak cukup data untuk baseline vs window
    terbaru (mis. semua data jatuh di satu window saja).
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return None
    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return None
    if date_column not in df.columns or "sentiment_score" not in df.columns:
        return None

    data = df[[date_column, "sentiment_score"]].dropna().copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data["sentiment_score"] = pd.to_numeric(data["sentiment_score"], errors="coerce")
    data = data.dropna(subset=[date_column, "sentiment_score"])
    if data.empty:
        return None

    daily = data.groupby(data[date_column].dt.normalize())["sentiment_score"].mean().sort_index()
    if len(daily) < 2:
        return None

    latest_date = daily.index.max()
    cutoff = latest_date - pd.Timedelta(days=window_days)
    recent = daily[daily.index > cutoff]
    baseline = daily[daily.index <= cutoff]
    if recent.empty or baseline.empty:
        return None

    baseline_mean = float(baseline.mean())
    baseline_std = float(baseline.std()) if len(baseline) > 1 else 0.0
    recent_mean = float(recent.mean())
    # Floor minimum pada std -- kalau baseline nyaris konstan (std~0), z-score
    # murni bisa meledak/tidak terdefinisi. Pakai batas bawah praktis (5% dari
    # rentang skor sentimen -1..1) supaya pergeseran besar TETAP terdeteksi
    # sebagai drift, bukan malah dianggap "tidak ada drift" karena baseline
    # kebetulan datar.
    effective_std = max(baseline_std, 0.05)
    drift_z_score = (recent_mean - baseline_mean) / effective_std

    return {
        "baseline_mean": round(baseline_mean, 4),
        "baseline_std": round(baseline_std, 4),
        "baseline_days": int(len(baseline)),
        "recent_mean": round(recent_mean, 4),
        "recent_days": int(len(recent)),
        "drift_z_score": round(drift_z_score, 2),
        "is_drifting": bool(abs(drift_z_score) > DRIFT_Z_THRESHOLD),
        "daily_series": [
            {"date": str(idx.date()), "avg_sentiment_score": round(float(value), 4)}
            for idx, value in daily.items()
        ],
    }
