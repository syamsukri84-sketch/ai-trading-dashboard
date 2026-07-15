"""Bandingkan engine sentimen produksi (TF-IDF+SVM) vs 3 alternatif berbasis
model bahasa pretrained (IndoBERT classifier, embedding Indo-E5+SVM, embedding
IndoBERT+SVM), diuji ULANG di 5 random_state berbeda (bukan satu split) --
lihat docstring `compare_sentiment_engines_repeated` di
src/nlp/sentiment_analyzer.py untuk alasan kenapa satu split saja bisa
menyesatkan pada dataset held-out sekecil ini (~54 baris test).

Dipindahkan dari streamlit_app.py (2026-07-13) -- ini alat riset/perbandingan
model, bukan kebutuhan pemantauan harian, dan butuh dependency berat
(transformers+torch, unduhan model ~500MB-1GB saat pertama kali dipakai).
Menjalankannya di dashboard produksi menambah beban & kebingungan tanpa
manfaat harian. Jalankan manual lewat CLI ini kapan pun ingin mengevaluasi
ulang apakah engine produksi masih menang melawan alternatif.

Penggunaan:
    python scripts/compare_sentiment_engines_cli.py
    python scripts/compare_sentiment_engines_cli.py --dataset data/sentiment/processed/financial_news_clean.csv
    python scripts/compare_sentiment_engines_cli.py --output data/sentiment/reports/engine_comparison.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nlp.sentiment_analyzer import compare_sentiment_engines_repeated, get_local_sentiment_dataset_path

ENGINE_LABELS = [
    ("tfidf_svm", "TF-IDF+SVM (produksi)"),
    ("indobert_pretrained", "IndoBERT Classifier Pretrained"),
    ("indo_e5_embedding_svm", "Embedding Indo-E5 + SVM"),
    ("indobert_embedding_svm", "Embedding IndoBERT + SVM"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bandingkan engine sentimen (robust, 5 split acak).")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path dataset sentimen lokal. Kosong = pakai get_local_sentiment_dataset_path() default.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Kalau diisi, simpan hasil lengkap sebagai JSON ke path ini.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else get_local_sentiment_dataset_path()
    if not dataset_path.exists():
        print(f"Dataset tidak ditemukan: {dataset_path}")
        return 1

    print(f"Dataset: {dataset_path}")
    print("Menjalankan evaluasi held-out di 5 split acak berbeda untuk semua engine "
          "(butuh unduhan model transformers ~500MB-1GB saat pertama kali dipakai, bisa beberapa menit)...")
    comparison = compare_sentiment_engines_repeated(dataset_path)

    print("=" * 72)
    print(f"VERDICT: {comparison['verdict']}")
    print(f"Alasan : {comparison['verdict_reason']}")
    print(f"Split yang diuji (random_state): {comparison['seeds']}")
    print("=" * 72)
    print(f"{'Engine':<32}{'Status':<14}{'Rata-rata (%)':<16}{'Std Dev':<10}{'Jumlah Split'}")
    for key, label in ENGINE_LABELS:
        summary = comparison["summary_by_engine"].get(key)
        if summary is None:
            print(f"{label:<32}{'Tidak tersedia':<14}{'-':<16}{'-':<10}{'-'}")
        else:
            print(
                f"{label:<32}{'OK':<14}{summary['mean_accuracy_pct']:<16}"
                f"{summary['stdev_accuracy_pct']:<10}{summary['n_seeds_evaluated']}"
            )
    print("=" * 72)
    print("Std Dev tinggi berarti performa engine tidak stabil antar-split -- jangan percaya "
          "angka rata-rata tanpa melihat variasinya juga.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nHasil lengkap disimpan: {output_path}")

    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
