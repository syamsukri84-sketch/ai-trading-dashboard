from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.nlp.sentiment_analyzer import build_local_sentiment_dataset, get_local_sentiment_dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangun dataset sentimen lokal AI Trading.")
    parser.add_argument(
        "--source",
        default="data/sentiment/market_issues.csv",
        help="CSV sumber berisi minimal kolom text.",
    )
    parser.add_argument(
        "--output",
        default=str(get_local_sentiment_dataset_path()),
        help="Lokasi output dataset lokal.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Jangan gabungkan contoh seed negatif/netral.",
    )
    args = parser.parse_args()

    dataset = build_local_sentiment_dataset(
        Path(args.source),
        Path(args.output),
        include_seed_examples=not args.no_seed,
    )
    print(f"Dataset tersimpan: {Path(args.output).resolve()}")
    print(f"Jumlah baris: {len(dataset)}")
    if not dataset.empty:
        print("Distribusi label:")
        print(dataset["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
