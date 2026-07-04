import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.global_models import predict_with_global_models


def load_tickers(config_path: str) -> list[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return [str(t).replace(".JK", "").upper().strip() for t in config.get("tickers", []) if str(t).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily predictions with saved global models.")
    parser.add_argument("--config", default="config/stocks.yaml")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker override.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--duplicate-policy", default="skip", choices=["skip", "overwrite", "intraday"])
    parser.add_argument("--run-type", default="FINAL", choices=["FINAL", "INTRADAY", "BACKFILL"])
    args = parser.parse_args()

    tickers = (
        [str(t).replace(".JK", "").upper().strip() for t in args.tickers.split(",") if str(t).strip()]
        if args.tickers
        else load_tickers(args.config)
    )
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]
    if not tickers:
        print("Tidak ada ticker untuk prediksi global.")
        return 1

    def progress(event):
        ticker = event.get("ticker") or "-"
        completed = int(event.get("completed") or 0)
        total = int(event.get("total") or len(tickers))
        print(f"[{completed}/{total}] {ticker}")

    print("Prediksi Global Model V1")
    print("=" * 72)
    summary = predict_with_global_models(
        tickers=tickers,
        duplicate_policy=args.duplicate_policy,
        prediction_run_type=args.run_type,
        progress_callback=progress,
    )
    print("\nRingkasan")
    print("=" * 72)
    print(f"Berhasil: {len(summary.get('predicted', []))}")
    print(f"Dilewati: {len(summary.get('skipped', []))}")
    print(f"Gagal   : {len(summary.get('failed', []))}")
    if summary.get("failed"):
        print("\nContoh gagal:")
        for row in summary["failed"][:20]:
            print(f"- {row.get('ticker')} {row.get('model_name', '')}: {row.get('reason')}")
    if summary.get("skipped"):
        print("\nDilewati:")
        for row in summary["skipped"][:20]:
            print(f"- {row.get('ticker')}: {row.get('reason')}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
