import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_analysis import run_full_analysis
from src.utils.model_store import model_store_status
from src.utils.training_policy import evaluate_training_policy_by_model


def load_config_tickers(config_path: str) -> list[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return [str(t).replace(".JK", "").upper().strip() for t in config.get("tickers", []) if str(t).strip()]


def parse_tickers(args) -> list[str]:
    if args.tickers:
        return [str(t).replace(".JK", "").upper().strip() for t in args.tickers.split(",") if str(t).strip()]
    tickers = load_config_tickers(args.config)
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]
    return tickers


def print_accuracy_report(min_accuracy: float, min_samples: int, lookback: int) -> None:
    report = evaluate_training_policy_by_model(
        min_direction_accuracy_pct=float(min_accuracy),
        min_recent_evaluations=int(min_samples),
        lookback_evaluations=int(lookback),
        prediction_purpose="NEXT_DAY_DIRECTION",
    )
    if report.empty:
        print("\nBelum ada data akurasi FINAL H+1 untuk membuat laporan target akurasi.")
        return

    print("\nLaporan akurasi historis FINAL H+1 per model")
    print("=" * 72)
    view = report.copy()
    view["target_tercapai"] = view["recent_accuracy_pct"].fillna(0) >= float(min_accuracy)
    for _, row in view.iterrows():
        acc = row.get("recent_accuracy_pct")
        acc_text = "-" if pd.isna(acc) else f"{float(acc):.2f}%"
        verdict = "OK" if bool(row.get("target_tercapai")) else "BELUM"
        print(
            f"{row['model_name']:<26} akurasi={acc_text:<8} "
            f"sampel={int(row['recent_evaluations']):<4} target>={min_accuracy:.1f}%: {verdict}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Training model AI trading dari terminal VS Code. "
            "Hasil training otomatis menyimpan artifact ke data/models."
        )
    )
    parser.add_argument("--config", default="config/stocks.yaml", help="Path config ticker.")
    parser.add_argument("--tickers", default="", help="Daftar ticker dipisah koma, contoh: BBCA,BBRI,BMRI.")
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah ticker pertama dari config untuk uji coba.")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch LSTM. Pakai 1-3 untuk training cepat.")
    parser.add_argument("--duplicate-policy", default="skip", choices=["skip", "overwrite", "intraday"])
    parser.add_argument("--run-type", default="FINAL", choices=["FINAL", "INTRADAY", "BACKFILL"])
    parser.add_argument("--force", action="store_true", help="Latih ulang walau prediksi inti tanggal terbaru sudah ada.")
    parser.add_argument("--min-accuracy", type=float, default=70.0, help="Target akurasi laporan historis.")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimal sampel untuk membaca target akurasi.")
    parser.add_argument("--lookback", type=int, default=100, help="Jumlah evaluasi terbaru untuk laporan akurasi.")
    args = parser.parse_args()

    tickers = parse_tickers(args)
    if not tickers:
        print("Tidak ada ticker untuk dilatih.")
        return 1

    print("Training model AI trading via CLI")
    print("=" * 72)
    print(f"Waktu mulai     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Jumlah ticker   : {len(tickers)}")
    print(f"Epoch LSTM      : {args.epochs}")
    print(f"Run type        : {args.run_type}")
    print(f"Duplicate policy: {args.duplicate_policy}")
    print(f"Force retrain   : {args.force}")
    print(f"Ticker awal     : {', '.join(tickers[:20])}{' ...' if len(tickers) > 20 else ''}")
    print("=" * 72)

    def progress(event):
        stage = event.get("stage", "")
        ticker = event.get("ticker") or "-"
        message = event.get("message", "")
        completed = event.get("completed", 0)
        total = event.get("total", len(tickers))
        if stage in {"ticker_started", "ticker_succeeded", "ticker_failed", "ticker_skipped", "done"}:
            print(f"[{completed}/{total}] {ticker:<8} {stage:<18} {message}")

    summary = run_full_analysis(
        tickers=tickers,
        lstm_epochs=int(args.epochs),
        progress_callback=progress,
        duplicate_policy=args.duplicate_policy,
        prediction_run_type=args.run_type,
        skip_completed=not args.force,
    )

    print("\nRingkasan training")
    print("=" * 72)
    print(f"Berhasil dianalisis : {len(summary.get('analyzed', []))}")
    print(f"Dilewati            : {len(summary.get('skipped', []))}")
    print(f"Gagal               : {len(summary.get('failed', []))}")
    if summary.get("failed"):
        print("\nContoh gagal:")
        for row in summary["failed"][:20]:
            print(f"- {row.get('ticker')}: {row.get('reason')}")

    store = model_store_status(tickers)
    print("\nModel store")
    print("=" * 72)
    print(f"Ticker punya model : {store['total_tickers_with_models']}")
    print(f"Artifact model     : {store['total_artifacts']}")
    print(f"Update terakhir    : {store['last_updated_at']}")
    missing_model_tickers = [
        row["ticker"]
        for row in store.get("rows", [])
        if int(row.get("model_count") or 0) == 0
    ]
    print(f"Ticker belum punya model: {len(missing_model_tickers)}")
    if missing_model_tickers:
        print("Contoh ticker belum punya model:")
        print(", ".join(missing_model_tickers[:50]) + (" ..." if len(missing_model_tickers) > 50 else ""))

    print_accuracy_report(args.min_accuracy, args.min_samples, args.lookback)
    print("\nCatatan: target akurasi >=70% tidak bisa dijamin oleh training. Script ini melatih dan melaporkan apakah histori evaluasi sudah memenuhi target.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
