import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.auto_updater import get_local_data_status, run_auto_updater
from src.models.global_models import predict_with_global_models
from src.nlp.sentiment_analyzer import build_local_sentiment_dataset, get_local_sentiment_dataset_path
from src.trading.market_regime import compute_market_breadth, log_regime_snapshot
from src.utils.accuracy_tracker import ACCURACY_FILE, PREDICTIONS_FILE, evaluate_pending_predictions
from src.utils.model_store import list_ticker_models


def load_tickers(config_path: str) -> list[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return [str(t).replace(".JK", "").upper().strip() for t in config.get("tickers", []) if str(t).strip()]


def parse_tickers(raw: str, config_path: str, limit: int = 0) -> list[str]:
    if raw.strip():
        tickers = [str(t).replace(".JK", "").upper().strip() for t in raw.split(",") if str(t).strip()]
    else:
        tickers = load_tickers(config_path)
    if limit and limit > 0:
        tickers = tickers[:limit]
    return tickers


def count_rows(path: str) -> int:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def count_pending_predictions(path: str = PREDICTIONS_FILE) -> int:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    if df.empty or "status" not in df.columns:
        return 0
    active = df.get("is_active", True)
    if not isinstance(active, pd.Series):
        active = pd.Series([active] * len(df))
    active = active.astype(str).str.lower().isin(["true", "1", "yes"])
    return int(((df["status"].astype(str).str.upper().str.strip() == "PENDING") & active).sum())


def write_summary(summary: dict, summary_dir: str) -> Path:
    output_dir = Path(summary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"daily_global_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Workflow harian Global Model: update data, evaluasi, prediksi.")
    parser.add_argument("--config", default="config/stocks.yaml", help="Path config ticker.")
    parser.add_argument("--tickers", default="", help="Daftar ticker dipisah koma. Kosong berarti dari config.")
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah ticker untuk uji coba.")
    parser.add_argument("--data-dir", default="data/raw", help="Folder data harga lokal.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Jeda update antar ticker.")
    parser.add_argument("--summary-dir", default="data/daily_workflows", help="Folder ringkasan workflow.")
    parser.add_argument("--skip-sentiment-dataset", action="store_true", help="Lewati build dataset sentimen lokal.")
    parser.add_argument("--skip-update", action="store_true", help="Lewati update data harga.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Lewati evaluasi prediksi pending.")
    parser.add_argument("--skip-prediction", action="store_true", help="Lewati prediksi Global Model.")
    parser.add_argument(
        "--strict-update",
        action="store_true",
        help="Return error jika ada ticker gagal update. Default tetap lanjut karena sebagian provider bisa kosong.",
    )
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers, args.config, args.limit)
    if not tickers:
        print("Tidak ada ticker untuk workflow harian.")
        return 1

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = {
        "workflow": "GLOBAL_DAILY_AFTER_MARKET",
        "started_at": started_at,
        "finished_at": None,
        "ticker_count": len(tickers),
        "steps": [],
        "status": "OK",
    }

    print("Workflow Harian Global Model")
    print("=" * 72)
    print(f"Mulai       : {started_at}")
    print(f"Jumlah saham: {len(tickers)}")
    print("=" * 72)

    if not args.skip_sentiment_dataset:
        print("\n[1/4] Build dataset sentimen lokal...")
        try:
            sentiment_dataset = build_local_sentiment_dataset()
            label_counts = sentiment_dataset["label"].value_counts().to_dict() if not sentiment_dataset.empty else {}
            summary["steps"].append({
                "step": "build_local_sentiment_dataset",
                "status": "DONE",
                "output": str(get_local_sentiment_dataset_path()),
                "rows": int(len(sentiment_dataset)),
                "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            })
            print(f"Dataset: {get_local_sentiment_dataset_path()}")
            print(f"Rows: {len(sentiment_dataset)} | Label: {label_counts}")
        except Exception as exc:
            summary["steps"].append({
                "step": "build_local_sentiment_dataset",
                "status": "FAILED",
                "reason": str(exc),
            })
            print(f"Build dataset sentimen gagal: {exc}")
    else:
        summary["steps"].append({"step": "build_local_sentiment_dataset", "status": "SKIPPED"})
        print("\n[1/4] Build dataset sentimen lokal dilewati.")

    if not args.skip_update:
        print("\n[2/4] Update data harga...")
        try:
            update_summary = run_auto_updater(
                config_path=args.config,
                data_dir=args.data_dir,
                tickers=tickers,
                sleep_seconds=float(args.sleep),
            )
            # Perbarui juga data indeks IHSG (^JKSE) -- TERPISAH dari daftar
            # `tickers` saham (sengaja tidak digabung supaya index ini tidak ikut
            # masuk ke loop prediksi/training saham di bawah). Fitur korelasi/beta
            # -terhadap-pasar di seluruh sistem bergantung pada data ini; sebelum
            # perbaikan bug simbol Yahoo Finance, file ini tidak pernah berhasil
            # terisi sama sekali (selalu fallback ke nilai netral).
            try:
                run_auto_updater(data_dir=args.data_dir, tickers=["^JKSE"], sleep_seconds=0.0)
            except Exception as exc:
                print(f"Peringatan: gagal memperbarui data indeks ^JKSE: {exc}")
            status_df = get_local_data_status(tickers, data_dir=args.data_dir)
            latest_dates = {}
            if status_df is not None and not status_df.empty:
                latest_dates = status_df["last_date"].value_counts(dropna=False).head(10).to_dict()
            step = {
                "step": "update_data",
                "status": "DONE",
                "total": update_summary.get("total", len(tickers)),
                "updated": len(update_summary.get("updated", [])),
                "skipped": len(update_summary.get("skipped", [])),
                "failed": len(update_summary.get("failed", [])),
                "latest_dates": {str(k): int(v) for k, v in latest_dates.items()},
                "failed_examples": update_summary.get("failed", [])[:20],
            }
            summary["steps"].append(step)
            print(f"Updated: {step['updated']} | Skipped: {step['skipped']} | Failed: {step['failed']}")
            if latest_dates:
                print("Tanggal data terakhir:")
                for date_value, count in latest_dates.items():
                    print(f"- {date_value}: {count}")
            if step["failed"] and args.strict_update:
                summary["status"] = "FAILED"
                summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                output_path = write_summary(summary, args.summary_dir)
                print(f"\nWorkflow berhenti karena update gagal. Ringkasan: {output_path}")
                return 1

            # Catat regime pasar hari ini (breadth naik/turun) ke riwayat --
            # dijalankan di sini (bukan hanya saat dashboard dibuka) supaya
            # otomatisasi terjadwal (Task Scheduler) tetap membangun riwayat
            # regime harian walau dashboard tidak pernah dibuka. Lihat
            # ROADMAP_COGNITIVE_DASHBOARD.md Bagian B3.
            market_breadth = compute_market_breadth(tickers, raw_dir=args.data_dir)
            log_regime_snapshot(market_breadth)
            print(f"Regime pasar hari ini: {market_breadth['market_regime']} (breadth naik {market_breadth['breadth_up_pct']:.1f}%)")
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["steps"].append({
                "step": "update_data",
                "status": "FAILED",
                "reason": str(exc),
            })
            summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_path = write_summary(summary, args.summary_dir)
            print(f"\nUpdate data harga gagal tak terduga: {exc}")
            print(f"Ringkasan: {output_path}")
            return 1
    else:
        summary["steps"].append({"step": "update_data", "status": "SKIPPED"})
        print("\n[2/4] Update data harga dilewati.")

    if not args.skip_evaluation:
        print("\n[3/4] Evaluasi prediksi pending...")
        try:
            accuracy_before = count_rows(ACCURACY_FILE)
            pending_before = count_pending_predictions()
            evaluate_pending_predictions()
            accuracy_after = count_rows(ACCURACY_FILE)
            pending_after = count_pending_predictions()
            evaluated_delta = max(accuracy_after - accuracy_before, 0)
            summary["steps"].append({
                "step": "evaluate_pending_predictions",
                "status": "DONE",
                "accuracy_rows_before": accuracy_before,
                "accuracy_rows_after": accuracy_after,
                "new_evaluations": evaluated_delta,
                "pending_before": pending_before,
                "pending_after": pending_after,
            })
            print(f"Evaluasi baru: {evaluated_delta} | Pending tersisa: {pending_after}")
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["steps"].append({
                "step": "evaluate_pending_predictions",
                "status": "FAILED",
                "reason": str(exc),
            })
            summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_path = write_summary(summary, args.summary_dir)
            print(f"\nEvaluasi prediksi pending gagal: {exc}")
            print(f"Ringkasan: {output_path}")
            return 1
    else:
        summary["steps"].append({"step": "evaluate_pending_predictions", "status": "SKIPPED"})
        print("\n[3/4] Evaluasi prediksi pending dilewati.")

    if not args.skip_prediction:
        print("\n[4/4] Prediksi Global Model...")
        try:
            global_records = list_ticker_models("GLOBAL")
            if not global_records:
                summary["status"] = "FAILED"
                summary["steps"].append({
                    "step": "predict_global_model",
                    "status": "FAILED",
                    "reason": "Global Model belum tersedia.",
                })
                summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                output_path = write_summary(summary, args.summary_dir)
                print("Global Model belum tersedia. Jalankan training global terlebih dahulu.")
                print(f"Ringkasan: {output_path}")
                return 1

            def progress(event):
                ticker = event.get("ticker") or "-"
                completed = int(event.get("completed") or 0)
                total = int(event.get("total") or len(tickers))
                if total and (completed == 0 or completed == total or completed % 25 == 0):
                    print(f"[{completed}/{total}] {ticker}")

            prediction_summary = predict_with_global_models(
                tickers=tickers,
                duplicate_policy="skip",
                prediction_run_type="FINAL",
                progress_callback=progress,
            )
            summary["steps"].append({
                "step": "predict_global_model",
                "status": "DONE",
                "predicted": len(prediction_summary.get("predicted", [])),
                "skipped": len(prediction_summary.get("skipped", [])),
                "failed": len(prediction_summary.get("failed", [])),
                "failed_examples": prediction_summary.get("failed", [])[:20],
                "skipped_examples": prediction_summary.get("skipped", [])[:20],
            })
            print(
                f"Predicted: {len(prediction_summary.get('predicted', []))} | "
                f"Skipped: {len(prediction_summary.get('skipped', []))} | "
                f"Failed: {len(prediction_summary.get('failed', []))}"
            )
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["steps"].append({
                "step": "predict_global_model",
                "status": "FAILED",
                "reason": str(exc),
            })
            summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_path = write_summary(summary, args.summary_dir)
            print(f"\nPrediksi Global Model gagal: {exc}")
            print(f"Ringkasan: {output_path}")
            return 1
    else:
        summary["steps"].append({"step": "predict_global_model", "status": "SKIPPED"})
        print("\n[4/4] Prediksi Global Model dilewati.")

    summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = write_summary(summary, args.summary_dir)
    print("\nWorkflow selesai.")
    print(f"Ringkasan disimpan: {output_path}")
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
