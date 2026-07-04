import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_analysis import run_full_analysis


JOB_DIR = os.path.join("data", "jobs")


def write_job_status(job_id, payload, max_attempts=8, retry_delay=0.15):
    os.makedirs(JOB_DIR, exist_ok=True)
    payload["job_id"] = job_id
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(JOB_DIR, f"analysis_{job_id}.json")
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    last_error = None
    for attempt in range(max_attempts):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError as e:
            last_error = e
            time.sleep(retry_delay * (attempt + 1))

    fallback_path = f"{path}.failed_write_{os.getpid()}.json"
    try:
        os.replace(temp_path, fallback_path)
    except OSError:
        pass
    raise last_error


def build_initial_ticker_statuses(tickers):
    return {
        ticker: {
            "ticker": ticker,
            "status": "MENUNGGU",
            "stage": "pending",
            "message": "Menunggu giliran analisis.",
            "started_at": None,
            "finished_at": None,
            "reason": "",
        }
        for ticker in tickers
    }


def main():
    parser = argparse.ArgumentParser(description="Run AI trading analysis as a background job.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker list.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--duplicate-policy", default="skip", choices=["skip", "overwrite", "intraday"])
    parser.add_argument("--run-type", default=None)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Latih ulang model walau prediksi FINAL tanggal terbaru sudah ada. Prediksi lama tetap mengikuti duplicate-policy.",
    )
    args = parser.parse_args()

    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    state = {
        "status": "RUNNING",
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "tickers": tickers,
        "total": len(tickers),
        "completed": 0,
        "analyzed_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "current_ticker": None,
        "message": "Job analisis background dimulai.",
        "events": [],
        "ticker_statuses": build_initial_ticker_statuses(tickers),
        "summary": None,
        "error": None,
        "duplicate_policy": args.duplicate_policy,
        "prediction_run_type": args.run_type or ("INTRADAY" if args.duplicate_policy == "intraday" else "FINAL"),
        "force_retrain": bool(args.force_retrain),
    }
    write_job_status(args.job_id, state)

    def progress_callback(event):
        stage = event.get("stage", "")
        ticker = event.get("ticker")
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["total"] = int(event.get("total") or state["total"] or 0)
        state["completed"] = int(event.get("completed") or 0)
        state["analyzed_count"] = int(event.get("analyzed_count") or state["analyzed_count"] or 0)
        state["failed_count"] = int(event.get("failed_count") or state["failed_count"] or 0)
        state["skipped_count"] = int(event.get("skipped_count") or state["skipped_count"] or 0)
        state["current_ticker"] = ticker or state.get("current_ticker")
        state["message"] = event.get("message", state["message"])
        state["stage"] = stage
        if ticker:
            ticker_status = state["ticker_statuses"].setdefault(ticker, {
                "ticker": ticker,
                "status": "MENUNGGU",
                "stage": "pending",
                "message": "",
                "started_at": None,
                "finished_at": None,
                "reason": "",
            })
            ticker_status["stage"] = stage
            ticker_status["message"] = event.get("message", "")
            if stage == "ticker_started":
                ticker_status["status"] = "BERJALAN"
                ticker_status["started_at"] = ticker_status.get("started_at") or now_text
                ticker_status["finished_at"] = None
                ticker_status["reason"] = ""
            elif stage == "ticker_succeeded":
                ticker_status["status"] = "SELESAI"
                ticker_status["finished_at"] = now_text
                ticker_status["reason"] = ""
            elif stage == "ticker_failed":
                ticker_status["status"] = "GAGAL"
                ticker_status["finished_at"] = now_text
                ticker_status["reason"] = event.get("reason", "")
            elif stage == "ticker_skipped":
                ticker_status["status"] = "DILEWATI"
                ticker_status["finished_at"] = now_text
                ticker_status["reason"] = event.get("reason", "")
        state["events"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": ticker or "-",
            "stage": stage,
            "message": event.get("message", ""),
        })
        state["events"] = state["events"][-50:]
        write_job_status(args.job_id, state)

    try:
        summary = run_full_analysis(
            tickers=tickers,
            lstm_epochs=args.epochs,
            progress_callback=progress_callback,
            duplicate_policy=args.duplicate_policy,
            prediction_run_type=args.run_type,
            skip_completed=not bool(args.force_retrain),
        )
        state["status"] = "DONE"
        state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["summary"] = summary
        state["message"] = "Analisis background selesai."
        state["completed"] = state["total"]
        state["analyzed_count"] = len(summary.get("analyzed", []))
        state["failed_count"] = len(summary.get("failed", []))
        state["skipped_count"] = len(summary.get("skipped", []))
        write_job_status(args.job_id, state)
    except Exception as e:
        state["status"] = "FAILED"
        state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["error"] = f"{e}\n{traceback.format_exc()}"
        state["message"] = f"Analisis background gagal: {e}"
        write_job_status(args.job_id, state)
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
