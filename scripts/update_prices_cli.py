import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.auto_updater import get_local_data_status, run_auto_updater


def parse_tickers(raw: str) -> list[str] | None:
    if not raw.strip():
        return None
    return [
        item.replace(".JK", "").upper().strip()
        for item in raw.split(",")
        if item.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Update data harga saham lokal dari terminal.")
    parser.add_argument("--tickers", default="", help="Daftar ticker dipisah koma, contoh: BBCA,BBRI,BMRI.")
    parser.add_argument("--config", default="config/stocks.yaml", help="Path config ticker.")
    parser.add_argument("--data-dir", default="data/raw", help="Folder CSV harga lokal.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Jeda antar ticker dalam detik.")
    parser.add_argument("--summary-dir", default="data", help="Folder output ringkasan JSON.")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers)
    summary = run_auto_updater(
        config_path=args.config,
        data_dir=args.data_dir,
        tickers=tickers,
        sleep_seconds=float(args.sleep),
    )

    output_dir = Path(args.summary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"price_update_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    checked_tickers = tickers or [row.get("ticker") for row in summary.get("updated", []) + summary.get("skipped", [])]
    status_df = get_local_data_status(checked_tickers, data_dir=args.data_dir) if checked_tickers else None

    print(f"Ringkasan disimpan: {output_path}")
    print(f"Total   : {summary.get('total', 0)}")
    print(f"Updated : {len(summary.get('updated', []))}")
    print(f"Skipped : {len(summary.get('skipped', []))}")
    print(f"Failed  : {len(summary.get('failed', []))}")
    if status_df is not None and not status_df.empty:
        print("Tanggal data terakhir:")
        print(status_df["last_date"].value_counts(dropna=False).head(10).to_string())
    if summary.get("failed"):
        print("Gagal:")
        for row in summary["failed"][:50]:
            print(f"- {row.get('ticker')}: {row.get('reason')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
