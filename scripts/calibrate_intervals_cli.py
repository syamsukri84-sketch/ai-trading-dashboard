"""Kalibrasi k interval EWMA lintas-universe lalu bekukan ke data/interval_calibration.csv.

Pemakaian (dari root proyek, venv aktif):
    python scripts/calibrate_intervals_cli.py [--horizon 10] [--max-tickers N] [--dry-run]

Jalankan JARANG (mis. kuartalan) atau ketika monitoring coverage berstatus
RESTRICTED -- bukan bagian workflow harian. Lihat docstring
src/trading/interval_forecaster.py untuk dasar metodologinya.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.interval_forecaster import CALIBRATION_FILENAME, calibrate_k  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,3,5,10",
        help="Daftar horizon hari bursa dipisah koma (default: 1,2,3,5,10)",
    )
    parser.add_argument("--max-tickers", type=int, default=None, help="Batasi jumlah ticker (untuk uji cepat)")
    parser.add_argument("--dry-run", action="store_true", help="Hitung tanpa menulis file kalibrasi")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(root, "data", "raw")
    save_to = None if args.dry_run else os.path.join(root, CALIBRATION_FILENAME)
    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())

    result = calibrate_k(
        raw_dir=raw_dir,
        horizons=horizons,
        max_tickers=args.max_tickers,
        save_to=save_to,
    )
    print(result.to_string(index=False))
    if save_to:
        print(f"\nTersimpan (atomic) ke: {save_to}")
    else:
        print("\n--dry-run: tidak ada file yang ditulis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
