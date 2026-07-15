"""Screening walk-forward genuine edge untuk seluruh ticker di config/stocks.yaml.

Beda dengan job analisis biasa (run_analysis.py lewat background_analysis_job.py)
yang melatih semua model (LSTM, GARCH, dst) dan menyimpan artifact -- script ini
HANYA menjalankan walk-forward validation (arah H+1, return H+3/H+5/H+10) memakai
hyperparameter produksi yang sama persis (DirectionClassifier/PriceProjector),
supaya bisa menyaring seluruh ticker dengan cepat tanpa training LSTM/GARCH yang
tidak relevan untuk pertanyaan "apakah ticker ini punya edge nyata di atas
baseline naif".

Hasil disimpan ke data/edge_screening_status.json, dipakai dashboard untuk
memfilter ticker yang benar-benar punya edge tervalidasi lewat walk-forward.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# HARUS di-set SEBELUM numpy/sklearn/lightgbm/xgboost ter-import (lihat impor
# di bawah) -- tiap worker proses paralel dibatasi 1 thread internal BLAS/OMP,
# supaya N proses x M thread masing-masing tidak oversubscribe CPU dan malah
# memperlambat total throughput. Pola sama seperti scripts/sync_mongodb_cli.py.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from data_loader import DataLoader
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.models.direction_classifier import DirectionClassifier
from src.models.price_projector import PriceProjector
from src.models.walk_forward import (
    EDGE_THRESHOLD_PCT,
    apply_fdr_correction,
    walk_forward_direction_validation,
    walk_forward_return_validation,
)
from src.utils.model_store import normalize_ticker

OUTPUT_PATH = os.path.join("data", "edge_screening_status.json")
MIN_FEATURE_ROWS = 300


def _load_tickers(config_path: str, raw_arg: str, limit: int) -> list[str]:
    if raw_arg.strip():
        tickers = [normalize_ticker(t) for t in raw_arg.split(",") if normalize_ticker(t)]
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        tickers = [normalize_ticker(t) for t in config.get("tickers", []) if normalize_ticker(t)]
    if limit and limit > 0:
        tickers = tickers[:limit]
    return tickers


def screen_ticker(ticker: str, loader: DataLoader, engineer: FeatureEngineer, idx_df) -> tuple[dict | None, str | None]:
    df = loader.load_data(ticker)
    if df is None:
        return None, "Data harga lokal tidak ditemukan/tidak cukup."
    features_df = engineer.generate_features(df, idx_df=idx_df)
    if features_df.empty or len(features_df) < MIN_FEATURE_ROWS:
        return None, f"Data fitur tidak cukup untuk walk-forward (baris={len(features_df)})."

    # Factory harus identik dengan DirectionClassifier/PriceProjector produksi
    # (lihat catatan di run_analysis.py) supaya verdict edge benar-benar
    # menguji model yang sama dengan yang dipakai untuk prediksi live --
    # termasuk wrapping CalibratedClassifierCV (calibrate=True default),
    # bukan estimator mentah. Lihat audit codebase 2026-07-12.
    direction_factory = lambda: DirectionClassifier(horizon_days=1, model_type="lightgbm").build_walk_forward_estimator(n_train_rows=252)
    return_factory = lambda: PriceProjector(projection_horizon=3, model_type="xgboost").model

    wf_h1 = walk_forward_direction_validation(features_df, direction_factory, horizon_days=1)
    wf_h3 = walk_forward_return_validation(features_df, return_factory, horizon_days=3)
    wf_h5 = walk_forward_return_validation(features_df, return_factory, horizon_days=5)
    wf_h10 = walk_forward_return_validation(features_df, return_factory, horizon_days=10)

    if wf_h1["samples"] == 0:
        return None, "Tidak cukup fold walk-forward yang bisa dibentuk dari data ini."

    # CATATAN: "edge_effect_size_ok_*" di sini HANYA gate effect-size (ambang
    # tetap EDGE_THRESHOLD_PCT), BUKAN verdict akhir. Verdict akhir
    # "has_genuine_edge_*" butuh JUGA lolos koreksi multiple-testing (FDR),
    # yang baru bisa dihitung setelah p-value SELURUH ticker terkumpul --
    # lihat main() di bawah. Menguji ratusan ticker sekaligus dengan ambang
    # effect-size tetap TANPA koreksi ini menghasilkan false positive yang
    # diharapkan murni dari varians sampel (lihat audit codebase 2026-07-12).
    edge_effect_size_ok_h1 = wf_h1["edge_vs_baseline_pct"] >= EDGE_THRESHOLD_PCT
    edge_effect_size_ok_h3 = (
        wf_h3["edge_vs_mean_mae_pct"] > 0
        and wf_h3["direction_accuracy_pct"] >= wf_h3["baseline_mean_direction_accuracy_pct"] + EDGE_THRESHOLD_PCT
    )
    edge_effect_size_ok_h5 = (
        wf_h5["edge_vs_mean_mae_pct"] > 0
        and wf_h5["direction_accuracy_pct"] >= wf_h5["baseline_mean_direction_accuracy_pct"] + EDGE_THRESHOLD_PCT
    )
    edge_effect_size_ok_h10 = (
        wf_h10["edge_vs_mean_mae_pct"] > 0
        and wf_h10["direction_accuracy_pct"] >= wf_h10["baseline_mean_direction_accuracy_pct"] + EDGE_THRESHOLD_PCT
    )

    result = {
        "ticker": ticker,
        "samples_h1": wf_h1["samples"],
        "walk_forward_h1_accuracy_pct": wf_h1["direction_accuracy_pct"],
        "walk_forward_h1_baseline_majority_pct": wf_h1["baseline_majority_accuracy_pct"],
        "walk_forward_h1_edge_pct": wf_h1["edge_vs_baseline_pct"],
        "walk_forward_h1_pvalue": wf_h1["p_value_vs_baseline"],
        "walk_forward_h1_n_folds": wf_h1["n_folds"],
        "edge_effect_size_ok_h1": bool(edge_effect_size_ok_h1),
        "walk_forward_h3_edge_mae_pct": wf_h3["edge_vs_mean_mae_pct"],
        "walk_forward_h3_pvalue": wf_h3["p_value_vs_baseline"],
        "edge_effect_size_ok_h3": bool(edge_effect_size_ok_h3),
        "walk_forward_h5_edge_mae_pct": wf_h5["edge_vs_mean_mae_pct"],
        "walk_forward_h5_pvalue": wf_h5["p_value_vs_baseline"],
        "edge_effect_size_ok_h5": bool(edge_effect_size_ok_h5),
        "walk_forward_h10_edge_mae_pct": wf_h10["edge_vs_mean_mae_pct"],
        "walk_forward_h10_pvalue": wf_h10["p_value_vs_baseline"],
        "edge_effect_size_ok_h10": bool(edge_effect_size_ok_h10),
        # Diisi setelah koreksi FDR lintas-ticker di main() -- placeholder di
        # sini supaya key selalu ada meski screen_ticker() dipanggil sendiri.
        "has_genuine_edge_h1": bool(edge_effect_size_ok_h1),
        "has_genuine_edge_h3": bool(edge_effect_size_ok_h3),
        "has_genuine_edge_h5": bool(edge_effect_size_ok_h5),
        "has_genuine_edge_h10": bool(edge_effect_size_ok_h10),
        "has_any_genuine_edge": bool(
            edge_effect_size_ok_h1 or edge_effect_size_ok_h3 or edge_effect_size_ok_h5 or edge_effect_size_ok_h10
        ),
    }
    return result, None


def _screen_ticker_worker(ticker: str, idx_df) -> tuple[str, dict | None, str | None]:
    """Entry point untuk tiap proses paralel (ProcessPoolExecutor). HARUS jadi
    fungsi top-level yang picklable -- di Windows (spawn mode), tiap proses
    anak mengimpor ulang modul ini dari nol, jadi `loader`/`engineer` dibuat
    fresh di sini, TIDAK diwariskan dari proses induk. `idx_df` (data ^JKSE)
    dikirim dari induk supaya tidak perlu dibaca ulang dari disk di tiap
    proses -- aman karena DataFrame murni data (immutable dalam pemakaian
    ini), tidak ada state yang perlu disinkronkan lintas proses."""
    loader = DataLoader(min_rows=252)
    engineer = FeatureEngineer(warmup_period=60)
    try:
        result, reason = screen_ticker(ticker, loader, engineer, idx_df)
    except Exception as e:
        result, reason = None, str(e)
    return ticker, result, reason


def apply_fdr_correction_across_universe(results: list[dict], alpha: float = 0.05) -> None:
    """Terapkan koreksi Benjamini-Hochberg PER HORIZON lintas seluruh ticker
    yang discreening, lalu perbarui `has_genuine_edge_*`/`has_any_genuine_edge`
    di tempat (in-place) supaya verdict akhir mensyaratkan effect-size DAN
    signifikansi setelah koreksi -- bukan cuma effect-size sendirian. Lihat
    docstring `apply_fdr_correction` di src/models/walk_forward.py."""
    if not results:
        return
    for horizon in ("h1", "h3", "h5", "h10"):
        p_values = [r[f"walk_forward_{horizon}_pvalue"] for r in results]
        significant = apply_fdr_correction(p_values, alpha=alpha)
        for row, is_significant in zip(results, significant):
            row[f"significant_after_fdr_{horizon}"] = bool(is_significant)
            row[f"has_genuine_edge_{horizon}"] = bool(row[f"edge_effect_size_ok_{horizon}"] and is_significant)
    for row in results:
        row["has_any_genuine_edge"] = bool(
            row["has_genuine_edge_h1"] or row["has_genuine_edge_h3"]
            or row["has_genuine_edge_h5"] or row["has_genuine_edge_h10"]
        )


def _print_ticker_progress(completed_count: int, total: int, ticker: str, result: dict | None, reason: str | None) -> None:
    if result is None:
        print(f"[{completed_count}/{total}] {ticker}: dilewati ({reason})")
        return
    print(
        f"[{completed_count}/{total}] {ticker}: H+1 edge={result['walk_forward_h1_edge_pct']:+.1f}pp, "
        f"p={result['walk_forward_h1_pvalue']:.3f} "
        f"({'lolos effect-size' if result['edge_effect_size_ok_h1'] else 'tidak ada'}, verdict FDR menyusul)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Screening walk-forward genuine edge untuk semua ticker.")
    parser.add_argument("--config", default="config/stocks.yaml")
    parser.add_argument("--tickers", default="", help="Daftar ticker dipisah koma. Kosong = semua ticker di config.")
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah ticker (untuk uji coba).")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Jumlah proses paralel. 0 (default) = otomatis pakai semua core CPU "
            "(os.cpu_count()). 1 = sekuensial (jalur lama, berguna untuk debug)."
        ),
    )
    args = parser.parse_args()

    tickers = _load_tickers(args.config, args.tickers, args.limit)
    if not tickers:
        print("Tidak ada ticker untuk discreening.")
        return 1

    loader = DataLoader(min_rows=252)
    idx_df = loader.load_data("^JKSE")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)
    workers = max(1, min(workers, len(tickers)))

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Screening genuine edge untuk {len(tickers)} ticker (mulai {started_at}, {workers} proses paralel)...")
    results = []
    failures = []
    completed_count = 0
    t_start = time.time()

    if workers == 1:
        # Jalur sekuensial dipertahankan (mis. mesin 1 core, atau debug --
        # traceback lebih mudah dibaca tanpa lapisan ProcessPoolExecutor).
        engineer = FeatureEngineer(warmup_period=60)
        for ticker in tickers:
            try:
                result, reason = screen_ticker(ticker, loader, engineer, idx_df)
            except Exception as e:
                result, reason = None, str(e)
            completed_count += 1
            if result is None:
                failures.append({"ticker": ticker, "reason": reason})
            else:
                results.append(result)
            _print_ticker_progress(completed_count, len(tickers), ticker, result, reason)
    else:
        # Tiap ticker independen (baca CSV sendiri, latih model sendiri, tidak
        # ada file bersama yang ditulis selama screening) -- aman diparalelkan
        # penuh. Ini peluang percepatan terbesar di script ini: screening
        # penuh ~265 ticker sebelumnya ~1 jam berjalan sekuensial satu-satu.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_screen_ticker_worker, ticker, idx_df): ticker for ticker in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    _, result, reason = future.result()
                except Exception as e:
                    result, reason = None, str(e)
                completed_count += 1
                if result is None:
                    failures.append({"ticker": ticker, "reason": reason})
                else:
                    results.append(result)
                _print_ticker_progress(completed_count, len(tickers), ticker, result, reason)

    # Koreksi multiple-testing PER HORIZON lintas seluruh ticker yang baru
    # discreening -- baru di sini karena butuh p-value SEMUA ticker terkumpul
    # dulu. Ini memperbarui has_genuine_edge_* di setiap `results[i]` in-place
    # jadi verdict akhir (dipakai `genuine_edge_h1`/`genuine_edge_any` di
    # bawah) mensyaratkan effect-size DAN signifikansi setelah FDR sekaligus.
    apply_fdr_correction_across_universe(results)

    elapsed = time.time() - t_start
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    genuine_edge_h1 = [r["ticker"] for r in results if r["has_genuine_edge_h1"]]
    genuine_edge_any = [r["ticker"] for r in results if r["has_any_genuine_edge"]]

    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed, 1),
        "total_requested": len(tickers),
        "total_screened": len(results),
        "total_failed": len(failures),
        "edge_threshold_pct": EDGE_THRESHOLD_PCT,
        # "has_genuine_edge_*" di tiap ticker (dalam "results") SEKARANG
        # mensyaratkan effect-size >= edge_threshold_pct DAN signifikan
        # setelah koreksi Benjamini-Hochberg FDR lintas seluruh ticker yang
        # discreening di horizon yang sama -- bukan cuma ambang effect-size
        # sendirian. Lihat apply_fdr_correction di src/models/walk_forward.py
        # dan audit codebase 2026-07-12.
        "fdr_alpha": 0.05,
        "tickers_with_genuine_edge_h1": genuine_edge_h1,
        "tickers_with_any_genuine_edge": genuine_edge_any,
        "results": results,
        "failures": failures,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Selesai dalam {elapsed:.1f}s. Discreening: {len(results)}/{len(tickers)}. Gagal: {len(failures)}.")
    print("Verdict di bawah SUDAH dikoreksi FDR (Benjamini-Hochberg, alpha=0.05) lintas seluruh ticker.")
    print(f"Ticker dengan edge nyata H+1: {len(genuine_edge_h1)} -> {genuine_edge_h1}")
    print(f"Ticker dengan edge nyata di horizon manapun: {len(genuine_edge_any)} -> {genuine_edge_any}")
    print(f"Hasil disimpan: {args.output}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
