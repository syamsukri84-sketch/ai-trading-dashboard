"""Validasi walk-forward genuine-edge KHUSUS untuk LSTMPriceProjector --
sebelumnya (audit codebase 2026-07-12) LSTM adalah SATU-SATUNYA model di
pipeline ini yang tidak pernah divalidasi walk-forward sama sekali, karena
interface train(df)/predict(df)-nya tidak kompatibel dengan
walk_forward_return_validation (yang menerima X, y datar). Sekarang dipakai
`walk_forward_sequence_model_validation` (src/models/walk_forward.py) yang
metodologinya identik (purge gap, baseline zero/mean return, p-value
berpasangan per-fold, koreksi FDR lintas ticker) dengan yang sudah dipakai
untuk model lain.

SENGAJA DIPISAH dari scripts/screen_genuine_edge.py: melatih ulang LSTM di
tiap fold jauh lebih mahal daripada model tree (LightGBM/XGBoost) -- tidak
cocok dijalankan rutin mingguan untuk ~265 ticker. Jalankan manual, per
kebutuhan riset, pada sample ticker (default: 24, mengikuti sample yang
dipakai sesi optimasi Juli 2026).

Penggunaan:
    python scripts/validate_lstm_walk_forward.py
    python scripts/validate_lstm_walk_forward.py --tickers BBCA,BBRI,TLKM --horizon 3
    python scripts/validate_lstm_walk_forward.py --limit 10 --epochs 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from data_loader import DataLoader
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.models.lstm_projector import LSTMPriceProjector
from src.models.walk_forward import EDGE_THRESHOLD_PCT, apply_fdr_correction, walk_forward_sequence_model_validation
from src.utils.model_store import normalize_ticker

OUTPUT_PATH = os.path.join("data", "lstm_edge_screening_status.json")
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


def validate_ticker(
    ticker: str,
    loader: DataLoader,
    engineer: FeatureEngineer,
    idx_df,
    horizon_days: int,
    epochs: int,
    lookback: int,
) -> tuple[dict | None, str | None]:
    df = loader.load_data(ticker)
    if df is None:
        return None, "Data harga lokal tidak ditemukan/tidak cukup."
    features_df = engineer.generate_features(df, idx_df=idx_df)
    if features_df.empty or len(features_df) < MIN_FEATURE_ROWS:
        return None, f"Data fitur tidak cukup untuk walk-forward (baris={len(features_df)})."

    factory = lambda: LSTMPriceProjector(projection_horizon=horizon_days, lookback=lookback)
    wf = walk_forward_sequence_model_validation(features_df, factory, horizon_days=horizon_days, epochs=epochs)

    if wf["samples"] == 0:
        return None, "Tidak cukup fold walk-forward yang bisa dibentuk dari data ini."

    edge_effect_size_ok = wf["edge_vs_mean_mae_pct"] > 0 and wf["direction_accuracy_pct"] >= wf["baseline_mean_direction_accuracy_pct"] + EDGE_THRESHOLD_PCT
    result = {
        "ticker": ticker,
        "horizon_days": horizon_days,
        "samples": wf["samples"],
        "n_folds": wf["n_folds"],
        "mae_pct": wf["mae_pct"],
        "baseline_mean_mae_pct": wf["baseline_mean_mae_pct"],
        "edge_vs_mean_mae_pct": wf["edge_vs_mean_mae_pct"],
        "direction_accuracy_pct": wf["direction_accuracy_pct"],
        "baseline_mean_direction_accuracy_pct": wf["baseline_mean_direction_accuracy_pct"],
        "pvalue": wf["p_value_vs_baseline"],
        "edge_effect_size_ok": bool(edge_effect_size_ok),
    }
    return result, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validasi walk-forward genuine-edge untuk LSTMPriceProjector.")
    parser.add_argument("--config", default="config/stocks.yaml")
    parser.add_argument("--tickers", default="", help="Daftar ticker dipisah koma. Kosong = 24 ticker pertama di config.")
    parser.add_argument("--limit", type=int, default=24, help="Batasi jumlah ticker (default 24, sample sesi optimasi Juli 2026).")
    parser.add_argument("--horizon", type=int, default=3, help="Horizon hari (default 3, sama seperti PriceProjector H+3).")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch training LSTM per fold (default 3 -- kecil sengaja, ini validasi bukan produksi).")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    tickers = _load_tickers(args.config, args.tickers, args.limit)
    if not tickers:
        print("Tidak ada ticker untuk divalidasi.")
        return 1

    loader = DataLoader(min_rows=252)
    engineer = FeatureEngineer(warmup_period=60)
    idx_df = loader.load_data("^JKSE")

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Validasi walk-forward LSTM H+{args.horizon} untuk {len(tickers)} ticker (epochs={args.epochs}, mulai {started_at})...")
    print("PERINGATAN: LSTM dilatih ulang tiap fold -- jauh lebih lambat dari screen_genuine_edge.py.")
    results = []
    failures = []
    t_start = time.time()
    for i, ticker in enumerate(tickers, start=1):
        ticker_t0 = time.time()
        try:
            result, reason = validate_ticker(ticker, loader, engineer, idx_df, args.horizon, args.epochs, args.lookback)
        except Exception as e:
            result, reason = None, str(e)
        ticker_elapsed = time.time() - ticker_t0
        if result is None:
            failures.append({"ticker": ticker, "reason": reason})
            print(f"[{i}/{len(tickers)}] {ticker}: dilewati ({reason}) [{ticker_elapsed:.1f}s]")
            continue
        results.append(result)
        print(
            f"[{i}/{len(tickers)}] {ticker}: edge={result['edge_vs_mean_mae_pct']:+.3f}pp, "
            f"p={result['pvalue']:.3f} [{ticker_elapsed:.1f}s]"
        )

    if results:
        p_values = [r["pvalue"] for r in results]
        significant = apply_fdr_correction(p_values, alpha=0.05)
        for row, is_significant in zip(results, significant):
            row["significant_after_fdr"] = bool(is_significant)
            row["has_genuine_edge"] = bool(row["edge_effect_size_ok"] and is_significant)
    genuine_edge_tickers = [r["ticker"] for r in results if r.get("has_genuine_edge")]

    elapsed = time.time() - t_start
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "model": "LSTMPriceProjector",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed, 1),
        "horizon_days": args.horizon,
        "epochs": args.epochs,
        "total_requested": len(tickers),
        "total_validated": len(results),
        "total_failed": len(failures),
        "edge_threshold_pct": EDGE_THRESHOLD_PCT,
        "fdr_alpha": 0.05,
        "tickers_with_genuine_edge": genuine_edge_tickers,
        "results": results,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Selesai dalam {elapsed:.1f}s. Divalidasi: {len(results)}/{len(tickers)}. Gagal: {len(failures)}.")
    print("Verdict di atas SUDAH dikoreksi FDR (Benjamini-Hochberg, alpha=0.05) lintas ticker yang divalidasi.")
    print(f"Ticker dengan edge nyata: {len(genuine_edge_tickers)} -> {genuine_edge_tickers}")
    print(f"Hasil disimpan: {args.output}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
