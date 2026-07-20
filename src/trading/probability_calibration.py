"""Kalibrasi probabilitas arah (Brier score) dari track record live.

Temuan baseline saat modul ini ditulis (2026-07-21, 7.250 prediksi ber-prob_up
yang sudah terevaluasi): SEMUA model punya Brier lebih buruk daripada baseline
base-rate (skill negatif), dan reliability bins menunjukkan bias pesimis
sistematis (prob_up rata-rata 0.34 padahal realisasi naik 51%). Artinya:
probabilitas mentah model BELUM layak dipakai sebagai derajat keyakinan.

Fungsi modul ini:
1. Mengukur -- compute_brier_by_model() / reliability_table() dari join
   predictions_log.csv (prob_up) x accuracy_log.csv (realisasi arah).
2. Menyediakan get_brier_weights() -- pembobot ensemble berbasis Brier skill
   dengan ambang sampel >=30 dan shrinkage ke bobot rata. Dengan track record
   sekarang (skill negatif semua) fungsi ini SENGAJA jatuh ke bobot rata --
   itu perilaku jujur, bukan bug.

PENTING (prinsip desain #3): mengganti pembobot produksi di
reliability_ensemble.get_reliability_weights dengan fungsi ini adalah
PERUBAHAN MODEL -- wajib revalidasi walk-forward dulu. Modul ini tidak
mengubah produksi apa pun; ia lapisan evaluasi.
"""

from __future__ import annotations

import os

import pandas as pd

PREDICTIONS_FILE = os.path.join("data", "tracking", "predictions_log.csv")
ACCURACY_FILE = os.path.join("data", "tracking", "accuracy_log.csv")
JOIN_KEYS = ["ticker", "model_name", "horizon_days", "prediction_purpose", "timestamp_prediction"]
MIN_EVALUATIONS_FOR_WEIGHT = 30


def load_probability_track_record(
    project_root: str = ".",
    prediction_purpose: str | None = "NEXT_DAY_DIRECTION",
) -> pd.DataFrame:
    """Join prediksi ber-probabilitas dengan realisasinya. Kolom inti:
    model_name, ticker, prob_up, y (1=NAIK terealisasi), horizon_days."""
    pred_path = os.path.join(project_root, PREDICTIONS_FILE)
    acc_path = os.path.join(project_root, ACCURACY_FILE)
    if not (os.path.exists(pred_path) and os.path.exists(acc_path)):
        return pd.DataFrame()
    p = pd.read_csv(pred_path, low_memory=False)
    a = pd.read_csv(acc_path, low_memory=False)
    for df in (p, a):
        df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce")
        df["timestamp_prediction"] = df["timestamp_prediction"].astype(str)
    p["prob_up"] = pd.to_numeric(p["prob_up"], errors="coerce")
    p = p[JOIN_KEYS + ["prob_up"]].dropna(subset=["prob_up"]).drop_duplicates(JOIN_KEYS)
    m = a.merge(p, on=JOIN_KEYS, how="inner")
    m = m[m["actual_direction"].notna()].copy()
    if prediction_purpose is not None:
        m = m[m["prediction_purpose"].astype(str) == prediction_purpose]
    if m.empty:
        return m
    m["y"] = m["actual_direction"].astype(str).str.upper().str.startswith("NAIK").astype(float)
    m["prob_up"] = m["prob_up"].clip(0.0, 1.0)
    return m


def compute_brier_by_model(track: pd.DataFrame) -> pd.DataFrame:
    """Per model: n, brier, brier baseline base-rate, skill = 1 - brier/baseline.

    skill > 0 berarti probabilitas model LEBIH baik daripada selalu menjawab
    base-rate; skill <= 0 berarti tidak lebih baik (kondisi saat ini).
    """
    if track.empty:
        return pd.DataFrame()
    rows = []
    for model, g in track.groupby("model_name"):
        base_rate = g["y"].mean()
        brier = float(((g["prob_up"] - g["y"]) ** 2).mean())
        brier_base = float(((base_rate - g["y"]) ** 2).mean())
        skill = 1.0 - brier / brier_base if brier_base > 0 else float("nan")
        rows.append(
            {
                "model_name": model,
                "n": int(len(g)),
                "brier": round(brier, 4),
                "brier_baserate": round(brier_base, 4),
                "brier_skill": round(skill, 4),
                "prob_up_mean": round(float(g["prob_up"].mean()), 4),
                "realized_up_rate": round(float(base_rate), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)


def reliability_table(track: pd.DataFrame, model_name: str | None = None, n_bins: int = 10) -> pd.DataFrame:
    """Tabel reliabilitas: per bin prob_up, bandingkan prob rata-rata vs
    frekuensi NAIK terealisasi. Terkalibrasi = kedua kolom mendekati."""
    df = track if model_name is None else track[track["model_name"] == model_name]
    if df.empty:
        return pd.DataFrame()
    bins = pd.cut(df["prob_up"], bins=[i / n_bins for i in range(n_bins + 1)], include_lowest=True)
    out = (
        df.groupby(bins, observed=True)
        .agg(n=("y", "size"), prob_up_mean=("prob_up", "mean"), realized_up_rate=("y", "mean"))
        .round(4)
        .reset_index()
        .rename(columns={"prob_up": "bin"})
    )
    out["gap"] = (out["realized_up_rate"] - out["prob_up_mean"]).round(4)
    return out


# ---------------------------------------------------------------------------
# Lapisan rekalibrasi (prototipe) + validasi walk-forward
# ---------------------------------------------------------------------------

def _logit(p):
    import numpy as np

    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_recalibrator(probs, outcomes, method: str = "platt"):
    """Fit pemetaan prob mentah -> prob terkalibrasi dari data historis.

    method:
      - "shift": koreksi intercept saja pada skala logit -- menggeser rata-rata
        prob ke base-rate aktual TANPA mengubah urutan/diskriminasi. Paling
        robust untuk sampel kecil; menyasar persis bias pesimis yang ditemukan.
      - "platt": logistic slope+intercept pada logit(p) (Platt scaling).
    Mengembalikan fungsi apply(probs)->probs terkalibrasi.
    """
    import numpy as np

    p = np.clip(np.asarray(probs, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(outcomes, dtype=float)
    if len(p) < 10 or len(set(y.tolist())) < 2:
        return lambda q: np.asarray(q, dtype=float)  # identitas: data tak cukup
    if method == "shift":
        # intercept b sehingga mean(sigmoid(logit(p)+b)) == mean(y)
        target = float(y.mean())
        z = _logit(p)
        lo, hi = -5.0, 5.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if float(_sigmoid(z + mid).mean()) < target:
                lo = mid
            else:
                hi = mid
        b = (lo + hi) / 2.0
        return lambda q: _sigmoid(_logit(q) + b)
    if method == "platt":
        from scipy.optimize import minimize

        z = _logit(p)

        def nll(theta):
            a, b = theta
            q = np.clip(_sigmoid(a * z + b), 1e-6, 1 - 1e-6)
            return -float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

        res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
        a, b = res.x
        return lambda q: _sigmoid(a * _logit(q) + b)
    raise ValueError(f"method tidak dikenal: {method}")


def walk_forward_recalibration_validation(
    track: pd.DataFrame,
    model_name: str,
    method: str = "shift",
    min_train: int = 500,
    step: int = 250,
) -> dict:
    """Uji walk-forward: apakah rekalibrasi memperbaiki Brier secara OOS?

    Kronologis (sort timestamp_prediction). Di tiap origin: fit rekalibrator +
    base-rate HANYA dari data sebelum origin (point-in-time), evaluasi pada
    `step` prediksi berikutnya. Keluaran: Brier OOS mentah vs terkalibrasi vs
    baseline base-rate train, per-fold dan agregat, plus p-value berpasangan
    satu-sisi (rekalibrasi < mentah).
    """
    import numpy as np
    from scipy import stats

    g = track[track["model_name"] == model_name].sort_values("timestamp_prediction")
    p = g["prob_up"].to_numpy(dtype=float)
    y = g["y"].to_numpy(dtype=float)
    n = len(g)
    folds = []
    raw_all, cal_all, base_all, y_all = [], [], [], []
    for start in range(min_train, n, step):
        end = min(start + step, n)
        if end - start < 20:
            break
        recal = fit_recalibrator(p[:start], y[:start], method=method)
        base_rate = float(y[:start].mean())
        q = np.asarray(recal(p[start:end]), dtype=float)
        raw = float(np.mean((p[start:end] - y[start:end]) ** 2))
        cal = float(np.mean((q - y[start:end]) ** 2))
        base = float(np.mean((base_rate - y[start:end]) ** 2))
        folds.append({"fold_start": start, "n": end - start, "brier_raw": raw, "brier_recal": cal, "brier_base": base})
        raw_all.append(p[start:end]); cal_all.append(q)
        base_all.append(np.full(end - start, base_rate)); y_all.append(y[start:end])
    if not folds:
        return {"model_name": model_name, "verdict": "DATA TIDAK CUKUP", "folds": pd.DataFrame()}
    fdf = pd.DataFrame(folds)
    yc = np.concatenate(y_all)
    brier_raw = float(np.mean((np.concatenate(raw_all) - yc) ** 2))
    brier_recal = float(np.mean((np.concatenate(cal_all) - yc) ** 2))
    brier_base = float(np.mean((np.concatenate(base_all) - yc) ** 2))
    diffs = (fdf["brier_raw"] - fdf["brier_recal"]).to_numpy()
    if len(diffs) >= 3 and np.std(diffs) > 0:
        t, p_two = stats.ttest_1samp(diffs, 0.0)
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    else:
        p_one = float("nan")
    beats_raw = brier_recal < brier_raw
    beats_base = brier_recal < brier_base
    verdict = (
        "LULUS -- rekalibrasi memperbaiki Brier OOS dan mengalahkan base-rate"
        if beats_raw and beats_base and (p_one == p_one and p_one < 0.05)
        else (
            "PERBAIKAN PARSIAL -- lebih baik dari mentah tapi belum mengalahkan base-rate"
            if beats_raw and not beats_base
            else "GAGAL -- tidak ada perbaikan OOS yang meyakinkan"
        )
    )
    return {
        "model_name": model_name,
        "method": method,
        "n_oos": int(len(yc)),
        "n_folds": int(len(fdf)),
        "brier_raw_oos": round(brier_raw, 4),
        "brier_recal_oos": round(brier_recal, 4),
        "brier_base_oos": round(brier_base, 4),
        "p_one_sided_recal_vs_raw": round(float(p_one), 4) if p_one == p_one else None,
        "verdict": verdict,
        "folds": fdf.round(4),
    }


def get_brier_weights(
    model_names,
    project_root: str = ".",
    prediction_purpose: str = "NEXT_DAY_DIRECTION",
    min_evaluations: int = MIN_EVALUATIONS_FOR_WEIGHT,
    shrinkage_evaluations: int = 100,
    track: pd.DataFrame | None = None,
) -> dict:
    """Bobot ensemble dari Brier skill, dengan dua guardrail kejujuran:

    1. Model dengan n < min_evaluations ATAU skill <= 0 mendapat skor 0 --
       kalau SEMUA model begitu (kondisi track record saat ini), hasilnya
       bobot rata (tidak pura-pura tahu model mana yang lebih baik).
    2. Shrinkage: bobot ditarik ke arah rata sebesar (1 - n/shrinkage_evaluations)
       supaya sampel kecil tidak menghasilkan bobot ekstrem.
    """
    models = [str(m).strip() for m in model_names if str(m).strip()]
    if not models:
        return {}
    equal = 1.0 / len(models)
    if track is None:
        track = load_probability_track_record(project_root, prediction_purpose)
    if track.empty:
        return {m: equal for m in models}
    summary = compute_brier_by_model(track[track["model_name"].isin(models)])
    if summary.empty:
        return {m: equal for m in models}
    scores = {}
    n_by_model = {}
    for m in models:
        row = summary[summary["model_name"] == m]
        if row.empty:
            scores[m] = 0.0
            n_by_model[m] = 0
            continue
        n = int(row.iloc[0]["n"])
        skill = float(row.iloc[0]["brier_skill"])
        n_by_model[m] = n
        scores[m] = max(skill, 0.0) if n >= min_evaluations else 0.0
    total = sum(scores.values())
    if total <= 0:
        return {m: equal for m in models}
    raw = {m: s / total for m, s in scores.items()}
    weights = {}
    for m in models:
        lam = min(n_by_model.get(m, 0) / max(shrinkage_evaluations, 1), 1.0)
        weights[m] = lam * raw[m] + (1.0 - lam) * equal
    norm = sum(weights.values())
    return {m: w / norm for m, w in weights.items()}
