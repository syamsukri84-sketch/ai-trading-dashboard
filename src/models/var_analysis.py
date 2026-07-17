"""
Analisis Value at Risk (VaR) multi-metodologi.

Melengkapi VaR parametric-normal tunggal yang sudah ada di GARCHModel
(satu metode, confidence 95% saja, asumsi mean=0) dengan metode
tambahan -- historical simulation, Cornish-Fisher, EWMA, Monte Carlo
bootstrap -- dan pemilihan metode yang DIREKOMENDASIKAN per confidence
level. Temuan ini berasal dari analisis empiris LPPF/MDS Retailing
(Jul 2026, lihat sesi chat terkait) memakai data harga asli proyek ini
(data/raw/LPPF_raw.csv, bersumber yfinance lewat
src/data_pipeline/auto_updater.py):

- Di confidence 95%, Cornish-Fisher konsisten dengan metode lain dan
  jadi koreksi yang wajar untuk skew/kurtosis ringan-sedang.
- Di confidence 99%, suku koreksi kurtosis pada ekspansi Cornish-Fisher
  (~(z^3-3z)*K/24) membesar SECARA KUBIK terhadap z. Begitu kurtosis
  riil (bukan hasil sampel kecil yang noisy) berada di kisaran
  moderat-tinggi, suku ini bisa melebih-lebihkan kerugian ekor jauh di
  atas historical simulation / Monte Carlo bootstrap -- yang tidak
  bergantung pada aproksimasi polinomial dan lebih bisa dipercaya di
  confidence sedalam ini. Konsisten dengan kritik metodologis di
  literatur (mis. Jaschke 2001; Maillard 2012, "A user's guide to the
  Cornish-Fisher expansion").

Modul ini murni fungsi (bukan class) -- tidak butuh training/state
seperti GARCHModel, cukup diberi deret return harian.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_WINDOW = 252  # ~1 tahun bursa -- jendela standar praktik VaR (RiskMetrics)
MIN_OBS_FLOOR = 10  # di bawah ini, VaR tidak dihitung sama sekali (terlalu tidak andal)
MIN_OBS_RELIABLE = 60  # di bawah ini, tandai data_terbatas=True (bukan cuma CF yang meragukan)
EWMA_LAMBDA = 0.94  # konvensi RiskMetrics
MC_SIMULATIONS = 100_000
CF_UNSTABLE_CONF_THRESHOLD = 0.99  # confidence >= ini: JANGAN pakai CF sbg estimasi utama


def _ewma_last_vol(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """Volatilitas EWMA (RiskMetrics) -- lebih berat ke observasi terbaru daripada std biasa."""
    n = len(returns)
    if n < 2:
        return float(np.std(returns)) if n else 0.0
    seed_n = min(20, n)
    var = float(np.var(returns[:seed_n])) if seed_n > 1 else float(returns[0] ** 2)
    for i in range(1, n):
        var = lam * var + (1 - lam) * float(returns[i - 1]) ** 2
    return float(np.sqrt(var))


def _cornish_fisher_z(z: float, skew: float, excess_kurt: float) -> float:
    """Ekspansi Cornish-Fisher: koreksi kuantil normal pakai skewness & excess kurtosis sampel."""
    return (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * excess_kurt / 24
        - (2 * z**3 - 5 * z) * (skew**2) / 36
    )


def compute_var_suite(
    returns: pd.Series,
    confidence_levels=(0.95, 0.99),
    window: int = DEFAULT_WINDOW,
    horizon_days: int = 1,
    mc_simulations: int = MC_SIMULATIONS,
    seed: int = 42,
) -> dict:
    """
    Hitung VaR (loss, dalam persen positif terhadap harga) dengan 5
    metodologi -- historical simulation, parametric normal,
    Cornish-Fisher, EWMA, Monte Carlo bootstrap -- lalu pilih metode
    yang DIREKOMENDASIKAN per confidence level (lihat docstring modul).

    Parameters
    ----------
    returns : pd.Series
        Return harian (mis. df["close"].pct_change().dropna()), urutan
        KRONOLOGIS (lama -> baru). Hanya `window` observasi TERAKHIR
        yang dipakai, supaya mencerminkan rezim volatilitas saat ini
        (bukan rezim beberapa tahun lalu yang mungkin sudah tidak
        relevan) -- konsisten dengan praktik standar VaR 1 tahun bursa.
    confidence_levels : tuple[float]
        Level confidence yang dihitung, mis. (0.95, 0.99).
    window : int
        Jumlah observasi trailing yang dipakai (default 252 ~ 1 tahun bursa).
    horizon_days : int
        Skala horizon lewat aturan akar-waktu (sqrt(horizon_days)).
        CATATAN: aturan akar-waktu adalah aproksimasi (asumsi return
        i.i.d.), bukan simulasi multi-hari sungguhan -- lihat
        `catatan_horizon` di hasil kalau horizon_days > 1.

    Returns
    -------
    dict dengan kunci: n_obs, window_diminta, data_terbatas, mean_pct,
    std_pct, skewness, excess_kurtosis, per_confidence (dict per
    confidence -> historical_pct/parametric_pct/cornish_fisher_pct/
    ewma_pct/mc_normal_pct/mc_bootstrap_pct/recommended_pct/
    recommended_method). Kalau data tidak cukup, dict hanya berisi
    key "error" + "n_obs".
    """
    if returns is None or len(returns) == 0:
        return {"error": "returns kosong", "n_obs": 0}

    r = returns.dropna().tail(window).to_numpy(dtype=float)
    n = len(r)
    if n < MIN_OBS_FLOOR:
        return {"error": f"Observasi terlalu sedikit (n={n}) -- VaR tidak dihitung.", "n_obs": n}

    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1)) if n > 1 else 0.0
    skew = float(stats.skew(r)) if n > 2 and std > 0 else 0.0
    kurt = float(stats.kurtosis(r, fisher=True)) if n > 3 and std > 0 else 0.0

    result = {
        "n_obs": n,
        "window_diminta": window,
        "data_terbatas": n < MIN_OBS_RELIABLE,
        "mean_pct": mean * 100,
        "std_pct": std * 100,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "per_confidence": {},
    }

    if std == 0:
        # Semua return identik (data flat/tidak wajar) -- VaR = 0 utk semua metode, tidak dipaksakan.
        for conf in confidence_levels:
            result["per_confidence"][conf] = {
                "historical_pct": 0.0, "parametric_pct": 0.0, "cornish_fisher_pct": 0.0,
                "ewma_pct": 0.0, "mc_normal_pct": 0.0, "mc_bootstrap_pct": 0.0,
                "recommended_pct": 0.0, "recommended_method": "n/a (std=0)",
            }
        result["catatan_data"] = "Standar deviasi return = 0 -- data harga kemungkinan flat/bermasalah."
        return result

    ewma_vol = _ewma_last_vol(r)
    rng = np.random.default_rng(seed)
    scale = float(np.sqrt(horizon_days))

    for conf in confidence_levels:
        z = float(stats.norm.ppf(1 - conf))  # negatif
        historical = -np.percentile(r, (1 - conf) * 100)
        parametric = -(mean + z * std)
        z_cf = _cornish_fisher_z(z, skew, kurt)
        cornish_fisher = -(mean + z_cf * std)
        ewma = -(z * ewma_vol)
        sims_normal = rng.normal(mean, std, mc_simulations)
        mc_normal = -np.percentile(sims_normal, (1 - conf) * 100)
        sims_boot = rng.choice(r, size=mc_simulations, replace=True)
        mc_bootstrap = -np.percentile(sims_boot, (1 - conf) * 100)

        if conf < CF_UNSTABLE_CONF_THRESHOLD:
            recommended_value = cornish_fisher
            recommended_method = "cornish_fisher"
        else:
            recommended_value = float(np.mean([historical, mc_bootstrap]))
            recommended_method = "historical_mc_bootstrap_avg"

        result["per_confidence"][conf] = {
            "historical_pct": float(historical * 100 * scale),
            "parametric_pct": float(parametric * 100 * scale),
            "cornish_fisher_pct": float(cornish_fisher * 100 * scale),
            "ewma_pct": float(ewma * 100 * scale),
            "mc_normal_pct": float(mc_normal * 100 * scale),
            "mc_bootstrap_pct": float(mc_bootstrap * 100 * scale),
            "recommended_pct": float(recommended_value * 100 * scale),
            "recommended_method": recommended_method,
        }

    if horizon_days > 1:
        result["catatan_horizon"] = (
            f"Skala {horizon_days} hari pakai aturan akar-waktu (asumsi return i.i.d.) -- "
            "aproksimasi, bukan simulasi multi-hari sungguhan."
        )
    if result["data_terbatas"]:
        result["catatan_data"] = (
            f"Hanya {n} observasi (< {MIN_OBS_RELIABLE}) -- skew/kurtosis sampel kemungkinan "
            "noisy, SEMUA metode di bawah kurang bisa diandalkan, bukan cuma Cornish-Fisher."
        )
    return result


def compute_var_from_price_df(
    df: pd.DataFrame,
    confidence_levels=(0.95, 0.99),
    window: int = DEFAULT_WINDOW,
    horizon_days: int = 1,
    mc_simulations: int = MC_SIMULATIONS,
    seed: int = 42,
) -> dict:
    """Wrapper: hitung compute_var_suite langsung dari df OHLCV (kolom 'close'), urutan kronologis."""
    returns = df["close"].pct_change().dropna()
    return compute_var_suite(
        returns,
        confidence_levels=confidence_levels,
        window=window,
        horizon_days=horizon_days,
        mc_simulations=mc_simulations,
        seed=seed,
    )
