"""Test src/trading/probability_calibration.py -- data sintetis, tanpa file produksi."""

import os
import random

import pandas as pd

from src.trading.probability_calibration import (
    compute_brier_by_model,
    get_brier_weights,
    load_probability_track_record,
    reliability_table,
)


def _write_logs(tmp_path, rows_pred, rows_acc):
    d = tmp_path / "data" / "tracking"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_pred).to_csv(d / "predictions_log.csv", index=False)
    pd.DataFrame(rows_acc).to_csv(d / "accuracy_log.csv", index=False)


def _synthetic_tracks(tmp_path, n=200, seed=11):
    """GOOD = diskriminatif & terkalibrasi (prob 0.8/0.2 mengikuti sinyal yang
    benar-benar menggerakkan outcome); BAD = probabilitas terbalik."""
    rng = random.Random(seed)
    preds, accs = [], []
    for i in range(n):
        ts = f"2026-01-01 09:{i:04d}"
        signal_up = rng.random() < 0.5
        p_true = 0.8 if signal_up else 0.2
        up = rng.random() < p_true
        actual = "NAIK" if up else "TURUN"
        for model, prob in (("GOOD", p_true), ("BAD", 1.0 - p_true)):
            key = dict(
                ticker="SYN",
                model_name=model,
                horizon_days=1,
                prediction_purpose="NEXT_DAY_DIRECTION",
                timestamp_prediction=ts,
            )
            preds.append({**key, "prob_up": prob})
            accs.append({**key, "actual_direction": actual, "direction_correct": up})
    _write_logs(tmp_path, preds, accs)


def test_load_join_dan_brier_ordering(tmp_path):
    _synthetic_tracks(tmp_path)
    track = load_probability_track_record(str(tmp_path))
    assert len(track) == 400 and set(track["model_name"]) == {"GOOD", "BAD"}
    summary = compute_brier_by_model(track)
    good = summary[summary["model_name"] == "GOOD"].iloc[0]
    bad = summary[summary["model_name"] == "BAD"].iloc[0]
    assert good["brier"] < bad["brier"]
    assert good["brier_skill"] > 0 > bad["brier_skill"]


def test_reliability_table_gap_kecil_untuk_model_terkalibrasi(tmp_path):
    _synthetic_tracks(tmp_path, n=400)
    track = load_probability_track_record(str(tmp_path))
    rel = reliability_table(track, "GOOD")
    assert not rel.empty
    assert rel["gap"].abs().max() < 0.12  # prob 0.8/0.2 vs realisasi ~0.8/0.2


def test_brier_weights_memihak_model_berskill(tmp_path):
    _synthetic_tracks(tmp_path, n=300)
    w = get_brier_weights(["GOOD", "BAD"], str(tmp_path), shrinkage_evaluations=100)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["GOOD"] > 0.9  # BAD skill negatif -> skor 0; GOOD n>=100 -> tanpa shrink


def test_brier_weights_jujur_saat_semua_tanpa_skill(tmp_path):
    # kedua model diberi probabilitas terbalik -> skill negatif semua -> bobot rata
    rng = random.Random(3)
    preds, accs = [], []
    for i in range(80):
        ts = f"2026-02-01 10:{i:04d}"
        up = rng.random() < 0.5
        actual = "NAIK" if up else "TURUN"
        for model in ("M1", "M2"):
            key = dict(ticker="SYN", model_name=model, horizon_days=1,
                       prediction_purpose="NEXT_DAY_DIRECTION", timestamp_prediction=ts)
            preds.append({**key, "prob_up": 0.1 if up else 0.9})
            accs.append({**key, "actual_direction": actual, "direction_correct": False})
    _write_logs(tmp_path, preds, accs)
    w = get_brier_weights(["M1", "M2"], str(tmp_path))
    assert w == {"M1": 0.5, "M2": 0.5}


def test_brier_weights_sampel_kecil_ke_bobot_rata(tmp_path):
    _synthetic_tracks(tmp_path, n=10)  # < min_evaluations 30
    w = get_brier_weights(["GOOD", "BAD"], str(tmp_path))
    assert w == {"GOOD": 0.5, "BAD": 0.5}


def test_track_record_kosong_bobot_rata(tmp_path):
    w = get_brier_weights(["A", "B", "C"], str(tmp_path))
    assert all(abs(v - 1 / 3) < 1e-9 for v in w.values())


# ---------------- rekalibrasi ----------------

def _biased_track(n=2000, seed=5, bias=-0.15):
    """Model diskriminatif tapi bias pesimis: prob dilaporkan = p_true + bias."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        signal = rng.random()
        p_true = 0.35 + 0.4 * signal  # 0.35..0.75
        up = rng.random() < p_true
        rows.append(
            dict(
                model_name="BIASED",
                ticker="SYN",
                horizon_days=1,
                prediction_purpose="NEXT_DAY_DIRECTION",
                timestamp_prediction=f"2026-01-01 09:{i:05d}",
                prob_up=max(min(p_true + bias, 0.99), 0.01),
                y=1.0 if up else 0.0,
            )
        )
    return pd.DataFrame(rows)


def test_fit_recalibrator_shift_menghilangkan_bias():
    from src.trading.probability_calibration import fit_recalibrator

    track = _biased_track(n=1500)
    recal = fit_recalibrator(track["prob_up"], track["y"], method="shift")
    q = recal(track["prob_up"].to_numpy())
    # rata-rata prob terkalibrasi mendekati base-rate aktual
    assert abs(float(q.mean()) - float(track["y"].mean())) < 0.02
    # urutan dipertahankan (shift monotonik)
    import numpy as np
    p = track["prob_up"].to_numpy()
    order = np.argsort(p)
    assert (np.diff(q[order]) >= -1e-12).all()


def test_fit_recalibrator_data_tak_cukup_identitas():
    from src.trading.probability_calibration import fit_recalibrator

    recal = fit_recalibrator([0.4, 0.5], [1.0, 1.0], method="shift")
    out = recal([0.3, 0.6])
    assert list(out) == [0.3, 0.6]


def test_walk_forward_recalibration_lulus_pada_model_bias_diskriminatif():
    from src.trading.probability_calibration import walk_forward_recalibration_validation

    track = _biased_track(n=2500)
    res = walk_forward_recalibration_validation(track, "BIASED", method="shift", min_train=500, step=400)
    assert res["n_folds"] >= 3
    assert res["brier_recal_oos"] < res["brier_raw_oos"]
    assert res["brier_recal_oos"] < res["brier_base_oos"]
    assert res["verdict"].startswith("LULUS")


def test_walk_forward_recalibration_data_kurang():
    from src.trading.probability_calibration import walk_forward_recalibration_validation

    track = _biased_track(n=100)
    res = walk_forward_recalibration_validation(track, "BIASED", min_train=500, step=250)
    assert res["verdict"] == "DATA TIDAK CUKUP"
