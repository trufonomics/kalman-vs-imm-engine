"""
Calibration Fix Comparison — Brier Reliability Gap Investigation.

Tests three fixes for the IMM's probability miscalibration:

Fix 1: Permanent likelihood tempering (Grunwald 2012 "Safe Bayesian")
  - Standard Bayesian update concentrates too fast under misspecification
  - Cap the tempering exponent at 0.85 instead of ramping to 1.0
  - Trades sharpness for calibration

Fix 2: Isotonic regression recalibration (Zadrozny & Elkan 2002)
  - Post-hoc monotone mapping from raw probs to calibrated frequencies
  - Time-series cross-validation: train on first 70%, evaluate on last 30%
  - Preserves probability ordering

Fix 3: Both combined

Also tests expanded stagflation ground truth (adding 1970s episodes)
and reports BSS alongside raw Brier.

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/calibration_fix_comparison.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    FACTORS,
    FACTOR_INDEX,
    N_FACTORS,
    ANOMALY_THRESHOLD,
)
from heimdall.imm_tracker import (
    IMMBranchTracker,
    LEVEL_A_TPM,
)
from heimdall.regime_noise import REGIME_R_TABLE, REGIME_Q_TABLE
from heimdall.state_tpm import build_state_adjusted_tpm
from heimdall.cumulator import (
    StreamCumulator,
    CUMULATED_STREAMS,
    compute_gap_adjusted_R,
)
from heimdall.gas_noise import GASNoiseTracker
from heimdall.regime_diagnostics import (
    run_full_diagnostics,
    get_ground_truth,
)
from heimdall.recalibration import (
    fit_isotonic_calibrator,
    recalibrate_probabilities,
)

from backtests.full_history_backtest import (
    fetch_data,
    compute_warmup_norms,
    RollingNormalizer,
    BRANCHES,
    REGIME_CHECKPOINTS,
)


REGIME_MAP = {
    "soft_landing": "expansion",
    "stagflation": "stagflation",
    "recession": "contraction",
}

PROB_KEY_FOR_REGIME = {
    "expansion": "soft_landing",
    "contraction": "recession",
    "stagflation": "stagflation",
}

CHECKPOINT_REGIME_TO_LABEL = {
    "soft_landing": "expansion",
    "recession": "contraction",
    "stagflation": "stagflation",
}


def run_engine(
    data: pd.DataFrame,
    engine_name: str,
    temper_ceiling: float = 1.0,
) -> dict:
    """Run full shipping stack (P0+P2b+P4+P5) with configurable tempering.

    Returns daily_log with regime probabilities at each timestep.
    """
    col_to_kalman = {}
    for col in data.columns:
        parts = col.split("_", 1)
        if len(parts) == 2:
            col_to_kalman[col] = parts[1]

    estimator = EconomicStateEstimator()
    imm = IMMBranchTracker()
    imm.initialize_branches(BRANCHES, baseline=estimator)

    norms = compute_warmup_norms(data, warmup_days=180)
    normalizer = RollingNormalizer(halflife_days=120)
    for col, n in norms.items():
        normalizer.initialize(col, n["mean"], n["std"])

    cumulator = StreamCumulator()
    gas_tracker = GASNoiseTracker()
    for stream_key, (_, base_R) in estimator.stream_registry.items():
        gas_tracker.initialize(stream_key, base_R)

    daily_log = []
    total_log_lik = 0.0
    anomaly_count = 0
    total_updates = 0
    warmup_end_idx = 180

    for day_idx, (date, row) in enumerate(data.iterrows()):
        is_warmup = day_idx < warmup_end_idx

        today_updates = []
        for col in data.columns:
            val = row[col]
            if pd.isna(val):
                continue
            kalman_key = col_to_kalman.get(col)
            if not kalman_key:
                continue
            if col not in normalizer.initialized:
                stream_vals = data[col].dropna()
                if len(stream_vals) >= 5:
                    mean = float(stream_vals.iloc[:50].mean())
                    std = float(stream_vals.iloc[:50].std())
                    normalizer.initialize(col, mean, max(std, 0.1))
                else:
                    continue
            today_updates.append((col, kalman_key, val))

        if not today_updates:
            state = estimator.get_state()
            for stream_key in CUMULATED_STREAMS:
                if stream_key in estimator.stream_registry:
                    H_row, _ = estimator.stream_registry[stream_key]
                    cumulator.accumulate(stream_key, H_row, state.mean)
            continue

        # State-dependent TPM
        if day_idx % 7 == 0:
            state = estimator.get_state()
            factor_values = {
                name: float(state.mean[i]) for i, name in enumerate(FACTORS)
            }
            adjusted_tpm = build_state_adjusted_tpm(LEVEL_A_TPM, factor_values)
            imm.tpm = adjusted_tpm

        probs = imm.get_probabilities()

        # Regime-dependent Q
        q_scale = 0.0
        for branch_id, prob in probs.items():
            regime = REGIME_MAP.get(branch_id, "expansion")
            q_scale += prob * REGIME_Q_TABLE.get(regime, 1.0)
        base_Q = estimator._build_Q()
        estimator.Q = base_Q * q_scale
        for branch in imm.branches:
            if branch.estimator:
                regime = REGIME_MAP.get(branch.branch_id, "expansion")
                branch.estimator.Q = base_Q * REGIME_Q_TABLE.get(regime, 1.0)

        estimator.predict()
        imm.predict()

        # Accumulate
        state = estimator.get_state()
        for stream_key in CUMULATED_STREAMS:
            if stream_key in estimator.stream_registry:
                H_row, _ = estimator.stream_registry[stream_key]
                cumulator.accumulate(stream_key, H_row, state.mean)

        for col, kalman_key, val in today_updates:
            z = normalizer.normalize(col, val)

            if kalman_key not in estimator.stream_registry:
                continue

            H_row, base_R = estimator.stream_registry[kalman_key]

            effective_R = base_R
            gas_R = gas_tracker.get_R(kalman_key)
            if gas_R is not None:
                effective_R = gas_R

            gap_days = 1
            if cumulator.should_cumulate(kalman_key):
                cumulated = cumulator.get_cumulated_prediction(kalman_key)
                if cumulated is not None:
                    _, gap_days = cumulated
                    effective_R = compute_gap_adjusted_R(effective_R, gap_days, kalman_key)

            r_blend = 0.0
            for branch_id, prob in probs.items():
                regime = REGIME_MAP.get(branch_id, "expansion")
                r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                r_blend += prob * r_mult
            effective_R = effective_R * r_blend

            use_cumulated_pred = (
                cumulator.should_cumulate(kalman_key)
                and cumulator.get_cumulated_prediction(kalman_key) is not None
            )

            if use_cumulated_pred:
                mean_pred, _ = cumulator.get_cumulated_prediction(kalman_key)
                innovation = z - mean_pred
            else:
                innovation = z - float(H_row @ estimator.x)

            estimator.stream_registry[kalman_key] = (H_row, effective_R)
            S = float(H_row @ estimator.P @ H_row.T + effective_R)
            K = estimator.P @ H_row.T / S
            innovation_zscore = innovation / np.sqrt(S)

            estimator.x = estimator.x + K * innovation
            I_KH = np.eye(N_FACTORS) - np.outer(K, H_row)
            estimator.P = I_KH @ estimator.P @ I_KH.T + np.outer(K, K) * effective_R

            estimator.stream_registry[kalman_key] = (H_row, base_R)
            gas_tracker.update(kalman_key, innovation, S)

            if cumulator.should_cumulate(kalman_key):
                cumulator.reset(kalman_key)

            # IMM update with CUSTOM tempering ceiling
            for branch in imm.branches:
                if branch.estimator and kalman_key in branch.estimator.stream_registry:
                    regime = REGIME_MAP.get(branch.branch_id, "expansion")
                    r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                    bH, bR = branch.estimator.stream_registry[kalman_key]
                    branch.estimator.stream_registry[kalman_key] = (bH, bR * r_mult)

            # Manually call update with custom tempering
            _imm_update_with_ceiling(imm, kalman_key, z, temper_ceiling)

            for branch in imm.branches:
                if branch.estimator and kalman_key in branch.estimator.stream_registry:
                    bH, bR_scaled = branch.estimator.stream_registry[kalman_key]
                    regime = REGIME_MAP.get(branch.branch_id, "expansion")
                    r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                    branch.estimator.stream_registry[kalman_key] = (bH, bR_scaled / r_mult)

            total_updates += 1
            if not is_warmup:
                S_ll = float(H_row @ estimator.P @ H_row.T + base_R)
                ll = -0.5 * (innovation_zscore**2 + np.log(2 * np.pi * S_ll))
                total_log_lik += ll
                if abs(float(innovation_zscore)) > ANOMALY_THRESHOLD:
                    anomaly_count += 1

        probs = imm.get_probabilities()
        state = estimator.get_state()

        if day_idx % 7 == 0 or day_idx == len(data) - 1:
            factors = {name: round(float(state.mean[i]), 4) for i, name in enumerate(FACTORS)}
            daily_log.append({
                "date": str(date.date()),
                "day": day_idx,
                "factors": factors,
                "probabilities": {b: round(p, 4) for b, p in probs.items()},
            })

    return {
        "name": engine_name,
        "daily_log": daily_log,
        "total_log_likelihood": total_log_lik,
        "anomaly_count": anomaly_count,
        "total_updates": total_updates,
        "temper_ceiling": temper_ceiling,
    }


def _imm_update_with_ceiling(
    imm: IMMBranchTracker,
    stream_key: str,
    value: float,
    temper_ceiling: float,
):
    """IMM update with configurable tempering ceiling.

    Replaces the standard imm.update() to allow permanent tempering.
    The key change: exponent ramps to temper_ceiling instead of 1.0.
    """
    from scipy.stats import norm

    likelihoods = []
    registry = None
    for branch in imm.branches:
        if branch.estimator and stream_key in branch.estimator.stream_registry:
            registry = branch.estimator.stream_registry
            break
    if registry is None:
        return

    H_row, R = registry[stream_key]

    for branch in imm.branches:
        if branch.estimator is None:
            likelihoods.append(1e-10)
            continue

        est = branch.estimator
        S_j = float(H_row @ est.P @ H_row.T + R)
        if S_j <= 0:
            S_j = R

        predicted = float(H_row @ est.x)
        innovation = value - predicted

        likelihood = float(norm.pdf(innovation, loc=0, scale=np.sqrt(S_j)))
        likelihood = max(likelihood, 1e-10)
        likelihoods.append(likelihood)

        # Per-branch Kalman update
        K = est.P @ H_row.T / S_j
        est.x = est.x + K * innovation
        I_KH = np.eye(N_FACTORS) - np.outer(K, H_row)
        est.P = I_KH @ est.P @ I_KH.T + np.outer(K, K) * R

    # Tempering with configurable ceiling
    imm._update_count += 1
    ramp = min(imm._update_count / imm.TEMPER_HORIZON, 1.0)
    exponent = imm.TEMPER_FLOOR + (temper_ceiling - imm.TEMPER_FLOOR) * ramp
    tempered = [l ** exponent for l in likelihoods]

    # Bayes' rule
    weighted = [l * b.probability for l, b in zip(tempered, imm.branches)]
    total = sum(weighted)

    if total > 0:
        for i, branch in enumerate(imm.branches):
            branch.probability = weighted[i] / total

    imm._clamp_probabilities()
    imm._normalize_probabilities()


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 90)
    print("  CALIBRATION FIX COMPARISON")
    print("  Investigating Brier reliability gap with 3 fixes")
    print("=" * 90)

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        print(f"\nLoading cached data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    # --- Config A: Full Stack as-is (ceiling=1.0) ---
    print("\n--- A: Full Stack (temper ceiling = 1.0) ---")
    eng_a = run_engine(data, "Full Stack", temper_ceiling=1.0)
    print(f"  LL = {eng_a['total_log_likelihood']:.0f}")

    # --- Config B: Permanent tempering (ceiling=0.85) ---
    print("\n--- B: Full Stack + Safe Bayesian (ceiling = 0.85) ---")
    eng_b = run_engine(data, "+Temper 0.85", temper_ceiling=0.85)
    print(f"  LL = {eng_b['total_log_likelihood']:.0f}")

    # --- Config C: Scan tempering values ---
    temper_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    temper_results = {}
    print("\n--- Tempering scan ---")
    for tc in temper_values:
        eng = run_engine(data, f"tc={tc}", temper_ceiling=tc)
        diag = run_full_diagnostics(eng["daily_log"], REGIME_CHECKPOINTS)
        mean_reliability = np.mean([
            diag["brier"][r].reliability
            for r in ["contraction", "stagflation", "expansion"]
        ])
        mean_bss = np.mean([
            diag["brier"][r].brier_skill_score
            for r in ["contraction", "stagflation", "expansion"]
        ])
        contraction_auc = diag["roc"]["contraction"].auc
        detected = sum(1 for dl in diag["detection_lag"] if dl.detection_lag_weeks is not None)
        sharp = diag["sharpness"].mean_max_prob
        temper_results[tc] = {
            "mean_reliability": mean_reliability,
            "mean_bss": mean_bss,
            "contraction_auc": contraction_auc,
            "sharpness": sharp,
            "detected": detected,
            "ll": eng["total_log_likelihood"],
        }
        print(f"  tc={tc:.2f}: rel={mean_reliability:.4f} BSS={mean_bss:+.4f} "
              f"AUC={contraction_auc:.4f} sharp={sharp:.4f} det={detected}/11 "
              f"LL={eng['total_log_likelihood']:.0f}")

    # Find optimal tempering
    best_tc = min(temper_results, key=lambda tc: temper_results[tc]["mean_reliability"])
    print(f"\n  Best tempering ceiling: {best_tc:.2f} "
          f"(reliability={temper_results[best_tc]['mean_reliability']:.4f})")

    # --- Config D: Isotonic recalibration (time-series CV) ---
    print("\n--- D: Isotonic Recalibration (train first 70%, eval last 30%) ---")

    # Use full stack as-is for raw probabilities
    daily_log_a = eng_a["daily_log"]
    n_total = len(daily_log_a)
    n_train = int(n_total * 0.70)

    train_log = daily_log_a[:n_train]
    test_log = daily_log_a[n_train:]

    train_cutoff = train_log[-1]["date"]
    test_start = test_log[0]["date"]
    print(f"  Train: {train_log[0]['date']} → {train_cutoff} ({n_train} entries)")
    print(f"  Test:  {test_start} → {test_log[-1]['date']} ({len(test_log)} entries)")

    # Fit calibrators on training data
    calibrators = {}
    for regime in ["contraction", "stagflation", "expansion"]:
        prob_key = PROB_KEY_FOR_REGIME[regime]
        forecasts = [e["probabilities"].get(prob_key, 0.0) for e in train_log]
        observations = [1 if get_ground_truth(e["date"]) == regime else 0 for e in train_log]
        calibrators[prob_key] = fit_isotonic_calibrator(forecasts, observations, regime)
        n_pos = sum(observations)
        print(f"  {regime}: {n_pos} positive examples in training set")

    # Apply to test set
    recalibrated_log = []
    for entry in test_log:
        new_probs = recalibrate_probabilities(entry["probabilities"], calibrators)
        recalibrated_log.append({
            **entry,
            "probabilities": {k: round(v, 4) for k, v in new_probs.items()},
        })

    # --- Config E: Best tempering + isotonic ---
    print(f"\n--- E: Temper {best_tc:.2f} + Isotonic ---")
    eng_best = run_engine(data, f"Temper {best_tc:.2f}", temper_ceiling=best_tc)
    daily_log_best = eng_best["daily_log"]

    train_log_best = daily_log_best[:n_train]
    test_log_best = daily_log_best[n_train:]

    calibrators_best = {}
    for regime in ["contraction", "stagflation", "expansion"]:
        prob_key = PROB_KEY_FOR_REGIME[regime]
        forecasts = [e["probabilities"].get(prob_key, 0.0) for e in train_log_best]
        observations = [1 if get_ground_truth(e["date"]) == regime else 0 for e in train_log_best]
        calibrators_best[prob_key] = fit_isotonic_calibrator(forecasts, observations, regime)

    recalibrated_log_best = []
    for entry in test_log_best:
        new_probs = recalibrate_probabilities(entry["probabilities"], calibrators_best)
        recalibrated_log_best.append({
            **entry,
            "probabilities": {k: round(v, 4) for k, v in new_probs.items()},
        })

    # ── Compare all configs on TEST SET ONLY ──────────────────────────
    print("\n" + "=" * 90)
    print(f"  RESULTS ON TEST SET ({test_start} → {test_log[-1]['date']})")
    print("=" * 90)

    configs = [
        ("A: Full Stack", test_log),
        (f"B: +Temper {best_tc:.2f}", test_log_best),
        ("D: +Isotonic", recalibrated_log),
        (f"E: Temper+Iso", recalibrated_log_best),
    ]

    # Filter checkpoints to test period only
    test_checkpoints = [
        cp for cp in REGIME_CHECKPOINTS
        if cp[1] >= test_start
    ]

    for regime in ["contraction", "stagflation", "expansion"]:
        print(f"\n  Regime: {regime}")
        header = (f"  {'Config':<22s} {'Brier':>7s} {'Reliab':>7s} "
                  f"{'Resol':>7s} {'BSS':>8s} {'AUC':>7s}")
        print(header)
        print("  " + "-" * 65)

        for name, log in configs:
            diag = run_full_diagnostics(log, test_checkpoints)
            b = diag["brier"].get(regime)
            r = diag["roc"].get(regime)
            if b and r:
                print(f"  {name:<22s} {b.brier_score:>7.4f} {b.reliability:>7.4f} "
                      f"{b.resolution:>7.4f} {b.brier_skill_score:>+8.4f} {r.auc:>7.4f}")

    # Sharpness comparison
    print(f"\n  Sharpness:")
    header = f"  {'Config':<22s} {'Mean max(P)':>12s} {'P>0.8':>7s} {'P>0.5':>7s}"
    print(header)
    print("  " + "-" * 50)

    for name, log in configs:
        diag = run_full_diagnostics(log, test_checkpoints)
        s = diag["sharpness"]
        print(f"  {name:<22s} {s.mean_max_prob:>12.4f} {s.frac_above_80:>6.1%} {s.frac_above_50:>6.1%}")

    # Detection lag
    print(f"\n  Detection Lag (events in test period):")
    for name, log in configs:
        diag = run_full_diagnostics(log, test_checkpoints)
        print(f"\n  {name}:")
        for dl in diag["detection_lag"]:
            lag_str = f"{dl.detection_lag_weeks:.1f}w" if dl.detection_lag_weeks is not None else "MISSED"
            print(f"    {lag_str:>8s}  peak={dl.peak_probability:.2f}  {dl.event}")

    # ── Save results ──────────────────────────────────────────────────
    results_dir = Path(__file__).parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "temper_scan": temper_results,
        "best_tempering_ceiling": best_tc,
        "train_period": f"{train_log[0]['date']} → {train_cutoff}",
        "test_period": f"{test_start} → {test_log[-1]['date']}",
    }

    for name, log in configs:
        diag = run_full_diagnostics(log, test_checkpoints)
        save_data[name] = {
            "brier": {
                regime: {
                    "score": diag["brier"][regime].brier_score,
                    "reliability": diag["brier"][regime].reliability,
                    "resolution": diag["brier"][regime].resolution,
                    "bss": diag["brier"][regime].brier_skill_score,
                }
                for regime in ["contraction", "stagflation", "expansion"]
            },
            "roc_auc": {
                regime: diag["roc"][regime].auc
                for regime in ["contraction", "stagflation", "expansion"]
            },
            "sharpness": diag["sharpness"].mean_max_prob,
        }

    out_path = results_dir / "calibration_fix_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
