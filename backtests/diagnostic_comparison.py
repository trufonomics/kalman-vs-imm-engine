"""
Proper Regime Diagnostics — Replacing Ljung-Box with Literature-Correct Tests.

Kim & Nelson (1999) prove innovations from Markov-switching models are
mixture distributions by construction — Ljung-Box fails for fundamental
mathematical reasons, not tuning failures.

Hashimzade et al. (2024), the closest published IMM macro application,
do NOT report Ljung-Box.  They evaluate by regime classification
against NBER dates.

This script runs:
  1. Brier Score decomposition (Murphy 1973) — calibration + resolution
  2. ROC/AUC (Berge & Jordà 2011) — threshold-independent classification
  3. Diebold-Mariano test (1995) — IMM vs single-regime significance
  4. PIT histogram (Diebold, Gunther & Tay 1998) — replaces Ljung-Box
  5. Sharpness (Gneiting et al. 2007) — paired with calibration
  6. Detection lag — per-checkpoint time to detection

Three configs compared:
  A = Baseline (no improvements)
  B = P0+P2b (regime noise + state TPM)
  C = P0+P2b+P4+P5 (full shipping stack)

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/diagnostic_comparison.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict

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
    brier_decomposition,
    roc_auc,
    diebold_mariano,
    pit_test,
    sharpness,
    detection_lag,
    get_ground_truth,
    run_full_diagnostics,
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


@dataclass
class EngineOutput:
    """Full backtest output with everything diagnostics need."""
    name: str
    daily_log: list[dict]            # regime probabilities per week
    innovation_log: dict[str, list[float]]
    variance_log: dict[str, list[float]]   # S_t values for PIT
    log_likelihoods: dict[str, list[float]]
    total_log_likelihood: float
    per_obs_loss: list[float]        # per-observation neg log-lik for DM test
    anomaly_count: int
    total_updates: int


def run_engine(
    data: pd.DataFrame,
    engine_name: str,
    use_regime_noise: bool = False,
    use_state_tpm: bool = False,
    use_cumulator: bool = False,
    use_gas: bool = False,
) -> EngineOutput:
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

    cumulator = StreamCumulator() if use_cumulator else None

    gas_tracker = None
    if use_gas:
        gas_tracker = GASNoiseTracker()
        for stream_key, (_, base_R) in estimator.stream_registry.items():
            gas_tracker.initialize(stream_key, base_R)

    daily_log = []
    innovation_log: dict[str, list[float]] = defaultdict(list)
    variance_log: dict[str, list[float]] = defaultdict(list)
    log_likelihood_log: dict[str, list[float]] = defaultdict(list)
    per_obs_loss: list[float] = []
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
            if cumulator is not None:
                state = estimator.get_state()
                for stream_key in CUMULATED_STREAMS:
                    if stream_key in estimator.stream_registry:
                        H_row, _ = estimator.stream_registry[stream_key]
                        cumulator.accumulate(stream_key, H_row, state.mean)
            continue

        # State-dependent TPM
        if use_state_tpm and day_idx % 7 == 0:
            state = estimator.get_state()
            factor_values = {
                name: float(state.mean[i]) for i, name in enumerate(FACTORS)
            }
            adjusted_tpm = build_state_adjusted_tpm(LEVEL_A_TPM, factor_values)
            imm.tpm = adjusted_tpm

        # Predict
        probs = imm.get_probabilities()

        if use_regime_noise:
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

        # Accumulate predictions after predict
        if cumulator is not None:
            state = estimator.get_state()
            for stream_key in CUMULATED_STREAMS:
                if stream_key in estimator.stream_registry:
                    H_row, _ = estimator.stream_registry[stream_key]
                    cumulator.accumulate(stream_key, H_row, state.mean)

        # Update
        for col, kalman_key, val in today_updates:
            z = normalizer.normalize(col, val)

            if kalman_key not in estimator.stream_registry:
                continue

            H_row, base_R = estimator.stream_registry[kalman_key]

            # --- Compute effective R ---
            effective_R = base_R

            # P5: GAS-adjusted R
            if gas_tracker is not None:
                gas_R = gas_tracker.get_R(kalman_key)
                if gas_R is not None:
                    effective_R = gas_R

            # P4: Gap adjustment for cumulated streams
            gap_days = 1
            if cumulator is not None and cumulator.should_cumulate(kalman_key):
                cumulated = cumulator.get_cumulated_prediction(kalman_key)
                if cumulated is not None:
                    _, gap_days = cumulated
                    effective_R = compute_gap_adjusted_R(effective_R, gap_days, kalman_key)

            # P0: Regime noise multiplier
            if use_regime_noise:
                r_blend = 0.0
                for branch_id, prob in probs.items():
                    regime = REGIME_MAP.get(branch_id, "expansion")
                    r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                    r_blend += prob * r_mult
                effective_R = effective_R * r_blend

            # --- Compute innovation ---
            use_cumulated_pred = (
                cumulator is not None
                and cumulator.should_cumulate(kalman_key)
                and cumulator.get_cumulated_prediction(kalman_key) is not None
            )

            if use_cumulated_pred:
                mean_pred, _ = cumulator.get_cumulated_prediction(kalman_key)
                innovation = z - mean_pred
            else:
                innovation = z - float(H_row @ estimator.x)

            # --- Kalman update ---
            estimator.stream_registry[kalman_key] = (H_row, effective_R)
            S = float(H_row @ estimator.P @ H_row.T + effective_R)
            K = estimator.P @ H_row.T / S
            innovation_zscore = innovation / np.sqrt(S)

            estimator.x = estimator.x + K * innovation
            I_KH = np.eye(N_FACTORS) - np.outer(K, H_row)
            estimator.P = I_KH @ estimator.P @ I_KH.T + np.outer(K, K) * effective_R

            # Restore base R
            estimator.stream_registry[kalman_key] = (H_row, base_R)

            # GAS update
            if gas_tracker is not None:
                gas_tracker.update(kalman_key, innovation, S)

            # Reset cumulator
            if cumulator is not None and cumulator.should_cumulate(kalman_key):
                cumulator.reset(kalman_key)

            # IMM update
            if use_regime_noise:
                for branch in imm.branches:
                    if branch.estimator and kalman_key in branch.estimator.stream_registry:
                        regime = REGIME_MAP.get(branch.branch_id, "expansion")
                        r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                        bH, bR = branch.estimator.stream_registry[kalman_key]
                        branch.estimator.stream_registry[kalman_key] = (bH, bR * r_mult)
                imm.update(kalman_key, z)
                for branch in imm.branches:
                    if branch.estimator and kalman_key in branch.estimator.stream_registry:
                        bH, bR_scaled = branch.estimator.stream_registry[kalman_key]
                        regime = REGIME_MAP.get(branch.branch_id, "expansion")
                        r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                        branch.estimator.stream_registry[kalman_key] = (bH, bR_scaled / r_mult)
            else:
                imm.update(kalman_key, z)

            # Log
            update_is_anomalous = abs(float(innovation_zscore)) > ANOMALY_THRESHOLD

            total_updates += 1
            if not is_warmup:
                innovation_log[kalman_key].append(float(innovation))
                variance_log[kalman_key].append(float(S))
                S_ll = float(H_row @ estimator.P @ H_row.T + base_R)
                ll = -0.5 * (innovation_zscore**2 + np.log(2 * np.pi * S_ll))
                log_likelihood_log[kalman_key].append(ll)
                total_log_lik += ll
                per_obs_loss.append(-ll)  # loss = negative log-lik
                if update_is_anomalous:
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

    return EngineOutput(
        name=engine_name,
        daily_log=daily_log,
        innovation_log=dict(innovation_log),
        variance_log=dict(variance_log),
        log_likelihoods=dict(log_likelihood_log),
        total_log_likelihood=total_log_lik,
        per_obs_loss=per_obs_loss,
        anomaly_count=anomaly_count,
        total_updates=total_updates,
    )


def format_brier(b) -> str:
    return (
        f"BS={b.brier_score:.4f}  "
        f"rel={b.reliability:.4f}  "
        f"res={b.resolution:.4f}  "
        f"unc={b.uncertainty:.4f}"
    )


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 90)
    print("  PROPER REGIME DIAGNOSTICS")
    print("  Kim & Nelson (1999): Ljung-Box is inappropriate for Markov-switching models")
    print("  Hashimzade et al. (2024): IMM evaluated by regime classification, not whiteness")
    print("=" * 90)

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        print(f"\nLoading cached data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    # --- Run three configs ---
    print("\n--- A: Baseline (no improvements) ---")
    baseline = run_engine(data, "Baseline")
    print(f"  {baseline.total_updates} updates, LL = {baseline.total_log_likelihood:.0f}")

    print("\n--- B: P0+P2b (regime noise + state TPM) ---")
    p0p2b = run_engine(data, "P0+P2b", use_regime_noise=True, use_state_tpm=True)
    print(f"  {p0p2b.total_updates} updates, LL = {p0p2b.total_log_likelihood:.0f}")

    print("\n--- C: P0+P2b+P4+P5 (full shipping stack) ---")
    full = run_engine(
        data, "Full Stack",
        use_regime_noise=True, use_state_tpm=True,
        use_cumulator=True, use_gas=True,
    )
    print(f"  {full.total_updates} updates, LL = {full.total_log_likelihood:.0f}")

    configs = [
        ("Baseline", baseline),
        ("P0+P2b", p0p2b),
        ("Full Stack", full),
    ]

    # ── 1. Brier Score ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  1. BRIER SCORE DECOMPOSITION (Murphy 1973)")
    print("     Lower BS = better.  Low reliability = well calibrated.")
    print("     High resolution = sharp discrimination.")
    print("=" * 90)

    for regime in ["contraction", "stagflation", "expansion"]:
        print(f"\n  Regime: {regime}")
        header = f"  {'Config':<20s} {'Brier':>8s}  {'Reliab':>8s}  {'Resol':>8s}  {'Uncert':>8s}"
        print(header)
        print("  " + "-" * 60)

        for name, eng in configs:
            diag = run_full_diagnostics(eng.daily_log, REGIME_CHECKPOINTS)
            b = diag["brier"][regime]
            print(
                f"  {name:<20s} {b.brier_score:>8.4f}  "
                f"{b.reliability:>8.4f}  {b.resolution:>8.4f}  "
                f"{b.uncertainty:>8.4f}"
            )

    # ── 2. ROC / AUC ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  2. ROC / AUC (Berge & Jordà 2011)")
    print("     >0.9 excellent, >0.8 good, >0.7 fair, 0.5 = random")
    print("=" * 90)

    for regime in ["contraction", "stagflation", "expansion"]:
        print(f"\n  Regime: {regime}")
        header = f"  {'Config':<20s} {'AUC':>8s}  {'Opt Thresh':>10s}  {'TPR@opt':>8s}  {'FPR@opt':>8s}"
        print(header)
        print("  " + "-" * 60)

        for name, eng in configs:
            diag = run_full_diagnostics(eng.daily_log, REGIME_CHECKPOINTS)
            r = diag["roc"][regime]
            print(
                f"  {name:<20s} {r.auc:>8.4f}  "
                f"{r.optimal_threshold:>10.4f}  "
                f"{r.tpr_at_optimal:>8.4f}  {r.fpr_at_optimal:>8.4f}"
            )

    # ── 3. Diebold-Mariano ────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  3. DIEBOLD-MARIANO TEST (1995)")
    print("     H0: both models have equal predictive accuracy")
    print("     Negative DM stat + p<0.05 = first model significantly better")
    print("=" * 90)

    # Compare: Full Stack vs Baseline
    min_len = min(len(full.per_obs_loss), len(baseline.per_obs_loss))
    if min_len > 0:
        dm_full_vs_base = diebold_mariano(
            full.per_obs_loss[:min_len],
            baseline.per_obs_loss[:min_len],
            model_a="Full Stack",
            model_b="Baseline",
        )
        print(f"\n  Full Stack vs Baseline:")
        print(f"    DM statistic: {dm_full_vs_base.dm_statistic:>8.4f}")
        print(f"    p-value:      {dm_full_vs_base.p_value:>8.6f}")
        print(f"    Mean loss diff: {dm_full_vs_base.mean_loss_diff:>8.6f}")
        print(f"    Verdict:      {dm_full_vs_base.verdict}")

    # Compare: P0+P2b vs Baseline
    min_len2 = min(len(p0p2b.per_obs_loss), len(baseline.per_obs_loss))
    if min_len2 > 0:
        dm_p0p2b_vs_base = diebold_mariano(
            p0p2b.per_obs_loss[:min_len2],
            baseline.per_obs_loss[:min_len2],
            model_a="P0+P2b",
            model_b="Baseline",
        )
        print(f"\n  P0+P2b vs Baseline:")
        print(f"    DM statistic: {dm_p0p2b_vs_base.dm_statistic:>8.4f}")
        print(f"    p-value:      {dm_p0p2b_vs_base.p_value:>8.6f}")
        print(f"    Verdict:      {dm_p0p2b_vs_base.verdict}")

    # Compare: Full Stack vs P0+P2b
    min_len3 = min(len(full.per_obs_loss), len(p0p2b.per_obs_loss))
    if min_len3 > 0:
        dm_full_vs_p0 = diebold_mariano(
            full.per_obs_loss[:min_len3],
            p0p2b.per_obs_loss[:min_len3],
            model_a="Full Stack",
            model_b="P0+P2b",
        )
        print(f"\n  Full Stack vs P0+P2b:")
        print(f"    DM statistic: {dm_full_vs_p0.dm_statistic:>8.4f}")
        print(f"    p-value:      {dm_full_vs_p0.p_value:>8.6f}")
        print(f"    Verdict:      {dm_full_vs_p0.verdict}")

    # ── 4. PIT Histogram ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  4. PIT HISTOGRAM (Diebold, Gunther & Tay 1998)")
    print("     Replaces Ljung-Box as density calibration check.")
    print("     Flat histogram = well-calibrated density. p>0.05 = uniform.")
    print("=" * 90)

    for name, eng in configs:
        # Aggregate all streams
        all_innovations = []
        all_variances = []
        for stream_key in sorted(eng.innovation_log.keys()):
            innov = eng.innovation_log[stream_key]
            var = eng.variance_log.get(stream_key, [])
            min_n = min(len(innov), len(var))
            all_innovations.extend(innov[:min_n])
            all_variances.extend(var[:min_n])

        if all_innovations:
            pit = pit_test(all_innovations, all_variances)
            uniform_str = "UNIFORM" if pit.is_uniform else "NOT UNIFORM"
            print(f"\n  {name}:")
            print(f"    chi2={pit.chi2_statistic:.2f}, p={pit.p_value:.4f} → {uniform_str}")
            print(f"    Bin counts (expect ~{pit.expected_count:.0f} each): {pit.bin_counts}")

    # ── 5. Sharpness ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  5. SHARPNESS (Gneiting et al. 2007)")
    print("     Mean max(P): 1.0=always certain, 0.33=always confused")
    print("     Meaningful ONLY paired with calibration (Brier reliability)")
    print("=" * 90)

    header = f"  {'Config':<20s} {'Mean max(P)':>12s}  {'P>0.8':>8s}  {'P>0.5':>8s}"
    print(header)
    print("  " + "-" * 50)

    for name, eng in configs:
        diag = run_full_diagnostics(eng.daily_log, REGIME_CHECKPOINTS)
        s = diag["sharpness"]
        print(
            f"  {name:<20s} {s.mean_max_prob:>12.4f}  "
            f"{s.frac_above_80:>7.1%}  {s.frac_above_50:>7.1%}"
        )

    # ── 6. Detection Lag ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  6. DETECTION LAG (weeks from event start to P>0.5)")
    print("=" * 90)

    for name, eng in configs:
        diag = run_full_diagnostics(eng.daily_log, REGIME_CHECKPOINTS)
        print(f"\n  {name}:")
        for dl in diag["detection_lag"]:
            lag_str = f"{dl.detection_lag_weeks:.1f}w" if dl.detection_lag_weeks is not None else "MISSED"
            peak_str = f"peak={dl.peak_probability:.2f}"
            print(f"    {lag_str:>8s}  {peak_str}  {dl.event}")

    # ── Summary verdict ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  SUMMARY — WHAT THE CORRECT DIAGNOSTICS SAY")
    print("=" * 90)

    full_diag = run_full_diagnostics(full.daily_log, REGIME_CHECKPOINTS)

    contraction_auc = full_diag["roc"]["contraction"].auc
    stagflation_auc = full_diag["roc"]["stagflation"].auc
    sharp = full_diag["sharpness"]

    detected = sum(
        1 for dl in full_diag["detection_lag"]
        if dl.detection_lag_weeks is not None
    )
    total_events = len(full_diag["detection_lag"])
    mean_lag = np.mean([
        dl.detection_lag_weeks
        for dl in full_diag["detection_lag"]
        if dl.detection_lag_weeks is not None
    ])

    print(f"\n  Full Stack (P0+P2b+P4+P5):")
    print(f"    Contraction ROC AUC:   {contraction_auc:.4f}")
    print(f"    Stagflation ROC AUC:   {stagflation_auc:.4f}")
    print(f"    Sharpness (mean max):  {sharp.mean_max_prob:.4f}")
    print(f"    Events detected:       {detected}/{total_events}")
    print(f"    Mean detection lag:    {mean_lag:.1f} weeks")
    print(f"    Total log-likelihood:  {full.total_log_likelihood:.0f}")

    if min_len > 0:
        print(f"    DM vs Baseline:        stat={dm_full_vs_base.dm_statistic:.4f}, "
              f"p={dm_full_vs_base.p_value:.6f} → {dm_full_vs_base.verdict}")

    # ── Save results ──────────────────────────────────────────────────
    results_dir = Path(__file__).parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for name, eng in configs:
        diag = run_full_diagnostics(eng.daily_log, REGIME_CHECKPOINTS)
        config_results = {
            "log_likelihood": round(eng.total_log_likelihood, 2),
            "anomaly_count": eng.anomaly_count,
            "brier": {},
            "roc_auc": {},
            "sharpness": {
                "mean_max_prob": diag["sharpness"].mean_max_prob,
                "frac_above_80": diag["sharpness"].frac_above_80,
                "frac_above_50": diag["sharpness"].frac_above_50,
            },
            "detection_lag": [
                {
                    "event": dl.event,
                    "lag_weeks": dl.detection_lag_weeks,
                    "peak_prob": dl.peak_probability,
                    "mean_prob": dl.mean_probability,
                }
                for dl in diag["detection_lag"]
            ],
        }
        for regime in ["contraction", "stagflation", "expansion"]:
            b = diag["brier"][regime]
            config_results["brier"][regime] = {
                "score": b.brier_score,
                "reliability": b.reliability,
                "resolution": b.resolution,
                "uncertainty": b.uncertainty,
            }
            r = diag["roc"][regime]
            config_results["roc_auc"][regime] = {
                "auc": r.auc,
                "optimal_threshold": r.optimal_threshold,
                "tpr_at_optimal": r.tpr_at_optimal,
                "fpr_at_optimal": r.fpr_at_optimal,
            }
        save_data[name] = config_results

    # Add DM tests
    if min_len > 0:
        save_data["diebold_mariano"] = {
            "full_vs_baseline": {
                "dm_stat": dm_full_vs_base.dm_statistic,
                "p_value": dm_full_vs_base.p_value,
                "verdict": dm_full_vs_base.verdict,
            },
            "p0p2b_vs_baseline": {
                "dm_stat": dm_p0p2b_vs_base.dm_statistic,
                "p_value": dm_p0p2b_vs_base.p_value,
                "verdict": dm_p0p2b_vs_base.verdict,
            },
            "full_vs_p0p2b": {
                "dm_stat": dm_full_vs_p0.dm_statistic,
                "p_value": dm_full_vs_p0.p_value,
                "verdict": dm_full_vs_p0.verdict,
            },
        }

    out_path = results_dir / "diagnostic_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
