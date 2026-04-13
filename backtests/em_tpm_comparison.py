"""
EM-Estimated TPM vs Hand-Tuned TPM Comparison.

The hand-tuned TPM in LEVEL_A_TPM is the single biggest methodological
vulnerability in the engine. Every published MS-DFM (Hamilton 1989,
Kim & Nelson 1999, Hashimzade et al. 2024) estimates the TPM via MLE or EM.

This script:
  1. Runs the shipping stack (P0+P2b+P4+P5+C1) with hand-tuned TPM
  2. Extracts filtered probabilities from the daily log
  3. Estimates the TPM via EM (Hamilton 1989 / Kim 1994)
  4. Re-runs the shipping stack with the EM-estimated TPM
  5. Compares all diagnostics: Brier, ROC/AUC, detection lag, DM test

Refs:
  Hamilton, J.D. (1989). Econometrica 57(2), 357-384.
  Kim, C.-J. (1994). J. Econometrics 60(1-2), 1-22.
  Dempster, Laird & Rubin (1977). JRSS B.

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/em_tpm_comparison.py
"""

import sys
import json
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

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
from heimdall.em_tpm import em_estimate_tpm, estimate_tpm_from_backtest
from heimdall.regime_diagnostics import (
    run_full_diagnostics,
    diebold_mariano,
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

# C1: Safe Bayesian tempering ceiling (Grunwald 2012)
TEMPER_CEILING = 0.70


@dataclass
class EngineOutput:
    name: str
    daily_log: list[dict]
    per_obs_loss: list[float]
    total_log_likelihood: float
    anomaly_count: int
    total_updates: int


def run_engine(
    data: pd.DataFrame,
    engine_name: str,
    tpm_override: np.ndarray | None = None,
    temper_ceiling: float = 1.0,
) -> EngineOutput:
    """Run the full shipping stack (P0+P2b+P4+P5) with optional TPM override."""
    col_to_kalman = {}
    for col in data.columns:
        parts = col.split("_", 1)
        if len(parts) == 2:
            col_to_kalman[col] = parts[1]

    estimator = EconomicStateEstimator()
    imm = IMMBranchTracker()
    imm.initialize_branches(BRANCHES, baseline=estimator)

    # Override TPM if provided
    if tpm_override is not None:
        imm.tpm = tpm_override.copy()

    norms = compute_warmup_norms(data, warmup_days=180)
    normalizer = RollingNormalizer(halflife_days=120)
    for col, n in norms.items():
        normalizer.initialize(col, n["mean"], n["std"])

    cumulator = StreamCumulator()
    gas_tracker = GASNoiseTracker()
    for stream_key, (_, base_R) in estimator.stream_registry.items():
        gas_tracker.initialize(stream_key, base_R)

    daily_log = []
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
            state = estimator.get_state()
            for stream_key in CUMULATED_STREAMS:
                if stream_key in estimator.stream_registry:
                    H_row, _ = estimator.stream_registry[stream_key]
                    cumulator.accumulate(stream_key, H_row, state.mean)
            continue

        # State-dependent TPM (P2b) — only if no override
        if day_idx % 7 == 0:
            state = estimator.get_state()
            factor_values = {
                name: float(state.mean[i]) for i, name in enumerate(FACTORS)
            }
            base_tpm = tpm_override if tpm_override is not None else LEVEL_A_TPM
            adjusted_tpm = build_state_adjusted_tpm(base_tpm, factor_values)
            imm.tpm = adjusted_tpm

        probs = imm.get_probabilities()

        # P0: Regime noise
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

            # P5: GAS-adjusted R
            effective_R = base_R
            gas_R = gas_tracker.get_R(kalman_key)
            if gas_R is not None:
                effective_R = gas_R

            # P4: Gap adjustment
            gap_days = 1
            if cumulator.should_cumulate(kalman_key):
                cumulated = cumulator.get_cumulated_prediction(kalman_key)
                if cumulated is not None:
                    _, gap_days = cumulated
                    effective_R = compute_gap_adjusted_R(effective_R, gap_days, kalman_key)

            # P0: Regime noise multiplier on R
            r_blend = 0.0
            for branch_id, prob in probs.items():
                regime = REGIME_MAP.get(branch_id, "expansion")
                r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                r_blend += prob * r_mult
            effective_R = effective_R * r_blend

            # Innovation
            use_cumulated_pred = (
                cumulator.should_cumulate(kalman_key)
                and cumulator.get_cumulated_prediction(kalman_key) is not None
            )
            if use_cumulated_pred:
                mean_pred, _ = cumulator.get_cumulated_prediction(kalman_key)
                innovation = z - mean_pred
            else:
                innovation = z - float(H_row @ estimator.x)

            # Kalman update
            estimator.stream_registry[kalman_key] = (H_row, effective_R)
            S = float(H_row @ estimator.P @ H_row.T + effective_R)
            K = estimator.P @ H_row.T / S
            innovation_zscore = innovation / np.sqrt(S)

            estimator.x = estimator.x + K * innovation
            I_KH = np.eye(N_FACTORS) - np.outer(K, H_row)
            estimator.P = I_KH @ estimator.P @ I_KH.T + np.outer(K, K) * effective_R

            estimator.stream_registry[kalman_key] = (H_row, base_R)

            # GAS update
            gas_tracker.update(kalman_key, innovation, S)

            # Cumulator reset
            if cumulator.should_cumulate(kalman_key):
                cumulator.reset(kalman_key)

            # IMM update with regime noise
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

            total_updates += 1
            if not is_warmup:
                S_ll = float(H_row @ estimator.P @ H_row.T + base_R)
                ll = -0.5 * (innovation_zscore**2 + np.log(2 * np.pi * S_ll))
                total_log_lik += ll
                per_obs_loss.append(-ll)
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

    return EngineOutput(
        name=engine_name,
        daily_log=daily_log,
        per_obs_loss=per_obs_loss,
        total_log_likelihood=total_log_lik,
        anomaly_count=anomaly_count,
        total_updates=total_updates,
    )


def main():
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 90)
    print("  EM-ESTIMATED TPM vs HAND-TUNED TPM")
    print("  Hamilton (1989) / Kim (1994) / Dempster-Laird-Rubin (1977)")
    print("=" * 90)

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        print(f"\nLoading cached data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    # ── Phase 1: Run with hand-tuned TPM ─────────────────────────────
    print("\n--- Phase 1: Run shipping stack with HAND-TUNED TPM ---")
    print(f"  LEVEL_A_TPM:\n{LEVEL_A_TPM}")
    hand_tuned = run_engine(data, "Hand-tuned TPM")
    print(f"  {hand_tuned.total_updates} updates, LL = {hand_tuned.total_log_likelihood:.0f}")

    # ── Phase 2: EM estimation from filtered probabilities ───────────
    print("\n--- Phase 2: EM estimation of TPM (20 iterations) ---")
    branch_order = ["soft_landing", "stagflation", "recession"]

    estimated_tpm, _, convergence = estimate_tpm_from_backtest(
        hand_tuned.daily_log,
        branch_order,
        LEVEL_A_TPM,
        n_iterations=20,
    )

    print(f"\n  EM-estimated TPM (after {len(convergence)} iterations):")
    for i, branch in enumerate(branch_order):
        row = "  ".join(f"{estimated_tpm[i, j]:.4f}" for j in range(len(branch_order)))
        print(f"    {branch:>15s}: [{row}]")

    print(f"\n  Convergence: {[f'{c:.6f}' for c in convergence]}")
    print(f"  Final delta: {convergence[-1]:.8f}")

    print(f"\n  Comparison with hand-tuned:")
    for i, branch in enumerate(branch_order):
        for j, target in enumerate(branch_order):
            hand = LEVEL_A_TPM[i, j]
            em = estimated_tpm[i, j]
            delta = em - hand
            print(f"    {branch:>15s}→{target:<15s}: "
                  f"hand={hand:.4f}  EM={em:.4f}  Δ={delta:+.4f}")

    # ── Phase 3: Re-run with EM-estimated TPM ────────────────────────
    print("\n--- Phase 3: Run shipping stack with EM-ESTIMATED TPM ---")
    em_result = run_engine(data, "EM-estimated TPM", tpm_override=estimated_tpm)
    print(f"  {em_result.total_updates} updates, LL = {em_result.total_log_likelihood:.0f}")

    # ── Phase 4: Full diagnostics comparison ──────────────────────────
    print("\n" + "=" * 90)
    print("  DIAGNOSTICS COMPARISON")
    print("=" * 90)

    configs = [
        ("Hand-tuned", hand_tuned),
        ("EM-estimated", em_result),
    ]

    # Brier scores
    print("\n  BRIER DECOMPOSITION (Murphy 1973)")
    for regime in ["contraction", "stagflation", "expansion"]:
        print(f"\n  Regime: {regime}")
        print(f"  {'Config':<20s} {'Brier':>8s}  {'Reliab':>8s}  {'Resol':>8s}  {'BSS':>8s}")
        print("  " + "-" * 56)
        for name, eng in configs:
            diag = run_full_diagnostics(eng.daily_log, REGIME_CHECKPOINTS)
            b = diag["brier"][regime]
            bss = getattr(b, 'brier_skill_score', None)
            bss_str = f"{bss:>8.4f}" if bss is not None else "     N/A"
            print(f"  {name:<20s} {b.brier_score:>8.4f}  "
                  f"{b.reliability:>8.4f}  {b.resolution:>8.4f}  {bss_str}")

    # ROC/AUC
    print(f"\n  ROC / AUC (Berge & Jordà 2011)")
    for regime in ["contraction", "stagflation", "expansion"]:
        print(f"\n  Regime: {regime}")
        print(f"  {'Config':<20s} {'AUC':>8s}  {'Opt Thresh':>10s}")
        print("  " + "-" * 40)
        for name, eng in configs:
            diag = run_full_diagnostics(eng.daily_log, REGIME_CHECKPOINTS)
            r = diag["roc"][regime]
            print(f"  {name:<20s} {r.auc:>8.4f}  {r.optimal_threshold:>10.4f}")

    # Detection lag
    print(f"\n  DETECTION LAG")
    for name, eng in configs:
        diag = run_full_diagnostics(eng.daily_log, REGIME_CHECKPOINTS)
        detected = sum(1 for dl in diag["detection_lag"] if dl.detection_lag_weeks is not None)
        total = len(diag["detection_lag"])
        print(f"\n  {name}: {detected}/{total} detected")
        for dl in diag["detection_lag"]:
            lag_str = f"{dl.detection_lag_weeks:.1f}w" if dl.detection_lag_weeks is not None else "MISSED"
            print(f"    {lag_str:>8s}  peak={dl.peak_probability:.2f}  {dl.event}")

    # Diebold-Mariano
    min_len = min(len(hand_tuned.per_obs_loss), len(em_result.per_obs_loss))
    if min_len > 0:
        dm = diebold_mariano(
            em_result.per_obs_loss[:min_len],
            hand_tuned.per_obs_loss[:min_len],
            model_a="EM-estimated",
            model_b="Hand-tuned",
        )
        print(f"\n  DIEBOLD-MARIANO: EM-estimated vs Hand-tuned")
        print(f"    DM stat: {dm.dm_statistic:.4f}")
        print(f"    p-value: {dm.p_value:.6f}")
        print(f"    Verdict: {dm.verdict}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    hand_diag = run_full_diagnostics(hand_tuned.daily_log, REGIME_CHECKPOINTS)
    em_diag = run_full_diagnostics(em_result.daily_log, REGIME_CHECKPOINTS)

    hand_auc = hand_diag["roc"]["contraction"].auc
    em_auc = em_diag["roc"]["contraction"].auc
    hand_det = sum(1 for dl in hand_diag["detection_lag"] if dl.detection_lag_weeks is not None)
    em_det = sum(1 for dl in em_diag["detection_lag"] if dl.detection_lag_weeks is not None)

    print(f"\n  {'Metric':<30s} {'Hand-tuned':>12s} {'EM-estimated':>12s} {'Δ':>10s}")
    print("  " + "-" * 66)
    print(f"  {'Log-likelihood':<30s} {hand_tuned.total_log_likelihood:>12.0f} "
          f"{em_result.total_log_likelihood:>12.0f} "
          f"{em_result.total_log_likelihood - hand_tuned.total_log_likelihood:>+10.0f}")
    print(f"  {'Contraction AUC':<30s} {hand_auc:>12.4f} {em_auc:>12.4f} "
          f"{em_auc - hand_auc:>+10.4f}")
    print(f"  {'Events detected':<30s} {hand_det:>12d} {em_det:>12d} "
          f"{em_det - hand_det:>+10d}")

    # ── Save results ──────────────────────────────────────────────────
    results_dir = Path(__file__).parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "hand_tuned_tpm": LEVEL_A_TPM.tolist(),
        "em_estimated_tpm": estimated_tpm.tolist(),
        "em_convergence": convergence,
        "branch_order": branch_order,
        "hand_tuned": {
            "log_likelihood": round(hand_tuned.total_log_likelihood, 2),
            "contraction_auc": round(hand_auc, 4),
            "events_detected": hand_det,
        },
        "em_estimated": {
            "log_likelihood": round(em_result.total_log_likelihood, 2),
            "contraction_auc": round(em_auc, 4),
            "events_detected": em_det,
        },
    }

    if min_len > 0:
        save_data["diebold_mariano"] = {
            "dm_stat": round(dm.dm_statistic, 4),
            "p_value": round(dm.p_value, 6),
            "verdict": dm.verdict,
        }

    out_path = results_dir / "em_tpm_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
