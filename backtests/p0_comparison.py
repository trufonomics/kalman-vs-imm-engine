"""
P0 A/B Comparison — Regime-Dependent R/Q vs Fixed R/Q

Runs the same 51-year FRED data through two engines:
  A) Baseline: fixed R and Q (current engine)
  B) P0: regime-dependent R and Q (each IMM branch gets its own noise)

Compares:
  1. Ljung-Box p-values (innovation whiteness)
  2. Mean innovation bias per stream
  3. Log-likelihood (total and per-stream)
  4. Regime checkpoint accuracy (11 checkpoints)
  5. Innovation z-score distributions (should approach N(0,1))

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/p0_comparison.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

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
    RECOMMENDED_BRANCH_ADJUSTMENTS,
    DEFAULT_TPM,
)
from heimdall.regime_noise import (
    REGIME_R_TABLE,
    REGIME_Q_TABLE,
)

# Import backtest infrastructure from existing script
from backtests.full_history_backtest import (
    fetch_data,
    compute_warmup_norms,
    RollingNormalizer,
    BRANCHES,
    REGIME_CHECKPOINTS,
)


@dataclass
class EngineMetrics:
    """Collected metrics for one engine run."""
    name: str
    innovation_log: dict[str, list[float]]  # stream -> z-scores
    log_likelihoods: dict[str, list[float]]  # stream -> log-liks
    total_log_likelihood: float
    daily_log: list[dict]
    anomaly_count: int
    total_updates: int


def run_engine(
    data: pd.DataFrame,
    engine_name: str,
    use_regime_noise: bool = False,
) -> EngineMetrics:
    """Run one engine variant on the full dataset.

    Args:
        data: The FRED + Yahoo dataframe
        engine_name: Label for this run
        use_regime_noise: If True, apply P0 regime-dependent R/Q
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

    daily_log = []
    innovation_log: dict[str, list[float]] = defaultdict(list)
    log_likelihood_log: dict[str, list[float]] = defaultdict(list)
    total_log_lik = 0.0
    anomaly_count = 0
    total_updates = 0
    warmup_end_idx = 180

    streams_seen = set()

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
            if col not in streams_seen:
                streams_seen.add(col)
            today_updates.append((col, kalman_key, val))

        if not today_updates:
            continue

        # --- Predict step ---
        if use_regime_noise:
            # P0: Apply regime-dependent Q before predict
            # Get current regime probabilities
            probs = imm.get_probabilities()
            # Blend Q scales across regimes
            q_scale = 0.0
            regime_map = {
                "soft_landing": "expansion",
                "stagflation": "stagflation",
                "recession": "contraction",
            }
            for branch_id, prob in probs.items():
                regime = regime_map.get(branch_id, "expansion")
                q_scale += prob * REGIME_Q_TABLE.get(regime, 1.0)

            # Scale Q for the shared estimator
            base_Q = estimator._build_Q()
            estimator.Q = base_Q * q_scale

            # Scale Q for each branch estimator too
            for branch in imm.branches:
                if branch.estimator:
                    regime = regime_map.get(branch.branch_id, "expansion")
                    branch_q_scale = REGIME_Q_TABLE.get(regime, 1.0)
                    branch.estimator.Q = base_Q * branch_q_scale

        estimator.predict()
        imm.predict()

        # --- Update step ---
        for col, kalman_key, val in today_updates:
            z = normalizer.normalize(col, val)

            if use_regime_noise:
                # P0: Blend R across regimes for the shared estimator
                if kalman_key in estimator.stream_registry:
                    H_row, base_R = estimator.stream_registry[kalman_key]
                    # Get base R (before any regime scaling)
                    # base_R already includes frequency scaling from init
                    r_blend = 0.0
                    for branch_id, prob in probs.items():
                        regime = regime_map.get(branch_id, "expansion")
                        r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                        r_blend += prob * r_mult
                    effective_R = base_R * r_blend
                    # Temporarily override R for this update
                    estimator.stream_registry[kalman_key] = (H_row, effective_R)
                    update = estimator.update(kalman_key, z)
                    # Restore base R
                    estimator.stream_registry[kalman_key] = (H_row, base_R)

                    # Also apply branch-specific R for IMM update
                    for branch in imm.branches:
                        if branch.estimator and kalman_key in branch.estimator.stream_registry:
                            regime = regime_map.get(branch.branch_id, "expansion")
                            r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                            bH, bR = branch.estimator.stream_registry[kalman_key]
                            branch.estimator.stream_registry[kalman_key] = (bH, bR * r_mult)

                    imm.update(kalman_key, z)

                    # Restore branch R values
                    for branch in imm.branches:
                        if branch.estimator and kalman_key in branch.estimator.stream_registry:
                            bH, bR_scaled = branch.estimator.stream_registry[kalman_key]
                            regime = regime_map.get(branch.branch_id, "expansion")
                            r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                            branch.estimator.stream_registry[kalman_key] = (bH, bR_scaled / r_mult)
                else:
                    update = estimator.update(kalman_key, z)
                    imm.update(kalman_key, z)
            else:
                # Baseline: standard update
                update = estimator.update(kalman_key, z)
                imm.update(kalman_key, z)

            if update:
                total_updates += 1
                if not is_warmup:
                    innovation_log[kalman_key].append(update.innovation_zscore)

                    # Log-likelihood: -0.5 * (z^2 + log(2*pi*S))
                    # where S = H @ P @ H' + R (innovation variance)
                    H_row_ll, R_ll = estimator.stream_registry[kalman_key]
                    S_ll = float(H_row_ll @ estimator.P @ H_row_ll.T + R_ll)
                    ll = -0.5 * (update.innovation_zscore**2 + np.log(2 * np.pi * S_ll))
                    log_likelihood_log[kalman_key].append(ll)
                    total_log_lik += ll

                    if abs(update.innovation_zscore) > ANOMALY_THRESHOLD:
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

    return EngineMetrics(
        name=engine_name,
        innovation_log=dict(innovation_log),
        log_likelihoods=dict(log_likelihood_log),
        total_log_likelihood=total_log_lik,
        daily_log=daily_log,
        anomaly_count=anomaly_count,
        total_updates=total_updates,
    )


def compute_diagnostics(metrics: EngineMetrics) -> dict:
    """Compute Ljung-Box, bias, kurtosis, and log-likelihood diagnostics."""
    results = {}
    lb_pass = 0
    lb_total = 0
    bias_pass = 0
    bias_total = 0

    for stream, innovations in sorted(metrics.innovation_log.items()):
        innovations = np.array(innovations)
        n = len(innovations)
        if n < 30:
            continue

        d = {"n": n}

        # Mean bias
        mean_z = float(np.mean(innovations))
        d["mean"] = round(mean_z, 4)
        d["mean_bias_ok"] = abs(mean_z) < 0.2
        bias_total += 1
        if d["mean_bias_ok"]:
            bias_pass += 1

        # Kurtosis
        d["kurtosis"] = round(float(stats.kurtosis(innovations, fisher=True)), 2)

        # Standard deviation (should be ~1.0 for calibrated filter)
        d["std"] = round(float(np.std(innovations)), 4)

        # Ljung-Box
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(innovations, lags=[10], return_df=True)
            lb_pval = float(lb_result["lb_pvalue"].iloc[0])
            d["ljung_box_p"] = round(lb_pval, 4)
            d["ljung_box_pass"] = lb_pval > 0.05
            lb_total += 1
            if d["ljung_box_pass"]:
                lb_pass += 1
        except Exception:
            d["ljung_box_p"] = None
            d["ljung_box_pass"] = None

        # Per-stream log-likelihood
        lls = metrics.log_likelihoods.get(stream, [])
        if lls:
            d["mean_log_lik"] = round(float(np.mean(lls)), 4)
            d["total_log_lik"] = round(float(np.sum(lls)), 2)

        results[stream] = d

    return {
        "per_stream": results,
        "ljung_box_pass": lb_pass,
        "ljung_box_total": lb_total,
        "ljung_box_rate": round(lb_pass / max(lb_total, 1), 4),
        "mean_bias_pass": bias_pass,
        "mean_bias_total": bias_total,
        "mean_bias_rate": round(bias_pass / max(bias_total, 1), 4),
        "total_log_likelihood": round(metrics.total_log_likelihood, 2),
        "anomaly_count": metrics.anomaly_count,
        "anomaly_rate": round(metrics.anomaly_count / max(metrics.total_updates - 180 * 3, 1), 4),
    }


def evaluate_checkpoints(metrics: EngineMetrics) -> list[dict]:
    """Evaluate regime detection against historical checkpoints."""
    checkpoint_results = []

    for start, end, expected, min_prob, description in REGIME_CHECKPOINTS:
        window = [
            entry for entry in metrics.daily_log
            if start <= entry["date"] <= end
        ]
        if not window:
            checkpoint_results.append({
                "regime": expected, "description": description,
                "result": "SKIP",
            })
            continue

        expected_probs = [entry["probabilities"].get(expected, 0) for entry in window]
        avg_prob = np.mean(expected_probs)
        max_prob = np.max(expected_probs)

        dominant_count = sum(
            1 for entry in window
            if max(entry["probabilities"], key=entry["probabilities"].get) == expected
        )
        dominance_pct = dominant_count / len(window)

        passed = avg_prob >= min_prob

        # Detection speed
        first_detect = None
        for entry in window:
            if entry["probabilities"].get(expected, 0) >= 0.5:
                first_detect = entry["date"]
                break
        if first_detect:
            from datetime import datetime as dt
            delta = (dt.strptime(first_detect, "%Y-%m-%d") - dt.strptime(start, "%Y-%m-%d")).days
            detect_weeks = delta / 7
        else:
            detect_weeks = None

        checkpoint_results.append({
            "regime": expected,
            "description": description,
            "result": "PASS" if passed else "FAIL",
            "avg_probability": round(avg_prob, 4),
            "max_probability": round(max_prob, 4),
            "dominance_pct": round(dominance_pct, 4),
            "threshold": min_prob,
            "first_detection_weeks": detect_weeks,
        })

    return checkpoint_results


def print_comparison(
    baseline_diag: dict,
    p0_diag: dict,
    baseline_cp: list[dict],
    p0_cp: list[dict],
):
    """Print side-by-side comparison."""
    print()
    print("=" * 80)
    print("  A/B COMPARISON: BASELINE vs P0 (Regime-Dependent R/Q)")
    print("=" * 80)

    # Summary
    print()
    print(f"{'Metric':<35s} {'Baseline':>15s} {'P0':>15s} {'Delta':>12s}")
    print("-" * 80)

    b_lb = baseline_diag["ljung_box_rate"]
    p_lb = p0_diag["ljung_box_rate"]
    print(f"{'Ljung-Box pass rate':<35s} {b_lb:>14.0%} {p_lb:>14.0%} {p_lb - b_lb:>+11.0%}")

    b_bias = baseline_diag["mean_bias_rate"]
    p_bias = p0_diag["mean_bias_rate"]
    print(f"{'Mean bias pass rate':<35s} {b_bias:>14.0%} {p_bias:>14.0%} {p_bias - b_bias:>+11.0%}")

    b_ll = baseline_diag["total_log_likelihood"]
    p_ll = p0_diag["total_log_likelihood"]
    print(f"{'Total log-likelihood':<35s} {b_ll:>15.1f} {p_ll:>15.1f} {p_ll - b_ll:>+12.1f}")

    b_anom = baseline_diag["anomaly_rate"]
    p_anom = p0_diag["anomaly_rate"]
    print(f"{'Anomaly rate':<35s} {b_anom:>14.1%} {p_anom:>14.1%} {p_anom - b_anom:>+11.1%}")

    b_passed = sum(1 for c in baseline_cp if c["result"] == "PASS")
    p_passed = sum(1 for c in p0_cp if c["result"] == "PASS")
    b_total = sum(1 for c in baseline_cp if c["result"] != "SKIP")
    p_total = sum(1 for c in p0_cp if c["result"] != "SKIP")
    print(f"{'Regime checkpoints':<35s} {f'{b_passed}/{b_total}':>15s} {f'{p_passed}/{p_total}':>15s}")

    # Per-stream detail
    print()
    print(f"{'Stream':<25s} {'LB_p (B)':>8s} {'LB_p (P0)':>9s} {'Bias (B)':>9s} {'Bias (P0)':>9s} {'LL (B)':>9s} {'LL (P0)':>9s}")
    print("-" * 80)

    for stream in sorted(baseline_diag["per_stream"].keys()):
        b = baseline_diag["per_stream"][stream]
        p = p0_diag["per_stream"].get(stream, {})

        b_lbp = f"{b.get('ljung_box_p', 0):.4f}" if b.get('ljung_box_p') is not None else "N/A"
        p_lbp = f"{p.get('ljung_box_p', 0):.4f}" if p.get('ljung_box_p') is not None else "N/A"
        b_mean = f"{b.get('mean', 0):+.3f}"
        p_mean = f"{p.get('mean', 0):+.3f}"
        b_mll = f"{b.get('mean_log_lik', 0):.3f}" if 'mean_log_lik' in b else "N/A"
        p_mll = f"{p.get('mean_log_lik', 0):.3f}" if 'mean_log_lik' in p else "N/A"

        print(f"{stream:<25s} {b_lbp:>8s} {p_lbp:>9s} {b_mean:>9s} {p_mean:>9s} {b_mll:>9s} {p_mll:>9s}")

    # Checkpoint comparison
    print()
    print(f"{'Checkpoint':<50s} {'B avg':>7s} {'P0 avg':>7s} {'B det':>6s} {'P0 det':>6s}")
    print("-" * 80)
    for b_cp, p_cp in zip(baseline_cp, p0_cp):
        if b_cp["result"] == "SKIP":
            continue
        b_det = f"{b_cp['first_detection_weeks']:.1f}w" if b_cp.get('first_detection_weeks') is not None else "N/D"
        p_det = f"{p_cp['first_detection_weeks']:.1f}w" if p_cp.get('first_detection_weeks') is not None else "N/D"
        marker = ""
        if b_cp["result"] != p_cp["result"]:
            marker = " <<<" if p_cp["result"] == "FAIL" else " +++"
        print(f"  {b_cp['description'][:48]:<48s} {b_cp['avg_probability']:>6.1%} {p_cp['avg_probability']:>6.1%} {b_det:>6s} {p_det:>6s}{marker}")


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 80)
    print("  P0 A/B COMPARISON — Regime-Dependent R/Q vs Fixed R/Q")
    print("  Running 51-year FRED backtest on BOTH engines")
    print("=" * 80)

    data = fetch_data()

    # Run baseline
    print("\n--- Running BASELINE engine ---")
    baseline = run_engine(data, "baseline", use_regime_noise=False)
    print(f"  Done. {baseline.total_updates} updates, {baseline.anomaly_count} anomalies")

    # Run P0
    print("\n--- Running P0 engine (regime-dependent R/Q) ---")
    p0 = run_engine(data, "p0_regime_noise", use_regime_noise=True)
    print(f"  Done. {p0.total_updates} updates, {p0.anomaly_count} anomalies")

    # Compute diagnostics
    baseline_diag = compute_diagnostics(baseline)
    p0_diag = compute_diagnostics(p0)

    # Evaluate checkpoints
    baseline_cp = evaluate_checkpoints(baseline)
    p0_cp = evaluate_checkpoints(p0)

    # Print comparison
    print_comparison(baseline_diag, p0_diag, baseline_cp, p0_cp)

    # Save results
    results = {
        "experiment": "p0_comparison",
        "baseline": {
            "diagnostics": baseline_diag,
            "checkpoints": baseline_cp,
        },
        "p0": {
            "diagnostics": p0_diag,
            "checkpoints": p0_cp,
        },
        "regime_r_table": {k: dict(v) for k, v in REGIME_R_TABLE.items()},
        "regime_q_table": dict(REGIME_Q_TABLE),
    }

    output_path = Path(__file__).parent.parent / "data" / "results" / "p0_comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Verdict
    print()
    print("=" * 80)
    ll_improved = p0_diag["total_log_likelihood"] > baseline_diag["total_log_likelihood"]
    lb_improved = p0_diag["ljung_box_rate"] >= baseline_diag["ljung_box_rate"]
    cp_held = sum(1 for c in p0_cp if c["result"] == "PASS") >= sum(1 for c in baseline_cp if c["result"] == "PASS")

    if ll_improved and cp_held:
        print("  VERDICT: P0 WINS — higher log-likelihood, checkpoints held")
    elif not cp_held:
        print("  VERDICT: P0 REGRESSED — lost regime checkpoints. DO NOT SHIP.")
    elif not ll_improved:
        print("  VERDICT: P0 NO IMPROVEMENT — log-likelihood did not improve.")
    else:
        print("  VERDICT: MIXED — review per-stream details.")
    print("=" * 80)


if __name__ == "__main__":
    main()
