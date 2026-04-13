"""
P2 A/B/C Comparison — Baseline vs P0 vs P0+P2 (Duration-Dependent TPM)

Durland & McCurdy (1994): recession exit probability increases with
duration. A fixed TPM treats transitions as memoryless — this fix makes
the contraction→expansion probability increase as contraction persists,
and gently increases expansion exit probability as expansions age.

Runs the same 51-year FRED data through three engines:
  A) Baseline: fixed R/Q, fixed TPM
  B) P0: regime-dependent R/Q, fixed TPM (validated Apr 13 2026)
  C) P0+P2: regime-dependent R/Q + duration-dependent TPM

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/p2_comparison.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
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
    LEVEL_A_TPM,
)
from heimdall.regime_noise import (
    REGIME_R_TABLE,
    REGIME_Q_TABLE,
)
from heimdall.duration_tpm import build_duration_adjusted_tpm

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
    innovation_log: dict[str, list[float]]
    log_likelihoods: dict[str, list[float]]
    total_log_likelihood: float
    daily_log: list[dict]
    anomaly_count: int
    total_updates: int


REGIME_MAP = {
    "soft_landing": "expansion",
    "stagflation": "stagflation",
    "recession": "contraction",
}


def run_engine(
    data: pd.DataFrame,
    engine_name: str,
    use_regime_noise: bool = False,
    use_duration_tpm: bool = False,
) -> EngineMetrics:
    """Run one engine variant on the full dataset."""
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

    # Duration tracking for P2
    contraction_streak = 0
    expansion_streak = 0
    last_dominant = None

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
            continue

        # --- Track regime duration (weekly granularity) ---
        if use_duration_tpm and day_idx % 7 == 0:
            probs = imm.get_probabilities()
            dominant = max(probs, key=probs.get)
            dominant_regime = REGIME_MAP.get(dominant, "expansion")

            if dominant_regime == "contraction":
                contraction_streak += 1
                expansion_streak = 0
            elif dominant_regime == "expansion":
                expansion_streak += 1
                contraction_streak = 0
            else:
                # Stagflation resets both streaks
                contraction_streak = 0
                expansion_streak = 0

            last_dominant = dominant_regime

        # --- P2: Update TPM based on duration ---
        if use_duration_tpm and day_idx % 7 == 0:
            adjusted_tpm = build_duration_adjusted_tpm(
                LEVEL_A_TPM,
                contraction_duration=contraction_streak,
                expansion_duration=expansion_streak,
            )
            imm.tpm = adjusted_tpm

        # --- Predict step ---
        probs = imm.get_probabilities()

        if use_regime_noise:
            # P0: regime-dependent Q
            q_scale = 0.0
            for branch_id, prob in probs.items():
                regime = REGIME_MAP.get(branch_id, "expansion")
                q_scale += prob * REGIME_Q_TABLE.get(regime, 1.0)

            base_Q = estimator._build_Q()
            estimator.Q = base_Q * q_scale

            for branch in imm.branches:
                if branch.estimator:
                    regime = REGIME_MAP.get(branch.branch_id, "expansion")
                    branch_q_scale = REGIME_Q_TABLE.get(regime, 1.0)
                    branch.estimator.Q = base_Q * branch_q_scale

        estimator.predict()
        imm.predict()

        # --- Update step ---
        for col, kalman_key, val in today_updates:
            z = normalizer.normalize(col, val)

            if use_regime_noise:
                # P0: blend R across regimes
                if kalman_key in estimator.stream_registry:
                    H_row, base_R = estimator.stream_registry[kalman_key]
                    r_blend = 0.0
                    for branch_id, prob in probs.items():
                        regime = REGIME_MAP.get(branch_id, "expansion")
                        r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                        r_blend += prob * r_mult
                    effective_R = base_R * r_blend
                    estimator.stream_registry[kalman_key] = (H_row, effective_R)
                    update = estimator.update(kalman_key, z)
                    estimator.stream_registry[kalman_key] = (H_row, base_R)

                    # Branch-specific R
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
                    update = estimator.update(kalman_key, z)
                    imm.update(kalman_key, z)
            else:
                update = estimator.update(kalman_key, z)
                imm.update(kalman_key, z)

            if update:
                total_updates += 1
                if not is_warmup:
                    innovation_log[kalman_key].append(update.innovation_zscore)
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
        mean_z = float(np.mean(innovations))
        d["mean"] = round(mean_z, 4)
        d["mean_bias_ok"] = abs(mean_z) < 0.2
        bias_total += 1
        if d["mean_bias_ok"]:
            bias_pass += 1

        d["kurtosis"] = round(float(stats.kurtosis(innovations, fisher=True)), 2)
        d["std"] = round(float(np.std(innovations)), 4)

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


def print_comparison(configs):
    """Print side-by-side comparison of all configs."""
    names = [c["name"] for c in configs]
    diags = [c["diag"] for c in configs]
    cps = [c["cp"] for c in configs]

    print()
    print("=" * 90)
    print("  A/B/C COMPARISON: BASELINE vs P0 vs P0+P2")
    print("=" * 90)

    # Summary table
    header = f"{'Metric':<35s}"
    for name in names:
        header += f" {name:>16s}"
    print()
    print(header)
    print("-" * 90)

    # Ljung-Box
    row = f"{'Ljung-Box pass rate':<35s}"
    for d in diags:
        row += f" {d['ljung_box_rate']:>15.0%}"
    print(row)

    # Mean bias
    row = f"{'Mean bias pass rate':<35s}"
    for d in diags:
        row += f" {d['mean_bias_rate']:>15.0%}"
    print(row)

    # Total LL
    row = f"{'Total log-likelihood':<35s}"
    for d in diags:
        row += f" {d['total_log_likelihood']:>16.0f}"
    print(row)

    # Anomaly rate
    row = f"{'Anomaly rate':<35s}"
    for d in diags:
        row += f" {d['anomaly_rate']:>15.1%}"
    print(row)

    # Checkpoints
    row = f"{'Regime checkpoints':<35s}"
    for cp in cps:
        passed = sum(1 for c in cp if c["result"] == "PASS")
        total = sum(1 for c in cp if c["result"] != "SKIP")
        row += f" {f'{passed}/{total}':>16s}"
    print(row)

    # Per-stream detail
    print()
    streams = sorted(diags[0]["per_stream"].keys())
    header = f"{'Stream':<25s}"
    for name in names:
        header += f"  {'LB_p':>6s} {'Bias':>6s} {'LL':>6s}"
    print(header)
    print("-" * 90)

    for stream in streams:
        row = f"{stream:<25s}"
        for d in diags:
            s = d["per_stream"].get(stream, {})
            lb = f"{s.get('ljung_box_p', 0):.3f}" if s.get('ljung_box_p') is not None else "N/A"
            bias = f"{s.get('mean', 0):+.2f}"
            ll = f"{s.get('mean_log_lik', 0):.2f}" if 'mean_log_lik' in s else "N/A"
            row += f"  {lb:>6s} {bias:>6s} {ll:>6s}"
        print(row)

    # Checkpoint comparison
    print()
    header = f"{'Checkpoint':<50s}"
    for name in names:
        header += f"  {'avg':>6s} {'det':>6s}"
    print(header)
    print("-" * 90)

    for i in range(len(cps[0])):
        if cps[0][i]["result"] == "SKIP":
            continue
        row = f"  {cps[0][i]['description'][:48]:<48s}"
        for cp in cps:
            avg = f"{cp[i]['avg_probability']:.1%}"
            det = f"{cp[i]['first_detection_weeks']:.1f}w" if cp[i].get('first_detection_weeks') is not None else "N/D"
            row += f"  {avg:>6s} {det:>6s}"
        print(row)


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 90)
    print("  P2 A/B/C COMPARISON — Baseline vs P0 vs P0+P2 (Duration-Dependent TPM)")
    print("=" * 90)

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        print(f"Loading cached data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    # A) Baseline
    print("\n--- Running BASELINE ---")
    baseline = run_engine(data, "baseline", use_regime_noise=False, use_duration_tpm=False)
    print(f"  Done. {baseline.total_updates} updates")

    # B) P0 only
    print("\n--- Running P0 (regime R/Q) ---")
    p0 = run_engine(data, "p0", use_regime_noise=True, use_duration_tpm=False)
    print(f"  Done. {p0.total_updates} updates")

    # C) P0 + P2
    print("\n--- Running P0+P2 (regime R/Q + duration TPM) ---")
    p0p2 = run_engine(data, "p0+p2", use_regime_noise=True, use_duration_tpm=True)
    print(f"  Done. {p0p2.total_updates} updates")

    # Compute diagnostics
    baseline_diag = compute_diagnostics(baseline)
    p0_diag = compute_diagnostics(p0)
    p0p2_diag = compute_diagnostics(p0p2)

    baseline_cp = evaluate_checkpoints(baseline)
    p0_cp = evaluate_checkpoints(p0)
    p0p2_cp = evaluate_checkpoints(p0p2)

    configs = [
        {"name": "Baseline", "diag": baseline_diag, "cp": baseline_cp},
        {"name": "P0", "diag": p0_diag, "cp": p0_cp},
        {"name": "P0+P2", "diag": p0p2_diag, "cp": p0p2_cp},
    ]

    print_comparison(configs)

    # Save results
    results = {
        "experiment": "p2_comparison",
        "baseline": {"diagnostics": baseline_diag, "checkpoints": baseline_cp},
        "p0": {"diagnostics": p0_diag, "checkpoints": p0_cp},
        "p0_p2": {"diagnostics": p0p2_diag, "checkpoints": p0p2_cp},
    }

    output_path = Path(__file__).parent.parent / "data" / "results" / "p2_comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Verdict
    p0_ll = p0_diag["total_log_likelihood"]
    p2_ll = p0p2_diag["total_log_likelihood"]
    p0_cp_pass = sum(1 for c in p0_cp if c["result"] == "PASS")
    p2_cp_pass = sum(1 for c in p0p2_cp if c["result"] == "PASS")
    baseline_ll = baseline_diag["total_log_likelihood"]

    print()
    print("=" * 90)
    ll_delta = p2_ll - p0_ll
    if ll_delta > 0 and p2_cp_pass >= p0_cp_pass:
        print(f"  VERDICT: P0+P2 WINS — LL delta {ll_delta:+.0f} over P0 alone")
    elif ll_delta > 0 and p2_cp_pass < p0_cp_pass:
        print(f"  VERDICT: P0+P2 MIXED — LL improved {ll_delta:+.0f} but lost {p0_cp_pass - p2_cp_pass} checkpoint(s)")
    elif ll_delta <= 0:
        print(f"  VERDICT: P0+P2 NO IMPROVEMENT over P0 — LL delta {ll_delta:+.0f}")
    print(f"  Cumulative improvement over baseline: LL {p2_ll - baseline_ll:+.0f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
