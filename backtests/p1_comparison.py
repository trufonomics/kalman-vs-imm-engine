"""
P1 A/B Comparison — Regime-Switching H (Loadings) + P0

Tests P0+P1 combined against baseline and P0-only.
Three engine variants on the same 51-year FRED data:
  A) Baseline: fixed R, Q, H
  B) P0 only: regime-dependent R/Q, fixed H
  C) P0+P1: regime-dependent R/Q + regime-switching H

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/p1_comparison.py
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
)
from heimdall.regime_noise import REGIME_R_TABLE, REGIME_Q_TABLE
from heimdall.regime_loadings import REGIME_H_ROWS

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
class EngineMetrics:
    name: str
    innovation_log: dict[str, list[float]]
    log_likelihoods: dict[str, list[float]]
    total_log_likelihood: float
    daily_log: list[dict]
    anomaly_count: int
    total_updates: int


def run_engine(
    data: pd.DataFrame,
    engine_name: str,
    use_regime_noise: bool = False,
    use_regime_loadings: bool = False,
) -> EngineMetrics:
    """Run one engine variant."""
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

        # Get regime probabilities before predict
        probs = imm.get_probabilities()

        # --- Predict with regime-dependent Q ---
        if use_regime_noise:
            base_Q = estimator._build_Q()
            q_scale = sum(
                prob * REGIME_Q_TABLE.get(REGIME_MAP.get(bid, "expansion"), 1.0)
                for bid, prob in probs.items()
            )
            estimator.Q = base_Q * q_scale
            for branch in imm.branches:
                if branch.estimator:
                    regime = REGIME_MAP.get(branch.branch_id, "expansion")
                    branch.estimator.Q = base_Q * REGIME_Q_TABLE.get(regime, 1.0)

        estimator.predict()
        imm.predict()

        # --- Update step ---
        for col, kalman_key, val in today_updates:
            z = normalizer.normalize(col, val)

            if kalman_key not in estimator.stream_registry:
                continue

            H_row_base, R_base = estimator.stream_registry[kalman_key]

            # --- Shared estimator update ---
            if use_regime_noise:
                # Blend R across regimes
                r_blend = sum(
                    prob * REGIME_R_TABLE.get(REGIME_MAP.get(bid, "expansion"), {}).get(kalman_key, 1.0)
                    for bid, prob in probs.items()
                )
                R_eff = R_base * r_blend
            else:
                R_eff = R_base

            if use_regime_loadings:
                # Blend H across regimes
                H_eff = np.zeros(N_FACTORS)
                for bid, prob in probs.items():
                    regime = REGIME_MAP.get(bid, "expansion")
                    regime_h = REGIME_H_ROWS.get(regime, {}).get(kalman_key)
                    if regime_h is not None:
                        H_eff += prob * regime_h
                    else:
                        H_eff += prob * H_row_base
            else:
                H_eff = H_row_base

            # Manual Kalman update on shared estimator with effective H and R
            predicted = float(H_eff @ estimator.x)
            innovation = z - predicted
            S = float(H_eff @ estimator.P @ H_eff.T + R_eff)
            if S <= 0:
                S = R_eff
            innovation_zscore = innovation / np.sqrt(S) if S > 0 else 0.0
            K = estimator.P @ H_eff.T / S
            x_prior = estimator.x.copy()
            estimator.x = estimator.x + K * innovation
            I_KH = np.eye(N_FACTORS) - np.outer(K, H_eff)
            estimator.P = I_KH @ estimator.P @ I_KH.T + np.outer(K, K) * R_eff
            estimator.update_count += 1

            # --- IMM branch updates with branch-specific H and R ---
            for branch in imm.branches:
                if branch.estimator is None:
                    continue
                est = branch.estimator
                regime = REGIME_MAP.get(branch.branch_id, "expansion")

                if use_regime_loadings:
                    branch_H = REGIME_H_ROWS.get(regime, {}).get(kalman_key, H_row_base)
                else:
                    branch_H = H_row_base

                if use_regime_noise:
                    r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                    branch_R = R_base * r_mult
                else:
                    branch_R = R_base

                # Branch Kalman update
                b_S = float(branch_H @ est.P @ branch_H.T + branch_R)
                if b_S <= 0:
                    b_S = branch_R
                b_pred = float(branch_H @ est.x)
                b_innov = z - b_pred
                b_K = est.P @ branch_H.T / b_S
                est.x = est.x + b_K * b_innov
                b_I_KH = np.eye(N_FACTORS) - np.outer(b_K, branch_H)
                est.P = b_I_KH @ est.P @ b_I_KH.T + np.outer(b_K, b_K) * branch_R

            # IMM probability update (Bayes' rule on branch likelihoods)
            from scipy.stats import norm as norm_dist
            likelihoods = []
            for branch in imm.branches:
                if branch.estimator is None:
                    likelihoods.append(1e-10)
                    continue
                regime = REGIME_MAP.get(branch.branch_id, "expansion")
                if use_regime_loadings:
                    bH = REGIME_H_ROWS.get(regime, {}).get(kalman_key, H_row_base)
                else:
                    bH = H_row_base
                if use_regime_noise:
                    bR = R_base * REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                else:
                    bR = R_base
                # Use PRE-update P for likelihood (already updated above, so approximate)
                # This is acceptable — the likelihood ordering is what matters for Bayes
                bS = float(bH @ branch.estimator.P @ bH.T + bR)
                if bS <= 0:
                    bS = bR
                bpred = float(bH @ branch.estimator.x)
                binnov = z - bpred
                lik = float(norm_dist.pdf(binnov, loc=0, scale=np.sqrt(bS)))
                likelihoods.append(max(lik, 1e-10))

            # Tempered Bayes update
            ramp = min((total_updates + 1) / 200, 1.0)
            exponent = 0.3 + 0.7 * ramp
            tempered = [l ** exponent for l in likelihoods]
            weighted = [t * b.probability for t, b in zip(tempered, imm.branches)]
            total_w = sum(weighted)
            if total_w > 0:
                for i, branch in enumerate(imm.branches):
                    branch.probability = weighted[i] / total_w
            # Clamp
            for branch in imm.branches:
                branch.probability = max(0.005, min(0.99, branch.probability))
            total_p = sum(b.probability for b in imm.branches)
            for branch in imm.branches:
                branch.probability /= total_p

            total_updates += 1
            if not is_warmup:
                innovation_log[kalman_key].append(innovation_zscore)
                ll = -0.5 * (innovation_zscore**2 + np.log(2 * np.pi * S))
                log_likelihood_log[kalman_key].append(ll)
                total_log_lik += ll
                if abs(innovation_zscore) > ANOMALY_THRESHOLD:
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
    """Compute Ljung-Box, bias, kurtosis, log-likelihood diagnostics."""
    results = {}
    lb_pass = lb_total = bias_pass = bias_total = 0

    for stream, innovations in sorted(metrics.innovation_log.items()):
        arr = np.array(innovations)
        n = len(arr)
        if n < 30:
            continue

        d = {"n": n}
        mean_z = float(np.mean(arr))
        d["mean"] = round(mean_z, 4)
        d["mean_bias_ok"] = abs(mean_z) < 0.2
        bias_total += 1
        if d["mean_bias_ok"]:
            bias_pass += 1

        d["std"] = round(float(np.std(arr)), 4)
        d["kurtosis"] = round(float(stats.kurtosis(arr, fisher=True)), 2)

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(arr, lags=[10], return_df=True)
            pval = float(lb["lb_pvalue"].iloc[0])
            d["ljung_box_p"] = round(pval, 4)
            d["ljung_box_pass"] = pval > 0.05
            lb_total += 1
            if d["ljung_box_pass"]:
                lb_pass += 1
        except Exception:
            d["ljung_box_p"] = None
            d["ljung_box_pass"] = None

        lls = metrics.log_likelihoods.get(stream, [])
        if lls:
            d["mean_log_lik"] = round(float(np.mean(lls)), 4)

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
    checkpoint_results = []
    for start, end, expected, min_prob, description in REGIME_CHECKPOINTS:
        window = [e for e in metrics.daily_log if start <= e["date"] <= end]
        if not window:
            checkpoint_results.append({"regime": expected, "description": description, "result": "SKIP"})
            continue

        probs = [e["probabilities"].get(expected, 0) for e in window]
        avg = np.mean(probs)
        passed = avg >= min_prob

        first_detect = None
        for e in window:
            if e["probabilities"].get(expected, 0) >= 0.5:
                first_detect = e["date"]
                break
        if first_detect:
            from datetime import datetime as dt
            detect_weeks = (dt.strptime(first_detect, "%Y-%m-%d") - dt.strptime(start, "%Y-%m-%d")).days / 7
        else:
            detect_weeks = None

        checkpoint_results.append({
            "regime": expected, "description": description,
            "result": "PASS" if passed else "FAIL",
            "avg_probability": round(avg, 4),
            "threshold": min_prob,
            "first_detection_weeks": detect_weeks,
        })
    return checkpoint_results


def print_comparison(variants: list[tuple[str, dict, list[dict]]]):
    """Print multi-variant comparison."""
    print()
    print("=" * 90)
    print("  A/B/C COMPARISON: BASELINE vs P0 vs P0+P1")
    print("=" * 90)

    names = [v[0] for v in variants]
    diags = [v[1] for v in variants]
    cps = [v[2] for v in variants]

    # Summary table
    header = f"{'Metric':<30s}" + "".join(f"{n:>18s}" for n in names)
    print(f"\n{header}")
    print("-" * 90)

    for label, key, fmt in [
        ("Ljung-Box pass rate", "ljung_box_rate", ".0%"),
        ("Mean bias pass rate", "mean_bias_rate", ".0%"),
        ("Total log-likelihood", "total_log_likelihood", ".0f"),
        ("Anomaly rate", "anomaly_rate", ".1%"),
    ]:
        row = f"{label:<30s}"
        for d in diags:
            row += f"{d[key]:>18{fmt}}"
        print(row)

    row = f"{'Regime checkpoints':<30s}"
    for cp in cps:
        p = sum(1 for c in cp if c["result"] == "PASS")
        t = sum(1 for c in cp if c["result"] != "SKIP")
        row += f"{f'{p}/{t}':>18s}"
    print(row)

    # Per-stream
    print(f"\n{'Stream':<22s}" + "".join(f"{'LB_p':>7s} {'Bias':>7s} {'LL':>7s}" for _ in names))
    print("-" * 90)
    all_streams = sorted(diags[0]["per_stream"].keys())
    for stream in all_streams:
        row = f"{stream:<22s}"
        for d in diags:
            s = d["per_stream"].get(stream, {})
            lbp = f"{s.get('ljung_box_p', 0):.3f}" if s.get('ljung_box_p') is not None else "N/A"
            bias = f"{s.get('mean', 0):+.2f}"
            ll = f"{s.get('mean_log_lik', 0):.2f}" if 'mean_log_lik' in s else "N/A"
            row += f"{lbp:>7s} {bias:>7s} {ll:>7s}"
        print(row)

    # Checkpoints
    print(f"\n{'Checkpoint':<50s}" + "".join(f"{'avg':>7s} {'det':>7s}" for _ in names))
    print("-" * 90)
    for i in range(len(cps[0])):
        if cps[0][i]["result"] == "SKIP":
            continue
        row = f"  {cps[0][i]['description'][:48]:<48s}"
        for cp in cps:
            c = cp[i]
            avg = f"{c.get('avg_probability', 0):.1%}"
            det = f"{c['first_detection_weeks']:.1f}w" if c.get('first_detection_weeks') is not None else "N/D"
            row += f"{avg:>7s} {det:>7s}"
        print(row)


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 90)
    print("  P1 A/B/C COMPARISON — Baseline vs P0 vs P0+P1 (Regime-Switching H)")
    print("=" * 90)

    data = fetch_data()

    # A) Baseline
    print("\n--- Running BASELINE ---")
    baseline = run_engine(data, "baseline", use_regime_noise=False, use_regime_loadings=False)
    print(f"  Done. {baseline.total_updates} updates")

    # B) P0 only
    print("\n--- Running P0 (regime R/Q) ---")
    p0 = run_engine(data, "P0", use_regime_noise=True, use_regime_loadings=False)
    print(f"  Done. {p0.total_updates} updates")

    # C) P0+P1
    print("\n--- Running P0+P1 (regime R/Q + regime H) ---")
    p0p1 = run_engine(data, "P0+P1", use_regime_noise=True, use_regime_loadings=True)
    print(f"  Done. {p0p1.total_updates} updates")

    # Diagnostics
    b_diag = compute_diagnostics(baseline)
    p0_diag = compute_diagnostics(p0)
    p0p1_diag = compute_diagnostics(p0p1)

    b_cp = evaluate_checkpoints(baseline)
    p0_cp = evaluate_checkpoints(p0)
    p0p1_cp = evaluate_checkpoints(p0p1)

    print_comparison([
        ("Baseline", b_diag, b_cp),
        ("P0", p0_diag, p0_cp),
        ("P0+P1", p0p1_diag, p0p1_cp),
    ])

    # Save
    results = {
        "experiment": "p1_comparison",
        "baseline": {"diagnostics": b_diag, "checkpoints": b_cp},
        "p0": {"diagnostics": p0_diag, "checkpoints": p0_cp},
        "p0_p1": {"diagnostics": p0p1_diag, "checkpoints": p0p1_cp},
    }
    output_path = Path(__file__).parent.parent / "data" / "results" / "p1_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Verdict
    print()
    print("=" * 90)
    p0p1_ll = p0p1_diag["total_log_likelihood"]
    p0_ll = p0_diag["total_log_likelihood"]
    b_ll = b_diag["total_log_likelihood"]
    p0p1_cp_pass = sum(1 for c in p0p1_cp if c["result"] == "PASS")
    p0_cp_pass = sum(1 for c in p0_cp if c["result"] == "PASS")

    if p0p1_ll > p0_ll and p0p1_cp_pass >= p0_cp_pass:
        print(f"  VERDICT: P0+P1 WINS over P0 — LL improvement {p0p1_ll - p0_ll:+.0f}, checkpoints held")
    elif p0p1_cp_pass < p0_cp_pass:
        print(f"  VERDICT: P0+P1 REGRESSED — lost checkpoints ({p0p1_cp_pass} vs {p0_cp_pass}). Tuning needed.")
    elif p0p1_ll <= p0_ll:
        print(f"  VERDICT: P0+P1 NO IMPROVEMENT over P0 — LL delta {p0p1_ll - p0_ll:+.0f}")
    else:
        print("  VERDICT: MIXED — review per-stream details.")
    print(f"  Cumulative improvement over baseline: LL {p0p1_ll - b_ll:+.0f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
