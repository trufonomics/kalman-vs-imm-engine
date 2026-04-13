"""
P3 A/B/C/D Comparison — Persistence Re-Estimation

Tests data-calibrated F_diag persistence values against the original
theory-based values. The hypothesis: Ljung-Box fails because the AR(1)
persistence is too low for daily steps, causing the filter to predict
excessive state decay between monthly observations.

Configs:
  A) Baseline: theory F_diag, fixed R/Q
  B) P0: theory F_diag, regime R/Q
  C) P0+P2b: theory F_diag, regime R/Q + state TPM
  D) P0+P2b+P3: calibrated F_diag, regime R/Q + state TPM

Also runs the Ljung-Box deep diagnostic on config D to see if
innovation ACF improves.

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/p3_comparison.py
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
    PERSISTENCE,
)
from heimdall.imm_tracker import (
    IMMBranchTracker,
    LEVEL_A_TPM,
)
from heimdall.regime_noise import REGIME_R_TABLE, REGIME_Q_TABLE
from heimdall.state_tpm import build_state_adjusted_tpm
from heimdall.calibrated_persistence import CALIBRATED_PERSISTENCE

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
    use_state_tpm: bool = False,
    persistence_override: dict[str, float] | None = None,
) -> EngineMetrics:
    col_to_kalman = {}
    for col in data.columns:
        parts = col.split("_", 1)
        if len(parts) == 2:
            col_to_kalman[col] = parts[1]

    estimator = EconomicStateEstimator()

    # P3: Override F matrix diagonal with calibrated persistence
    if persistence_override:
        F_new = estimator.F.copy()
        for factor, phi in persistence_override.items():
            idx = FACTOR_INDEX.get(factor)
            if idx is not None:
                F_new[idx, idx] = phi
        estimator.F = F_new

        # Also rebuild Q to match new persistence
        # Q_ii = (1 - phi^2) * scale
        Q_new = estimator.Q.copy()
        for i, factor in enumerate(FACTORS):
            old_phi = PERSISTENCE[factor]
            new_phi = persistence_override.get(factor, old_phi)
            if old_phi != new_phi:
                # Scale Q_ii proportionally: Q_new / Q_old = (1-new^2) / (1-old^2)
                old_var = 1 - old_phi ** 2
                new_var = 1 - new_phi ** 2
                if old_var > 0:
                    Q_new[i, i] *= new_var / old_var
        estimator.Q = Q_new

    imm = IMMBranchTracker()
    imm.initialize_branches(BRANCHES, baseline=estimator)

    # P3: Also override F for each branch estimator
    if persistence_override:
        for branch in imm.branches:
            if branch.estimator:
                branch.estimator.F = estimator.F.copy()
                branch.estimator.Q = estimator.Q.copy()

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

        # P2b: State-dependent TPM
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
            if persistence_override:
                # Rebuild Q with calibrated persistence
                diag = [(1 - persistence_override.get(f, PERSISTENCE[f]) ** 2) for f in FACTORS]
                base_Q = np.diag(diag) * 0.01  # Same scale factor as original
            estimator.Q = base_Q * q_scale
            for branch in imm.branches:
                if branch.estimator:
                    regime = REGIME_MAP.get(branch.branch_id, "expansion")
                    branch.estimator.Q = base_Q * REGIME_Q_TABLE.get(regime, 1.0)

        estimator.predict()
        imm.predict()

        # Update
        for col, kalman_key, val in today_updates:
            z = normalizer.normalize(col, val)

            if use_regime_noise:
                if kalman_key in estimator.stream_registry:
                    H_row, base_R = estimator.stream_registry[kalman_key]
                    r_blend = 0.0
                    for branch_id, prob in probs.items():
                        regime = REGIME_MAP.get(branch_id, "expansion")
                        r_mult = REGIME_R_TABLE.get(regime, {}).get(kalman_key, 1.0)
                        r_blend += prob * r_mult
                    estimator.stream_registry[kalman_key] = (H_row, base_R * r_blend)
                    update = estimator.update(kalman_key, z)
                    estimator.stream_registry[kalman_key] = (H_row, base_R)

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
    results = {}
    lb_pass = lb_total = bias_pass = bias_total = 0

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
            from statsmodels.tsa.stattools import acf
            lb_result = acorr_ljungbox(innovations, lags=[10], return_df=True)
            lb_pval = float(lb_result["lb_pvalue"].iloc[0])
            d["ljung_box_p"] = round(lb_pval, 4)
            d["ljung_box_pass"] = lb_pval > 0.05
            lb_total += 1
            if d["ljung_box_pass"]:
                lb_pass += 1

            acf_vals = acf(innovations, nlags=5, fft=True)
            d["acf_lag1"] = round(float(acf_vals[1]), 4)
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
        expected_probs = [e["probabilities"].get(expected, 0) for e in window]
        avg_prob = np.mean(expected_probs)
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
            "regime": expected, "description": description,
            "result": "PASS" if passed else "FAIL",
            "avg_probability": round(avg_prob, 4),
            "first_detection_weeks": detect_weeks,
        })
    return checkpoint_results


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 100)
    print("  P3 COMPARISON — Calibrated F_diag Persistence")
    print("=" * 100)

    # Show what we're changing
    print("\n  Persistence values:")
    print(f"  {'Factor':<24s} {'Theory':>8s} {'Calibrated':>11s} {'Monthly (T)':>12s} {'Monthly (C)':>12s}")
    print("  " + "-" * 70)
    for f in FACTORS:
        t = PERSISTENCE[f]
        c = CALIBRATED_PERSISTENCE[f]
        print(f"  {f:<24s} {t:>8.3f} {c:>11.4f} {t**22:>12.3f} {c**22:>12.3f}")

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        print(f"\nLoading cached data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    # A) Baseline
    print("\n--- Running BASELINE ---")
    baseline = run_engine(data, "baseline")
    print(f"  Done. {baseline.total_updates} updates")

    # B) P0+P2b (best so far)
    print("\n--- Running P0+P2b ---")
    best_so_far = run_engine(data, "p0+p2b", use_regime_noise=True, use_state_tpm=True)
    print(f"  Done. {best_so_far.total_updates} updates")

    # C) P0+P2b+P3 (calibrated persistence)
    print("\n--- Running P0+P2b+P3 (calibrated F_diag) ---")
    p3 = run_engine(data, "p0+p2b+p3", use_regime_noise=True, use_state_tpm=True,
                    persistence_override=CALIBRATED_PERSISTENCE)
    print(f"  Done. {p3.total_updates} updates")

    # Diagnostics
    configs = []
    for label, m in [("Baseline", baseline), ("P0+P2b", best_so_far), ("P0+P2b+P3", p3)]:
        configs.append({
            "name": label,
            "diag": compute_diagnostics(m),
            "cp": evaluate_checkpoints(m),
        })

    names = [c["name"] for c in configs]
    diags = [c["diag"] for c in configs]
    cps = [c["cp"] for c in configs]

    # Print comparison
    print()
    print("=" * 100)
    print("  COMPARISON")
    print("=" * 100)
    print()

    header = f"{'Metric':<35s}"
    for name in names:
        header += f" {name:>18s}"
    print(header)
    print("-" * 100)

    for label, key, fmt in [
        ("Ljung-Box pass rate", "ljung_box_rate", "{:>17.0%}"),
        ("Mean bias pass rate", "mean_bias_rate", "{:>17.0%}"),
        ("Total log-likelihood", "total_log_likelihood", "{:>18.0f}"),
        ("Anomaly rate", "anomaly_rate", "{:>17.1%}"),
    ]:
        row = f"{label:<35s}"
        for d in diags:
            row += " " + fmt.format(d[key])
        print(row)

    row = f"{'Regime checkpoints':<35s}"
    for cp in cps:
        passed = sum(1 for c in cp if c["result"] == "PASS")
        total = sum(1 for c in cp if c["result"] != "SKIP")
        row += f" {f'{passed}/{total}':>18s}"
    print(row)

    # Per-stream with ACF(1)
    print()
    streams = sorted(diags[0]["per_stream"].keys())
    header = f"{'Stream':<25s}"
    for name in names:
        header += f"  {'LB_p':>5s} {'ACF1':>6s} {'Bias':>6s} {'LL':>6s}"
    print(header)
    print("-" * 100)
    for stream in streams:
        row = f"{stream:<25s}"
        for d in diags:
            s = d["per_stream"].get(stream, {})
            lb = f"{s.get('ljung_box_p', 0):.3f}"
            acf1 = f"{s.get('acf_lag1', 0):.3f}" if 'acf_lag1' in s else "N/A"
            bias = f"{s.get('mean', 0):+.2f}"
            ll = f"{s.get('mean_log_lik', 0):.2f}" if 'mean_log_lik' in s else "N/A"
            row += f"  {lb:>5s} {acf1:>6s} {bias:>6s} {ll:>6s}"
        print(row)

    # Checkpoints
    print()
    header = f"{'Checkpoint':<50s}"
    for name in names:
        header += f"  {'avg':>6s} {'det':>6s}"
    print(header)
    print("-" * 100)
    for i in range(len(cps[0])):
        if cps[0][i]["result"] == "SKIP":
            continue
        row = f"  {cps[0][i]['description'][:48]:<48s}"
        for cp in cps:
            avg = f"{cp[i]['avg_probability']:.1%}"
            det = f"{cp[i]['first_detection_weeks']:.1f}w" if cp[i].get('first_detection_weeks') is not None else "N/D"
            row += f"  {avg:>6s} {det:>6s}"
        print(row)

    # Save results
    results = {
        "experiment": "p3_comparison",
        "persistence_values": {
            "theory": dict(PERSISTENCE),
            "calibrated": dict(CALIBRATED_PERSISTENCE),
        },
    }
    for label, d, cp in zip(names, diags, cps):
        key = label.lower().replace("+", "_").replace(" ", "_")
        results[key] = {"diagnostics": d, "checkpoints": cp}

    output_path = Path(__file__).parent.parent / "data" / "results" / "p3_comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Verdict
    prev_ll = diags[1]["total_log_likelihood"]
    p3_ll = diags[2]["total_log_likelihood"]
    baseline_ll = diags[0]["total_log_likelihood"]
    prev_lb = diags[1]["ljung_box_rate"]
    p3_lb = diags[2]["ljung_box_rate"]
    prev_cp = sum(1 for c in cps[1] if c["result"] == "PASS")
    p3_cp = sum(1 for c in cps[2] if c["result"] == "PASS")

    print()
    print("=" * 100)
    ll_delta = p3_ll - prev_ll
    if p3_lb > prev_lb:
        print(f"  LJUNG-BOX IMPROVED: {prev_lb:.0%} → {p3_lb:.0%}")
    else:
        print(f"  LJUNG-BOX UNCHANGED: {p3_lb:.0%}")

    # Show ACF improvement
    print()
    print("  ACF(1) comparison (lower = better):")
    print(f"  {'Stream':<25s} {'P0+P2b':>8s} {'P0+P2b+P3':>10s} {'Delta':>8s}")
    for stream in streams:
        prev_acf = diags[1]["per_stream"].get(stream, {}).get("acf_lag1", 0)
        p3_acf = diags[2]["per_stream"].get(stream, {}).get("acf_lag1", 0)
        if prev_acf and p3_acf:
            delta = p3_acf - prev_acf
            marker = " ✓" if delta < -0.05 else ""
            print(f"  {stream:<25s} {prev_acf:>8.3f} {p3_acf:>10.3f} {delta:>+8.3f}{marker}")

    print()
    if ll_delta > 0 and p3_cp >= prev_cp:
        print(f"  VERDICT: P3 WINS — LL {ll_delta:+.0f}, checkpoints held")
    elif ll_delta > 0:
        print(f"  VERDICT: P3 MIXED — LL {ll_delta:+.0f} but lost {prev_cp - p3_cp} checkpoint(s)")
    else:
        print(f"  VERDICT: P3 NO IMPROVEMENT — LL {ll_delta:+.0f}")
    print(f"  Cumulative over baseline: LL {p3_ll - baseline_ll:+.0f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
