"""
P2b A/B/C Comparison — Baseline vs P0 vs P0+P2b (State-Dependent TPM)

Filardo (1994): factor values drive transition probabilities via logistic
link. When financial_conditions deteriorate, expansion→contraction
probability increases. When inflation_trend rises, expansion→stagflation
probability increases. The TPM becomes a function of the economy's
current state, not a fixed matrix.

Also includes detailed Ljung-Box diagnostics to understand WHY the
autocorrelation persists across all variants.

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/p2b_comparison.py
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
    LEVEL_A_TPM,
)
from heimdall.regime_noise import REGIME_R_TABLE, REGIME_Q_TABLE
from heimdall.state_tpm import build_state_adjusted_tpm

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
) -> EngineMetrics:
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

        # --- P2b: State-dependent TPM ---
        if use_state_tpm and day_idx % 7 == 0:
            state = estimator.get_state()
            factor_values = {
                name: float(state.mean[i]) for i, name in enumerate(FACTORS)
            }
            adjusted_tpm = build_state_adjusted_tpm(LEVEL_A_TPM, factor_values)
            imm.tpm = adjusted_tpm

        # --- Predict ---
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

        # --- Update ---
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


def ljung_box_deep_diagnostic(metrics: EngineMetrics):
    """Deep diagnostic: WHY is Ljung-Box failing for every stream?

    Computes:
    1. ACF at lags 1-10 for each stream
    2. Identifies dominant lag structure
    3. Checks for observation frequency effects
    4. Tests whether innovations are autocorrelated because of
       repeated observations (monthly data fed daily)
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    print()
    print("=" * 90)
    print("  LJUNG-BOX DEEP DIAGNOSTIC — Why is autocorrelation persistent?")
    print("=" * 90)

    acf_data = {}

    for stream in sorted(metrics.innovation_log.keys()):
        innovations = np.array(metrics.innovation_log[stream])
        n = len(innovations)
        if n < 50:
            continue

        # ACF at lags 1-10
        acf_vals = acf(innovations, nlags=10, fft=True)

        # Per-lag Ljung-Box
        lb_result = acorr_ljungbox(innovations, lags=list(range(1, 11)), return_df=True)

        # Count consecutive identical innovations (proxy for repeated observations)
        diffs = np.diff(innovations)
        zero_diffs = np.sum(np.abs(diffs) < 1e-10)
        repeat_pct = zero_diffs / len(diffs)

        # Run-length of repeated values
        runs = []
        current_run = 1
        for i in range(1, len(innovations)):
            if abs(innovations[i] - innovations[i-1]) < 1e-10:
                current_run += 1
            else:
                if current_run > 1:
                    runs.append(current_run)
                current_run = 1
        if current_run > 1:
            runs.append(current_run)
        max_run = max(runs) if runs else 0
        avg_run = np.mean(runs) if runs else 0
        n_runs = len(runs)

        # First lag where Ljung-Box passes (if any)
        first_pass_lag = None
        for lag in range(1, 11):
            p = float(lb_result.loc[lag, "lb_pvalue"])
            if p > 0.05:
                first_pass_lag = lag
                break

        acf_data[stream] = {
            "acf_lag1": round(acf_vals[1], 4),
            "acf_lag2": round(acf_vals[2], 4),
            "acf_lag5": round(acf_vals[5], 4),
            "acf_lag10": round(acf_vals[10], 4),
            "repeat_pct": round(repeat_pct, 4),
            "max_run": max_run,
            "avg_run": round(avg_run, 1),
            "n_runs": n_runs,
            "n_obs": n,
            "first_pass_lag": first_pass_lag,
        }

    # Print results
    print()
    print(f"{'Stream':<25s} {'ACF(1)':>7s} {'ACF(2)':>7s} {'ACF(5)':>7s} {'ACF(10)':>7s} "
          f"{'%repeat':>8s} {'max_run':>8s} {'avg_run':>8s} {'n_runs':>7s} {'n_obs':>7s}")
    print("-" * 90)

    for stream, d in sorted(acf_data.items()):
        print(f"{stream:<25s} {d['acf_lag1']:>7.3f} {d['acf_lag2']:>7.3f} "
              f"{d['acf_lag5']:>7.3f} {d['acf_lag10']:>7.3f} "
              f"{d['repeat_pct']:>7.1%} {d['max_run']:>8d} "
              f"{d['avg_run']:>8.1f} {d['n_runs']:>7d} {d['n_obs']:>7d}")

    # Summary interpretation
    print()
    print("--- INTERPRETATION ---")
    high_repeat = [s for s, d in acf_data.items() if d["repeat_pct"] > 0.5]
    high_acf1 = [s for s, d in acf_data.items() if abs(d["acf_lag1"]) > 0.3]
    low_repeat_high_acf = [s for s, d in acf_data.items()
                           if d["repeat_pct"] < 0.3 and abs(d["acf_lag1"]) > 0.1]

    if high_repeat:
        print(f"\n  REPEATED OBSERVATIONS ({len(high_repeat)} streams):")
        print(f"    {', '.join(high_repeat)}")
        print(f"    These streams have >50% identical consecutive innovations.")
        print(f"    Root cause: monthly/quarterly data fed as daily observations.")
        print(f"    The filter sees the same z-score for 20-30 days, producing")
        print(f"    near-identical innovations each day. This is structural —")
        print(f"    Ljung-Box WILL always fail for these streams unless we only")
        print(f"    update on new observations.")

    if low_repeat_high_acf:
        print(f"\n  GENUINE AUTOCORRELATION ({len(low_repeat_high_acf)} streams):")
        print(f"    {', '.join(low_repeat_high_acf)}")
        print(f"    These streams have low repeat rate but high ACF(1).")
        print(f"    Root cause: F matrix misspecification (AR(1) doesn't capture")
        print(f"    the actual dynamics) OR H matrix mismatch OR the rolling")
        print(f"    normalizer introduces correlation.")

    all_streams = list(acf_data.keys())
    neither = [s for s in all_streams if s not in high_repeat and s not in low_repeat_high_acf]
    if neither:
        print(f"\n  MILD AUTOCORRELATION ({len(neither)} streams):")
        print(f"    {', '.join(neither)}")
        print(f"    Low repeat rate AND low ACF(1), but Ljung-Box still fails.")
        print(f"    Joint test at 10 lags may be catching small but persistent")
        print(f"    autocorrelation across multiple lags.")

    return acf_data


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 90)
    print("  P2b A/B/C — Baseline vs P0 vs P0+P2b (State-Dependent TPM)")
    print("=" * 90)

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        print(f"Loading cached data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    print("\n--- Running BASELINE ---")
    baseline = run_engine(data, "baseline")
    print(f"  Done. {baseline.total_updates} updates")

    print("\n--- Running P0 (regime R/Q) ---")
    p0 = run_engine(data, "p0", use_regime_noise=True)
    print(f"  Done. {p0.total_updates} updates")

    print("\n--- Running P0+P2b (regime R/Q + state TPM) ---")
    p0p2b = run_engine(data, "p0+p2b", use_regime_noise=True, use_state_tpm=True)
    print(f"  Done. {p0p2b.total_updates} updates")

    # Diagnostics
    configs = []
    for label, m in [("Baseline", baseline), ("P0", p0), ("P0+P2b", p0p2b)]:
        configs.append({
            "name": label,
            "diag": compute_diagnostics(m),
            "cp": evaluate_checkpoints(m),
        })

    # Print comparison
    names = [c["name"] for c in configs]
    diags = [c["diag"] for c in configs]
    cps = [c["cp"] for c in configs]

    print()
    print("=" * 90)
    print("  A/B/C COMPARISON")
    print("=" * 90)
    print()

    header = f"{'Metric':<35s}"
    for name in names:
        header += f" {name:>16s}"
    print(header)
    print("-" * 90)

    row = f"{'Ljung-Box pass rate':<35s}"
    for d in diags:
        row += f" {d['ljung_box_rate']:>15.0%}"
    print(row)

    row = f"{'Mean bias pass rate':<35s}"
    for d in diags:
        row += f" {d['mean_bias_rate']:>15.0%}"
    print(row)

    row = f"{'Total log-likelihood':<35s}"
    for d in diags:
        row += f" {d['total_log_likelihood']:>16.0f}"
    print(row)

    row = f"{'Anomaly rate':<35s}"
    for d in diags:
        row += f" {d['anomaly_rate']:>15.1%}"
    print(row)

    row = f"{'Regime checkpoints':<35s}"
    for cp in cps:
        passed = sum(1 for c in cp if c["result"] == "PASS")
        total = sum(1 for c in cp if c["result"] != "SKIP")
        row += f" {f'{passed}/{total}':>16s}"
    print(row)

    # Per-stream
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
            lb = f"{s.get('ljung_box_p', 0):.3f}"
            bias = f"{s.get('mean', 0):+.2f}"
            ll = f"{s.get('mean_log_lik', 0):.2f}" if 'mean_log_lik' in s else "N/A"
            row += f"  {lb:>6s} {bias:>6s} {ll:>6s}"
        print(row)

    # Checkpoints
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

    # Deep Ljung-Box diagnostic on P0 (the best engine)
    acf_data = ljung_box_deep_diagnostic(p0)

    # Save results
    p0_ll = diags[1]["total_log_likelihood"]
    p2b_ll = diags[2]["total_log_likelihood"]
    baseline_ll = diags[0]["total_log_likelihood"]

    results = {
        "experiment": "p2b_comparison",
        "baseline": {"diagnostics": diags[0], "checkpoints": cps[0]},
        "p0": {"diagnostics": diags[1], "checkpoints": cps[1]},
        "p0_p2b": {"diagnostics": diags[2], "checkpoints": cps[2]},
        "ljung_box_diagnostic": acf_data,
    }

    output_path = Path(__file__).parent.parent / "data" / "results" / "p2b_comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Verdict
    ll_delta = p2b_ll - p0_ll
    p0_cp = sum(1 for c in cps[1] if c["result"] == "PASS")
    p2b_cp = sum(1 for c in cps[2] if c["result"] == "PASS")
    print()
    print("=" * 90)
    if ll_delta > 0 and p2b_cp >= p0_cp:
        print(f"  VERDICT: P0+P2b WINS — LL delta {ll_delta:+.0f} over P0 alone")
    elif ll_delta > 0 and p2b_cp < p0_cp:
        print(f"  VERDICT: MIXED — LL {ll_delta:+.0f} but lost {p0_cp - p2b_cp} checkpoint(s)")
    else:
        print(f"  VERDICT: P0+P2b NO IMPROVEMENT over P0 — LL delta {ll_delta:+.0f}")
    print(f"  Cumulative over baseline: LL {p2b_ll - baseline_ll:+.0f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
