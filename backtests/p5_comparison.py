"""
P5 A/B/C — P0+P2b vs P0+P2b+P4 vs P0+P2b+P4+P5 (GAS Noise)

Score-driven (GAS) observation noise: R adapts at each observation
using the score of the conditional log-likelihood.  When innovations
are larger than expected, R increases.  When smaller, R decreases.

This targets the heteroskedasticity that P0 (regime-dependent R) only
partially captures.  P0 uses 3 discrete noise profiles (expansion,
stagflation, contraction).  GAS provides continuous, stream-specific
adaptation — R for CPI can increase during supply chain disruptions
without waiting for a regime switch.

A = P0+P2b (previous shipping config)
B = P0+P2b+P4 (with cumulator)
C = P0+P2b+P4+P5 (cumulator + GAS noise)

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/p5_comparison.py
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
from heimdall.cumulator import (
    StreamCumulator,
    CUMULATED_STREAMS,
    compute_gap_adjusted_R,
)
from heimdall.gas_noise import GASNoiseTracker

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
    use_cumulator: bool = False,
    use_gas: bool = False,
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

    cumulator = StreamCumulator() if use_cumulator else None

    # P5: Initialize GAS tracker with base R for each stream
    gas_tracker = None
    if use_gas:
        gas_tracker = GASNoiseTracker()
        for stream_key, (_, base_R) in estimator.stream_registry.items():
            gas_tracker.initialize(stream_key, base_R)

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

            # P5: GAS-adjusted R (replaces base_R)
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

            # --- Kalman update with effective R ---
            estimator.stream_registry[kalman_key] = (H_row, effective_R)
            S = float(H_row @ estimator.P @ H_row.T + effective_R)
            K = estimator.P @ H_row.T / S
            innovation_zscore = innovation / np.sqrt(S)

            estimator.x = estimator.x + K * innovation
            I_KH = np.eye(N_FACTORS) - np.outer(K, H_row)
            estimator.P = I_KH @ estimator.P @ I_KH.T + np.outer(K, K) * effective_R

            # Restore base R
            estimator.stream_registry[kalman_key] = (H_row, base_R)

            # P5: Update GAS tracker with raw innovation
            if gas_tracker is not None:
                gas_tracker.update(kalman_key, innovation, S)

            # Reset cumulator
            if cumulator is not None and cumulator.should_cumulate(kalman_key):
                cumulator.reset(kalman_key)

            # IMM update (standard path with regime noise)
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

            # Construct update object
            update = type('Update', (), {
                'stream_key': kalman_key,
                'innovation_zscore': float(innovation_zscore),
                'is_anomalous': abs(float(innovation_zscore)) > ANOMALY_THRESHOLD,
            })()

            total_updates += 1
            if not is_warmup:
                innovation_log[kalman_key].append(update.innovation_zscore)
                S_ll = float(H_row @ estimator.P @ H_row.T + base_R)
                ll = -0.5 * (update.innovation_zscore**2 + np.log(2 * np.pi * S_ll))
                log_likelihood_log[kalman_key].append(ll)
                total_log_lik += ll
                if update.is_anomalous:
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
            from statsmodels.tsa.stattools import acf as compute_acf
            lb_result = acorr_ljungbox(innovations, lags=[10], return_df=True)
            lb_pval = float(lb_result["lb_pvalue"].iloc[0])
            d["ljung_box_p"] = round(lb_pval, 4)
            d["ljung_box_pass"] = lb_pval > 0.05
            lb_total += 1
            if d["ljung_box_pass"]:
                lb_pass += 1

            acf_vals = compute_acf(innovations, nlags=1, fft=True)
            d["acf_lag1"] = round(float(acf_vals[1]), 4)
        except Exception:
            d["ljung_box_p"] = None
            d["ljung_box_pass"] = None
            d["acf_lag1"] = None

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

    print("=" * 90)
    print("  P5 A/B/C — P0+P2b vs P0+P2b+P4 vs P0+P2b+P4+P5 (GAS Noise)")
    print("=" * 90)

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        print(f"Loading cached data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    print("\n--- Running P0+P2b (shipping config) ---")
    p0p2b = run_engine(data, "p0+p2b", use_regime_noise=True, use_state_tpm=True)
    print(f"  Done. {p0p2b.total_updates} updates")

    print("\n--- Running P0+P2b+P4 (cumulator) ---")
    p0p2b_p4 = run_engine(
        data, "p0+p2b+p4",
        use_regime_noise=True, use_state_tpm=True, use_cumulator=True,
    )
    print(f"  Done. {p0p2b_p4.total_updates} updates")

    print("\n--- Running P0+P2b+P4+P5 (cumulator + GAS) ---")
    full = run_engine(
        data, "p0+p2b+p4+p5",
        use_regime_noise=True, use_state_tpm=True,
        use_cumulator=True, use_gas=True,
    )
    print(f"  Done. {full.total_updates} updates")

    configs = []
    for label, m in [("P0+P2b", p0p2b), ("P0+P2b+P4", p0p2b_p4), ("+P5 GAS", full)]:
        configs.append({"name": label, "diag": compute_diagnostics(m), "cp": evaluate_checkpoints(m)})

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

    for metric_name, key in [
        ("Ljung-Box pass rate", "ljung_box_rate"),
        ("Mean bias pass rate", "mean_bias_rate"),
        ("Total log-likelihood", "total_log_likelihood"),
        ("Anomaly rate", "anomaly_rate"),
    ]:
        row = f"{metric_name:<35s}"
        for d in diags:
            val = d[key]
            if key == "total_log_likelihood":
                row += f" {val:>16.0f}"
            elif key == "anomaly_rate":
                row += f" {val:>15.1%}"
            else:
                row += f" {val:>15.0%}"
        print(row)

    row = f"{'Regime checkpoints':<35s}"
    for cp in cps:
        passed = sum(1 for c in cp if c["result"] == "PASS")
        total = sum(1 for c in cp if c["result"] != "SKIP")
        row += f" {f'{passed}/{total}':>16s}"
    print(row)

    # Per-stream ACF(1) — the key diagnostic
    print()
    print("  ACF(1) and Ljung-Box per stream")
    print()
    streams = sorted(diags[0]["per_stream"].keys())
    header = f"{'Stream':<25s}"
    for name in names:
        header += f"  {'ACF(1)':>7s} {'LBp':>6s}"
    print(header)
    print("-" * 90)
    for stream in streams:
        row = f"{stream:<25s}"
        for d in diags:
            s = d["per_stream"].get(stream, {})
            acf1 = f"{s.get('acf_lag1', 0):.3f}" if s.get("acf_lag1") is not None else "  N/A"
            lb = f"{s.get('ljung_box_p', 0):.3f}"
            row += f"  {acf1:>7s} {lb:>6s}"
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
            det = (
                f"{cp[i]['first_detection_weeks']:.1f}w"
                if cp[i].get('first_detection_weeks') is not None
                else "N/D"
            )
            row += f"  {avg:>6s} {det:>6s}"
        print(row)

    # Save
    results = {
        "experiment": "p5_comparison",
        "p0_p2b": {"diagnostics": diags[0], "checkpoints": cps[0]},
        "p0_p2b_p4": {"diagnostics": diags[1], "checkpoints": cps[1]},
        "p0_p2b_p4_p5": {"diagnostics": diags[2], "checkpoints": cps[2]},
    }
    output_path = Path(__file__).parent.parent / "data" / "results" / "p5_comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Verdict
    p4_ll = diags[1]["total_log_likelihood"]
    p5_ll = diags[2]["total_log_likelihood"]
    ll_delta = p5_ll - p4_ll
    p4_cp = sum(1 for c in cps[1] if c["result"] == "PASS")
    p5_cp = sum(1 for c in cps[2] if c["result"] == "PASS")
    p4_lb = diags[1]["ljung_box_rate"]
    p5_lb = diags[2]["ljung_box_rate"]

    print()
    print("=" * 90)
    print(f"  LL delta (P5 vs P4): {ll_delta:+.0f}")
    print(f"  Checkpoints: P4 {p4_cp}/11, P5 {p5_cp}/11")
    print(f"  Ljung-Box: P4 {p4_lb:.0%}, P5 {p5_lb:.0%}")
    if ll_delta > 0 and p5_cp >= p4_cp:
        print("  VERDICT: P5 SHIPS")
    elif p5_cp < p4_cp:
        print(f"  VERDICT: P5 REGRESSES — lost {p4_cp - p5_cp} checkpoint(s)")
    else:
        print("  VERDICT: P5 NEUTRAL")
    print("=" * 90)


if __name__ == "__main__":
    main()
