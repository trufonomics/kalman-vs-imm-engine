"""
Stagflation Branch Adjustment Fix — Correcting growth_trend Sign.

The current stagflation branch has growth_trend = +0.195 (positive),
derived from the 2021-22 episode alone. During 2021-22, GDP growth was
actually positive — making it an "inflationary boom," not canonical
stagflation.

Classic stagflation (1970s: oil embargo, second oil shock) features:
  - High inflation (CPI > 5%)
  - Weak or negative growth
  - Rising unemployment
  - Depressed consumer sentiment

This script tests corrected stagflation adjustments against the
current ones to see if stagflation AUC improves.

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/stagflation_fix_comparison.py
"""

import sys
import json
import os
import copy
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
    RECOMMENDED_BRANCH_ADJUSTMENTS,
)
from heimdall.regime_noise import REGIME_R_TABLE, REGIME_Q_TABLE
from heimdall.state_tpm import build_state_adjusted_tpm
from heimdall.cumulator import (
    StreamCumulator,
    CUMULATED_STREAMS,
    compute_gap_adjusted_R,
)
from heimdall.gas_noise import GASNoiseTracker
from heimdall.regime_diagnostics import run_full_diagnostics

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


# ── Corrected stagflation adjustments ─────────────────────────────────
# Theory-based corrections for canonical stagflation (1970s + 2021-22):
#
# growth_trend: 0.195 → -0.10 (stagnant/weak growth, not positive)
# consumer_sentiment: 0.100 → -0.20 (depressed during stagflation)
# housing_momentum: 0.360 → 0.0 (remove — specific to 2021-22 housing bubble)
#
# Keep inflation_trend (1.063), labor_pressure (0.320),
# financial_conditions (0.247), policy_stance (0.171) — these match
# the canonical pattern.

CORRECTED_STAGFLATION = {
    "inflation_trend": 1.063,       # HIGH inflation — unchanged (correct)
    "growth_trend": -0.10,          # WEAK growth (was +0.195)
    "labor_pressure": 0.320,        # unchanged
    "financial_conditions": 0.247,  # unchanged
    "consumer_sentiment": -0.20,    # DEPRESSED sentiment (was +0.10)
    "policy_stance": 0.171,         # unchanged
}

# Also test a blend: average of current (2021-22 derived) and theory
BLENDED_STAGFLATION = {
    "inflation_trend": 1.063,
    "growth_trend": 0.05,           # blend: (0.195 + -0.10) / 2 ≈ 0.05
    "labor_pressure": 0.320,
    "housing_momentum": 0.180,      # halved
    "financial_conditions": 0.247,
    "consumer_sentiment": -0.05,    # blend: (0.10 + -0.20) / 2 = -0.05
    "policy_stance": 0.171,
}


def make_branches(stagflation_adj: dict) -> list[dict]:
    """Create branch configs with custom stagflation adjustments."""
    return [
        {
            "branch_id": "soft_landing",
            "name": "Soft Landing",
            "probability": 1 / 3,
            "state_adjustments": RECOMMENDED_BRANCH_ADJUSTMENTS["soft_landing"],
        },
        {
            "branch_id": "stagflation",
            "name": "Stagflation",
            "probability": 1 / 3,
            "state_adjustments": stagflation_adj,
        },
        {
            "branch_id": "recession",
            "name": "Recession",
            "probability": 1 / 3,
            "state_adjustments": RECOMMENDED_BRANCH_ADJUSTMENTS["recession"],
        },
    ]


def run_engine(data, engine_name, branches):
    """Run full shipping stack with custom branches."""
    col_to_kalman = {}
    for col in data.columns:
        parts = col.split("_", 1)
        if len(parts) == 2:
            col_to_kalman[col] = parts[1]

    estimator = EconomicStateEstimator()
    imm = IMMBranchTracker()
    imm.initialize_branches(branches, baseline=estimator)

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

        # P2b: State-dependent TPM
        if day_idx % 7 == 0:
            state = estimator.get_state()
            factor_values = {
                name: float(state.mean[i]) for i, name in enumerate(FACTORS)
            }
            adjusted_tpm = build_state_adjusted_tpm(LEVEL_A_TPM, factor_values)
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
            effective_R = base_R

            gas_R = gas_tracker.get_R(kalman_key)
            if gas_R is not None:
                effective_R = gas_R

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

            use_cumulated = (
                cumulator.should_cumulate(kalman_key)
                and cumulator.get_cumulated_prediction(kalman_key) is not None
            )
            if use_cumulated:
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

    return daily_log, total_log_lik, anomaly_count, total_updates


def main():
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 90)
    print("  STAGFLATION BRANCH ADJUSTMENT FIX")
    print("  Current: growth_trend=+0.195 (from 2021-22 only — positive growth)")
    print("  Theory:  growth_trend=-0.10  (canonical stagflation = weak growth)")
    print("=" * 90)

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    configs = {
        "Current (growth=+0.195)": make_branches(
            RECOMMENDED_BRANCH_ADJUSTMENTS["stagflation"]
        ),
        "Theory (growth=-0.10)": make_branches(CORRECTED_STAGFLATION),
        "Blended (growth=+0.05)": make_branches(BLENDED_STAGFLATION),
    }

    results = {}
    for name, branches in configs.items():
        print(f"\n--- {name} ---")
        daily_log, ll, anomaly, updates = run_engine(data, name, branches)
        print(f"  {updates} updates, LL = {ll:.0f}, anomaly = {anomaly}")

        diag = run_full_diagnostics(daily_log, REGIME_CHECKPOINTS)
        results[name] = {
            "daily_log": daily_log,
            "ll": ll,
            "diag": diag,
        }

    # ── Comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  RESULTS")
    print("=" * 90)

    print(f"\n  {'Config':<30s} {'Stag AUC':>10s} {'Contr AUC':>10s} "
          f"{'Stag Brier':>10s} {'LL':>10s}")
    print("  " + "-" * 74)

    for name, r in results.items():
        d = r["diag"]
        stag_auc = d["roc"]["stagflation"].auc
        contr_auc = d["roc"]["contraction"].auc
        stag_brier = d["brier"]["stagflation"].brier_score
        ll = r["ll"]
        print(f"  {name:<30s} {stag_auc:>10.4f} {contr_auc:>10.4f} "
              f"{stag_brier:>10.4f} {ll:>10.0f}")

    # Detection comparison
    print(f"\n  DETECTION (50% / 80% thresholds)")
    for name, r in results.items():
        d = r["diag"]
        det_50 = sum(1 for dl in d["detection_lag"] if dl.detection_lag_weeks is not None)
        det_80 = sum(1 for dl in d["detection_lag"] if dl.detection_lag_weeks_80 is not None)
        print(f"\n  {name}: {det_50}/11 @50%, {det_80}/11 @80%")
        for dl in d["detection_lag"]:
            lag50 = f"{dl.detection_lag_weeks:.1f}w" if dl.detection_lag_weeks is not None else "MISSED"
            lag80 = f"{dl.detection_lag_weeks_80:.1f}w" if dl.detection_lag_weeks_80 is not None else "MISSED"
            print(f"    {lag50:>8s}/{lag80:>8s}  {dl.event}")

    # Save
    results_dir = Path(__file__).parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save = {}
    for name, r in results.items():
        d = r["diag"]
        save[name] = {
            "ll": round(r["ll"], 2),
            "stagflation_auc": d["roc"]["stagflation"].auc,
            "contraction_auc": d["roc"]["contraction"].auc,
            "stagflation_brier": d["brier"]["stagflation"].brier_score,
            "events_detected_50": sum(1 for dl in d["detection_lag"]
                                      if dl.detection_lag_weeks is not None),
            "events_detected_80": sum(1 for dl in d["detection_lag"]
                                      if dl.detection_lag_weeks_80 is not None),
        }

    out_path = results_dir / "stagflation_fix_results.json"
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
