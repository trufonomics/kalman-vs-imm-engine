"""
Sensitivity Analysis — Fix 7

Perturbs key parameters (branch adjustments, F matrix persistence,
R observation noise, TPM diagonal) and checks whether regime checkpoint
results remain robust. A well-calibrated system should tolerate ±30-50%
perturbation without catastrophic failure.

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/sensitivity_analysis.py
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    FACTORS,
    FACTOR_INDEX,
    N_FACTORS,
)
from heimdall.imm_tracker import (
    IMMBranchTracker,
    EMPIRICAL_BRANCH_ADJUSTMENTS,
    DEFAULT_TPM,
)

# ── Backtest column → Kalman stream key mapping ─────────────────────
COLUMN_TO_KALMAN = {
    "FRED_US_CPI_YOY": "US_CPI_YOY",
    "FRED_CORE_CPI": "CORE_CPI",
    "FRED_PPI": "PPI",
    "FRED_UNEMPLOYMENT_RATE": "UNEMPLOYMENT_RATE",
    "FRED_INITIAL_CLAIMS": "INITIAL_CLAIMS",
    "FRED_HOUSING_STARTS": "HOUSING_STARTS",
    "FRED_HOME_PRICES": "HOME_PRICES",
    "FRED_RETAIL_SALES": "RETAIL_SALES",
    "FRED_CONSUMER_CONFIDENCE": "CONSUMER_CONFIDENCE",
    "FRED_FED_FUNDS_RATE": "FED_FUNDS_RATE",
    "FRED_10Y_YIELD": "10Y_YIELD",
    "YAHOO_SP500": "SP500",
    "YAHOO_OIL_PRICE": "OIL_PRICE",
    "YAHOO_GOLD_PRICE": "GOLD_PRICE",
    "YAHOO_BTC_USD": "BTC_USD",
}

# Regime checkpoints: (start, end, expected, threshold, description)
REGIME_CHECKPOINTS = [
    ("2008-06-01", "2009-06-01", "recession", 0.35,
     "Great Recession"),
    ("2020-03-01", "2020-06-01", "recession", 0.30,
     "COVID crash"),
    ("2021-06-01", "2022-06-01", "stagflation", 0.30,
     "Inflation surge"),
    ("2023-06-01", "2025-12-01", "soft_landing", 0.35,
     "Disinflation"),
]


class RollingNormalizer:
    """EMA-based normalizer."""

    def __init__(self, halflife_days: int = 90):
        self.alpha = 1 - np.exp(-np.log(2) / halflife_days)
        self.ema_mean: dict[str, float] = {}
        self.ema_var: dict[str, float] = {}
        self.initialized: dict[str, bool] = {}

    def initialize(self, key: str, mean: float, std: float):
        self.ema_mean[key] = mean
        self.ema_var[key] = max(std, 0.01) ** 2
        self.initialized[key] = True

    def normalize(self, key: str, raw: float) -> float:
        if key not in self.initialized:
            return 0.0
        mean = self.ema_mean[key]
        std = max(np.sqrt(self.ema_var[key]), 1e-8)
        z = (raw - mean) / std
        self.ema_mean[key] = (1 - self.alpha) * mean + self.alpha * raw
        deviation_sq = (raw - self.ema_mean[key]) ** 2
        self.ema_var[key] = (1 - self.alpha) * self.ema_var[key] + self.alpha * deviation_sq
        return z


def perturb_adjustments(scale: float) -> dict:
    """Scale all branch adjustments by a factor."""
    perturbed = {}
    for regime, adjs in EMPIRICAL_BRANCH_ADJUSTMENTS.items():
        perturbed[regime] = {k: v * scale for k, v in adjs.items()}
    return perturbed


def perturb_tpm(diagonal_delta: float) -> np.ndarray:
    """Shift TPM diagonal by delta, keeping rows summing to 1."""
    tpm = DEFAULT_TPM.copy()
    n = tpm.shape[0]
    for i in range(n):
        old_diag = tpm[i, i]
        new_diag = np.clip(old_diag + diagonal_delta, 0.80, 0.999)
        off_diag_sum = 1.0 - old_diag
        if off_diag_sum > 1e-10:
            scale = (1.0 - new_diag) / off_diag_sum
            for j in range(n):
                if j != i:
                    tpm[i, j] *= scale
        tpm[i, i] = new_diag
    return tpm


def run_trial(
    data: pd.DataFrame,
    adjustments: dict,
    tpm: np.ndarray,
    f_perturbation: float = 0.0,
    r_perturbation: float = 0.0,
) -> dict:
    """Run full 2007-2026 backtest with perturbed parameters."""
    # Build branches
    branches = []
    for bid, name in [("soft_landing", "Soft Landing"),
                       ("stagflation", "Stagflation"),
                       ("recession", "Recession")]:
        branches.append({
            "id": bid,
            "name": name,
            "probability": 1.0 / 3,
            "state_adjustments": adjustments.get(bid, {}),
            "transition_overrides": [],
        })

    # Create baseline estimator
    baseline = EconomicStateEstimator()

    if f_perturbation != 0:
        for i in range(N_FACTORS):
            baseline.F[i, i] = np.clip(baseline.F[i, i] + f_perturbation, 0.70, 0.999)

    if r_perturbation != 0:
        new_registry = {}
        for stream_key, (H_row, R) in baseline.stream_registry.items():
            new_R = R * (1.0 + r_perturbation)
            new_registry[stream_key] = (H_row, max(new_R, 0.001))
        baseline.stream_registry = new_registry

    # Initialize tracker
    tracker = IMMBranchTracker(tpm=tpm.copy())
    tracker.initialize_branches(branches, baseline)

    # Initialize normalizer from first 90 days
    normalizer = RollingNormalizer(halflife_days=90)
    warmup_end = min(90, len(data))
    warmup = data.iloc[:warmup_end]
    for col in data.columns:
        kalman_key = COLUMN_TO_KALMAN.get(col)
        if kalman_key:
            valid = warmup[col].dropna()
            if len(valid) > 5:
                normalizer.initialize(kalman_key, float(valid.mean()), float(valid.std()))

    # Run through all data
    daily_probs = []
    for date, row in data.iterrows():
        tracker.predict()
        for col, val in row.items():
            if pd.notna(val):
                kalman_key = COLUMN_TO_KALMAN.get(col)
                if kalman_key and kalman_key in normalizer.initialized:
                    z = normalizer.normalize(kalman_key, float(val))
                    z = np.clip(z, -4, 4)
                    tracker.update(kalman_key, z)

        probs = tracker.get_probabilities()
        daily_probs.append((date, probs))

    # Evaluate checkpoints
    results = {"checkpoints_passed": 0, "checkpoint_details": {}}

    for start, end, expected, threshold, desc in REGIME_CHECKPOINTS:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        period_probs = [
            p.get(expected, 0)
            for d, p in daily_probs
            if start_ts <= d <= end_ts
        ]

        if not period_probs:
            results["checkpoint_details"][desc] = {
                "passed": False, "avg_prob": 0, "max_prob": 0,
            }
            continue

        avg_prob = float(np.mean(period_probs))
        max_prob = float(np.max(period_probs))
        dominant_pct = float(np.mean([1 if p >= threshold else 0 for p in period_probs]))

        passed = avg_prob >= threshold
        if passed:
            results["checkpoints_passed"] += 1

        results["checkpoint_details"][desc] = {
            "passed": passed,
            "avg_prob": round(avg_prob, 3),
            "max_prob": round(max_prob, 3),
            "dominant_pct": round(dominant_pct, 3),
        }

    return results


def main():
    cache_path = Path(__file__).parent.parent / "data" / "backtest_data_cache.csv"
    if not cache_path.exists():
        print("ERROR: Run multi_regime_backtest.py first to create cache")
        sys.exit(1)

    data = pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Data: {len(data)} days, {len(data.columns)} streams")
    print()

    # Define perturbation scenarios
    scenarios = [
        # Adjustment scaling
        {"name": "Adj ×0.5 (halved)", "adj_scale": 0.5},
        {"name": "Adj ×0.7", "adj_scale": 0.7},
        {"name": "Adj ×1.0 (BASELINE)", "adj_scale": 1.0},
        {"name": "Adj ×1.5", "adj_scale": 1.5},
        {"name": "Adj ×2.0 (doubled)", "adj_scale": 2.0},
        # F matrix perturbation
        {"name": "F persistence −0.03", "f_delta": -0.03},
        {"name": "F persistence +0.03", "f_delta": 0.03},
        # R observation noise
        {"name": "R noise ×0.5 (halved)", "r_delta": -0.5},
        {"name": "R noise ×1.5", "r_delta": 0.5},
        {"name": "R noise ×2.0 (doubled)", "r_delta": 1.0},
        # TPM diagonal
        {"name": "TPM −0.02 (less sticky)", "tpm_delta": -0.02},
        {"name": "TPM +0.02 (more sticky)", "tpm_delta": 0.02},
    ]

    results_all = {}
    cp_names = [desc for _, _, _, _, desc in REGIME_CHECKPOINTS]

    print(f"  {'Scenario':35s} {'Pass':>5s}  ", end="")
    for name in cp_names:
        print(f" {name[:8]:>8s}", end="")
    print()
    print(f"  {'-'*35} {'-----':>5s}  ", end="")
    for _ in cp_names:
        print(f" {'--------':>8s}", end="")
    print()

    for scenario in scenarios:
        adj_scale = scenario.get("adj_scale", 1.0)
        f_delta = scenario.get("f_delta", 0.0)
        r_delta = scenario.get("r_delta", 0.0)
        tpm_delta = scenario.get("tpm_delta", 0.0)

        adjustments = perturb_adjustments(adj_scale)
        tpm = perturb_tpm(tpm_delta) if tpm_delta != 0 else DEFAULT_TPM.copy()

        result = run_trial(
            data,
            adjustments=adjustments,
            tpm=tpm,
            f_perturbation=f_delta,
            r_perturbation=r_delta,
        )

        passed = result["checkpoints_passed"]
        print(f"  {scenario['name']:35s} {passed:>3d}/4  ", end="")

        for desc in cp_names:
            cp = result["checkpoint_details"].get(desc, {})
            avg = cp.get("avg_prob", 0)
            ok = "✓" if cp.get("passed", False) else "✗"
            print(f" {ok}{avg:5.1%}  ", end="")
        print()

        results_all[scenario["name"]] = result

    # Summary
    print(f"\n{'='*70}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'='*70}")

    robust_count = sum(
        1 for r in results_all.values()
        if r["checkpoints_passed"] >= 3
    )
    total = len(results_all)
    perfect_count = sum(
        1 for r in results_all.values()
        if r["checkpoints_passed"] == 4
    )

    print(f"\n  Scenarios tested: {total}")
    print(f"  4/4 passed: {perfect_count}/{total} ({perfect_count/total:.0%})")
    print(f"  ≥3/4 passed: {robust_count}/{total} ({robust_count/total:.0%})")

    if robust_count >= total * 0.75:
        print(f"  → ROBUST: System tolerates parameter perturbation well")
    elif robust_count >= total * 0.50:
        print(f"  → MODERATE: Some sensitivity to parameter choices")
    else:
        print(f"  → FRAGILE: Results depend heavily on parameter tuning")

    # Save
    output_path = Path(__file__).parent.parent / "data" / "results" / "sensitivity_results.json"
    with open(output_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
