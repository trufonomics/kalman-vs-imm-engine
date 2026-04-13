"""
Bootstrap Confidence Intervals for the Shipping Stack.

Adds 95% CIs to all key metrics using Politis & Romano (1994) stationary
block bootstrap. Point estimates without CIs are unpublishable — this
script closes that gap.

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/bootstrap_ci_comparison.py
"""

import sys
import json
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall.bootstrap_ci import run_bootstrap_suite
from heimdall.regime_diagnostics import run_full_diagnostics

from backtests.diagnostic_comparison import run_engine, EngineOutput
from backtests.full_history_backtest import (
    fetch_data,
    REGIME_CHECKPOINTS,
)


def main():
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 90)
    print("  BOOTSTRAP CONFIDENCE INTERVALS — SHIPPING STACK")
    print("  Politis & Romano (1994) stationary block bootstrap")
    print("  500 resamples, 95% percentile CIs")
    print("  Block length: Politis-White (2004) + Patton (2009) auto-selection")
    print("=" * 90)

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        print(f"\nLoading cached data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    # Run full shipping stack
    print("\n--- Running Full Stack (P0+P2b+P4+P5) ---")
    full = run_engine(
        data, "Full Stack",
        use_regime_noise=True, use_state_tpm=True,
        use_cumulator=True, use_gas=True,
    )
    print(f"  {full.total_updates} updates, LL = {full.total_log_likelihood:.0f}")

    # Point estimates
    diag = run_full_diagnostics(full.daily_log, REGIME_CHECKPOINTS)
    print("\n--- Point Estimates ---")
    for regime in ["contraction", "stagflation"]:
        r = diag["roc"][regime]
        b = diag["brier"][regime]
        print(f"  {regime}: AUC={r.auc:.4f}, Brier={b.brier_score:.4f}, "
              f"Reliab={b.reliability:.4f}, Resol={b.resolution:.4f}")

    detected = sum(1 for dl in diag["detection_lag"] if dl.detection_lag_weeks is not None)
    lags = [dl.detection_lag_weeks for dl in diag["detection_lag"] if dl.detection_lag_weeks is not None]
    mean_lag = np.mean(lags) if lags else 0
    print(f"  Detection: {detected}/{len(diag['detection_lag'])}, mean lag = {mean_lag:.1f}w")

    # Bootstrap CIs
    print("\n--- Bootstrap CIs (500 resamples, this takes a few minutes) ---")
    boot_results = run_bootstrap_suite(
        full.daily_log,
        REGIME_CHECKPOINTS,
        n_bootstrap=500,
        seed=42,
    )

    print("\n" + "=" * 90)
    print("  RESULTS — 95% Confidence Intervals")
    print("=" * 90)
    print(f"\n  {'Metric':<30s} {'Point':>8s} {'Lower':>8s} {'Upper':>8s} {'SE':>8s}")
    print("  " + "-" * 66)

    for key, ci in sorted(boot_results.items()):
        print(f"  {key:<30s} {ci.point_estimate:>8.4f} "
              f"{ci.ci_lower:>8.4f} {ci.ci_upper:>8.4f} {ci.std_error:>8.4f}")

    # Save results
    results_dir = Path(__file__).parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for key, ci in boot_results.items():
        save_data[key] = {
            "point_estimate": round(ci.point_estimate, 4),
            "ci_lower": round(ci.ci_lower, 4),
            "ci_upper": round(ci.ci_upper, 4),
            "std_error": round(ci.std_error, 4),
            "n_bootstrap": ci.n_bootstrap,
        }

    out_path = results_dir / "bootstrap_ci_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
