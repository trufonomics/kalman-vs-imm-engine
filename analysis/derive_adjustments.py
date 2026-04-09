"""
Derive Branch Adjustments from Historical Data — Fix 4

Computes mean factor z-scores during NBER recession, inflation surge,
and soft landing periods. These empirical values replace hand-tuned
adjustments, eliminating the "tuned for 40pp swings" criticism.

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/derive_adjustments.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

from heimdall.kalman_filter import FACTORS

# ── NBER regime windows ──────────────────────────────────────────────
# Using broadly accepted dates for each regime type.
# Recession dates from NBER Business Cycle Dating Committee.
# Inflation surge dates from CPI > 5% YoY period.
# Soft landing = everything else post-GFC that's not recession/inflation.

REGIME_WINDOWS = {
    "recession": [
        ("2008-01-01", "2009-06-30"),   # Great Recession (NBER: Dec 2007 - Jun 2009)
        ("2020-02-01", "2020-04-30"),   # COVID recession (NBER: Feb 2020 - Apr 2020)
    ],
    "stagflation": [
        ("2021-03-01", "2022-09-30"),   # CPI surged above 5% from May 2021
    ],
    "soft_landing": [
        ("2009-07-01", "2019-12-31"),   # Post-GFC expansion
        ("2023-01-01", "2025-12-31"),   # Disinflation period
    ],
}


def main():
    results_path = Path(__file__).parent.parent / "data" / "results" / "multi_regime_results.json"
    if not results_path.exists():
        print("ERROR: Run multi_regime_backtest.py first")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    log = results["daily_log"]
    dates = pd.to_datetime([e["date"] for e in log])
    factor_data = {f: [e["factors"][f] for e in log] for f in FACTORS}
    factor_df = pd.DataFrame(factor_data, index=dates)

    print("=" * 70)
    print("DATA-DERIVED BRANCH ADJUSTMENTS")
    print("=" * 70)
    print(f"Factor trajectories: {len(factor_df)} weekly snapshots")
    print(f"Date range: {factor_df.index[0].date()} → {factor_df.index[-1].date()}")
    print()

    # ── Compute mean factor values per regime ────────────────────────
    derived = {}
    for regime, windows in REGIME_WINDOWS.items():
        mask = pd.Series(False, index=factor_df.index)
        for start, end in windows:
            mask |= (factor_df.index >= start) & (factor_df.index <= end)

        regime_data = factor_df[mask]
        n_obs = len(regime_data)

        if n_obs < 5:
            print(f"  {regime}: only {n_obs} observations, skipping")
            continue

        means = regime_data.mean()
        stds = regime_data.std()

        print(f"  {regime.upper()} ({n_obs} weekly snapshots)")
        print(f"  {'Factor':25s} {'Mean':>8s} {'Std':>8s} {'Current Adj':>12s} {'Match?':>8s}")
        print(f"  {'-'*65}")

        adjustments = {}
        for factor in FACTORS:
            mean_val = float(means[factor])
            std_val = float(stds[factor])

            # Only include factors with meaningful signal (|mean| > 0.5 * std)
            if abs(mean_val) > 0.05:
                adjustments[factor] = round(mean_val, 3)

            # Compare to current hand-tuned values
            from heimdall.imm_tracker import RECOMMENDED_BRANCH_ADJUSTMENTS
            current = RECOMMENDED_BRANCH_ADJUSTMENTS.get(regime, {}).get(factor, 0)
            if abs(mean_val) > 0.05 or abs(current) > 0.05:
                diff = abs(mean_val - current)
                match = "✓" if diff < 0.25 else "≠"
                print(f"  {factor:25s} {mean_val:+8.3f} {std_val:8.3f} {current:+12.3f} {match:>8s}")

        derived[regime] = adjustments
        print()

    # ── Print derived adjustments in code-ready format ────────────────
    print("=" * 70)
    print("DERIVED ADJUSTMENTS (paste into imm_tracker.py)")
    print("=" * 70)
    print()
    print("EMPIRICAL_BRANCH_ADJUSTMENTS = {")
    for regime, adjs in derived.items():
        print(f'    "{regime}": {{')
        for factor, val in sorted(adjs.items()):
            print(f'        "{factor}": {val},')
        print("    },")
    print("}")

    # ── Compare to current ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("COMPARISON: CURRENT vs DATA-DERIVED")
    print("=" * 70)
    print()

    from heimdall.imm_tracker import RECOMMENDED_BRANCH_ADJUSTMENTS

    total_factors = 0
    close_count = 0
    for regime in derived:
        current = RECOMMENDED_BRANCH_ADJUSTMENTS.get(regime, {})
        all_factors = set(list(derived[regime].keys()) + list(current.keys()))
        for factor in sorted(all_factors):
            d_val = derived[regime].get(factor, 0)
            c_val = current.get(factor, 0)
            if abs(d_val) > 0.05 or abs(c_val) > 0.05:
                total_factors += 1
                diff = abs(d_val - c_val)
                if diff < 0.25:
                    close_count += 1
                    status = "CLOSE"
                elif (d_val > 0) == (c_val > 0) and abs(c_val) > 0.01:
                    status = "SAME DIR"
                else:
                    status = "DIFFERS"
                print(f"  {regime:12s} {factor:25s} current={c_val:+.3f}  data={d_val:+.3f}  {status}")

    print(f"\n  Match rate: {close_count}/{total_factors} within ±0.25")
    print(f"  ({close_count/max(total_factors,1):.0%} of adjustments are close to data-derived values)")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "derived_adjustments": derived,
        "current_adjustments": {
            k: dict(v) for k, v in RECOMMENDED_BRANCH_ADJUSTMENTS.items()
        },
        "regime_windows": REGIME_WINDOWS,
        "match_rate": round(close_count / max(total_factors, 1), 4),
    }
    output_path = Path(__file__).parent.parent / "data" / "results" / "derived_adjustments.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
