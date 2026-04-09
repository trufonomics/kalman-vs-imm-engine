"""
Derive Level B Sub-Regime Adjustments from Historical Data — Mar 19 2026

Extends derive_adjustments.py to compute sub-regime factor adjustments
for the VS-IMM hierarchical architecture.

Methodology:
  1. Label historical sub-periods using NBER dates + economic event dates
  2. Compute mean factor z-scores during each sub-period
  3. Subtract the Level A parent adjustment to get Level B deltas
  4. Output code-ready constants for imm_tracker.py

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/derive_sub_regime_adjustments.py
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
from heimdall.imm_tracker import LEVEL_A_ADJUSTMENTS

# ── Sub-Regime Period Labels ─────────────────────────────────────────
# Each sub-period is labeled based on economic history.
# Sources: NBER, FRED, economic consensus.
#
# NOTE: These windows don't cover 100% of history — only periods where
# the sub-regime is clearly identifiable. Ambiguous transition periods
# are excluded to keep the signal clean.

SUB_REGIME_WINDOWS = {
    # ── EXPANSION sub-regimes ──
    "expansion_goldilocks": [
        # Mid-cycle steady growth: moderate GDP, low inflation, stable labor
        ("2013-01-01", "2015-12-31"),   # Post-taper, pre-rate-hike calm
        ("2017-01-01", "2018-09-30"),   # Steady growth, low vol
        ("2024-06-01", "2025-06-30"),   # Soft landing achieved, disinflation done
    ],
    "expansion_boom": [
        # Overheating: strong growth, tight labor, rising assets, inflation turning up
        ("1997-01-01", "2000-03-31"),   # Dot-com boom (only 2005+ in our data)
        ("2005-01-01", "2006-12-31"),   # Housing boom, pre-GFC
        ("2021-03-01", "2021-09-30"),   # Post-COVID reopening surge (before inflation took hold)
    ],
    "expansion_disinflation": [
        # Active Fed easing cycle: inflation falling, growth holding
        ("2019-07-01", "2019-12-31"),   # Fed mid-cycle cuts
        ("2023-01-01", "2024-05-31"),   # Post-inflation tightening working, inflation dropping
    ],

    # ── STAGFLATION sub-regimes ──
    "stagflation_cost_push": [
        # Supply-driven: commodity surge, growth weakening
        ("2008-01-01", "2008-06-30"),   # Oil to $147, pre-Lehman (stagflationary phase)
        ("2022-01-01", "2022-06-30"),   # Ukraine war → oil/food/energy spike
    ],
    "stagflation_demand_pull": [
        # Demand-driven: strong growth + inflation from fiscal/monetary excess
        ("2021-10-01", "2022-01-31"),   # GDP still strong, CPI surging from demand
    ],

    # ── CONTRACTION sub-regimes ──
    "contraction_credit_crunch": [
        # Financial system stress → real economy drag
        ("2008-07-01", "2009-06-30"),   # GFC: Lehman, bank failures, credit freeze
    ],
    "contraction_demand_shock": [
        # Sudden external hit to demand
        ("2020-02-01", "2020-04-30"),   # COVID lockdowns: instant demand collapse
    ],
}


def main():
    results_path = Path(__file__).parent.parent / "data" / "results" / "multi_regime_results.json"
    if not results_path.exists():
        print("ERROR: Run multi_regime_backtest.py first to generate factor trajectories")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    log = results["daily_log"]
    dates = pd.to_datetime([e["date"] for e in log])
    factor_data = {f: [e["factors"][f] for e in log] for f in FACTORS}
    factor_df = pd.DataFrame(factor_data, index=dates)

    print("=" * 70)
    print("LEVEL B SUB-REGIME ADJUSTMENT DERIVATION")
    print("=" * 70)
    print(f"Factor trajectories: {len(factor_df)} weekly snapshots")
    print(f"Date range: {factor_df.index[0].date()} → {factor_df.index[-1].date()}")
    print()

    # ── Compute mean factor values per sub-regime ─────────────────────
    raw_means = {}
    for sub_regime, windows in SUB_REGIME_WINDOWS.items():
        mask = pd.Series(False, index=factor_df.index)
        total_in_range = 0

        for start, end in windows:
            window_mask = (factor_df.index >= start) & (factor_df.index <= end)
            mask |= window_mask
            total_in_range += window_mask.sum()

        regime_data = factor_df[mask]
        n_obs = len(regime_data)

        if n_obs < 3:
            print(f"  WARNING: {sub_regime} has only {n_obs} observations (need ≥3)")
            # Still include — some sub-regimes are short (COVID = 13 weeks)
            if n_obs == 0:
                print(f"  SKIP: {sub_regime} — no data in range")
                continue

        means = regime_data.mean()
        stds = regime_data.std()

        raw_means[sub_regime] = {
            factor: round(float(means[factor]), 3) for factor in FACTORS
        }

        print(f"  {sub_regime.upper()} ({n_obs} snapshots)")
        print(f"  {'Factor':25s} {'Mean z-score':>12s} {'Std':>8s}")
        print(f"  {'-'*50}")
        for factor in FACTORS:
            mean_val = float(means[factor])
            std_val = float(stds[factor]) if n_obs > 1 else 0
            if abs(mean_val) > 0.03:
                print(f"  {factor:25s} {mean_val:+12.3f} {std_val:8.3f}")
        print()

    # ── Compute Level B deltas (subtract parent Level A adjustment) ───
    print("=" * 70)
    print("LEVEL B DELTAS (raw mean - parent Level A adjustment)")
    print("=" * 70)
    print()

    # Map sub-regime → parent regime
    parent_map = {
        "expansion_goldilocks": "expansion",
        "expansion_boom": "expansion",
        "expansion_disinflation": "expansion",
        "stagflation_cost_push": "stagflation",
        "stagflation_demand_pull": "stagflation",
        "contraction_credit_crunch": "contraction",
        "contraction_demand_shock": "contraction",
    }

    level_b_deltas = {}
    for sub_regime, means in raw_means.items():
        parent = parent_map[sub_regime]
        parent_adj = LEVEL_A_ADJUSTMENTS.get(parent, {})

        # Delta = sub-regime mean - parent adjustment
        deltas = {}
        for factor in FACTORS:
            raw_mean = means.get(factor, 0.0)
            parent_val = parent_adj.get(factor, 0.0)
            delta = round(raw_mean - parent_val, 3)
            if abs(delta) > 0.03:  # Only include meaningful deltas
                deltas[factor] = delta

        level_b_deltas[sub_regime] = deltas

        # Strip prefix for display
        short_name = sub_regime.split("_", 1)[1] if "_" in sub_regime else sub_regime
        print(f"  {sub_regime}")
        print(f"  Parent: {parent} | Delta = raw_mean - parent_adj")
        print(f"  {'Factor':25s} {'Raw Mean':>10s} {'Parent':>10s} {'Delta':>10s}")
        print(f"  {'-'*60}")
        for factor in FACTORS:
            raw_mean = means.get(factor, 0.0)
            parent_val = parent_adj.get(factor, 0.0)
            delta = round(raw_mean - parent_val, 3)
            if abs(raw_mean) > 0.03 or abs(parent_val) > 0.03:
                marker = " ←" if abs(delta) > 0.15 else ""
                print(f"  {factor:25s} {raw_mean:+10.3f} {parent_val:+10.3f} {delta:+10.3f}{marker}")
        print()

    # ── Print code-ready constants ────────────────────────────────────
    print("=" * 70)
    print("CODE-READY CONSTANTS (paste into imm_tracker.py)")
    print("=" * 70)
    print()

    # Group by parent
    for parent in ["expansion", "stagflation", "contraction"]:
        var_name = f"{parent.upper()}_SUB_ADJUSTMENTS"
        print(f"{var_name} = {{")
        for sub_regime, deltas in level_b_deltas.items():
            if not sub_regime.startswith(parent):
                continue
            # Strip parent prefix
            short_name = sub_regime[len(parent) + 1:]
            print(f'    "{short_name}": {{')
            for factor, val in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f'        "{factor}": {val},')
            print("    },")
        print("}")
        print()

    # ── Discriminator analysis ────────────────────────────────────────
    print("=" * 70)
    print("DISCRIMINATOR ANALYSIS (which factors best separate sub-regimes?)")
    print("=" * 70)
    print()

    for parent in ["expansion", "stagflation", "contraction"]:
        subs = {k: v for k, v in raw_means.items() if k.startswith(parent)}
        if len(subs) < 2:
            continue

        print(f"  {parent.upper()} sub-regime discrimination:")
        sub_names = list(subs.keys())

        for factor in FACTORS:
            values = [subs[s].get(factor, 0.0) for s in sub_names]
            spread = max(values) - min(values)
            if spread > 0.10:
                labels = "  vs  ".join(
                    f"{s.split('_', 1)[1]}={subs[s].get(factor, 0):+.3f}"
                    for s in sub_names
                )
                quality = "STRONG" if spread > 0.30 else "MODERATE"
                print(f"    {factor:25s} spread={spread:.3f} [{quality}]  {labels}")
        print()

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "raw_means": raw_means,
        "level_b_deltas": level_b_deltas,
        "parent_adjustments": {k: dict(v) for k, v in LEVEL_A_ADJUSTMENTS.items()},
        "sub_regime_windows": {k: v for k, v in SUB_REGIME_WINDOWS.items()},
        "factors": FACTORS,
    }
    output_path = Path(__file__).parent.parent / "data" / "results" / "sub_regime_adjustments.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
