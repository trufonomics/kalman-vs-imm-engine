"""
Estimate Factor Loadings via PCA — Fix 3

Runs PCA on the 15-stream backtest data, rotates via varimax,
and compares to hand-specified H loadings. Also estimates R
from residual variance.

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/estimate_loadings.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

from heimdall.kalman_filter import (
    FACTORS,
    FACTOR_INDEX,
    STREAM_LOADINGS,
    STREAM_NOISE,
    FREQUENCY_SCALE,
    N_FACTORS,
)

# Map backtest column names to Kalman stream keys
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


def varimax_rotation(loadings: np.ndarray, max_iter: int = 100,
                     tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Varimax rotation for interpretability."""
    n, k = loadings.shape
    rotation = np.eye(k)
    d = 0

    for _ in range(max_iter):
        old_d = d
        comp = loadings @ rotation
        u, s, vt = np.linalg.svd(
            loadings.T @ (comp ** 3 - (1.0 / n) * comp @ np.diag(np.sum(comp ** 2, axis=0)))
        )
        rotation = u @ vt
        d = np.sum(s)
        if abs(d - old_d) < tol:
            break

    return loadings @ rotation, rotation


def main():
    cache_path = Path(__file__).parent.parent / "data" / "backtest_data_cache.csv"
    if not cache_path.exists():
        print("ERROR: Run multi_regime_backtest.py first")
        sys.exit(1)

    data = pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print("=" * 70)
    print("PCA FACTOR LOADING ESTIMATION")
    print("=" * 70)
    print(f"Data: {len(data)} days, {len(data.columns)} streams")

    # ── Prepare data for PCA ─────────────────────────────────────────
    # Resample to weekly (reduces noise, matches factor update frequency)
    weekly = data.resample("W").mean()

    # Drop columns with >50% missing
    thresh = len(weekly) * 0.5
    weekly = weekly.dropna(axis=1, thresh=int(thresh))
    weekly = weekly.ffill().dropna()

    print(f"Weekly data: {len(weekly)} weeks, {len(weekly.columns)} streams")

    # Standardize
    means = weekly.mean()
    stds = weekly.std()
    standardized = (weekly - means) / stds

    # ── Run PCA ──────────────────────────────────────────────────────
    n_components = min(N_FACTORS, len(standardized.columns))
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(standardized)
    raw_loadings = pca.components_.T  # (n_streams × n_components)

    print(f"\nPCA: {n_components} components")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    for i, ev in enumerate(pca.explained_variance_ratio_):
        cumulative = sum(pca.explained_variance_ratio_[:i+1])
        print(f"  PC{i+1}: {ev:.1%} (cumulative: {cumulative:.1%})")

    # ── Varimax rotation ─────────────────────────────────────────────
    rotated_loadings, rotation_matrix = varimax_rotation(raw_loadings)

    print(f"\nRotated loadings (varimax):")
    print(f"  {'Stream':30s}", end="")
    for i in range(n_components):
        print(f"  {'PC'+str(i+1):>6s}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for _ in range(n_components):
        print(f"  {'------':>6s}", end="")
    print()

    for j, col in enumerate(standardized.columns):
        print(f"  {col:30s}", end="")
        for i in range(n_components):
            val = rotated_loadings[j, i]
            marker = "*" if abs(val) > 0.4 else " "
            print(f"  {val:+5.2f}{marker}", end="")
        print()

    # ── Match PCA components to Kalman factors ───────────────────────
    # For each Kalman factor, find the PCA component with highest
    # correlation to the corresponding hand-specified loading pattern
    print(f"\n{'='*70}")
    print("PCA vs HAND-SPECIFIED LOADINGS")
    print(f"{'='*70}")

    # Build hand-specified H matrix for the streams we have
    available_streams = []
    for col in standardized.columns:
        kalman_key = COLUMN_TO_KALMAN.get(col)
        if kalman_key and kalman_key in STREAM_LOADINGS:
            available_streams.append((col, kalman_key))

    n_available = len(available_streams)
    H_hand = np.zeros((n_available, N_FACTORS))
    H_pca = np.zeros((n_available, n_components))

    for s_idx, (col, kalman_key) in enumerate(available_streams):
        # Hand-specified
        for factor, loading in STREAM_LOADINGS[kalman_key].items():
            f_idx = FACTOR_INDEX.get(factor)
            if f_idx is not None:
                H_hand[s_idx, f_idx] = loading

        # PCA
        col_idx = list(standardized.columns).index(col)
        H_pca[s_idx, :] = rotated_loadings[col_idx, :]

    # Correlate each hand-specified factor column with each PCA component
    print(f"\nCorrelation matrix (hand-specified factors × PCA components):")
    print(f"  {'Factor':25s}", end="")
    for i in range(n_components):
        print(f"  {'PC'+str(i+1):>6s}", end="")
    print(f"  {'Best PC':>8s} {'Corr':>6s}")
    print(f"  {'-'*25}", end="")
    for _ in range(n_components):
        print(f"  {'------':>6s}", end="")
    print(f"  {'--------':>8s} {'------':>6s}")

    factor_pc_matches = {}
    overall_correlations = []

    for f_idx, factor in enumerate(FACTORS):
        h_col = H_hand[:, f_idx]
        if np.std(h_col) < 1e-10:
            print(f"  {factor:25s}  (no loadings)")
            continue

        print(f"  {factor:25s}", end="")
        best_corr = 0
        best_pc = -1
        for pc_idx in range(n_components):
            pc_col = H_pca[:, pc_idx]
            if np.std(pc_col) < 1e-10:
                corr = 0
            else:
                corr = float(np.corrcoef(h_col, pc_col)[0, 1])
            print(f"  {corr:+5.2f} ", end="")
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_pc = pc_idx
        print(f"    PC{best_pc+1:d}   {best_corr:+.3f}")
        factor_pc_matches[factor] = {
            "best_pc": best_pc,
            "correlation": round(best_corr, 4),
        }
        overall_correlations.append(abs(best_corr))

    avg_corr = np.mean(overall_correlations) if overall_correlations else 0
    print(f"\n  Average |correlation| between hand-specified and PCA loadings: {avg_corr:.3f}")

    if avg_corr > 0.5:
        print(f"  → GOOD: Theory-based loadings are broadly consistent with data-driven PCA")
    elif avg_corr > 0.3:
        print(f"  → MODERATE: Some alignment but significant differences exist")
    else:
        print(f"  → WEAK: Hand-specified loadings diverge from data structure")

    # ── Estimate R from PCA residuals ────────────────────────────────
    print(f"\n{'='*70}")
    print("ESTIMATED vs HAND-SPECIFIED OBSERVATION NOISE (R)")
    print(f"{'='*70}")

    # Reconstruct data from PCA and compute residual variance
    reconstructed = scores @ pca.components_
    residuals = standardized.values - reconstructed
    residual_var = np.var(residuals, axis=0)

    print(f"\n  {'Stream':30s} {'R (hand)':>10s} {'R (PCA)':>10s} {'Ratio':>8s}")
    print(f"  {'-'*60}")

    r_comparisons = {}
    for j, col in enumerate(standardized.columns):
        kalman_key = COLUMN_TO_KALMAN.get(col)
        if not kalman_key:
            continue

        base_r = STREAM_NOISE.get(kalman_key, 0.10)
        freq_scale = FREQUENCY_SCALE.get(kalman_key, 1.0)
        hand_r = base_r * freq_scale

        # PCA residual variance (already in standardized units)
        pca_r = float(residual_var[j])

        ratio = pca_r / hand_r if hand_r > 0 else float("inf")
        print(f"  {col:30s} {hand_r:10.4f} {pca_r:10.4f} {ratio:8.2f}x")

        r_comparisons[kalman_key] = {
            "hand_r": round(float(hand_r), 4),
            "pca_residual_var": round(float(pca_r), 4),
            "ratio": round(float(ratio), 2),
        }

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "n_components": n_components,
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "total_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "factor_pc_matches": factor_pc_matches,
        "avg_correlation": round(float(avg_corr), 4),
        "r_comparisons": r_comparisons,
        "rotated_loadings": rotated_loadings.tolist(),
        "streams": list(standardized.columns),
    }

    output_path = Path(__file__).parent.parent / "data" / "results" / "estimated_loadings.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
