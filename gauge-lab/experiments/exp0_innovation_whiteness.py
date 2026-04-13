"""Experiment 0: Innovation Whiteness Test.

If the Kalman engine's flat-space assumption is correct, innovations should
be white noise (uncorrelated, zero mean). If innovations show systematic
autocorrelation or cross-correlation, the flat connection is wrong —
curvature is present.

This experiment requires NO new math. It just analyzes the existing engine's
output. If innovations are non-white, that's the smoking gun for curvature.

Usage:
    python experiments/exp0_innovation_whiteness.py --data backtests/full_history_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from heimdall.kalman_filter import FACTORS, EconomicStateEstimator


def load_innovation_history(data_path: str) -> dict:
    """Load innovation history from backtest results."""
    with open(data_path) as f:
        return json.load(f)


def extract_innovations_per_factor(history: list[dict]) -> dict[str, np.ndarray]:
    """Extract per-factor innovation z-scores from update history.

    Each update has kalman_gain (8-element) and innovation_zscore (scalar).
    The per-factor contribution is: kalman_gain[i] * innovation_zscore.
    """
    factor_innovations: dict[str, list[float]] = {f: [] for f in FACTORS}

    for update in history:
        zscore = update.get("innovation_zscore", 0.0)
        kg = update.get("kalman_gain", [0.0] * len(FACTORS))

        for i, factor in enumerate(FACTORS):
            if i < len(kg):
                factor_innovations[factor].append(kg[i] * zscore)

    return {f: np.array(v) for f, v in factor_innovations.items()}


def ljung_box_test(innovations: np.ndarray, max_lag: int = 20) -> dict:
    """Ljung-Box test for autocorrelation (whiteness).

    H0: innovations are white noise (no autocorrelation up to lag k)
    H1: at least one autocorrelation is non-zero

    Returns dict with test statistic, p-value, and per-lag autocorrelations.
    """
    n = len(innovations)
    if n < max_lag + 10:
        max_lag = max(1, n // 3)

    # Compute autocorrelations
    mean = np.mean(innovations)
    centered = innovations - mean
    var = np.sum(centered ** 2) / n

    acf_values = []
    for lag in range(1, max_lag + 1):
        if var > 0:
            acf = np.sum(centered[lag:] * centered[:-lag]) / (n * var)
        else:
            acf = 0.0
        acf_values.append(acf)

    # Ljung-Box statistic: Q = n(n+2) Σ_{k=1}^{m} r_k^2 / (n-k)
    Q = 0.0
    for k, r_k in enumerate(acf_values, 1):
        Q += (r_k ** 2) / (n - k)
    Q *= n * (n + 2)

    # Under H0, Q ~ chi-squared(max_lag)
    p_value = 1.0 - stats.chi2.cdf(Q, max_lag)

    return {
        "statistic": float(Q),
        "p_value": float(p_value),
        "max_lag": max_lag,
        "autocorrelations": [float(a) for a in acf_values],
        "is_white": p_value > 0.05,  # fail to reject H0 at 5%
    }


def cross_correlation_matrix(factor_innovations: dict[str, np.ndarray]) -> np.ndarray:
    """Compute cross-correlation between factors' innovations.

    If innovations are truly white AND independent, the cross-correlation
    matrix should be approximately identity. Off-diagonal structure = the
    factors are systematically co-surprised, suggesting the flat-space model
    is missing something.
    """
    factors = list(factor_innovations.keys())
    n = len(factors)
    C = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            xi = factor_innovations[factors[i]]
            xj = factor_innovations[factors[j]]
            min_len = min(len(xi), len(xj))
            if min_len > 0:
                C[i, j] = float(np.corrcoef(xi[:min_len], xj[:min_len])[0, 1])

    return C


def regime_segmented_analysis(
    innovations: np.ndarray,
    regime_labels: list[str],
) -> dict:
    """Test whiteness separately in each regime.

    If innovations are white in one regime but not another, the curvature
    is regime-dependent (the manifold has different geometry in different regions).
    """
    unique_regimes = sorted(set(regime_labels))
    results = {}

    for regime in unique_regimes:
        mask = np.array([r == regime for r in regime_labels])
        regime_innov = innovations[mask[:len(innovations)]]
        if len(regime_innov) > 20:
            results[regime] = ljung_box_test(regime_innov)
            results[regime]["n_observations"] = int(len(regime_innov))

    return results


def run_experiment(data_path: str) -> dict:
    """Run the full innovation whiteness experiment."""
    print("=" * 60)
    print("EXPERIMENT 0: Innovation Whiteness Test")
    print("Is the economic state space flat or curved?")
    print("=" * 60)

    # Load data
    data = load_innovation_history(data_path)
    history = data.get("innovation_history", data.get("updates", []))

    if not history:
        print("ERROR: No innovation history found in data file.")
        print("Expected keys: 'innovation_history' or 'updates'")
        return {"error": "no_data"}

    print(f"\nLoaded {len(history)} observations")

    # Extract per-factor innovations
    factor_innov = extract_innovations_per_factor(history)

    # Test 1: Per-factor Ljung-Box whiteness
    print("\n--- Test 1: Per-Factor Autocorrelation (Ljung-Box) ---")
    lb_results = {}
    for factor in FACTORS:
        innov = factor_innov[factor]
        if len(innov) > 30:
            result = ljung_box_test(innov)
            lb_results[factor] = result
            status = "WHITE" if result["is_white"] else "NON-WHITE ***"
            print(f"  {factor:25s}: Q={result['statistic']:8.2f}  p={result['p_value']:.4f}  [{status}]")
            if not result["is_white"]:
                # Show strongest autocorrelation lag
                acf = np.abs(result["autocorrelations"])
                peak_lag = int(np.argmax(acf)) + 1
                peak_val = float(acf[peak_lag - 1])
                print(f"    Peak autocorrelation: lag {peak_lag}, |r| = {peak_val:.3f}")

    # Test 2: Cross-correlation matrix
    print("\n--- Test 2: Cross-Factor Innovation Correlation ---")
    C = cross_correlation_matrix(factor_innov)
    off_diag = C[np.triu_indices_from(C, k=1)]
    print(f"  Mean |off-diagonal|: {np.mean(np.abs(off_diag)):.4f}")
    print(f"  Max  |off-diagonal|: {np.max(np.abs(off_diag)):.4f}")

    # Find strongest cross-correlations
    n = len(FACTORS)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((FACTORS[i], FACTORS[j], C[i, j]))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)

    print("  Top 5 cross-correlations:")
    for f1, f2, corr in pairs[:5]:
        print(f"    {f1:25s} × {f2:25s}: r = {corr:+.4f}")

    # Test 3: Innovation kurtosis (Gaussianity)
    print("\n--- Test 3: Innovation Distribution (Gaussianity) ---")
    for factor in FACTORS:
        innov = factor_innov[factor]
        if len(innov) > 30:
            kurt = float(stats.kurtosis(innov, fisher=True))  # excess kurtosis
            skew = float(stats.skew(innov))
            _, jb_p = stats.jarque_bera(innov)
            status = "GAUSSIAN" if jb_p > 0.05 else "NON-GAUSSIAN ***"
            print(f"  {factor:25s}: kurt={kurt:+6.2f}  skew={skew:+6.2f}  JB p={jb_p:.4f}  [{status}]")

    # Summary
    n_nonwhite = sum(1 for r in lb_results.values() if not r["is_white"])
    n_tested = len(lb_results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Factors tested: {n_tested}")
    print(f"  Non-white innovations: {n_nonwhite}/{n_tested}")
    print(f"  Mean cross-correlation: {np.mean(np.abs(off_diag)):.4f}")

    if n_nonwhite > 0:
        print(f"\n  CONCLUSION: {n_nonwhite} factors show systematic autocorrelation.")
        print("  The flat-space assumption is producing systematic prediction errors.")
        print("  → CURVATURE IS PRESENT in the economic state space.")
        print("  → Proceed to Experiment 2 (connection estimation).")
    else:
        print("\n  CONCLUSION: All innovations appear white.")
        print("  The flat-space assumption may be adequate (curvature < noise).")
        print("  → Check with larger sample or regime-segmented analysis.")

    return {
        "ljung_box": lb_results,
        "cross_correlation": C.tolist(),
        "top_cross_pairs": [(f1, f2, float(c)) for f1, f2, c in pairs[:10]],
        "n_nonwhite": n_nonwhite,
        "n_tested": n_tested,
    }


def run_on_synthetic(n_steps: int = 5000) -> dict:
    """Run on synthetic data to validate the test works.

    Generates data from a KNOWN curved model (state-dependent F),
    runs standard (flat) Kalman, checks that innovations are non-white.
    Then runs on data from a flat model, checks innovations ARE white.

    Key: we feed MULTIPLE streams per time step so the filter is well-observed,
    and use the filter's OWN Q/F/H/R to generate data (perfect DGP match for flat).
    """
    from heimdall.kalman_filter import EconomicStateEstimator, FACTORS, N_FACTORS

    print("=" * 60)
    print("SYNTHETIC VALIDATION: Does the test detect known curvature?")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Streams to cycle through (covers multiple factors)
    multi_streams = [
        "US_CPI_YOY", "NONFARM_PAYROLLS", "SP500", "OIL_PRICE",
        "HOUSING_STARTS", "FED_FUNDS_RATE", "RETAIL_SALES", "BTC_USD",
    ]

    # --- Phase A: Flat data (constant F, perfect DGP match) ---
    print("\n--- Phase A: Flat data (constant F, multi-stream) ---")
    estimator = EconomicStateEstimator()
    x_true = np.zeros(N_FACTORS)
    flat_innovations = []

    # Use filter's own process noise for DGP
    Q_chol = np.linalg.cholesky(estimator.Q + 1e-10 * np.eye(N_FACTORS))

    for t in range(n_steps):
        x_true = estimator.F @ x_true + Q_chol @ rng.normal(0, 1, N_FACTORS)
        estimator.predict()

        # Feed one stream per step, cycling
        stream_key = multi_streams[t % len(multi_streams)]
        H_row, R = estimator.stream_registry[stream_key]
        z = float(H_row @ x_true + rng.normal(0, np.sqrt(R)))
        update = estimator.update(stream_key, z)
        if update:
            flat_innovations.append(update.innovation_zscore)

    flat_innov = np.array(flat_innovations)
    flat_result = ljung_box_test(flat_innov)
    print(f"  Ljung-Box: Q={flat_result['statistic']:.2f}, p={flat_result['p_value']:.4f}")
    print(f"  White: {flat_result['is_white']}")

    # --- Phase B: Curved data (state-dependent F) ---
    print("\n--- Phase B: Curved data (state-dependent F, multi-stream) ---")
    estimator2 = EconomicStateEstimator()
    x_true2 = np.zeros(N_FACTORS)
    curved_innovations = []

    for t in range(n_steps):
        F_curved = estimator2.F.copy()
        # STRONG curvature: inflation persistence is state-dependent
        # When inflation is high → more persistent (self-reinforcing spiral)
        F_curved[0, 0] = 0.97 + 0.025 * np.tanh(3.0 * x_true2[0])
        # When growth is negative → commodity→inflation channel strengthens
        F_curved[0, 5] = 0.06 * (1.0 + np.tanh(-2.0 * x_true2[1]))
        # When financial conditions tighten → growth decays faster
        F_curved[1, 1] = 0.96 - 0.03 * np.tanh(2.0 * x_true2[4])

        x_true2 = F_curved @ x_true2 + Q_chol @ rng.normal(0, 1, N_FACTORS)
        estimator2.predict()

        stream_key = multi_streams[t % len(multi_streams)]
        H_row, R = estimator2.stream_registry[stream_key]
        z = float(H_row @ x_true2 + rng.normal(0, np.sqrt(R)))
        update = estimator2.update(stream_key, z)
        if update:
            curved_innovations.append(update.innovation_zscore)

    curved_innov = np.array(curved_innovations)
    curved_result = ljung_box_test(curved_innov)
    print(f"  Ljung-Box: Q={curved_result['statistic']:.2f}, p={curved_result['p_value']:.4f}")
    print(f"  White: {curved_result['is_white']}")

    print(f"\n--- Result ---")
    if flat_result["is_white"] and not curved_result["is_white"]:
        print("  TEST VALIDATED: Detects flat (white) and curved (non-white) correctly.")
    elif not flat_result["is_white"]:
        print("  WARNING: Even flat data shows non-white innovations.")
        print("  Possible cause: filter parameters not perfectly matched to DGP.")
    else:
        print("  WARNING: Curved data still shows white innovations.")
        print("  Possible cause: curvature too small relative to noise.")

    return {
        "flat": flat_result,
        "curved": curved_result,
        "validated": flat_result["is_white"] and not curved_result["is_white"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Innovation Whiteness Test")
    parser.add_argument("--data", type=str, help="Path to backtest results JSON")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic data")
    args = parser.parse_args()

    if args.synthetic:
        results = run_on_synthetic()
    elif args.data:
        results = run_experiment(args.data)
    else:
        print("Running synthetic validation (no --data provided)...")
        print("To run on real data: python exp0_innovation_whiteness.py --data <path>")
        print()
        results = run_on_synthetic()

    # Save results
    output_path = Path(__file__).resolve().parent.parent / "results" / "exp0_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(f"\nResults saved to {output_path}")
