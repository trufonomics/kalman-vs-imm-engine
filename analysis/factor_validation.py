"""
Factor Validation — Phase 4 & 5 of "Bulletproof Architecture"

Phase 4: Factor-vs-Reference Correlation
  For each Kalman factor, correlate the estimated trajectory with a known
  reference series from FRED. If inflation_trend doesn't correlate with
  actual CPI, the factor isn't tracking what it claims.

Phase 5: Cross-Dynamics Estimation
  Instead of assuming diagonal F (factors evolve independently), estimate
  a VAR(1) from the backtest factor trajectories. Discovers which factors
  lead/lag each other (e.g., does commodity_pressure predict inflation_trend?).
  Compare fitted F to the current diagonal F to see what we're missing.

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/factor_validation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall.kalman_filter import FACTORS, FACTOR_INDEX

# ── Phase 4: Reference series for each factor ─────────────────────────

# Map each Kalman factor to the FRED backtest column that should correlate.
# Some factors have multiple reasonable references.
FACTOR_REFERENCES = {
    "inflation_trend": ["FRED_US_CPI_YOY", "FRED_CORE_CPI", "FRED_PPI"],
    "growth_trend": ["YAHOO_SP500", "FRED_RETAIL_SALES"],
    "labor_pressure": ["FRED_UNEMPLOYMENT_RATE", "FRED_INITIAL_CLAIMS"],
    "housing_momentum": ["FRED_HOME_PRICES", "FRED_HOUSING_STARTS"],
    "financial_conditions": ["FRED_10Y_YIELD", "FRED_FED_FUNDS_RATE"],
    "commodity_pressure": ["YAHOO_OIL_PRICE", "YAHOO_GOLD_PRICE"],
    "consumer_sentiment": ["FRED_CONSUMER_CONFIDENCE", "FRED_RETAIL_SALES"],
    "policy_stance": ["FRED_FED_FUNDS_RATE", "FRED_10Y_YIELD"],
}

# Minimum acceptable correlation (absolute) for a factor to "pass"
MIN_CORRELATION = 0.15


def load_data():
    """Load backtest results and raw data."""
    results_path = Path(__file__).parent.parent / "data" / "results" / "multi_regime_results.json"
    cache_path = Path(__file__).parent.parent / "data" / "backtest_data_cache.csv"

    if not results_path.exists():
        print("ERROR: Run multi_regime_backtest.py first")
        sys.exit(1)

    results = json.load(open(results_path))
    raw_data = pd.read_csv(cache_path, index_col=0, parse_dates=True)

    # Build factor trajectory DataFrame from weekly log
    log = results["daily_log"]
    factor_dates = [e["date"] for e in log]
    factor_data = {f: [e["factors"][f] for e in log] for f in FACTORS}
    factor_df = pd.DataFrame(factor_data, index=pd.to_datetime(factor_dates))

    return results, raw_data, factor_df


def run_phase4(factor_df: pd.DataFrame, raw_data: pd.DataFrame) -> dict:
    """Phase 4: Factor-vs-reference correlation checks."""
    print(f"\n{'='*70}")
    print("PHASE 4: FACTOR-VS-REFERENCE CORRELATION")
    print(f"{'='*70}")
    print(f"Factor trajectories: {len(factor_df)} weekly snapshots")
    print(f"Raw data: {len(raw_data)} days, {len(raw_data.columns)} streams")
    print()

    results = {}
    pass_count = 0
    total_count = 0

    for factor in FACTORS:
        refs = FACTOR_REFERENCES.get(factor, [])
        if not refs:
            continue

        factor_series = factor_df[factor]
        best_corr = 0.0
        best_ref = ""
        ref_results = []

        for ref_col in refs:
            if ref_col not in raw_data.columns:
                continue

            # Resample raw data to weekly to match factor snapshots
            ref_weekly = raw_data[ref_col].resample("W").mean().dropna()

            # Align on common dates
            common = factor_series.index.intersection(ref_weekly.index)
            if len(common) < 20:
                # Try nearest-date alignment with 3-day tolerance
                aligned_factor = []
                aligned_ref = []
                for date in factor_series.index:
                    # Find nearest ref date within 3 days
                    diffs = abs(ref_weekly.index - date)
                    min_idx = diffs.argmin()
                    if diffs[min_idx] <= pd.Timedelta(days=3):
                        aligned_factor.append(factor_series[date])
                        aligned_ref.append(ref_weekly.iloc[min_idx])

                if len(aligned_factor) < 20:
                    ref_results.append({
                        "ref": ref_col, "n": len(aligned_factor),
                        "corr": None, "p_value": None, "note": "insufficient overlap"
                    })
                    continue

                f_arr = np.array(aligned_factor)
                r_arr = np.array(aligned_ref)
            else:
                f_arr = factor_series.loc[common].values
                r_arr = ref_weekly.loc[common].values

            # Remove any NaN pairs
            mask = ~(np.isnan(f_arr) | np.isnan(r_arr))
            f_arr = f_arr[mask]
            r_arr = r_arr[mask]

            if len(f_arr) < 20:
                ref_results.append({
                    "ref": ref_col, "n": len(f_arr),
                    "corr": None, "p_value": None, "note": "insufficient data"
                })
                continue

            corr, p_value = stats.pearsonr(f_arr, r_arr)

            ref_results.append({
                "ref": ref_col,
                "n": int(len(f_arr)),
                "corr": round(float(corr), 4),
                "p_value": round(float(p_value), 6),
            })

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_ref = ref_col

        passed = abs(best_corr) >= MIN_CORRELATION
        total_count += 1
        if passed:
            pass_count += 1

        status = "PASS" if passed else "FAIL"
        print(f"  {factor:25s} best_corr={best_corr:+.3f} ({best_ref:30s}) {status}")
        for r in ref_results:
            if r.get("corr") is not None:
                sig = "*" if r["p_value"] < 0.05 else " "
                print(f"    {r['ref']:35s} r={r['corr']:+.3f} p={r['p_value']:.4f}{sig} n={r['n']}")
            else:
                print(f"    {r['ref']:35s} {r.get('note', 'N/A')}")

        results[factor] = {
            "best_corr": round(float(best_corr), 4),
            "best_ref": best_ref,
            "passed": passed,
            "references": ref_results,
        }

    print(f"\n  Factor correlation pass: {pass_count}/{total_count} "
          f"({pass_count / max(total_count, 1):.0%})")
    print(f"  Threshold: |r| ≥ {MIN_CORRELATION}")

    results["_summary"] = {
        "pass_count": pass_count,
        "total": total_count,
        "pass_rate": round(pass_count / max(total_count, 1), 4),
    }
    return results


def run_phase5(factor_df: pd.DataFrame) -> dict:
    """Phase 5: Cross-dynamics estimation via VAR(1).

    Fits x_t = A @ x_{t-1} + noise to the factor trajectories.
    The diagonal of A ≈ persistence (should match our F diagonal).
    Off-diagonals reveal cross-factor lead/lag relationships.
    """
    print(f"\n{'='*70}")
    print("PHASE 5: CROSS-DYNAMICS ESTIMATION (VAR(1))")
    print(f"{'='*70}")

    # Remove warmup (first 12 weeks)
    data = factor_df.iloc[12:].copy()
    print(f"Using {len(data)} weekly snapshots (warmup removed)")

    # Build lagged matrices
    X = data.iloc[:-1].values  # x_{t-1}
    Y = data.iloc[1:].values   # x_t

    n_obs, n_factors = X.shape
    print(f"Fitting VAR(1): {n_obs} observations, {n_factors} factors\n")

    # OLS: A = (X'X)^{-1} X'Y
    try:
        A_hat = np.linalg.lstsq(X, Y, rcond=None)[0].T  # [n_factors × n_factors]
    except np.linalg.LinAlgError:
        print("  ERROR: Singular matrix in OLS fit")
        return {"error": "singular_matrix"}

    # Residuals
    Y_pred = (A_hat @ X.T).T
    residuals = Y - Y_pred
    residual_cov = np.cov(residuals.T)

    # ── Report diagonal (persistence) ──
    print("Estimated Persistence (diagonal of A) vs Current F:")
    print(f"  {'Factor':25s} {'Estimated':>10s} {'Current F':>10s} {'Delta':>10s}")
    print(f"  {'-'*55}")

    # Current F diagonal from production
    from heimdall.kalman_filter import EconomicStateEstimator
    est = EconomicStateEstimator()
    current_F_diag = np.diag(est.F)

    persistence_results = {}
    for i, factor in enumerate(FACTORS):
        estimated = A_hat[i, i]
        current = current_F_diag[i]
        delta = estimated - current
        persistence_results[factor] = {
            "estimated": round(float(estimated), 4),
            "current": round(float(current), 4),
            "delta": round(float(delta), 4),
        }
        flag = " !!" if abs(delta) > 0.1 else ""
        print(f"  {factor:25s} {estimated:10.4f} {current:10.4f} {delta:+10.4f}{flag}")

    # ── Report significant off-diagonals (cross-dynamics) ──
    print(f"\nSignificant Cross-Dynamics (|A_ij| > 0.05):")
    print(f"  {'From':25s} → {'To':25s} {'Coeff':>10s} {'Interpretation'}")
    print(f"  {'-'*80}")

    cross_dynamics = []
    for i in range(n_factors):
        for j in range(n_factors):
            if i == j:
                continue
            coeff = A_hat[i, j]
            if abs(coeff) > 0.05:
                from_factor = FACTORS[j]
                to_factor = FACTORS[i]

                # Interpret the direction
                if coeff > 0:
                    interp = f"↑ {from_factor} → ↑ {to_factor}"
                else:
                    interp = f"↑ {from_factor} → ↓ {to_factor}"

                cross_dynamics.append({
                    "from": from_factor,
                    "to": to_factor,
                    "coefficient": round(float(coeff), 4),
                    "interpretation": interp,
                })
                print(f"  {from_factor:25s} → {to_factor:25s} {coeff:+10.4f}   {interp}")

    if not cross_dynamics:
        print("  (none found — factors are roughly independent at weekly frequency)")

    # ── Residual correlation matrix (what F misses) ──
    print(f"\nResidual Correlation Matrix (top off-diagonal pairs):")
    res_corr = np.corrcoef(residuals.T)
    corr_pairs = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            corr_pairs.append((FACTORS[i], FACTORS[j], res_corr[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"  {'Factor A':25s} {'Factor B':25s} {'Residual r':>10s}")
    print(f"  {'-'*60}")
    for a, b, r in corr_pairs[:8]:
        flag = " !!" if abs(r) > 0.3 else ""
        print(f"  {a:25s} {b:25s} {r:+10.4f}{flag}")

    # ── Suggested F matrix improvements ──
    print(f"\nSuggested F Matrix Improvements:")
    improvements = []

    for entry in cross_dynamics:
        if abs(entry["coefficient"]) > 0.10:
            improvements.append(entry)
            print(f"  Add F[{entry['to']}, {entry['from']}] = {entry['coefficient']:+.3f}")

    for factor, p in persistence_results.items():
        if abs(p["delta"]) > 0.1:
            improvements.append({
                "type": "persistence",
                "factor": factor,
                "suggested": p["estimated"],
                "current": p["current"],
            })
            print(f"  Adjust F[{factor}, {factor}]: {p['current']:.3f} → {p['estimated']:.3f}")

    if not improvements:
        print("  Current F matrix is adequate — no changes recommended")

    # ── Granger-like causality test (simplified) ──
    print(f"\nGranger-like Lead/Lag Analysis (1-week lead):")
    print(f"  {'Predictor':25s} {'Target':25s} {'ΔR²':>10s} {'Significant'}")
    print(f"  {'-'*70}")

    granger_results = []
    for j in range(n_factors):
        for i in range(n_factors):
            if i == j:
                continue

            # Restricted model: target_t ~ target_{t-1}
            y = Y[:, i]
            x_restricted = X[:, i:i + 1]
            x_full = X[:, [i, j]]  # target_{t-1} + predictor_{t-1}

            # R² restricted
            beta_r = np.linalg.lstsq(x_restricted, y, rcond=None)[0]
            ss_res_r = np.sum((y - x_restricted @ beta_r) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_restricted = 1 - ss_res_r / max(ss_tot, 1e-10)

            # R² full
            beta_f = np.linalg.lstsq(x_full, y, rcond=None)[0]
            ss_res_f = np.sum((y - x_full @ beta_f) ** 2)
            r2_full = 1 - ss_res_f / max(ss_tot, 1e-10)

            delta_r2 = r2_full - r2_restricted

            # F-test for significance
            df1 = 1  # one extra predictor
            df2 = n_obs - 2  # full model df
            if ss_res_f > 0:
                f_stat = (ss_res_r - ss_res_f) / df1 / (ss_res_f / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)
            else:
                f_stat = 0
                p_value = 1.0

            if delta_r2 > 0.01 and p_value < 0.05:
                granger_results.append({
                    "predictor": FACTORS[j],
                    "target": FACTORS[i],
                    "delta_r2": round(float(delta_r2), 4),
                    "p_value": round(float(p_value), 6),
                })
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
                print(f"  {FACTORS[j]:25s} {FACTORS[i]:25s} {delta_r2:+10.4f}   {sig}")

    if not granger_results:
        print("  No significant lead/lag relationships found at weekly frequency")

    return {
        "persistence": persistence_results,
        "cross_dynamics": cross_dynamics,
        "residual_correlations": [
            {"a": a, "b": b, "r": round(float(r), 4)} for a, b, r in corr_pairs
        ],
        "improvements": improvements,
        "granger_results": granger_results,
        "A_hat": A_hat.tolist(),
        "residual_cov": residual_cov.tolist(),
    }


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 70)
    print("FACTOR VALIDATION (Phases 4 & 5)")
    print("=" * 70)

    results, raw_data, factor_df = load_data()

    # Phase 4
    phase4 = run_phase4(factor_df, raw_data)

    # Phase 5
    phase5 = run_phase5(factor_df)

    # Save combined results
    output = {
        "phase4_correlation": phase4,
        "phase5_cross_dynamics": phase5,
    }
    output_path = Path(__file__).parent.parent / "data" / "results" / "factor_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
