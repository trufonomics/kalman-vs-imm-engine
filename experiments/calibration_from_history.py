"""
Research: Estimate Kalman filter parameters from 51 years of FRED data.

Compare data-driven estimates to the current theory-based hardcoded values
in kalman_filter.py. Tests whether starting from the 1970s gives stronger
calibration than the current 7-step warm-up approach.

Estimates:
  1. Persistence (AR(1) coefficients) → F diagonal
  2. Cross-dynamics (VAR(1) off-diagonal) → F off-diagonal
  3. Factor loadings (PCA) → H matrix
  4. Observation noise (residual variance) → R diagonal
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "backtest_data_cache_51yr.csv"

# Current hardcoded parameters from kalman_filter.py
CURRENT_PERSISTENCE = {
    "inflation_trend": 0.97,
    "growth_trend": 0.93,
    "labor_pressure": 0.96,
    "housing_momentum": 0.95,
    "financial_conditions": 0.88,
    "commodity_pressure": 0.85,
    "consumer_sentiment": 0.90,
    "policy_stance": 0.98,
}

CURRENT_CROSS_DYNAMICS = [
    ("policy_stance", "financial_conditions", 0.05),
    ("financial_conditions", "housing_momentum", 0.04),
    ("commodity_pressure", "inflation_trend", 0.06),
    ("labor_pressure", "consumer_sentiment", 0.03),
    ("growth_trend", "labor_pressure", 0.04),
]

# Map CSV columns to factor proxies
FACTOR_PROXIES = {
    "inflation_trend": ["FRED_US_CPI_YOY", "FRED_CORE_CPI", "FRED_PPI"],
    "growth_trend": ["FRED_RETAIL_SALES", "YAHOO_SP500"],
    "labor_pressure": ["FRED_UNEMPLOYMENT_RATE", "FRED_INITIAL_CLAIMS"],
    "housing_momentum": ["FRED_HOUSING_STARTS", "FRED_HOME_PRICES"],
    "financial_conditions": ["FRED_10Y_YIELD", "YAHOO_SP500"],
    "commodity_pressure": ["YAHOO_OIL_PRICE", "YAHOO_GOLD_PRICE"],
    "consumer_sentiment": ["FRED_CONSUMER_CONFIDENCE"],
    "policy_stance": ["FRED_FED_FUNDS_RATE"],
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Columns: {list(df.columns)}")
    return df


def estimate_persistence(df: pd.DataFrame) -> dict:
    """Estimate AR(1) coefficients for each factor proxy (monthly resampled)."""
    results = {}

    for factor, proxies in FACTOR_PROXIES.items():
        ar1_values = []
        for col in proxies:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 50:
                continue

            # Resample to monthly for comparable persistence
            monthly = series.resample("ME").last().dropna()
            if len(monthly) < 24:
                continue

            # AR(1): correlation of x_t with x_{t-1}
            ar1 = monthly.autocorr(lag=1)
            if not np.isnan(ar1):
                ar1_values.append(ar1)

        if ar1_values:
            avg_ar1 = float(np.mean(ar1_values))
            # Convert monthly AR(1) to daily: daily = monthly^(1/22)
            daily_equiv = abs(avg_ar1) ** (1 / 22) if avg_ar1 > 0 else 0.5
            results[factor] = {
                "monthly_ar1": round(avg_ar1, 4),
                "daily_equivalent": round(daily_equiv, 4),
                "current_daily": CURRENT_PERSISTENCE[factor],
                "proxies_used": [p for p in proxies if p in df.columns],
                "n_observations": len(ar1_values),
            }

    return results


def estimate_cross_dynamics(df: pd.DataFrame) -> list:
    """Estimate VAR(1) cross-factor coefficients from monthly data."""
    # Build monthly factor index from primary proxies
    primary = {
        "inflation_trend": "FRED_US_CPI_YOY",
        "growth_trend": "FRED_RETAIL_SALES",
        "labor_pressure": "FRED_UNEMPLOYMENT_RATE",
        "housing_momentum": "FRED_HOUSING_STARTS",
        "financial_conditions": "FRED_10Y_YIELD",
        "commodity_pressure": "YAHOO_OIL_PRICE",
        "consumer_sentiment": "FRED_CONSUMER_CONFIDENCE",
        "policy_stance": "FRED_FED_FUNDS_RATE",
    }

    monthly_factors = {}
    for factor, col in primary.items():
        if col in df.columns:
            series = df[col].resample("ME").last().dropna()
            # Standardize
            if series.std() > 0:
                monthly_factors[factor] = (series - series.mean()) / series.std()

    if len(monthly_factors) < 4:
        return []

    factor_df = pd.DataFrame(monthly_factors).dropna()

    # VAR(1): correlate each factor with lagged values of other factors
    lagged = factor_df.shift(1).dropna()
    current = factor_df.loc[lagged.index]

    results = []
    for target in current.columns:
        for source in lagged.columns:
            if source == target:
                continue
            try:
                corr = float(np.corrcoef(lagged[source], current[target])[0, 1])
                if abs(corr) > 0.15:
                    results.append({
                        "from": source,
                        "to": target,
                        "correlation": round(corr, 4),
                        "suggested_coeff": round(corr * 0.1, 4),
                    })
            except Exception:
                continue

    return results


def estimate_observation_noise(df: pd.DataFrame) -> dict:
    """Estimate observation noise from residual variance after removing trend."""
    results = {}

    col_to_stream = {
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
    }

    current_noise = {
        "US_CPI_YOY": 0.03, "CORE_CPI": 0.03, "PPI": 0.05,
        "UNEMPLOYMENT_RATE": 0.08, "INITIAL_CLAIMS": 0.10,
        "HOUSING_STARTS": 0.12, "HOME_PRICES": 0.05,
        "RETAIL_SALES": 0.08, "CONSUMER_CONFIDENCE": 0.20,
        "FED_FUNDS_RATE": 0.01, "10Y_YIELD": 0.03,
        "SP500": 0.05, "OIL_PRICE": 0.05, "GOLD_PRICE": 0.05,
    }

    for col, stream_key in col_to_stream.items():
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < 100:
            continue

        # Monthly resample
        monthly = series.resample("ME").last().dropna()
        if len(monthly) < 24:
            continue

        # Remove 12-month rolling trend
        trend = monthly.rolling(12, min_periods=6).mean()
        residuals = (monthly - trend).dropna()

        if len(residuals) > 12 and residuals.std() > 0:
            # Normalize: residual variance relative to series range
            normalized_noise = float(residuals.std() / (monthly.max() - monthly.min()))
            results[stream_key] = {
                "estimated_noise": round(normalized_noise, 4),
                "current_noise": current_noise.get(stream_key, 0.10),
                "n_months": len(monthly),
            }

    return results


def run_pca(df: pd.DataFrame) -> dict:
    """Run PCA on monthly factor proxies to estimate factor structure."""
    col_to_stream = {
        "FRED_US_CPI_YOY": "US_CPI_YOY",
        "FRED_CORE_CPI": "CORE_CPI",
        "FRED_PPI": "PPI",
        "FRED_UNEMPLOYMENT_RATE": "UNEMPLOYMENT_RATE",
        "FRED_HOUSING_STARTS": "HOUSING_STARTS",
        "FRED_RETAIL_SALES": "RETAIL_SALES",
        "FRED_CONSUMER_CONFIDENCE": "CONSUMER_CONFIDENCE",
        "FRED_FED_FUNDS_RATE": "FED_FUNDS_RATE",
        "FRED_10Y_YIELD": "10Y_YIELD",
        "YAHOO_SP500": "SP500",
        "YAHOO_OIL_PRICE": "OIL_PRICE",
        "YAHOO_GOLD_PRICE": "GOLD_PRICE",
    }

    available = {k: v for k, v in col_to_stream.items() if k in df.columns}
    monthly = df[list(available.keys())].resample("ME").last().dropna()
    monthly.columns = [available[c] for c in monthly.columns]

    # Standardize
    std_monthly = (monthly - monthly.mean()) / monthly.std()
    std_monthly = std_monthly.dropna()

    if len(std_monthly) < 50:
        return {"error": "Not enough data for PCA"}

    # PCA
    from numpy.linalg import svd
    X = std_monthly.values
    U, S, Vt = svd(X - X.mean(axis=0), full_matrices=False)

    # Explained variance
    explained = (S ** 2) / (S ** 2).sum()

    # First 8 components (matching our 8 factors)
    n_components = min(8, len(S))
    loadings = Vt[:n_components].T  # (n_streams, n_components)

    result = {
        "explained_variance": [round(float(e), 4) for e in explained[:n_components]],
        "cumulative_variance": round(float(explained[:n_components].sum()), 4),
        "n_months": len(std_monthly),
        "n_streams": len(available),
        "top_loadings_per_component": {},
    }

    for comp_i in range(n_components):
        comp_loadings = {}
        for stream_i, stream_name in enumerate(std_monthly.columns):
            val = float(loadings[stream_i, comp_i])
            if abs(val) > 0.2:
                comp_loadings[stream_name] = round(val, 3)
        # Sort by absolute value
        comp_loadings = dict(sorted(comp_loadings.items(), key=lambda x: abs(x[1]), reverse=True))
        result["top_loadings_per_component"][f"PC{comp_i + 1}"] = comp_loadings

    return result


def compare_eras(df: pd.DataFrame) -> dict:
    """Compare persistence estimates across economic eras to test regime stability."""
    eras = {
        "1975-1985 (Volcker)": ("1975-01-01", "1985-12-31"),
        "1986-2000 (Great Moderation)": ("1986-01-01", "2000-12-31"),
        "2001-2008 (Dot-com to GFC)": ("2001-01-01", "2008-12-31"),
        "2009-2019 (ZIRP)": ("2009-01-01", "2019-12-31"),
        "2020-2026 (Post-COVID)": ("2020-01-01", "2026-12-31"),
    }

    primary = {
        "inflation": "FRED_US_CPI_YOY",
        "growth": "FRED_RETAIL_SALES",
        "labor": "FRED_UNEMPLOYMENT_RATE",
        "rates": "FRED_FED_FUNDS_RATE",
    }

    results = {}
    for era_name, (start, end) in eras.items():
        era_df = df.loc[start:end]
        era_results = {}
        for factor_name, col in primary.items():
            if col not in era_df.columns:
                continue
            monthly = era_df[col].resample("ME").last().dropna()
            if len(monthly) < 12:
                continue
            ar1 = monthly.autocorr(lag=1)
            if not np.isnan(ar1):
                daily = abs(ar1) ** (1 / 22) if ar1 > 0 else 0.5
                era_results[factor_name] = {
                    "monthly_ar1": round(float(ar1), 4),
                    "daily_equiv": round(float(daily), 4),
                }
        results[era_name] = era_results

    return results


def main():
    df = load_data()

    print("\n" + "=" * 70)
    print("1. PERSISTENCE ESTIMATES (AR(1) → F diagonal)")
    print("=" * 70)
    persistence = estimate_persistence(df)
    for factor, vals in sorted(persistence.items()):
        current = vals["current_daily"]
        estimated = vals["daily_equivalent"]
        diff = estimated - current
        marker = "<<< DIVERGENT" if abs(diff) > 0.03 else ""
        print(f"  {factor:25s}  current={current:.2f}  estimated={estimated:.4f}  "
              f"monthly_ar1={vals['monthly_ar1']:.4f}  diff={diff:+.4f} {marker}")

    print("\n" + "=" * 70)
    print("2. CROSS-DYNAMICS (VAR(1) off-diagonal)")
    print("=" * 70)
    cross = estimate_cross_dynamics(df)
    # Compare to current
    current_cross_set = {(c[0], c[1]): c[2] for c in CURRENT_CROSS_DYNAMICS}
    for item in sorted(cross, key=lambda x: abs(x["correlation"]), reverse=True)[:15]:
        current_val = current_cross_set.get((item["from"], item["to"]), None)
        marker = f"  (current: {current_val})" if current_val else "  NEW"
        print(f"  {item['from']:25s} → {item['to']:25s}  corr={item['correlation']:+.4f}  "
              f"suggested={item['suggested_coeff']:+.4f}{marker}")

    print("\n" + "=" * 70)
    print("3. OBSERVATION NOISE (R diagonal)")
    print("=" * 70)
    noise = estimate_observation_noise(df)
    for stream, vals in sorted(noise.items()):
        current = vals["current_noise"]
        estimated = vals["estimated_noise"]
        diff = estimated - current
        marker = "<<< DIVERGENT" if abs(diff) > 0.05 else ""
        print(f"  {stream:25s}  current={current:.2f}  estimated={estimated:.4f}  "
              f"months={vals['n_months']}  diff={diff:+.4f} {marker}")

    print("\n" + "=" * 70)
    print("4. PCA FACTOR STRUCTURE")
    print("=" * 70)
    pca = run_pca(df)
    if "error" not in pca:
        print(f"  {pca['n_months']} months, {pca['n_streams']} streams")
        print(f"  Explained variance (first 8 PCs): {pca['explained_variance']}")
        print(f"  Cumulative: {pca['cumulative_variance']:.1%}")
        print()
        for pc, loadings in pca["top_loadings_per_component"].items():
            print(f"  {pc}: {loadings}")
    else:
        print(f"  {pca['error']}")

    print("\n" + "=" * 70)
    print("5. ERA COMPARISON (regime stability test)")
    print("=" * 70)
    eras = compare_eras(df)
    # Print as table
    factors = ["inflation", "growth", "labor", "rates"]
    header = f"  {'Era':35s}" + "".join(f"  {f:12s}" for f in factors)
    print(header)
    print("  " + "-" * (35 + 14 * len(factors)))
    for era_name, era_vals in eras.items():
        row = f"  {era_name:35s}"
        for f in factors:
            if f in era_vals:
                row += f"  {era_vals[f]['daily_equiv']:12.4f}"
            else:
                row += f"  {'--':>12s}"
        print(row)

    # Current values for comparison
    current_row = f"  {'CURRENT (hardcoded)':35s}"
    current_map = {"inflation": 0.97, "growth": 0.93, "labor": 0.96, "rates": 0.98}
    for f in factors:
        current_row += f"  {current_map[f]:12.2f}"
    print(current_row)

    # Save full results
    output = {
        "persistence": persistence,
        "cross_dynamics": cross,
        "noise": noise,
        "pca": pca,
        "era_comparison": eras,
    }
    out_path = Path(__file__).parent.parent / "data" / "results" / "calibration_from_history_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
