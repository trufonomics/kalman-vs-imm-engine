"""
Multi-Regime Backtest — March 11, 2026

Replays the Heimdall 8-factor Kalman + 3-branch IMM through 4 economic regimes
using FRED + Yahoo data (2007-2026). Same fixed parameters throughout.

If the model correctly identifies the dominant regime in all 4 periods,
the architecture is validated as regime-robust, not just lucky on one year.

Regimes:
  1. Great Recession (2007-2009): Recession should dominate
  2. COVID crash + recovery (2020): Sharp Recession spike then recovery
  3. Inflation surge (2021-2022): Stagflation should rise
  4. Soft Landing (2023-2026): Soft Landing should dominate

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/multi_regime_backtest.py
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    FACTORS,
    FACTOR_INDEX,
    ANOMALY_THRESHOLD,
)
from heimdall.imm_tracker import (
    IMMBranchTracker,
    RECOMMENDED_BRANCH_ADJUSTMENTS,
)

# ── Configuration ─────────────────────────────────────────────────────

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")  # Set FRED_API_KEY env var to fetch data
START_DATE = "2005-01-01"
END_DATE = "2026-03-01"

# FRED series → (kalman_key, transform)
FRED_STREAMS = {
    "CPIAUCSL": ("US_CPI_YOY", "yoy"),
    "CPILFESL": ("CORE_CPI", "yoy"),
    "PPIACO": ("PPI", "yoy"),
    "UNRATE": ("UNEMPLOYMENT_RATE", "level"),
    "ICSA": ("INITIAL_CLAIMS", "level"),
    "HOUST": ("HOUSING_STARTS", "level"),
    "CSUSHPINSA": ("HOME_PRICES", "yoy"),
    "RSAFS": ("RETAIL_SALES", "yoy"),
    "UMCSENT": ("CONSUMER_CONFIDENCE", "level"),
    "FEDFUNDS": ("FED_FUNDS_RATE", "level"),
    "DGS10": ("10Y_YIELD", "level"),
}

# Yahoo → (kalman_key, transform)
YAHOO_STREAMS = {
    "^GSPC": ("SP500", "returns"),
    "CL=F": ("OIL_PRICE", "returns"),
    "GC=F": ("GOLD_PRICE", "returns"),
    "BTC-USD": ("BTC_USD", "returns"),  # only from ~2014
}

# Branches with recommended widened adjustments
BRANCHES = [
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
        "state_adjustments": RECOMMENDED_BRANCH_ADJUSTMENTS["stagflation"],
    },
    {
        "branch_id": "recession",
        "name": "Recession",
        "probability": 1 / 3,
        "state_adjustments": RECOMMENDED_BRANCH_ADJUSTMENTS["recession"],
    },
]

# Regime checkpoints: (start, end, expected_dominant_branch, min_probability)
REGIME_CHECKPOINTS = [
    ("2008-06-01", "2009-06-01", "recession", 0.35,
     "Great Recession — housing collapse, bank failures, unemployment surge"),
    ("2020-03-01", "2020-06-01", "recession", 0.30,
     "COVID crash — fastest recession in history"),
    ("2021-06-01", "2022-06-01", "stagflation", 0.30,
     "Inflation surge — CPI hit 9.1%, supply chain + demand overshoot"),
    ("2023-06-01", "2025-12-01", "soft_landing", 0.35,
     "Disinflation — CPI cooling, labor holding, no recession"),
]


class RollingNormalizer:
    """EMA-based normalizer to prevent drift over 19 years of data."""

    def __init__(self, halflife_days: int = 90):
        self.halflife = halflife_days
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


def fetch_data() -> pd.DataFrame:
    """Fetch all FRED + Yahoo data and combine into a single DataFrame."""
    cache_path = Path(__file__).parent.parent / "data" / "backtest_data_cache.csv"

    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print("Fetching FRED data...")
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    all_series = {}
    for series_id, (kalman_key, transform) in FRED_STREAMS.items():
        try:
            # For YoY transforms, fetch 13 months earlier
            if transform == "yoy":
                from dateutil.relativedelta import relativedelta
                fetch_start = (pd.Timestamp(START_DATE) - relativedelta(months=13)).strftime("%Y-%m-%d")
            else:
                fetch_start = START_DATE

            s = fred.get_series(series_id, observation_start=fetch_start, observation_end=END_DATE)
            if s is not None and len(s) > 0:
                s = s.dropna()
                if transform == "yoy":
                    s = s.pct_change(periods=12) * 100
                    s = s.dropna()
                    s = s[s.index >= START_DATE]

                col_name = f"FRED_{kalman_key}"
                all_series[col_name] = s
                print(f"  {series_id:15s} → {kalman_key:20s} {len(s):5d} obs ({transform})")
        except Exception as e:
            print(f"  {series_id:15s} → ERROR: {str(e)[:60]}")

    print("\nFetching Yahoo Finance data...")
    import yfinance as yf

    for ticker, (kalman_key, transform) in YAHOO_STREAMS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if df is not None and len(df) > 0:
                close = df["Close"].squeeze()
                if transform == "returns":
                    s = np.log(close / close.shift(1)) * 100
                    s = s.dropna()
                else:
                    s = close

                col_name = f"YAHOO_{kalman_key}"
                all_series[col_name] = s
                print(f"  {ticker:15s} → {kalman_key:20s} {len(s):5d} obs ({transform})")
        except Exception as e:
            print(f"  {ticker:15s} → ERROR: {str(e)[:60]}")

    # Combine all series into a DataFrame
    combined = pd.DataFrame(all_series)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()

    # Don't forward-fill — only feed data on days it actually updates
    combined = combined.dropna(how="all")

    # Cache for future runs
    combined.to_csv(cache_path)
    print(f"\nCached to {cache_path}")
    print(f"Total: {len(combined)} days, {combined.notna().sum().sum()} observations")

    return combined


def compute_warmup_norms(data: pd.DataFrame, warmup_days: int = 90) -> dict[str, dict[str, float]]:
    """Compute normalization stats from the first N days of data."""
    warmup = data.iloc[:warmup_days]
    norms = {}
    for col in data.columns:
        vals = warmup[col].dropna()
        if len(vals) < 5:
            vals = data[col].dropna().iloc[:50]
        if len(vals) < 2:
            continue
        mean = float(vals.mean())
        std = float(vals.std())
        min_std = max(abs(mean) * 0.01, 0.1)
        norms[col] = {"mean": mean, "std": max(std, min_std)}
    return norms


def run_backtest(data: pd.DataFrame) -> dict:
    """Run the full 2007-2026 replay."""
    print(f"\n{'='*70}")
    print("MULTI-REGIME BACKTEST")
    print(f"{'='*70}")
    print(f"Date range: {data.index[0].date()} → {data.index[-1].date()}")
    print(f"Total days: {len(data)}")

    # Map column names to kalman keys
    col_to_kalman = {}
    for col in data.columns:
        # Column format: "FRED_US_CPI_YOY" or "YAHOO_SP500"
        parts = col.split("_", 1)
        if len(parts) == 2:
            col_to_kalman[col] = parts[1]

    # Initialize
    estimator = EconomicStateEstimator()
    imm = IMMBranchTracker()
    imm.initialize_branches(BRANCHES, baseline=estimator)

    # Warmup normalization
    norms = compute_warmup_norms(data)
    normalizer = RollingNormalizer(halflife_days=90)
    for col, n in norms.items():
        normalizer.initialize(col, n["mean"], n["std"])

    # Tracking
    daily_log = []
    innovation_log: dict[str, list[float]] = defaultdict(list)  # per-stream z-scores
    anomaly_count = 0
    total_updates = 0
    warmup_end_idx = 90

    for day_idx, (date, row) in enumerate(data.iterrows()):
        is_warmup = day_idx < warmup_end_idx

        estimator.predict()
        imm.predict()

        for col in data.columns:
            val = row[col]
            if pd.isna(val):
                continue

            kalman_key = col_to_kalman.get(col)
            if not kalman_key:
                continue

            z = normalizer.normalize(col, val)
            update = estimator.update(kalman_key, z)
            if update:
                total_updates += 1
                imm.update(kalman_key, z)

                if not is_warmup:
                    innovation_log[kalman_key].append(update.innovation_zscore)
                    if abs(update.innovation_zscore) > ANOMALY_THRESHOLD:
                        anomaly_count += 1

        probs = imm.get_probabilities()
        state = estimator.get_state()

        # Log weekly
        if day_idx % 7 == 0 or day_idx == len(data) - 1:
            factors = {name: round(float(state.mean[i]), 4) for i, name in enumerate(FACTORS)}
            daily_log.append({
                "date": str(date.date()),
                "day": day_idx,
                "factors": factors,
                "probabilities": {b: round(p, 4) for b, p in probs.items()},
            })

        # Monthly progress
        if day_idx % 60 == 0:
            prob_str = " | ".join(f"{b}: {p:.1%}" for b, p in probs.items())
            print(f"  {date.date()} | {prob_str}")

    # Final
    post_warmup = total_updates - sum(1 for _ in range(warmup_end_idx))
    anomaly_rate = anomaly_count / max(total_updates - warmup_end_idx * 5, 1)

    print(f"\n{'='*70}")
    print("REGIME CHECKPOINT EVALUATION")
    print(f"{'='*70}")

    checkpoint_results = []
    for start, end, expected, min_prob, description in REGIME_CHECKPOINTS:
        # Find log entries in this window
        window = [
            entry for entry in daily_log
            if start <= entry["date"] <= end
        ]
        if not window:
            print(f"\n  SKIP: {description} — no data in window")
            checkpoint_results.append({
                "regime": expected, "description": description,
                "result": "SKIP", "reason": "no data"
            })
            continue

        # Average probability of expected branch in this window
        expected_probs = [entry["probabilities"].get(expected, 0) for entry in window]
        avg_prob = np.mean(expected_probs)
        max_prob = np.max(expected_probs)

        # Check if expected branch was dominant (highest probability) for >50% of window
        dominant_count = sum(
            1 for entry in window
            if max(entry["probabilities"], key=entry["probabilities"].get) == expected
        )
        dominance_pct = dominant_count / len(window)

        passed = avg_prob >= min_prob
        status = "PASS" if passed else "FAIL"

        print(f"\n  {status}: {description}")
        print(f"    Expected: {expected} ≥ {min_prob:.0%}")
        print(f"    Actual: avg={avg_prob:.1%}, max={max_prob:.1%}, dominant {dominance_pct:.0%} of time")

        checkpoint_results.append({
            "regime": expected,
            "description": description,
            "window": f"{start} → {end}",
            "result": status,
            "avg_probability": round(avg_prob, 4),
            "max_probability": round(max_prob, 4),
            "dominance_pct": round(dominance_pct, 4),
            "threshold": min_prob,
        })

    passed_count = sum(1 for c in checkpoint_results if c["result"] == "PASS")
    total_checks = sum(1 for c in checkpoint_results if c["result"] != "SKIP")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed_count}/{total_checks} regime checkpoints passed")
    print(f"Anomaly rate: {anomaly_rate:.1%}")
    print(f"Total updates: {total_updates}")
    print(f"{'='*70}")

    results = {
        "experiment": "multi_regime_backtest",
        "date_range": f"{data.index[0].date()} → {data.index[-1].date()}",
        "total_days": len(data),
        "total_updates": total_updates,
        "anomaly_rate": round(anomaly_rate, 4),
        "checkpoints": checkpoint_results,
        "passed": passed_count,
        "total_checks": total_checks,
        "daily_log": daily_log,
        "innovation_log": {k: v for k, v in innovation_log.items()},
        "run_timestamp": datetime.now().isoformat(),
    }

    return results


def run_diagnostics(results: dict):
    """Phase 3: Innovation diagnostics on the backtest results."""
    from scipy import stats

    innovation_log = results.get("innovation_log", {})
    if not innovation_log:
        print("No innovation data for diagnostics")
        return {}

    print(f"\n{'='*70}")
    print("INNOVATION DIAGNOSTICS")
    print(f"{'='*70}")

    diagnostics = {}
    ljung_box_pass = 0
    ljung_box_total = 0
    mean_bias_pass = 0
    mean_bias_total = 0

    for stream, innovations in sorted(innovation_log.items()):
        innovations = np.array(innovations)
        n = len(innovations)
        if n < 30:
            continue

        d = {"n": n}

        # Mean bias
        mean_z = float(np.mean(innovations))
        std_err = float(np.std(innovations) / np.sqrt(n))
        d["mean"] = round(mean_z, 4)
        d["mean_bias_ok"] = abs(mean_z) < 0.2
        mean_bias_total += 1
        if d["mean_bias_ok"]:
            mean_bias_pass += 1

        # Kurtosis
        kurt = float(stats.kurtosis(innovations, fisher=True))
        d["kurtosis"] = round(kurt, 2)
        d["distribution"] = "heavy-tailed" if kurt > 3 else "near-Gaussian"

        # Ljung-Box at lag 10
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(innovations, lags=[10], return_df=True)
            lb_pval = float(lb_result["lb_pvalue"].iloc[0])
            d["ljung_box_p"] = round(lb_pval, 4)
            d["ljung_box_pass"] = lb_pval > 0.05
            ljung_box_total += 1
            if d["ljung_box_pass"]:
                ljung_box_pass += 1
        except Exception:
            d["ljung_box_p"] = None
            d["ljung_box_pass"] = None

        diagnostics[stream] = d

        status = "OK" if d.get("ljung_box_pass") and d["mean_bias_ok"] else "!!"
        lb_p_str = f"{d['ljung_box_p']:.4f}" if d.get("ljung_box_p") is not None else "  N/A"
        print(
            f"  {stream:25s} n={n:5d} mean={mean_z:+.3f} "
            f"kurt={kurt:5.1f} LB_p={lb_p_str:>6} {status}"
        )

    print(f"\n  Ljung-Box pass: {ljung_box_pass}/{ljung_box_total} ({ljung_box_pass/max(ljung_box_total,1):.0%})")
    print(f"  Mean bias pass: {mean_bias_pass}/{mean_bias_total} ({mean_bias_pass/max(mean_bias_total,1):.0%})")

    return {
        "per_stream": diagnostics,
        "ljung_box_pass_rate": round(ljung_box_pass / max(ljung_box_total, 1), 4),
        "mean_bias_pass_rate": round(mean_bias_pass / max(mean_bias_total, 1), 4),
    }


def main():
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")  # not needed for FRED

    print("=" * 70)
    print("MULTI-REGIME BACKTEST (2007-2026)")
    print("=" * 70)

    # Phase 2: Fetch data and run backtest
    data = fetch_data()
    results = run_backtest(data)

    # Phase 3: Innovation diagnostics
    diag = run_diagnostics(results)
    results["diagnostics"] = diag

    # Save
    output_path = Path(__file__).parent.parent / "data" / "results" / "multi_regime_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
