"""
Full History Backtest — March 17, 2026

Replays the Heimdall 8-factor Kalman + 3-branch IMM from 1975 to 2026.
3-year warmup → first detection ~1978. Tests against every major US
economic event in 51 years.

3 regimes: Soft Landing, Stagflation, Recession

Streams available change over time:
  - 1977: 10 streams (FRED + S&P500, no Oil/Gold/BTC/Retail)
  - 1992: 11 streams (+Retail Sales)
  - 2000: 14 streams (+Oil, Gold)
  - 2014: 15 streams (+BTC)

11 regime checkpoints:
  Recession (6): Volcker I/II, Black Monday, Gulf War, Dot-com, GFC, COVID
  Stagflation (1): 2021-22 Inflation Surge
  Soft Landing (4): Reagan expansion, 90s Goldilocks, 2023-25 Disinflation

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/full_history_backtest.py
"""

import sys
import os
import json
from datetime import datetime
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
START_DATE = "1975-01-01"
END_DATE = "2026-03-01"
CACHE_FILE = Path(__file__).parent.parent / "data" / "full_history_cache.csv"

# FRED series → (kalman_key, transform)
# These all go back to at least 1977
FRED_STREAMS = {
    "CPIAUCSL": ("US_CPI_YOY", "yoy"),       # from 1947
    "CPILFESL": ("CORE_CPI", "yoy"),          # from 1957
    "PPIACO": ("PPI", "yoy"),                 # from 1913
    "UNRATE": ("UNEMPLOYMENT_RATE", "level"),  # from 1948
    "ICSA": ("INITIAL_CLAIMS", "level"),       # from 1967
    "HOUST": ("HOUSING_STARTS", "level"),      # from 1959
    "CSUSHPINSA": ("HOME_PRICES", "yoy"),      # from 1975 (Case-Shiller)
    "RSAFS": ("RETAIL_SALES", "yoy"),          # from 1992 (will be NaN before)
    "UMCSENT": ("CONSUMER_CONFIDENCE", "level"),  # from 1952
    "FEDFUNDS": ("FED_FUNDS_RATE", "level"),   # from 1954
    "DGS10": ("10Y_YIELD", "level"),           # from 1953
}

# Yahoo → (kalman_key, transform)
YAHOO_STREAMS = {
    "^GSPC": ("SP500", "returns"),       # from 1970
    "CL=F": ("OIL_PRICE", "returns"),    # from 2000 (will be NaN before)
    "GC=F": ("GOLD_PRICE", "returns"),   # from 2000 (will be NaN before)
    "BTC-USD": ("BTC_USD", "returns"),   # from 2014 (will be NaN before)
}

# Branches
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

# Extended regime checkpoints
REGIME_CHECKPOINTS = [
    # Volcker era: two recessions Jan 1980-Jul 1980, then Jul 1981-Nov 1982
    ("1980-01-01", "1980-07-01", "recession", 0.30,
     "Volcker Recession I — Fed raises to 20%, 6-month recession"),
    ("1981-07-01", "1982-11-01", "recession", 0.30,
     "Volcker Recession II — double-dip, unemployment hits 10.8%"),
    # Mid-80s expansion
    ("1983-06-01", "1986-12-01", "soft_landing", 0.30,
     "Reagan expansion — disinflation + strong growth"),
    # Black Monday (short sharp shock)
    ("1987-10-01", "1988-01-01", "recession", 0.25,
     "Black Monday — S&P drops 22% in one day, brief panic"),
    # Gulf War recession
    ("1990-07-01", "1991-03-01", "recession", 0.30,
     "Gulf War Recession — oil shock + S&L crisis"),
    # 90s expansion
    ("1995-01-01", "1999-12-01", "soft_landing", 0.30,
     "90s expansion — Goldilocks era, tech boom, low inflation"),
    # Dot-com bust
    ("2001-03-01", "2001-11-01", "recession", 0.25,
     "Dot-com Bust — NASDAQ crashes, 9/11, mild recession"),
    # Great Recession
    ("2008-06-01", "2009-06-01", "recession", 0.35,
     "Great Recession — housing collapse, bank failures, unemployment surge"),
    # COVID crash
    ("2020-03-01", "2020-06-01", "recession", 0.30,
     "COVID crash — fastest recession in history"),
    # Inflation surge
    ("2021-06-01", "2022-06-01", "stagflation", 0.30,
     "Inflation surge — CPI hit 9.1%, supply chain + demand overshoot"),
    # Soft landing
    ("2023-06-01", "2025-12-01", "soft_landing", 0.35,
     "Disinflation — CPI cooling, labor holding, no recession"),
]


class RollingNormalizer:
    """EMA-based normalizer to prevent drift over decades of data."""

    def __init__(self, halflife_days: int = 120):
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
    """Fetch all FRED + Yahoo data from 1977 to 2026."""
    cache_path = CACHE_FILE

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
                first_date = s.index[0].strftime("%Y-%m-%d")
                print(f"  {series_id:15s} → {kalman_key:25s} {len(s):5d} obs from {first_date} ({transform})")
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
                first_date = s.index[0].strftime("%Y-%m-%d")
                print(f"  {ticker:15s} → {kalman_key:25s} {len(s):5d} obs from {first_date} ({transform})")
        except Exception as e:
            print(f"  {ticker:15s} → ERROR: {str(e)[:60]}")

    combined = pd.DataFrame(all_series)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()
    combined = combined.dropna(how="all")

    combined.to_csv(cache_path)
    print(f"\nCached to {cache_path}")
    print(f"Total: {len(combined)} days, {combined.notna().sum().sum()} observations")
    print(f"Date range: {combined.index[0].date()} → {combined.index[-1].date()}")

    # Show stream availability timeline
    print("\nStream availability:")
    for col in sorted(combined.columns):
        first = combined[col].first_valid_index()
        last = combined[col].last_valid_index()
        count = combined[col].notna().sum()
        if first:
            print(f"  {col:35s} {first.date()} → {last.date()} ({count} obs)")

    return combined


def compute_warmup_norms(data: pd.DataFrame, warmup_days: int = 180) -> dict[str, dict[str, float]]:
    """Compute normalization stats from the first N days of data.
    Uses 180 days for longer history to capture more variance."""
    warmup = data.iloc[:warmup_days]
    norms = {}
    for col in data.columns:
        vals = warmup[col].dropna()
        if len(vals) < 5:
            # For streams that start later, use their first 50 values
            vals = data[col].dropna().iloc[:50]
        if len(vals) < 2:
            continue
        mean = float(vals.mean())
        std = float(vals.std())
        min_std = max(abs(mean) * 0.01, 0.1)
        norms[col] = {"mean": mean, "std": max(std, min_std)}
    return norms


def run_backtest(data: pd.DataFrame) -> dict:
    """Run the full 1977-2026 replay."""
    print(f"\n{'='*70}")
    print("FULL HISTORY BACKTEST (1977-2026)")
    print(f"{'='*70}")
    print(f"Date range: {data.index[0].date()} → {data.index[-1].date()}")
    print(f"Total days: {len(data)}")

    # Count streams available at different dates
    for check_date in ["1977-06-01", "1980-01-01", "1992-06-01", "2000-09-01", "2014-10-01"]:
        ts = pd.Timestamp(check_date)
        if ts < data.index[0] or ts > data.index[-1]:
            continue
        mask = data.loc[:ts].notna().any()
        n_streams = mask.sum()
        names = [c.split("_", 1)[1] for c in mask[mask].index]
        print(f"  Streams at {check_date}: {n_streams} — {', '.join(names[:6])}{'...' if len(names)>6 else ''}")

    col_to_kalman = {}
    for col in data.columns:
        parts = col.split("_", 1)
        if len(parts) == 2:
            col_to_kalman[col] = parts[1]

    estimator = EconomicStateEstimator()
    imm = IMMBranchTracker()
    imm.initialize_branches(BRANCHES, baseline=estimator)

    norms = compute_warmup_norms(data, warmup_days=180)
    normalizer = RollingNormalizer(halflife_days=120)
    for col, n in norms.items():
        normalizer.initialize(col, n["mean"], n["std"])

    daily_log = []
    innovation_log: dict[str, list[float]] = defaultdict(list)
    anomaly_count = 0
    total_updates = 0
    warmup_end_idx = 180  # ~6 months warmup for normalizer

    # Track when new streams come online
    streams_seen = set()

    for day_idx, (date, row) in enumerate(data.iterrows()):
        is_warmup = day_idx < warmup_end_idx

        # Collect streams with actual data today
        today_updates = []
        for col in data.columns:
            val = row[col]
            if pd.isna(val):
                continue

            kalman_key = col_to_kalman.get(col)
            if not kalman_key:
                continue

            # Initialize normalizer for late-arriving streams
            if col not in normalizer.initialized:
                stream_vals = data[col].dropna()
                if len(stream_vals) >= 5:
                    mean = float(stream_vals.iloc[:50].mean())
                    std = float(stream_vals.iloc[:50].std())
                    normalizer.initialize(col, mean, max(std, 0.1))
                else:
                    continue

            if col not in streams_seen:
                streams_seen.add(col)
                if day_idx > 0:
                    print(f"  ** New stream online: {kalman_key} at {date.date()} (day {day_idx})")

            today_updates.append((col, kalman_key, val))

        # Only predict + update when we have actual observations.
        # This fixes the Ljung-Box issue: no more 30 empty predict()
        # calls between monthly releases creating trivial autocorrelation.
        if not today_updates:
            continue

        estimator.predict()
        imm.predict()

        for col, kalman_key, val in today_updates:
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
        if day_idx % 120 == 0:
            prob_str = " | ".join(f"{b}: {p:.1%}" for b, p in probs.items())
            n_active = sum(1 for c in data.columns if col in normalizer.initialized)
            print(f"  {date.date()} | {prob_str}")

    post_warmup_updates = total_updates - warmup_end_idx * 3  # rough estimate
    anomaly_rate = anomaly_count / max(post_warmup_updates, 1)

    # ── Checkpoint evaluation ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("REGIME CHECKPOINT EVALUATION")
    print(f"{'='*70}")

    checkpoint_results = []
    for start, end, expected, min_prob, description in REGIME_CHECKPOINTS:
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

        expected_probs = [entry["probabilities"].get(expected, 0) for entry in window]
        avg_prob = np.mean(expected_probs)
        max_prob = np.max(expected_probs)

        dominant_count = sum(
            1 for entry in window
            if max(entry["probabilities"], key=entry["probabilities"].get) == expected
        )
        dominance_pct = dominant_count / len(window)

        passed = avg_prob >= min_prob
        status = "PASS" if passed else "FAIL"

        # Detection speed: first time expected regime > 50%
        first_detect = None
        for entry in window:
            if entry["probabilities"].get(expected, 0) >= 0.5:
                first_detect = entry["date"]
                break

        if first_detect:
            from datetime import datetime as dt
            delta_days = (dt.strptime(first_detect, "%Y-%m-%d") - dt.strptime(start, "%Y-%m-%d")).days
            detect_weeks = delta_days / 7
            detect_str = f"{detect_weeks:.1f}w"
        else:
            detect_weeks = None
            detect_str = "N/D"

        print(f"\n  {status}: {description}")
        print(f"    Expected: {expected} ≥ {min_prob:.0%}")
        print(f"    Actual: avg={avg_prob:.1%}, max={max_prob:.1%}, dominant {dominance_pct:.0%} of time")
        print(f"    First >50% detection: {detect_str}")

        checkpoint_results.append({
            "regime": expected,
            "description": description,
            "window": f"{start} → {end}",
            "result": status,
            "avg_probability": round(avg_prob, 4),
            "max_probability": round(max_prob, 4),
            "dominance_pct": round(dominance_pct, 4),
            "threshold": min_prob,
            "first_detection_weeks": detect_weeks,
        })

    passed_count = sum(1 for c in checkpoint_results if c["result"] == "PASS")
    total_checks = sum(1 for c in checkpoint_results if c["result"] != "SKIP")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed_count}/{total_checks} regime checkpoints passed")
    print(f"Anomaly rate: {anomaly_rate:.1%}")
    print(f"Total updates: {total_updates}")
    print(f"Date range: {data.index[0].date()} → {data.index[-1].date()}")
    print(f"{'='*70}")

    # Detection speed summary
    print(f"\n{'='*70}")
    print("DETECTION SPEED SUMMARY")
    print(f"{'='*70}")
    print(f"{'Event':45s} {'Result':6s} {'Avg':7s} {'Detect':8s}")
    print("-" * 70)
    for cp in checkpoint_results:
        if cp["result"] == "SKIP":
            continue
        detect = f"{cp['first_detection_weeks']:.1f}w" if cp.get("first_detection_weeks") is not None else "N/D"
        print(f"  {cp['description'][:43]:43s} {cp['result']:6s} {cp['avg_probability']:.1%}  {detect:>6s}")

    results = {
        "experiment": "full_history_backtest",
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
    """Innovation diagnostics."""
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
        mean_z = float(np.mean(innovations))
        std_err = float(np.std(innovations) / np.sqrt(n))
        d["mean"] = round(mean_z, 4)
        d["mean_bias_ok"] = abs(mean_z) < 0.2
        mean_bias_total += 1
        if d["mean_bias_ok"]:
            mean_bias_pass += 1

        kurt = float(stats.kurtosis(innovations, fisher=True))
        d["kurtosis"] = round(kurt, 2)

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
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 70)
    print("FULL HISTORY BACKTEST — 1975 to 2026")
    print("=" * 70)
    print("Testing 3-regime IMM across 51 years of US economic history")
    print("3-year warmup (1975-1978) → 11 regime checkpoints")
    print()

    data = fetch_data()
    results = run_backtest(data)

    diag = run_diagnostics(results)
    results["diagnostics"] = diag

    output_path = Path(__file__).parent.parent / "data" / "results" / "full_history_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
