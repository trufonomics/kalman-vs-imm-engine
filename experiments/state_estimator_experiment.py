"""
State Estimator Experiment — March 11, 2026

Tests the core claims of the Heimdall Kalman filter + IMM architecture
against 1 year of real macro data from FRED + Yahoo Finance.

Hypothesis: The 8-factor Kalman filter, when fed real data chronologically,
produces coherent state estimates and the IMM correctly shifts probability
toward branches that best predict incoming data.

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/state_estimator_experiment.py
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    FACTORS,
    FACTOR_INDEX,
    STREAM_LOADINGS,
)
from heimdall.imm_tracker import IMMBranchTracker
from heimdall.trigger_service import TriggerService


# ── Data Configuration ──────────────────────────────────────────────

# FRED series → Kalman stream key mapping
FRED_STREAMS = {
    # Series ID: (kalman_key, transform)
    # transform: "level" = use as-is, "yoy" = compute YoY %, "diff" = first diff
    "CPIAUCSL": ("US_CPI_YOY", "yoy"),       # CPI All Urban, monthly
    "CPILFESL": ("CORE_CPI", "yoy"),          # Core CPI, monthly
    "PPIACO": ("PPI", "yoy"),                 # PPI All Commodities, monthly
    "UNRATE": ("UNEMPLOYMENT_RATE", "level"),  # Unemployment Rate, monthly
    "ICSA": ("INITIAL_CLAIMS", "level"),       # Initial Claims, weekly
    "JTSJOL": ("RETAIL_SALES", "level"),       # Job Openings (proxy activity)
    "HOUST": ("HOUSING_STARTS", "level"),      # Housing Starts, monthly
    "CSUSHPINSA": ("HOME_PRICES", "yoy"),      # Case-Shiller Home Price, monthly
    "RSAFS": ("RETAIL_SALES", "yoy"),          # Retail Sales, monthly
    "UMCSENT": ("CONSUMER_CONFIDENCE", "level"),  # Michigan Consumer Sentiment
    "FEDFUNDS": ("FED_FUNDS_RATE", "level"),   # Fed Funds Rate, monthly
    "DGS10": ("10Y_YIELD", "level"),           # 10Y Treasury Yield, daily
}

# Yahoo Finance tickers → (Kalman stream key, transform)
# "returns" = use daily log-returns (stationary) instead of levels (non-stationary)
# This is critical: financial time series levels are non-stationary, which
# causes z-scores to explode. Returns capture the *change* which is what
# the Kalman filter actually needs to estimate momentum.
YAHOO_STREAMS = {
    "^GSPC": ("SP500", "returns"),
    "CL=F": ("OIL_PRICE", "returns"),
    "GC=F": ("GOLD_PRICE", "returns"),
    "BTC-USD": ("BTC_USD", "returns"),
}

# 3 scenario branches for the IMM
BRANCHES = [
    {
        "branch_id": "soft_landing",
        "name": "Soft Landing",
        "probability": 0.50,
        "state_adjustments": {
            "inflation_trend": -0.05,
            "growth_trend": 0.10,
            "labor_pressure": -0.05,
            "financial_conditions": 0.10,
            "policy_stance": -0.15,
        },
    },
    {
        "branch_id": "stagflation",
        "name": "Stagflation",
        "probability": 0.30,
        "state_adjustments": {
            "inflation_trend": 0.20,
            "growth_trend": -0.10,
            "commodity_pressure": 0.25,
            "labor_pressure": 0.05,
            "policy_stance": 0.10,
        },
    },
    {
        "branch_id": "recession",
        "name": "Recession",
        "probability": 0.20,
        "state_adjustments": {
            "growth_trend": -0.30,
            "labor_pressure": -0.25,
            "consumer_sentiment": -0.20,
            "financial_conditions": -0.15,
            "housing_momentum": -0.20,
        },
    },
]


# ── Data Fetching ───────────────────────────────────────────────────

def fetch_fred_data(start: str, end: str) -> pd.DataFrame:
    """Fetch FRED series and return a DataFrame with kalman_key columns."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("  No FRED_API_KEY set — skipping FRED data.")
        print("  (Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html)")
        return pd.DataFrame()

    try:
        from fredapi import Fred
    except ImportError:
        print("  fredapi not installed — skipping FRED data.")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    frames = {}

    for series_id, (kalman_key, transform) in FRED_STREAMS.items():
        try:
            # For YoY transforms, fetch extra 13 months of history
            fetch_start = start
            if transform == "yoy":
                from dateutil.relativedelta import relativedelta
                fetch_start = (pd.Timestamp(start) - relativedelta(months=13)).strftime("%Y-%m-%d")
            data = fred.get_series(series_id, fetch_start, end)
            if data is None or data.empty:
                print(f"  SKIP {series_id} → {kalman_key} (no data)")
                continue

            if transform == "yoy":
                # Year-over-year percentage change
                data = data.pct_change(periods=12) * 100
                data = data.dropna()
                # Trim to original date range
                data = data[data.index >= start]
            elif transform == "diff":
                data = data.diff().dropna()

            # Avoid duplicate kalman_key columns — keep first
            if kalman_key not in frames:
                frames[kalman_key] = data.rename(kalman_key)
                print(f"  OK   {series_id} → {kalman_key} ({len(data)} obs, {transform})")
            else:
                print(f"  SKIP {series_id} → {kalman_key} (duplicate key, keeping first)")
        except Exception as e:
            print(f"  FAIL {series_id}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_yahoo_data(start: str, end: str) -> pd.DataFrame:
    """Fetch Yahoo Finance daily data."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    frames = {}
    for ticker, (kalman_key, transform) in YAHOO_STREAMS.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data is not None and not data.empty:
                close = data["Close"]
                if hasattr(close, "iloc"):
                    close = close.squeeze()

                if transform == "returns":
                    # Log returns: stationary, mean ~0, comparable across assets
                    series = np.log(close / close.shift(1)).dropna() * 100  # percent
                    print(f"  OK   {ticker} → {kalman_key} ({len(series)} obs, log-returns %)")
                else:
                    series = close
                    print(f"  OK   {ticker} → {kalman_key} ({len(series)} obs, levels)")

                frames[kalman_key] = series
            else:
                print(f"  SKIP {ticker} → {kalman_key} (no data)")
        except Exception as e:
            print(f"  FAIL {ticker}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df


# ── Normalization ───────────────────────────────────────────────────

def compute_normalization(df: pd.DataFrame, warmup_days: int = 60) -> dict:
    """Compute per-stream mean/std from the first N days of data."""
    cutoff = df.index.min() + timedelta(days=warmup_days)
    warmup = df[df.index <= cutoff]

    norms = {}
    for col in df.columns:
        series = warmup[col].dropna()
        if len(series) < 5:
            # Not enough warmup data, use full series
            series = df[col].dropna()
        if len(series) < 2:
            continue
        std = float(series.std())
        mean = float(series.mean())
        # Minimum std: 1% of mean (or 0.1 if mean is near zero)
        # This prevents division-by-zero for series that don't change
        # during warmup (e.g., fed funds rate held constant for months)
        min_std = max(abs(mean) * 0.01, 0.1)
        norms[col] = {
            "mean": mean,
            "std": max(std, min_std),
        }
    return norms


class RollingNormalizer:
    """
    Rolling window normalization that adapts as data arrives.
    Uses exponentially weighted mean/std to prevent normalization drift.

    This solves Open Question #5 from "Creating the Economy's State Estimator":
    baselines computed from a fixed window go stale when markets trend.
    """

    def __init__(self, halflife_days: int = 90):
        self.halflife = halflife_days
        self.alpha = 1 - np.exp(-np.log(2) / halflife_days)
        self.ema_mean: dict[str, float] = {}
        self.ema_var: dict[str, float] = {}
        self.initialized: dict[str, bool] = {}

    def initialize(self, norms: dict):
        """Seed from warmup statistics."""
        for key, n in norms.items():
            self.ema_mean[key] = n["mean"]
            self.ema_var[key] = n["std"] ** 2
            self.initialized[key] = True

    def normalize(self, key: str, raw: float) -> float:
        """Normalize value and update rolling stats."""
        if key not in self.initialized:
            return 0.0

        mean = self.ema_mean[key]
        std = max(np.sqrt(self.ema_var[key]), 1e-8)
        z = (raw - mean) / std

        # Update EMA
        self.ema_mean[key] = (1 - self.alpha) * mean + self.alpha * raw
        deviation_sq = (raw - self.ema_mean[key]) ** 2
        self.ema_var[key] = (1 - self.alpha) * self.ema_var[key] + self.alpha * deviation_sq

        return z


def normalize_value(raw: float, norm: dict) -> float:
    """Z-score normalization (static, for backward compat)."""
    return (raw - norm["mean"]) / norm["std"]


# ── Replay Engine ───────────────────────────────────────────────────

def run_replay(
    df: pd.DataFrame,
    norms: dict,
    warmup_days: int = 60,
) -> dict:
    """
    Feed data day-by-day through Kalman filter + IMM.
    Returns full log of the experiment.
    """
    # Initialize
    estimator = EconomicStateEstimator()
    tracker = IMMBranchTracker()
    trigger_service = TriggerService()

    # Rolling normalizer (solves normalization drift — Open Question #5)
    normalizer = RollingNormalizer(halflife_days=90)
    normalizer.initialize(norms)

    # Initialize IMM branches
    tracker.initialize_branches(BRANCHES, estimator)

    # Sort by date
    dates = sorted(df.index.unique())
    start_date = df.index.min() + timedelta(days=warmup_days)

    # Logs
    daily_log = []
    event_log = []
    trigger_log = []
    total_updates = 0
    anomalous_count = 0
    post_warmup_updates = 0
    post_warmup_anomalous = 0

    print(f"\n{'='*70}")
    print(f"REPLAY: {dates[0].date()} → {dates[-1].date()} ({len(dates)} days)")
    print(f"Warmup: first {warmup_days} days (normalization period)")
    print(f"Streams: {list(df.columns)}")
    print(f"Using: Rolling normalization (EMA halflife=90 days)")
    print(f"{'='*70}\n")

    for date in dates:
        row = df.loc[date]
        day_updates = []
        day_triggers = []
        is_post_warmup = date >= start_date

        # Predict step (time update)
        estimator.predict()
        tracker.predict()

        # Update with each available stream
        for kalman_key in df.columns:
            val = row.get(kalman_key) if isinstance(row, pd.Series) else row.get(kalman_key, None)

            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if kalman_key not in STREAM_LOADINGS:
                continue

            raw_val = float(val)
            z = normalizer.normalize(kalman_key, raw_val)

            # Kalman update
            state_update = estimator.update(kalman_key, z)
            if state_update is None:
                continue

            total_updates += 1
            if state_update.is_anomalous:
                anomalous_count += 1

            if is_post_warmup:
                post_warmup_updates += 1
                if state_update.is_anomalous:
                    post_warmup_anomalous += 1

            # IMM update
            imm_update = tracker.update(kalman_key, z)

            # Trigger checks
            k_triggers = trigger_service.process_kalman_update(state_update)
            i_triggers = trigger_service.process_imm_update(imm_update)

            day_updates.append({
                "stream": kalman_key,
                "raw": raw_val,
                "normalized": round(z, 4),
                "innovation": round(state_update.innovation, 4),
                "innovation_z": round(state_update.innovation_zscore, 4),
                "is_anomalous": state_update.is_anomalous,
            })

            for t in k_triggers + i_triggers:
                day_triggers.append(t.to_dict())
                trigger_log.append({
                    "date": str(date.date()),
                    "type": t.trigger_type.value if hasattr(t.trigger_type, 'value') else str(t.trigger_type),
                    "detail": t.detail,
                    "branch_id": t.branch_id,
                })

        # Get state and probabilities
        state = estimator.get_state()
        probs = tracker.get_probabilities()

        factors = {
            FACTORS[i]: round(float(state.mean[i]), 4)
            for i in range(len(FACTORS))
        }

        entry = {
            "date": str(date.date()),
            "factors": factors,
            "probabilities": probs,
            "updates": len(day_updates),
            "anomalous": sum(1 for u in day_updates if u["is_anomalous"]),
            "triggers": len(day_triggers),
        }
        daily_log.append(entry)

        # Print notable days
        has_anomaly = any(u["is_anomalous"] for u in day_updates)
        has_trigger = len(day_triggers) > 0
        if (has_anomaly or has_trigger) and is_post_warmup:
            event_log.append({
                "date": str(date.date()),
                "updates": day_updates,
                "triggers": day_triggers,
                "probabilities": probs,
                "factors": factors,
            })

            print(f"[{date.date()}] ", end="")
            if has_anomaly:
                anomalies = [u for u in day_updates if u["is_anomalous"]]
                for a in anomalies:
                    direction = "+" if a["innovation_z"] > 0 else ""
                    print(f"{a['stream']} z={direction}{a['innovation_z']:.1f} ", end="")
            print()
            print(f"  Probs: SL={probs.get('soft_landing', 0):.1%}  "
                  f"SF={probs.get('stagflation', 0):.1%}  "
                  f"RC={probs.get('recession', 0):.1%}")
            if day_triggers:
                for t in day_triggers:
                    ttype = t.get("trigger_type", t.get("type", "?"))
                    print(f"  TRIGGER: {ttype} — {t.get('detail', '')}")
            print()

    # Print periodic summaries (every 30 days after warmup)
    print(f"\n{'='*70}")
    print("PROBABILITY TIMELINE (every 30 days)")
    print(f"{'='*70}")
    post_warmup_entries = [e for e in daily_log if e["date"] >= str(start_date.date())]
    for i, entry in enumerate(post_warmup_entries):
        if i % 30 == 0 or i == len(post_warmup_entries) - 1:
            p = entry["probabilities"]
            f = entry["factors"]
            print(f"\n[{entry['date']}]")
            print(f"  Soft Landing: {p.get('soft_landing', 0):.1%}  "
                  f"Stagflation: {p.get('stagflation', 0):.1%}  "
                  f"Recession: {p.get('recession', 0):.1%}")
            print(f"  infl={f['inflation_trend']:+.2f}  "
                  f"grow={f['growth_trend']:+.2f}  "
                  f"labr={f['labor_pressure']:+.2f}  "
                  f"hous={f['housing_momentum']:+.2f}  "
                  f"fncl={f['financial_conditions']:+.2f}  "
                  f"comm={f['commodity_pressure']:+.2f}  "
                  f"cnsr={f['consumer_sentiment']:+.2f}  "
                  f"plcy={f['policy_stance']:+.2f}")

    return {
        "daily_log": daily_log,
        "event_log": event_log,
        "trigger_log": trigger_log,
        "stats": {
            "total_days": len(dates),
            "total_updates": total_updates,
            "total_anomalous": anomalous_count,
            "post_warmup_updates": post_warmup_updates,
            "post_warmup_anomalous": post_warmup_anomalous,
            "anomaly_rate_post_warmup": (
                post_warmup_anomalous / post_warmup_updates
                if post_warmup_updates > 0
                else 0
            ),
            "total_triggers": len(trigger_log),
        },
    }


# ── Analysis ────────────────────────────────────────────────────────

def analyze_results(results: dict) -> None:
    """Check success criteria."""
    stats = results["stats"]

    print(f"\n{'='*70}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*70}\n")

    # 1. Innovation z-scores
    anomaly_rate = stats["anomaly_rate_post_warmup"]
    z_pass = anomaly_rate < 0.20
    print(f"1. Innovation z-scores (post-warmup)")
    print(f"   Total updates: {stats['post_warmup_updates']}")
    print(f"   Anomalous (|z| > 2.5): {stats['post_warmup_anomalous']}")
    print(f"   Anomaly rate: {anomaly_rate:.1%}")
    print(f"   Criterion: <20% anomalous after warmup")
    print(f"   Result: {'PASS' if z_pass else 'FAIL'}")
    print()

    # 2. Final probabilities
    daily_log = results["daily_log"]
    if daily_log:
        final = daily_log[-1]
        p = final["probabilities"]
        f = final["factors"]
        print(f"2. Final state ({final['date']})")
        print(f"   Soft Landing: {p.get('soft_landing', 0):.1%}")
        print(f"   Stagflation:  {p.get('stagflation', 0):.1%}")
        print(f"   Recession:    {p.get('recession', 0):.1%}")
        print(f"   Factors: infl={f['inflation_trend']:+.3f}  "
              f"grow={f['growth_trend']:+.3f}  "
              f"labr={f['labor_pressure']:+.3f}  "
              f"comm={f['commodity_pressure']:+.3f}")
        print()

    # 3. Trigger summary
    trigger_log = results["trigger_log"]
    print(f"3. Trigger summary")
    print(f"   Total triggers: {stats['total_triggers']}")
    if trigger_log:
        from collections import Counter
        types = Counter(t["type"] for t in trigger_log)
        for ttype, count in types.most_common():
            print(f"   {ttype}: {count}")
    print()

    # 4. Probability trajectory
    print(f"4. Probability trajectory (start → end)")
    if len(daily_log) > 1:
        first_real = None
        for entry in daily_log:
            if entry["updates"] > 0:
                first_real = entry
                break
        last = daily_log[-1]
        if first_real:
            for branch in ["soft_landing", "stagflation", "recession"]:
                p0 = first_real["probabilities"].get(branch, 0)
                p1 = last["probabilities"].get(branch, 0)
                delta = p1 - p0
                direction = "+" if delta > 0 else ""
                print(f"   {branch}: {p0:.1%} → {p1:.1%} ({direction}{delta:.1%})")
    print()

    # 5. Notable events
    event_log = results["event_log"]
    print(f"5. Notable events ({len(event_log)} days with anomalies/triggers)")
    if event_log:
        for evt in event_log[:10]:  # Show first 10
            anomalies = [u for u in evt["updates"] if u["is_anomalous"]]
            if anomalies:
                streams = ", ".join(f"{a['stream']}(z={a['innovation_z']:+.1f})" for a in anomalies)
                print(f"   [{evt['date']}] {streams}")
    print()

    # Overall assessment
    print(f"{'='*70}")
    print("ASSESSMENT")
    print(f"{'='*70}")
    if z_pass:
        print("Filter normalization is working — innovation z-scores are in range.")
    else:
        print("WARNING: High anomaly rate suggests normalization or loading issues.")

    if daily_log:
        final_p = daily_log[-1]["probabilities"]
        dominant = max(final_p, key=final_p.get)
        print(f"Dominant scenario: {dominant} ({final_p[dominant]:.1%})")
        print("Check: Does this match what actually happened in the economy?")


# ── Main ────────────────────────────────────────────────────────────

def main():
    start_date = "2025-03-01"
    end_date = "2026-03-01"
    warmup_days = 60

    print("State Estimator Experiment")
    print(f"Period: {start_date} → {end_date}")
    print(f"Warmup: {warmup_days} days\n")

    # Step 1: Fetch data
    print("── Fetching FRED data ──")
    fred_df = fetch_fred_data(start_date, end_date)

    print("\n── Fetching Yahoo Finance data ──")
    yahoo_df = fetch_yahoo_data(start_date, end_date)

    # Combine (forward-fill monthly data to daily)
    if fred_df.empty and yahoo_df.empty:
        print("ERROR: No data fetched. Check API keys and internet connection.")
        sys.exit(1)

    # Merge on date index
    if not fred_df.empty and not yahoo_df.empty:
        combined = fred_df.join(yahoo_df, how="outer")
    elif not fred_df.empty:
        combined = fred_df
    else:
        combined = yahoo_df

    # DO NOT forward-fill. Only feed observations on days they actually update.
    # Forward-filling monthly data creates fake "no-change" observations that
    # wreck the Kalman filter's innovation statistics.
    combined = combined.dropna(how="all")

    print(f"\n── Combined dataset ──")
    print(f"Date range: {combined.index.min().date()} → {combined.index.max().date()}")
    print(f"Streams: {list(combined.columns)}")
    print(f"Total rows: {len(combined)}")

    # Filter to only streams the Kalman filter knows about
    valid_streams = [c for c in combined.columns if c in STREAM_LOADINGS]
    unknown = [c for c in combined.columns if c not in STREAM_LOADINGS]
    if unknown:
        print(f"Dropping unknown streams: {unknown}")
    combined = combined[valid_streams]
    print(f"Valid streams for Kalman: {valid_streams}")

    # Step 2: Compute normalization from warmup period
    print(f"\n── Computing normalization (first {warmup_days} days) ──")
    norms = compute_normalization(combined, warmup_days)
    for key, n in norms.items():
        print(f"  {key}: mean={n['mean']:.2f}, std={n['std']:.4f}")

    # Step 3: Run replay
    print("\n── Running replay ──")
    results = run_replay(combined, norms, warmup_days)

    # Step 4: Analyze
    analyze_results(results)

    # Step 5: Save results
    output_path = Path(__file__).parent.parent / "data" / "results" / "state_estimator_results.json"
    # Convert for JSON serialization
    serializable = {
        "stats": results["stats"],
        "trigger_log": results["trigger_log"],
        "event_log": results["event_log"][:50],  # Cap at 50 events
        "probability_timeline": [
            {
                "date": e["date"],
                "probabilities": e["probabilities"],
                "factors": e["factors"],
            }
            for i, e in enumerate(results["daily_log"])
            if i % 7 == 0 or i == len(results["daily_log"]) - 1
        ],
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
