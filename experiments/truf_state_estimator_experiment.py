"""
State Estimator Experiment (TRUF Streams) — March 11, 2026

Same experiment as state_estimator_experiment.py but using ACTUAL TRUF Network
data streams instead of FRED/Yahoo proxies. This tests the exact data pipeline
that Heimdall uses in production.

Uses:
- trufnetwork-sdk-py to fetch 365 days of history for all mapped streams
- TRUF_TO_KALMAN mappings from stream_pipeline.py
- NORMALIZATION from stream_pipeline.py (calibrated from real TRUF data)
- Same 8-factor Kalman + 3-branch IMM as the FRED experiment

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/truf_state_estimator_experiment.py
"""

import sys
import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

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
from heimdall.stream_pipeline import (
    TRUF_TO_KALMAN,
    NORMALIZATION,
)
# NOTE: Requires TRUF client for live stream data.
# from app.services.truf_client import KNOWN_STREAMS, TRUFLATION_PROVIDER

# ── Configuration ─────────────────────────────────────────────────────

TRUF_GATEWAY = "https://gateway.mainnet.truf.network"
DAYS_HISTORY = 365

# Streams to skip (no real TRUF history)
SKIP_STREAMS = {
    "CONSUMER_CONFIDENCE",  # no real TRUF history (default normalization)
}

# Streams that are price LEVELS and need log-returns transformation.
# Without this, raw prices are non-stationary and every observation
# looks anomalous as prices drift away from the normalization mean.
PRICE_LEVEL_STREAMS = {
    "BITCOIN", "TOTAL_MARKET_CAP", "SP500",
    "CRUDE_OIL_BRENT", "NICKEL_FUTURES", "EV_COMMODITY", "SILICON_USD",
    # Index levels (monthly but still non-stationary)
    "BLS_CPI", "BLS_CPI_CORE", "PCE_INDEX", "PCE_CORE",
    "TRUFLATION_PCE", "PPI", "PPI_CORE",
    # Housing/retail levels
    "EXISTING_HOME_PRICE", "FOR_SALE_INVENTORY", "MORTGAGE_DEBT",
    "TOTAL_RETAIL_SALES", "FOOD_SERVICE_SALES",
    # Labor counts
    "INITIAL_CLAIMS", "CONTINUED_CLAIMS",
    "JOB_OPENINGS", "JOB_HIRES", "JOB_SEPARATIONS",
}

# Streams that are already rates/percentages — feed as-is
# US_INFLATION (YoY %), UNEMPLOYMENT (%), MORTGAGE_30YR (%),
# RENTAL_VACANCY (%), WAGE_INFLATION ($/hr), BIG_MAC_US ($)

# ── Tier 1 Fix #1: Anomaly threshold ──
# 2.5σ assumes Gaussian. Financial data has kurtosis 5-10.
# 3.5σ captures 99.95% of Gaussian, ~97% of Student-t(5).
ANOMALY_THRESHOLD = 3.5

# ── Tier 1 Fix #3: Widened branch adjustments (3-5x) ──
# Old adjustments were ~0.05-0.25 — less than 1/10th of daily noise.
# New adjustments are in units of ~1σ of the factor's annual variation.
BRANCHES = [
    {
        "branch_id": "soft_landing",
        "name": "Soft Landing",
        "probability": 0.50,
        "state_adjustments": {
            "inflation_trend": -0.20,
            "growth_trend": 0.40,
            "labor_pressure": -0.20,
            "financial_conditions": 0.30,
            "policy_stance": -0.50,
        },
    },
    {
        "branch_id": "stagflation",
        "name": "Stagflation",
        "probability": 0.30,
        "state_adjustments": {
            "inflation_trend": 0.60,
            "growth_trend": -0.40,
            "commodity_pressure": 0.80,
            "labor_pressure": 0.15,
            "policy_stance": 0.30,
        },
    },
    {
        "branch_id": "recession",
        "name": "Recession",
        "probability": 0.20,
        "state_adjustments": {
            "growth_trend": -1.00,
            "labor_pressure": -0.80,
            "consumer_sentiment": -0.70,
            "financial_conditions": -0.50,
            "housing_momentum": -0.60,
        },
    },
]


class RollingNormalizer:
    """EMA-based rolling normalizer (same as FRED experiment).

    Prevents normalization drift by adapting mean/std with halflife.
    """

    def __init__(self, halflife_days: int = 90):
        self.halflife = halflife_days
        self.alpha = 1 - np.exp(-np.log(2) / halflife_days)
        self.ema_mean: dict[str, float] = {}
        self.ema_var: dict[str, float] = {}
        self.initialized: dict[str, bool] = {}

    def initialize_from_stream_normalization(self):
        """Seed from NORMALIZATION (calibrated from 1yr TRUF history).

        For price-level streams that get log-return transformed,
        we use log-return normalization (mean~0, std based on asset class).
        """
        for key, norm in NORMALIZATION.items():
            if key in PRICE_LEVEL_STREAMS:
                # Log-returns: mean ≈ 0, std depends on asset volatility
                # Daily log-return stds (approximate):
                #   crypto: ~3-5%  equity: ~1-2%  commodity: ~1.5-3%
                #   index levels (CPI etc): ~0.05-0.2%
                if key in ("BITCOIN", "TOTAL_MARKET_CAP"):
                    self.ema_mean[key] = 0.0
                    self.ema_var[key] = 3.0 ** 2  # ~3% daily vol
                elif key in ("SP500",):
                    self.ema_mean[key] = 0.0
                    self.ema_var[key] = 1.2 ** 2
                elif key in ("CRUDE_OIL_BRENT", "NICKEL_FUTURES", "EV_COMMODITY", "SILICON_USD"):
                    self.ema_mean[key] = 0.0
                    self.ema_var[key] = 2.0 ** 2
                elif key in ("INITIAL_CLAIMS", "CONTINUED_CLAIMS",
                             "JOB_OPENINGS", "JOB_HIRES", "JOB_SEPARATIONS"):
                    self.ema_mean[key] = 0.0
                    self.ema_var[key] = 2.0 ** 2  # monthly jumps
                else:
                    # CPI/PCE/PPI index levels → very small monthly changes
                    self.ema_mean[key] = 0.0
                    self.ema_var[key] = 0.3 ** 2
            else:
                self.ema_mean[key] = norm["mean"]
                self.ema_var[key] = norm["std"] ** 2
            self.initialized[key] = True

    def normalize(self, key: str, raw: float) -> float:
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


def fetch_all_truf_streams() -> dict[str, pd.DataFrame]:
    """Fetch historical data for all TRUF streams mapped to Kalman keys.

    Returns: {truf_key: DataFrame with columns [date, value]}
    """
    from trufnetwork_sdk_py.client import TNClient

    key = os.environ.get("TRUF_PRIVATE_KEY", "")
    if key.startswith("0x"):
        key = key[2:]

    client = TNClient(TRUF_GATEWAY, key)

    to_time = datetime.now(timezone.utc)
    from_time = to_time - timedelta(days=DAYS_HISTORY)
    date_from = int(from_time.timestamp())
    date_to = int(to_time.timestamp())

    all_data: dict[str, pd.DataFrame] = {}
    stream_keys = [k for k in TRUF_TO_KALMAN.keys() if k not in SKIP_STREAMS]

    print(f"\nFetching {len(stream_keys)} TRUF streams ({DAYS_HISTORY} days)...")

    for i, truf_key in enumerate(stream_keys):
        meta = KNOWN_STREAMS.get(truf_key)
        if not meta:
            print(f"  [{i+1}/{len(stream_keys)}] {truf_key}: no metadata, skipping")
            continue

        try:
            resp = client.get_records(
                stream_id=meta["streamId"],
                data_provider=meta["provider"],
                date_from=date_from,
                date_to=date_to,
                use_cache=False,
            )
            records = resp.data if hasattr(resp, "data") else resp

            if not records:
                print(f"  [{i+1}/{len(stream_keys)}] {truf_key}: empty")
                continue

            rows = []
            for rec in records:
                try:
                    value = float(rec.Value)
                    event_time = int(rec.EventTime)
                    dt = datetime.fromtimestamp(event_time, tz=timezone.utc).date()
                    rows.append({"date": dt, "value": value})
                except (ValueError, TypeError, AttributeError):
                    continue

            if rows:
                df = pd.DataFrame(rows)
                # Deduplicate by date (keep last if multiple records same day)
                df = df.drop_duplicates(subset="date", keep="last")
                df = df.sort_values("date").reset_index(drop=True)
                all_data[truf_key] = df
                kalman_key = TRUF_TO_KALMAN[truf_key]
                print(
                    f"  [{i+1}/{len(stream_keys)}] {truf_key:25s} → {kalman_key:20s} "
                    f"{len(df):4d} obs  [{df['value'].min():.2f}, {df['value'].max():.2f}]"
                )

        except Exception as e:
            print(f"  [{i+1}/{len(stream_keys)}] {truf_key}: ERROR - {str(e)[:80]}")

    return all_data


def build_daily_timeline(all_data: dict[str, pd.DataFrame]) -> list[dict]:
    """Build a chronological list of (date, observations) for replay.

    For price-level streams, computes log-returns instead of feeding
    raw levels. This makes the observations stationary and bounded.

    For rate/percentage streams, feeds raw values as-is.
    """
    # For price-level streams, compute log-returns
    transformed: dict[str, pd.DataFrame] = {}
    for truf_key, df in all_data.items():
        if truf_key in PRICE_LEVEL_STREAMS:
            # Log-returns: ln(p_t / p_{t-1}) * 100
            df = df.copy()
            df["value"] = np.log(df["value"] / df["value"].shift(1)) * 100
            df = df.dropna()  # first row has no return
            transformed[truf_key] = df
        else:
            transformed[truf_key] = df

    # Collect all (date, truf_key, value) triples
    events = []
    for truf_key, df in transformed.items():
        for _, row in df.iterrows():
            events.append({
                "date": row["date"],
                "truf_key": truf_key,
                "value": row["value"],
            })

    if not events:
        return []

    # Group by date
    by_date: dict = defaultdict(list)
    for ev in events:
        by_date[ev["date"]].append({"truf_key": ev["truf_key"], "value": ev["value"]})

    # Sort by date
    timeline = [
        {"date": d, "observations": obs}
        for d, obs in sorted(by_date.items())
    ]

    return timeline


def run_experiment(all_data: dict[str, pd.DataFrame]) -> dict:
    """Run the Kalman + IMM replay on TRUF data."""
    timeline = build_daily_timeline(all_data)
    if not timeline:
        print("ERROR: No data in timeline")
        return {}

    print(f"\n{'='*70}")
    print(f"TRUF EXPERIMENT REPLAY")
    print(f"{'='*70}")
    print(f"Date range: {timeline[0]['date']} → {timeline[-1]['date']}")
    print(f"Total days with data: {len(timeline)}")
    total_obs = sum(len(d["observations"]) for d in timeline)
    print(f"Total observations: {total_obs}")

    # Initialize
    estimator = EconomicStateEstimator()

    # ── Tier 1 Fix #2: Frequency-weighted observation noise ──
    # Daily streams get R × 16 (√252). Monthly get R × 3.5 (√12).
    # This equalizes annual information content across frequencies.
    FREQ_SCALE = {
        # Daily streams → high R (low per-observation weight)
        "BTC_USD": 16.0,
        "SP500": 16.0,
        "OIL_PRICE": 16.0,
        "COPPER_PRICE": 16.0,
        "COMMODITY_PRESSURE_RAW": 16.0,
        # Weekly streams → moderate R
        "INITIAL_CLAIMS": 7.0,
        "HOUSING_STARTS": 7.0,  # mortgage rates are weekly
        # Monthly streams → low R (high per-observation weight)
        "US_CPI_YOY": 3.5,
        "CORE_CPI": 3.5,
        "PPI": 3.5,
        "NONFARM_PAYROLLS": 3.5,
        "UNEMPLOYMENT_RATE": 3.5,
        "RETAIL_SALES": 3.5,
        "HOME_PRICES": 3.5,
        "CONSUMER_CONFIDENCE": 2.0,  # semi-annual, very high weight
    }
    for stream_key, (H_row, R_base) in estimator.stream_registry.items():
        scale = FREQ_SCALE.get(stream_key, 1.0)
        estimator.stream_registry[stream_key] = (H_row, R_base * scale)

    imm = IMMBranchTracker()
    imm.initialize_branches(BRANCHES, baseline=estimator)
    normalizer = RollingNormalizer(halflife_days=90)
    normalizer.initialize_from_stream_normalization()

    # Tracking
    daily_log = []
    anomaly_count = 0
    total_updates = 0
    kalman_key_counts: dict[str, int] = defaultdict(int)

    # Warmup: first 60 days just update normalizer, don't count anomalies
    warmup_end = timeline[min(59, len(timeline) - 1)]["date"]

    for day_idx, day in enumerate(timeline):
        date = day["date"]
        is_warmup = day_idx < 60

        # Predict step
        estimator.predict()

        day_updates = []
        for obs in day["observations"]:
            truf_key = obs["truf_key"]
            raw_value = obs["value"]
            kalman_key = TRUF_TO_KALMAN.get(truf_key)
            if not kalman_key:
                continue

            # Normalize using rolling normalizer
            z = normalizer.normalize(truf_key, raw_value)

            # Kalman update
            update = estimator.update(kalman_key, z)
            if update:
                total_updates += 1
                kalman_key_counts[kalman_key] += 1
                if not is_warmup and abs(update.innovation_zscore) > ANOMALY_THRESHOLD:
                    anomaly_count += 1
                day_updates.append({
                    "truf_key": truf_key,
                    "kalman_key": kalman_key,
                    "raw": raw_value,
                    "z": round(z, 4),
                    "innovation_z": round(update.innovation_zscore, 4),
                    "anomalous": abs(update.innovation_zscore) > ANOMALY_THRESHOLD,
                })

        # IMM update with current state
        state = estimator.get_state()
        imm.predict()
        for upd in day_updates:
            imm.update(upd["kalman_key"], upd["z"])

        probs = imm.get_probabilities()
        factors = {name: round(float(state.mean[i]), 6) for i, name in enumerate(FACTORS)}

        # Log every 7th day or last day
        if day_idx % 7 == 0 or day_idx == len(timeline) - 1:
            daily_log.append({
                "date": str(date),
                "day": day_idx,
                "observations": len(day["observations"]),
                "factors": factors,
                "probabilities": {b: round(p, 4) for b, p in probs.items()},
                "updates_today": len(day_updates),
            })

        # Progress
        if day_idx % 30 == 0 or day_idx == len(timeline) - 1:
            prob_str = " | ".join(f"{b}: {p:.1%}" for b, p in probs.items())
            print(f"  Day {day_idx:3d} ({date}) | {len(day['observations']):2d} obs | {prob_str}")

    # Final state
    final_state = estimator.get_state()
    final_probs = imm.get_probabilities()
    post_warmup_updates = total_updates - sum(
        len(d["observations"]) for d in timeline[:60]
    )
    anomaly_rate = anomaly_count / max(post_warmup_updates, 1)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total updates: {total_updates}")
    print(f"Anomaly rate (post-warmup): {anomaly_rate:.1%} ({anomaly_count}/{post_warmup_updates})")
    print(f"\nFinal factors:")
    for i, name in enumerate(FACTORS):
        val = float(final_state.mean[i])
        unc = float(np.sqrt(final_state.covariance[i, i]))
        print(f"  {name:25s} = {val:+.4f} ± {unc:.4f}")

    print(f"\nFinal probabilities:")
    for b, p in final_probs.items():
        print(f"  {b:20s}: {p:.1%}")

    print(f"\nUpdates per Kalman key:")
    for k, c in sorted(kalman_key_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {c:4d}")

    # Build results
    results = {
        "experiment": "truf_state_estimator",
        "date_range": f"{timeline[0]['date']} → {timeline[-1]['date']}",
        "total_days": len(timeline),
        "total_observations": total_obs,
        "total_updates": total_updates,
        "anomaly_rate": round(anomaly_rate, 4),
        "anomaly_count": anomaly_count,
        "streams_used": len(all_data),
        "streams_list": list(all_data.keys()),
        "kalman_key_counts": dict(kalman_key_counts),
        "final_factors": {
            name: {
                "value": round(float(final_state.mean[i]), 6),
                "uncertainty": round(float(np.sqrt(final_state.covariance[i, i])), 6),
            }
            for i, name in enumerate(FACTORS)
        },
        "final_probabilities": {b: round(p, 4) for b, p in final_probs.items()},
        "daily_log": daily_log,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return results


def main():
    # Ensure TRUF key is set
    if not os.environ.get("TRUF_PRIVATE_KEY"):
        from dotenv import load_dotenv
        load_dotenv()

    key = os.environ.get("TRUF_PRIVATE_KEY", "")
    if not key:
        print("ERROR: TRUF_PRIVATE_KEY not set")
        sys.exit(1)

    print("=" * 70)
    print("TRUF STATE ESTIMATOR EXPERIMENT")
    print("=" * 70)
    print(f"Gateway: {TRUF_GATEWAY}")
    print(f"History: {DAYS_HISTORY} days")
    print(f"Mapped streams: {len(TRUF_TO_KALMAN)} (minus {len(SKIP_STREAMS)} skipped)")

    # Phase 1: Fetch all TRUF data
    all_data = fetch_all_truf_streams()
    if not all_data:
        print("ERROR: No data fetched from TRUF Network")
        sys.exit(1)

    print(f"\n--- Fetched {len(all_data)} streams successfully ---")

    # Phase 2: Run the replay
    results = run_experiment(all_data)

    # Phase 3: Save results
    output_path = Path(__file__).parent.parent / "data" / "results" / "truf_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
