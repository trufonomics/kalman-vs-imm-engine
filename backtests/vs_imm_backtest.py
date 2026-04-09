"""
VS-IMM Hierarchical Backtest — March 19, 2026

Replays the HierarchicalIMMTracker through 1975-2026 using FRED + Yahoo data.
Fetches data directly from FRED API when cache doesn't cover the full range.

Compares flat IMM (3 branches) vs VS-IMM (3 Level A + 7 Level B).
Measures: checkpoint accuracy, false positive rate, sub-regime detection.

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/vs_imm_backtest.py
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
os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    FACTORS,
    ANOMALY_THRESHOLD,
    STREAM_LOADINGS,
)
from heimdall.imm_tracker import (
    IMMBranchTracker,
    HierarchicalIMMTracker,
    ParallelHierarchicalTracker,
    EnsembleHierarchicalTracker,
    ShadowStateTracker,
    RECOMMENDED_BRANCH_ADJUSTMENTS,
)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")  # Set FRED_API_KEY env var to fetch data
START_DATE = "1975-01-01"
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

YAHOO_STREAMS = {
    "^GSPC": ("SP500", "returns"),
    "CL=F": ("OIL_PRICE", "returns"),
    "GC=F": ("GOLD_PRICE", "returns"),
}


# ── Checkpoints ──────────────────────────────────────────────────────
# Level A checkpoints: the full 51-year set
LEVEL_A_CHECKPOINTS = [
    # 1970s-80s
    ("1975-01-01", "1975-06-01", "contraction", 0.25,
     "1973-75 recession — oil embargo, OPEC quadrupling prices, deep downturn"),
    ("1977-01-01", "1978-12-31", "stagflation", 0.25,
     "Late 70s stagflation — rising inflation, sluggish growth, Volcker not yet chair"),
    ("1980-01-01", "1980-07-01", "contraction", 0.25,
     "1980 recession — Volcker rate shock, credit controls, brief sharp downturn"),
    ("1981-07-01", "1982-11-01", "contraction", 0.30,
     "1981-82 recession — Volcker double-dip, unemployment hit 10.8%"),
    ("1983-01-01", "1985-12-31", "expansion", 0.25,
     "Reagan expansion — strong recovery, falling inflation, Morning in America"),
    # 1990s
    ("1990-07-01", "1991-03-01", "contraction", 0.25,
     "1990-91 recession — S&L crisis, oil price spike, Gulf War uncertainty"),
    ("1995-01-01", "1999-12-31", "expansion", 0.25,
     "Late 90s expansion — tech boom, Goldilocks economy, low inflation + growth"),
    # 2000s-2020s (existing)
    ("2001-03-01", "2001-11-01", "contraction", 0.25,
     "Dot-com bust — NASDAQ crash, 9/11, mild recession"),
    ("2008-06-01", "2009-06-01", "contraction", 0.35,
     "Great Recession — housing collapse, bank failures, unemployment surge"),
    ("2020-03-01", "2020-06-01", "contraction", 0.30,
     "COVID crash — fastest contraction in history"),
    ("2021-06-01", "2022-06-01", "stagflation", 0.30,
     "Inflation surge — CPI hit 9.1%, supply chain + demand overshoot"),
    ("2023-06-01", "2025-12-01", "expansion", 0.35,
     "Post-inflation expansion — CPI cooling, labor holding, no contraction"),
]

# Level B sub-regime checkpoints: finer-grained validation
LEVEL_B_CHECKPOINTS = [
    # Format: (start, end, parent, expected_sub, min_conditional, description)
    # Threshold = beat uniform: 34% for 3-way (expansion), 51% for 2-way (stag/contraction)

    # 1970s-80s sub-regimes
    ("1977-01-01", "1978-12-31", "stagflation", "stagflation_cost_push", 0.51,
     "Late 70s cost-push — OPEC-driven inflation, weak growth"),
    ("1983-01-01", "1985-12-31", "expansion", "expansion_disinflation", 0.34,
     "Volcker disinflation — inflation falling fast, strong recovery"),
    ("1995-01-01", "1999-12-31", "expansion", "expansion_boom", 0.34,
     "Dot-com boom — tech euphoria, tight labor, surging markets"),

    # Within expansion periods
    ("2013-01-01", "2015-12-31", "expansion", "expansion_goldilocks", 0.34,
     "Mid-cycle Goldilocks — steady growth, low vol, no drama"),
    ("2017-01-01", "2018-09-30", "expansion", "expansion_goldilocks", 0.34,
     "Pre-tightening Goldilocks — steady before Fed hiking"),
    ("2005-01-01", "2006-12-31", "expansion", "expansion_boom", 0.34,
     "Housing boom — pre-GFC overheating"),
    ("2023-06-01", "2024-05-31", "expansion", "expansion_disinflation", 0.34,
     "Disinflation cycle — Fed tightening working, inflation dropping"),

    # Within contraction periods
    ("2008-09-01", "2009-06-01", "contraction", "contraction_credit_crunch", 0.51,
     "Credit crunch — Lehman, bank failures, credit freeze"),
    ("2020-03-01", "2020-05-01", "contraction", "contraction_demand_shock", 0.51,
     "Demand shock — COVID lockdowns, instant demand collapse"),

    # Within stagflation periods
    ("2022-01-01", "2022-06-30", "stagflation", "stagflation_cost_push", 0.51,
     "Cost-push stagflation — Ukraine war, oil/food spike"),
    ("2021-10-01", "2022-01-31", "stagflation", "stagflation_demand_pull", 0.51,
     "Demand-pull stagflation — GDP strong, CPI surging from demand"),
]

# False positive windows: periods where contraction should NOT dominate
FP_WINDOWS = [
    ("1983-01-01", "1989-12-31", "Reagan/Bush expansion"),
    ("1992-01-01", "2000-12-31", "90s expansion"),
    ("2010-01-01", "2019-12-31", "Post-GFC expansion"),
    ("2023-01-01", "2025-12-01", "Post-inflation soft landing"),
    ("2013-01-01", "2015-12-31", "Mid-cycle calm"),
    ("2017-01-01", "2018-09-30", "Pre-tightening expansion"),
]
FP_THRESHOLD = 0.35  # If contraction > 35% during these windows, it's a false positive


class RollingNormalizer:
    """EMA-based normalizer — identical to multi_regime_backtest.py."""

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


def load_data() -> pd.DataFrame:
    """Load or fetch backtest data covering 1975-2026."""
    cache_path = Path(__file__).parent.parent / "data" / "backtest_data_cache_51yr.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if df.index[0].year <= 1976:
            print(f"Loaded cached 51-year data: {len(df)} days, {df.notna().sum().sum()} observations")
            return df

    print("Fetching 51-year data from FRED + Yahoo...")
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
                print(f"  {series_id:15s} → {kalman_key:20s} {len(s):5d} obs ({s.index[0].date()} → {s.index[-1].date()})")
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
                print(f"  {ticker:15s} → {kalman_key:20s} {len(s):5d} obs ({s.index[0].date()} → {s.index[-1].date()})")
        except Exception as e:
            print(f"  {ticker:15s} → ERROR: {str(e)[:60]}")

    combined = pd.DataFrame(all_series)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()
    combined = combined.dropna(how="all")

    combined.to_csv(cache_path)
    print(f"\nCached to {cache_path}: {len(combined)} days")
    print(f"Loaded {len(combined)} days, {combined.notna().sum().sum()} observations")
    return combined


def compute_warmup_norms(data: pd.DataFrame, warmup_days: int = 90) -> dict:
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


def run_hierarchical_backtest(data: pd.DataFrame, smoothing: bool = True, label: str = "") -> dict:
    """Run VS-IMM through 1975-2026 data."""
    print(f"\n{'='*70}")
    print(f"VS-IMM HIERARCHICAL BACKTEST{f' [{label}]' if label else ''}")
    print(f"{'='*70}")
    print(f"Date range: {data.index[0].date()} → {data.index[-1].date()}")

    # Map column names to kalman keys
    col_to_kalman = {}
    for col in data.columns:
        parts = col.split("_", 1)
        if len(parts) == 2:
            col_to_kalman[col] = parts[1]

    # Initialize
    estimator = EconomicStateEstimator()
    hier_imm = ShadowStateTracker(smoothing=smoothing)
    hier_imm.initialize(estimator)

    # Also run flat IMM for comparison
    flat_branches = [
        {"branch_id": "soft_landing", "name": "Soft Landing", "probability": 1/3,
         "state_adjustments": RECOMMENDED_BRANCH_ADJUSTMENTS["soft_landing"]},
        {"branch_id": "stagflation", "name": "Stagflation", "probability": 1/3,
         "state_adjustments": RECOMMENDED_BRANCH_ADJUSTMENTS["stagflation"]},
        {"branch_id": "recession", "name": "Recession", "probability": 1/3,
         "state_adjustments": RECOMMENDED_BRANCH_ADJUSTMENTS["recession"]},
    ]
    flat_estimator = EconomicStateEstimator()
    flat_imm = IMMBranchTracker()
    flat_imm.initialize_branches(flat_branches, flat_estimator)

    # Warmup
    norms = compute_warmup_norms(data)
    hier_normalizer = RollingNormalizer(halflife_days=90)
    flat_normalizer = RollingNormalizer(halflife_days=90)
    for col, n in norms.items():
        hier_normalizer.initialize(col, n["mean"], n["std"])
        flat_normalizer.initialize(col, n["mean"], n["std"])

    # Tracking
    daily_log = []
    total_updates = 0
    warmup_end_idx = 90

    for day_idx, (date, row) in enumerate(data.iterrows()):
        # Predict
        estimator.predict()
        hier_imm.predict()
        flat_estimator.predict()
        flat_imm.predict()

        for col in data.columns:
            val = row[col]
            if pd.isna(val):
                continue

            kalman_key = col_to_kalman.get(col)
            if not kalman_key or kalman_key not in STREAM_LOADINGS:
                continue

            # Hierarchical
            z_hier = hier_normalizer.normalize(col, val)
            estimator.update(kalman_key, z_hier)
            hier_imm.update(kalman_key, z_hier)

            # Flat (separate estimator to keep independent)
            z_flat = flat_normalizer.normalize(col, val)
            flat_estimator.update(kalman_key, z_flat)
            flat_imm.update(kalman_key, z_flat)

            total_updates += 1

        # Log weekly
        if day_idx % 7 == 0 or day_idx == len(data) - 1:
            hier_probs = hier_imm.get_probabilities()
            joint_probs = hier_imm.get_joint_probabilities()
            flat_probs = flat_imm.get_probabilities()

            state = estimator.get_state()
            factors = {name: round(float(state.mean[i]), 4) for i, name in enumerate(FACTORS)}

            daily_log.append({
                "date": str(date.date()),
                "day": day_idx,
                "factors": factors,
                "hier_level_a": {b: round(p, 4) for b, p in hier_probs.items()},
                "hier_joint": {b: round(p, 4) for b, p in joint_probs.items()},
                "flat": {b: round(p, 4) for b, p in flat_probs.items()},
            })

        # Monthly progress
        if day_idx % 60 == 0:
            hp = hier_imm.get_probabilities()
            fp = flat_imm.get_probabilities()
            jp = hier_imm.get_joint_probabilities()
            sub_str = ""
            for regime_id in ["expansion", "stagflation", "contraction"]:
                sub_probs = hier_imm.get_sub_probabilities(regime_id)
                if sub_probs:
                    top_sub = max(sub_probs, key=sub_probs.get)
                    short = top_sub.split("_", 1)[-1] if "_" in top_sub else top_sub
                    sub_str += f" [{short}={sub_probs[top_sub]:.0%}]"

            print(f"  {date.date()} | "
                  f"VS-IMM: E={hp.get('expansion',0):.0%} S={hp.get('stagflation',0):.0%} C={hp.get('contraction',0):.0%}{sub_str} | "
                  f"Flat: SL={fp.get('soft_landing',0):.0%} SF={fp.get('stagflation',0):.0%} RC={fp.get('recession',0):.0%}")

    return {
        "total_updates": total_updates,
        "daily_log": daily_log,
    }


def evaluate_checkpoints(results: dict) -> dict:
    """Evaluate Level A and Level B checkpoints."""
    daily_log = results["daily_log"]

    print(f"\n{'='*70}")
    print("LEVEL A CHECKPOINT EVALUATION")
    print(f"{'='*70}")

    level_a_results = []
    for start, end, expected, min_prob, description in LEVEL_A_CHECKPOINTS:
        window = [e for e in daily_log if start <= e["date"] <= end]
        if not window:
            print(f"\n  SKIP: {description}")
            level_a_results.append({"result": "SKIP", "description": description})
            continue

        # VS-IMM Level A
        hier_probs = [e["hier_level_a"].get(expected, 0) for e in window]
        hier_avg = np.mean(hier_probs)
        hier_max = np.max(hier_probs)

        # Flat IMM (map regime names)
        flat_map = {"expansion": "soft_landing", "contraction": "recession", "stagflation": "stagflation"}
        flat_key = flat_map.get(expected, expected)
        flat_probs = [e["flat"].get(flat_key, 0) for e in window]
        flat_avg = np.mean(flat_probs)

        hier_pass = hier_avg >= min_prob
        flat_pass = flat_avg >= min_prob

        status = "PASS" if hier_pass else "FAIL"
        comp = "BETTER" if hier_avg > flat_avg else ("SAME" if abs(hier_avg - flat_avg) < 0.02 else "WORSE")

        print(f"\n  {status}: {description}")
        print(f"    VS-IMM: avg={hier_avg:.1%}, max={hier_max:.1%}")
        print(f"    Flat:   avg={flat_avg:.1%}")
        print(f"    Comparison: {comp} ({hier_avg - flat_avg:+.1%}pp)")

        level_a_results.append({
            "description": description,
            "expected": expected,
            "result": status,
            "hier_avg": round(hier_avg, 4),
            "hier_max": round(hier_max, 4),
            "flat_avg": round(flat_avg, 4),
            "comparison": comp,
        })

    passed = sum(1 for r in level_a_results if r.get("result") == "PASS")
    total = sum(1 for r in level_a_results if r.get("result") != "SKIP")
    print(f"\n  Level A: {passed}/{total} checkpoints passed")

    # Level B sub-regime checkpoints
    print(f"\n{'='*70}")
    print("LEVEL B SUB-REGIME CHECKPOINT EVALUATION")
    print(f"{'='*70}")

    level_b_results = []
    for start, end, parent, expected_sub, min_prob, description in LEVEL_B_CHECKPOINTS:
        window = [e for e in daily_log if start <= e["date"] <= end]
        if not window:
            print(f"\n  SKIP: {description}")
            level_b_results.append({"result": "SKIP", "description": description})
            continue

        # Joint probability of the sub-regime
        sub_probs = [e["hier_joint"].get(expected_sub, 0) for e in window]
        avg_prob = np.mean(sub_probs)
        max_prob = np.max(sub_probs)

        # Parent Level A probability
        parent_probs = [e["hier_level_a"].get(parent, 0) for e in window]
        parent_avg = np.mean(parent_probs)

        # Conditional probability (sub given parent) — only when parent is meaningful
        cond_probs = [
            s / p for s, p in zip(sub_probs, parent_probs) if p > 0.05
        ]
        cond_avg = np.mean(cond_probs) if cond_probs else 0.0

        passed = cond_avg >= min_prob
        status = "PASS" if passed else "FAIL"

        print(f"\n  {status}: {description}")
        print(f"    Joint P({expected_sub}): avg={avg_prob:.1%}, max={max_prob:.1%}")
        print(f"    Parent P({parent}): avg={parent_avg:.1%}")
        print(f"    Conditional P({expected_sub}|{parent}): avg={cond_avg:.1%}")

        level_b_results.append({
            "description": description,
            "expected_sub": expected_sub,
            "parent": parent,
            "result": status,
            "joint_avg": round(avg_prob, 4),
            "joint_max": round(max_prob, 4),
            "parent_avg": round(parent_avg, 4),
            "conditional_avg": round(cond_avg, 4),
        })

    passed_b = sum(1 for r in level_b_results if r.get("result") == "PASS")
    total_b = sum(1 for r in level_b_results if r.get("result") != "SKIP")
    print(f"\n  Level B: {passed_b}/{total_b} sub-regime checkpoints passed")

    return {
        "level_a": level_a_results,
        "level_b": level_b_results,
        "level_a_pass_rate": f"{passed}/{total}",
        "level_b_pass_rate": f"{passed_b}/{total_b}",
    }


def compute_false_positives(results: dict) -> dict:
    """Measure FP rate: how often contraction > 35% during expansion windows."""
    daily_log = results["daily_log"]

    print(f"\n{'='*70}")
    print("FALSE POSITIVE ANALYSIS")
    print(f"{'='*70}")

    hier_fp_weeks = 0
    flat_fp_weeks = 0
    total_weeks = 0

    for window_start, window_end, description in FP_WINDOWS:
        window = [e for e in daily_log if window_start <= e["date"] <= window_end]
        if not window:
            continue

        hier_fps = sum(1 for e in window if e["hier_level_a"].get("contraction", 0) > FP_THRESHOLD)
        flat_fps = sum(1 for e in window if e["flat"].get("recession", 0) > FP_THRESHOLD)

        hier_fp_weeks += hier_fps
        flat_fp_weeks += flat_fps
        total_weeks += len(window)

        hier_rate = hier_fps / len(window)
        flat_rate = flat_fps / len(window)
        improvement = flat_rate - hier_rate

        print(f"\n  {description} ({window_start} → {window_end}, {len(window)} weeks)")
        print(f"    VS-IMM FP: {hier_fps}/{len(window)} = {hier_rate:.1%}")
        print(f"    Flat FP:   {flat_fps}/{len(window)} = {flat_rate:.1%}")
        if improvement > 0:
            print(f"    Improvement: {improvement:.1%}pp reduction")
        elif improvement < 0:
            print(f"    Regression: {-improvement:.1%}pp increase")

    hier_total_rate = hier_fp_weeks / max(total_weeks, 1)
    flat_total_rate = flat_fp_weeks / max(total_weeks, 1)

    print(f"\n{'='*70}")
    print(f"OVERALL FALSE POSITIVE RATE (contraction > {FP_THRESHOLD:.0%} during expansion windows)")
    print(f"  VS-IMM: {hier_fp_weeks}/{total_weeks} = {hier_total_rate:.1%}")
    print(f"  Flat:   {flat_fp_weeks}/{total_weeks} = {flat_total_rate:.1%}")
    improvement = flat_total_rate - hier_total_rate
    if improvement > 0:
        print(f"  Improvement: {improvement:.1%}pp ({improvement/max(flat_total_rate, 0.001):.0%} reduction)")
    print(f"{'='*70}")

    return {
        "hier_fp_rate": round(hier_total_rate, 4),
        "flat_fp_rate": round(flat_total_rate, 4),
        "hier_fp_weeks": hier_fp_weeks,
        "flat_fp_weeks": flat_fp_weeks,
        "total_weeks": total_weeks,
        "improvement_pp": round(improvement, 4),
    }


def run_variant(data, smoothing, label, tpm_diag=None):
    """Run one backtest variant and return summary dict."""
    results = run_hierarchical_backtest(data, smoothing=smoothing, label=label)
    checkpoint_results = evaluate_checkpoints(results)
    fp_results = compute_false_positives(results)
    return {
        "label": label,
        "smoothing": smoothing,
        "tpm_diagonal": tpm_diag,
        "total_updates": results["total_updates"],
        "checkpoints": checkpoint_results,
        "false_positives": fp_results,
        "probability_timeline": [
            {
                "date": e["date"],
                "hier_level_a": e["hier_level_a"],
                "hier_joint": e["hier_joint"],
                "flat": e["flat"],
                "active_subs": e.get("active_subs", []),
            }
            for i, e in enumerate(results["daily_log"])
            if i % 4 == 0 or i == len(results["daily_log"]) - 1
        ],
    }


def main():
    print("=" * 70)
    print("VS-IMM HIERARCHICAL BACKTEST — SMOOTHING COMPARISON")
    print("3 variants: raw scorecards, mild smoothing, stronger smoothing")
    print("=" * 70)

    data = load_data()

    # Monkey-patch ShadowStateTracker to accept tpm_diag override in backtest
    # Variant 1: Raw (no smoothing)
    variant_raw = run_variant(data, smoothing=False, label="RAW (no smoothing)")

    # Variant 2: Mild smoothing (default TPMs: ~0.78-0.88 diagonal)
    variant_mild = run_variant(data, smoothing=True, label="MILD smoothing (default TPMs)")

    # Variant 3: Stronger smoothing — temporarily override TPMs
    # Save originals
    orig_tpms = {}
    for regime_id, cfg in ShadowStateTracker.FAMILY_CONFIG.items():
        orig_tpms[regime_id] = cfg["tpm"].copy()

    # Override class-level TPMs for stronger smoothing
    for regime_id, cfg in ShadowStateTracker.FAMILY_CONFIG.items():
        n = cfg["tpm"].shape[0]
        off_diag = 0.08 / max(1, n - 1)
        new_tpm = np.full((n, n), off_diag)
        np.fill_diagonal(new_tpm, 0.92)
        cfg["tpm"] = new_tpm

    variant_strong = run_variant(data, smoothing=True, label="STRONG smoothing (0.92 diag)", tpm_diag=0.92)

    # Restore originals
    for regime_id, cfg in ShadowStateTracker.FAMILY_CONFIG.items():
        cfg["tpm"] = orig_tpms[regime_id]

    # ── Comparison summary ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SMOOTHING COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    for v in [variant_raw, variant_mild, variant_strong]:
        cp = v["checkpoints"]
        la_results = cp.get("level_a", [])
        lb_results = cp.get("level_b", [])
        la_pass = sum(1 for r in la_results if r.get("result") == "PASS")
        la_total = sum(1 for r in la_results if r.get("result") != "SKIP")
        lb_pass = sum(1 for r in lb_results if r.get("result") == "PASS")
        lb_total = sum(1 for r in lb_results if r.get("result") != "SKIP")
        fp = v["false_positives"]
        fp_rate = fp.get("hier_fp_rate", 0) if fp else 0
        print(f"  {v['label']:45s}  Level A: {la_pass}/{la_total}  Level B: {lb_pass}/{lb_total}  FP: {fp_rate:.1%}")

    print()

    # Save all variants
    output = {
        "experiment": "vs_imm_smoothing_comparison",
        "date_range": f"{data.index[0].date()} → {data.index[-1].date()}",
        "variants": {
            "raw": variant_raw,
            "mild": variant_mild,
            "strong": variant_strong,
        },
        "run_timestamp": datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "results" / "vs_imm_backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
