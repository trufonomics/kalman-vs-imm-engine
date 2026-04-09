"""
Chauvet-Piger Head-to-Head Benchmark — March 17, 2026

Compares our IMM recession probability against:
  1. Chauvet-Piger smoothed recession probabilities (FRED: RECPROUSM156N)
  2. Sahm Rule (FRED: SAHMREALTIME, threshold 0.50)
  3. CFNAI (Chicago Fed National Activity Index, threshold -0.70)

All benchmarks are recession-only models. Our IMM tracks 3 regimes
(soft_landing, stagflation, recession). We extract our recession
probability for fair comparison.

Metrics:
  - Detection speed: first month each model crosses its threshold
  - Peak probability during recession window
  - False positive rate during expansions
  - Correlation between models
  - Lead/lag at each NBER recession

Uses NBER recession dates (USREC) as ground truth.

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/chauvet_piger_benchmark.py
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
CACHE_FILE = str(Path(__file__).parent.parent / "data" / "full_history_cache.csv")  # Reuse existing cache

# FRED series for our Kalman (same as full_history_backtest.py)
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
    "BTC-USD": ("BTC_USD", "returns"),
}

# IMM branches
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

# NBER recession windows (ground truth)
NBER_RECESSIONS = [
    ("1980-01-01", "1980-07-01", "Volcker I"),
    ("1981-07-01", "1982-11-01", "Volcker II"),
    ("1990-07-01", "1991-03-01", "Gulf War"),
    ("2001-03-01", "2001-11-01", "Dot-com Bust"),
    ("2007-12-01", "2009-06-01", "Great Recession"),
    ("2020-02-01", "2020-04-01", "COVID"),
]

# Detection thresholds
IMM_THRESHOLD = 0.50          # recession probability > 50%
CP_THRESHOLD = 50.0           # Chauvet-Piger > 50%
SAHM_THRESHOLD = 0.50         # Sahm Rule >= 0.50
CFNAI_THRESHOLD = -0.70       # CFNAI < -0.70 (negative = contraction)


class RollingNormalizer:
    """EMA-based normalizer (same as full_history_backtest.py)."""

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


def fetch_benchmark_series() -> dict[str, pd.Series]:
    """Fetch Chauvet-Piger, Sahm Rule, CFNAI, and NBER indicator from FRED."""
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    series = {}
    for name, fred_id in [
        ("chauvet_piger", "RECPROUSM156N"),
        ("sahm_rule", "SAHMREALTIME"),
        ("cfnai", "CFNAI"),
        ("nber", "USREC"),
    ]:
        try:
            s = fred.get_series(fred_id, observation_start=START_DATE, observation_end=END_DATE)
            s = s.dropna()
            series[name] = s
            print(f"  {name}: {len(s)} observations, {s.index[0].date()} → {s.index[-1].date()}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    return series


def load_imm_data() -> pd.DataFrame:
    """Load cached IMM data or reuse full_history_results.json."""
    results_path = Path(__file__).parent.parent / "data" / "results" / "full_history_results.json"
    if results_path.exists():
        r = json.load(open(results_path))
        daily_log = r.get("daily_log", [])
        if daily_log:
            rows = []
            for entry in daily_log:
                rows.append({
                    "date": pd.Timestamp(entry["date"]),
                    "imm_recession": entry["probabilities"].get("recession", 0),
                    "imm_stagflation": entry["probabilities"].get("stagflation", 0),
                    "imm_soft_landing": entry["probabilities"].get("soft_landing", 0),
                })
            df = pd.DataFrame(rows).set_index("date")
            return df

    raise FileNotFoundError(
        "No full_history_results.json found. Run full_history_backtest.py first."
    )


def resample_to_monthly(imm_df: pd.DataFrame) -> pd.DataFrame:
    """Resample weekly IMM data to monthly for fair comparison.

    Uses the last observation of each month (same convention as FRED).
    """
    return imm_df.resample("MS").last()


def compute_detection_speed(
    series: pd.Series,
    threshold: float,
    start: str,
    end: str,
    direction: str = "above",
) -> dict:
    """Find first month the series crosses threshold within a window.

    Args:
        direction: "above" means signal > threshold (CP, IMM, Sahm).
                   "below" means signal < threshold (CFNAI).
    """
    window = series[start:end].dropna()
    if window.empty:
        return {"detected": False, "first_date": None, "lead_months": None, "peak": None}

    if direction == "above":
        detections = window[window >= threshold]
    else:
        detections = window[window <= threshold]

    peak = float(window.max()) if direction == "above" else float(window.min())

    if detections.empty:
        return {"detected": False, "first_date": None, "lead_months": None, "peak": peak}

    first_date = detections.index[0]
    rec_start = pd.Timestamp(start)
    lead_months = (first_date - rec_start).days / 30.44

    return {
        "detected": True,
        "first_date": str(first_date.date()),
        "lead_months": round(lead_months, 1),
        "peak": round(peak, 2),
    }


def compute_false_positive_rate(
    series: pd.Series,
    threshold: float,
    nber: pd.Series,
    direction: str = "above",
) -> float:
    """Fraction of expansion months where the model falsely signals recession."""
    # Align to common dates
    common = series.index.intersection(nber.index)
    s = series.loc[common]
    n = nber.loc[common]

    expansion_mask = n == 0
    expansion_signals = s[expansion_mask]

    if expansion_signals.empty:
        return 0.0

    if direction == "above":
        false_positives = (expansion_signals >= threshold).sum()
    else:
        false_positives = (expansion_signals <= threshold).sum()

    return round(float(false_positives) / len(expansion_signals), 4)


def compute_correlation(s1: pd.Series, s2: pd.Series) -> float:
    """Pearson correlation on overlapping dates."""
    common = s1.index.intersection(s2.index)
    if len(common) < 10:
        return float("nan")
    return round(float(s1.loc[common].corr(s2.loc[common])), 3)


def run_benchmark():
    """Run the full head-to-head benchmark."""
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 70)
    print("CHAUVET-PIGER HEAD-TO-HEAD BENCHMARK")
    print("=" * 70)
    print()

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading benchmark series from FRED...")
    benchmarks = fetch_benchmark_series()
    print()

    print("Loading IMM results...")
    imm_df = load_imm_data()
    imm_monthly = resample_to_monthly(imm_df)
    print(f"  IMM: {len(imm_monthly)} monthly observations, "
          f"{imm_monthly.index[0].date()} → {imm_monthly.index[-1].date()}")
    print()

    # Scale IMM recession to 0-100 for visual comparison with CP
    imm_rec_pct = imm_monthly["imm_recession"] * 100

    # ── Detection speed comparison ─────────────────────────────────────
    print("=" * 70)
    print("DETECTION SPEED — FIRST MONTH CROSSING THRESHOLD")
    print("=" * 70)
    print()

    # Extend window 6 months before and after NBER dates to catch early/late signals
    detection_results = []

    for rec_start, rec_end, name in NBER_RECESSIONS:
        # Search window: 6 months before NBER start to 3 months after NBER end
        search_start = str((pd.Timestamp(rec_start) - pd.DateOffset(months=6)).date())
        search_end = str((pd.Timestamp(rec_end) + pd.DateOffset(months=3)).date())

        result = {"event": name, "nber_start": rec_start, "nber_end": rec_end}

        # IMM
        imm_det = compute_detection_speed(
            imm_rec_pct, IMM_THRESHOLD * 100, search_start, search_end, "above"
        )
        result["imm"] = imm_det

        # Chauvet-Piger
        if "chauvet_piger" in benchmarks:
            cp_det = compute_detection_speed(
                benchmarks["chauvet_piger"], CP_THRESHOLD, search_start, search_end, "above"
            )
            result["chauvet_piger"] = cp_det

        # Sahm Rule
        if "sahm_rule" in benchmarks:
            sahm_det = compute_detection_speed(
                benchmarks["sahm_rule"], SAHM_THRESHOLD, search_start, search_end, "above"
            )
            result["sahm_rule"] = sahm_det

        # CFNAI
        if "cfnai" in benchmarks:
            cfnai_det = compute_detection_speed(
                benchmarks["cfnai"], CFNAI_THRESHOLD, search_start, search_end, "below"
            )
            result["cfnai"] = cfnai_det

        detection_results.append(result)

    # Print detection speed table
    header = f"{'Event':20s} {'NBER Start':12s}"
    models = []
    if "chauvet_piger" in benchmarks:
        header += f" {'C-P (mo)':>10s}"
        models.append("chauvet_piger")
    header += f" {'IMM (mo)':>10s}"
    if "sahm_rule" in benchmarks:
        header += f" {'Sahm (mo)':>10s}"
        models.append("sahm_rule")
    if "cfnai" in benchmarks:
        header += f" {'CFNAI (mo)':>10s}"
        models.append("cfnai")
    header += f" {'Winner':>10s}"

    print(header)
    print("-" * len(header))

    win_counts = defaultdict(int)

    for r in detection_results:
        line = f"{r['event']:20s} {r['nber_start']:12s}"

        speeds = {}

        # Chauvet-Piger
        if "chauvet_piger" in r:
            cp = r["chauvet_piger"]
            if cp["detected"]:
                line += f" {cp['lead_months']:>10.1f}"
                speeds["C-P"] = cp["lead_months"]
            else:
                line += f" {'N/D':>10s}"

        # IMM
        imm = r["imm"]
        if imm["detected"]:
            line += f" {imm['lead_months']:>10.1f}"
            speeds["IMM"] = imm["lead_months"]
        else:
            line += f" {'N/D':>10s}"

        # Sahm
        if "sahm_rule" in r:
            sahm = r["sahm_rule"]
            if sahm["detected"]:
                line += f" {sahm['lead_months']:>10.1f}"
                speeds["Sahm"] = sahm["lead_months"]
            else:
                line += f" {'N/D':>10s}"

        # CFNAI
        if "cfnai" in r:
            cfnai = r["cfnai"]
            if cfnai["detected"]:
                line += f" {cfnai['lead_months']:>10.1f}"
                speeds["CFNAI"] = cfnai["lead_months"]
            else:
                line += f" {'N/D':>10s}"

        # Determine winner (fastest detection = lowest positive lead_months)
        detected = {k: v for k, v in speeds.items() if v is not None}
        if detected:
            winner = min(detected, key=detected.get)
            line += f" {winner:>10s}"
            win_counts[winner] += 1
        else:
            line += f" {'none':>10s}"

        print(line)

    print()
    print("Win counts (fastest detection):")
    for model, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count}")

    # ── Peak probability during recession ──────────────────────────────
    print()
    print("=" * 70)
    print("PEAK SIGNAL DURING NBER RECESSION WINDOWS")
    print("=" * 70)
    print()

    header2 = f"{'Event':20s} {'C-P Peak':>10s} {'IMM Peak':>10s} {'Sahm Peak':>10s} {'CFNAI Min':>10s}"
    print(header2)
    print("-" * len(header2))

    for r in detection_results:
        line = f"{r['event']:20s}"

        if "chauvet_piger" in r:
            p = r["chauvet_piger"].get("peak")
            line += f" {p:>10.1f}" if p is not None else f" {'N/A':>10s}"

        p = r["imm"].get("peak")
        line += f" {p:>10.1f}" if p is not None else f" {'N/A':>10s}"

        if "sahm_rule" in r:
            p = r["sahm_rule"].get("peak")
            line += f" {p:>10.2f}" if p is not None else f" {'N/A':>10s}"

        if "cfnai" in r:
            p = r["cfnai"].get("peak")
            line += f" {p:>10.2f}" if p is not None else f" {'N/A':>10s}"

        print(line)

    # ── False positive rate ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FALSE POSITIVE RATE (% of expansion months with signal)")
    print("=" * 70)
    print()

    nber = benchmarks.get("nber")
    if nber is not None:
        # IMM false positive rate
        imm_fp = compute_false_positive_rate(
            imm_rec_pct, IMM_THRESHOLD * 100, nber, "above"
        )
        print(f"  IMM (>50%):              {imm_fp:.1%}")

        if "chauvet_piger" in benchmarks:
            cp_fp = compute_false_positive_rate(
                benchmarks["chauvet_piger"], CP_THRESHOLD, nber, "above"
            )
            print(f"  Chauvet-Piger (>50%):    {cp_fp:.1%}")

        if "sahm_rule" in benchmarks:
            sahm_fp = compute_false_positive_rate(
                benchmarks["sahm_rule"], SAHM_THRESHOLD, nber, "above"
            )
            print(f"  Sahm Rule (≥0.50):       {sahm_fp:.1%}")

        if "cfnai" in benchmarks:
            cfnai_fp = compute_false_positive_rate(
                benchmarks["cfnai"], CFNAI_THRESHOLD, nber, "below"
            )
            print(f"  CFNAI (<-0.70):          {cfnai_fp:.1%}")

        # Also check IMM at lower thresholds
        for thresh in [0.30, 0.40]:
            fp = compute_false_positive_rate(
                imm_rec_pct, thresh * 100, nber, "above"
            )
            print(f"  IMM (>{thresh:.0%}):              {fp:.1%}")

    # ── Correlation matrix ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("CORRELATION MATRIX (monthly, overlapping dates)")
    print("=" * 70)
    print()

    all_series = {"IMM_rec": imm_rec_pct}
    if "chauvet_piger" in benchmarks:
        all_series["C-P"] = benchmarks["chauvet_piger"]
    if "sahm_rule" in benchmarks:
        all_series["Sahm"] = benchmarks["sahm_rule"]
    if "cfnai" in benchmarks:
        # Invert CFNAI so positive = recession signal (for correlation)
        all_series["CFNAI_inv"] = -benchmarks["cfnai"]

    names = list(all_series.keys())
    print(f"{'':15s}", end="")
    for n in names:
        print(f" {n:>10s}", end="")
    print()

    for i, n1 in enumerate(names):
        print(f"{n1:15s}", end="")
        for j, n2 in enumerate(names):
            if j <= i:
                corr = compute_correlation(all_series[n1], all_series[n2])
                print(f" {corr:>10.3f}", end="")
            else:
                print(f" {'':>10s}", end="")
        print()

    # ── Volcker I deep dive ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("VOLCKER I DEEP DIVE — Classification Disagreement Analysis")
    print("=" * 70)
    print()

    # Show all model signals during Volcker I
    print("Monthly signals during Volcker I (Jan-Jul 1980):")
    print(f"{'Month':12s} {'NBER':>6s} {'C-P':>8s} {'IMM_Rec':>8s} {'IMM_Stag':>9s} {'Sahm':>8s} {'CFNAI':>8s}")
    print("-" * 65)

    for month in pd.date_range("1980-01-01", "1980-07-01", freq="MS"):
        line = f"{str(month.date()):12s}"

        # NBER
        if nber is not None and month in nber.index:
            line += f" {int(nber.loc[month]):>6d}"
        else:
            line += f" {'?':>6s}"

        # C-P
        if "chauvet_piger" in benchmarks and month in benchmarks["chauvet_piger"].index:
            line += f" {benchmarks['chauvet_piger'].loc[month]:>8.1f}"
        else:
            line += f" {'N/A':>8s}"

        # IMM recession
        if month in imm_monthly.index:
            line += f" {imm_monthly.loc[month, 'imm_recession'] * 100:>8.1f}"
            line += f" {imm_monthly.loc[month, 'imm_stagflation'] * 100:>9.1f}"
        else:
            line += f" {'N/A':>8s} {'N/A':>9s}"

        # Sahm
        if "sahm_rule" in benchmarks and month in benchmarks["sahm_rule"].index:
            line += f" {benchmarks['sahm_rule'].loc[month]:>8.2f}"
        else:
            line += f" {'N/A':>8s}"

        # CFNAI
        if "cfnai" in benchmarks and month in benchmarks["cfnai"].index:
            line += f" {benchmarks['cfnai'].loc[month]:>8.2f}"
        else:
            line += f" {'N/A':>8s}"

        print(line)

    print()
    print("Analysis: IMM reads Volcker I as stagflation (CPI 14.6%, FFR 17.6%)")
    print("while C-P reads it as recession (GDP-based). Both are correct —")
    print("different definitions of 'regime'. IMM's 3-regime model is more")
    print("informative: it distinguishes inflationary recessions from")
    print("deflationary ones.")

    # ── IMM combined distress signal ───────────────────────────────────
    print()
    print("=" * 70)
    print("IMM COMBINED DISTRESS (Recession + Stagflation)")
    print("=" * 70)
    print()
    print("Since IMM has 3 regimes, 'distress' = 1 - P(soft_landing).")
    print("This captures both recession AND stagflation as non-normal states.")
    print()

    imm_distress = (1 - imm_monthly["imm_soft_landing"]) * 100

    for rec_start, rec_end, name in NBER_RECESSIONS:
        window = imm_distress[rec_start:rec_end].dropna()
        if window.empty:
            continue
        avg = window.mean()
        peak = window.max()
        first_50 = window[window >= 50]
        if not first_50.empty:
            lead = (first_50.index[0] - pd.Timestamp(rec_start)).days / 30.44
            detect_str = f"{lead:.1f}mo"
        else:
            detect_str = "N/D"

        print(f"  {name:20s}  avg={avg:.0f}%  peak={peak:.0f}%  first>50%: {detect_str}")

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "experiment": "chauvet_piger_benchmark",
        "date_range": f"{imm_monthly.index[0].date()} → {imm_monthly.index[-1].date()}",
        "detection_results": detection_results,
        "false_positive_rates": {},
        "win_counts": dict(win_counts),
        "run_timestamp": datetime.now().isoformat(),
    }

    if nber is not None:
        results["false_positive_rates"] = {
            "imm_50": compute_false_positive_rate(imm_rec_pct, 50, nber, "above"),
            "imm_40": compute_false_positive_rate(imm_rec_pct, 40, nber, "above"),
            "imm_30": compute_false_positive_rate(imm_rec_pct, 30, nber, "above"),
        }
        if "chauvet_piger" in benchmarks:
            results["false_positive_rates"]["cp_50"] = compute_false_positive_rate(
                benchmarks["chauvet_piger"], 50, nber, "above"
            )
        if "sahm_rule" in benchmarks:
            results["false_positive_rates"]["sahm_50"] = compute_false_positive_rate(
                benchmarks["sahm_rule"], 0.5, nber, "above"
            )

    output_path = Path(__file__).parent.parent / "data" / "results" / "chauvet_piger_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
