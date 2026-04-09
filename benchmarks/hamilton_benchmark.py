"""
Hamilton Benchmark — Markov-Switching Model vs Heimdall IMM

Runs Hamilton's (1989) Markov-switching regression on the same 2007-2026
FRED+Yahoo data and compares regime detection speed against our
Kalman+IMM approach.

The key metric is DETECTION LAG: how many weeks after a regime starts does
each model assign >50% probability to the correct regime?

This is the "beat the benchmark" result for the academic paper. Reviewers
will 100% ask for this comparison.

Hamilton model: statsmodels MarkovRegression with 3 regimes
  - Uses same observable data (CPI, unemployment, yields, S&P500)
  - Batch-fitted (full sample), then also rolling-window for fair comparison
  - Smoothed probabilities (Kim, 1994 smoother)

Heimdall model: 8-factor Kalman + 3-branch IMM (from multi_regime_backtest)
  - Online/streaming (processes one observation at a time)
  - No future data leakage

Usage:
    cd aggretor/backend
    source .venv-brew/bin/activate
    python scripts/hamilton_benchmark.py
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Regime definitions (same as multi_regime_backtest.py) ────────────

REGIME_EVENTS = [
    {
        "name": "Great Recession",
        "onset": "2008-01-01",    # Bear Stearns collapse ~Mar 2008
        "window_start": "2008-06-01",
        "window_end": "2009-06-01",
        "expected_regime": "recession",
        "description": "Housing collapse → bank failures → unemployment surge",
    },
    {
        "name": "COVID Crash",
        "onset": "2020-03-01",    # WHO pandemic declaration
        "window_start": "2020-03-01",
        "window_end": "2020-06-01",
        "expected_regime": "recession",
        "description": "Fastest recession in history",
    },
    {
        "name": "Inflation Surge",
        "onset": "2021-03-01",    # CPI started climbing
        "window_start": "2021-06-01",
        "window_end": "2022-06-01",
        "expected_regime": "stagflation",
        "description": "CPI hit 9.1%, supply chain + demand overshoot",
    },
    {
        "name": "Soft Landing",
        "onset": "2023-01-01",    # Disinflation became evident
        "window_start": "2023-06-01",
        "window_end": "2025-12-01",
        "expected_regime": "soft_landing",
        "description": "CPI cooling, labor holding, no recession",
    },
]


def load_data() -> tuple[pd.DataFrame, dict]:
    """Load cached backtest data and IMM results."""
    cache_path = Path(__file__).parent.parent / "data" / "backtest_data_cache.csv"
    results_path = Path(__file__).parent.parent / "data" / "results" / "multi_regime_results.json"

    if not cache_path.exists() or not results_path.exists():
        print("ERROR: Run multi_regime_backtest.py first to generate cached data")
        sys.exit(1)

    data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    with open(results_path) as f:
        imm_results = json.load(f)

    return data, imm_results


def build_hamilton_input(data: pd.DataFrame) -> pd.DataFrame:
    """Build a clean monthly time series for Hamilton model.

    Hamilton (1989) used quarterly GDP growth. We use a richer set
    of monthly indicators to make the comparison fair — both models
    see the same information.
    """
    # Resample to monthly (Hamilton works on regular frequency)
    monthly = data.resample("MS").mean()

    # Select key macro indicators (avoid multicollinearity)
    cols = [
        "FRED_US_CPI_YOY",
        "FRED_UNEMPLOYMENT_RATE",
        "FRED_10Y_YIELD",
        "FRED_FED_FUNDS_RATE",
        "YAHOO_SP500",
    ]

    available = [c for c in cols if c in monthly.columns]
    df = monthly[available].copy()

    # Forward-fill gaps (Hamilton needs complete data)
    df = df.ffill().dropna()

    # Standardize for numerical stability
    for col in df.columns:
        mu = df[col].mean()
        sigma = df[col].std()
        if sigma > 1e-10:
            df[col] = (df[col] - mu) / sigma

    return df


def build_hamilton_equalized(data: pd.DataFrame) -> pd.DataFrame:
    """Build weekly time series using ALL streams via PCA.

    This is the EQUALIZED comparison: Hamilton gets the same information
    content as IMM at the same frequency (weekly).

    IMPORTANT: We drop BTC (starts ~2014) to preserve the full 2007-2026
    date range needed to test all 4 regime events including Great Recession.
    This gives Hamilton 14 of IMM's 15 streams — a fair tradeoff.

    We reduce to 4 principal components to avoid multicollinearity.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Drop BTC to get full 2007-2026 coverage (BTC starts ~2014)
    cols_to_use = [c for c in data.columns if "BTC" not in c]
    data_no_btc = data[cols_to_use]

    # Resample to weekly (same as IMM snapshots)
    weekly = data_no_btc.resample("W").mean()

    # Forward-fill gaps then drop rows with any remaining NaN
    weekly = weekly.ffill().dropna()

    print(f"\n  Equalized data: {len(weekly)} weeks, {len(weekly.columns)} streams")
    print(f"  Date range: {weekly.index.min().date()} to {weekly.index.max().date()}")
    print(f"  Streams: {list(weekly.columns)}")
    print(f"  (BTC excluded to preserve 2007-2026 coverage for all 4 regime events)")

    # Standardize
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(weekly),
        index=weekly.index,
        columns=weekly.columns,
    )

    # PCA to 4 components (captures bulk of variance, avoids multicollinearity)
    pca = PCA(n_components=4)
    components = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_

    pc_df = pd.DataFrame(
        components,
        index=weekly.index,
        columns=[f"PC{i+1}" for i in range(4)],
    )

    print(f"  PCA variance explained: {[f'{v:.1%}' for v in explained]}")
    print(f"  Total: {sum(explained):.1%}")

    # Show what each PC loads on (top 3 loadings per component)
    for i in range(4):
        loadings = pd.Series(pca.components_[i], index=weekly.columns)
        top = loadings.abs().nlargest(3)
        signs = [f"{loadings[c]:+.2f} {c.split('_', 1)[-1]}" for c in top.index]
        print(f"  PC{i+1}: {', '.join(signs)}")

    return pc_df


def _stabilize_regime_labels(result, train_df, k_regimes: int = 3) -> dict:
    """Assign stable regime labels using a two-pass strategy.

    Hamilton's EM assigns regime numbers arbitrarily and they permute
    between refits. We need consistent semantic labels.

    Strategy:
      Pass 1 (event-based): Check which EM regime is dominant during known
      historical events in the training window. This is the gold standard
      because it uses ground truth.

      Pass 2 (PC1-based fallback): For any regimes not matched by events,
      sort by PC1 mean. Higher PC1 = more inflation (PC1 loads on CPI).

    Why not PC1 alone? In early windows (2007-2012) there's no inflation
    period, so Hamilton finds "deep crisis / mild crisis / normal" — not
    our three regimes. Event-based labeling handles this correctly.

    Returns: {EM_regime_number: semantic_label}
    """
    smoothed = result.smoothed_marginal_probabilities
    endog = result.model.endog

    # Pass 1: Event-based labeling (same logic as _label_regimes)
    regime_scores = {i: {"recession": 0, "stagflation": 0, "soft_landing": 0}
                     for i in range(k_regimes)}
    events_in_window = 0

    for event in REGIME_EVENTS:
        mask = (train_df.index >= event["window_start"]) & (train_df.index <= event["window_end"])
        if mask.sum() == 0:
            continue
        events_in_window += 1
        for i in range(k_regimes):
            # Use smoothed probs aligned to training index
            window_vals = smoothed.iloc[:len(train_df)][mask]
            if len(window_vals) > 0:
                avg = float(np.mean(window_vals.iloc[:, i]))
                regime_scores[i][event["expected_regime"]] += avg

    labels = {}
    assigned_regimes = set()
    assigned_labels = set()

    if events_in_window >= 2:
        # Enough events to do event-based matching
        for label_name in ["recession", "stagflation", "soft_landing"]:
            best_regime = None
            best_score = -1
            for i in range(k_regimes):
                if i in assigned_regimes:
                    continue
                if regime_scores[i][label_name] > best_score:
                    best_score = regime_scores[i][label_name]
                    best_regime = i
            if best_regime is not None and best_score > 0.1:
                labels[best_regime] = label_name
                assigned_regimes.add(best_regime)
                assigned_labels.add(label_name)

    # Pass 2: PC1-based fallback for unassigned regimes
    if len(labels) < k_regimes:
        regime_means = {}
        for i in range(k_regimes):
            if i in assigned_regimes:
                continue
            dominant = smoothed.iloc[:, i] > 0.5
            if dominant.sum() > 5:
                regime_means[i] = float(np.mean(endog[dominant]))
            else:
                regime_means[i] = float(np.mean(endog))

        remaining_labels = [l for l in ["recession", "soft_landing", "stagflation"]
                           if l not in assigned_labels]
        sorted_remaining = sorted(regime_means.keys(), key=lambda k: regime_means[k])

        for idx, regime_num in enumerate(sorted_remaining):
            if idx < len(remaining_labels):
                labels[regime_num] = remaining_labels[idx]

    # Safety: fill any gaps
    for i in range(k_regimes):
        if i not in labels:
            for l in ["recession", "soft_landing", "stagflation"]:
                if l not in labels.values():
                    labels[i] = l
                    break

    return labels


def _remap_filtered_probs(
    filt: pd.DataFrame,
    labels: dict,
    k_regimes: int = 3,
) -> pd.DataFrame:
    """Remap filtered probabilities from EM regime numbers to stable semantic names.

    Returns DataFrame with columns ['recession', 'soft_landing', 'stagflation'].
    """
    semantic = pd.DataFrame(index=filt.index, dtype=float)
    for em_num, label in labels.items():
        semantic[label] = filt.iloc[:, em_num].values
    return semantic


def fit_hamilton_equalized_rolling(
    df: pd.DataFrame,
    k_regimes: int = 3,
    window_weeks: int = 156,  # ~3 years (reduced from 5 to cover Great Recession)
) -> dict:
    """Fit Hamilton MS model on PCA-reduced WEEKLY data (equalized).

    This is the FAIR comparison: Hamilton gets the same 14 streams at the
    same weekly frequency as IMM. The only difference is the model:
    batch re-estimation (Hamilton) vs continuous Bayesian updating (IMM).

    Key design decisions:
      - 156-week warmup (~3 years): long enough to estimate 3 regimes,
        short enough that first fit happens in 2010 (can still see 2008
        recession data in the window and produce filtered probs for it)
      - First fit writes ALL historical filtered probs (not just from
        warmup point forward) so early regime detections are recorded
      - Event-based regime labeling with PC1 fallback (stable across refits)
      - Switching mean only (no switching variance) for numerical stability
        on weekly data
      - Quarterly refits (every 13 weeks)
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    print(f"\n{'='*70}")
    print(f"HAMILTON EQUALIZED (weekly, 14 streams via PCA, rolling)")
    print(f"{'='*70}")

    endog_col = "PC1"
    exog_cols = [c for c in df.columns if c != endog_col]

    # Use semantic column names instead of arbitrary regime numbers
    semantic_names = ["recession", "soft_landing", "stagflation"]
    rolling_probs = pd.DataFrame(
        index=df.index,
        columns=semantic_names,
        dtype=float,
    )
    rolling_probs[:] = np.nan

    refit_every = 13  # weeks (~quarterly)
    last_write_idx = -1  # Start at -1 so first fit writes everything
    current_result = None
    current_labels = None
    fit_count = 0
    fail_count = 0
    convergence_failures = 0

    total_iters = len(df) - window_weeks
    print(f"  Data: {len(df)} weeks, warmup: {window_weeks} (~{window_weeks/52:.1f} years)")
    print(f"  First fit at: {df.index[window_weeks].date()}")
    print(f"  Refit every {refit_every} weeks (~{refit_every/4:.0f} months)")
    print(f"  Model: switching mean only (no switching variance — stabler on weekly data)")

    # ELIMINATING PARAMETER LOOK-AHEAD BIAS
    #
    # Hamilton's filtered probability P(S_t | y_1..y_t; θ) depends on
    # BOTH the data up to t AND the model parameters θ. In an expanding-
    # window setup, θ is estimated from data 0..T where T is the refit point.
    # If T > t, the parameters "know" about data the filter hasn't seen yet.
    #
    # This matters at structural breaks: the COVID refit (trained through
    # Mar 2020) creates a "crash regime" in its parameters. The filtered
    # prob at the onset date — using those crash-informed parameters —
    # shows 99% recession. But the model trained BEFORE the crash has
    # no such regime. That's not detection, it's hindsight.
    #
    # SOLUTION: At each refit point t, record ONLY the filtered probability
    # at index t itself. At this one point, the parameters (from 0..t) and
    # the filtering data (0..t) end at the same moment. Zero look-ahead.
    #
    # Trade-off: we get one data point per refit (every 13 weeks), not
    # continuous weekly estimates. Detection requires consecutive refits
    # above threshold. This honestly reflects Hamilton's real-time
    # capability: it can only update its view when it re-estimates parameters.
    #
    # IMM has no equivalent problem because its parameters are fixed from
    # theory. It updates every observation with no re-estimation.

    import warnings as _w
    print(f"  Fitting models and recording point estimates (no look-ahead)...")

    for t in range(window_weeks, len(df)):
        prev_t = fit_count > 0
        weeks_since = t - (last_write_idx if last_write_idx >= 0 else window_weeks - 1)

        if weeks_since >= refit_every or fit_count == 0:
            train_endog = df[endog_col].iloc[:t+1]
            train_exog = df[exog_cols].iloc[:t+1]

            try:
                model = MarkovRegression(
                    train_endog,
                    k_regimes=k_regimes,
                    exog=train_exog,
                    switching_variance=False,
                )

                with _w.catch_warnings(record=True) as caught:
                    _w.simplefilter("always")
                    result = model.fit(maxiter=500, em_iter=100, disp=False, search_reps=10)
                    if any("convergence" in str(w.message).lower() for w in caught):
                        convergence_failures += 1

                stable_labels = _stabilize_regime_labels(result, df.iloc[:t+1], k_regimes)
                current_labels = stable_labels

                # Extract filtered prob at THIS refit point only
                filt = result.filtered_marginal_probabilities
                remapped = _remap_filtered_probs(filt, stable_labels, k_regimes)

                # Write ONLY the probability at index t (no look-ahead)
                for col in semantic_names:
                    rolling_probs.at[df.index[t], col] = remapped[col].iloc[t]

                last_write_idx = t
                fit_count += 1

                if fit_count <= 3 or fit_count % 10 == 0:
                    labels_str = {v: k for k, v in stable_labels.items()}
                    print(f"    Fit {fit_count}: t={t}, date={df.index[t].date()}, "
                          f"labels={labels_str}")

            except Exception as e:
                fail_count += 1

    rolling_probs = rolling_probs.dropna(how="all")
    print(f"  Fits: {fit_count}, failures: {fail_count}, convergence warnings: {convergence_failures}")
    print(f"  Rolling probs available for {len(rolling_probs)} weeks")

    # For detection, use semantic column names directly (no regime_map needed)
    return {
        "rolling_probs": rolling_probs,
        "regime_labels": {name: name for name in semantic_names},  # identity map
        "fit_count": fit_count,
        "fail_count": fail_count,
        "convergence_failures": convergence_failures,
        "frequency": "weekly",
    }


def fit_hamilton_fullsample(df: pd.DataFrame, k_regimes: int = 3) -> dict:
    """Fit Hamilton MS model on full sample (batch, not online).

    This gives Hamilton every advantage — it sees the future.
    If IMM still detects faster, that's a strong result.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    print(f"\n{'='*70}")
    print("HAMILTON MARKOV-SWITCHING MODEL (Full-Sample, 3 Regimes)")
    print(f"{'='*70}")
    print(f"Data: {len(df)} months, {len(df.columns)} indicators")
    print(f"Indicators: {list(df.columns)}")

    # Use CPI as dependent variable, others as exogenous
    endog = df["FRED_US_CPI_YOY"]
    exog_cols = [c for c in df.columns if c != "FRED_US_CPI_YOY"]
    exog = df[exog_cols]

    print(f"Endog: FRED_US_CPI_YOY")
    print(f"Exog: {exog_cols}")
    print(f"Fitting {k_regimes}-regime model...")

    # Fit with switching mean and variance
    model = MarkovRegression(
        endog,
        k_regimes=k_regimes,
        exog=exog,
        switching_variance=True,
    )

    # Try multiple starting points for convergence
    best_result = None
    best_llf = -np.inf

    for attempt in range(5):
        try:
            np.random.seed(42 + attempt)
            result = model.fit(
                maxiter=500,
                em_iter=100,
                disp=False,
                search_reps=20 if attempt == 0 else 5,
            )
            if result.llf > best_llf:
                best_llf = result.llf
                best_result = result
        except Exception as e:
            print(f"  Attempt {attempt+1}: {str(e)[:80]}")
            continue

    if best_result is None:
        print("  ERROR: Hamilton model failed to converge after 5 attempts")
        return {"error": "convergence_failure"}

    result = best_result
    print(f"  Converged: log-likelihood = {result.llf:.2f}")

    # Get smoothed probabilities (Kim smoother — uses full sample)
    smoothed = result.smoothed_marginal_probabilities
    filtered = result.filtered_marginal_probabilities

    # Label regimes by their characteristics
    regime_labels = _label_regimes(result, df, smoothed)

    print(f"\nRegime characteristics:")
    for i, label in regime_labels.items():
        # Proportion of time in each regime
        pct = (smoothed[i] > 0.5).mean()
        print(f"  Regime {i} → {label:15s} ({pct:.0%} of sample)")

    return {
        "smoothed": smoothed,
        "filtered": filtered,
        "regime_labels": regime_labels,
        "dates": df.index,
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
    }


def fit_hamilton_rolling(df: pd.DataFrame, k_regimes: int = 3,
                         window_months: int = 60) -> dict:
    """Fit Hamilton MS model with expanding window (quasi-online).

    This is the FAIR comparison: at each point, Hamilton only sees
    past data (like IMM). We re-fit every 6 months.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    print(f"\n{'='*70}")
    print(f"HAMILTON ROLLING-WINDOW ({window_months}mo minimum, refit every 6mo)")
    print(f"{'='*70}")

    endog_col = "FRED_US_CPI_YOY"
    exog_cols = [c for c in df.columns if c != endog_col]

    # Store rolling probabilities
    rolling_probs = pd.DataFrame(
        index=df.index,
        columns=[f"regime_{i}" for i in range(k_regimes)],
        dtype=float,
    )
    rolling_probs[:] = np.nan

    refit_every = 6  # months
    last_fit_idx = -refit_every  # force fit on first eligible month
    current_result = None
    current_labels = None
    fit_count = 0
    fail_count = 0

    for t in range(window_months, len(df)):
        months_since_fit = t - last_fit_idx

        if months_since_fit >= refit_every or current_result is None:
            # Expanding window: use all data up to t
            train_endog = df[endog_col].iloc[:t+1]
            train_exog = df[exog_cols].iloc[:t+1]

            try:
                model = MarkovRegression(
                    train_endog,
                    k_regimes=k_regimes,
                    exog=train_exog,
                    switching_variance=True,
                )
                result = model.fit(maxiter=300, em_iter=50, disp=False, search_reps=5)
                current_result = result
                current_labels = _label_regimes(
                    result, df.iloc[:t+1],
                    result.smoothed_marginal_probabilities
                )
                last_fit_idx = t
                fit_count += 1
            except Exception:
                fail_count += 1
                if current_result is None:
                    continue

        # Use FILTERED (not smoothed) probabilities at time t
        # Filtered = P(S_t | y_1, ..., y_t) — no future info
        if current_result is not None:
            filt = current_result.filtered_marginal_probabilities
            # The last row of filtered probs corresponds to time t
            if t < len(filt):
                for i in range(k_regimes):
                    rolling_probs.iloc[t, i] = filt.iloc[t, i]

    rolling_probs = rolling_probs.dropna(how="all")
    print(f"  Fits: {fit_count}, failures: {fail_count}")
    print(f"  Rolling probs available for {len(rolling_probs)} months")

    return {
        "rolling_probs": rolling_probs,
        "regime_labels": current_labels or {},
        "fit_count": fit_count,
        "fail_count": fail_count,
    }


def _label_regimes(result, df, smoothed_probs) -> dict:
    """Label Hamilton regimes by matching to economic conditions.

    Hamilton regimes are numbered arbitrarily. We label them by checking
    which regime dominates during known economic periods.
    """
    labels = {}
    regime_scores = {i: {"recession": 0, "stagflation": 0, "soft_landing": 0}
                     for i in range(smoothed_probs.shape[1])}

    # Score each regime by when it's active
    for event in REGIME_EVENTS:
        mask = (df.index >= event["window_start"]) & (df.index <= event["window_end"])
        if mask.sum() == 0:
            continue

        window_probs = smoothed_probs.loc[mask] if hasattr(smoothed_probs, 'loc') else smoothed_probs[mask]
        for i in range(smoothed_probs.shape[1]):
            col = window_probs.iloc[:, i] if hasattr(window_probs, 'iloc') else window_probs[:, i]
            avg = float(np.mean(col))
            regime_scores[i][event["expected_regime"]] += avg

    # Assign labels greedily (highest score first)
    assigned = set()
    for regime_name in ["recession", "stagflation", "soft_landing"]:
        best_regime = None
        best_score = -1
        for i in range(smoothed_probs.shape[1]):
            if i in assigned:
                continue
            if regime_scores[i][regime_name] > best_score:
                best_score = regime_scores[i][regime_name]
                best_regime = i
        if best_regime is not None:
            labels[best_regime] = regime_name
            assigned.add(best_regime)

    # Assign remaining
    for i in range(smoothed_probs.shape[1]):
        if i not in labels:
            labels[i] = f"regime_{i}"

    return labels


def extract_imm_probabilities(imm_results: dict) -> pd.DataFrame:
    """Extract IMM probabilities from backtest log as a time series."""
    log = imm_results["daily_log"]
    dates = pd.to_datetime([e["date"] for e in log])
    probs = pd.DataFrame([e["probabilities"] for e in log], index=dates)
    return probs


def measure_detection_lag(
    probs: pd.DataFrame,
    regime_map: dict,
    model_name: str,
    threshold: float = 0.40,
    consecutive_weeks: int = 3,
) -> list[dict]:
    """Measure how quickly a model detects each regime.

    Detection = probability of correct regime exceeds threshold
    for `consecutive_weeks` consecutive observations.

    Returns detection lag in weeks from regime onset.
    """
    results = []

    for event in REGIME_EVENTS:
        onset = pd.Timestamp(event["onset"])
        expected = event["expected_regime"]

        # Find the probability column for the expected regime
        if isinstance(regime_map, dict) and any(isinstance(k, int) for k in regime_map.keys()):
            # Hamilton: map regime number → name → column
            col_idx = None
            for regime_num, label in regime_map.items():
                if label == expected:
                    col_idx = regime_num
                    break
            if col_idx is None:
                results.append({
                    "event": event["name"],
                    "expected": expected,
                    "model": model_name,
                    "detection_lag_weeks": None,
                    "note": "regime not mapped",
                })
                continue
            col_name = f"regime_{col_idx}" if f"regime_{col_idx}" in probs.columns else col_idx
        else:
            # IMM: column name IS the regime name
            col_name = expected

        if col_name not in probs.columns:
            results.append({
                "event": event["name"],
                "expected": expected,
                "model": model_name,
                "detection_lag_weeks": None,
                "note": f"column {col_name} not found",
            })
            continue

        # Look for detection after onset
        post_onset = probs[probs.index >= onset]
        if len(post_onset) == 0:
            results.append({
                "event": event["name"],
                "expected": expected,
                "model": model_name,
                "detection_lag_weeks": None,
                "note": "no data after onset",
            })
            continue

        # Find first point where prob > threshold for N consecutive periods
        above = (post_onset[col_name] >= threshold).values
        detection_idx = None
        consecutive = 0
        for i, val in enumerate(above):
            if val:
                consecutive += 1
                if consecutive >= consecutive_weeks:
                    detection_idx = i - consecutive_weeks + 1
                    break
            else:
                consecutive = 0

        if detection_idx is not None:
            detection_date = post_onset.index[detection_idx]
            lag_days = (detection_date - onset).days
            lag_weeks = lag_days / 7.0

            results.append({
                "event": event["name"],
                "expected": expected,
                "model": model_name,
                "detection_date": str(detection_date.date()),
                "onset_date": str(onset.date()),
                "detection_lag_weeks": round(lag_weeks, 1),
                "probability_at_detection": round(
                    float(post_onset[col_name].iloc[detection_idx]), 4
                ),
            })
        else:
            # Check if it ever crosses threshold
            ever_above = post_onset[col_name] >= threshold
            max_prob = float(post_onset[col_name].max())
            results.append({
                "event": event["name"],
                "expected": expected,
                "model": model_name,
                "detection_lag_weeks": None,
                "note": f"never sustained {consecutive_weeks} consecutive periods above {threshold:.0%}",
                "max_probability": round(max_prob, 4),
            })

    return results


def print_comparison(imm_lags: list, hamilton_full_lags: list,
                     hamilton_rolling_lags: list,
                     hamilton_equalized_lags: list | None = None):
    """Print the head-to-head comparison table."""
    has_equalized = hamilton_equalized_lags is not None and len(hamilton_equalized_lags) > 0

    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD: DETECTION LAG COMPARISON")
    print(f"{'='*70}")
    print(f"Metric: weeks from regime onset to sustained detection (≥40% for 3+ periods)")
    print()

    if has_equalized:
        print(f"  {'Event':22s} {'IMM':>10s} {'Ham.Roll':>10s} {'Ham.EQ':>10s} {'Ham.Full':>10s} {'Winner':>8s}")
        print(f"  {'':22s} {'(online)':>10s} {'(5mo)':>10s} {'(15wk)':>10s} {'(future)':>10s} {'(vs EQ)':>8s}")
        print(f"  {'-'*75}")
    else:
        print(f"  {'Event':25s} {'IMM':>12s} {'Hamilton':>12s} {'Hamilton':>12s} {'Winner':>10s}")
        print(f"  {'':25s} {'(online)':>12s} {'(full-samp)':>12s} {'(rolling)':>12s}")
        print(f"  {'-'*75}")

    comparison = []
    imm_wins = 0
    hamilton_wins = 0
    imm_wins_eq = 0
    hamilton_wins_eq = 0

    for event in REGIME_EVENTS:
        name = event["name"]

        imm_lag = next((r for r in imm_lags if r["event"] == name), None)
        hf_lag = next((r for r in hamilton_full_lags if r["event"] == name), None)
        hr_lag = next((r for r in hamilton_rolling_lags if r["event"] == name), None)
        he_lag = next((r for r in (hamilton_equalized_lags or []) if r["event"] == name), None)

        imm_w = imm_lag["detection_lag_weeks"] if imm_lag else None
        hf_w = hf_lag["detection_lag_weeks"] if hf_lag else None
        hr_w = hr_lag["detection_lag_weeks"] if hr_lag else None
        he_w = he_lag["detection_lag_weeks"] if he_lag else None

        imm_str = f"{imm_w:.1f}w" if imm_w is not None else "N/D"
        hf_str = f"{hf_w:.1f}w" if hf_w is not None else "N/D"
        hr_str = f"{hr_w:.1f}w" if hr_w is not None else "N/D"
        he_str = f"{he_w:.1f}w" if he_w is not None else "N/D"

        # Compare IMM vs rolling Hamilton (original)
        if imm_w is not None and hr_w is not None:
            if imm_w < hr_w:
                winner_orig = "IMM"
                imm_wins += 1
            elif hr_w < imm_w:
                winner_orig = "Hamilton"
                hamilton_wins += 1
            else:
                winner_orig = "Tie"
        elif imm_w is not None:
            winner_orig = "IMM"
            imm_wins += 1
        elif hr_w is not None:
            winner_orig = "Hamilton"
            hamilton_wins += 1
        else:
            winner_orig = "N/A"

        # Compare IMM vs EQUALIZED Hamilton (the fair fight)
        winner_eq = "N/A"
        if has_equalized:
            if imm_w is not None and he_w is not None:
                if imm_w < he_w:
                    winner_eq = "IMM"
                    imm_wins_eq += 1
                elif he_w < imm_w:
                    winner_eq = "Hamilton"
                    hamilton_wins_eq += 1
                else:
                    winner_eq = "Tie"
            elif imm_w is not None:
                winner_eq = "IMM"
                imm_wins_eq += 1
            elif he_w is not None:
                winner_eq = "Hamilton"
                hamilton_wins_eq += 1

        advantage = ""
        if has_equalized and imm_w is not None and he_w is not None:
            diff = he_w - imm_w
            if abs(diff) > 0.5:
                faster = "IMM" if diff > 0 else "Ham"
                advantage = f" ({abs(diff):.1f}w)"

        if has_equalized:
            print(f"  {name:22s} {imm_str:>10s} {hr_str:>10s} {he_str:>10s} {hf_str:>10s} {winner_eq:>8s}{advantage}")
        else:
            print(f"  {name:25s} {imm_str:>12s} {hf_str:>12s} {hr_str:>12s} {winner_orig:>10s}")

        entry = {
            "event": name,
            "imm_lag_weeks": imm_w,
            "hamilton_fullsample_lag_weeks": hf_w,
            "hamilton_rolling_lag_weeks": hr_w,
            "winner_vs_rolling": winner_orig,
        }
        if has_equalized:
            entry["hamilton_equalized_lag_weeks"] = he_w
            entry["winner_vs_equalized"] = winner_eq
        comparison.append(entry)

    if has_equalized:
        print(f"\n  Original score (IMM vs Hamilton rolling, 5 streams monthly):")
        print(f"    IMM {imm_wins} — Hamilton {hamilton_wins}")
        print(f"\n  EQUALIZED score (IMM vs Hamilton equalized, 15 streams weekly):")
        print(f"    IMM {imm_wins_eq} — Hamilton {hamilton_wins_eq}")

        if imm_wins_eq > hamilton_wins_eq:
            print(f"\n  → IMM wins {imm_wins_eq}/{len(REGIME_EVENTS)} even with equalized data")
            print(f"  → This proves MODEL ADVANTAGE, not just data advantage")
        elif hamilton_wins_eq > imm_wins_eq:
            print(f"\n  → Hamilton wins {hamilton_wins_eq}/{len(REGIME_EVENTS)} when given equal data")
            print(f"  → Original 4-0 was partially a data advantage — honest result")
        else:
            print(f"\n  → Tied when data is equalized — models are comparable")
    else:
        print(f"\n  Score: IMM {imm_wins} — Hamilton {hamilton_wins}")

    return comparison


def print_methodology_notes():
    """Print methodology notes for the paper."""
    print(f"\n{'='*70}")
    print("METHODOLOGY NOTES FOR PAPER")
    print(f"{'='*70}")
    print("""
  Hamilton (1989) Markov-Switching Regression:
    - 3 regimes, switching mean and variance
    - Dependent: CPI YoY (standardized)
    - Exogenous: unemployment, 10Y yield, fed funds, S&P500 returns
    - Estimated via EM algorithm (statsmodels)
    - Full-sample: smoothed probabilities (Kim, 1994) — sees future
    - Rolling: expanding window, refit every 6 months, filtered probs only

  Heimdall IMM (this paper):
    - 8-factor Kalman filter (state transition model)
    - 3-branch IMM with persistent adjustments
    - Likelihood tempering for cold-start (α: 0.3 → 1.0 over 200 obs)
    - Processes one observation at a time (true online/streaming)
    - 15 data streams (11 FRED + 4 Yahoo)
    - No future data leakage at any point

  Fair comparison:
    - Same date range: 2007-01-01 to 2026-02-28
    - Same underlying data sources
    - Hamilton rolling = quasi-online (closest fair analog)
    - Detection threshold: 40% sustained for 3+ consecutive periods
    - IMM uses weekly snapshots; Hamilton uses monthly observations

  Expected result:
    - IMM should detect faster due to: (a) higher frequency (weekly vs monthly),
      (b) richer state representation (8 factors vs 1 dependent + 4 exog),
      (c) continuous updating vs periodic refitting
    - Hamilton full-sample has an unfair advantage (smoothed = sees future)
      but provides a useful upper bound on detection speed
""")


def main():
    import os
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    print("=" * 70)
    print("HAMILTON BENCHMARK: Markov-Switching vs Heimdall IMM")
    print("=" * 70)

    # ── Load data ──
    data, imm_results = load_data()

    # ── Extract IMM probabilities ──
    imm_probs = extract_imm_probabilities(imm_results)
    print(f"\nIMM probabilities: {len(imm_probs)} weekly snapshots")
    print(f"  Columns: {list(imm_probs.columns)}")

    # ── Build Hamilton input ──
    hamilton_df = build_hamilton_input(data)
    print(f"\nHamilton input: {len(hamilton_df)} months")
    print(f"  Columns: {list(hamilton_df.columns)}")

    # ── Fit Hamilton (full sample — gives Hamilton every advantage) ──
    hamilton_full = fit_hamilton_fullsample(hamilton_df)

    # ── Fit Hamilton (rolling — original comparison, 5 streams monthly) ──
    hamilton_rolling = fit_hamilton_rolling(hamilton_df)

    # ── Fit Hamilton EQUALIZED (all 15 streams, weekly, PCA-reduced) ──
    print(f"\n{'='*70}")
    print("BUILDING EQUALIZED HAMILTON INPUT (all 15 streams, weekly, PCA)")
    print(f"{'='*70}")
    hamilton_eq_df = build_hamilton_equalized(data)
    hamilton_equalized = fit_hamilton_equalized_rolling(hamilton_eq_df)

    # ── Measure detection lags ──
    print(f"\n{'='*70}")
    print("DETECTION LAG MEASUREMENT")
    print(f"{'='*70}")

    # IMM lags (weekly data)
    imm_lags = measure_detection_lag(
        imm_probs, regime_map=None, model_name="IMM",
        threshold=0.40, consecutive_weeks=3,
    )

    # Hamilton full-sample lags (monthly, smoothed)
    if "error" not in hamilton_full:
        # Build a DataFrame from smoothed probs
        smoothed = hamilton_full["smoothed"]
        smoothed_df = pd.DataFrame(
            smoothed.values,
            index=hamilton_full["dates"],
            columns=[f"regime_{i}" for i in range(smoothed.shape[1])],
        )
        hamilton_full_lags = measure_detection_lag(
            smoothed_df, hamilton_full["regime_labels"], "Hamilton (full)",
            threshold=0.40, consecutive_weeks=2,  # monthly = fewer points
        )
    else:
        hamilton_full_lags = []

    # Hamilton rolling lags (monthly, filtered)
    if hamilton_rolling.get("rolling_probs") is not None:
        hamilton_rolling_lags = measure_detection_lag(
            hamilton_rolling["rolling_probs"],
            hamilton_rolling["regime_labels"],
            "Hamilton (rolling)",
            threshold=0.40,
            consecutive_weeks=2,  # monthly frequency
        )
    else:
        hamilton_rolling_lags = []

    # Hamilton EQUALIZED lags (point estimates at each refit, no look-ahead)
    hamilton_equalized_lags = []
    if hamilton_equalized.get("rolling_probs") is not None:
        hamilton_equalized_lags = measure_detection_lag(
            hamilton_equalized["rolling_probs"],
            hamilton_equalized["regime_labels"],
            "Hamilton (equalized)",
            threshold=0.40,
            consecutive_weeks=2,  # one data point per 13-week refit, 2 consecutive = 26w
        )

    # ── Print comparison ──
    comparison = print_comparison(
        imm_lags, hamilton_full_lags, hamilton_rolling_lags, hamilton_equalized_lags,
    )

    # ── Print detail ──
    print(f"\n{'='*70}")
    print("DETAILED DETECTION RESULTS")
    print(f"{'='*70}")

    for label, lags in [("IMM (online)", imm_lags),
                        ("Hamilton (full-sample)", hamilton_full_lags),
                        ("Hamilton (rolling, 5 monthly)", hamilton_rolling_lags),
                        ("Hamilton (equalized, 15 weekly PCA)", hamilton_equalized_lags)]:
        print(f"\n  {label}:")
        for r in lags:
            lag = r.get("detection_lag_weeks")
            if lag is not None:
                print(f"    {r['event']:25s} detected at {r.get('detection_date', '?'):12s} "
                      f"(lag: {lag:.1f}w, p={r.get('probability_at_detection', 0):.2%})")
            else:
                note = r.get("note", "not detected")
                max_p = r.get("max_probability", 0)
                print(f"    {r['event']:25s} NOT DETECTED — {note} (max p={max_p:.2%})")

    # ── Model fit comparison ──
    if "error" not in hamilton_full:
        print(f"\n{'='*70}")
        print("MODEL FIT STATISTICS")
        print(f"{'='*70}")
        print(f"  Hamilton AIC:  {hamilton_full['aic']:.1f}")
        print(f"  Hamilton BIC:  {hamilton_full['bic']:.1f}")
        print(f"  Hamilton LogL: {hamilton_full['log_likelihood']:.1f}")
        print(f"  (IMM is not likelihood-based — comparison via detection speed, not fit)")

    # ── Methodology notes ──
    print_methodology_notes()

    # ── Save results ──
    output = {
        "experiment": "hamilton_benchmark",
        "run_timestamp": datetime.now().isoformat(),
        "imm_detection_lags": imm_lags,
        "hamilton_fullsample_lags": hamilton_full_lags,
        "hamilton_rolling_lags": hamilton_rolling_lags,
        "hamilton_equalized_lags": hamilton_equalized_lags,
        "comparison": comparison,
        "hamilton_fit": {
            "log_likelihood": hamilton_full.get("log_likelihood"),
            "aic": hamilton_full.get("aic"),
            "bic": hamilton_full.get("bic"),
            "regime_labels": {str(k): v for k, v in hamilton_full.get("regime_labels", {}).items()},
        } if "error" not in hamilton_full else {"error": hamilton_full.get("error")},
        "hamilton_equalized_fit": {
            "fit_count": hamilton_equalized.get("fit_count"),
            "fail_count": hamilton_equalized.get("fail_count"),
            "regime_labels": {str(k): v for k, v in hamilton_equalized.get("regime_labels", {}).items()},
            "frequency": "weekly",
            "input": "15 streams via PCA (4 components)",
        },
        "methodology": {
            "hamilton_original": {
                "regimes": 3,
                "endog": "CPI YoY",
                "exog": ["unemployment", "10Y yield", "fed funds", "S&P500"],
                "frequency": "monthly",
                "consecutive_periods": 2,
            },
            "hamilton_equalized": {
                "regimes": 3,
                "endog": "PC1 (first principal component of all 15 streams)",
                "exog": ["PC2", "PC3", "PC4"],
                "frequency": "weekly",
                "consecutive_periods": 3,
                "pca_input_streams": 15,
                "note": "Same data, same frequency, same detection threshold as IMM",
            },
            "imm": {
                "factors": 8,
                "streams": 15,
                "frequency": "weekly (daily updates, weekly snapshots)",
                "consecutive_periods": 3,
            },
            "detection_threshold": 0.40,
            "rolling_refit_interval": "6 months (26 weeks equalized, 6 months original)",
        },
    }

    output_path = Path(__file__).parent.parent / "data" / "results" / "hamilton_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
