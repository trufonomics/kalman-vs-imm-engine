"""
Block Bootstrap Confidence Intervals for Regime Diagnostics.

Point estimates without confidence intervals are unpublishable.
This module provides stationary block bootstrap (Politis & Romano 1994)
CIs for all key metrics: AUC, Brier score, detection count, and
log-likelihood.

Block bootstrap is required (not i.i.d. bootstrap) because the daily_log
has strong serial correlation — regime probabilities are a filtered
Markov process.

Block length selection uses Politis & White (2004) with the Patton,
Politis & White (2009) correction when the `arch` package is available.
Falls back to a heuristic (n^(1/3)) otherwise.

Refs:
  Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap."
      JASA 89(428), 1303-1313.
  Politis, D.N. & White, H. (2004). "Automatic Block-Length Selection
      for the Dependent Bootstrap." Econometric Reviews 23(1), 53-70.
  Patton, A., Politis, D.N. & White, H. (2009). "Correction to
      'Automatic Block-Length Selection'." Econometric Reviews 28(4).
  DiCiccio, T.J. & Efron, B. (1996). "Bootstrap Confidence Intervals."
      Statistical Science 11(3), 189-212.
"""

import numpy as np
from dataclasses import dataclass
from heimdall.regime_diagnostics import (
    brier_decomposition,
    roc_auc,
    detection_lag,
    get_ground_truth,
)


def select_block_length(
    daily_log: list[dict],
    regime: str = "contraction",
) -> float:
    """Select optimal block length using Politis-White-Patton.

    Uses the arch.bootstrap.optimal_block_length implementation
    (Politis & White 2004 with Patton et al. 2009 correction).
    Falls back to n^(1/3) heuristic if arch is unavailable.

    Args:
        daily_log: weekly regime probability snapshots
        regime: which regime's probability series to use for
                autocorrelation estimation
    """
    prob_key = {
        "contraction": "recession",
        "stagflation": "stagflation",
        "expansion": "soft_landing",
    }.get(regime, regime)

    probs = np.array([
        entry["probabilities"].get(prob_key, 0.0) for entry in daily_log
    ])

    try:
        from arch.bootstrap import optimal_block_length
        result = optimal_block_length(probs)
        # Use the stationary bootstrap column
        block_len = float(result.loc["stationary", "stationary"])
        return max(block_len, 2.0)
    except (ImportError, Exception):
        # Heuristic: n^(1/3) — standard for weakly dependent data
        return max(float(len(probs) ** (1.0 / 3.0)), 2.0)


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a single metric."""
    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int
    confidence_level: float

    def __str__(self) -> str:
        pct = int(self.confidence_level * 100)
        return (
            f"{self.metric}: {self.point_estimate:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"({pct}% CI, SE={self.std_error:.4f})"
        )


def stationary_block_bootstrap_indices(
    n: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one bootstrap resample using Politis & Romano (1994).

    Each block starts at a random position. Block lengths are geometric
    with expectation mean_block_length. Wraps around at end.

    Args:
        n: length of original series
        mean_block_length: expected block length (geometric distribution)
        rng: numpy random generator

    Returns:
        Array of indices (length n) for the bootstrap resample.
    """
    p = 1.0 / mean_block_length  # geometric parameter
    indices = []
    while len(indices) < n:
        # Random start position
        start = rng.integers(0, n)
        # Geometric block length
        block_len = rng.geometric(p)
        for j in range(block_len):
            if len(indices) >= n:
                break
            indices.append((start + j) % n)
    return np.array(indices[:n])


PROB_KEY_MAP = {
    "expansion": "soft_landing",
    "contraction": "recession",
    "stagflation": "stagflation",
}


def _extract_forecasts_observations(
    daily_log: list[dict],
    regime: str,
) -> tuple[list[float], list[int]]:
    """Extract forecast probabilities and ground truth from daily_log."""
    prob_key = PROB_KEY_MAP.get(regime, regime)
    forecasts = []
    observations = []
    for entry in daily_log:
        p = entry["probabilities"].get(prob_key, 0.0)
        gt = get_ground_truth(entry["date"])
        forecasts.append(p)
        observations.append(1 if gt == regime else 0)
    return forecasts, observations


def compute_metric_on_resample(
    daily_log: list[dict],
    indices: np.ndarray,
    regime_checkpoints: list[tuple],
    metric: str,
    regime: str = "contraction",
) -> float | None:
    """Compute a single metric on a bootstrap resample of daily_log.

    For AUC/Brier, resamples the weekly snapshots and extracts
    forecasts/observations for the diagnostic functions.
    For detection, uses the resampled log directly (block structure
    preserves temporal ordering within blocks).

    Returns None if metric cannot be computed on this resample.
    """
    resampled = [daily_log[i] for i in indices]

    if metric in ("auc", "brier", "reliability", "resolution"):
        forecasts, observations = _extract_forecasts_observations(resampled, regime)
        if not forecasts or sum(observations) == 0:
            return None

    if metric == "auc":
        result = roc_auc(forecasts, observations, regime)
        return result.auc if result.auc > 0 else None

    elif metric == "brier":
        result = brier_decomposition(forecasts, observations, regime)
        return result.brier_score

    elif metric == "reliability":
        result = brier_decomposition(forecasts, observations, regime)
        return result.reliability

    elif metric == "resolution":
        result = brier_decomposition(forecasts, observations, regime)
        return result.resolution

    elif metric == "detection_count":
        # Detection uses full log with dates — checkpoint matching
        checkpoint_regime_to_label = {
            "soft_landing": "expansion",
            "recession": "contraction",
            "stagflation": "stagflation",
        }
        count = 0
        for start, end, ckpt_regime, _min_prob, desc in regime_checkpoints:
            prob_key = PROB_KEY_MAP.get(
                checkpoint_regime_to_label.get(ckpt_regime, ckpt_regime),
                ckpt_regime,
            )
            result = detection_lag(resampled, start, end, prob_key, desc)
            if result.detection_lag_weeks is not None:
                count += 1
        return float(count)

    elif metric == "mean_detection_lag":
        checkpoint_regime_to_label = {
            "soft_landing": "expansion",
            "recession": "contraction",
            "stagflation": "stagflation",
        }
        lags = []
        for start, end, ckpt_regime, _min_prob, desc in regime_checkpoints:
            prob_key = PROB_KEY_MAP.get(
                checkpoint_regime_to_label.get(ckpt_regime, ckpt_regime),
                ckpt_regime,
            )
            result = detection_lag(resampled, start, end, prob_key, desc)
            if result.detection_lag_weeks is not None:
                lags.append(result.detection_lag_weeks)
        return float(np.mean(lags)) if lags else None

    return None


def bootstrap_confidence_interval(
    daily_log: list[dict],
    regime_checkpoints: list[tuple],
    metric: str,
    regime: str = "contraction",
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    mean_block_length: float | None = None,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap CI for a regime diagnostic metric.

    Uses Politis & Romano (1994) stationary block bootstrap with
    percentile intervals (DiCiccio & Efron 1996).

    Block length is selected automatically via Politis & White (2004)
    with Patton et al. (2009) correction when mean_block_length is None.

    Args:
        daily_log: weekly regime probability snapshots
        regime_checkpoints: NBER-dated event windows
        metric: one of "auc", "brier", "reliability", "resolution",
                "detection_count", "mean_detection_lag"
        regime: regime to evaluate (for auc/brier)
        n_bootstrap: number of bootstrap resamples
        confidence_level: CI coverage (default 0.95)
        mean_block_length: expected block length (None = auto-select)
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    n = len(daily_log)

    # Auto-select block length if not specified
    if mean_block_length is None:
        mean_block_length = select_block_length(daily_log, regime)

    # Point estimate on original data
    point = compute_metric_on_resample(
        daily_log, np.arange(n), regime_checkpoints, metric, regime,
    )
    if point is None:
        return BootstrapCI(
            metric=metric, point_estimate=0.0,
            ci_lower=0.0, ci_upper=0.0, std_error=0.0,
            n_bootstrap=n_bootstrap, confidence_level=confidence_level,
        )

    # Bootstrap resamples
    boot_values = []
    for _ in range(n_bootstrap):
        idx = stationary_block_bootstrap_indices(n, mean_block_length, rng)
        val = compute_metric_on_resample(
            daily_log, idx, regime_checkpoints, metric, regime,
        )
        if val is not None:
            boot_values.append(val)

    if len(boot_values) < 10:
        return BootstrapCI(
            metric=metric, point_estimate=point,
            ci_lower=point, ci_upper=point, std_error=0.0,
            n_bootstrap=n_bootstrap, confidence_level=confidence_level,
        )

    boot_arr = np.array(boot_values)
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(boot_arr, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    std_error = float(np.std(boot_arr, ddof=1))

    return BootstrapCI(
        metric=metric,
        point_estimate=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def run_bootstrap_suite(
    daily_log: list[dict],
    regime_checkpoints: list[tuple],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, BootstrapCI]:
    """Run bootstrap CIs for all key metrics.

    Returns dict of metric_name -> BootstrapCI.
    """
    results = {}

    for regime in ["contraction", "stagflation"]:
        for metric in ["auc", "brier", "reliability", "resolution"]:
            key = f"{regime}_{metric}"
            results[key] = bootstrap_confidence_interval(
                daily_log, regime_checkpoints,
                metric=metric, regime=regime,
                n_bootstrap=n_bootstrap, seed=seed,
            )

    # Detection metrics (regime-agnostic)
    results["detection_count"] = bootstrap_confidence_interval(
        daily_log, regime_checkpoints,
        metric="detection_count",
        n_bootstrap=n_bootstrap, seed=seed,
    )
    results["mean_detection_lag"] = bootstrap_confidence_interval(
        daily_log, regime_checkpoints,
        metric="mean_detection_lag",
        n_bootstrap=n_bootstrap, seed=seed,
    )

    return results
