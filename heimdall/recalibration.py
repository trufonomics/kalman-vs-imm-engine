"""
Post-Hoc Recalibration for IMM Regime Probabilities.

The IMM's Bayesian update produces miscalibrated probabilities because:
1. The TPM persistence prior (0.97 self-transition) anchors probabilities
   toward the dominant regime — a well-known property of Markov-switching
   filtered probabilities (Hamilton 1989, Kim & Nelson 1999 Ch. 4)
2. The model is intentionally misspecified (hand-tuned adjustments, not MLE)
3. The observation model assumes Gaussian innovations

Isotonic regression (Zadrozny & Elkan 2002) maps raw IMM probabilities
to calibrated frequencies while preserving probability ordering.

Likelihood tempering (power posterior) attacks the structural cause:
standard Bayesian updates concentrate too fast under misspecification,
producing overconfident posteriors. The tempering exponent eta < 1 slows
posterior concentration. Following the power-posterior framework
(Grunwald & van Ommen 2017, Bhattacharya et al. 2019), we cap eta at
0.70 — an empirically chosen constant validated via grid search on the
51-year backtest. Grunwald (2012) motivates learning eta from data;
our fixed cap is a pragmatic simplification.

Refs:
  Zadrozny, B. & Elkan, C. (2002). "Transforming Classifier Scores
      into Accurate Multiclass Probability Estimates." KDD-02.
  Grunwald, P. & van Ommen, T. (2017). "Inconsistency of Bayesian
      Inference for Misspecified Linear Models, and a Proposal for
      Repairing It." Bayesian Analysis 12(4), 1069-1103.
  Grunwald, P. (2012). "The Safe Bayesian." Algorithmic Learning Theory.
  Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines."
      Advances in Large Margin Classifiers.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class IsotonicCalibrator:
    """Monotone recalibration function learned from historical data.

    Uses the pool-adjacent-violators algorithm (PAVA) to fit a
    non-decreasing mapping from raw probabilities to calibrated
    frequencies. Equivalent to sklearn's IsotonicRegression but
    dependency-free.
    """
    x_thresholds: list[float]
    y_values: list[float]
    regime: str

    def predict(self, raw_prob: float) -> float:
        """Map a raw IMM probability to a calibrated probability."""
        if not self.x_thresholds:
            return raw_prob

        # Clamp to [0, 1]
        raw_prob = max(0.0, min(1.0, raw_prob))

        # Binary search for position
        if raw_prob <= self.x_thresholds[0]:
            return self.y_values[0]
        if raw_prob >= self.x_thresholds[-1]:
            return self.y_values[-1]

        # Linear interpolation between thresholds
        for i in range(len(self.x_thresholds) - 1):
            if self.x_thresholds[i] <= raw_prob <= self.x_thresholds[i + 1]:
                x0, x1 = self.x_thresholds[i], self.x_thresholds[i + 1]
                y0, y1 = self.y_values[i], self.y_values[i + 1]
                if abs(x1 - x0) < 1e-12:
                    return y0
                t = (raw_prob - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        return raw_prob


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm.

    Fits a non-decreasing sequence that minimizes weighted squared error.
    This is the core of isotonic regression.
    """
    n = len(y)
    result = y.astype(float).copy()
    weights = w.astype(float).copy()

    # Forward pass: merge violating pairs
    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Pool: weighted average of the block
            block_start = i
            total_w = weights[i] + weights[i + 1]
            total_wy = result[i] * weights[i] + result[i + 1] * weights[i + 1]

            # Extend block backward while violated
            while block_start > 0 and result[block_start - 1] > total_wy / total_w:
                block_start -= 1
                total_w += weights[block_start]
                total_wy += result[block_start] * weights[block_start]

            # Set all elements in block to pooled value
            pooled = total_wy / total_w
            for j in range(block_start, i + 2):
                result[j] = pooled
                weights[j] = total_w

            i = block_start
        else:
            i += 1

    return result


def fit_isotonic_calibrator(
    forecasts: list[float],
    observations: list[int],
    regime: str,
    n_bins: int = 20,
) -> IsotonicCalibrator:
    """Fit isotonic regression calibrator from historical data.

    Bins forecasts, computes observed frequency per bin, then
    fits isotonic regression to ensure monotonicity.

    Args:
        forecasts: raw IMM probabilities
        observations: binary ground truth (1 = regime active)
        regime: label
        n_bins: number of bins for initial binning
    """
    f = np.array(forecasts)
    o = np.array(observations, dtype=float)

    # Sort by forecast probability
    order = np.argsort(f)
    f_sorted = f[order]
    o_sorted = o[order]

    # Bin into n_bins groups
    bin_size = max(1, len(f_sorted) // n_bins)
    x_centers = []
    y_freq = []
    w_counts = []

    for i in range(0, len(f_sorted), bin_size):
        chunk_f = f_sorted[i:i + bin_size]
        chunk_o = o_sorted[i:i + bin_size]
        if len(chunk_f) == 0:
            continue
        x_centers.append(float(np.mean(chunk_f)))
        y_freq.append(float(np.mean(chunk_o)))
        w_counts.append(float(len(chunk_f)))

    if not x_centers:
        return IsotonicCalibrator([], [], regime)

    x_arr = np.array(x_centers)
    y_arr = np.array(y_freq)
    w_arr = np.array(w_counts)

    # Apply PAVA for monotonicity
    y_isotonic = _pava(y_arr, w_arr)

    return IsotonicCalibrator(
        x_thresholds=[round(x, 6) for x in x_arr],
        y_values=[round(float(y), 6) for y in y_isotonic],
        regime=regime,
    )


def recalibrate_probabilities(
    probabilities: dict[str, float],
    calibrators: dict[str, IsotonicCalibrator],
) -> dict[str, float]:
    """Recalibrate a set of regime probabilities and renormalize.

    Args:
        probabilities: {regime_key: raw_probability}
        calibrators: {regime_key: IsotonicCalibrator}

    Returns:
        Recalibrated probabilities summing to 1.0
    """
    recal = {}
    for key, raw_p in probabilities.items():
        cal = calibrators.get(key)
        if cal is not None:
            recal[key] = cal.predict(raw_p)
        else:
            recal[key] = raw_p

    # Renormalize to sum to 1
    total = sum(recal.values())
    if total > 1e-12:
        recal = {k: v / total for k, v in recal.items()}

    return recal
