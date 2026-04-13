"""
P5 Fix: Score-Driven (GAS) Observation Noise Update.

The Ljung-Box 0% rate persists because observation noise R is fixed.
In reality, R varies dramatically: CPI measurement uncertainty during
COVID (supply chain disruptions, collection gaps) was nothing like
2019.  Initial claims went from 200K to 6.6M in two weeks.  Fixed R
treats these identically, producing innovations with time-varying
variance that are mechanically autocorrelated.

The GAS (Generalized Autoregressive Score) framework (Creal, Koopman
& Lucas 2013) updates R at each observation using the score of the
conditional log-likelihood:

    log R_t = omega + beta * log R_{t-1} + alpha * score_t

where score_t = (innovation_t^2 / S_t - 1) — the normalized squared
innovation minus its expected value under the model.  When the
innovation is larger than expected, R increases.  When smaller,
R decreases.

This is observation-driven (not parameter-driven like stochastic
volatility), so:
- No additional latent states
- Likelihood available in closed form
- O(1) update per observation
- Compatible with the existing IMM architecture

Parameters calibrated conservatively:
- beta = 0.95 (high persistence — noise doesn't change fast)
- alpha = 0.10 (moderate response to score — avoids over-reaction)
- omega = log(R_base) * (1 - beta) (ensures unconditional mean = R_base)

Ref: Creal, D., Koopman, S.J. & Lucas, A. (2013). "Generalized
     Autoregressive Score Models with Applications." J. Applied
     Econometrics 28(5), 777-795.
     Harvey, A.C. (2013). "Dynamic Models for Volatility and Heavy
     Tails." Cambridge University Press.
"""

import numpy as np


# GAS parameters — conservative defaults
GAS_BETA = 0.95       # Persistence of log-variance (high = slow adaptation)
GAS_ALPHA = 0.10      # Score scaling (low = less reactive to individual obs)

# Bounds to prevent numerical issues
LOG_R_MIN = -6.0      # R_min ≈ 0.0025
LOG_R_MAX = 4.0       # R_max ≈ 55


class GASNoiseTracker:
    """Score-driven observation noise tracker per stream.

    Maintains a running estimate of log(R) for each stream.  Updated
    at each observation using the GAS score.  Returns the current R
    value for use in the Kalman update.
    """

    def __init__(self, beta: float = GAS_BETA, alpha: float = GAS_ALPHA):
        self._beta = beta
        self._alpha = alpha
        # stream_key -> log(R_current)
        self._log_R: dict[str, float] = {}
        # stream_key -> log(R_base) for omega computation
        self._log_R_base: dict[str, float] = {}

    def initialize(self, stream_key: str, base_R: float) -> None:
        """Set the unconditional mean for a stream."""
        log_base = np.log(max(base_R, 1e-6))
        self._log_R[stream_key] = log_base
        self._log_R_base[stream_key] = log_base

    def get_R(self, stream_key: str) -> float | None:
        """Return current R estimate for this stream."""
        log_r = self._log_R.get(stream_key)
        if log_r is None:
            return None
        return np.exp(log_r)

    def update(
        self,
        stream_key: str,
        innovation: float,
        S: float,
    ) -> float:
        """Update R using the GAS score after a Kalman observation.

        The score for a Gaussian observation model is:
            score = innovation^2 / S - 1

        When innovation^2 > S (observation surprised the filter),
        score > 0 → R increases.  When innovation^2 < S, score < 0
        → R decreases.

        Args:
            stream_key: Stream identifier
            innovation: Raw innovation (z - H @ x), NOT z-scored
            S: Innovation variance (H @ P @ H.T + R)

        Returns:
            Updated R value.
        """
        if stream_key not in self._log_R:
            return S  # Fallback: return current S

        log_r = self._log_R[stream_key]
        log_r_base = self._log_R_base[stream_key]

        # Omega ensures unconditional mean = R_base
        omega = log_r_base * (1.0 - self._beta)

        # GAS score for Gaussian: (innovation^2 / S) - 1
        # Clamp to prevent extreme score values
        standardized_sq = min(innovation ** 2 / max(S, 1e-10), 50.0)
        score = standardized_sq - 1.0

        # GAS recursion in log-space
        log_r_new = omega + self._beta * log_r + self._alpha * score

        # Clamp to bounds
        log_r_new = max(LOG_R_MIN, min(LOG_R_MAX, log_r_new))

        self._log_R[stream_key] = log_r_new
        return np.exp(log_r_new)

    def get_all_R(self) -> dict[str, float]:
        """Return current R for all tracked streams."""
        return {
            stream: np.exp(log_r)
            for stream, log_r in self._log_R.items()
        }
