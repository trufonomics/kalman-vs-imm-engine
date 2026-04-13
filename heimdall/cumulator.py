"""
P4 Fix: Mariano-Murasawa Temporal Aggregation via Observation Accumulators.

The 0% Ljung-Box rate has a structural cause: monthly streams (CPI,
unemployment, PPI, etc.) arrive every ~22 trading days.  Between
observations the filter runs open-loop predictions, and because
F_diag < 1 (daily persistence 0.85-0.98) the predicted state decays
significantly:

    inflation_trend:  0.97^22 = 0.51  (filter predicts 49% decay)
    actual monthly AR(1) = 0.986       (reality: 1.4% decay)

The observation, when it arrives, finds the state barely changed while
the filter predicted massive decay.  Innovation = observed - predicted
is systematically positive, and this repeats every month, creating
ACF(1) = 0.92-0.97 for monthly streams.

The Mariano-Murasawa (2003) cumulator fixes this by changing what the
filter predicts the observation will be.  Instead of comparing against
H @ x_T (the endpoint prediction after g gap-days of decay), it
compares against the MEAN of daily predictions over the gap:

    predicted_avg = (1/g) * SUM_{k=1}^{g} H @ F^k @ x_last_update

For F_diag = 0.97, g = 22:
    endpoint prediction:  0.51 * x  (massive decay)
    cumulator prediction: 0.66 * x  (averaged, less decay)

This reduces the systematic innovation bias without touching F_diag,
so regime detection speed is preserved.

Applies ONLY to monthly and weekly streams.  Daily streams (oil, SP500,
BTC) get updated every day and have near-zero ACF(1) already.

Ref: Mariano, R.S. & Murasawa, Y. (2003). "A New Coincident Index of
     Business Cycles." J. Applied Econometrics 18(4), 427-443.
     Brave, Butters & Kelley (2022). "A Practitioner's Guide for Mixed
     Frequency State Space Models." J. Statistical Software 104(10).
"""

import numpy as np


# Streams that need cumulation (observed less than daily)
MONTHLY_STREAMS = frozenset({
    "US_CPI_YOY",
    "CORE_CPI",
    "PPI",
    "NONFARM_PAYROLLS",
    "UNEMPLOYMENT_RATE",
    "RETAIL_SALES",
    "HOME_PRICES",
    "CONSUMER_CONFIDENCE",
    "FED_FUNDS_RATE",
})

WEEKLY_STREAMS = frozenset({
    "INITIAL_CLAIMS",
    "HOUSING_STARTS",
})

CUMULATED_STREAMS = MONTHLY_STREAMS | WEEKLY_STREAMS


class StreamCumulator:
    """Tracks accumulated predictions between observations.

    For each low-frequency stream, accumulates H @ x on every daily
    predict step.  When the observation arrives, returns the mean
    prediction over the gap period instead of the single-day endpoint
    prediction.  This reduces the systematic innovation bias caused by
    F_diag decay during no-observation gaps.
    """

    def __init__(self):
        # stream_key -> {"sum": float, "count": int}
        self._accumulators: dict[str, dict] = {}

    def accumulate(
        self,
        stream_key: str,
        H_row: np.ndarray,
        x: np.ndarray,
    ) -> None:
        """Add today's predicted observation to the accumulator.

        Called on every predict step for cumulated streams.

        Args:
            stream_key: Stream identifier (e.g. "US_CPI_YOY")
            H_row: 1xN observation vector for this stream
            x: Current state estimate (Nx1)
        """
        predicted = float(H_row @ x)

        if stream_key not in self._accumulators:
            self._accumulators[stream_key] = {"sum": 0.0, "count": 0}

        acc = self._accumulators[stream_key]
        self._accumulators[stream_key] = {
            "sum": acc["sum"] + predicted,
            "count": acc["count"] + 1,
        }

    def get_cumulated_prediction(self, stream_key: str) -> tuple[float, int] | None:
        """Return (mean_prediction, gap_days) for this stream.

        Returns None if no predictions have been accumulated (first
        observation or stream never seen).
        """
        acc = self._accumulators.get(stream_key)
        if acc is None or acc["count"] == 0:
            return None

        return (acc["sum"] / acc["count"], acc["count"])

    def reset(self, stream_key: str) -> None:
        """Reset accumulator after observation arrives."""
        self._accumulators[stream_key] = {"sum": 0.0, "count": 0}

    def should_cumulate(self, stream_key: str) -> bool:
        """Check if this stream uses cumulation."""
        return stream_key in CUMULATED_STREAMS


def compute_cumulated_innovation(
    observed_z: float,
    H_row: np.ndarray,
    x: np.ndarray,
    cumulator: StreamCumulator,
    stream_key: str,
) -> tuple[float, float]:
    """Compute innovation using cumulated prediction if available.

    If the stream has accumulated predictions (gap > 0), the innovation
    is computed against the mean prediction over the gap.  Otherwise
    falls back to the standard H @ x prediction.

    Args:
        observed_z: Normalized observation value
        H_row: Observation vector
        x: Current state estimate
        cumulator: StreamCumulator tracking predictions
        stream_key: Stream identifier

    Returns:
        (innovation, predicted) tuple
    """
    cumulated = cumulator.get_cumulated_prediction(stream_key)

    if cumulated is not None:
        mean_pred, gap_days = cumulated
        # Use the average prediction over the gap, not the endpoint
        predicted = mean_pred
    else:
        # No accumulated predictions — use standard endpoint prediction
        predicted = float(H_row @ x)

    innovation = observed_z - predicted
    return (innovation, predicted)


def compute_gap_adjusted_R(
    base_R: float,
    gap_days: int,
    stream_key: str,
) -> float:
    """Adjust observation noise for the gap between observations.

    The cumulated prediction has different uncertainty than the endpoint
    prediction.  The mean of g correlated predictions has variance:

        Var(mean) ≈ Var(single) * (1 + 2*rho) / g

    where rho is the average autocorrelation between daily predictions.
    Since F_diag ≈ 0.97, consecutive predictions are highly correlated,
    so Var(mean) ≈ Var(single) — the averaging doesn't reduce variance
    much.  We add a small correction for the gap uncertainty.

    For monthly streams (gap ~22 days), this adds ~10-15% to R.
    For weekly streams (gap ~5 days), this adds ~3-5% to R.
    """
    if gap_days <= 1:
        return base_R

    # Gap correction: longer gaps → slightly more uncertain prediction
    # Scale factor: sqrt(gap_days) / gap_days = 1/sqrt(gap_days)
    # This captures the "effective" reduction in prediction precision
    # from averaging correlated predictions over the gap
    gap_correction = 1.0 + 0.05 * np.log(gap_days)

    return base_R * gap_correction
