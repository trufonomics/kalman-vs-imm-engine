"""
P1 Fix: Regime-Switching Factor Loadings (H matrix).

Urga & Wang (2024) showed that regime-switching factor loadings beat
random-walk drifting loadings for medium-sized models like ours.
The same CPI reading should load differently onto inflation_trend
in expansion vs stagflation — because the economic interpretation
of the data changes with the regime.

Implementation: Each regime gets a per-stream scaling vector that
adjusts the baseline H_row. This is computationally cheap — just
K copies of the loading row selected by branch_id.

Key economic intuitions:
- In stagflation, inflation streams load MORE on commodity_pressure
  (supply shocks drive inflation) and LESS on growth_trend (growth
  is disconnected from inflation).
- In contraction, labor streams load MORE on labor_pressure and
  financial_conditions (these are the dominant signals), while
  inflation loadings attenuate (deflation makes CPI less informative).
- In expansion, baseline loadings apply.

Ref: Urga, G. & Wang, F. (2024). "Estimation and Inference for High
     Dimensional Factor Model with Regime Switching." J. Econometrics
     241(2). arXiv:2205.12126.

     Feldkircher, M. et al. (2024). "Sophisticated and Small versus
     Simple and Sizeable: When Does It Pay Off to Introduce Drifting
     Coefficients in Bayesian VARs?" J. Forecasting.
     (Warns against overfitting — keep adjustments conservative.)
"""

import numpy as np
from heimdall.kalman_filter import FACTORS, FACTOR_INDEX, N_FACTORS, STREAM_LOADINGS

# Per-factor loading scale by regime.
# Values > 1.0 amplify the loading; < 1.0 attenuate it.
# Conservative magnitudes (1.2-1.5x) per Feldkircher et al. warning.

_REGIME_FACTOR_SCALES: dict[str, dict[str, float]] = {
    "expansion": {
        # Baseline — no scaling
    },
    "stagflation": {
        # Inflation and commodity channels amplified
        "inflation_trend": 1.3,      # Inflation IS the story
        "commodity_pressure": 1.4,   # Supply shocks dominate
        "growth_trend": 0.7,         # Growth decouples from inflation
        "consumer_sentiment": 0.8,   # Sentiment less informative
        "policy_stance": 1.2,        # Policy tightening is central
    },
    "contraction": {
        # Labor and financial channels amplified
        "labor_pressure": 1.4,       # Labor is THE recession signal
        "financial_conditions": 1.3, # Financial stress dominates
        "housing_momentum": 1.3,     # Housing collapse is central
        "growth_trend": 1.2,         # Growth signal is strong
        "inflation_trend": 0.7,      # Deflation makes CPI less useful
        "commodity_pressure": 0.8,   # Commodity drops follow, don't lead
        "consumer_sentiment": 1.2,   # Sentiment collapse informative
    },
}


def get_regime_h_row(
    regime: str,
    stream_key: str,
    base_h_row: np.ndarray,
) -> np.ndarray:
    """Return regime-adjusted H row for a stream.

    Applies per-factor scaling to the baseline loading vector.
    Expansion returns unchanged baseline.

    Args:
        regime: One of "expansion", "stagflation", "contraction"
        stream_key: Stream identifier
        base_h_row: The baseline H row from STREAM_LOADINGS

    Returns:
        Scaled H row (new array, does not mutate input)
    """
    regime_lower = regime.lower().replace(" ", "_")
    if regime_lower in ("soft_landing",):
        regime_lower = "expansion"
    if regime_lower in ("recession",):
        regime_lower = "contraction"

    scales = _REGIME_FACTOR_SCALES.get(regime_lower, {})
    if not scales:
        return base_h_row.copy()

    h_adjusted = base_h_row.copy()
    for factor, scale in scales.items():
        idx = FACTOR_INDEX.get(factor)
        if idx is not None:
            h_adjusted[idx] *= scale

    return h_adjusted


# Pre-compute all regime H rows for fast lookup during backtest
REGIME_H_ROWS: dict[str, dict[str, np.ndarray]] = {}

for _regime in ("expansion", "stagflation", "contraction"):
    REGIME_H_ROWS[_regime] = {}
    for _stream, _loadings in STREAM_LOADINGS.items():
        base_h = np.zeros(N_FACTORS)
        for _factor, _loading in _loadings.items():
            _idx = FACTOR_INDEX.get(_factor)
            if _idx is not None:
                base_h[_idx] = _loading
        REGIME_H_ROWS[_regime][_stream] = get_regime_h_row(_regime, _stream, base_h)
