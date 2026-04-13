"""
P0 Fix: Regime-Dependent Observation Noise (R).

Chan & Eisenstat (2018) showed that stochastic volatility on R and Q
is where the largest empirical gains come from — more than time-varying
F or H. This module implements the cheapest version: regime-dependent
R multipliers using existing IMM branch probabilities.

During contraction/crisis, observation noise is inflated — the filter
trusts each data point less because the data is genuinely noisier
(claims spike 10x, oil swings ±30%, equities gap ±10%). During
expansion, R stays at the calibrated baseline.

This adds zero new latent states. The IMM mixing step already blends
branches, so the effective R adapts automatically as regime
probabilities shift.

Ref: Chan, J.C.C. & Eisenstat, E. (2018). "Bayesian Model Comparison
     for Time-Varying Parameter VARs with Stochastic Volatility."
     J. Applied Econometrics 33(4), 509-532.
"""

import numpy as np

# Regime-specific R multipliers per stream category.
#
# These are applied ON TOP of the base R × frequency_scale values.
# A multiplier of 1.0 = no change from baseline.
#
# Rationale for each category:
#
# HIGH VOLATILITY IN CONTRACTION (3.0x):
#   - INITIAL_CLAIMS: kurtosis 395 in backtest (COVID spike). Claims can
#     10x in a week during recession. Fixed R treats this as 50+ sigma.
#   - SP500: kurtosis 17.7. Black Monday = -22% in one day. COVID = -34%
#     in 3 weeks. These are not 50-sigma events; they're regime-dependent.
#   - BTC_USD: kurtosis 12.1. Correlates with risk-off in contraction.
#
# MODERATE VOLATILITY IN STAGFLATION (2.0x):
#   - OIL_PRICE: oil shocks define stagflation. ±30% moves are common.
#   - PPI: producer prices swing with commodity inputs.
#   - COMMODITY_PRESSURE_RAW: direct commodity exposure.
#
# STABLE ACROSS REGIMES (1.0x):
#   - US_CPI_YOY, CORE_CPI: CPI is a smoothed index, low measurement noise
#     regardless of regime. The LEVEL changes, not the noise.
#   - FED_FUNDS_RATE: set by FOMC, extremely low noise by construction.
#   - 10Y_YIELD: moves more in vol regimes but still mean-reverting.

# Streams categorized by volatility behavior across regimes
_FINANCIAL_STRESS_STREAMS = {
    "SP500", "BTC_USD", "INITIAL_CLAIMS", "GOLD_PRICE",
}
_COMMODITY_STREAMS = {
    "OIL_PRICE", "COPPER_PRICE", "PPI", "COMMODITY_PRESSURE_RAW",
}
_HOUSING_STREAMS = {
    "HOUSING_STARTS", "HOME_PRICES",
}
_STABLE_STREAMS = {
    "US_CPI_YOY", "CORE_CPI", "FED_FUNDS_RATE", "10Y_YIELD",
    "UNEMPLOYMENT_RATE", "NONFARM_PAYROLLS", "CONSUMER_CONFIDENCE",
    "RETAIL_SALES",
}


def get_regime_r_multiplier(regime: str, stream_key: str) -> float:
    """Return R multiplier for a given regime and stream.

    Args:
        regime: One of "expansion", "soft_landing", "stagflation",
                "contraction", "recession"
        stream_key: The stream identifier (e.g., "US_CPI_YOY")

    Returns:
        Multiplier to apply to base R value. 1.0 = no change.
    """
    # Normalize regime names
    regime_lower = regime.lower().replace(" ", "_")
    is_contraction = regime_lower in ("contraction", "recession")
    is_stagflation = regime_lower in ("stagflation",)

    if is_contraction:
        if stream_key in _FINANCIAL_STRESS_STREAMS:
            return 3.0   # Financial markets are 3x noisier in crisis
        if stream_key in _COMMODITY_STREAMS:
            return 2.0   # Commodities swing in crisis too
        if stream_key in _HOUSING_STREAMS:
            return 2.5   # Housing data gets volatile in recessions
        return 1.5       # Everything else gets a mild inflation

    if is_stagflation:
        if stream_key in _COMMODITY_STREAMS:
            return 2.5   # Commodity shocks define stagflation
        if stream_key in _FINANCIAL_STRESS_STREAMS:
            return 2.0   # Financial stress elevated
        if stream_key in _HOUSING_STREAMS:
            return 1.5   # Housing moderately affected
        return 1.2       # Mild inflation for stable streams

    # Expansion / soft_landing — baseline noise
    return 1.0


def get_regime_q_scale(regime: str) -> float:
    """Return Q (process noise) scale for a given regime.

    During crisis, the economy's state itself evolves faster — factors
    can shift dramatically week to week. Higher Q means the filter
    forgets old state faster and responds more to new data.

    During expansion, state evolves slowly and predictably.

    Args:
        regime: Regime identifier

    Returns:
        Scale factor for Q matrix. 1.0 = baseline.
    """
    regime_lower = regime.lower().replace(" ", "_")

    if regime_lower in ("contraction", "recession"):
        return 2.0   # State evolves 2x faster in crisis
    if regime_lower in ("stagflation",):
        return 1.5   # Moderately faster evolution
    return 1.0       # Baseline for expansion


# Pre-computed multiplier tables for fast lookup during backtest
REGIME_R_TABLE: dict[str, dict[str, float]] = {}
REGIME_Q_TABLE: dict[str, float] = {}

for _regime in ("expansion", "stagflation", "contraction"):
    REGIME_Q_TABLE[_regime] = get_regime_q_scale(_regime)
    REGIME_R_TABLE[_regime] = {}
    # Build for all known streams
    from heimdall.kalman_filter import STREAM_LOADINGS
    for _stream in STREAM_LOADINGS:
        REGIME_R_TABLE[_regime][_stream] = get_regime_r_multiplier(_regime, _stream)
