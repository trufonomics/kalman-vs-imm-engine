"""
P6 Fix: Sparse Off-Diagonal Process Noise (Correlated Shocks).

The current Q is diagonal: shocks to inflation_trend and
commodity_pressure are treated as independent.  But an oil price
shock hits BOTH simultaneously — the inflation factor and commodity
factor receive correlated process noise.

Primiceri (2005) showed that correlated shocks matter for macro VARs.
However, a full off-diagonal Q on 8 factors has 36 free parameters
(8×9/2), which is unidentifiable from 15 streams.

This module adds only ECONOMICALLY MOTIVATED off-diagonal terms,
following the same philosophy as CROSS_DYNAMICS in the F matrix.
Each term represents a known co-movement channel where factor shocks
are not independent.

Only 4 terms are added (vs 28 possible off-diagonal entries).
Conservative cross-noise coefficients (0.15-0.30 of geometric mean
of diagonal entries) to avoid over-fitting.

Ref: Primiceri, G.E. (2005). "Time Varying Structural Vector
     Autoregressions and Monetary Policy." Review of Economic
     Studies 72(3), 821-852.
     Fischer et al. (2023). "General Bayesian TVP-VARs for
     Government Bond Yields." J. Applied Econometrics 38(1).
"""

import numpy as np
from heimdall.kalman_filter import FACTORS, FACTOR_INDEX, PERSISTENCE


# Economically motivated cross-noise channels
# (factor_a, factor_b, correlation_strength)
# Strength is fraction of sqrt(Q_aa * Q_bb): how much of a shock
# to factor_a also shocks factor_b (and vice versa).
CROSS_NOISE_CHANNELS = [
    # Oil shocks hit inflation and commodity simultaneously
    ("commodity_pressure", "inflation_trend", 0.30),

    # Financial stress and housing are co-shocked (credit channel)
    ("financial_conditions", "housing_momentum", 0.25),

    # Policy shocks hit financial conditions directly
    ("policy_stance", "financial_conditions", 0.25),

    # Growth shocks and labor shocks are co-determined
    ("growth_trend", "labor_pressure", 0.20),
]


def build_correlated_Q(base_Q: np.ndarray) -> np.ndarray:
    """Build Q with sparse off-diagonal terms.

    Adds economically motivated cross-noise entries to the diagonal Q.
    Each cross-noise entry is computed as:

        Q[i,j] = rho * sqrt(Q[i,i] * Q[j,j])

    where rho is the channel correlation strength.

    The resulting Q is guaranteed symmetric and PSD (since rho < 1.0
    for all channels, the matrix is diagonally dominant).

    Args:
        base_Q: Diagonal process noise matrix (N×N)

    Returns:
        New Q with off-diagonal entries added (N×N, symmetric PSD)
    """
    Q = base_Q.copy()

    for factor_a, factor_b, rho in CROSS_NOISE_CHANNELS:
        i = FACTOR_INDEX[factor_a]
        j = FACTOR_INDEX[factor_b]

        # Cross-noise = rho * geometric mean of diagonal entries
        cross = rho * np.sqrt(Q[i, i] * Q[j, j])
        Q[i, j] = cross
        Q[j, i] = cross

    return Q
