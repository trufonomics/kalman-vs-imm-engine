"""
P2b Fix: State-Dependent Transition Probability Matrix.

Filardo (1994) showed that economic state variables can drive regime
transition probabilities via a logistic link — the probability of
leaving expansion should depend on the CURRENT values of factors like
financial_conditions and growth_trend, not just on how long you've
been in the regime.

This is complementary to P2a (duration-dependent TPM, which was neutral).
The hypothesis here is that factor values contain transition-relevant
information that the fixed TPM ignores.

Logistic link: for each off-diagonal TPM entry, a subset of factors
modulates the transition probability. When the relevant factors are
at their mean (zero), the base TPM applies. When they deviate, the
transition probability shifts via sigmoid.

Implementation adds zero latent states. Uses current Kalman state
estimates (the factor vector x) to adjust the TPM each cycle.

Ref: Filardo, A.J. (1994). "Business-Cycle Phases and Their
     Transitional Dynamics." JBES, 12(3), 299-308.
"""

import numpy as np

# ── Factor-driven transition rules ──
#
# Each rule specifies: which TPM entry to modify, which factors drive it,
# weights on each factor, and the scaling range.
#
# Convention: positive weight means "when this factor is HIGH, increase
# the transition probability." Negative weight means "when this factor
# is HIGH, decrease the transition probability."
#
# TPM indices: [expansion=0, stagflation=1, contraction=2]

TRANSITION_RULES = [
    {
        "name": "expansion→contraction",
        "from_idx": 0,
        "to_idx": 2,
        "factors": {
            "financial_conditions": -0.8,   # Deteriorating financials → more likely to contract
            "growth_trend": -0.6,           # Falling growth → more likely to contract
            "consumer_sentiment": -0.4,     # Falling sentiment → early warning
        },
        "base_prob": 0.010,
        "min_prob": 0.005,
        "max_prob": 0.050,
    },
    {
        "name": "expansion→stagflation",
        "from_idx": 0,
        "to_idx": 1,
        "factors": {
            "inflation_trend": 0.8,         # Rising inflation → more likely to stagflate
            "commodity_pressure": 0.6,      # Rising commodities → supply-side pressure
            "growth_trend": -0.3,           # Weakening growth + inflation = stagflation
        },
        "base_prob": 0.020,
        "min_prob": 0.005,
        "max_prob": 0.060,
    },
    {
        "name": "contraction→expansion",
        "from_idx": 2,
        "to_idx": 0,
        "factors": {
            "financial_conditions": 0.7,    # Improving financials → recovery signal
            "growth_trend": 0.6,            # Growth turning positive → recovery
            "consumer_sentiment": 0.4,      # Sentiment rebounding → demand returning
        },
        "base_prob": 0.050,
        "min_prob": 0.020,
        "max_prob": 0.120,
    },
    {
        "name": "stagflation→expansion",
        "from_idx": 1,
        "to_idx": 0,
        "factors": {
            "inflation_trend": -0.8,        # Falling inflation → stagflation resolving
            "commodity_pressure": -0.5,     # Commodity prices easing
            "growth_trend": 0.4,            # Growth recovering
        },
        "base_prob": 0.030,
        "min_prob": 0.010,
        "max_prob": 0.080,
    },
    {
        "name": "stagflation→contraction",
        "from_idx": 1,
        "to_idx": 2,
        "factors": {
            "growth_trend": -0.7,           # Growth collapsing → tipping into recession
            "financial_conditions": -0.6,   # Financial stress mounting
            "labor_pressure": -0.4,         # Labor shedding
        },
        "base_prob": 0.020,
        "min_prob": 0.005,
        "max_prob": 0.060,
    },
    {
        "name": "contraction→stagflation",
        "from_idx": 2,
        "to_idx": 1,
        "factors": {
            "inflation_trend": 0.7,         # Rising inflation during contraction → stagflation
            "commodity_pressure": 0.6,      # Commodity shock during recession
        },
        "base_prob": 0.020,
        "min_prob": 0.005,
        "max_prob": 0.050,
    },
]


def _logistic_shift(
    factor_values: dict[str, float],
    factor_weights: dict[str, float],
    base_prob: float,
    min_prob: float,
    max_prob: float,
) -> float:
    """Compute state-dependent transition probability.

    The logistic function maps the weighted factor sum to a probability
    shift. When the weighted sum is zero (factors at mean), the base
    probability applies. Positive sum → probability increases toward
    max_prob. Negative sum → probability decreases toward min_prob.

    Args:
        factor_values: Current Kalman state estimates {factor: z-score}
        factor_weights: Weights for each factor in the logistic link
        base_prob: TPM entry when factors are at mean
        min_prob: Floor for this transition
        max_prob: Ceiling for this transition

    Returns:
        Adjusted transition probability.
    """
    # Weighted sum of factor values
    z = 0.0
    for factor, weight in factor_weights.items():
        z += weight * factor_values.get(factor, 0.0)

    # Sigmoid maps z to [0, 1]
    sigmoid = 1.0 / (1.0 + np.exp(-z))

    # Map [0, 1] to [min_prob, max_prob] with base_prob at sigmoid=0.5
    return min_prob + (max_prob - min_prob) * sigmoid


def build_state_adjusted_tpm(
    base_tpm: np.ndarray,
    factor_values: dict[str, float],
) -> np.ndarray:
    """Return a state-adjusted TPM based on current factor values.

    Modifies off-diagonal entries of the base TPM using factor-driven
    logistic links, then adjusts the diagonal to maintain row sums of 1.0.

    Args:
        base_tpm: The fixed 3x3 TPM (rows sum to 1.0)
        factor_values: Current Kalman state {factor_name: z-score}

    Returns:
        New 3x3 TPM with state-adjusted transitions (rows sum to 1.0)
    """
    tpm = base_tpm.copy()

    for rule in TRANSITION_RULES:
        i = rule["from_idx"]
        j = rule["to_idx"]

        new_prob = _logistic_shift(
            factor_values=factor_values,
            factor_weights=rule["factors"],
            base_prob=rule["base_prob"],
            min_prob=rule["min_prob"],
            max_prob=rule["max_prob"],
        )

        tpm[i, j] = new_prob

    # Re-normalize: adjust diagonal to maintain row sums of 1.0
    for i in range(3):
        off_diag_sum = sum(tpm[i, j] for j in range(3) if j != i)
        tpm[i, i] = max(1.0 - off_diag_sum, 0.50)  # Floor at 50% self-transition
        # Re-normalize entire row
        tpm[i] /= tpm[i].sum()

    return tpm
