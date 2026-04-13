"""
P3 Fix: Data-Calibrated F Matrix Persistence.

The original PERSISTENCE values (0.85-0.98) were set from literature
references and treated as daily step parameters. But the predict() step
runs daily, and at those values the state decays far too fast between
monthly observations:

    inflation_trend:  0.97^22 = 0.51 (50% decay per month)
    policy_stance:    0.98^22 = 0.64 (36% decay per month)
    labor_pressure:   0.96^22 = 0.40 (60% decay per month)

The calibration study (calibration_from_history.py) computed the actual
monthly AR(1) from 51 years of FRED data and converted to daily equivalents:

    inflation_trend:  monthly AR(1) = 0.986 → daily = 0.9994
    policy_stance:    monthly AR(1) = 0.991 → daily = 0.9996
    labor_pressure:   monthly AR(1) = 0.769 → daily = 0.9881

This mismatch is why Ljung-Box fails on every stream at 0% pass rate.
The filter's too-low persistence causes it to predict systematic decay
toward zero between observations, but the actual state barely changes.
Innovation_t ≈ innovation_{t-1} because both reflect the same
"still elevated, filter predicted decay" pattern.

The fix uses a BLEND of theory and data values to avoid over-fitting
to the calibration sample while correcting the worst mismatches.
Commodity_pressure is left at the theory value because its monthly
AR(1) is -0.017 (mean-reverting, not persistent).

Ref: Ang, A., Bekaert, G. & Wei, M. (2007). "Do Macro Variables,
     Asset Markets, or Surveys Forecast Inflation Better?" JME 54(4).
     (Cross-validates AR(1) estimates for macro factors.)
"""

# Current (theory) values for reference
THEORY_PERSISTENCE = {
    "inflation_trend": 0.97,
    "growth_trend": 0.96,
    "labor_pressure": 0.96,
    "housing_momentum": 0.98,
    "financial_conditions": 0.95,
    "commodity_pressure": 0.85,
    "consumer_sentiment": 0.96,
    "policy_stance": 0.98,
}

# Data-derived daily equivalents from calibration_from_history.py
DATA_PERSISTENCE = {
    "inflation_trend": 0.9994,
    "growth_trend": 0.9617,
    "labor_pressure": 0.9881,
    "housing_momentum": 0.9989,
    "financial_conditions": 0.9698,
    "commodity_pressure": 0.50,      # Mean-reverting — use theory
    "consumer_sentiment": 0.9979,
    "policy_stance": 0.9996,
}

# Blend: 30% theory + 70% data for most factors
# Exception: commodity_pressure stays at theory (data AR(1) is negative)
# Exception: growth_trend stays closer to current (data and theory agree)
BLEND_WEIGHT = 0.70  # Weight on data-derived value


def compute_blended_persistence() -> dict[str, float]:
    """Return blended persistence values."""
    result = {}
    for factor in THEORY_PERSISTENCE:
        theory = THEORY_PERSISTENCE[factor]
        data = DATA_PERSISTENCE[factor]

        if factor == "commodity_pressure":
            # Data AR(1) is -0.017 — not meaningful for daily persistence
            result[factor] = theory
        else:
            blended = (1 - BLEND_WEIGHT) * theory + BLEND_WEIGHT * data
            # Cap at 0.999 to maintain some mean-reversion
            result[factor] = min(round(blended, 4), 0.999)

    return result


# Pre-computed blended values
CALIBRATED_PERSISTENCE = compute_blended_persistence()
