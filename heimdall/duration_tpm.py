"""
P2 Fix: Duration-Dependent Transition Probability Matrix.

Durland & McCurdy (1994) showed that recession exit probability increases
with duration — the longer you've been in contraction, the more likely
recovery becomes. This is a well-established empirical regularity:
NBER recessions have a finite expected duration (~10 months post-1945),
and the hazard function is non-constant.

A fixed TPM treats the contraction→expansion transition as memoryless
(0.050 every week regardless of how long contraction has persisted).
This causes two problems:
  1. The filter stays in contraction too long after recovery begins
  2. Innovation residuals become autocorrelated (systematic over-prediction
     of contraction persistence)

Fix: Track contraction duration (consecutive weeks with contraction as
dominant regime). Apply a logistic ramp to the contraction→expansion
transition probability as duration increases.

The same logic applies symmetrically to expansion (which also has
duration-dependent exit probability, though weaker — expansions CAN
last decades). We apply a gentler ramp to expansion→contraction.

Implementation adds zero latent states. The TPM modification uses only
the existing regime probability history.

Ref: Durland, J.M. & McCurdy, T.H. (1994). "Duration-Dependent
     Transitions in a Markov Model of U.S. GNP Growth."
     JBES, 12(3), 279-288.

     Filardo, A.J. (1994). "Business-Cycle Phases and Their Transitional
     Dynamics." JBES, 12(3), 299-308.
"""

import numpy as np


# ── NBER recession durations (months) for calibration ──
# Post-1970 recessions used to calibrate the logistic ramp:
#   1973-75: 16 months  (~70 weeks)
#   1980:     6 months  (~26 weeks)
#   1981-82: 16 months  (~70 weeks)
#   1990-91:  8 months  (~35 weeks)
#   2001:     8 months  (~35 weeks)
#   2007-09: 18 months  (~78 weeks)
#   2020:     2 months  (~9 weeks)
#
# Median: ~10 months (~43 weeks)
# Mean (excl. COVID outlier): ~12 months (~52 weeks)

# Logistic ramp parameters for contraction exit
CONTRACTION_D_HALF = 39      # Inflection point (weeks) — ~9 months
CONTRACTION_P_BASE = 0.050   # Baseline exit probability (matches TPM)
CONTRACTION_P_MAX = 0.150    # Maximum exit probability (3x baseline)
CONTRACTION_K = 0.08         # Logistic slope (moderate — not too sharp)

# Logistic ramp parameters for expansion exit (gentler)
# Expansions last much longer (avg 64 months post-1945) so the ramp
# should be slower and the ceiling lower.
EXPANSION_D_HALF = 180       # Inflection point (weeks) — ~3.5 years
EXPANSION_P_BASE = 0.030     # Baseline: expansion→contraction + stagflation
EXPANSION_P_MAX = 0.060      # Maximum (2x baseline, not 3x)
EXPANSION_K = 0.03           # Very gentle slope


def _logistic(d: float, d_half: float, k: float) -> float:
    """Standard logistic function: 0 at d=0, 0.5 at d=d_half, →1 as d→∞."""
    return 1.0 / (1.0 + np.exp(-k * (d - d_half)))


def get_contraction_exit_prob(duration_weeks: int) -> float:
    """Return duration-adjusted contraction→expansion transition probability.

    Args:
        duration_weeks: Consecutive weeks with contraction as dominant regime.

    Returns:
        Adjusted exit probability (always between P_BASE and P_MAX).
    """
    if duration_weeks <= 0:
        return CONTRACTION_P_BASE

    ramp = _logistic(duration_weeks, CONTRACTION_D_HALF, CONTRACTION_K)
    return CONTRACTION_P_BASE + (CONTRACTION_P_MAX - CONTRACTION_P_BASE) * ramp


def get_expansion_exit_prob(duration_weeks: int) -> float:
    """Return duration-adjusted expansion exit probability.

    This increases the total probability of leaving expansion (split
    between stagflation and contraction destinations) as expansion
    persists longer.

    Args:
        duration_weeks: Consecutive weeks with expansion as dominant regime.

    Returns:
        Adjusted total exit probability.
    """
    if duration_weeks <= 0:
        return EXPANSION_P_BASE

    ramp = _logistic(duration_weeks, EXPANSION_D_HALF, EXPANSION_K)
    return EXPANSION_P_BASE + (EXPANSION_P_MAX - EXPANSION_P_BASE) * ramp


def build_duration_adjusted_tpm(
    base_tpm: np.ndarray,
    contraction_duration: int,
    expansion_duration: int,
) -> np.ndarray:
    """Return a duration-adjusted TPM.

    Modifies the base TPM by:
    1. Increasing contraction→expansion as contraction persists
    2. Gently increasing expansion exit probability as expansion persists

    Branch order: [expansion, stagflation, contraction] (indices 0, 1, 2)

    Args:
        base_tpm: The fixed 3x3 TPM (rows sum to 1.0)
        contraction_duration: Weeks with contraction dominant
        expansion_duration: Weeks with expansion dominant

    Returns:
        New 3x3 TPM with duration-adjusted transitions (rows sum to 1.0)
    """
    tpm = base_tpm.copy()

    # ── Contraction duration adjustment ──
    # Increase contraction→expansion (row 2, col 0)
    new_exit = get_contraction_exit_prob(contraction_duration)
    old_exit = tpm[2, 0]  # Current contraction→expansion
    if new_exit > old_exit:
        delta = new_exit - old_exit
        tpm[2, 0] = new_exit
        # Remove delta from contraction→contraction (self-loop)
        tpm[2, 2] = max(tpm[2, 2] - delta, 0.50)  # Floor at 50%
        # Re-normalize row 2
        tpm[2] /= tpm[2].sum()

    # ── Expansion duration adjustment ──
    # Increase total expansion exit probability
    new_total_exit = get_expansion_exit_prob(expansion_duration)
    old_total_exit = tpm[0, 1] + tpm[0, 2]  # stagflation + contraction exits
    if new_total_exit > old_total_exit:
        # Distribute extra exit probability proportionally between
        # stagflation and contraction destinations
        delta = new_total_exit - old_total_exit
        ratio_stag = tpm[0, 1] / old_total_exit if old_total_exit > 0 else 0.67
        tpm[0, 1] += delta * ratio_stag
        tpm[0, 2] += delta * (1.0 - ratio_stag)
        # Remove from expansion→expansion
        tpm[0, 0] = max(tpm[0, 0] - delta, 0.50)
        # Re-normalize row 0
        tpm[0] /= tpm[0].sum()

    return tpm
