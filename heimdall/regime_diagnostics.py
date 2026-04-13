"""
Regime Detector Diagnostics — Proper Evaluation for IMM.

The previous diagnostic suite used Ljung-Box as the primary metric.
As is well-known in the regime-switching literature (Hamilton 1989,
Kim & Nelson 1999), innovations from Markov-switching models follow
mixture distributions — the composite innovation inherits
autocorrelation from regime-mixing even when each regime's innovations
are individually white. Ljung-Box, which assumes homoskedastic
Gaussian innovations, is therefore inappropriate.

Hashimzade et al. (2024) — closest published IMM macro application —
do NOT report Ljung-Box. They evaluate via regime classification
accuracy against NBER dates.

This module implements the diagnostics the regime-switching literature
actually uses:

Priority 1 (primary):
  - Brier Score decomposition (Murphy 1973, Chauvet & Piger 2008)
  - ROC/AUC (Berge & Jordà 2011)
  - Diebold-Mariano test (Diebold & Mariano 1995)

Priority 2 (supporting):
  - PIT histogram (Diebold, Gunther & Tay 1998)
  - Sharpness (Gneiting, Balabdaoui & Raftery 2007)
  - Detection lag (standard in all regime-switching evaluations)

Refs:
  Murphy, A.H. (1973). "A New Vector Partition of the Probability
      Score." J. Applied Meteorology 12(4), 595-600.
  Chauvet, M. & Piger, J. (2008). "A Comparison of the Real-Time
      Performance of Business Cycle Dating Methods." J. Business &
      Economic Statistics 26(1), 42-49.
  Berge, T.J. & Jordà, Ò. (2011). "Evaluating the Classification
      of Economic Activity into Recessions and Expansions."
      American Economic J.: Macroeconomics 3(2), 246-277.
  Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive
      Accuracy." J. Business & Economic Statistics 13(3), 253-263.
  Diebold, F.X., Gunther, T.A. & Tay, A.S. (1998). "Evaluating
      Density Forecasts with Applications to Financial Risk
      Management." International Economic Review 39(4), 863-883.
  Gneiting, T., Balabdaoui, F. & Raftery, A.E. (2007). "Probabilistic
      Forecasts, Calibration and Sharpness." J. Royal Statistical
      Society B 69(2), 243-268.
  Ferro, C.A.T. & Fricker, T.E. (2012). "A Bias-Corrected Decomposition
      of the Brier Score." QJRMS 138(668), 1954-1960.
  Berkowitz, J. (2001). "Testing Density Forecasts, with Applications
      to Risk Management." JBES 19(4), 465-474.
  Kim, C.-J. & Nelson, C.R. (1999). "State-Space Models with Regime
      Switching." MIT Press, Ch. 4.
  Hashimzade, N. et al. (2024). "Interacting Multiple Model..."
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


# ── NBER ground truth ─────────────────────────────────────────────────
# Binary regime indicators at each date.  We define "contraction" and
# "stagflation" windows from NBER + CPI data.  Everything else is
# "expansion."  These match the REGIME_CHECKPOINTS in the backtest but
# expressed as continuous date ranges for per-observation evaluation.

NBER_CONTRACTION_WINDOWS = [
    # (start, end) — NBER recession dates
    ("1980-01-01", "1980-07-31"),   # Volcker I
    ("1981-07-01", "1982-11-30"),   # Volcker II
    ("1987-10-01", "1988-01-31"),   # Black Monday
    ("1990-07-01", "1991-03-31"),   # Gulf War
    ("2001-03-01", "2001-11-30"),   # Dot-com
    ("2007-12-01", "2009-06-30"),   # Great Recession (full NBER window)
    ("2020-02-01", "2020-04-30"),   # COVID
]

STAGFLATION_WINDOWS = [
    # High inflation + weak growth episodes (CPI > 5% sustained)
    # Expanded from single window to include 1970s episodes —
    # essential for Brier resolution on rare-event regimes.
    ("1975-01-01", "1975-12-31"),   # Post oil embargo stagflation
    ("1979-06-01", "1980-07-31"),   # Second oil shock + Volcker
    ("2021-06-01", "2022-06-30"),   # Post-COVID inflation surge
]


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _date_in_windows(date_str: str, windows: list[tuple[str, str]]) -> bool:
    for start, end in windows:
        if start <= date_str <= end:
            return True
    return False


def get_ground_truth(date_str: str) -> str:
    """Return ground-truth regime label for a date."""
    if _date_in_windows(date_str, NBER_CONTRACTION_WINDOWS):
        return "contraction"
    if _date_in_windows(date_str, STAGFLATION_WINDOWS):
        return "stagflation"
    return "expansion"


# ── Brier Score Decomposition (Murphy 1973) ───────────────────────────

@dataclass
class BrierDecomposition:
    """Murphy (1973) decomposition: BS = reliability - resolution + uncertainty.

    Lower BS is better (0 = perfect, 1 = worst).
    reliability: lower is better (calibration — do 70% forecasts verify 70%?)
    resolution: higher is better (sharpness — can the model distinguish events?)
    uncertainty: fixed by the data (base rate of the event)
    """
    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    brier_skill_score: float  # BSS = 1 - BS/uncertainty; >0 = beats climatology
    n_bins: int
    n_obs: int
    regime: str


def brier_decomposition(
    forecasts: list[float],
    observations: list[int],
    regime: str,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Compute Murphy (1973) Brier Score decomposition.

    Args:
        forecasts: predicted probabilities p_t for the regime
        observations: binary indicators y_t (1 = regime active, 0 = not)
        regime: label for reporting
        n_bins: number of probability bins for decomposition

    Returns:
        BrierDecomposition with all components
    """
    f = np.array(forecasts)
    o = np.array(observations)
    n = len(f)

    # Raw Brier Score
    bs = float(np.mean((f - o) ** 2))

    # Base rate
    o_bar = float(np.mean(o))
    uncertainty = o_bar * (1.0 - o_bar)

    # Bin forecasts into n_bins equal-width bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for k in range(n_bins):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        if k == n_bins - 1:
            mask = (f >= lo) & (f <= hi)
        else:
            mask = (f >= lo) & (f < hi)

        n_k = int(np.sum(mask))
        if n_k == 0:
            continue

        f_bar_k = float(np.mean(f[mask]))
        o_bar_k = float(np.mean(o[mask]))

        reliability += n_k * (f_bar_k - o_bar_k) ** 2
        resolution += n_k * (o_bar_k - o_bar) ** 2

    reliability /= n
    resolution /= n

    # Brier Skill Score: BSS = 1 - BS/uncertainty
    # >0 means better than climatological forecast; <0 means worse
    bss = 1.0 - (bs / uncertainty) if uncertainty > 1e-8 else 0.0

    return BrierDecomposition(
        brier_score=round(bs, 6),
        reliability=round(reliability, 6),
        resolution=round(resolution, 6),
        uncertainty=round(uncertainty, 6),
        brier_skill_score=round(bss, 6),
        n_bins=n_bins,
        n_obs=n,
        regime=regime,
    )


# ── Ferro-Fricker (2012) Bias-Corrected Brier ────────────────────────

@dataclass
class FerroFrickerBrier:
    """Bias-corrected Brier decomposition (Ferro & Fricker 2012, QJRMS).

    The Murphy (1973) decomposition overestimates resolution in bins
    with few observations — critical for rare events like stagflation.
    Ferro & Fricker derive the O(1/n_k) bias and provide unbiased
    estimators for reliability and resolution individually.

    Note: The total Brier score is already unbiased (biases cancel in
    the sum). The correction matters for interpreting the components.
    """
    brier_score: float
    reliability: float           # bias-corrected
    resolution: float            # bias-corrected
    uncertainty: float
    reliability_uncorrected: float
    resolution_uncorrected: float
    bias_correction: float       # magnitude of the correction term
    n_bins: int
    n_obs: int
    regime: str


def ferro_fricker_brier(
    forecasts: list[float],
    observations: list[int],
    regime: str,
    n_bins: int = 10,
) -> FerroFrickerBrier:
    """Ferro & Fricker (2012) bias-corrected Brier decomposition.

    The within-bin bias for both reliability and resolution is:
        bias_k = o_bar_k * (1 - o_bar_k) / (n_k - 1)

    Corrected estimators:
        reliability_bc = (1/N) sum_k [n_k(f_bar_k - o_bar_k)^2 - o_bar_k(1-o_bar_k)*n_k/(n_k-1)]
        resolution_bc  = resolution_murphy - same_term + o_bar(1-o_bar)/(N-1)

    Args:
        forecasts: predicted probabilities p_t
        observations: binary indicators y_t
        regime: label for reporting
        n_bins: number of equal-width probability bins
    """
    f = np.array(forecasts)
    o = np.array(observations)
    n = len(f)

    bs = float(np.mean((f - o) ** 2))
    o_bar = float(np.mean(o))
    uncertainty = o_bar * (1.0 - o_bar)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability_raw = 0.0
    resolution_raw = 0.0
    bias_term = 0.0

    for k in range(n_bins):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        if k == n_bins - 1:
            mask = (f >= lo) & (f <= hi)
        else:
            mask = (f >= lo) & (f < hi)

        n_k = int(np.sum(mask))
        if n_k == 0:
            continue

        f_bar_k = float(np.mean(f[mask]))
        o_bar_k = float(np.mean(o[mask]))

        reliability_raw += n_k * (f_bar_k - o_bar_k) ** 2
        resolution_raw += n_k * (o_bar_k - o_bar) ** 2

        # Ferro-Fricker bias: var(o|k) / (n_k - 1)
        if n_k > 1:
            bias_term += o_bar_k * (1.0 - o_bar_k) * n_k / (n_k - 1)

    reliability_raw /= n
    resolution_raw /= n
    bias_term /= n

    # Bias-corrected components
    reliability_bc = reliability_raw - bias_term
    # Resolution correction also includes the overall variance term
    overall_correction = o_bar * (1.0 - o_bar) / max(n - 1, 1)
    resolution_bc = resolution_raw - bias_term + overall_correction

    return FerroFrickerBrier(
        brier_score=round(bs, 6),
        reliability=round(max(reliability_bc, 0.0), 6),
        resolution=round(max(resolution_bc, 0.0), 6),
        uncertainty=round(uncertainty, 6),
        reliability_uncorrected=round(reliability_raw, 6),
        resolution_uncorrected=round(resolution_raw, 6),
        bias_correction=round(bias_term, 6),
        n_bins=n_bins,
        n_obs=n,
        regime=regime,
    )


# ── ROC / AUC (Berge & Jordà 2011) ───────────────────────────────────

@dataclass
class ROCResult:
    """ROC analysis for regime detection.

    AUC interpretation (Berge & Jordà 2011):
      > 0.9 = excellent
      > 0.8 = good
      > 0.7 = fair
      > 0.5 = no skill (random)
    """
    auc: float
    fpr: list[float]  # false positive rates at each threshold
    tpr: list[float]  # true positive rates at each threshold
    thresholds: list[float]
    optimal_threshold: float  # Youden's J maximizer
    tpr_at_optimal: float
    fpr_at_optimal: float
    regime: str


def roc_auc(
    forecasts: list[float],
    observations: list[int],
    regime: str,
    n_thresholds: int = 200,
) -> ROCResult:
    """Compute ROC curve and AUC for regime detection.

    Uses the trapezoidal rule for AUC (equivalent to Mann-Whitney U).

    Args:
        forecasts: predicted probabilities
        observations: binary indicators
        regime: label for reporting
        n_thresholds: number of threshold points
    """
    f = np.array(forecasts)
    o = np.array(observations, dtype=bool)

    n_pos = int(np.sum(o))
    n_neg = len(o) - n_pos

    if n_pos == 0 or n_neg == 0:
        return ROCResult(
            auc=0.5, fpr=[0.0, 1.0], tpr=[0.0, 1.0],
            thresholds=[1.0, 0.0], optimal_threshold=0.5,
            tpr_at_optimal=0.5, fpr_at_optimal=0.5, regime=regime,
        )

    thresholds = np.linspace(1.0, 0.0, n_thresholds)
    fpr_list = []
    tpr_list = []

    for thresh in thresholds:
        predicted_pos = f >= thresh
        tp = int(np.sum(predicted_pos & o))
        fp = int(np.sum(predicted_pos & ~o))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal AUC
    auc = float(np.trapz(tpr_list, fpr_list))

    # Youden's J statistic: maximize TPR - FPR
    j_scores = np.array(tpr_list) - np.array(fpr_list)
    best_idx = int(np.argmax(j_scores))
    optimal_thresh = float(thresholds[best_idx])

    return ROCResult(
        auc=round(abs(auc), 4),
        fpr=[round(x, 4) for x in fpr_list],
        tpr=[round(x, 4) for x in tpr_list],
        thresholds=[round(float(x), 4) for x in thresholds],
        optimal_threshold=round(optimal_thresh, 4),
        tpr_at_optimal=round(tpr_list[best_idx], 4),
        fpr_at_optimal=round(fpr_list[best_idx], 4),
        regime=regime,
    )


# ── Diebold-Mariano Test (1995) ──────────────────────────────────────

@dataclass
class DieboldMarianoResult:
    """Diebold-Mariano test for comparing two forecast models.

    Null hypothesis: both models have equal predictive accuracy.
    Negative DM statistic + small p-value = model A is significantly
    better than model B (lower loss).
    """
    dm_statistic: float
    p_value: float
    mean_loss_diff: float
    model_a: str
    model_b: str
    n_obs: int
    verdict: str  # "A better", "B better", "no significant difference"


def diebold_mariano(
    loss_a: list[float],
    loss_b: list[float],
    model_a: str = "IMM",
    model_b: str = "Hamilton",
    h: int = 1,
) -> DieboldMarianoResult:
    """Diebold-Mariano test comparing two loss series.

    Uses the Harvey, Leybourne & Newbold (1997) small-sample correction.

    Args:
        loss_a: per-observation loss for model A (e.g. negative log-lik)
        loss_b: per-observation loss for model B
        model_a: label
        model_b: label
        h: forecast horizon (1 for contemporaneous)
    """
    d = np.array(loss_a) - np.array(loss_b)
    n = len(d)
    d_bar = float(np.mean(d))

    # Autocovariance at lags 0..h-1 (Newey-West style)
    gamma_0 = float(np.var(d, ddof=1))
    V = gamma_0
    for k in range(1, h):
        gamma_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        V += 2 * gamma_k

    se = np.sqrt(V / n)
    if se < 1e-12:
        return DieboldMarianoResult(
            dm_statistic=0.0, p_value=1.0, mean_loss_diff=d_bar,
            model_a=model_a, model_b=model_b, n_obs=n,
            verdict="no significant difference",
        )

    dm_raw = d_bar / se

    # Harvey-Leybourne-Newbold (1997) small-sample correction
    hln_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = dm_raw * hln_correction

    # Two-sided p-value from standard normal
    from scipy.stats import norm
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))

    if p_value < 0.05:
        verdict = f"{model_a} better" if dm_stat < 0 else f"{model_b} better"
    else:
        verdict = "no significant difference"

    return DieboldMarianoResult(
        dm_statistic=round(float(dm_stat), 4),
        p_value=round(float(p_value), 6),
        mean_loss_diff=round(d_bar, 6),
        model_a=model_a,
        model_b=model_b,
        n_obs=n,
        verdict=verdict,
    )


# ── PIT Histogram (Diebold, Gunther & Tay 1998) ──────────────────────

@dataclass
class PITResult:
    """Probability Integral Transform uniformity test.

    If the model's density forecasts are correct, the PIT values
    z_t = Phi(innovation_t / sqrt(S_t)) should be Uniform(0,1).

    We test uniformity via chi-squared goodness of fit.
    A flat histogram = well-calibrated density.
    """
    chi2_statistic: float
    p_value: float
    n_bins: int
    bin_counts: list[int]
    expected_count: float
    n_obs: int
    is_uniform: bool  # p > 0.05


def pit_test(
    innovations: list[float],
    variances: list[float],
    n_bins: int = 10,
) -> PITResult:
    """Compute PIT histogram and uniformity test.

    Args:
        innovations: raw innovations (y - Hx)
        variances: innovation variances S_t = H P H' + R
        n_bins: number of histogram bins
    """
    from scipy.stats import norm as norm_dist, chi2

    z = np.array(innovations)
    s = np.sqrt(np.maximum(np.array(variances), 1e-12))
    standardized = z / s

    # PIT: Phi(standardized innovation)
    pit_values = norm_dist.cdf(standardized)
    n = len(pit_values)

    # Bin into n_bins equal-width bins on [0, 1]
    bin_counts, _ = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))
    expected = n / n_bins

    # Chi-squared goodness of fit against Uniform
    chi2_stat = float(np.sum((bin_counts - expected) ** 2 / expected))
    p_val = 1.0 - chi2.cdf(chi2_stat, df=n_bins - 1)

    return PITResult(
        chi2_statistic=round(chi2_stat, 4),
        p_value=round(float(p_val), 4),
        n_bins=n_bins,
        bin_counts=[int(c) for c in bin_counts],
        expected_count=round(expected, 1),
        n_obs=n,
        is_uniform=float(p_val) > 0.05,
    )


# ── Berkowitz (2001) PIT LR Test ──────────────────────────────────────

@dataclass
class BerkowitzResult:
    """Berkowitz (2001) inverse-normal PIT likelihood ratio test.

    Transforms PIT values u_t = F(y_t) via Φ^{-1} to z_t. Under correct
    specification, z_t ~ i.i.d. N(0,1). The LR test jointly tests:
        H0: μ = 0, σ² = 1, ρ = 0
    against the AR(1) alternative z_t = μ + ρ(z_{t-1} - μ) + σε_t.

    Advantages over the chi-squared PIT histogram test (Diebold et al. 1998):
      - Better small-sample power (parametric vs nonparametric)
      - Detects both miscalibration (μ≠0, σ²≠1) and independence
        violations (ρ≠0) in a single test
    """
    lr_statistic: float
    p_value: float        # χ²(3) under H0
    mu_hat: float         # estimated mean (0 if correct)
    sigma_hat: float      # estimated std (1.0 if correct)
    rho_hat: float        # estimated AR(1) coefficient (0 if correct)
    log_lik_restricted: float
    log_lik_unrestricted: float
    n_obs: int
    is_correct: bool      # p > 0.05


def berkowitz_pit_test(
    innovations: list[float],
    variances: list[float],
) -> BerkowitzResult:
    """Berkowitz (2001) inverse-normal PIT + likelihood ratio test.

    More powerful than chi-squared histogram test for small samples.
    Detects miscalibration AND serial dependence in one test.

    Args:
        innovations: raw innovations (y - Hx)
        variances: innovation variances S_t = H P H' + R
    """
    from scipy.stats import norm as norm_dist, chi2

    z_raw = np.array(innovations)
    s = np.sqrt(np.maximum(np.array(variances), 1e-12))
    standardized = z_raw / s

    # PIT: u_t = Φ(standardized)
    pit_values = norm_dist.cdf(standardized)

    # Clip to avoid infinities at boundaries
    pit_clipped = np.clip(pit_values, 1e-6, 1.0 - 1e-6)

    # Inverse normal transform: z_t = Φ^{-1}(u_t)
    z = norm_dist.ppf(pit_clipped)
    n = len(z)

    # Restricted log-likelihood: z ~ i.i.d. N(0,1)
    ll_restricted = float(np.sum(norm_dist.logpdf(z)))

    # Unrestricted: AR(1) model z_t = μ + ρ(z_{t-1} - μ) + σε_t
    # Estimate parameters via conditional MLE
    z_lag = z[:-1]
    z_now = z[1:]
    n_eff = len(z_now)

    mu_hat = float(np.mean(z))

    # AR(1) coefficient via OLS: regress (z_t - μ) on (z_{t-1} - μ)
    z_dm = z_now - mu_hat
    z_lag_dm = z_lag - mu_hat

    denom = float(np.sum(z_lag_dm ** 2))
    if denom > 1e-12:
        rho_hat = float(np.sum(z_dm * z_lag_dm) / denom)
    else:
        rho_hat = 0.0

    # Residual variance
    residuals = z_dm - rho_hat * z_lag_dm
    sigma_hat = float(np.sqrt(np.mean(residuals ** 2)))
    sigma_hat = max(sigma_hat, 1e-6)

    # Unrestricted conditional log-likelihood (conditions on z_0)
    ll_unrestricted = float(np.sum(
        norm_dist.logpdf(residuals / sigma_hat) - np.log(sigma_hat)
    ))

    # LR statistic: -2(L_R - L_U) ~ χ²(3)
    # Use restricted LL on same observations (z_1,...,z_T conditioned on z_0)
    ll_restricted_cond = float(np.sum(norm_dist.logpdf(z_now)))
    lr_stat = -2.0 * (ll_restricted_cond - ll_unrestricted)
    lr_stat = max(lr_stat, 0.0)

    p_val = 1.0 - chi2.cdf(lr_stat, df=3)

    return BerkowitzResult(
        lr_statistic=round(lr_stat, 4),
        p_value=round(float(p_val), 4),
        mu_hat=round(mu_hat, 4),
        sigma_hat=round(sigma_hat, 4),
        rho_hat=round(rho_hat, 4),
        log_lik_restricted=round(ll_restricted_cond, 2),
        log_lik_unrestricted=round(ll_unrestricted, 2),
        n_obs=n,
        is_correct=float(p_val) > 0.05,
    )


# ── Sharpness (Gneiting et al. 2007) ─────────────────────────────────

@dataclass
class SharpnessResult:
    """Measures how decisive the regime probabilities are.

    Perfect sharpness = 1.0 (always 100% confident in one regime).
    Minimum sharpness = 1/K (always uniform across K regimes).
    Higher is better, BUT only meaningful when paired with calibration.
    """
    mean_max_prob: float
    std_max_prob: float
    frac_above_80: float   # fraction of timesteps where max(p) > 0.8
    frac_above_50: float   # fraction where max(p) > 0.5
    n_regimes: int
    n_obs: int


def sharpness(
    probability_series: list[dict[str, float]],
) -> SharpnessResult:
    """Compute sharpness metrics from a time series of regime probabilities.

    Args:
        probability_series: list of {regime_id: probability} dicts
    """
    max_probs = []
    for probs in probability_series:
        if probs:
            max_probs.append(max(probs.values()))

    max_probs = np.array(max_probs)
    n = len(max_probs)
    n_regimes = len(probability_series[0]) if probability_series else 3

    return SharpnessResult(
        mean_max_prob=round(float(np.mean(max_probs)), 4),
        std_max_prob=round(float(np.std(max_probs)), 4),
        frac_above_80=round(float(np.mean(max_probs > 0.8)), 4),
        frac_above_50=round(float(np.mean(max_probs > 0.5)), 4),
        n_regimes=n_regimes,
        n_obs=n,
    )


# ── Detection Lag ─────────────────────────────────────────────────────

@dataclass
class DetectionLagResult:
    """How quickly does the detector identify regime changes?

    Two thresholds are reported:
      - detection_lag_weeks: first P(regime) > 0.5 (early warning)
      - detection_lag_weeks_80: first P(regime) > 0.8 (Chauvet-Piger standard)

    The regime-switching literature (Chauvet & Hamilton 2006,
    Chauvet & Piger 2008) uses 80% as the standard threshold for
    declaring a regime shift. The 50% threshold is an early-warning
    measure with higher false positive risk.
    """
    event: str
    expected_regime: str
    detection_lag_weeks: float | None       # threshold = 0.5
    detection_lag_weeks_80: float | None    # threshold = 0.8 (Chauvet-Piger)
    peak_probability: float
    mean_probability: float


def detection_lag(
    daily_log: list[dict],
    event_start: str,
    event_end: str,
    expected_regime: str,
    event_name: str,
) -> DetectionLagResult:
    """Compute detection lag for a single regime event.

    Reports lag at both 50% (early warning) and 80% (Chauvet-Piger
    standard) thresholds.

    Args:
        daily_log: list of {"date": ..., "probabilities": {...}} entries
        event_start: NBER start date
        event_end: NBER end date
        expected_regime: regime key in probabilities dict
        event_name: human-readable description
    """
    window = [
        e for e in daily_log
        if event_start <= e["date"] <= event_end
    ]

    if not window:
        return DetectionLagResult(
            event=event_name, expected_regime=expected_regime,
            detection_lag_weeks=None, detection_lag_weeks_80=None,
            peak_probability=0.0, mean_probability=0.0,
        )

    probs = [e["probabilities"].get(expected_regime, 0.0) for e in window]

    def _find_crossing(threshold: float) -> float | None:
        for entry in window:
            p = entry["probabilities"].get(expected_regime, 0.0)
            if p >= threshold:
                start_dt = _parse_date(event_start)
                detect_dt = _parse_date(entry["date"])
                return (detect_dt - start_dt).days / 7.0
        return None

    lag_50 = _find_crossing(0.5)
    lag_80 = _find_crossing(0.8)

    return DetectionLagResult(
        event=event_name,
        expected_regime=expected_regime,
        detection_lag_weeks=round(lag_50, 1) if lag_50 is not None else None,
        detection_lag_weeks_80=round(lag_80, 1) if lag_80 is not None else None,
        peak_probability=round(float(max(probs)), 4),
        mean_probability=round(float(np.mean(probs)), 4),
    )


# ── Convenience: run full diagnostic suite ────────────────────────────

def run_full_diagnostics(
    daily_log: list[dict],
    regime_checkpoints: list[tuple],
    prob_key_map: dict[str, str] | None = None,
) -> dict:
    """Run all regime diagnostics on a backtest's daily_log.

    Args:
        daily_log: list of {"date", "probabilities": {regime: prob}} dicts
        regime_checkpoints: list of (start, end, regime, min_prob, desc) tuples
        prob_key_map: optional mapping from checkpoint regime keys to
                      probability dict keys (e.g. {"recession": "recession"})

    Returns:
        dict with all diagnostic results
    """
    if prob_key_map is None:
        prob_key_map = {}

    results = {}

    # --- Brier Score per regime ---
    regime_labels = ["expansion", "contraction", "stagflation"]
    checkpoint_regime_to_label = {
        "soft_landing": "expansion",
        "recession": "contraction",
        "stagflation": "stagflation",
    }

    prob_key_for_regime = {
        "expansion": "soft_landing",
        "contraction": "recession",
        "stagflation": "stagflation",
    }
    prob_key_for_regime.update(prob_key_map)

    brier_results = {}
    for regime in regime_labels:
        prob_key = prob_key_for_regime.get(regime, regime)
        forecasts = []
        observations = []
        for entry in daily_log:
            date_str = entry["date"]
            gt = get_ground_truth(date_str)
            p = entry["probabilities"].get(prob_key, 0.0)
            forecasts.append(p)
            observations.append(1 if gt == regime else 0)

        if forecasts:
            brier_results[regime] = brier_decomposition(
                forecasts, observations, regime,
            )

    results["brier"] = brier_results

    # --- Ferro-Fricker (2012) bias-corrected Brier per regime ---
    ff_results = {}
    for regime in regime_labels:
        prob_key = prob_key_for_regime.get(regime, regime)
        forecasts = []
        observations = []
        for entry in daily_log:
            gt = get_ground_truth(entry["date"])
            forecasts.append(entry["probabilities"].get(prob_key, 0.0))
            observations.append(1 if gt == regime else 0)
        if forecasts:
            ff_results[regime] = ferro_fricker_brier(
                forecasts, observations, regime,
            )
    results["ferro_fricker"] = ff_results

    # --- ROC/AUC per regime ---
    roc_results = {}
    for regime in regime_labels:
        prob_key = prob_key_for_regime.get(regime, regime)
        forecasts = []
        observations = []
        for entry in daily_log:
            gt = get_ground_truth(entry["date"])
            forecasts.append(entry["probabilities"].get(prob_key, 0.0))
            observations.append(1 if gt == regime else 0)

        if forecasts:
            roc_results[regime] = roc_auc(forecasts, observations, regime)

    results["roc"] = roc_results

    # --- Sharpness ---
    prob_series = [entry["probabilities"] for entry in daily_log]
    results["sharpness"] = sharpness(prob_series)

    # --- Detection lag per checkpoint ---
    lag_results = []
    for start, end, regime, _min_prob, desc in regime_checkpoints:
        prob_key = prob_key_for_regime.get(
            checkpoint_regime_to_label.get(regime, regime),
            regime,
        )
        lag_results.append(detection_lag(
            daily_log, start, end, prob_key, desc,
        ))
    results["detection_lag"] = lag_results

    return results
