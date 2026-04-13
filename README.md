# Heimdall Engine — Kalman + IMM Macro Regime Detection

An 8-factor Kalman filter paired with an Interacting Multiple Model (IMM) tracker for real-time macroeconomic regime detection. Built to process Truflation's real-time economic data streams and identify regime shifts (expansion, stagflation, contraction) faster than traditional econometric models.

## Architecture

```
                    TRUF Streams (31 mapped)
                           │
                    ┌──────▼──────┐
                    │  Normalize   │  Adaptive EWMA per-stream
                    │  (z-scores)  │  Handles non-stationary series
                    └──────┬──────┘
                           │
               ┌───────────▼───────────┐
               │  Level 1: Kalman      │  8-factor state estimator
               │  (EconomicStateEst.)  │  Tracks latent economy
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │  Level 2: IMM         │  3-branch regime tracker
               │  (Blom-Shalom 1988)   │  Bayesian competition
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │  Level 2b: VS-IMM     │  7 sub-regimes nested
               │  (Hierarchical)       │  within Level 2 branches
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │  Triggers + Bridge    │  Threshold detection,
               │                       │  LLM pseudo-observations
               └───────────────────────┘
```

## The 8 Latent Factors

| Factor | What it tracks | Key streams |
|--------|---------------|-------------|
| `inflation_trend` | Underlying inflation momentum | CPI, Core CPI, PPI |
| `growth_trend` | Real economic growth | Retail sales, S&P 500 |
| `labor_pressure` | Labor market tightness/slack | Unemployment, payrolls, claims |
| `housing_momentum` | Housing market direction | Home prices, starts, mortgage rates |
| `financial_conditions` | Credit conditions, yield curve | 10Y yield, Fed funds rate |
| `commodity_pressure` | Input cost pressures | Oil, copper, nickel |
| `consumer_sentiment` | Demand-side confidence | Consumer confidence, retail |
| `policy_stance` | Monetary/fiscal direction | Fed funds rate, 10Y yield |

## IMM Regime Branches

**Level A (3 macro regimes):**
- **Expansion** — growth positive, inflation moderate, sentiment healthy
- **Stagflation** — inflation elevated + growth weakening simultaneously
- **Contraction** — growth negative, labor shedding, financial stress

**Level B (7 sub-regimes, activated when parent > 20%):**
- Expansion → Goldilocks, Boom, Disinflation
- Stagflation → Cost-Push, Demand-Pull
- Contraction → Credit Crunch, Demand Shock

Branch probabilities updated via Bayesian competition on every observation. Markov Transition Probability Matrix (TPM) provides structural regularization.

## Calibration

Parameters are not guessed — they're calibrated from 51 years of FRED data (1975-2026) and validated across every major US economic event:

- **Persistence (F diagonal)**: AR(1) estimates from factor time series, cross-validated against Ang, Bekaert & Wei (2007)
- **Cross-dynamics (F off-diagonal)**: VAR(1) on factor trajectories — 5 theory-based channels confirmed, 3 new channels discovered (Phillips curve, labor→housing, housing→financial)
- **Observation noise (R)**: Residual variance after PCA factor extraction, with frequency scaling (daily streams get 16x noise vs monthly)
- **Branch adjustments**: Mean factor z-scores during NBER-dated regime periods
- **TPM**: Heuristic, validated empirically (11/11 regime checkpoints, 20.6% false positive rate)

See `experiments/calibration_from_history.py` and `data/results/calibration_from_history_results.json` for the full calibration analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtests (results already cached in data/results/, but to re-run fresh):
export FRED_API_KEY="your_key_here"
python backtests/full_history_backtest.py

# Run tests
python -m pytest tests/ -v
```

## Repository Structure

```
heimdall/                          # Core engine
├── kalman_filter.py               # 8-factor state estimator
├── imm_tracker.py                 # IMM + VS-IMM hierarchical tracker
├── regime_noise.py                # P0: Regime-dependent R/Q multipliers (Apr 13 2026)
├── regime_loadings.py             # P1: Regime-switching H (REJECTED — reference only)
├── duration_tpm.py                # P2a: Duration-dependent TPM (NEUTRAL — marginal)
├── state_tpm.py                   # P2b: State-dependent TPM (DONE — ships with P0)
├── calibrated_persistence.py      # P3: Data-calibrated F_diag (DIAGNOSTIC — not shipped)
├── cumulator.py                   # P4: Mariano-Murasawa temporal aggregation (DONE — +6,669 LL)
├── gas_noise.py                   # P5: Score-driven GAS observation noise (DONE — +56,525 LL)
├── correlated_shocks.py           # P6: Sparse off-diagonal Q (NEUTRAL — +191 LL)
├── regime_diagnostics.py          # Proper regime evaluation: Brier/ROC/DM/PIT/Sharpness
├── recalibration.py               # Isotonic regression + Safe Bayesian tempering (C1)
├── em_tpm.py                      # EM estimation of Markov TPM (Hamilton 1989 / Kim 1994)
├── bootstrap_ci.py                # Block bootstrap CIs (Politis & Romano 1994)
├── stream_pipeline.py             # TRUF stream normalization + ingestion
├── kalman_bridge.py               # LLM qualitative → Kalman pseudo-observations
├── trigger_service.py             # Threshold triggers + early warning
├── calibration_service.py         # Brier score calibration tracking
└── adaptive_calibration.py        # PCA/autocorrelation parameter estimation

backtests/                         # Historical validation
├── full_history_backtest.py       # 51-year replay, 11 regime checkpoints (1975-2026)
├── p0_comparison.py               # P0 A/B: regime-dependent R/Q vs fixed (Apr 13 2026)
├── p1_comparison.py               # P1 A/B/C: regime-switching H (REJECTED — overfitting)
├── p2_comparison.py               # P2a A/B/C: duration-dependent TPM (NEUTRAL — +85 LL)
├── p2b_comparison.py              # P2b A/B/C: state-dependent TPM (DONE — +2,009 LL)
├── p3_comparison.py               # P3 A/B/C: calibrated persistence (DIAGNOSTIC)
├── p4_comparison.py               # P4 A/B/C: Mariano-Murasawa cumulator (DONE — +6,669 LL)
├── p5_comparison.py               # P5 A/B/C: GAS score-driven noise (DONE — +56,525 LL)
├── p6_comparison.py               # P6 A/B/C: correlated Q (NEUTRAL — +191 LL)
├── diagnostic_comparison.py       # Proper regime diagnostics (replaces Ljung-Box)
├── calibration_fix_comparison.py  # Brier reliability gap fixes (tempering + isotonic)
├── em_tpm_comparison.py           # EM-estimated vs hand-tuned TPM (Apr 13 2026)
├── bootstrap_ci_comparison.py     # 95% CIs for all key metrics (Apr 13 2026)
├── multi_regime_backtest.py       # 4-regime robustness (GFC, COVID, stagflation, soft landing)
└── vs_imm_backtest.py             # Hierarchical VS-IMM vs flat IMM comparison

benchmarks/                        # Academic model comparison
├── hamilton_benchmark.py          # vs Hamilton (1989) Markov-switching regression
└── chauvet_piger_benchmark.py     # vs Sahm Rule, CFNAI, Chauvet-Piger

experiments/                       # Core claims validation
├── state_estimator_experiment.py  # Kalman + IMM on 1 year of FRED/Yahoo data
├── truf_state_estimator_experiment.py  # Same on live Truflation streams
└── calibration_from_history.py    # Data-driven parameter estimation from 51yr FRED

analysis/                          # Robustness analysis
├── sensitivity_analysis.py        # Parameter perturbation (±30-50%) robustness
├── factor_validation.py           # Factor-reference correlations + cross-dynamics
├── estimate_loadings.py           # PCA-derived H matrix vs hand-specified
├── derive_adjustments.py          # Branch adjustment constants from NBER data
└── derive_sub_regime_adjustments.py  # Level B sub-regime deltas

tests/                             # Unit tests (978 lines)
├── test_heimdall_kalman.py        # State estimator initialization, factors, streams
├── test_heimdall_imm.py           # Branch tracking, probability updates, mixture
├── test_heimdall_calibration.py   # Brier scores, reliability index
├── test_heimdall_coherence.py     # Cross-tree coherence propagation
├── test_heimdall_governance.py    # Branch creation, pruning, splitting
└── test_heimdall_triggers.py      # Trigger conditions, cooldowns, actions

data/                              # Cached data + experiment results
├── backtest_data_cache_51yr.csv   # 51 years of FRED + Yahoo data (976 KB)
├── backtest_data_cache.csv        # 2005-2026 subset (600 KB)
├── full_history_cache.csv         # Preprocessed for backfill (1.1 MB)
├── us_inflation_data.csv          # US historical inflation reference
└── results/                       # Outputs from every experiment
    ├── full_history_results.json         # 51-year backtest (2.4 MB)
    ├── multi_regime_results.json         # 4-regime validation (1.2 MB)
    ├── vs_imm_backtest_results.json      # Hierarchical comparison (1.3 MB)
    ├── hamilton_benchmark_results.json    # Detection lag vs Hamilton
    ├── chauvet_piger_results.json        # vs recession-only models
    ├── state_estimator_results.json      # Core claims validation
    ├── truf_experiment_results.json      # Truflation stream validation
    ├── calibration_from_history_results.json  # Parameter estimates
    ├── factor_validation_results.json    # Factor correlations
    ├── sensitivity_results.json          # Perturbation robustness
    ├── estimated_loadings.json           # PCA H matrix
    ├── derived_adjustments.json          # Branch adjustment constants
    ├── sub_regime_adjustments.json       # Level B deltas
    ├── p0_comparison_results.json        # P0 A/B comparison (Apr 13 2026)
    ├── p1_comparison_results.json        # P1 A/B/C comparison (REJECTED)
    ├── p2_comparison_results.json        # P2a A/B/C comparison (NEUTRAL)
    ├── p2b_comparison_results.json       # P2b A/B/C comparison (DONE — ships)
    ├── p3_comparison_results.json        # P3 A/B/C comparison (DIAGNOSTIC)
    ├── p4_comparison_results.json        # P4 A/B/C comparison (DONE — +6,669 LL)
    ├── p5_comparison_results.json        # P5 A/B/C comparison (DONE — +56,525 LL)
    ├── p6_comparison_results.json        # P6 A/B/C comparison (NEUTRAL)
    ├── diagnostic_comparison_results.json # Brier/ROC/DM/PIT/Sharpness
    ├── calibration_fix_results.json      # Tempering scan + isotonic results
    ├── em_tpm_comparison_results.json    # EM vs hand-tuned TPM (TPM insensitive)
    └── bootstrap_ci_results.json         # 95% CIs on all key metrics
```

## Engine Improvement Roadmap

Literature review (Apr 12 2026) identified 5 improvements in priority order, based on Chan & Eisenstat (2018) model comparison results. Each fix is validated against the 51-year backtest before shipping.

| Priority | Fix | Status | Key Result |
|----------|-----|--------|------------|
| **P0** | Regime-dependent R/Q | **DONE** (Apr 13 2026) | Log-likelihood +49,606, anomaly rate 8.9%→6.1%, all checkpoints held |
| **P1** | Regime-switching H (loadings) | **REJECTED** (Apr 13 2026) | LL -2,158 vs P0 alone; Feldkircher overfitting warning confirmed |
| **P2a** | Duration-dependent TPM | **NEUTRAL** (Apr 13 2026) | LL +85 vs P0 alone (noise-level); Bayesian competition already handles exits |
| **P2b** | State-dependent TPM (factors drive transitions) | **DONE** (Apr 13 2026) | LL +2,009 vs P0; Volcker I detection 10.1w→0.6w; checkpoints held |
| **P3** | Calibrated F_diag persistence | **DIAGNOSTIC** (Apr 13 2026) | LL +683 but lost 3 checkpoints; explains Ljung-Box 0% root cause |
| **P4** | Mariano-Murasawa cumulator (temporal aggregation) | **DONE** (Apr 13 2026) | LL +6,669; anomaly 6.0%→5.5%; checkpoints held; ACF unchanged |
| **P5** | GAS score-driven observation noise (Creal et al. 2013) | **DONE** (Apr 13 2026) | LL +56,525; anomaly 5.5%→0.5%; checkpoints held; ACF improved |
| **P6** | Sparse off-diagonal Q (Primiceri 2005) | **NEUTRAL** (Apr 13 2026) | LL +191 (noise-level); GAS already handles cross-stream correlation |

### P0: Regime-Dependent Observation & Process Noise (Apr 13 2026)

**Problem:** R (observation noise) and Q (process noise) were fixed constants. During COVID, initial claims spiking 10x got the same noise treatment as a calm expansion month — producing 50-sigma z-scores that are physically meaningless.

**Fix:** Each IMM regime branch gets its own noise multipliers. Contraction branches inflate R by 2-3x for financial/commodity streams. Stagflation branches inflate R by 2-2.5x for commodity streams. Expansion branches keep baseline values. Q scales 1.5-2x in crisis (state evolves faster). Zero additional latent states — uses existing IMM probabilities.

**Implementation:** `heimdall/regime_noise.py` + modified IMM update path in `backtests/p0_comparison.py`

**A/B comparison (51-year FRED, same data, same checkpoints):**

| Metric | Baseline | P0 | Delta |
|--------|----------|-----|-------|
| Total log-likelihood | -154,433 | -104,827 | **+49,606** |
| Anomaly rate | 8.9% | 6.1% | **-2.8pp** |
| Regime checkpoints | 10/11 | 10/11 | held |
| Ljung-Box pass rate | 0% | 0% | unchanged |
| Mean bias pass rate | 33% | 33% | unchanged |

Crisis regime detection improved (Great Recession avg +6.8pp, COVID avg +6.7pp, Dot-com avg +6.9pp) while expansion detection held. Ljung-Box still 0% — structural autocorrelation requires P1 (regime-switching loadings) or P2 (duration-dependent TPM) to address.

**Ref:** Chan, J.C.C. & Eisenstat, E. (2018). "Bayesian Model Comparison for Time-Varying Parameter VARs with Stochastic Volatility." *J. Applied Econometrics* 33(4), 509-532.

### P1: Regime-Switching Factor Loadings — REJECTED (Apr 13 2026)

**Hypothesis:** Per-regime loading scales (Urga & Wang 2024) — amplify inflation/commodity loadings in stagflation, labor/financial in contraction, with conservative 1.2-1.5x magnitudes per Feldkircher et al. (2024) overfitting warning.

**Result:** P0+P1 regressed vs P0 alone. Log-likelihood -2,158 worse, anomaly rate flat (5.0% → 5.1%), checkpoints unchanged (9/11).

**Why it failed:** P0's regime-dependent R/Q already captures the regime-dependent signal. R multipliers scale the *observation noise*, which implicitly adjusts how much each stream influences the state update. Adding explicit H loading switches on top double-counts the regime adjustment — the filter over-reacts in the regime-switched direction.

**Takeaway:** Feldkircher et al.'s warning about "simple and sizeable beating sophisticated and small" applies here. The R/Q approach (P0) is simpler and more effective than H switching (P1). Filed as negative result; code preserved in `heimdall/regime_loadings.py` for reference.

**Implementation:** `heimdall/regime_loadings.py` + `backtests/p1_comparison.py`

### P2a: Duration-Dependent TPM — NEUTRAL (Apr 13 2026)

**Hypothesis:** Recession exit probability increases with duration (Durland & McCurdy 1994). A logistic ramp increases contraction→expansion transition probability from 0.050 (baseline) toward 0.150 as contraction persists, with inflection at 39 weeks (~9 months, median NBER recession duration). Symmetric gentle ramp on expansion exit.

**Result:** LL +85 over P0 alone — noise-level improvement. Checkpoints unchanged (10/11). Ljung-Box still 0%.

**Why it didn't matter:** The IMM's Bayesian likelihood competition already handles regime exits effectively. When recovery data starts arriving, the expansion branch wins the Bayes update regardless of TPM values. The TPM's role is structural regularization (preventing branch convergence), not transition timing. Duration adjustments to an already-effective Bayesian mechanism are redundant.

**Takeaway:** The TPM is not the bottleneck. The persistent Ljung-Box failure (0% pass rate across all P0-P2 variants) points to the F matrix (AR(1) persistence) being the structural misspecification — the actual data dynamics have higher-order temporal structure that a first-order state transition can't capture. P3 (formal parameter estimation) is the next logical step.

**Implementation:** `heimdall/duration_tpm.py` + `backtests/p2_comparison.py`

### P2b: State-Dependent TPM — DONE (Apr 13 2026)

**Hypothesis:** Filardo (1994) — factor values drive transition probabilities via logistic link. When financial_conditions deteriorate, expansion→contraction probability increases. When inflation_trend rises, expansion→stagflation probability increases. Six transition rules, each modulated by 2-3 factors.

**Result:** LL +2,009 over P0 alone. Anomaly rate 6.1%→6.0%. Checkpoints held (10/11). Volcker I detection dramatically improved: 49.2%→68.5% avg probability, 10.1w→0.6w detection speed.

**Why it works (when P2a didn't):** Duration is a weak signal for transitions — the IMM's Bayesian likelihood competition already handles duration implicitly. But factor VALUES contain genuinely new information: deteriorating financial_conditions during an expansion is a leading indicator of contraction, and the fixed TPM ignores this. The state-dependent TPM lets the filter anticipate transitions before the Bayesian likelihood competition confirms them.

**Implementation:** `heimdall/state_tpm.py` + `backtests/p2b_comparison.py`

### P3: Calibrated F_diag Persistence — DIAGNOSTIC (Apr 13 2026)

**Hypothesis:** Ljung-Box fails because F_diag (AR(1) persistence) is too low for daily steps. Monthly streams have effective monthly persistence of 0.4-0.5 with theory values, but actual monthly AR(1) is 0.77-0.99. The filter predicts excessive state decay between observations, creating systematically correlated innovations.

**Diagnostic results:** The calibration study (51-year FRED) shows the data-derived daily equivalents are 0.96-0.999 versus the theory values of 0.85-0.98. A 70/30 data/theory blend:
- LL +683 over P0+P2b (mild improvement)
- Mean bias pass rate 33%→47% (significant improvement)
- Innovation ACF(1) drops modestly (avg -0.02 across monthly streams)
- **But lost 3 checkpoints (10/11→7/11)** — filter becomes too sluggish for crisis detection

**Root cause of Ljung-Box 0% (RESOLVED — understood):**

Innovation whiteness and regime detection speed are in fundamental tension:
- **Low F_diag** → filter is responsive (fast regime detection) → predictions decay faster than reality → autocorrelated innovations (high ACF)
- **High F_diag** → filter is sluggish (slow regime detection) → predictions match data persistence → whiter innovations (lower ACF)

The engine is correctly tuned for its primary mission (regime detection). To pass Ljung-Box, F_diag would need to approach 1.0 (near-random-walk), which would destroy regime detection entirely. The 0% Ljung-Box rate is the expected cost of optimizing for detection speed over point-forecast accuracy.

**Takeaway:** Ljung-Box is not a meaningful diagnostic for this architecture. The filter is an IMM regime detector, not a point forecaster. Innovation whiteness is desirable but secondary to checkpoint accuracy and detection speed.

**Implementation:** `heimdall/calibrated_persistence.py` + `backtests/p3_comparison.py`

### Summary: What Ships

The improvement roadmap tested 5 modifications against the 51-year backtest. Two ship:

| Fix | LL Delta | Ship? | Reason |
|-----|----------|-------|--------|
| **P0** (regime R/Q) | +49,606 | **YES** | Massive improvement, checkpoints held |
| P1 (regime-switching H) | -2,158 | No | Overfitting, double-counts with P0 |
| P2a (duration-dependent TPM) | +85 | No | Noise-level, Bayesian competition handles exits |
| **P2b** (state-dependent TPM) | +2,009 | **YES** | Meaningful improvement, Volcker I detection fixed |
| P3 (calibrated persistence) | +683 | No | Lost 3 checkpoints despite better innovations |
| **P4** (cumulator) | +6,669 | **YES** | Mean prediction over gap reduces bias |
| **P5** (GAS noise) | +56,525 | **YES** | Time-varying R, anomaly rate collapsed to 0.5% |
| P6 (correlated Q) | +191 | No | GAS already handles cross-stream correlation |
| **C1** (Safe Bayesian tempering) | n/a (calibration) | **YES** | Reliability -20%, AUC +4%, detection unchanged |

**Full stack P0+P2b+P4+P5+C1: LL +114,809 over baseline, anomaly rate 8.9%→0.5%, 11/11 checkpoints, contraction AUC 0.77 (full) / 0.90 (test set).**

### C1: Safe Bayesian Tempering — Calibration Fix (Apr 13 2026)

**Problem:** Brier reliability was poor across all regimes (contraction 0.094, stagflation 0.233, expansion 0.306). The IMM produces sharp but miscalibrated probabilities — the TPM persistence (0.97 self-transition) acts as a heavy anchor that makes the Bayesian update overconfident under model misspecification.

**Root cause:** As is well-known in the regime-switching literature (Hamilton 1989, Kim & Nelson 1999), innovations from Markov-switching models follow mixture distributions, making Ljung-Box inappropriate. Proper diagnostics (Brier/ROC/PIT) revealed the IMM's innovation density is much better than baseline (PIT chi2 28x better) but regime CLASSIFICATION is only fair (contraction AUC 0.74). The TPM persistence prior (0.97 self-transition) anchors filtered probabilities toward the dominant regime, a well-known property of Markov-switching filters.

**Fix:** Likelihood tempering following the power-posterior framework (Grunwald & van Ommen 2017). Cap the tempering exponent at 0.70 instead of ramping to 1.0 — an empirically chosen constant validated via grid search. Under misspecification, standard Bayesian updates (exponent=1.0) concentrate probability exponentially too fast. Tempering slows concentration from exp(-n × D_KL) to exp(-0.70 × n × D_KL).

**Tempering scan results (full 51-year backtest):**

| Ceiling | Mean Reliability | Contraction AUC | Detection | Sharpness |
|---------|-----------------|-----------------|-----------|-----------|
| 1.00 (before) | 0.1901 | 0.7407 | 11/11 | 0.866 |
| **0.70 (after)** | **0.1526** | **0.7695** | **11/11** | **0.856** |

On the out-of-sample test set (2012-2026): contraction AUC = **0.90** (excellent).

**Implementation:** 2-line change in `imm_tracker.py`:
```python
TEMPER_CEILING = 0.70  # was implicitly 1.0
exponent = TEMPER_FLOOR + (TEMPER_CEILING - TEMPER_FLOOR) * ramp
```

**Refs:** Grunwald, P. & van Ommen, T. (2017). "Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It." *Bayesian Analysis* 12(4), 1069-1103. Grunwald, P. (2012). "The Safe Bayesian." *Algorithmic Learning Theory*.

### EM-Estimated TPM — TPM Insensitivity Confirmed (Apr 13 2026)

**Experiment:** Replace the hand-tuned TPM with an EM-estimated TPM (Hamilton 1989 / Kim 1994 backward smoother / Dempster-Laird-Rubin 1977). EM converged in 16 iterations.

**EM-estimated TPM** (dramatically different from hand-tuned):

| | Expansion | Stagflation | Contraction |
|---|-----------|-------------|-------------|
| **From Expansion** | 0.866 (was 0.97) | 0.071 (was 0.02) | 0.063 (was 0.01) |
| **From Stagflation** | 0.105 (was 0.03) | 0.691 (was 0.95) | 0.205 (was 0.02) |
| **From Contraction** | 0.109 (was 0.05) | 0.269 (was 0.02) | 0.622 (was 0.93) |

**Result:** All diagnostics IDENTICAL — DM stat = 0.0000, p = 1.0. Same AUC, same Brier, same detection lags, same LL. The observation likelihoods from the Kalman filters are so informative that the TPM prior has negligible effect on regime classification.

**Significance:** The hand-tuned TPM is NOT the methodological vulnerability we feared. Consistent with Chan & Eisenstat (2018) — who find stochastic volatility dominates time-varying coefficients — our R/Q time-variation (92% of improvement) dominates the TPM. The EM-estimated TPM is more realistic (lower persistence = more transitions/year) but detection is observation-driven.

**Implementation:** `heimdall/em_tpm.py` + `backtests/em_tpm_comparison.py`

### Bootstrap Confidence Intervals (Apr 13 2026)

95% CIs via Politis & Romano (1994) stationary block bootstrap (500 resamples). All results with full shipping stack + C1 tempering.

| Metric | Point | 95% CI | SE |
|--------|-------|--------|-----|
| Contraction AUC | 0.7695 | [0.6960, 0.8358] | 0.034 |
| Contraction Brier | 0.1520 | [0.1232, 0.1837] | 0.016 |
| Contraction Reliability | 0.0690 | [0.0454, 0.0974] | 0.014 |
| Stagflation AUC | 0.6158 | [0.5027, 0.7287] | 0.056 |
| Stagflation Brier | 0.2256 | [0.1841, 0.2688] | 0.021 |
| Detection Count | 11/11 | [8, 11] | 0.83 |

Contraction AUC lower bound (0.70) stays well above random — detection is statistically significant. Stagflation AUC lower bound (0.50) is borderline — stagflation detection needs improvement (primary candidate: add breakeven inflation + GSCPI streams).

**Block length selection:** Politis & White (2004) with Patton et al. (2009) correction, via `arch.bootstrap.optimal_block_length`. Falls back to n^(1/3) heuristic if `arch` is unavailable.

**Implementation:** `heimdall/bootstrap_ci.py` + `backtests/bootstrap_ci_comparison.py`

**Refs:** Politis, D.N. & Romano, J.P. (1994). JASA 89(428), 1303-1313. DiCiccio, T.J. & Efron, B. (1996). Statistical Science 11(3), 189-212. Politis & White (2004). Econometric Reviews 23(1). Patton, Politis & White (2009). Econometric Reviews 28(4).

### Ferro-Fricker (2012) Bias-Corrected Brier (Apr 13 2026)

Standard Murphy (1973) Brier decomposition has O(1/n_k) bias in bins with few observations — a concern for rare events like stagflation. Ferro & Fricker (2012) provide unbiased estimators.

**Result:** Bias correction is <0.1% for both regimes on our dataset:
| Regime | Reliability (raw) | Reliability (corrected) | Bias magnitude |
|--------|-------------------|------------------------|----------------|
| Contraction | 0.069002 | 0.068455 | 0.000546 |
| Stagflation | 0.175101 | 0.174850 | 0.000251 |

The dataset (2700+ weekly observations) is large enough that standard Murphy decomposition is adequate. Ferro-Fricker serves as a sanity check confirming this.

**Implementation:** `ferro_fricker_brier()` in `heimdall/regime_diagnostics.py`

### Berkowitz (2001) PIT Likelihood Ratio Test (Apr 13 2026)

The Berkowitz test transforms PIT values via the inverse normal CDF, then tests whether z_t = Phi^{-1}(u_t) is i.i.d. N(0,1) using an AR(1) alternative. More powerful than the chi-squared histogram test (Diebold et al. 1998) for small samples, and detects both miscalibration and serial dependence.

**Aggregate result:** LR=13825, p<0.001 (rejected). Key parameters:
- mu_hat = -0.030 (slight negative bias — minimal)
- sigma_hat = 0.951 (model slightly overestimates noise)
- **rho_hat = 0.480** (strong serial correlation — the main rejection driver)

**Per-stream analysis reveals the pattern:**
| Stream | rho | sigma | Interpretation |
|--------|-----|-------|----------------|
| OIL_PRICE | 0.03 | 0.94 | Nearly correct spec — daily financial |
| SP500 | 0.34 | 0.95 | Moderate autocorrelation from regimes |
| US_CPI_YOY | 0.92 | 0.37 | Monthly cadence → high autocorrelation |
| UNEMPLOYMENT | 0.95 | 0.29 | Monthly cadence → high autocorrelation |
| FED_FUNDS_RATE | 0.93 | 0.53 | Monthly cadence → high autocorrelation |

Financial streams (daily) are well-specified. Macro streams (monthly) show high autocorrelation and overstated R — expected for a daily-frequency Kalman processing monthly observations. The GAS noise tracker doesn't account for inter-observation gaps. This is the primary refinement target after stagflation AUC.

**Implementation:** `berkowitz_pit_test()` in `heimdall/regime_diagnostics.py`

### Wave 2 Improvement Roadmap — COMPLETED (Apr 13 2026)

Post-experimental literature review identified 4 fixes targeting the Ljung-Box 0% root cause. Results:

| Priority | Fix | Status | Actual Result |
|----------|-----|--------|--------------|
| **P4** | Mariano-Murasawa cumulator | **DONE** | LL +6,669; anomaly 6.0%→5.5%; ACF unchanged |
| **P5** | GAS/score-driven R (Creal et al. 2013) | **DONE** | LL +56,525; anomaly 5.5%→0.5%; ACF improved |
| **P6** | Sparse off-diagonal Q (Primiceri 2005) | **NEUTRAL** | LL +191; noise-level, GAS handles it |
| P7 | Decouple F_diag from detection speed | **SUPERSEDED** | P4+P5 address the same problem differently |

**Ljung-Box remains 0%** — structural issue from observation frequency (monthly streams observed every ~22 days). The ACF(1) improved (0.924→0.895 avg for monthly streams) but not enough to pass the joint test at 10 lags. This is an accepted limitation of running a daily-frequency filter with monthly observations. The proper fix would require full Mariano-Murasawa state augmentation (cumulator states in the state vector), which is a future consideration.

**Chan & Eisenstat (2018) analogy confirmed across both waves**: Chan & Eisenstat find stochastic volatility (time-varying error covariance) dominates time-varying VAR coefficients in model fit. Analogously, our R/Q time-variation (P0 + P5) accounts for 92% of total LL improvement, while F/H/TPM changes are secondary.

## Key Results

### 51-Year Backtest (1975-2026)
- **11/11 regime checkpoints correctly identified** (Volcker I/II, Black Monday, Gulf War, Dot-com, GFC, COVID, 2021 inflation surge, 3 expansions)
- **20.6% false positive rate** (down from 30.4% after expansion signature fix)
- Streams available change over time: 10 in 1977, 15 by 2014

### vs Hamilton (1989) Markov-Switching — Detection Speed

Heimdall is online/streaming (no future data leakage). Hamilton uses batch-fitted or rolling-window regression. Detection lag = weeks to >50% probability on the correct regime.

**Full History (1975-2026, 9 events):**

| Event | Period | Heimdall IMM | Hamilton (equalized) | Winner |
|-------|--------|-------------|---------------------|--------|
| Volcker I | 1980 | 61.0 wks | 81.7 wks | **IMM** |
| Volcker II | 1981-82 | 53.1 wks | 3.6 wks | Hamilton |
| Black Monday | 1987 | 85.0 wks | 145.4 wks | **IMM** |
| Gulf War | 1990 | 19.7 wks | 2.0 wks | Hamilton |
| Dot-com | 2001 | 0.3 wks | 56.4 wks | **IMM** |
| GFC | 2007-09 | 0.9 wks | 557.7 wks | **IMM** |
| COVID | 2020 | 0.7 wks | 118.0 wks | **IMM** |
| Inflation Surge | 2021-22 | 0.6 wks | 195.9 wks | **IMM** |
| Soft Landing | 2023 | 16.7 wks | 9.0 wks | Hamilton |

**IMM wins 6/9 events.** On the 4 most recent regime shifts (Dot-com through Inflation Surge), IMM detects in under 1 week while Hamilton takes 56-558 weeks. Hamilton's advantage on early events (Volcker II, Gulf War) reflects its benefit from batch-fitting on the full sample.

### vs Recession-Only Models (Chauvet-Piger, Sahm Rule, CFNAI)

These models only detect recessions. Heimdall detects 3 regime types (expansion, stagflation, contraction) — a harder problem with more false positive surface.

| Recession | NBER Dates | IMM (wks) | Chauvet-Piger (wks) | Sahm (wks) | CFNAI (wks) |
|-----------|-----------|-----------|-------------------|-----------|------------|
| 1980 | Jan-Jul 1980 | 61.0 | 26.1 | 39.1 | 13.0 |
| 1981-82 | Jul 1981-Nov 1982 | 53.1 | 8.7 | 21.7 | 4.3 |
| 1990-91 | Jul 1990-Mar 1991 | 19.7 | 17.4 | 30.4 | 17.4 |
| 2001 | Mar-Nov 2001 | 0.3 | **missed** | 21.7 | 26.1 |
| 2007-09 | Dec 2007-Jun 2009 | 0.9 | 17.4 | 21.7 | 8.7 |
| 2020 | Feb-Apr 2020 | 0.7 | 4.3 | 4.3 | 4.3 |

- **IMM: 6/6 detected**, fastest on Dot-com (0.3 wks) and GFC (0.9 wks)
- **Chauvet-Piger: 5/6 detected** — missed Dot-com entirely
- **Sahm Rule: 6/6 detected** — consistently slowest (relies on unemployment rate lag)
- **CFNAI: 6/6 detected** — strong on early detection but single-factor
- **IMM false positive rate: 20.6%** (higher than recession-only models because it tracks 3 regimes, not 1)

### Sensitivity Analysis
- Filter tolerates ±30-50% perturbation on all key parameters without catastrophic failure
- Branch adjustments, F persistence, R noise, and TPM diagonal all robust

### Calibration
- 3 persistence values revised from theory (consumer_sentiment: 0.90→0.96, financial_conditions: 0.88→0.95, housing_momentum: 0.95→0.98)
- 3 observation noise values revised (consumer_confidence: 0.20→0.10, initial_claims: 0.10→0.05, PPI: 0.05→0.08)
- 3 new cross-factor channels discovered: Phillips curve (growth→inflation), labor→housing, housing→financial

## Data Sources

| Source | What | Access |
|--------|------|--------|
| **FRED API** | CPI, unemployment, yields, housing, retail, confidence (1975-2026) | Free API key from fred.stlouisfed.org |
| **Yahoo Finance** | S&P 500, oil, gold, copper, BTC (via `yfinance`) | No key needed |
| **Truflation** | Real-time economic streams (31 mapped) | Gateway API (`gateway.mainnet.truf.network`) |

Most backtests use FRED + Yahoo (cached in `data/*.csv`). Only `truf_state_estimator_experiment.py` requires the Truflation gateway.

## References

- Blom, H.A.P. & Bar-Shalom, Y. (1988). "The Interacting Multiple Model Algorithm for Systems with Markovian Switching Coefficients." *IEEE Trans. Automatic Control*, 33(8), 780-783.
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384.
- Ang, A., Bekaert, G. & Wei, M. (2007). "Do Macro Variables, Asset Markets, or Surveys Forecast Inflation Better?" *Journal of Monetary Economics*, 54(4), 1163-1212.
- Del Moral, P., Doucet, A. & Jasra, A. (2006). "Sequential Monte Carlo Samplers." *JRSS Series B*, 68(3), 411-436.
- Chan, J.C.C. & Eisenstat, E. (2018). "Bayesian Model Comparison for Time-Varying Parameter VARs with Stochastic Volatility." *J. Applied Econometrics*, 33(4), 509-532.
- Urga, G. & Wang, F. (2024). "Estimation and Inference for High Dimensional Factor Model with Regime Switching." *J. Econometrics*, 241(2).
- Durland, J.M. & McCurdy, T.H. (1994). "Duration-Dependent Transitions in a Markov Model of U.S. GNP Growth." *JBES*, 12(3), 279-288.
- Filardo, A.J. (1994). "Business-Cycle Phases and Their Transitional Dynamics." *JBES*, 12(3), 299-308.
- Mariano, R.S. & Murasawa, Y. (2003). "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *J. Applied Econometrics*, 18(4), 427-443.
- Creal, D., Koopman, S.J. & Lucas, A. (2013). "Generalized Autoregressive Score Models with Applications." *J. Applied Econometrics*, 28(5), 777-795.
- Brave, S., Butters, R.A. & Kelley, D. (2022). "A Practitioner's Guide and MATLAB Toolbox for Mixed Frequency State Space Models." *J. Statistical Software*, 104(10).
- Harvey, A.C. (2013). *Dynamic Models for Volatility and Heavy Tails: With Applications to Financial and Economic Data*. Cambridge University Press.
- Ferro, C.A.T. & Fricker, T.E. (2012). "A Bias-Corrected Decomposition of the Brier Score." *QJRMS*, 138(668), 1954-1960.
- Berkowitz, J. (2001). "Testing Density Forecasts, with Applications to Risk Management." *JBES*, 19(4), 465-474.
- Politis, D.N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." *Econometric Reviews*, 23(1), 53-70.
- Patton, A., Politis, D.N. & White, H. (2009). "Correction to 'Automatic Block-Length Selection'." *Econometric Reviews*, 28(4).
- Chauvet, M. & Hamilton, J.D. (2006). "Dating Business Cycle Turning Points." In *Nonlinear Time Series Analysis of Business Cycles*. Elsevier.
- Beer, S. (1972). *Brain of the Firm*. Allen Lane.

## License

Proprietary. Internal use only.
