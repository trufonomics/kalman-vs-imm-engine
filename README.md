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
heimdall/                          # Core engine (5,549 lines)
├── kalman_filter.py               # 8-factor state estimator (487 lines)
├── imm_tracker.py                 # IMM + VS-IMM hierarchical tracker (2,433 lines)
├── stream_pipeline.py             # TRUF stream normalization + ingestion (704 lines)
├── kalman_bridge.py               # LLM qualitative → Kalman pseudo-observations (438 lines)
├── trigger_service.py             # Threshold triggers + early warning (692 lines)
├── calibration_service.py         # Brier score calibration tracking (294 lines)
└── adaptive_calibration.py        # PCA/autocorrelation parameter estimation (500 lines)

backtests/                         # Historical validation
├── full_history_backtest.py       # 51-year replay, 11 regime checkpoints (1975-2026)
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
    └── sub_regime_adjustments.json       # Level B deltas
```

## Key Results

### 51-Year Backtest (1975-2026)
- **11/11 regime checkpoints correctly identified** (Volcker I/II, Black Monday, Gulf War, Dot-com, GFC, COVID, 2021 inflation surge, 3 expansions)
- **20.6% false positive rate** (down from 30.4% after expansion signature fix)
- Streams available change over time: 10 in 1977, 15 by 2014

### Hamilton Benchmark
- Heimdall detects regime shifts **faster** than Hamilton's (1989) Markov-switching regression
- Hamilton uses batch-fitted or rolling-window; Heimdall is online/streaming (no future data leakage)
- Key metric: detection lag (weeks to >50% probability on correct regime)

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
- Beer, S. (1972). *Brain of the Firm*. Allen Lane.

## License

Proprietary. Internal use only.
