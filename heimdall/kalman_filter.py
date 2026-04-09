"""
Level 1: Economic State Estimator — 8-factor Kalman filter.

Estimates the hidden economic state from noisy TRUF stream observations.
The economy has a true state we never observe directly. TRUF streams are
noisy windows into it. This filter tracks 8 latent factors:

  1. inflation_trend      — Underlying inflation momentum
  2. growth_trend         — Real economic growth momentum
  3. labor_pressure       — Tightness/slack in labor market
  4. housing_momentum     — Housing market direction and strength
  5. financial_conditions — Credit conditions, yield curve, spreads
  6. commodity_pressure   — Input cost pressures
  7. consumer_sentiment   — Demand-side confidence
  8. policy_stance        — Monetary/fiscal policy direction

The filter runs the standard Kalman predict/update cycle:
  x_t = F @ x_{t-1} + w_t       (state transition)
  z_t = H @ x_t + v_t           (observation)

Ref: Implementation Architecture doc (Level 1 section)
     Factor loading table, persistence parameters, R values
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Factor names (indices into the state vector)
FACTORS = [
    "inflation_trend",
    "growth_trend",
    "labor_pressure",
    "housing_momentum",
    "financial_conditions",
    "commodity_pressure",
    "consumer_sentiment",
    "policy_stance",
]
N_FACTORS = len(FACTORS)
FACTOR_INDEX = {name: i for i, name in enumerate(FACTORS)}

# TIME STEP: F and Q are calibrated for DAILY updates.
# predict() advances state by one day. Called once per calendar day.
# On days with no observations, state decays toward zero (mean-reverting).
#
# CALIBRATION SOURCE: 51 years of FRED data (1975-2026), validated across 5 eras
# (Volcker, Great Moderation, Dot-com→GFC, ZIRP, Post-COVID).
# See scripts/calibration_from_history.py + calibration_from_history_results.json.
#
# Original values came from Ang, Bekaert & Wei (2007) and similar literature.
# The 51-year backtest confirmed the core macro factors (inflation, policy, labor)
# were within 0.03 of the data-driven estimates. Three factors were revised:
#   consumer_sentiment: 0.90 → 0.96 (data shows monthly AR1=0.955, very sticky)
#   financial_conditions: 0.88 → 0.95 (rates more persistent than assumed)
#   housing_momentum: 0.95 → 0.98 (housing trends barely decay month-to-month)
#
# Daily persistence → monthly equivalent (×22 trading days):
#   0.97^22 = 0.51 (inflation_trend: ~50% of shock persists after 1 month)
#   0.96^22 = 0.40 (growth_trend: responsive to new data)
#   0.85^22 = 0.03 (commodity_pressure: very responsive, little memory)

# Persistence parameters (diagonal of F matrix)
# Higher = factor moves slowly, old observations matter more
PERSISTENCE = {
    "inflation_trend": 0.97,       # Literature: 0.97, Data: 0.999 → keep (theory-regime-robust)
    "growth_trend": 0.96,          # Literature: 0.93, Data: 0.962 → bump up
    "labor_pressure": 0.96,        # Literature: 0.96, Data: 0.988 → confirmed
    "housing_momentum": 0.98,      # Literature: 0.95, Data: 0.999 → bump up (housing is sticky)
    "financial_conditions": 0.95,  # Literature: 0.88, Data: 0.970 → bump up (rates persist)
    "commodity_pressure": 0.85,    # Literature: 0.85, Data: N/A (returns ≠ levels) → keep
    "consumer_sentiment": 0.96,    # Literature: 0.90, Data: 0.998 → bump up (sentiment sticky)
    "policy_stance": 0.98,         # Literature: 0.98, Data: 0.999 → confirmed
}

# Cross-factor dynamics (off-diagonal F entries)
# Original 5 channels from Implementation Architecture doc, validated by
# VAR(1) backtest (factor_validation.py). 51-year FRED backtest (Mar 2026)
# confirmed all 5 and revealed 3 additional significant channels.
# Coefficients kept conservative (0.03-0.07) for regime-robustness.
# See scripts/calibration_from_history_results.json for full VAR(1) analysis.
CROSS_DYNAMICS = [
    # (from_factor, to_factor, coefficient)
    # ── Original 5 (confirmed by 51yr data) ──
    ("policy_stance", "financial_conditions", 0.07),   # Data: 0.073, was 0.05
    ("financial_conditions", "housing_momentum", 0.04), # Data: 0.040, confirmed
    ("commodity_pressure", "inflation_trend", 0.06),    # Confirmed
    ("labor_pressure", "consumer_sentiment", 0.03),     # Confirmed
    ("growth_trend", "labor_pressure", 0.04),           # Confirmed
    # ── New channels from 51yr backtest ──
    ("growth_trend", "inflation_trend", 0.05),          # Data: 0.055 (Phillips curve)
    ("labor_pressure", "housing_momentum", -0.05),      # Data: -0.059 (tight labor → less construction)
    ("housing_momentum", "financial_conditions", 0.04), # Data: 0.043 (housing wealth → risk appetite)
]

# Stream-to-factor loading matrix (H rows)
# From Implementation Architecture doc — representative loadings
# Each entry: {factor_name: loading_value}
# Bold values in the spec table are primary loadings
STREAM_LOADINGS: dict[str, dict[str, float]] = {
    "US_CPI_YOY": {"inflation_trend": 0.85, "growth_trend": 0.10, "labor_pressure": 0.05,
                    "housing_momentum": 0.15, "commodity_pressure": 0.20, "policy_stance": 0.10},
    "CORE_CPI": {"inflation_trend": 0.90, "growth_trend": 0.05, "labor_pressure": 0.10,
                 "housing_momentum": 0.20, "policy_stance": 0.10},
    "PPI": {"inflation_trend": 0.60, "growth_trend": 0.15, "commodity_pressure": 0.50},
    "NONFARM_PAYROLLS": {"growth_trend": 0.30, "labor_pressure": 0.85,
                         "financial_conditions": 0.10, "consumer_sentiment": 0.15},
    "UNEMPLOYMENT_RATE": {"growth_trend": -0.25, "labor_pressure": -0.80,
                          "financial_conditions": -0.10, "consumer_sentiment": -0.20},
    "INITIAL_CLAIMS": {"growth_trend": -0.20, "labor_pressure": -0.70,
                       "consumer_sentiment": -0.15},
    "HOUSING_STARTS": {"growth_trend": 0.20, "housing_momentum": 0.85,
                       "financial_conditions": 0.30, "consumer_sentiment": 0.20, "policy_stance": -0.25},
    "HOME_PRICES": {"inflation_trend": 0.20, "growth_trend": 0.15, "housing_momentum": 0.75,
                    "financial_conditions": 0.20, "consumer_sentiment": 0.15, "policy_stance": -0.15},
    "RETAIL_SALES": {"growth_trend": 0.50, "labor_pressure": 0.15,
                     "financial_conditions": 0.10, "consumer_sentiment": 0.60},
    "CONSUMER_CONFIDENCE": {"growth_trend": 0.25, "labor_pressure": 0.10,
                            "consumer_sentiment": 0.85},
    "OIL_PRICE": {"inflation_trend": 0.25, "growth_trend": 0.15,
                  "financial_conditions": 0.10, "commodity_pressure": 0.90},
    "GOLD_PRICE": {"inflation_trend": 0.10, "growth_trend": -0.10,
                   "financial_conditions": -0.40, "commodity_pressure": 0.30, "policy_stance": -0.20},
    "COPPER_PRICE": {"growth_trend": 0.40, "housing_momentum": 0.20,
                     "commodity_pressure": 0.60},
    "FED_FUNDS_RATE": {"inflation_trend": 0.15, "financial_conditions": 0.20,
                       "policy_stance": 0.90},
    "10Y_YIELD": {"inflation_trend": 0.30, "growth_trend": 0.25,
                  "financial_conditions": 0.50, "policy_stance": 0.40},
    "SP500": {"growth_trend": 0.35, "financial_conditions": 0.50,
              "consumer_sentiment": 0.30, "policy_stance": -0.20},
    "BTC_USD": {"growth_trend": 0.15, "financial_conditions": 0.40,
                "consumer_sentiment": 0.25, "policy_stance": -0.15},
    # Raw commodity pressure (for streams that directly measure input costs)
    "COMMODITY_PRESSURE_RAW": {"commodity_pressure": 0.80, "inflation_trend": 0.15,
                                "growth_trend": 0.10},
}

# Anomaly threshold for innovation z-scores.
# 2.5σ assumes Gaussian, but financial data has kurtosis 5-10.
# 3.5σ captures 99.95% of Gaussian and ~97% of Student-t(5).
# Validated: TRUF experiment anomaly rate dropped 41.7% → 4.0%.
ANOMALY_THRESHOLD = 3.5

# Frequency scaling for observation noise (R).
# Daily streams get R × 16 (√252), weekly R × 7 (√52), monthly R × 3.5 (√12).
# Equalizes annual information content: 252 small daily updates ≈ 12 bigger monthly updates.
# Without this, BTC (678 updates/yr) dominates CPI (11 updates/yr).
# Validated: TRUF experiment showed proper damping of daily noise.
FREQUENCY_SCALE: dict[str, float] = {
    # Daily financial streams → high R (low per-observation weight)
    "BTC_USD": 16.0,
    "SP500": 16.0,
    "OIL_PRICE": 16.0,
    "COPPER_PRICE": 16.0,
    "COMMODITY_PRESSURE_RAW": 16.0,
    # Weekly streams → moderate R
    "INITIAL_CLAIMS": 7.0,
    "HOUSING_STARTS": 7.0,
    # Monthly streams → low R (high per-observation weight)
    "US_CPI_YOY": 3.5,
    "CORE_CPI": 3.5,
    "PPI": 3.5,
    "NONFARM_PAYROLLS": 3.5,
    "UNEMPLOYMENT_RATE": 3.5,
    "RETAIL_SALES": 3.5,
    "HOME_PRICES": 3.5,
    # Semi-annual → very high weight per observation
    "CONSUMER_CONFIDENCE": 2.0,
    # Rates — update frequency varies, keep default
    "FED_FUNDS_RATE": 1.0,
    "10Y_YIELD": 1.0,
    "GOLD_PRICE": 1.0,
}

# Observation noise (R) by stream type — BASE values before frequency scaling.
# Original: Implementation Architecture doc. Revised Mar 2026 using 51yr FRED
# residual variance analysis (scripts/calibration_from_history_results.json).
# 3 values adjusted: CONSUMER_CONFIDENCE (0.20→0.10), INITIAL_CLAIMS (0.10→0.05),
# PPI (0.05→0.08). Others confirmed within tolerance.
STREAM_NOISE: dict[str, float] = {
    "US_CPI_YOY": 0.03,
    "CORE_CPI": 0.03,
    "PPI": 0.08,              # Data: 0.10, was 0.05 (PPI noisier than CPI)
    "NONFARM_PAYROLLS": 0.10,
    "UNEMPLOYMENT_RATE": 0.08,
    "INITIAL_CLAIMS": 0.05,   # Data: 0.04, was 0.10 (claims less noisy than assumed)
    "HOUSING_STARTS": 0.12,
    "HOME_PRICES": 0.05,
    "RETAIL_SALES": 0.08,
    "CONSUMER_CONFIDENCE": 0.10,  # Data: 0.10, was 0.20 (confidence less noisy than assumed)
    "OIL_PRICE": 0.05,
    "GOLD_PRICE": 0.05,
    "COPPER_PRICE": 0.06,
    "FED_FUNDS_RATE": 0.01,
    "10Y_YIELD": 0.03,
    "SP500": 0.05,
    "BTC_USD": 0.08,
    "COMMODITY_PRESSURE_RAW": 0.06,
}


@dataclass
class StateEstimate:
    """Current state estimate with covariance."""
    mean: np.ndarray          # 8-element state vector
    covariance: np.ndarray    # 8x8 covariance matrix
    factor_names: list[str] = field(default_factory=lambda: list(FACTORS))

    def to_dict(self) -> dict:
        """Serialize for JSONB storage."""
        return {
            "factors": {
                name: float(self.mean[i])
                for i, name in enumerate(self.factor_names)
            },
            "covariance": self.covariance.tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateEstimate":
        """Deserialize from JSONB."""
        factors = data.get("factors", {})
        mean = np.array([factors.get(f, 0.0) for f in FACTORS])
        cov = np.array(data.get("covariance", np.eye(N_FACTORS).tolist()))
        return cls(mean=mean, covariance=cov)


@dataclass
class StateUpdate:
    """Result of a single observation update.

    Every probability number must be traceable back to its source data,
    model reasoning, and constraints. This dataclass captures the full
    audit trail for each Kalman update.
    """
    stream_key: str
    observed: float
    predicted: float
    innovation: float          # observed - predicted
    innovation_zscore: float   # standardized innovation
    kalman_gain: list[float]
    state_before: list[float]
    state_after: list[float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_anomalous(self) -> bool:
        """Innovation z-score exceeds ANOMALY_THRESHOLD (3.5σ).

        Uses 3.5σ instead of 2.5σ because financial data is fat-tailed
        (kurtosis 5-10). Validated: 41.7% → 4.0% false anomaly rate.
        """
        return abs(self.innovation_zscore) > ANOMALY_THRESHOLD

    def to_dict(self) -> dict:
        """Serialize for JSONB storage — prior audit trail."""
        return {
            "stream_key": self.stream_key,
            "observed": self.observed,
            "predicted": self.predicted,
            "innovation": self.innovation,
            "innovation_zscore": self.innovation_zscore,
            "kalman_gain": self.kalman_gain,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "is_anomalous": bool(self.is_anomalous),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateUpdate":
        """Deserialize from JSONB."""
        return cls(
            stream_key=data["stream_key"],
            observed=data["observed"],
            predicted=data["predicted"],
            innovation=data["innovation"],
            innovation_zscore=data["innovation_zscore"],
            kalman_gain=data["kalman_gain"],
            state_before=data["state_before"],
            state_after=data["state_after"],
            timestamp=data.get("timestamp", ""),
        )


class EconomicStateEstimator:
    """Level 1: Estimates the hidden economic state from TRUF streams.

    ~200 lines of numpy implementing the standard Kalman filter.
    Processes TRUF data as it arrives, maintaining a running estimate
    of the 8 latent economic factors.
    """

    def __init__(self):
        # State estimate (initialized at zero = neutral economy)
        self.x = np.zeros(N_FACTORS)
        # State covariance (moderate initial uncertainty)
        self.P = np.eye(N_FACTORS) * 0.1

        # Build F (state transition matrix) from persistence + cross-dynamics
        self.F = self._build_F()
        # Build Q (process noise) from persistence
        self.Q = self._build_Q()
        # Stream registry: stream_key -> (H_row, R_value)
        self.stream_registry: dict[str, tuple[np.ndarray, float]] = {}

        # Register all known streams
        self._register_default_streams()

        # Update history for threshold triggers
        self.update_count = 0
        self.recent_innovations: list[StateUpdate] = []

    def _build_F(self) -> np.ndarray:
        """Build state transition matrix from persistence params and cross-dynamics."""
        F = np.diag([PERSISTENCE[f] for f in FACTORS])

        for from_factor, to_factor, coeff in CROSS_DYNAMICS:
            i = FACTOR_INDEX[to_factor]
            j = FACTOR_INDEX[from_factor]
            F[i, j] = coeff

        return F

    def _build_Q(self) -> np.ndarray:
        """Build process noise matrix.

        Q_ii = (1 - phi_i^2) × scale, where phi_i is the AR(1) persistence.
        The (1 - phi^2) term is the unconditional variance of a unit-innovation
        AR(1) process, ensuring the state variance remains bounded.

        The 0.01 scale factor sets the state-to-noise ratio: factors evolve
        slowly relative to observation noise. This means the filter trusts
        accumulated state more than any single observation — appropriate for
        latent economic factors estimated from noisy daily data.
        """
        diag = [(1 - PERSISTENCE[f] ** 2) for f in FACTORS]
        return np.diag(diag) * 0.01

    def _register_default_streams(self):
        """Register all known TRUF streams with their loadings and noise.

        Applies FREQUENCY_SCALE to base R values so daily streams (BTC, SP500)
        have lower per-observation weight than monthly streams (CPI, unemployment).
        """
        for stream_key, loadings in STREAM_LOADINGS.items():
            H_row = np.zeros(N_FACTORS)
            for factor, loading in loadings.items():
                idx = FACTOR_INDEX.get(factor)
                if idx is not None:
                    H_row[idx] = loading

            base_noise = STREAM_NOISE.get(stream_key, 0.10)
            freq_scale = FREQUENCY_SCALE.get(stream_key, 1.0)
            noise = base_noise * freq_scale
            self.stream_registry[stream_key] = (H_row, noise)

    def register_stream(self, stream_key: str, loadings: dict[str, float], noise: float):
        """Register a custom TRUF stream with factor loadings and noise."""
        H_row = np.zeros(N_FACTORS)
        for factor, loading in loadings.items():
            idx = FACTOR_INDEX.get(factor)
            if idx is not None:
                H_row[idx] = loading
        self.stream_registry[stream_key] = (H_row, noise)

    def predict(self) -> StateEstimate:
        """Time update: project state forward one step.

        x_t = F @ x_{t-1}
        P_t = F @ P_{t-1} @ F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return StateEstimate(mean=self.x.copy(), covariance=self.P.copy())

    def update(self, stream_key: str, value: float) -> Optional[StateUpdate]:
        """Measurement update: incorporate one TRUF observation.

        Standard Kalman update equations:
        innovation = z - H @ x
        S = H @ P @ H^T + R
        K = P @ H^T / S
        x = x + K * innovation
        P = (I - K @ H) @ P
        """
        if stream_key not in self.stream_registry:
            logger.warning(f"Unknown stream: {stream_key}")
            return None

        H_row, R = self.stream_registry[stream_key]

        # Innovation (surprise)
        predicted = float(H_row @ self.x)
        innovation = value - predicted

        # Innovation variance
        S = float(H_row @ self.P @ H_row.T + R)
        if S <= 0:
            S = R  # Fallback to measurement noise

        # Standardized innovation (z-score)
        innovation_zscore = innovation / np.sqrt(S) if S > 0 else 0.0

        # Kalman gain
        K = self.P @ H_row.T / S

        # State update
        x_prior = self.x.copy()
        self.x = self.x + K * innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(N_FACTORS) - np.outer(K, H_row)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R

        self.update_count += 1

        update = StateUpdate(
            stream_key=stream_key,
            observed=value,
            predicted=predicted,
            innovation=innovation,
            innovation_zscore=innovation_zscore,
            kalman_gain=K.tolist(),
            state_before=x_prior.tolist(),
            state_after=self.x.tolist(),
        )

        # Keep recent innovations for trigger checking
        self.recent_innovations.append(update)
        if len(self.recent_innovations) > 100:
            self.recent_innovations = self.recent_innovations[-100:]

        return update

    def get_state(self) -> StateEstimate:
        """Get current state estimate."""
        return StateEstimate(mean=self.x.copy(), covariance=self.P.copy())

    def get_factor_value(self, factor: str) -> Optional[float]:
        """Get a single factor's current estimate."""
        idx = FACTOR_INDEX.get(factor)
        if idx is None:
            return None
        return float(self.x[idx])

    def get_factor_uncertainty(self, factor: str) -> Optional[float]:
        """Get a single factor's current uncertainty (std dev)."""
        idx = FACTOR_INDEX.get(factor)
        if idx is None:
            return None
        return float(np.sqrt(self.P[idx, idx]))

    def to_dict(self) -> dict:
        """Serialize full filter state for JSONB storage.

        Includes innovation history — the prior audit trail.
        Every update is stored so probability changes are traceable
        back to which stream, what value, what the filter predicted,
        and how the state shifted.
        """
        return {
            "factors": {name: float(self.x[i]) for i, name in enumerate(FACTORS)},
            "covariance": self.P.tolist(),
            "update_count": self.update_count,
            "innovation_history": [u.to_dict() for u in self.recent_innovations],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def load_state(self, state_dict: dict):
        """Restore filter state from JSONB, including innovation history."""
        factors = state_dict.get("factors", {})
        self.x = np.array([factors.get(f, 0.0) for f in FACTORS])
        cov = state_dict.get("covariance")
        if cov:
            self.P = np.array(cov)
        self.update_count = state_dict.get("update_count", 0)
        # Restore innovation history for audit trail continuity
        history = state_dict.get("innovation_history", [])
        self.recent_innovations = [StateUpdate.from_dict(h) for h in history]
