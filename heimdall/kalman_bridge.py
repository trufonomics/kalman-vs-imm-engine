"""
LLM-Kalman Bridge: Translates qualitative signals into pseudo-observations.

When Sonar detects "3 dovish Fed articles" or Gemini finds "historical
pattern match to 2019 mid-cycle adjustment," these qualitative signals
need to enter the quantitative Kalman system.

The bridge converts qualitative signals to pseudo-observations — synthetic
data points with inflated noise terms (high R) that nudge the state
estimate gently. Qualitative signals influence the state, but never
dominate it. The data always wins in the long run.

Signal Strength Mapping (from Implementation Architecture):
    Minor   → R=5.0, Kalman gain ~0.02  (barely nudges state)
    Moderate → R=1.0, Kalman gain ~0.10  (noticeable shift)
    Major   → R=0.3, Kalman gain ~0.25  (strong shift, like a surprising TRUF reading)

For context: typical TRUF stream R is 0.01-0.30. Pseudo-observations
ALWAYS have higher R than quantitative data.

Ref: Implementation Architecture doc (LLM-Kalman Bridge section)
     NY Fed nowcast model (soft data integration)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    FACTORS,
    FACTOR_INDEX,
    N_FACTORS,
    StateUpdate,
)

logger = logging.getLogger(__name__)


class SignalSource(str, Enum):
    """Which model or system generated the signal."""
    OPUS = "opus"
    GPT = "gpt"
    SONAR = "sonar"
    GEMINI = "gemini"
    HAIKU = "haiku"
    USER = "user"          # Manual user override
    SYSTEM = "system"      # Automated detection (e.g., structural break)


class SignalType(str, Enum):
    """Category of qualitative signal. Determines decay half-life."""
    BREAKING_NEWS = "breaking_news"
    MARKET_SENTIMENT = "market_sentiment"
    POLICY_COMMENTARY = "policy_commentary"
    POLICY_ACTION = "policy_action"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    HISTORICAL_PATTERN = "historical_pattern"
    GEOPOLITICAL_EVENT = "geopolitical_event"
    SCENARIO_GENERATION = "scenario_generation"


class Magnitude(str, Enum):
    """Signal strength. Maps to pseudo-observation noise (R)."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


# Noise (R) per magnitude — from Implementation Architecture spec
MAGNITUDE_NOISE: dict[Magnitude, float] = {
    Magnitude.MINOR: 5.0,
    Magnitude.MODERATE: 1.0,
    Magnitude.MAJOR: 0.3,
}

# Decay half-lives in hours — from Implementation Architecture spec
SIGNAL_HALF_LIVES: dict[SignalType, float] = {
    SignalType.BREAKING_NEWS: 3 * 24,          # 3 days
    SignalType.MARKET_SENTIMENT: 5 * 24,       # 5 days
    SignalType.POLICY_COMMENTARY: 14 * 24,     # 14 days
    SignalType.POLICY_ACTION: 60 * 24,         # 60 days
    SignalType.STRUCTURAL_ANALYSIS: 30 * 24,   # 30 days
    SignalType.HISTORICAL_PATTERN: 45 * 24,    # 45 days
    SignalType.GEOPOLITICAL_EVENT: 21 * 24,    # 21 days
    SignalType.SCENARIO_GENERATION: 14 * 24,   # 14 days
}


@dataclass
class FactorImpact:
    """How a signal affects a single economic factor."""
    factor: str                 # e.g. "policy_stance"
    direction: int              # -1, 0, or +1
    magnitude: Magnitude
    pseudo_value: float = 0.0   # Computed from direction + magnitude
    noise: float = 1.0          # R value (set from MAGNITUDE_NOISE)

    def to_dict(self) -> dict:
        return {
            "factor": self.factor,
            "direction": self.direction,
            "magnitude": self.magnitude.value,
            "pseudo_value": self.pseudo_value,
            "noise": self.noise,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FactorImpact":
        return cls(
            factor=data["factor"],
            direction=data["direction"],
            magnitude=Magnitude(data["magnitude"]),
            pseudo_value=data.get("pseudo_value", 0.0),
            noise=data.get("noise", 1.0),
        )


@dataclass
class PseudoObservation:
    """A qualitative signal translated into the Kalman state space.

    Each pseudo-observation carries:
    - Factor impacts with direction and magnitude
    - Source model and signal type (for audit trail)
    - Creation timestamp and decay half-life
    - LLM reasoning (why this translation)
    """
    id: str
    source: SignalSource
    signal_type: SignalType
    signal: str                                # Human-readable description
    impacts: list[FactorImpact]
    reasoning: str                             # Why this signal was translated this way
    confidence: float                          # LLM's self-assessed confidence (0-1)
    decay_half_life_hours: float               # Hours until half influence
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    applied_count: int = 0                     # How many times fed to Kalman

    @property
    def age_hours(self) -> float:
        """Hours since creation."""
        created = datetime.fromisoformat(self.created_at)
        now = datetime.now(timezone.utc)
        return (now - created).total_seconds() / 3600

    @property
    def decay_factor(self) -> float:
        """Current decay multiplier: 2^(age/half_life).

        At age=0: 1.0 (full influence)
        At age=half_life: 2.0 (half influence)
        At age=2*half_life: 4.0 (quarter influence)
        At age=5*half_life: 32.0 (effectively zero)
        """
        if self.decay_half_life_hours <= 0:
            return 1.0
        return 2.0 ** (self.age_hours / self.decay_half_life_hours)

    @property
    def is_expired(self) -> bool:
        """Past 5 half-lives = effectively zero influence."""
        return self.age_hours > (self.decay_half_life_hours * 5)

    def effective_noise(self, base_noise: float) -> float:
        """R_effective = R_original * 2^(age / half_life)."""
        return base_noise * self.decay_factor

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source.value,
            "signal_type": self.signal_type.value,
            "signal": self.signal,
            "impacts": [i.to_dict() for i in self.impacts],
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "decay_half_life_hours": self.decay_half_life_hours,
            "created_at": self.created_at,
            "applied_count": self.applied_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PseudoObservation":
        return cls(
            id=data["id"],
            source=SignalSource(data["source"]),
            signal_type=SignalType(data["signal_type"]),
            signal=data["signal"],
            impacts=[FactorImpact.from_dict(i) for i in data.get("impacts", [])],
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5),
            decay_half_life_hours=data.get("decay_half_life_hours", 72),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            applied_count=data.get("applied_count", 0),
        )


class KalmanBridge:
    """Translates qualitative LLM signals into Kalman pseudo-observations.

    Usage:
        bridge = KalmanBridge()

        # LLM detects geopolitical signal
        obs = bridge.create_pseudo_observation(
            source=SignalSource.SONAR,
            signal_type=SignalType.GEOPOLITICAL_EVENT,
            signal="US-Iran tensions escalate, Strait of Hormuz at risk",
            impacts=[
                {"factor": "commodity_pressure", "direction": 1, "magnitude": "major"},
                {"factor": "inflation_trend", "direction": 1, "magnitude": "moderate"},
                {"factor": "growth_trend", "direction": -1, "magnitude": "moderate"},
                {"factor": "financial_conditions", "direction": -1, "magnitude": "minor"},
            ],
            reasoning="Oil supply disruption raises commodity costs, feeds into inflation, "
                      "tightens financial conditions, slows growth",
            confidence=0.7,
        )

        # Apply to Kalman filter
        updates = bridge.apply_to_estimator(estimator)
    """

    # Direction → base pseudo-value mapping
    # These are in normalized state-space units (roughly standard deviations)
    DIRECTION_VALUES = {
        Magnitude.MINOR: 0.15,      # Small nudge
        Magnitude.MODERATE: 0.40,    # Noticeable shift
        Magnitude.MAJOR: 0.80,       # Strong move
    }

    def __init__(self):
        self.observations: list[PseudoObservation] = []
        self._counter = 0

    def create_pseudo_observation(
        self,
        source: SignalSource,
        signal_type: SignalType,
        signal: str,
        impacts: list[dict],
        reasoning: str,
        confidence: float = 0.5,
        half_life_override: Optional[float] = None,
    ) -> PseudoObservation:
        """Create a pseudo-observation from a qualitative signal.

        Args:
            source: Which model/system generated this
            signal_type: Category (determines decay half-life)
            signal: Human-readable description
            impacts: List of dicts with {factor, direction, magnitude}
            reasoning: Why this translation was made
            confidence: LLM's self-assessed confidence (0-1)
            half_life_override: Override default half-life (hours)

        Returns:
            PseudoObservation ready to apply to Kalman filter
        """
        self._counter += 1
        obs_id = f"pseudo_{self._counter}_{int(datetime.now(timezone.utc).timestamp())}"

        half_life = half_life_override or SIGNAL_HALF_LIVES.get(
            signal_type, 72  # Default 3 days
        )

        factor_impacts = []
        for impact in impacts:
            factor = impact["factor"]
            if factor not in FACTOR_INDEX:
                logger.warning(f"Unknown factor in pseudo-observation: {factor}")
                continue

            direction = impact.get("direction", 0)
            magnitude = Magnitude(impact.get("magnitude", "moderate"))

            # Compute pseudo-value: direction * base_value * confidence
            base_value = self.DIRECTION_VALUES[magnitude]
            pseudo_value = direction * base_value * confidence

            # Noise from magnitude
            noise = MAGNITUDE_NOISE[magnitude]

            factor_impacts.append(FactorImpact(
                factor=factor,
                direction=direction,
                magnitude=magnitude,
                pseudo_value=pseudo_value,
                noise=noise,
            ))

        obs = PseudoObservation(
            id=obs_id,
            source=source,
            signal_type=signal_type,
            signal=signal,
            impacts=factor_impacts,
            reasoning=reasoning,
            confidence=confidence,
            decay_half_life_hours=half_life,
        )

        self.observations.append(obs)
        logger.info(
            f"Created pseudo-observation: {obs_id} ({signal_type.value}) "
            f"affecting {len(factor_impacts)} factors"
        )

        return obs

    def apply_to_estimator(
        self,
        estimator: EconomicStateEstimator,
        observation: Optional[PseudoObservation] = None,
    ) -> list[StateUpdate]:
        """Apply pseudo-observation(s) to the Kalman filter.

        If observation is provided, apply only that one.
        Otherwise, apply all active (non-expired) observations.

        Each factor impact becomes a single-factor Kalman update with
        inflated R (incorporating both magnitude noise and decay).
        """
        targets = [observation] if observation else self.get_active_observations()
        updates = []

        for obs in targets:
            if obs.is_expired:
                continue

            for impact in obs.impacts:
                idx = FACTOR_INDEX.get(impact.factor)
                if idx is None:
                    continue

                # Build single-factor observation vector
                # H row: 1.0 on the target factor, 0 elsewhere
                stream_key = f"pseudo:{obs.id}:{impact.factor}"
                H_row = np.zeros(N_FACTORS)
                H_row[idx] = 1.0

                # R = base noise * decay factor
                effective_R = obs.effective_noise(impact.noise)

                # Register temporarily, update, then remove
                estimator.stream_registry[stream_key] = (H_row, effective_R)
                update = estimator.update(stream_key, impact.pseudo_value)
                del estimator.stream_registry[stream_key]

                if update:
                    updates.append(update)

            obs.applied_count += 1

        return updates

    def apply_single_to_estimator(
        self,
        estimator: EconomicStateEstimator,
        observation: PseudoObservation,
    ) -> list[StateUpdate]:
        """Apply a single pseudo-observation to the Kalman filter."""
        return self.apply_to_estimator(estimator, observation)

    def get_active_observations(self) -> list[PseudoObservation]:
        """Get all non-expired pseudo-observations."""
        return [obs for obs in self.observations if not obs.is_expired]

    def get_expired_observations(self) -> list[PseudoObservation]:
        """Get expired observations (for audit trail, not influence)."""
        return [obs for obs in self.observations if obs.is_expired]

    def cleanup_expired(self, keep_for_audit: int = 100) -> int:
        """Remove expired observations beyond audit retention limit.

        Returns number of observations removed.
        """
        active = self.get_active_observations()
        expired = self.get_expired_observations()

        # Keep the most recent expired for audit trail
        expired_to_keep = expired[-keep_for_audit:] if len(expired) > keep_for_audit else expired
        removed = len(expired) - len(expired_to_keep)

        self.observations = active + expired_to_keep
        return removed

    def get_influence_summary(self) -> list[dict]:
        """Summarize current active pseudo-observations and their influence.

        Returns a list sorted by effective influence (lowest R = most influence).
        """
        summary = []
        for obs in self.get_active_observations():
            for impact in obs.impacts:
                effective_R = obs.effective_noise(impact.noise)
                # Approximate influence: lower R = more influence
                influence = 1.0 / (1.0 + effective_R)
                summary.append({
                    "obs_id": obs.id,
                    "source": obs.source.value,
                    "signal_type": obs.signal_type.value,
                    "signal": obs.signal[:80],
                    "factor": impact.factor,
                    "direction": impact.direction,
                    "magnitude": impact.magnitude.value,
                    "pseudo_value": round(impact.pseudo_value, 4),
                    "base_noise": impact.noise,
                    "effective_noise": round(effective_R, 4),
                    "influence": round(influence, 4),
                    "age_hours": round(obs.age_hours, 1),
                    "decay_factor": round(obs.decay_factor, 4),
                })

        return sorted(summary, key=lambda x: x["effective_noise"])

    def to_dict(self) -> dict:
        """Serialize bridge state for JSONB storage."""
        return {
            "observations": [obs.to_dict() for obs in self.observations],
            "counter": self._counter,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def load_state(self, state_dict: dict):
        """Restore bridge state from JSONB."""
        self.observations = [
            PseudoObservation.from_dict(o)
            for o in state_dict.get("observations", [])
        ]
        self._counter = state_dict.get("counter", len(self.observations))
