"""
Threshold Trigger Service: Monitors the Heimdall pipeline for events
that should trigger LLM re-evaluation.

Collects triggers from three sources:
  1. IMM branch updates (tier crossings, sustained drift, convergence)
  2. Kalman innovations (z-score > 3.5 = anomalous data, fat-tail adjusted)
  3. Narrative freshness (branch narrative older than half-life)

Manages cooldowns to prevent trigger spam and queues actions for
the orchestrator to execute (LLM re-evaluation, merge suggestion, etc.).

Ref: Implementation Architecture doc (Threshold Triggers, Tree Governance)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from heimdall.imm_tracker import IMMBranchTracker, IMMUpdate
from heimdall.kalman_filter import FACTORS, StateUpdate

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Categories of threshold trigger."""
    TIER_CROSSING = "tier_crossing"           # Branch crosses 25/50/75%
    ANOMALOUS_DATA = "anomalous_data"         # |z-score| > 2.5
    SUSTAINED_DRIFT = "sustained_drift"       # 5+ consecutive same direction
    CONVERGENCE = "convergence"               # Two branches within 5pp
    NARRATIVE_STALE = "narrative_stale"        # Branch narrative past half-life
    BRANCH_EXTINCTION = "branch_extinction"   # Branch below 2% for 5 updates
    STRUCTURAL_BREAK = "structural_break"     # State outside all branch ranges
    # ── Early warning signals (Mar 23 2026) ──
    INNOVATION_STREAK = "innovation_streak"   # 5+ consecutive |z| > 2.0 on same factor
    MULTI_FACTOR_SURPRISE = "multi_factor_surprise"  # 3+ factors elevated in same window
    BOUNDARY_APPROACH = "boundary_approach"   # Dominant regime drifting toward 50%


class TriggerAction(str, Enum):
    """What should happen when a trigger fires."""
    LLM_REEVALUATE = "llm_reevaluate"         # Full LLM re-evaluation of branch
    REFRESH_NARRATIVE = "refresh_narrative"     # Update prose, not probabilities
    SUGGEST_MERGE = "suggest_merge"            # Surface merge suggestion to user
    SUGGEST_PRUNE = "suggest_prune"            # Surface prune suggestion
    SUGGEST_SPLIT = "suggest_split"            # Surface split suggestion
    ALERT_USER = "alert_user"                  # Notify user of significant change
    LOG_ONLY = "log_only"                      # Record but don't act


@dataclass
class Trigger:
    """A single trigger event with recommended action."""
    id: str
    trigger_type: TriggerType
    action: TriggerAction
    branch_id: Optional[str]
    detail: str
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trigger_type": self.trigger_type.value,
            "action": self.action.value,
            "branch_id": self.branch_id,
            "detail": self.detail,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trigger":
        return cls(
            id=data["id"],
            trigger_type=TriggerType(data["trigger_type"]),
            action=TriggerAction(data["action"]),
            branch_id=data.get("branch_id"),
            detail=data["detail"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", ""),
            resolved=data.get("resolved", False),
        )


class TriggerService:
    """Monitors Heimdall pipeline for threshold triggers.

    Usage:
        service = TriggerService()

        # After IMM update
        triggers = service.process_imm_update(imm_update)

        # After Kalman update
        triggers = service.process_kalman_update(state_update)

        # Check narrative freshness
        triggers = service.check_narrative_freshness(branches, half_life_hours=336)

        # Get pending actions
        pending = service.get_pending_triggers()
    """

    # Cooldown: minimum hours between triggers of the same type for the same branch
    COOLDOWN_HOURS: dict[TriggerType, float] = {
        TriggerType.TIER_CROSSING: 1.0,        # Max once per hour
        TriggerType.ANOMALOUS_DATA: 0.5,       # Can fire often, data is data
        TriggerType.SUSTAINED_DRIFT: 6.0,      # Once per 6 hours
        TriggerType.CONVERGENCE: 12.0,         # Slow-moving
        TriggerType.NARRATIVE_STALE: 24.0,     # Once per day
        TriggerType.BRANCH_EXTINCTION: 12.0,   # Once per 12 hours
        TriggerType.STRUCTURAL_BREAK: 1.0,     # Urgent, short cooldown
        TriggerType.INNOVATION_STREAK: 6.0,    # Once per 6 hours
        TriggerType.MULTI_FACTOR_SURPRISE: 8.0,  # Once per 8 hours
        TriggerType.BOUNDARY_APPROACH: 4.0,    # Fairly urgent
    }

    # Map trigger type → default action
    DEFAULT_ACTIONS: dict[TriggerType, TriggerAction] = {
        TriggerType.TIER_CROSSING: TriggerAction.LLM_REEVALUATE,
        TriggerType.ANOMALOUS_DATA: TriggerAction.ALERT_USER,
        TriggerType.SUSTAINED_DRIFT: TriggerAction.LLM_REEVALUATE,
        TriggerType.CONVERGENCE: TriggerAction.SUGGEST_MERGE,
        TriggerType.NARRATIVE_STALE: TriggerAction.REFRESH_NARRATIVE,
        TriggerType.BRANCH_EXTINCTION: TriggerAction.SUGGEST_PRUNE,
        TriggerType.STRUCTURAL_BREAK: TriggerAction.LLM_REEVALUATE,
        TriggerType.INNOVATION_STREAK: TriggerAction.ALERT_USER,
        TriggerType.MULTI_FACTOR_SURPRISE: TriggerAction.LLM_REEVALUATE,
        TriggerType.BOUNDARY_APPROACH: TriggerAction.ALERT_USER,
    }

    # Early warning thresholds
    INNOVATION_STREAK_THRESHOLD = 2.0   # |z| above this counts as "elevated"
    INNOVATION_STREAK_LENGTH = 5        # consecutive elevated observations to trigger
    MULTI_FACTOR_WINDOW = 10            # observations to look back for correlated surprises
    MULTI_FACTOR_MIN_FACTORS = 3        # factors that must be elevated simultaneously
    MULTI_FACTOR_Z_THRESHOLD = 1.5      # |z| threshold for factor elevation
    BOUNDARY_DISTANCE_THRESHOLD = 0.08  # trigger when dominant regime within 8pp of 50%

    def __init__(self):
        self.triggers: list[Trigger] = []
        self._counter = 0
        # Last trigger time per (type, branch_id) for cooldown
        self._last_triggered: dict[tuple[str, str], str] = {}
        # Track low-probability streaks per branch
        self._low_prob_streak: dict[str, int] = {}
        # ── Early warning state ──
        # Per-factor streak of elevated innovations: factor_name → consecutive count
        self._innovation_streaks: dict[str, int] = {}
        # Rolling window of recent innovations for multi-factor analysis
        self._recent_factor_zscores: list[dict[str, float]] = []
        # Previous dominant regime probability for boundary approach detection
        self._prev_dominant_prob: Optional[float] = None
        self._prev_dominant_regime: Optional[str] = None
        # Consecutive observations where dominant regime is approaching 50%
        self._boundary_approach_streak: int = 0

    def process_imm_update(self, imm_update: IMMUpdate) -> list[Trigger]:
        """Extract triggers from an IMM update result.

        The IMM tracker already detects tier crossings, sustained drift,
        and convergence. This method wraps them as Trigger objects with
        cooldown checks and action assignment.
        """
        new_triggers = []

        for raw in imm_update.triggers:
            trigger_type_str = raw.get("type", "")

            # Map IMM trigger types to our enum
            type_map = {
                "tier_crossing": TriggerType.TIER_CROSSING,
                "sustained_drift": TriggerType.SUSTAINED_DRIFT,
                "convergence": TriggerType.CONVERGENCE,
            }
            trigger_type = type_map.get(trigger_type_str)
            if not trigger_type:
                continue

            branch_id = raw.get("branch_id", raw.get("branch_id_a", ""))

            if not self._check_cooldown(trigger_type, branch_id):
                continue

            action = self.DEFAULT_ACTIONS.get(trigger_type, TriggerAction.LOG_ONLY)
            trigger = self._create_trigger(
                trigger_type=trigger_type,
                action=action,
                branch_id=branch_id,
                detail=raw.get("detail", str(raw)),
                metadata={
                    "stream_key": imm_update.stream_key,
                    "observed": imm_update.observed,
                    **{k: v for k, v in raw.items() if k not in ("type", "detail")},
                },
            )
            new_triggers.append(trigger)

        # Check for branch extinction (< 2% probability)
        for score in imm_update.branch_scores:
            bid = score.get("branch_id", "")
            prob_after = score.get("prob_after", 1.0)

            if prob_after < 0.02:
                self._low_prob_streak[bid] = self._low_prob_streak.get(bid, 0) + 1
            else:
                self._low_prob_streak[bid] = 0

            if self._low_prob_streak.get(bid, 0) >= 5:
                if self._check_cooldown(TriggerType.BRANCH_EXTINCTION, bid):
                    trigger = self._create_trigger(
                        trigger_type=TriggerType.BRANCH_EXTINCTION,
                        action=TriggerAction.SUGGEST_PRUNE,
                        branch_id=bid,
                        detail=f"Branch {score.get('name', bid)} below 2% for "
                               f"{self._low_prob_streak[bid]} consecutive updates",
                        metadata={
                            "probability": prob_after,
                            "consecutive_below": self._low_prob_streak[bid],
                        },
                    )
                    new_triggers.append(trigger)

        return new_triggers

    def process_kalman_update(self, state_update: StateUpdate) -> list[Trigger]:
        """Check a Kalman state update for anomalies and early warning signals.

        Three checks:
        1. Anomalous data (|z| > 3.5) — existing behavior
        2. Innovation streak — filter consistently surprised on same factor(s)
        3. Multi-factor surprise — multiple factors elevated in rolling window
        """
        new_triggers: list[Trigger] = []

        # ── 1. Single-observation anomaly (existing) ──
        if state_update.is_anomalous:
            if self._check_cooldown(
                TriggerType.ANOMALOUS_DATA, state_update.stream_key
            ):
                trigger = self._create_trigger(
                    trigger_type=TriggerType.ANOMALOUS_DATA,
                    action=TriggerAction.ALERT_USER,
                    branch_id=None,
                    detail=f"{state_update.stream_key} z-score={state_update.innovation_zscore:+.2f} "
                           f"(observed={state_update.observed:.4f}, "
                           f"predicted={state_update.predicted:.4f})",
                    metadata={
                        "stream_key": state_update.stream_key,
                        "innovation_zscore": state_update.innovation_zscore,
                        "observed": state_update.observed,
                        "predicted": state_update.predicted,
                        "innovation": state_update.innovation,
                    },
                )
                new_triggers.append(trigger)

        # ── 2. Innovation streak detection ──
        # Track which factors were most affected by this observation
        # via Kalman gain magnitude (which factors moved most)
        streak_triggers = self._check_innovation_streak(state_update)
        new_triggers.extend(streak_triggers)

        # ── 3. Multi-factor surprise detection ──
        multifactor_triggers = self._check_multi_factor_surprise(state_update)
        new_triggers.extend(multifactor_triggers)

        return new_triggers

    def _check_innovation_streak(self, state_update: StateUpdate) -> list[Trigger]:
        """Detect sustained elevated innovations on specific factors.

        When the filter is consistently surprised (|z| > 2.0) by observations
        that load onto the same factor, that factor's regime dynamics are
        shifting — the filter's model of that factor is becoming stale.
        This precedes IMM regime transitions by 5-15 observations.

        We track per-factor streaks by checking which factors received the
        largest Kalman gain updates (i.e., which factors the observation
        informed most).
        """
        abs_z = abs(state_update.innovation_zscore)
        is_elevated = abs_z > self.INNOVATION_STREAK_THRESHOLD

        # Determine which factors this observation most affected
        # Kalman gain K is an 8-vector; large |K[i]| means factor i was updated
        gain = state_update.kalman_gain
        if not gain or len(gain) != len(FACTORS):
            return []

        # Find factors with above-median gain (the ones this stream informs)
        abs_gains = [abs(g) for g in gain]
        median_gain = sorted(abs_gains)[len(abs_gains) // 2]
        affected_factors = [
            FACTORS[i] for i, g in enumerate(abs_gains)
            if g > max(median_gain, 0.001)
        ]

        triggers = []
        for factor in affected_factors:
            if is_elevated:
                self._innovation_streaks[factor] = (
                    self._innovation_streaks.get(factor, 0) + 1
                )
            else:
                self._innovation_streaks[factor] = 0

            streak = self._innovation_streaks.get(factor, 0)
            if streak >= self.INNOVATION_STREAK_LENGTH:
                if self._check_cooldown(TriggerType.INNOVATION_STREAK, factor):
                    trigger = self._create_trigger(
                        trigger_type=TriggerType.INNOVATION_STREAK,
                        action=TriggerAction.ALERT_USER,
                        branch_id=None,
                        detail=(
                            f"Filter consistently surprised on {factor}: "
                            f"{streak} consecutive observations with |z| > "
                            f"{self.INNOVATION_STREAK_THRESHOLD:.1f} "
                            f"(latest: {state_update.stream_key} z={state_update.innovation_zscore:+.2f}). "
                            f"Regime dynamics may be shifting."
                        ),
                        metadata={
                            "factor": factor,
                            "streak_length": streak,
                            "latest_stream": state_update.stream_key,
                            "latest_zscore": state_update.innovation_zscore,
                            "threshold": self.INNOVATION_STREAK_THRESHOLD,
                        },
                    )
                    triggers.append(trigger)
                    # Reset after firing to avoid spam
                    self._innovation_streaks[factor] = 0

        return triggers

    def _check_multi_factor_surprise(self, state_update: StateUpdate) -> list[Trigger]:
        """Detect correlated surprises across multiple factors.

        When 3+ factors show elevated innovation z-scores within a short
        window, something systemic is happening. Individual factor surprises
        are normal (noisy data). Correlated multi-factor surprises indicate
        a broad economic shift — the kind that precedes regime changes.

        Tracks a rolling window of per-factor z-score contributions.
        """
        gain = state_update.kalman_gain
        if not gain or len(gain) != len(FACTORS):
            return []

        # Compute per-factor contribution to this innovation:
        # factor_z[i] ≈ |K[i] * innovation| / sqrt(P[i,i])
        # Simplified: use |K[i] * z| as a proxy for factor-level surprise
        abs_z = abs(state_update.innovation_zscore)
        factor_zscores = {}
        for i, factor in enumerate(FACTORS):
            # Factor-level surprise = how much gain this factor got × overall surprise
            factor_z = abs(gain[i]) * abs_z
            if factor_z > 0.001:
                factor_zscores[factor] = factor_z

        # Add to rolling window
        self._recent_factor_zscores.append(factor_zscores)
        if len(self._recent_factor_zscores) > self.MULTI_FACTOR_WINDOW:
            self._recent_factor_zscores = self._recent_factor_zscores[
                -self.MULTI_FACTOR_WINDOW:
            ]

        # Count how many factors are elevated across the window
        factor_max_z: dict[str, float] = {}
        for entry in self._recent_factor_zscores:
            for factor, z in entry.items():
                factor_max_z[factor] = max(factor_max_z.get(factor, 0), z)

        elevated_factors = [
            (f, z) for f, z in factor_max_z.items()
            if z > self.MULTI_FACTOR_Z_THRESHOLD
        ]

        if len(elevated_factors) >= self.MULTI_FACTOR_MIN_FACTORS:
            if self._check_cooldown(TriggerType.MULTI_FACTOR_SURPRISE, "global"):
                elevated_factors.sort(key=lambda x: -x[1])
                factor_list = ", ".join(
                    f"{f} ({z:.2f})" for f, z in elevated_factors[:5]
                )
                trigger = self._create_trigger(
                    trigger_type=TriggerType.MULTI_FACTOR_SURPRISE,
                    action=TriggerAction.LLM_REEVALUATE,
                    branch_id=None,
                    detail=(
                        f"{len(elevated_factors)} factors elevated in last "
                        f"{self.MULTI_FACTOR_WINDOW} observations: {factor_list}. "
                        f"Broad economic shift detected."
                    ),
                    metadata={
                        "elevated_factors": [
                            {"factor": f, "max_z": round(z, 3)}
                            for f, z in elevated_factors
                        ],
                        "window_size": self.MULTI_FACTOR_WINDOW,
                        "threshold": self.MULTI_FACTOR_Z_THRESHOLD,
                    },
                )
                # Clear window after firing
                self._recent_factor_zscores.clear()
                return [trigger]

        return []

    def check_boundary_approach(
        self,
        branch_probabilities: dict[str, float],
    ) -> list[Trigger]:
        """Detect when the dominant regime is drifting toward 50%.

        When the most-probable regime's probability drops toward 50%, the IMM
        is losing confidence — a regime change is imminent. This fires BEFORE
        a tier crossing because it tracks the *trajectory*, not a single
        threshold crossing.

        Args:
            branch_probabilities: regime_name → probability mapping from IMM
                                  (e.g. {"expansion": 0.72, "stagflation": 0.18, ...})
        """
        if not branch_probabilities:
            return []

        # Find current dominant regime
        dominant_regime = max(branch_probabilities, key=branch_probabilities.get)
        dominant_prob = branch_probabilities[dominant_regime]

        triggers: list[Trigger] = []

        # Compare with previous: is the dominant regime approaching 50%?
        if self._prev_dominant_prob is not None and self._prev_dominant_regime is not None:
            distance_from_boundary = dominant_prob - 0.50

            # Track if same regime is losing probability toward 50%
            if (
                dominant_regime == self._prev_dominant_regime
                and dominant_prob < self._prev_dominant_prob
                and distance_from_boundary < self.BOUNDARY_DISTANCE_THRESHOLD
                and distance_from_boundary > 0  # Still above 50%, but barely
            ):
                self._boundary_approach_streak += 1
            else:
                self._boundary_approach_streak = 0

            # Fire when sustained approach (3+ consecutive observations heading toward 50%)
            if self._boundary_approach_streak >= 3:
                if self._check_cooldown(TriggerType.BOUNDARY_APPROACH, dominant_regime):
                    trigger = self._create_trigger(
                        trigger_type=TriggerType.BOUNDARY_APPROACH,
                        action=TriggerAction.ALERT_USER,
                        branch_id=None,
                        detail=(
                            f"Dominant regime '{dominant_regime}' approaching 50% boundary: "
                            f"{dominant_prob:.1%} (was {self._prev_dominant_prob:.1%}). "
                            f"{self._boundary_approach_streak} consecutive observations "
                            f"declining. Regime change may be imminent."
                        ),
                        metadata={
                            "dominant_regime": dominant_regime,
                            "current_prob": round(dominant_prob, 4),
                            "previous_prob": round(self._prev_dominant_prob, 4),
                            "distance_from_boundary": round(distance_from_boundary, 4),
                            "streak": self._boundary_approach_streak,
                            "all_probabilities": {
                                k: round(v, 4) for k, v in branch_probabilities.items()
                            },
                        },
                    )
                    triggers.append(trigger)
                    self._boundary_approach_streak = 0

        # Store for next comparison
        self._prev_dominant_prob = dominant_prob
        self._prev_dominant_regime = dominant_regime

        return triggers

    def check_narrative_freshness(
        self,
        branches: list[dict],
        half_life_hours: float = 336,  # 14 days default
    ) -> list[Trigger]:
        """Check if branch narratives are stale (older than half-life).

        Args:
            branches: List of branch dicts with 'branch_id', 'name',
                      'last_narrative_update' (ISO timestamp)
            half_life_hours: How long before narrative is considered stale
        """
        now = datetime.now(timezone.utc)
        new_triggers = []

        for branch in branches:
            bid = branch.get("branch_id", branch.get("id", ""))
            last_update_str = branch.get("last_narrative_update")
            if not last_update_str:
                continue

            last_update = datetime.fromisoformat(last_update_str)
            age_hours = (now - last_update).total_seconds() / 3600

            if age_hours > half_life_hours:
                if not self._check_cooldown(TriggerType.NARRATIVE_STALE, bid):
                    continue

                trigger = self._create_trigger(
                    trigger_type=TriggerType.NARRATIVE_STALE,
                    action=TriggerAction.REFRESH_NARRATIVE,
                    branch_id=bid,
                    detail=f"{branch.get('name', bid)} narrative is "
                           f"{age_hours:.0f}h old (half-life: {half_life_hours:.0f}h)",
                    metadata={
                        "age_hours": age_hours,
                        "half_life_hours": half_life_hours,
                        "last_update": last_update_str,
                    },
                )
                new_triggers.append(trigger)

        return new_triggers

    def check_structural_break(
        self,
        estimator_state: dict[str, float],
        branch_states: dict[str, dict],
        threshold_sigma: float = 3.0,
    ) -> list[Trigger]:
        """Check if the Level 1 state is outside all branch models' ranges.

        If no branch predicted the current state, something structural
        has changed. This is a strong signal for LLM re-evaluation.
        """
        if not branch_states:
            return []

        # Check each factor: is the estimator value within any branch's range?
        factors_outside = []
        for factor, value in estimator_state.items():
            covered = False
            for _bid, bstate in branch_states.items():
                branch_factors = bstate.get("factors", {})
                branch_value = branch_factors.get(factor, 0.0)
                # A branch "covers" a factor if its estimate is within threshold_sigma
                if abs(value - branch_value) < threshold_sigma * 0.3:
                    covered = True
                    break

            if not covered:
                factors_outside.append((factor, value))

        # If >2 factors are outside all branch ranges, structural break
        if len(factors_outside) >= 2:
            if not self._check_cooldown(TriggerType.STRUCTURAL_BREAK, "global"):
                return []

            trigger = self._create_trigger(
                trigger_type=TriggerType.STRUCTURAL_BREAK,
                action=TriggerAction.LLM_REEVALUATE,
                branch_id=None,
                detail=f"State outside all branch models on {len(factors_outside)} factors: "
                       + ", ".join(f"{f}={v:+.3f}" for f, v in factors_outside[:4]),
                metadata={
                    "factors_outside": [
                        {"factor": f, "value": v} for f, v in factors_outside
                    ],
                },
            )
            return [trigger]

        return []

    def get_pending_triggers(self) -> list[Trigger]:
        """Get unresolved triggers, ordered by urgency."""
        # Priority: structural_break > boundary_approach > anomalous > multi_factor > streak > tier > drift > rest
        priority = {
            TriggerType.STRUCTURAL_BREAK: 0,
            TriggerType.BOUNDARY_APPROACH: 1,
            TriggerType.ANOMALOUS_DATA: 2,
            TriggerType.MULTI_FACTOR_SURPRISE: 3,
            TriggerType.INNOVATION_STREAK: 4,
            TriggerType.TIER_CROSSING: 5,
            TriggerType.SUSTAINED_DRIFT: 6,
            TriggerType.BRANCH_EXTINCTION: 7,
            TriggerType.CONVERGENCE: 8,
            TriggerType.NARRATIVE_STALE: 9,
        }
        pending = [t for t in self.triggers if not t.resolved]
        return sorted(pending, key=lambda t: priority.get(t.trigger_type, 99))

    def resolve_trigger(self, trigger_id: str):
        """Mark a trigger as resolved (action taken)."""
        for trigger in self.triggers:
            if trigger.id == trigger_id:
                trigger.resolved = True
                return

    def _check_cooldown(self, trigger_type: TriggerType, key: str) -> bool:
        """Check if enough time has passed since the last trigger of this type/key."""
        cooldown_key = (trigger_type.value, key)
        last = self._last_triggered.get(cooldown_key)

        if last:
            last_time = datetime.fromisoformat(last)
            now = datetime.now(timezone.utc)
            hours_since = (now - last_time).total_seconds() / 3600
            cooldown = self.COOLDOWN_HOURS.get(trigger_type, 1.0)
            if hours_since < cooldown:
                return False

        # Record this trigger time
        self._last_triggered[cooldown_key] = datetime.now(timezone.utc).isoformat()
        return True

    def _create_trigger(
        self,
        trigger_type: TriggerType,
        action: TriggerAction,
        branch_id: Optional[str],
        detail: str,
        metadata: dict,
    ) -> Trigger:
        """Create and store a new trigger."""
        self._counter += 1
        trigger = Trigger(
            id=f"trigger_{self._counter}",
            trigger_type=trigger_type,
            action=action,
            branch_id=branch_id,
            detail=detail,
            metadata=metadata,
        )
        self.triggers.append(trigger)

        # Keep history bounded
        if len(self.triggers) > 500:
            # Keep resolved history + all unresolved
            resolved = [t for t in self.triggers if t.resolved]
            unresolved = [t for t in self.triggers if not t.resolved]
            self.triggers = resolved[-200:] + unresolved

        logger.info(f"Trigger: [{trigger_type.value}] {detail}")
        return trigger

    def to_dict(self) -> dict:
        """Serialize for JSONB storage."""
        return {
            "triggers": [t.to_dict() for t in self.triggers[-200:]],
            "counter": self._counter,
            "low_prob_streak": dict(self._low_prob_streak),
            "last_triggered": {
                f"{k[0]}:{k[1]}": v for k, v in self._last_triggered.items()
            },
            # ── Early warning state ──
            "innovation_streaks": dict(self._innovation_streaks),
            "recent_factor_zscores": list(self._recent_factor_zscores),
            "prev_dominant_prob": self._prev_dominant_prob,
            "prev_dominant_regime": self._prev_dominant_regime,
            "boundary_approach_streak": self._boundary_approach_streak,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def load_state(self, state_dict: dict):
        """Restore from JSONB."""
        self.triggers = [
            Trigger.from_dict(t) for t in state_dict.get("triggers", [])
        ]
        self._counter = state_dict.get("counter", len(self.triggers))
        self._low_prob_streak = state_dict.get("low_prob_streak", {})
        raw_last = state_dict.get("last_triggered", {})
        self._last_triggered = {}
        for key_str, val in raw_last.items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                self._last_triggered[(parts[0], parts[1])] = val
        # ── Early warning state ──
        self._innovation_streaks = state_dict.get("innovation_streaks", {})
        self._recent_factor_zscores = state_dict.get("recent_factor_zscores", [])
        self._prev_dominant_prob = state_dict.get("prev_dominant_prob")
        self._prev_dominant_regime = state_dict.get("prev_dominant_regime")
        self._boundary_approach_streak = state_dict.get("boundary_approach_streak", 0)
