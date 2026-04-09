"""Unit tests for Heimdall early warning trigger signals."""

import pytest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

from heimdall.kalman_filter import FACTORS, StateUpdate
from heimdall.trigger_service import (
    TriggerService,
    TriggerType,
    TriggerAction,
)


def _make_state_update(
    stream_key: str = "us-cpi",
    z_score: float = 0.5,
    kalman_gain: list[float] | None = None,
) -> StateUpdate:
    """Build a minimal StateUpdate for testing."""
    if kalman_gain is None:
        kalman_gain = [0.1] * len(FACTORS)
    return StateUpdate(
        stream_key=stream_key,
        observed=100.0,
        predicted=99.5,
        innovation=0.5,
        innovation_zscore=z_score,
        kalman_gain=kalman_gain,
        state_before=[0.0] * len(FACTORS),
        state_after=[0.01] * len(FACTORS),
    )


class TestInnovationStreak:
    """Innovation streak fires after 5+ consecutive elevated z-scores on same factor."""

    def test_no_trigger_below_threshold(self):
        svc = TriggerService()
        # 10 mild observations — no trigger
        for _ in range(10):
            triggers = svc.process_kalman_update(
                _make_state_update(z_score=1.5)
            )
            assert not any(
                t.trigger_type == TriggerType.INNOVATION_STREAK for t in triggers
            )

    def test_trigger_after_streak(self):
        svc = TriggerService()
        # Focus gain on factor 0 (inflation_trend)
        gain = [0.0] * len(FACTORS)
        gain[0] = 0.5  # High gain on inflation_trend
        fired = False
        for _ in range(6):
            triggers = svc.process_kalman_update(
                _make_state_update(z_score=2.5, kalman_gain=gain)
            )
            if any(t.trigger_type == TriggerType.INNOVATION_STREAK for t in triggers):
                fired = True
        assert fired

    def test_streak_resets_on_low_z(self):
        svc = TriggerService()
        gain = [0.0] * len(FACTORS)
        gain[0] = 0.5
        # 4 elevated, then 1 low → resets
        for _ in range(4):
            svc.process_kalman_update(
                _make_state_update(z_score=2.5, kalman_gain=gain)
            )
        svc.process_kalman_update(
            _make_state_update(z_score=0.3, kalman_gain=gain)
        )
        # 4 more elevated — still not 5 consecutive
        for _ in range(4):
            triggers = svc.process_kalman_update(
                _make_state_update(z_score=2.5, kalman_gain=gain)
            )
            assert not any(
                t.trigger_type == TriggerType.INNOVATION_STREAK for t in triggers
            )


class TestMultiFactorSurprise:
    """Multi-factor surprise fires when 3+ factors elevated in rolling window."""

    def test_no_trigger_single_factor(self):
        svc = TriggerService()
        gain = [0.0] * len(FACTORS)
        gain[0] = 0.5  # Only inflation_trend
        for _ in range(12):
            triggers = svc.process_kalman_update(
                _make_state_update(z_score=3.0, kalman_gain=gain)
            )
            assert not any(
                t.trigger_type == TriggerType.MULTI_FACTOR_SURPRISE
                for t in triggers
            )

    def test_trigger_with_broad_gain(self):
        svc = TriggerService()
        # High gain on 4 factors — broad surprise
        # factor_z = |gain[i]| * |z| must exceed MULTI_FACTOR_Z_THRESHOLD (1.5)
        # 0.6 * 3.0 = 1.8 > 1.5 ✓
        gain = [0.0] * len(FACTORS)
        gain[0] = 0.6  # inflation_trend
        gain[1] = 0.6  # growth_trend
        gain[2] = 0.6  # labor_pressure
        gain[3] = 0.6  # housing_momentum
        fired = False
        for _ in range(5):
            triggers = svc.process_kalman_update(
                _make_state_update(z_score=3.0, kalman_gain=gain)
            )
            if any(
                t.trigger_type == TriggerType.MULTI_FACTOR_SURPRISE
                for t in triggers
            ):
                fired = True
        assert fired


class TestBoundaryApproach:
    """Boundary approach fires when dominant regime drifts toward 50%."""

    def test_no_trigger_stable_regime(self):
        svc = TriggerService()
        # Stable at 75% — no trigger
        for _ in range(5):
            triggers = svc.check_boundary_approach(
                {"expansion": 0.75, "stagflation": 0.15, "contraction": 0.10}
            )
            assert not triggers

    def test_trigger_approaching_boundary(self):
        svc = TriggerService()
        # Simulate dominant regime declining toward 50%
        probs_sequence = [
            {"expansion": 0.60, "stagflation": 0.25, "contraction": 0.15},
            {"expansion": 0.57, "stagflation": 0.27, "contraction": 0.16},
            {"expansion": 0.55, "stagflation": 0.28, "contraction": 0.17},
            {"expansion": 0.53, "stagflation": 0.29, "contraction": 0.18},
        ]
        fired = False
        for probs in probs_sequence:
            triggers = svc.check_boundary_approach(probs)
            if triggers:
                fired = True
                assert triggers[0].trigger_type == TriggerType.BOUNDARY_APPROACH
                assert "expansion" in triggers[0].detail
        assert fired

    def test_no_trigger_when_regime_switches(self):
        svc = TriggerService()
        # Dominant switches from expansion to stagflation — resets streak
        svc.check_boundary_approach(
            {"expansion": 0.55, "stagflation": 0.30, "contraction": 0.15}
        )
        svc.check_boundary_approach(
            {"expansion": 0.53, "stagflation": 0.31, "contraction": 0.16}
        )
        # Now stagflation takes over
        triggers = svc.check_boundary_approach(
            {"stagflation": 0.52, "expansion": 0.30, "contraction": 0.18}
        )
        # Streak resets — shouldn't fire (only 1 observation of new dominant)
        assert not triggers

    def test_no_trigger_above_threshold(self):
        svc = TriggerService()
        # Declining but still far from 50% (above 0.08 threshold)
        probs_sequence = [
            {"expansion": 0.80, "stagflation": 0.12, "contraction": 0.08},
            {"expansion": 0.75, "stagflation": 0.15, "contraction": 0.10},
            {"expansion": 0.70, "stagflation": 0.18, "contraction": 0.12},
            {"expansion": 0.65, "stagflation": 0.20, "contraction": 0.15},
        ]
        for probs in probs_sequence:
            triggers = svc.check_boundary_approach(probs)
            assert not triggers


class TestSerializationRoundTrip:
    """Early warning state survives to_dict/load_state cycle."""

    def test_roundtrip_preserves_innovation_streaks(self):
        svc = TriggerService()
        gain = [0.0] * len(FACTORS)
        gain[0] = 0.5
        # Build up streak state
        for _ in range(3):
            svc.process_kalman_update(
                _make_state_update(z_score=2.5, kalman_gain=gain)
            )

        # Serialize and restore
        state = svc.to_dict()
        svc2 = TriggerService()
        svc2.load_state(state)

        assert svc2._innovation_streaks == svc._innovation_streaks

    def test_roundtrip_preserves_boundary_state(self):
        svc = TriggerService()
        svc.check_boundary_approach(
            {"expansion": 0.60, "stagflation": 0.25, "contraction": 0.15}
        )
        svc.check_boundary_approach(
            {"expansion": 0.57, "stagflation": 0.27, "contraction": 0.16}
        )

        state = svc.to_dict()
        svc2 = TriggerService()
        svc2.load_state(state)

        assert svc2._prev_dominant_prob == svc._prev_dominant_prob
        assert svc2._prev_dominant_regime == svc._prev_dominant_regime
        assert svc2._boundary_approach_streak == svc._boundary_approach_streak

    def test_roundtrip_preserves_factor_zscores(self):
        svc = TriggerService()
        gain = [0.1] * len(FACTORS)
        for _ in range(5):
            svc.process_kalman_update(
                _make_state_update(z_score=2.0, kalman_gain=gain)
            )

        state = svc.to_dict()
        svc2 = TriggerService()
        svc2.load_state(state)

        assert len(svc2._recent_factor_zscores) == len(svc._recent_factor_zscores)


class TestPriorityOrdering:
    """get_pending_triggers respects priority order."""

    def test_early_warnings_in_priority(self):
        svc = TriggerService()
        # Manually insert triggers of different types
        svc._create_trigger(
            TriggerType.NARRATIVE_STALE,
            TriggerAction.REFRESH_NARRATIVE,
            None, "stale", {},
        )
        svc._create_trigger(
            TriggerType.BOUNDARY_APPROACH,
            TriggerAction.ALERT_USER,
            None, "approaching", {},
        )
        svc._create_trigger(
            TriggerType.INNOVATION_STREAK,
            TriggerAction.ALERT_USER,
            None, "streak", {},
        )
        svc._create_trigger(
            TriggerType.MULTI_FACTOR_SURPRISE,
            TriggerAction.LLM_REEVALUATE,
            None, "multi", {},
        )

        pending = svc.get_pending_triggers()
        types = [t.trigger_type for t in pending]
        assert types.index(TriggerType.BOUNDARY_APPROACH) < types.index(TriggerType.MULTI_FACTOR_SURPRISE)
        assert types.index(TriggerType.MULTI_FACTOR_SURPRISE) < types.index(TriggerType.INNOVATION_STREAK)
        assert types.index(TriggerType.INNOVATION_STREAK) < types.index(TriggerType.NARRATIVE_STALE)
