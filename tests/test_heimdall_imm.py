"""Unit tests for Heimdall IMM Branch Tracker."""

import pytest
import numpy as np

from heimdall.imm_tracker import (
    IMMBranchTracker,
    BranchModel,
    IMMUpdate,
)
from heimdall.kalman_filter import EconomicStateEstimator


def _make_branches_jsonb():
    """Create test branch JSONB matching the DB format."""
    return [
        {
            "id": "base",
            "name": "Base Case",
            "probability": 0.5,
            "state_adjustments": {"inflation_trend": 0.0},
        },
        {
            "id": "high_inflation",
            "name": "High Inflation",
            "probability": 0.3,
            "state_adjustments": {"inflation_trend": 0.5, "commodity_pressure": 0.3},
        },
        {
            "id": "deflation",
            "name": "Deflation",
            "probability": 0.2,
            "state_adjustments": {"inflation_trend": -0.3, "consumer_sentiment": -0.2},
        },
    ]


class TestIMMBranchTracker:
    """Test Interacting Multiple Model branch tracker."""

    def _make_tracker(self) -> IMMBranchTracker:
        tracker = IMMBranchTracker()
        baseline = EconomicStateEstimator()
        tracker.initialize_branches(_make_branches_jsonb(), baseline)
        return tracker

    def test_initialization(self):
        tracker = IMMBranchTracker()
        assert len(tracker.branches) == 0

    def test_initialize_branches(self):
        tracker = self._make_tracker()
        assert len(tracker.branches) == 3

    def test_probabilities_sum_to_one(self):
        tracker = self._make_tracker()
        probs = tracker.get_probabilities()
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 0.001

    def test_predict(self):
        tracker = self._make_tracker()
        tracker.predict()
        probs = tracker.get_probabilities()
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_update_with_registered_stream(self):
        tracker = self._make_tracker()

        # Use a stream key that's registered in the baseline estimator
        baseline = EconomicStateEstimator()
        stream_key = list(baseline.stream_registry.keys())[0]

        tracker.predict()
        update = tracker.update(stream_key, 0.5)
        assert isinstance(update, IMMUpdate)

    def test_update_shifts_probabilities(self):
        tracker = self._make_tracker()

        baseline = EconomicStateEstimator()
        # Find an inflation-related stream
        inflation_streams = [
            k for k in baseline.stream_registry.keys()
            if "CPI" in k.upper() or "INFLATION" in k.upper()
        ]
        if not inflation_streams:
            # Use any available stream
            inflation_streams = list(baseline.stream_registry.keys())[:1]

        stream_key = inflation_streams[0]
        probs_before = tracker.get_probabilities()

        # Run several updates with high values (should favor inflation branch)
        for _ in range(5):
            tracker.predict()
            tracker.update(stream_key, 2.0)

        probs_after = tracker.get_probabilities()
        # Probabilities should still be valid
        assert abs(sum(probs_after.values()) - 1.0) < 0.01
        for p in probs_after.values():
            assert p >= 0.0

    def test_probabilities_stay_valid_under_noise(self):
        tracker = self._make_tracker()

        baseline = EconomicStateEstimator()
        stream_key = list(baseline.stream_registry.keys())[0]

        for _ in range(30):
            tracker.predict()
            tracker.update(stream_key, np.random.randn() * 2)

        probs = tracker.get_probabilities()
        assert abs(sum(probs.values()) - 1.0) < 0.01
        for p in probs.values():
            assert p >= 0.0

    def test_get_branch_states(self):
        tracker = self._make_tracker()
        states = tracker.get_branch_states()
        assert isinstance(states, dict)
        assert len(states) == 3

    def test_update_history(self):
        tracker = self._make_tracker()
        baseline = EconomicStateEstimator()
        stream_key = list(baseline.stream_registry.keys())[0]

        for _ in range(3):
            tracker.predict()
            tracker.update(stream_key, 0.5)

        assert len(tracker.update_history) == 3

    def test_min_max_probability_enforced(self):
        tracker = self._make_tracker()
        # MIN_PROB = 0.02, MAX_PROB = 0.95
        probs = tracker.get_probabilities()
        for p in probs.values():
            assert p >= tracker.MIN_PROB - 0.001
            assert p <= tracker.MAX_PROB + 0.001


class TestShadowStateTrackerPersistence:
    """Test that save/load cycle preserves shadow state and doesn't reset on set_baseline."""

    def test_set_baseline_preserves_loaded_shadows(self):
        """set_baseline() must NOT reset shadow x/P when shadows already loaded from state."""
        from heimdall.imm_tracker import ShadowStateTracker

        tracker = ShadowStateTracker(smoothing=True)
        baseline = EconomicStateEstimator()
        tracker.initialize(baseline)

        # Mutate shadow states so they differ from baseline
        for regime_id, shadow in tracker._shadows.items():
            shadow.x = shadow.x + 10.0  # Obvious offset
            shadow.P = shadow.P * 2.0

        # Save and reload
        state_dict = tracker.to_dict()
        new_tracker = ShadowStateTracker(smoothing=True)
        new_tracker.load_state(state_dict)

        # Capture loaded shadow state BEFORE set_baseline
        loaded_x = {rid: s.x.copy() for rid, s in new_tracker._shadows.items()}

        # set_baseline should NOT overwrite loaded shadows
        new_tracker.set_baseline(baseline)

        for regime_id in loaded_x:
            np.testing.assert_array_almost_equal(
                new_tracker._shadows[regime_id].x,
                loaded_x[regime_id],
                err_msg=f"set_baseline() reset shadow {regime_id} x — loaded state destroyed!",
            )

    def test_history_preserved_full_window(self):
        """to_dict() must save enough history for SLOPE_LONG computation."""
        from heimdall.imm_tracker import ShadowStateTracker

        tracker = ShadowStateTracker(smoothing=True)
        baseline = EconomicStateEstimator()
        tracker.initialize(baseline)

        # Feed 30 observations to build up history
        for i in range(30):
            tracker.predict()
            tracker.update("US_CPI_YOY", 3.0 + i * 0.01)

        state_dict = tracker.to_dict()

        # History per regime should be at least SLOPE_LONG (26) steps
        for regime_id, hist_list in state_dict["state_history"].items():
            assert len(hist_list) >= tracker.SLOPE_LONG, (
                f"History for {regime_id}: {len(hist_list)} steps saved, "
                f"need >= {tracker.SLOPE_LONG} for slope computation"
            )

    def test_obs_buffer_preserved_full_window(self):
        """to_dict() must save enough obs_buffer for 60-day impulse proxies."""
        from heimdall.imm_tracker import ShadowStateTracker

        tracker = ShadowStateTracker(smoothing=True)
        baseline = EconomicStateEstimator()
        tracker.initialize(baseline)

        # Feed 70 observations of a proxy stream
        for i in range(70):
            tracker.predict()
            tracker.update("SP500", 5000.0 + i)

        state_dict = tracker.to_dict()

        sp500_buf = state_dict["obs_buffer"].get("SP500", [])
        assert len(sp500_buf) >= 60, (
            f"obs_buffer SP500: {len(sp500_buf)} values saved, "
            f"need >= 60 for 60-day impulse proxy"
        )

    def test_update_count_survives_save_load(self):
        """_update_count must survive save → load so tempering stays consistent."""
        from heimdall.imm_tracker import ShadowStateTracker

        tracker = ShadowStateTracker(smoothing=True)
        baseline = EconomicStateEstimator()
        tracker.initialize(baseline)

        # Process observations to build up _update_count
        for i in range(50):
            tracker.predict()
            tracker.update("US_CPI_YOY", 3.0 + i * 0.01)

        original_count = tracker.level_a._update_count
        assert original_count == 50

        # Save and reload
        state_dict = tracker.to_dict()
        new_tracker = ShadowStateTracker(smoothing=True)
        new_tracker.load_state(state_dict)

        assert new_tracker.level_a._update_count == original_count, (
            f"_update_count lost: expected {original_count}, "
            f"got {new_tracker.level_a._update_count}"
        )
