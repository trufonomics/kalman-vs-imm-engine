"""Unit tests for Heimdall Kalman filter (Economic State Estimator)."""

import pytest
import numpy as np

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    StateEstimate,
    StateUpdate,
    FACTORS,
    STREAM_LOADINGS,
)


class TestEconomicStateEstimator:
    """Test the 8-factor Kalman filter."""

    def test_initialization(self):
        estimator = EconomicStateEstimator()
        state = estimator.get_state()
        assert isinstance(state, StateEstimate)
        assert len(state.mean) == 8
        assert state.covariance.shape == (8, 8)
        assert estimator.update_count == 0

    def test_factors_list(self):
        assert len(FACTORS) == 8
        assert "inflation_trend" in FACTORS
        assert "growth_trend" in FACTORS
        assert "labor_pressure" in FACTORS
        assert "housing_momentum" in FACTORS
        assert "financial_conditions" in FACTORS
        assert "commodity_pressure" in FACTORS
        assert "consumer_sentiment" in FACTORS
        assert "policy_stance" in FACTORS

    def test_stream_loadings_exist(self):
        assert len(STREAM_LOADINGS) > 0
        for key, loading in STREAM_LOADINGS.items():
            assert isinstance(key, str)
            assert isinstance(loading, dict)

    def test_stream_registry_populated(self):
        estimator = EconomicStateEstimator()
        assert len(estimator.stream_registry) > 0
        for key, (H_row, R) in estimator.stream_registry.items():
            assert isinstance(key, str)
            assert H_row.shape == (8,)
            assert isinstance(R, float)

    def test_predict_step(self):
        estimator = EconomicStateEstimator()
        state_before = estimator.x.copy()
        estimator.predict()
        # After predict, state should have changed (F @ x)
        # and covariance should be valid PSD
        eigenvalues = np.linalg.eigvalsh(estimator.P)
        assert np.all(eigenvalues >= -1e-10)

    def test_update_with_registered_stream(self):
        estimator = EconomicStateEstimator()
        # Use a real stream key from the registry
        stream_key = list(estimator.stream_registry.keys())[0]
        estimator.predict()
        update = estimator.update(stream_key, 0.5)
        assert update is not None
        assert isinstance(update, StateUpdate)
        assert estimator.update_count == 1

    def test_update_with_unknown_stream(self):
        estimator = EconomicStateEstimator()
        estimator.predict()
        update = estimator.update("NONEXISTENT_STREAM", 0.5)
        assert update is None
        assert estimator.update_count == 0

    def test_multiple_updates(self):
        estimator = EconomicStateEstimator()
        stream_key = list(estimator.stream_registry.keys())[0]

        for _ in range(10):
            estimator.predict()
            estimator.update(stream_key, 1.0)

        assert estimator.update_count == 10
        # State should have shifted from zero
        state = estimator.get_state()
        assert not np.allclose(state.mean, np.zeros(8))

    def test_covariance_stays_positive_semidefinite(self):
        estimator = EconomicStateEstimator()
        stream_key = list(estimator.stream_registry.keys())[0]

        for _ in range(50):
            estimator.predict()
            estimator.update(stream_key, np.random.randn())

        eigenvalues = np.linalg.eigvalsh(estimator.P)
        assert np.all(eigenvalues >= -1e-10)

    def test_recent_innovations_tracked(self):
        estimator = EconomicStateEstimator()
        stream_key = list(estimator.stream_registry.keys())[0]

        for i in range(5):
            estimator.predict()
            estimator.update(stream_key, float(i))

        assert len(estimator.recent_innovations) == 5

    def test_register_custom_stream(self):
        estimator = EconomicStateEstimator()
        estimator.register_stream(
            "CUSTOM_STREAM",
            {"inflation_trend": 0.8, "growth_trend": 0.2},
            noise=0.15,
        )
        assert "CUSTOM_STREAM" in estimator.stream_registry


class TestStateEstimate:
    def test_to_dict(self):
        mean = np.zeros(8)
        cov = np.eye(8)
        est = StateEstimate(mean=mean, covariance=cov)
        d = est.to_dict()
        assert "factors" in d
        assert "covariance" in d
        assert len(d["factors"]) == 8

    def test_from_dict_roundtrip(self):
        mean = np.array([0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.05, 0.15])
        cov = np.eye(8) * 0.05
        est = StateEstimate(mean=mean, covariance=cov)
        d = est.to_dict()
        est2 = StateEstimate.from_dict(d)
        np.testing.assert_allclose(est2.mean, mean, atol=1e-10)
