"""Unit tests for Heimdall Calibration Service."""

import pytest
from datetime import datetime, timezone

from heimdall.calibration_service import (
    CalibrationService,
    ResolvedBranch,
    CalibrationBucket,
    ModelScore,
)


class TestCalibrationService:
    """Test Brier score calibration."""

    def _make_branch(self, model_id: str, prob: float, outcome: int) -> ResolvedBranch:
        return ResolvedBranch(
            tree_id="tree_1",
            branch_id=f"branch_{prob}_{outcome}",
            branch_name=f"Test {prob}",
            scenario_type="inflation",
            model_id=model_id,
            forecasted_probability=prob,
            outcome=outcome,
            resolved_at=datetime.now(timezone.utc),
        )

    def test_empty_service(self):
        svc = CalibrationService()
        scores = svc.compute_scores()
        assert scores == []
        assert svc.get_reliability_index() == 1.0

    def test_perfect_calibration(self):
        svc = CalibrationService()
        svc.record_resolution(self._make_branch("opus", 1.0, 1))
        svc.record_resolution(self._make_branch("opus", 0.0, 0))

        score = svc.compute_brier_score("opus")
        assert score == 0.0

    def test_worst_calibration(self):
        svc = CalibrationService()
        svc.record_resolution(self._make_branch("opus", 1.0, 0))
        svc.record_resolution(self._make_branch("opus", 0.0, 1))

        score = svc.compute_brier_score("opus")
        assert score == 1.0

    def test_moderate_calibration(self):
        svc = CalibrationService()
        svc.record_resolution(self._make_branch("opus", 0.7, 1))
        svc.record_resolution(self._make_branch("opus", 0.3, 0))

        score = svc.compute_brier_score("opus")
        # (0.7-1)^2 + (0.3-0)^2 / 2 = (0.09 + 0.09) / 2 = 0.09
        assert abs(score - 0.09) < 0.001

    def test_multiple_models(self):
        svc = CalibrationService()
        svc.record_resolution(self._make_branch("opus", 1.0, 1))
        svc.record_resolution(self._make_branch("opus", 0.0, 0))
        svc.record_resolution(self._make_branch("sonar", 0.5, 1))
        svc.record_resolution(self._make_branch("sonar", 0.5, 0))

        scores = svc.compute_scores()
        assert len(scores) == 2
        opus_score = next(s for s in scores if s.model_id == "opus")
        sonar_score = next(s for s in scores if s.model_id == "sonar")
        assert opus_score.brier_score < sonar_score.brier_score

    def test_fusion_weights(self):
        svc = CalibrationService()
        svc.record_resolution(self._make_branch("opus", 1.0, 1))
        svc.record_resolution(self._make_branch("opus", 0.0, 0))
        svc.record_resolution(self._make_branch("sonar", 0.5, 1))
        svc.record_resolution(self._make_branch("sonar", 0.5, 0))

        weights = svc.get_fusion_weights()
        assert "opus" in weights
        assert "sonar" in weights
        assert weights["opus"] > weights["sonar"]
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_reliability_index(self):
        svc = CalibrationService()
        svc.record_resolution(self._make_branch("opus", 0.8, 1))
        svc.record_resolution(self._make_branch("opus", 0.2, 0))

        idx = svc.get_reliability_index()
        assert 0.0 <= idx <= 1.0

    def test_calibration_curve(self):
        svc = CalibrationService()
        for _ in range(20):
            svc.record_resolution(self._make_branch("opus", 0.8, 1))
            svc.record_resolution(self._make_branch("opus", 0.2, 0))
            svc.record_resolution(self._make_branch("opus", 0.5, 1))
            svc.record_resolution(self._make_branch("opus", 0.5, 0))

        curve = svc.compute_calibration_curve("opus")
        # Returns only non-empty buckets
        assert len(curve) > 0
        assert len(curve) <= 10
        for bucket in curve:
            assert isinstance(bucket, CalibrationBucket)
            assert 0.0 <= bucket.lower <= 1.0
            assert 0.0 <= bucket.upper <= 1.0
            assert bucket.count > 0

    def test_record_batch(self):
        svc = CalibrationService()
        branches = [
            self._make_branch("opus", 0.8, 1),
            self._make_branch("opus", 0.2, 0),
            self._make_branch("sonar", 0.6, 1),
        ]
        svc.record_batch(branches)
        assert len(svc.resolutions) == 3

    def test_serialization(self):
        svc = CalibrationService()
        svc.record_resolution(self._make_branch("opus", 0.7, 1))

        data = svc.to_dict()
        assert "resolutions" in data

        svc2 = CalibrationService()
        svc2.load_state(data)
        assert len(svc2.resolutions) == 1

    def test_by_scenario_type(self):
        svc = CalibrationService()
        svc.record_resolution(self._make_branch("opus", 0.8, 1))

        score = svc.compute_brier_score("opus", scenario_type="inflation")
        assert isinstance(score, float)

        score_other = svc.compute_brier_score("opus", scenario_type="growth")
        # No growth resolutions → None
        assert score_other is None

    def test_brier_score_no_data(self):
        svc = CalibrationService()
        assert svc.compute_brier_score("nonexistent") is None
