"""
S3* Calibration Service — Tracks prediction accuracy with Brier scores.

Stores resolved branches with actual outcomes, computes Brier scores
and calibration curves per model and scenario type. Feeds back into
ModelFusion as weights (better Brier = higher weight).

Brier Score = (1/N) Σ (forecast_probability - outcome)²
  Perfect: 0.0
  Coin flip: 0.25
  Always wrong: 1.0

Calibration Curve: bucket forecasts by probability (0-10%, 10-20%, ...),
compare forecast % to actual % of outcomes in each bucket.

Ref: Implementation Architecture doc (S3* Calibration section)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResolvedBranch:
    """A branch whose outcome is now known."""

    tree_id: str
    branch_id: str
    branch_name: str
    scenario_type: str  # e.g. "tariff", "rate_hike", "recession"
    model_id: str  # which model generated this branch
    forecasted_probability: float  # what was predicted
    outcome: float  # 1.0 = happened, 0.0 = didn't happen
    resolved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    context: dict = field(default_factory=dict)

    def brier_contribution(self) -> float:
        """This branch's contribution to the Brier score."""
        return (self.forecasted_probability - self.outcome) ** 2

    def to_dict(self) -> dict:
        return {
            "tree_id": self.tree_id,
            "branch_id": self.branch_id,
            "branch_name": self.branch_name,
            "scenario_type": self.scenario_type,
            "model_id": self.model_id,
            "forecasted_probability": self.forecasted_probability,
            "outcome": self.outcome,
            "resolved_at": self.resolved_at,
            "brier_contribution": round(self.brier_contribution(), 6),
            "context": self.context,
        }


@dataclass
class CalibrationBucket:
    """One bucket in a calibration curve (e.g. 20-30% forecasts)."""

    lower: float
    upper: float
    count: int = 0
    sum_forecasted: float = 0.0
    sum_outcomes: float = 0.0

    @property
    def avg_forecasted(self) -> float:
        return self.sum_forecasted / self.count if self.count > 0 else 0.0

    @property
    def avg_outcome(self) -> float:
        return self.sum_outcomes / self.count if self.count > 0 else 0.0

    @property
    def deviation(self) -> float:
        """How far off calibration (0 = perfect)."""
        return abs(self.avg_forecasted - self.avg_outcome)

    def to_dict(self) -> dict:
        return {
            "range": f"{self.lower:.0%}-{self.upper:.0%}",
            "count": self.count,
            "avg_forecasted": round(self.avg_forecasted, 4),
            "avg_outcome": round(self.avg_outcome, 4),
            "deviation": round(self.deviation, 4),
        }


@dataclass
class ModelScore:
    """Aggregate score for one model."""

    model_id: str
    brier_score: float
    n_predictions: int
    calibration_curve: list[CalibrationBucket]
    by_scenario_type: dict[str, float]  # scenario_type → Brier score

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "brier_score": round(self.brier_score, 6),
            "n_predictions": self.n_predictions,
            "calibration_curve": [b.to_dict() for b in self.calibration_curve],
            "by_scenario_type": {
                k: round(v, 6) for k, v in self.by_scenario_type.items()
            },
        }


class CalibrationService:
    """Tracks prediction accuracy across models and scenario types.

    Usage:
        svc = CalibrationService()

        # When a branch resolves:
        svc.record_resolution(ResolvedBranch(
            tree_id="tree-1", branch_id="hawkish",
            branch_name="Hawkish Pivot", scenario_type="rate_hike",
            model_id="opus", forecasted_probability=0.65, outcome=1.0,
        ))

        # Get scores:
        scores = svc.compute_scores()
        brier_weights = svc.get_fusion_weights()
    """

    N_BUCKETS = 10

    def __init__(self):
        self.resolutions: list[ResolvedBranch] = []

    def record_resolution(self, resolution: ResolvedBranch):
        """Record a resolved branch outcome."""
        self.resolutions.append(resolution)
        logger.info(
            f"Recorded resolution: {resolution.branch_name} "
            f"(forecast={resolution.forecasted_probability:.2%}, "
            f"outcome={resolution.outcome})"
        )

    def record_batch(self, resolutions: list[ResolvedBranch]):
        """Record multiple resolutions at once."""
        self.resolutions.extend(resolutions)

    def compute_brier_score(
        self,
        model_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
    ) -> Optional[float]:
        """Compute Brier score, optionally filtered by model/scenario."""
        filtered = self._filter(model_id, scenario_type)
        if not filtered:
            return None
        return float(np.mean([r.brier_contribution() for r in filtered]))

    def compute_calibration_curve(
        self,
        model_id: Optional[str] = None,
    ) -> list[CalibrationBucket]:
        """Compute calibration curve for a model (or all models)."""
        filtered = self._filter(model_id)
        if not filtered:
            return []

        buckets = [
            CalibrationBucket(lower=i / self.N_BUCKETS, upper=(i + 1) / self.N_BUCKETS)
            for i in range(self.N_BUCKETS)
        ]

        for r in filtered:
            idx = min(int(r.forecasted_probability * self.N_BUCKETS), self.N_BUCKETS - 1)
            buckets[idx].count += 1
            buckets[idx].sum_forecasted += r.forecasted_probability
            buckets[idx].sum_outcomes += r.outcome

        return [b for b in buckets if b.count > 0]

    def compute_scores(self) -> list[ModelScore]:
        """Compute full scores for all models."""
        models = set(r.model_id for r in self.resolutions)
        scores = []

        for mid in sorted(models):
            model_resolutions = self._filter(mid)
            if not model_resolutions:
                continue

            brier = float(np.mean([r.brier_contribution() for r in model_resolutions]))
            curve = self.compute_calibration_curve(mid)

            # Per scenario type
            by_type: dict[str, list[float]] = defaultdict(list)
            for r in model_resolutions:
                by_type[r.scenario_type].append(r.brier_contribution())

            by_type_scores = {
                st: float(np.mean(contributions))
                for st, contributions in by_type.items()
            }

            scores.append(ModelScore(
                model_id=mid,
                brier_score=brier,
                n_predictions=len(model_resolutions),
                calibration_curve=curve,
                by_scenario_type=by_type_scores,
            ))

        return sorted(scores, key=lambda s: s.brier_score)

    def get_fusion_weights(self) -> dict[str, float]:
        """Get weights for ModelFusion based on Brier scores.

        weight = 1 / brier_score (lower Brier = higher weight).
        Returns normalized weights summing to 1.0.
        """
        scores = self.compute_scores()
        if not scores:
            return {}

        raw_weights = {}
        for s in scores:
            if s.brier_score > 0:
                raw_weights[s.model_id] = 1.0 / s.brier_score
            else:
                raw_weights[s.model_id] = 10.0  # Perfect score → high weight

        total = sum(raw_weights.values())
        if total <= 0:
            return {}

        return {k: round(v / total, 4) for k, v in raw_weights.items()}

    def get_reliability_index(self) -> float:
        """Overall reliability index: avg calibration deviation across all buckets.

        Lower = better calibrated.
        """
        curve = self.compute_calibration_curve()
        if not curve:
            return 1.0
        deviations = [b.deviation for b in curve if b.count >= 3]
        return float(np.mean(deviations)) if deviations else 1.0

    def to_dict(self) -> dict:
        """Serialize for JSONB storage."""
        return {
            "resolutions": [r.to_dict() for r in self.resolutions[-500:]],
            "scores": [s.to_dict() for s in self.compute_scores()],
            "fusion_weights": self.get_fusion_weights(),
            "reliability_index": round(self.get_reliability_index(), 4),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def load_state(self, state_dict: dict):
        """Restore from JSONB."""
        self.resolutions = []
        for r in state_dict.get("resolutions", []):
            self.resolutions.append(ResolvedBranch(
                tree_id=r["tree_id"],
                branch_id=r["branch_id"],
                branch_name=r["branch_name"],
                scenario_type=r["scenario_type"],
                model_id=r["model_id"],
                forecasted_probability=r["forecasted_probability"],
                outcome=r["outcome"],
                resolved_at=r.get("resolved_at", ""),
                context=r.get("context", {}),
            ))

    def _filter(
        self,
        model_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
    ) -> list[ResolvedBranch]:
        filtered = self.resolutions
        if model_id:
            filtered = [r for r in filtered if r.model_id == model_id]
        if scenario_type:
            filtered = [r for r in filtered if r.scenario_type == scenario_type]
        return filtered
