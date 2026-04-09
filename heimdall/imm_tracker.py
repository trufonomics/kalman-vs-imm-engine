"""
Level 2: Interacting Multiple Model (IMM) Branch Tracker.

Implements the full IMM algorithm (Blom & Bar-Shalom, 1988) for
macroeconomic regime detection. Each branch is a HYPOTHESIS about
the economy with its own Kalman filter. The IMM interaction step
mixes state estimates across branches before prediction, preventing
filter convergence while maintaining branch-specific dynamics.

Algorithm (per observation cycle):
    Step 1: INTERACTION — mix branch states via Markov TPM
    Step 2: PREDICT — each branch runs its Kalman prediction
    Step 3: UPDATE — each branch scores the observation (innovation likelihood)
    Step 4: COMBINE — Bayes' rule updates branch probabilities
    Step 5: TRIGGERS — check threshold crossings for LLM re-evaluation

The Markov Transition Probability Matrix (TPM) encodes regime persistence
and transition rates. It provides structural regularization: branches
always maintain minimum probability through transition priors, eliminating
the need for artificial probability clamping.

Ref: Blom & Bar-Shalom (1988) "The Interacting Multiple Model Algorithm
     for Systems with Markovian Switching Coefficients"
     IEEE Trans. Automatic Control, 33(8), 780-783.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.stats import norm

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    FACTORS,
    FACTOR_INDEX,
    N_FACTORS,
    StateUpdate,
)

logger = logging.getLogger(__name__)

# ── Branch adjustments derived from historical data ──────────────────
# Recession/stagflation: computed from mean factor z-scores during
# NBER-dated regime periods (scripts/derive_adjustments.py).
#
# Soft landing: explicit expansion signature added Mar 17 2026 to fix
# false positive rate. Root cause: with soft_landing = zero (baseline),
# recession won the Bayesian competition during benign low-inflation
# periods (inflation_trend & policy_stance both negative in expansions
# AND recessions). Adding positive growth/sentiment adjustments to
# soft_landing lets it compete, reducing FP from 30.4% → 20.6% while
# improving checkpoints from 10/11 → 11/11.
#
# 51-year backtest (1975-2026): 11/11 checkpoints, 20.6% FP rate.
# Recession: NBER 2008-01→2009-06 + 2020-02→2020-04 (79 weekly snapshots)
# Stagflation: CPI>5% period 2021-03→2022-09 (82 weekly snapshots)
# Soft landing: expansion signature (growth+sentiment positive, mild disinflation)

EMPIRICAL_BRANCH_ADJUSTMENTS = {
    "soft_landing": {
        # Explicit expansion signature — competes with recession during
        # benign low-inflation periods. Key: positive consumer_sentiment
        # and positive growth_trend distinguish expansion from recession,
        # while mild negative inflation_trend matches what expansions
        # actually look like (inflation below long-run trailing average).
        "inflation_trend": -0.200,       # healthy disinflation
        "growth_trend": 0.200,           # above-trend growth
        "labor_pressure": -0.150,        # low unemployment pressure
        "consumer_sentiment": 0.150,     # above-average confidence
        "policy_stance": -0.100,         # mildly accommodative
    },
    "stagflation": {
        "inflation_trend": 1.063,
        "growth_trend": 0.195,
        "labor_pressure": 0.320,
        "housing_momentum": 0.360,
        "financial_conditions": 0.247,
        "consumer_sentiment": 0.100,
        "policy_stance": 0.171,
    },
    "recession": {
        # Original values from derive_adjustments.py (2007-2026 NBER windows).
        # False positive analysis (Mar 17 2026) showed inflation_trend and
        # policy_stance fire during any low-inflation/easy-money period.
        # FP reduction handled via TPM stickiness, not adjustment scaling.
        "inflation_trend": -0.712,
        "growth_trend": -0.215,
        "labor_pressure": -0.421,       # KEY discriminator (3.5x stronger in real recessions)
        "housing_momentum": -0.329,
        "financial_conditions": -0.365,
        "consumer_sentiment": -0.154,   # KEY discriminator (near-zero in false positives)
        "policy_stance": -0.453,
        "commodity_pressure": -0.060,
    },
}

# ── VS-IMM: Hierarchical Regime Architecture (Mar 19 2026) ────────
#
# Level A: 3 macro regimes (calibrated 3x3 TPM, 51-year backtest)
# Level B: 2-3 sub-regimes nested within each Level A regime
#
# Joint probability: P(boom) = P(expansion) × P(boom|expansion)
#
# Sub-regime trackers activate only when parent exceeds ACTIVATION_THRESHOLD.
# This keeps computational cost bounded: typically 3 + 3 = 6 branch updates
# per observation during steady expansion.

HIERARCHICAL_ACTIVATION_THRESHOLD = 0.20

# Level A adjustments — renamed from soft_landing/recession
# Expansion has a REAL positive signature (not baseline-zero).
# This is the key FP fix: expansion actively competes with contraction.
LEVEL_A_ADJUSTMENTS = {
    "expansion": {
        # IDENTICAL to soft_landing adjustments — these values are tuned
        # from the Mar 17 fix (30.4% → 20.6% FP, 10/11 → 11/11 checkpoints).
        # Do NOT modify without re-running the 51-year backtest.
        "inflation_trend": -0.200,
        "growth_trend": 0.200,
        "labor_pressure": -0.150,
        "consumer_sentiment": 0.150,
        "policy_stance": -0.100,
    },
    "stagflation": {
        "inflation_trend": 1.063,
        "growth_trend": 0.195,
        "labor_pressure": 0.320,
        "housing_momentum": 0.360,
        "financial_conditions": 0.247,
        "consumer_sentiment": 0.100,
        "policy_stance": 0.171,
    },
    "contraction": {
        "inflation_trend": -0.712,
        "growth_trend": -0.215,
        "labor_pressure": -0.421,
        "housing_momentum": -0.329,
        "financial_conditions": -0.365,
        "consumer_sentiment": -0.154,
        "policy_stance": -0.453,
        "commodity_pressure": -0.060,
    },
}

# Level A TPM — same calibration as DEFAULT_TPM, renamed labels
# Branch order: [expansion, stagflation, contraction]
LEVEL_A_TPM = np.array([
    [0.970, 0.020, 0.010],   # Expansion: very sticky
    [0.030, 0.950, 0.020],   # Stagflation: can resolve or tip
    [0.050, 0.020, 0.930],   # Contraction: ends faster
], dtype=np.float64)

# Level B sub-regime adjustments — RELATIVE to the parent Level A adjustment
# These are deltas applied on top of the parent's adjustment vector.
# Derived from scripts/derive_sub_regime_adjustments.py against 1025 weekly
# snapshots (2005-2026). Each value = sub-regime raw mean z-score MINUS
# the Level A parent adjustment. These are DELTAS applied on top of the
# parent's adjustment vector.
#
# Do NOT hand-tune. Re-run derive_sub_regime_adjustments.py to update.

EXPANSION_SUB_ADJUSTMENTS = {
    "goldilocks": {
        # Mid-cycle calm: moderate growth, low inflation, stable labor.
        # Near the expansion parent centroid — small deltas.
        # Key discriminator: balanced, nothing extreme.
        "growth_trend": 0.050,
        "inflation_trend": -0.050,
        "consumer_sentiment": 0.100,
        "financial_conditions": 0.050,
    },
    "boom": {
        # Overheating: strong growth, rising inflation, tight labor,
        # hot housing & financial markets. Above-trend everything.
        # Key discriminators vs goldilocks: growth +, inflation +, housing +.
        "growth_trend": 0.300,
        "inflation_trend": 0.350,
        "labor_pressure": 0.250,
        "housing_momentum": 0.300,
        "financial_conditions": 0.150,
        "consumer_sentiment": 0.100,
    },
    "disinflation": {
        # Active easing cycle: inflation falling, policy tight (or easing),
        # growth moderate. Fed engineering soft landing.
        # Key discriminators: inflation NEGATIVE, policy POSITIVE (tight).
        "inflation_trend": -0.350,
        "policy_stance": 0.300,
        "growth_trend": -0.100,
        "labor_pressure": -0.100,
    },
}

STAGFLATION_SUB_ADJUSTMENTS = {
    "cost_push": {
        # Supply-driven: commodity surge (oil, food, energy) → inflation,
        # with weakening growth and tightening financial conditions.
        # Key discriminator vs demand_pull: commodity_pressure POSITIVE,
        # growth_trend NEGATIVE, financial_conditions NEGATIVE.
        "commodity_pressure": 0.350,
        "growth_trend": -0.300,
        "financial_conditions": -0.200,
        "consumer_sentiment": -0.150,
        "labor_pressure": -0.100,
    },
    "demand_pull": {
        # Demand-driven: fiscal/monetary excess → strong demand → inflation.
        # Key discriminator vs cost_push: growth POSITIVE, consumer strong,
        # commodities neutral.
        "growth_trend": 0.300,
        "consumer_sentiment": 0.200,
        "labor_pressure": 0.150,
        "commodity_pressure": -0.050,
        "financial_conditions": 0.100,
    },
}

CONTRACTION_SUB_ADJUSTMENTS = {
    "credit_crunch": {
        # Financial system stress → real economy drag. Banks failing, credit
        # freezing, contagion. Growth declines gradually but financial
        # conditions collapse. Housing stress is central.
        # Key discriminator vs demand_shock: financial MORE negative,
        # housing MORE negative, growth LESS extreme (gradual).
        "financial_conditions": -0.400,
        "housing_momentum": -0.300,
        "growth_trend": 0.100,
        "policy_stance": -0.200,
        "inflation_trend": 0.050,
    },
    "demand_shock": {
        # Sudden external hit — everything drops simultaneously and fast.
        # Growth collapses instantly, labor sheds, consumers freeze.
        # Key discriminator vs credit_crunch: growth MORE negative,
        # labor MORE negative, financial less extreme.
        "growth_trend": -0.350,
        "labor_pressure": -0.300,
        "consumer_sentiment": -0.200,
        "inflation_trend": -0.250,
        "financial_conditions": 0.050,
    },
}

# Level B TPMs
EXPANSION_SUB_TPM = np.array([
    [0.950, 0.030, 0.020],   # Goldilocks: very sticky
    [0.050, 0.920, 0.030],   # Boom: can cool to goldilocks or disinflation
    [0.060, 0.010, 0.930],   # Disinflation: resolves to goldilocks eventually
], dtype=np.float64)

STAGFLATION_SUB_TPM = np.array([
    [0.940, 0.060],   # Cost-push: can shift to demand-pull
    [0.080, 0.920],   # Demand-pull: can shift to cost-push
], dtype=np.float64)

CONTRACTION_SUB_TPM = np.array([
    [0.960, 0.040],   # Credit crunch: persistent
    [0.100, 0.900],   # Demand shock: ends fast
], dtype=np.float64)

# All sub-regime configs bundled for HierarchicalIMMTracker
SUB_REGIME_CONFIG = {
    "expansion": {
        "adjustments": EXPANSION_SUB_ADJUSTMENTS,
        "tpm": EXPANSION_SUB_TPM,
        "names": {"goldilocks": "Goldilocks", "boom": "Boom", "disinflation": "Disinflation"},
    },
    "stagflation": {
        "adjustments": STAGFLATION_SUB_ADJUSTMENTS,
        "tpm": STAGFLATION_SUB_TPM,
        "names": {"cost_push": "Cost-Push", "demand_pull": "Demand-Pull"},
    },
    "contraction": {
        "adjustments": CONTRACTION_SUB_ADJUSTMENTS,
        "tpm": CONTRACTION_SUB_TPM,
        "names": {"credit_crunch": "Credit Crunch", "demand_shock": "Demand Shock"},
    },
}

# Keep old adjustments available for backward compatibility
RECOMMENDED_BRANCH_ADJUSTMENTS = EMPIRICAL_BRANCH_ADJUSTMENTS

# ── Markov Transition Probability Matrix ─────────────────────────────
# Encodes regime persistence and transition rates.
# Row i, column j = P(transition to regime j | currently in regime i).
#
# These values are HEURISTIC, validated empirically against the 51-year
# backtest (11/11 checkpoints, 20.6% FP rate). They are NOT derived from
# NBER durations via a closed-form formula. NBER avg expansion (64 months)
# would imply ~0.9964 weekly persistence, but 0.970 works better in
# practice because the IMM needs faster mixing to detect transitions.
#
# Do NOT modify without re-running the 51-year backtest.
#
# Branch order: [soft_landing, stagflation, recession]
# Rows must sum to 1.0.
DEFAULT_TPM = np.array([
    [0.970, 0.020, 0.010],   # From soft landing: very sticky, rare to enter recession
    [0.030, 0.950, 0.020],   # From stagflation: can resolve to SL or tip to recession
    [0.050, 0.020, 0.930],   # From recession: ends faster (recoveries), rarely → stagflation
], dtype=np.float64)


@dataclass
class BranchModel:
    """A single branch's Kalman filter variant.

    Each branch has its own Kalman filter initialized from the Level 1
    baseline state + regime-specific adjustments. The IMM interaction
    step mixes states across branches before each prediction cycle,
    preventing filter convergence while preserving branch identity.
    """
    branch_id: str
    name: str
    probability: float

    # State adjustments shift this branch's estimate from the Level 1 baseline
    state_adjustments: dict[str, float] = field(default_factory=dict)

    # Transition overrides (factor-level persistence/drift changes)
    transition_overrides: list[dict] = field(default_factory=list)

    # This branch's Kalman filter
    estimator: Optional[EconomicStateEstimator] = field(default=None, repr=False)

    # Probability history for audit trail
    probability_history: list[dict] = field(default_factory=list)

    def initialize_from_baseline(self, baseline: EconomicStateEstimator):
        """Create this branch's Kalman variant from the shared Level 1 state."""
        self.estimator = EconomicStateEstimator()

        # Copy baseline state
        self.estimator.x = baseline.x.copy()
        self.estimator.P = baseline.P.copy()

        # Apply state adjustments
        for factor, adjustment in self.state_adjustments.items():
            idx = FACTOR_INDEX.get(factor)
            if idx is not None:
                self.estimator.x[idx] += adjustment

        # Apply transition overrides to F matrix
        for override in self.transition_overrides:
            factor = override.get("factor", "")
            idx = FACTOR_INDEX.get(factor)
            if idx is None:
                continue
            if "persistence" in override and override["persistence"] is not None:
                self.estimator.F[idx, idx] = override["persistence"]
            if "drift" in override and override["drift"] is not None:
                if not hasattr(self.estimator, '_drift'):
                    self.estimator._drift = np.zeros(N_FACTORS)
                self.estimator._drift[idx] = override["drift"]

    def to_dict(self) -> dict:
        """Serialize for JSONB storage."""
        return {
            "branch_id": self.branch_id,
            "name": self.name,
            "probability": self.probability,
            "state_adjustments": self.state_adjustments,
            "transition_overrides": self.transition_overrides,
            "kalman_state": self.estimator.to_dict() if self.estimator else None,
            "probability_history": self.probability_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BranchModel":
        """Deserialize from JSONB."""
        model = cls(
            branch_id=data["branch_id"],
            name=data["name"],
            probability=data["probability"],
            state_adjustments=data.get("state_adjustments", {}),
            transition_overrides=data.get("transition_overrides", []),
            probability_history=data.get("probability_history", []),
        )
        kalman = data.get("kalman_state")
        if kalman:
            model.estimator = EconomicStateEstimator()
            model.estimator.load_state(kalman)
        return model


@dataclass
class IMMUpdate:
    """Result of one IMM update cycle across all branches."""
    stream_key: str
    observed: float
    branch_scores: list[dict]
    triggers: list[dict]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "stream_key": self.stream_key,
            "observed": self.observed,
            "branch_scores": self.branch_scores,
            "triggers": self.triggers,
            "timestamp": self.timestamp,
        }


class IMMBranchTracker:
    """Level 2: Tracks scenario branch probabilities via full IMM.

    Implements the complete Blom & Bar-Shalom (1988) algorithm:
    1. Interaction: mix branch states via Markov transition probabilities
    2. Prediction: each branch runs its own Kalman predict
    3. Update: each branch scores observation via innovation likelihood
    4. Combination: Bayes' rule updates branch probabilities

    The Markov TPM provides structural regularization — branches maintain
    minimum probability through transition priors, replacing artificial
    clamping. The mixing step re-diversifies branch states every cycle,
    preventing the convergence problem that occurs with independent filters.

    Usage:
        tracker = IMMBranchTracker()
        tracker.initialize_branches(branches_jsonb, baseline_estimator)
        tracker.predict()            # IMM interaction + per-branch predict
        imm_update = tracker.update("US_CPI_YOY", normalized_value)
    """

    # Probability floor/ceiling — widened because TPM provides structural floors.
    # These are safety nets only, not the primary regularization mechanism.
    MIN_PROB = 0.005
    MAX_PROB = 0.99

    # Likelihood tempering for cold-start stability.
    # Standard technique in recursive Bayesian estimation (cf. tempered SMC
    # filters, Del Moral et al. 2006). Prevents wild swings when Kalman P
    # is large during the first observations.
    TEMPER_FLOOR = 0.3
    TEMPER_HORIZON = 200

    def __init__(self, tpm: Optional[np.ndarray] = None):
        self.branches: list[BranchModel] = []
        self.update_history: list[IMMUpdate] = []
        self._consecutive_drift: dict[str, int] = {}
        self._convergence_streak: dict[str, int] = {}
        self._baseline: Optional[EconomicStateEstimator] = None
        self._update_count: int = 0
        self._branch_order: list[str] = []  # maps index → branch_id
        self._adjustment_vectors: dict[str, np.ndarray] = {}

        # Markov transition probability matrix
        self.tpm = tpm if tpm is not None else DEFAULT_TPM.copy()

    def initialize_branches(
        self,
        branches_jsonb: list[dict],
        baseline: EconomicStateEstimator,
    ):
        """Initialize branch models from RedwoodTree.branches JSONB.

        Each branch gets its own Kalman filter initialized from the Level 1
        baseline + regime-specific state adjustments. The IMM interaction
        step will mix these states before each prediction cycle.
        """
        self._baseline = baseline
        self.branches = []
        self._branch_order = []

        for b in branches_jsonb:
            model = BranchModel(
                branch_id=b.get("id", b.get("branch_id", "")),
                name=b["name"],
                probability=b["probability"],
                state_adjustments=b.get("state_adjustments", {}),
                transition_overrides=b.get("transition_overrides", []),
                probability_history=b.get("probability_history", []),
            )
            model.initialize_from_baseline(baseline)
            self.branches.append(model)
            self._branch_order.append(model.branch_id)

        # Pre-compute adjustment vectors
        self._adjustment_vectors = {}
        for branch in self.branches:
            adj_vec = np.zeros(N_FACTORS)
            for factor, adj in branch.state_adjustments.items():
                idx = FACTOR_INDEX.get(factor)
                if idx is not None:
                    adj_vec[idx] = adj
            self._adjustment_vectors[branch.branch_id] = adj_vec

        # Resize TPM if branch count doesn't match default
        n = len(self.branches)
        if self.tpm.shape[0] != n:
            # Build a uniform TPM with high self-transition
            self.tpm = np.full((n, n), 0.02 / max(n - 1, 1))
            np.fill_diagonal(self.tpm, 0.98)
            # Normalize rows
            self.tpm /= self.tpm.sum(axis=1, keepdims=True)

        self._normalize_probabilities()

    def bootstrap_from_spec(
        self,
        spec: "ScenarioSpec",
        baseline: EconomicStateEstimator,
    ):
        """Initialize IMM from a ScenarioSpec (tree structure designed by Opus).

        This is the Step 1→Step 2 bridge in the restructured pipeline:
        Opus designs the tree (structure + adjustment vectors), then IMM
        bootstraps branch tracking with uniform priors.

        If a branch has probability=None (the new default), IMM assigns
        uniform prior (1/N). If probability is provided (legacy mode),
        it is used as-is.

        Does NOT modify the core IMM algorithm — only sets up initial state.
        """
        from heimdall.scenario_generator import ScenarioSpec, BranchSpec

        self._baseline = baseline
        self.branches = []
        self._branch_order = []

        n = len(spec.branches)
        uniform_prior = 1.0 / n if n > 0 else 1.0

        for branch_spec in spec.branches:
            # Use provided probability or uniform prior
            prob = branch_spec.probability if branch_spec.probability is not None else uniform_prior

            model = BranchModel(
                branch_id=branch_spec.id,
                name=branch_spec.name,
                probability=prob,
                state_adjustments=branch_spec.state_adjustments,
                transition_overrides=[
                    t.model_dump() for t in branch_spec.transition_overrides
                ],
                probability_history=[{
                    "probability": round(prob, 4),
                    "prior": None,
                    "trigger": "imm_bootstrap",
                    "reasoning": (
                        f"IMM bootstrap from tree design — "
                        f"{'uniform prior (1/{})'.format(n) if branch_spec.probability is None else 'LLM-provided prior'}"
                    ),
                    "streams_used": branch_spec.key_indicators,
                    "model": "imm_kalman",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            )
            model.initialize_from_baseline(baseline)
            self.branches.append(model)
            self._branch_order.append(model.branch_id)

        # Pre-compute adjustment vectors
        self._adjustment_vectors = {}
        for branch in self.branches:
            adj_vec = np.zeros(N_FACTORS)
            for factor, adj in branch.state_adjustments.items():
                idx = FACTOR_INDEX.get(factor)
                if idx is not None:
                    adj_vec[idx] = adj
            self._adjustment_vectors[branch.branch_id] = adj_vec

        # Build TPM — use default 3x3 if exactly 3 branches with matching names,
        # otherwise construct uniform TPM
        n = len(self.branches)
        names = [b.name.lower().replace(" ", "_").replace("-", "_") for b in self.branches]
        default_names = ["soft_landing", "stagflation", "recession"]

        if n == 3 and all(any(dn in name for dn in default_names) for name in names):
            # Map to default TPM order
            self.tpm = DEFAULT_TPM.copy()
        else:
            self.tpm = np.full((n, n), 0.02 / max(n - 1, 1))
            np.fill_diagonal(self.tpm, 0.98)
            self.tpm /= self.tpm.sum(axis=1, keepdims=True)

        self._normalize_probabilities()

        logger.info(
            f"IMM bootstrapped from spec: {n} branches, "
            f"priors={[round(b.probability, 3) for b in self.branches]}, "
            f"adjustments={[list(b.state_adjustments.keys()) for b in self.branches]}"
        )

    def predict(self):
        """IMM Step 1+2: Interaction (mixing) + per-branch prediction.

        Step 1 (Interaction): For each target branch j, compute a mixed
        initial condition by blending all branch states weighted by the
        Markov transition probabilities P(i→j). This re-diversifies
        branch states every cycle, preventing convergence.

        Step 2 (Prediction): Each branch runs its own Kalman predict
        from its mixed initial condition.

        Ref: Blom & Bar-Shalom (1988), Equations 2-5.
        """
        n = len(self.branches)
        if n == 0:
            return

        # If no baseline set, fall back to independent prediction
        if self._baseline is None:
            for branch in self.branches:
                if branch.estimator:
                    drift = getattr(branch.estimator, '_drift', None)
                    branch.estimator.predict()
                    if drift is not None:
                        branch.estimator.x += drift
            return

        # ── Step 1: IMM Interaction (Mixing) ──

        # Current probabilities as array
        mu = np.array([b.probability for b in self.branches])

        # Predicted mode probabilities: c_j = Σ_i TPM[i,j] × μ_i
        c_bar = self.tpm.T @ mu  # (n,)
        c_bar = np.maximum(c_bar, 1e-15)  # prevent division by zero

        # Mixing probabilities: μ(i|j) = TPM[i,j] × μ_i / c_j
        mixing_probs = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mixing_probs[i, j] = self.tpm[i, j] * mu[i] / c_bar[j]

        # Collect current branch states
        states = []
        covs = []
        for branch in self.branches:
            if branch.estimator:
                states.append(branch.estimator.x.copy())
                covs.append(branch.estimator.P.copy())
            else:
                states.append(self._baseline.x.copy())
                covs.append(self._baseline.P.copy())

        # Compute mixed initial conditions for each target branch j
        for j, branch in enumerate(self.branches):
            if branch.estimator is None:
                continue

            # Mixed state: x̄_j = Σ_i μ(i|j) × x_i
            x_mixed = np.zeros(N_FACTORS)
            for i in range(n):
                x_mixed += mixing_probs[i, j] * states[i]

            # Mixed covariance: P̄_j = Σ_i μ(i|j) × [P_i + (x_i - x̄_j)(x_i - x̄_j)']
            P_mixed = np.zeros((N_FACTORS, N_FACTORS))
            for i in range(n):
                diff = states[i] - x_mixed
                P_mixed += mixing_probs[i, j] * (covs[i] + np.outer(diff, diff))

            # Set mixed state as the starting point for prediction
            branch.estimator.x = x_mixed
            branch.estimator.P = P_mixed

        # ── Step 2: Per-branch Kalman prediction ──
        for branch in self.branches:
            if branch.estimator:
                drift = getattr(branch.estimator, '_drift', None)
                branch.estimator.predict()
                if drift is not None:
                    branch.estimator.x += drift

                # Re-apply branch identity after prediction to counteract
                # mixing-step contamination. Without this, mixing pulls all
                # branches toward the probability-weighted mean.
                adj_vec = self._adjustment_vectors.get(branch.branch_id, np.zeros(N_FACTORS))
                if np.any(adj_vec != 0):
                    # Regime branches: push toward characteristic state
                    branch.estimator.x += adj_vec * 0.10
                else:
                    # Baseline branch (soft_landing): actively pull toward
                    # zero (mean-reverting baseline). This is soft landing's
                    # identity — it predicts "return to normal."
                    branch.estimator.x *= 0.90  # 10% pull toward zero

    def update(self, stream_key: str, value: float) -> IMMUpdate:
        """IMM Steps 3+4: Per-branch measurement update + probability combination.

        Step 3: Each branch runs its own Kalman update and computes the
        innovation likelihood (how well it predicted this observation).

        Step 4: Bayes' rule combines likelihoods with prior probabilities.

        Each branch has its own P (covariance), so innovation variances
        differ across branches — a branch with higher uncertainty in the
        relevant factor will produce a wider likelihood, naturally
        penalizing uncertain models.
        """
        prob_before = {b.branch_id: b.probability for b in self.branches}
        branch_scores = []
        likelihoods = []

        # Check if any branch can process this stream
        registry = None
        for branch in self.branches:
            if branch.estimator and stream_key in branch.estimator.stream_registry:
                registry = branch.estimator.stream_registry
                break

        if registry is None and self._baseline and stream_key in self._baseline.stream_registry:
            registry = self._baseline.stream_registry

        if registry is None or stream_key not in registry:
            return IMMUpdate(
                stream_key=stream_key, observed=value,
                branch_scores=[], triggers=[],
            )

        H_row, R = registry[stream_key]

        for branch in self.branches:
            if branch.estimator is None:
                likelihoods.append(1e-10)
                continue

            est = branch.estimator

            # Branch-specific innovation variance (each branch has own P)
            S_j = float(H_row @ est.P @ H_row.T + R)
            if S_j <= 0:
                S_j = R

            # Innovation
            predicted = float(H_row @ est.x)
            innovation = value - predicted
            innovation_zscore = innovation / np.sqrt(S_j) if S_j > 0 else 0.0

            # Innovation likelihood: N(innovation | 0, S_j)
            likelihood = float(norm.pdf(innovation, loc=0, scale=np.sqrt(S_j)))
            likelihood = max(likelihood, 1e-10)
            likelihoods.append(likelihood)

            # Run per-branch Kalman update
            K = est.P @ H_row.T / S_j
            est.x = est.x + K * innovation
            I_KH = np.eye(N_FACTORS) - np.outer(K, H_row)
            est.P = I_KH @ est.P @ I_KH.T + np.outer(K, K) * R

            branch_scores.append({
                "branch_id": branch.branch_id,
                "name": branch.name,
                "predicted": predicted,
                "innovation": innovation,
                "innovation_zscore": innovation_zscore,
                "likelihood": likelihood,
                "S": S_j,
            })

        # ── Likelihood tempering during warmup ──
        # Standard in recursive Bayesian estimation (Del Moral et al., 2006).
        # Prevents wild swings when P is large during early observations.
        self._update_count += 1
        ramp = min(self._update_count / self.TEMPER_HORIZON, 1.0)
        exponent = self.TEMPER_FLOOR + (1.0 - self.TEMPER_FLOOR) * ramp
        tempered = [l ** exponent for l in likelihoods]

        # ── Bayes' rule: P(model_j | z) ∝ L_j^α × P(model_j) ──
        weighted = [
            l * b.probability
            for l, b in zip(tempered, self.branches)
        ]
        total = sum(weighted)

        if total > 0:
            for i, branch in enumerate(self.branches):
                branch.probability = weighted[i] / total
        self._clamp_probabilities()
        self._normalize_probabilities()

        # ── Record probability changes ──
        for score in branch_scores:
            bid = score["branch_id"]
            score["prob_before"] = prob_before.get(bid, 0)
            score["prob_after"] = next(
                (b.probability for b in self.branches if b.branch_id == bid), 0
            )

        now = datetime.now(timezone.utc).isoformat()
        for branch in self.branches:
            prior = prob_before.get(branch.branch_id, branch.probability)
            if abs(branch.probability - prior) > 0.001:
                branch.probability_history.append({
                    "probability": round(branch.probability, 4),
                    "prior": round(prior, 4),
                    "trigger": f"{stream_key} observation",
                    "reasoning": f"IMM Bayesian update from {stream_key}",
                    "streams_used": [stream_key],
                    "model": "imm_kalman",
                    "timestamp": now,
                })

        triggers = self._check_triggers(prob_before)

        imm_update = IMMUpdate(
            stream_key=stream_key,
            observed=value,
            branch_scores=branch_scores,
            triggers=triggers,
        )
        self.update_history.append(imm_update)

        if len(self.update_history) > 200:
            self.update_history = self.update_history[-200:]

        return imm_update

    def _check_triggers(self, prob_before: dict[str, float]) -> list[dict]:
        """Check for events that should trigger LLM re-evaluation."""
        triggers = []
        tiers = [0.25, 0.50, 0.75]

        for branch in self.branches:
            bid = branch.branch_id
            prior = prob_before.get(bid, branch.probability)
            current = branch.probability

            for tier in tiers:
                crossed_up = prior < tier <= current
                crossed_down = prior >= tier > current
                if crossed_up or crossed_down:
                    direction = "above" if crossed_up else "below"
                    triggers.append({
                        "type": "tier_crossing",
                        "branch_id": bid,
                        "branch_name": branch.name,
                        "tier": tier,
                        "direction": direction,
                        "probability": round(current, 4),
                        "detail": f"{branch.name} crossed {tier:.0%} ({direction}): {prior:.1%} → {current:.1%}",
                    })

            if current > prior + 0.001:
                direction = 1
            elif current < prior - 0.001:
                direction = -1
            else:
                direction = 0

            prev_drift = self._consecutive_drift.get(bid, 0)
            if direction != 0 and (
                (prev_drift > 0 and direction > 0) or
                (prev_drift < 0 and direction < 0)
            ):
                self._consecutive_drift[bid] = prev_drift + direction
            else:
                self._consecutive_drift[bid] = direction

            if abs(self._consecutive_drift.get(bid, 0)) >= 5:
                drift_dir = "upward" if self._consecutive_drift[bid] > 0 else "downward"
                triggers.append({
                    "type": "sustained_drift",
                    "branch_id": bid,
                    "branch_name": branch.name,
                    "consecutive": abs(self._consecutive_drift[bid]),
                    "direction": drift_dir,
                    "probability": round(current, 4),
                    "detail": f"{branch.name} has drifted {drift_dir} for {abs(self._consecutive_drift[bid])} consecutive updates",
                })

        for i, a in enumerate(self.branches):
            for b in self.branches[i + 1:]:
                key = f"{a.branch_id}:{b.branch_id}"
                spread = abs(a.probability - b.probability)

                if spread < 0.05:
                    self._convergence_streak[key] = self._convergence_streak.get(key, 0) + 1
                else:
                    self._convergence_streak[key] = 0

                if self._convergence_streak.get(key, 0) >= 10:
                    triggers.append({
                        "type": "convergence",
                        "branch_id_a": a.branch_id,
                        "branch_id_b": b.branch_id,
                        "consecutive": self._convergence_streak[key],
                        "detail": f"{a.name} ({a.probability:.1%}) and {b.name} ({b.probability:.1%}) within 5pp for {self._convergence_streak[key]} consecutive updates — consider merging",
                    })

        return triggers

    def _clamp_probabilities(self):
        """Safety clamp — widened because TPM provides structural floors."""
        for branch in self.branches:
            branch.probability = max(self.MIN_PROB, min(self.MAX_PROB, branch.probability))

    def _normalize_probabilities(self):
        """Ensure probabilities sum to 1.0."""
        total = sum(b.probability for b in self.branches)
        if total > 0:
            for branch in self.branches:
                branch.probability = branch.probability / total

    def get_probabilities(self) -> dict[str, float]:
        """Get current branch probabilities."""
        return {b.branch_id: round(b.probability, 4) for b in self.branches}

    def get_branch_states(self) -> dict[str, dict]:
        """Get current state estimates for all branches."""
        result = {}
        for branch in self.branches:
            if branch.estimator:
                state = branch.estimator.get_state()
                result[branch.branch_id] = {
                    "name": branch.name,
                    "probability": round(branch.probability, 4),
                    "factors": {
                        FACTORS[i]: round(float(state.mean[i]), 4)
                        for i in range(N_FACTORS)
                    },
                }
        return result

    def to_dict(self) -> dict:
        """Serialize full IMM state for JSONB storage."""
        return {
            "branches": [b.to_dict() for b in self.branches],
            "update_history": [u.to_dict() for u in self.update_history[-50:]],
            "consecutive_drift": dict(self._consecutive_drift),
            "convergence_streak": dict(self._convergence_streak),
            "update_count": self._update_count,
            "tpm": self.tpm.tolist(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def set_baseline(self, baseline: EconomicStateEstimator):
        """Set or update the shared Level 1 estimator reference.

        Used for initial calibration and by the coherence service to
        sync across Redwood trees. After setting baseline, branch filters
        are re-initialized from the new baseline + their adjustments.
        """
        self._baseline = baseline
        self._adjustment_vectors = {}
        for branch in self.branches:
            adj_vec = np.zeros(N_FACTORS)
            for factor, adj in branch.state_adjustments.items():
                idx = FACTOR_INDEX.get(factor)
                if idx is not None:
                    adj_vec[idx] = adj
            self._adjustment_vectors[branch.branch_id] = adj_vec

    def load_state(self, state_dict: dict):
        """Restore IMM state from JSONB.

        Note: call set_baseline() after this to reconnect the shared estimator.
        """
        self.branches = [
            BranchModel.from_dict(b)
            for b in state_dict.get("branches", [])
        ]
        self.update_history = [
            IMMUpdate(
                stream_key=u["stream_key"],
                observed=u["observed"],
                branch_scores=u.get("branch_scores", []),
                triggers=u.get("triggers", []),
                timestamp=u.get("timestamp", ""),
            )
            for u in state_dict.get("update_history", [])
        ]
        self._consecutive_drift = state_dict.get("consecutive_drift", {})
        self._convergence_streak = state_dict.get("convergence_streak", {})
        self._update_count = state_dict.get("update_count", 0)

        # Restore TPM
        tpm_data = state_dict.get("tpm")
        if tpm_data:
            self.tpm = np.array(tpm_data, dtype=np.float64)

        # Rebuild adjustment vectors and branch order
        self._adjustment_vectors = {}
        self._branch_order = []
        for branch in self.branches:
            adj_vec = np.zeros(N_FACTORS)
            for factor, adj in branch.state_adjustments.items():
                idx = FACTOR_INDEX.get(factor)
                if idx is not None:
                    adj_vec[idx] = adj
            self._adjustment_vectors[branch.branch_id] = adj_vec
            self._branch_order.append(branch.branch_id)


class HierarchicalIMMTracker:
    """VS-IMM: Variable-Structure Interacting Multiple Model tracker.

    Two-level hierarchy:
      Level A: Full IMM with 3 macro regimes (expansion, stagflation, contraction).
               Uses 3x3 TPM validated against 51-year backtest (heuristic, not NBER-derived).
               Algorithm: Kalman predict/update + Bayesian likelihood competition.

      Level B: Discriminant scoring for 2-3 sub-regimes within each Level A regime.
               Projects the parent branch's Kalman state onto sub-regime
               characteristic directions (delta adjustment vectors) and applies
               softmax. No separate Kalman filters — avoids the covariance
               convergence problem that makes IMM ineffective when branches
               share identical dynamics and differ by <1σ in state space.

    Joint probability:
      P(boom) = P(expansion) × P(boom | expansion)

    Why discriminant scoring for Level B:
      IMM discrimination requires innovation likelihood differences across
      branches. Innovation variance S = H@P@H' + R. When sub-branches share
      identical Kalman dynamics (F, Q matrices), their P converges, making S
      nearly identical. Sub-regime states differ by ~0.5 units vs measurement
      noise R≈1.0 → SNR=0.5, below Bayesian discrimination threshold.
      Discriminant scoring bypasses this by projecting the state directly
      onto the sub-regime's characteristic direction.

    Ref: VS-IMM Hierarchical Regime Architecture — Mar 19 2026 (Obsidian)
         Blom & Bar-Shalom (1988), Springer adaptive TPM (2023, 2025)
    """

    # Softmax temperature for sub-regime scoring. Controls decisiveness.
    # Lower = more decisive (picks one sub-regime), higher = more uniform.
    # T=1.0 with typical 0.5-2.0 score differences gives 62-88% dominant.
    SOFTMAX_TEMPERATURE = 0.5

    # Delta amplification: empirical deltas are mean z-score differences.
    # Amplification increases the discriminant weight without changing
    # the relative ordering. 2x is conservative.
    DELTA_AMPLIFICATION = 3.0

    def __init__(self):
        # Level A: macro regime tracker (full IMM)
        self.level_a = IMMBranchTracker(tpm=LEVEL_A_TPM.copy())

        # Level B: sub-regime discriminant vectors (precomputed on init)
        # Maps regime_id → {sub_id: np.array of amplified delta weights}
        self._sub_discriminants: dict[str, dict[str, np.ndarray]] = {}

        # Discrimination weights per factor per regime: how much does each
        # factor help separate sub-regimes? Computed from cross-sub variance.
        self._disc_weights: dict[str, np.ndarray] = {}

        # Track which sub-regimes are active (parent > threshold)
        self._sub_active: dict[str, bool] = {
            "expansion": False,
            "stagflation": False,
            "contraction": False,
        }

        # Cache last sub-regime probabilities for get_sub_probabilities()
        self._sub_probs_cache: dict[str, dict[str, float]] = {}

        self._baseline: Optional[EconomicStateEstimator] = None
        self._initialized = False

    def initialize(self, baseline: EconomicStateEstimator):
        """Initialize Level A (IMM) and Level B (discriminant vectors)."""
        self._baseline = baseline

        # Build Level A branches
        level_a_branches = []
        priors = {"expansion": 0.50, "stagflation": 0.25, "contraction": 0.25}

        for regime_id, adjustments in LEVEL_A_ADJUSTMENTS.items():
            level_a_branches.append({
                "branch_id": regime_id,
                "name": regime_id.replace("_", " ").title(),
                "probability": priors[regime_id],
                "state_adjustments": adjustments,
            })

        self.level_a.initialize_branches(level_a_branches, baseline)

        # Precompute Level B discriminant vectors and factor weights.
        for regime_id, config in SUB_REGIME_CONFIG.items():
            disc_vectors = {}
            # Collect all delta vectors to compute discrimination weights
            all_deltas = []
            for sub_id, delta_adj in config["adjustments"].items():
                vec = np.zeros(N_FACTORS)
                for factor, weight in delta_adj.items():
                    idx = FACTOR_INDEX.get(factor)
                    if idx is not None:
                        vec[idx] = weight * self.DELTA_AMPLIFICATION
                disc_vectors[f"{regime_id}_{sub_id}"] = vec
                all_deltas.append(vec)

            # Unit-normalize discriminant vectors
            for k in disc_vectors:
                norm = np.linalg.norm(disc_vectors[k])
                if norm > 1e-10:
                    disc_vectors[k] = disc_vectors[k] / norm

            self._sub_discriminants[regime_id] = disc_vectors

            # Compute per-factor discrimination weights: standard deviation
            # of each factor's delta across sub-regimes. Factors where subs
            # agree (low std) get low weight; factors where they disagree
            # (high std) get high weight. This focuses scoring on what
            # actually distinguishes sub-regimes.
            delta_matrix = np.array(all_deltas)  # (n_subs, n_factors)
            factor_stds = np.std(delta_matrix, axis=0)  # (n_factors,)
            # Normalize so weights sum to n_factors (preserves scale)
            total_std = np.sum(factor_stds)
            if total_std > 1e-10:
                self._disc_weights[regime_id] = factor_stds / total_std * N_FACTORS
            else:
                self._disc_weights[regime_id] = np.ones(N_FACTORS)

        self._initialized = True

        logger.info(
            "HierarchicalIMM initialized: Level A = IMM %s, "
            "Level B = discriminant scoring, threshold = %.0f%%",
            list(LEVEL_A_ADJUSTMENTS.keys()),
            HIERARCHICAL_ACTIVATION_THRESHOLD * 100,
        )

    def predict(self):
        """Run predict on Level A. Level B has no predict step (stateless scoring)."""
        if not self._initialized:
            return
        self.level_a.predict()

    def update(self, stream_key: str, value: float) -> IMMUpdate:
        """Run update on Level A, then score Level B sub-regimes.

        Returns the Level A IMMUpdate. Level B results accessible via
        get_sub_probabilities() or get_joint_probabilities().
        """
        if not self._initialized:
            return IMMUpdate(
                stream_key=stream_key, observed=value,
                branch_scores=[], triggers=[],
            )

        # Level A update (full IMM)
        level_a_result = self.level_a.update(stream_key, value)

        # Score Level B sub-regimes via discriminant scoring.
        # Always compute scores (even below threshold) so joint probs are
        # available for analytics. The activation flag controls UX display only.
        level_a_probs = self.level_a.get_probabilities()
        for regime_id in SUB_REGIME_CONFIG:
            parent_prob = level_a_probs.get(regime_id, 0)
            should_be_active = parent_prob >= HIERARCHICAL_ACTIVATION_THRESHOLD
            self._sub_active[regime_id] = should_be_active

            # Always score so joint probs are available for analytics/backtest.
            # The _sub_active flag controls UX display only.
            self._sub_probs_cache[regime_id] = self._score_sub_regimes(regime_id)

        return level_a_result

    def _score_sub_regimes(self, regime_id: str) -> dict[str, float]:
        """Score sub-regimes using discriminant-weighted proximity.

        Key insight: sub-regimes within a parent share most characteristics.
        Standard L2 distance is dominated by factors where they AGREE (high
        values but similar across subs) rather than factors where they DIFFER.
        The fix: weight each factor by its discrimination power (cross-sub
        standard deviation). Factors where sub-regimes disagree get amplified;
        factors where they agree get dampened.

        This focuses scoring on what actually distinguishes boom from goldilocks
        (inflation trajectory), or credit_crunch from demand_shock (financial
        stress vs. growth collapse depth), rather than the shared characteristics
        of the parent regime.

        This replaces the full IMM for Level B — the Kalman state from Level A
        already contains all the information we need. No separate filters needed.
        """
        config = SUB_REGIME_CONFIG.get(regime_id)
        if not config:
            return {}

        parent_branch = next(
            (b for b in self.level_a.branches if b.branch_id == regime_id),
            None,
        )
        if not parent_branch or not parent_branch.estimator:
            return {}

        state = parent_branch.estimator.x
        parent_adj = LEVEL_A_ADJUSTMENTS.get(regime_id, {})

        # Build parent target vector
        parent_vec = np.zeros(N_FACTORS)
        for factor, val in parent_adj.items():
            idx = FACTOR_INDEX.get(factor)
            if idx is not None:
                parent_vec[idx] = val

        # Per-factor discrimination weights (precomputed in initialize)
        weights = self._disc_weights.get(regime_id, np.ones(N_FACTORS))

        scores = {}
        for sub_id, delta_adj in config["adjustments"].items():
            full_sub_id = f"{regime_id}_{sub_id}"

            # Build sub-regime target: parent + amplified delta
            target = parent_vec.copy()
            for factor, delta in delta_adj.items():
                idx = FACTOR_INDEX.get(factor)
                if idx is not None:
                    target[idx] += delta * self.DELTA_AMPLIFICATION

            # Discriminant-weighted distance: emphasize factors that separate
            # this sub-regime from its siblings
            diff = state - target
            weighted_dist = np.sqrt(np.sum(weights * diff**2))
            scores[full_sub_id] = -weighted_dist

        # Softmax with temperature
        max_s = max(scores.values()) if scores else 0
        exp_scores = {
            k: np.exp((v - max_s) / self.SOFTMAX_TEMPERATURE)
            for k, v in scores.items()
        }
        total = sum(exp_scores.values())
        if total < 1e-15:
            n = len(exp_scores)
            return {k: round(1.0 / n, 4) for k in exp_scores}

        return {k: round(v / total, 4) for k, v in exp_scores.items()}

    def _activate_sub_tracker(self, regime_id: str):
        """Backward-compatible activation (used by tests). Now just marks active."""
        self._sub_active[regime_id] = True
        self._sub_probs_cache[regime_id] = self._score_sub_regimes(regime_id)

    def _deactivate_sub_tracker(self, regime_id: str):
        """Backward-compatible deactivation (used by tests)."""
        self._sub_active[regime_id] = False
        self._sub_probs_cache.pop(regime_id, None)

    def get_probabilities(self) -> dict[str, float]:
        """Get Level A probabilities."""
        return self.level_a.get_probabilities()

    def get_joint_probabilities(self) -> dict[str, float]:
        """Get joint probabilities: P(sub) = P(parent) × P(sub|parent).

        Returns flat dict with both Level A and Level B entries:
          {"expansion": 0.70, "expansion_goldilocks": 0.49, "expansion_boom": 0.14, ...}
        """
        result = {}
        level_a_probs = self.level_a.get_probabilities()

        for regime_id, parent_prob in level_a_probs.items():
            result[regime_id] = parent_prob

            sub_probs = self._sub_probs_cache.get(regime_id)
            if sub_probs:
                for sub_id, sub_prob in sub_probs.items():
                    result[sub_id] = round(parent_prob * sub_prob, 4)

        return result

    def get_sub_probabilities(self, regime_id: str) -> dict[str, float]:
        """Get conditional sub-regime probabilities P(sub|parent)."""
        if not self._sub_active.get(regime_id, False):
            return {}
        return dict(self._sub_probs_cache.get(regime_id, {}))

    def get_branch_states(self) -> dict[str, dict]:
        """Get state estimates for Level A and active Level B sub-regimes."""
        result = self.level_a.get_branch_states()

        level_a_probs = self.level_a.get_probabilities()
        for regime_id, active in self._sub_active.items():
            if not active:
                continue
            sub_probs = self._sub_probs_cache.get(regime_id, {})
            parent_prob = level_a_probs.get(regime_id, 0)
            for sub_id, sub_prob in sub_probs.items():
                result[sub_id] = {
                    "probability": sub_prob,
                    "joint_probability": round(parent_prob * sub_prob, 4),
                    "parent": regime_id,
                    "scoring": "discriminant",
                }

        return result

    def set_baseline(self, baseline: EconomicStateEstimator):
        """Update shared Level 1 estimator."""
        self._baseline = baseline
        self.level_a.set_baseline(baseline)

    def to_dict(self) -> dict:
        """Serialize full hierarchical state for JSONB storage."""
        return {
            "type": "hierarchical",
            "level_a": self.level_a.to_dict(),
            "sub_probs": dict(self._sub_probs_cache),
            "sub_active": dict(self._sub_active),
        }

    def load_state(self, state_dict: dict):
        """Restore hierarchical state from JSONB."""
        level_a_data = state_dict.get("level_a", {})
        self.level_a.load_state(level_a_data)
        self._initialized = True

        self._sub_active = state_dict.get("sub_active", {
            "expansion": False,
            "stagflation": False,
            "contraction": False,
        })

        self._sub_probs_cache = state_dict.get("sub_probs", {})

        # Rebuild discriminant vectors and weights if not already set
        if not self._sub_discriminants:
            for regime_id, config in SUB_REGIME_CONFIG.items():
                disc_vectors = {}
                all_deltas = []
                for sub_id, delta_adj in config["adjustments"].items():
                    vec = np.zeros(N_FACTORS)
                    for factor, weight in delta_adj.items():
                        idx = FACTOR_INDEX.get(factor)
                        if idx is not None:
                            vec[idx] = weight * self.DELTA_AMPLIFICATION
                    disc_vectors[f"{regime_id}_{sub_id}"] = vec
                    all_deltas.append(vec)

                # Unit-normalize
                for k in disc_vectors:
                    norm = np.linalg.norm(disc_vectors[k])
                    if norm > 1e-10:
                        disc_vectors[k] = disc_vectors[k] / norm
                self._sub_discriminants[regime_id] = disc_vectors

                # Rebuild discrimination weights
                delta_matrix = np.array(all_deltas)
                factor_stds = np.std(delta_matrix, axis=0)
                total_std = np.sum(factor_stds)
                if total_std > 1e-10:
                    self._disc_weights[regime_id] = factor_stds / total_std * N_FACTORS
                else:
                    self._disc_weights[regime_id] = np.ones(N_FACTORS)


class ParallelHierarchicalTracker:
    """Parallel architecture: Level A + independent Level B IMM trackers.

    Instead of scoring sub-regimes from the parent's Kalman state (discriminant),
    each Level B group runs its OWN IMMBranchTracker with its own Kalman filters.
    All trackers receive the same observations in parallel.

    Architecture:
      Level A:   [expansion | stagflation | contraction]     (3-branch IMM)
      Level B-1: [goldilocks | boom | disinflation]          (3-branch IMM, independent)
      Level B-2: [cost_push | demand_pull]                   (2-branch IMM, independent)
      Level B-3: [credit_crunch | demand_shock]              (2-branch IMM, independent)

    Joint probability: P(boom) = P_A(expansion) × P_B1(boom)

    Sub-regime adjustments are COMBINED: parent_adj + delta_adj. This gives each
    sub-regime branch a meaningful absolute position in factor space.
    """

    # Minimum probability floor for Level B branches. Prevents branch death
    # from covariance convergence. The TPM's mixing step alone isn't enough
    # for 2-branch trackers where sub-regime differences are small.
    LEVEL_B_PROB_FLOOR = 0.05

    def __init__(self):
        self.level_a = IMMBranchTracker(tpm=LEVEL_A_TPM.copy())
        self.level_b: dict[str, IMMBranchTracker] = {}
        self._baseline: Optional[EconomicStateEstimator] = None
        self._initialized = False

    def initialize(self, baseline: EconomicStateEstimator):
        """Initialize Level A and all Level B trackers."""
        self._baseline = baseline

        # Level A: unchanged
        level_a_branches = []
        priors = {"expansion": 0.50, "stagflation": 0.25, "contraction": 0.25}
        for regime_id, adjustments in LEVEL_A_ADJUSTMENTS.items():
            level_a_branches.append({
                "branch_id": regime_id,
                "name": regime_id.replace("_", " ").title(),
                "probability": priors[regime_id],
                "state_adjustments": adjustments,
            })
        self.level_a.initialize_branches(level_a_branches, baseline)

        # Level B: one independent IMM tracker per parent regime
        for regime_id, config in SUB_REGIME_CONFIG.items():
            parent_adj = LEVEL_A_ADJUSTMENTS.get(regime_id, {})
            tpm = config["tpm"].copy()
            tracker = IMMBranchTracker(tpm=tpm)

            branches = []
            n_subs = len(config["adjustments"])
            uniform_prior = round(1.0 / n_subs, 4)

            for sub_id, delta_adj in config["adjustments"].items():
                # Combined adjustment: parent + delta
                combined = dict(parent_adj)
                for factor, delta in delta_adj.items():
                    combined[factor] = combined.get(factor, 0.0) + delta

                branches.append({
                    "branch_id": f"{regime_id}_{sub_id}",
                    "name": config["names"].get(sub_id, sub_id),
                    "probability": uniform_prior,
                    "state_adjustments": combined,
                })

            tracker.initialize_branches(branches, baseline)
            self.level_b[regime_id] = tracker

        self._initialized = True
        logger.info(
            "ParallelHierarchicalIMM initialized: Level A = %s, "
            "Level B = %s parallel IMM trackers",
            list(LEVEL_A_ADJUSTMENTS.keys()),
            {k: len(v.branches) for k, v in self.level_b.items()},
        )

    def predict(self):
        """Run predict on ALL trackers in parallel."""
        if not self._initialized:
            return
        self.level_a.predict()
        for tracker in self.level_b.values():
            tracker.predict()

    def update(self, stream_key: str, value: float) -> IMMUpdate:
        """Run update on ALL trackers in parallel. Returns Level A result."""
        if not self._initialized:
            return IMMUpdate(
                stream_key=stream_key, observed=value,
                branch_scores=[], triggers=[],
            )

        level_a_result = self.level_a.update(stream_key, value)
        for tracker in self.level_b.values():
            tracker.update(stream_key, value)
            # Clamp Level B probabilities to prevent branch death
            self._clamp_probabilities(tracker)

        return level_a_result

    def _clamp_probabilities(self, tracker: IMMBranchTracker):
        """Prevent any Level B branch from dying below the floor."""
        needs_clamp = any(
            b.probability < self.LEVEL_B_PROB_FLOOR for b in tracker.branches
        )
        if not needs_clamp:
            return

        for branch in tracker.branches:
            if branch.probability < self.LEVEL_B_PROB_FLOOR:
                branch.probability = self.LEVEL_B_PROB_FLOOR

        # Renormalize
        total = sum(b.probability for b in tracker.branches)
        if total > 0:
            for branch in tracker.branches:
                branch.probability = branch.probability / total

    def get_probabilities(self) -> dict[str, float]:
        """Get Level A probabilities."""
        return self.level_a.get_probabilities()

    def get_sub_probabilities(self, regime_id: str) -> dict[str, float]:
        """Get conditional sub-regime probabilities from Level B tracker."""
        tracker = self.level_b.get(regime_id)
        if not tracker:
            return {}
        return tracker.get_probabilities()

    def get_joint_probabilities(self) -> dict[str, float]:
        """Get joint probabilities: P(sub) = P_A(parent) × P_B(sub)."""
        result = {}
        level_a_probs = self.level_a.get_probabilities()

        for regime_id, parent_prob in level_a_probs.items():
            result[regime_id] = parent_prob

            tracker = self.level_b.get(regime_id)
            if tracker:
                sub_probs = tracker.get_probabilities()
                for sub_id, sub_prob in sub_probs.items():
                    result[sub_id] = round(parent_prob * sub_prob, 4)

        return result

    def get_branch_states(self) -> dict[str, dict]:
        """Get state estimates for all trackers."""
        result = self.level_a.get_branch_states()
        level_a_probs = self.level_a.get_probabilities()

        for regime_id, tracker in self.level_b.items():
            parent_prob = level_a_probs.get(regime_id, 0)
            sub_probs = tracker.get_probabilities()
            for branch in tracker.branches:
                result[branch.branch_id] = {
                    "probability": sub_probs.get(branch.branch_id, 0),
                    "joint_probability": round(
                        parent_prob * sub_probs.get(branch.branch_id, 0), 4
                    ),
                    "parent": regime_id,
                    "scoring": "parallel_imm",
                    "factors": {
                        f: round(float(branch.estimator.x[i]), 4)
                        for f, i in FACTOR_INDEX.items()
                    } if branch.estimator else {},
                }

        return result

    def set_baseline(self, baseline: EconomicStateEstimator):
        """Update shared baseline for all trackers."""
        self._baseline = baseline
        self.level_a.set_baseline(baseline)
        for tracker in self.level_b.values():
            tracker.set_baseline(baseline)

    def to_dict(self) -> dict:
        """Serialize full parallel hierarchical state."""
        return {
            "type": "parallel_hierarchical",
            "level_a": self.level_a.to_dict(),
            "level_b": {
                regime_id: tracker.to_dict()
                for regime_id, tracker in self.level_b.items()
            },
        }

    def load_state(self, state_dict: dict):
        """Restore parallel hierarchical state from JSONB."""
        level_a_data = state_dict.get("level_a", {})
        self.level_a.load_state(level_a_data)

        for regime_id, tracker_data in state_dict.get("level_b", {}).items():
            if regime_id in self.level_b:
                self.level_b[regime_id].load_state(tracker_data)
            else:
                config = SUB_REGIME_CONFIG.get(regime_id)
                if config:
                    tracker = IMMBranchTracker(tpm=config["tpm"].copy())
                    tracker.load_state(tracker_data)
                    self.level_b[regime_id] = tracker

        self._initialized = True


class EnsembleHierarchicalTracker:
    """Ensemble: blends parallel IMM + discriminant scoring for Level B.

    Parallel IMM gives strong signal when sub-regimes are state-separable
    (cost-push vs demand-pull, boom vs goldilocks during their respective eras).
    Discriminant scoring provides stable fallback when IMM branches die from
    covariance convergence.

    P(sub|parent) = α × P_imm(sub|parent) + (1-α) × P_disc(sub|parent)

    This preserves the best of both: IMM's ability to independently track
    state + discriminant's resistance to branch death.
    """

    IMM_WEIGHT = 0.6

    def __init__(self):
        self._parallel = ParallelHierarchicalTracker()
        self._discriminant = HierarchicalIMMTracker()
        self._baseline: Optional[EconomicStateEstimator] = None
        self._initialized = False

    @property
    def level_a(self):
        return self._parallel.level_a

    def initialize(self, baseline: EconomicStateEstimator):
        self._baseline = baseline
        self._parallel.initialize(baseline)
        self._discriminant.initialize(baseline)
        self._initialized = True
        logger.info(
            "EnsembleHierarchicalIMM: α=%.1f IMM / %.1f disc",
            self.IMM_WEIGHT, 1 - self.IMM_WEIGHT,
        )

    def predict(self):
        if not self._initialized:
            return
        self._parallel.predict()
        self._discriminant.predict()

    def update(self, stream_key: str, value: float) -> IMMUpdate:
        if not self._initialized:
            return IMMUpdate(
                stream_key=stream_key, observed=value,
                branch_scores=[], triggers=[],
            )
        result = self._parallel.update(stream_key, value)
        self._discriminant.update(stream_key, value)
        return result

    def get_probabilities(self) -> dict[str, float]:
        return self._parallel.get_probabilities()

    def get_sub_probabilities(self, regime_id: str) -> dict[str, float]:
        imm_probs = self._parallel.get_sub_probabilities(regime_id)
        disc_probs = self._discriminant._sub_probs_cache.get(regime_id, {})

        if not imm_probs and not disc_probs:
            return {}

        all_keys = set(imm_probs.keys()) | set(disc_probs.keys())
        n = len(all_keys)
        uniform = 1.0 / n if n > 0 else 0

        α = self.IMM_WEIGHT
        blended = {}
        for k in all_keys:
            p_imm = imm_probs.get(k, uniform)
            p_disc = disc_probs.get(k, uniform)
            blended[k] = α * p_imm + (1 - α) * p_disc

        total = sum(blended.values())
        if total > 0:
            blended = {k: round(v / total, 4) for k, v in blended.items()}

        return blended

    def get_joint_probabilities(self) -> dict[str, float]:
        result = {}
        level_a_probs = self._parallel.get_probabilities()

        for regime_id, parent_prob in level_a_probs.items():
            result[regime_id] = parent_prob
            sub_probs = self.get_sub_probabilities(regime_id)
            for sub_id, sub_prob in sub_probs.items():
                result[sub_id] = round(parent_prob * sub_prob, 4)

        return result

    def get_branch_states(self) -> dict[str, dict]:
        return self._parallel.get_branch_states()

    def set_baseline(self, baseline: EconomicStateEstimator):
        self._baseline = baseline
        self._parallel.set_baseline(baseline)
        self._discriminant.set_baseline(baseline)

    def to_dict(self) -> dict:
        return {
            "type": "ensemble_hierarchical",
            "parallel": self._parallel.to_dict(),
            "discriminant": self._discriminant.to_dict(),
        }

    def load_state(self, state_dict: dict):
        parallel_data = state_dict.get("parallel", {})
        disc_data = state_dict.get("discriminant", {})
        if parallel_data:
            self._parallel.load_state(parallel_data)
        if disc_data:
            self._discriminant.load_state(disc_data)
        self._initialized = True


class ShadowStateTracker:
    """Shadow-state evidence scorecard scorer for Level B.

    Architecture:
      Level A:  3-branch IMM (unchanged, proven 11/12)
      Shadow:   3 non-competitive Kalman filters (one per parent regime)
                Always updated, never mixed — fresh state estimates at all times
      Level B:  Theory-coded additive evidence scorecards per subtype

    Key design principles:
    - Shadow filters solve stale-vs-dead: always fresh, no branch competition
    - Evidence scorecards replace geometry: interpretable additive terms, not
      distance-to-centroid
    - Path features (slopes, acceleration, innovations) solve snapshot-vs-trajectory
    - Observation proxies (oil impulse, PPI-CPI spread, SP500 drawdown) provide
      causal signal that latent factors alone cannot capture
    - Distance is ONE component (~20%), not the dominant scorer
    """

    # History windows for slope computation (step-based)
    SLOPE_SHORT = 5     # ~1 month
    SLOPE_MEDIUM = 13   # ~3 months
    SLOPE_LONG = 26     # ~6 months

    # Observation proxy buffer length
    OBS_BUFFER_LEN = 65  # ~3 months of daily data

    # Innovation rolling window
    INNOV_WINDOW = 5

    # Softmax temperature per family
    TEMPERATURE = {"expansion": 0.5, "stagflation": 0.5, "contraction": 0.5}

    # Distance penalty weight (low — evidence dominates)
    DIST_LAMBDA = {"expansion": 0.25, "stagflation": 0.15, "contraction": 0.25}

    # ── Sticky transition matrices per family ─────────────────────────
    # Moderate persistence, plausible transitions. Not ultra-sticky —
    # raw scorecard should do the heavy lifting, smoothing just reduces chatter.
    #
    # Expansion: goldilocks ↔ boom (moderate), boom → disinflation (plausible),
    #            disinflation → goldilocks (common end-state)
    EXP_TPM = np.array([
        #              gold  boom  disinfl
        [0.85, 0.10, 0.05],  # goldilocks
        [0.12, 0.78, 0.10],  # boom
        [0.15, 0.05, 0.80],  # disinflation
    ])
    EXP_SUBTYPES = ["expansion_goldilocks", "expansion_boom", "expansion_disinflation"]

    # Stagflation: mostly sticky, slow transitions
    STAG_TPM = np.array([
        #              cost_push  demand_pull
        [0.88, 0.12],  # cost_push
        [0.12, 0.88],  # demand_pull
    ])
    STAG_SUBTYPES = ["stagflation_cost_push", "stagflation_demand_pull"]

    # Contraction: moderately sticky
    CONT_TPM = np.array([
        #              credit_crunch  demand_shock
        [0.85, 0.15],  # credit_crunch
        [0.15, 0.85],  # demand_shock
    ])
    CONT_SUBTYPES = ["contraction_credit_crunch", "contraction_demand_shock"]

    FAMILY_CONFIG = {
        "expansion": {"tpm": EXP_TPM, "subtypes": EXP_SUBTYPES},
        "stagflation": {"tpm": STAG_TPM, "subtypes": STAG_SUBTYPES},
        "contraction": {"tpm": CONT_TPM, "subtypes": CONT_SUBTYPES},
    }

    def __init__(self, smoothing: bool = True):
        self.level_a = IMMBranchTracker(tpm=LEVEL_A_TPM.copy())
        self._shadows: dict[str, EconomicStateEstimator] = {}
        self._state_history: dict[str, list[np.ndarray]] = {}
        self._max_history = self.SLOPE_LONG + 5

        # Innovation tracking per shadow filter
        self._innovations: dict[str, list[dict]] = {}

        # Observation-side proxy buffers (raw values, not z-scores)
        self._obs_buffer: dict[str, list[float]] = {}
        self._proxy_streams = {
            "OIL_PRICE", "PPI", "US_CPI_YOY", "CORE_CPI",
            "SP500", "UNEMPLOYMENT_RATE", "INITIAL_CLAIMS",
            "FED_FUNDS_RATE", "10Y_YIELD", "HOME_PRICES",
            "CONSUMER_CONFIDENCE", "HOUSING_STARTS",
        }

        # Smoothing state
        self._smoothing = smoothing
        self._smooth_probs: dict[str, np.ndarray] = {}  # regime → last smoothed probs

        self._baseline: Optional[EconomicStateEstimator] = None
        self._initialized = False
        self._step_count = 0

    def set_tpm_diagonal(self, diag_strength: float):
        """Override TPM diagonal persistence for all families.

        Args:
            diag_strength: diagonal value (e.g. 0.85 = mild, 0.92 = stronger).
                Off-diagonal mass is split uniformly.
        """
        for regime_id, cfg in self.FAMILY_CONFIG.items():
            n = cfg["tpm"].shape[0]
            off_diag = (1.0 - diag_strength) / max(1, n - 1)
            new_tpm = np.full((n, n), off_diag)
            np.fill_diagonal(new_tpm, diag_strength)
            cfg["tpm"] = new_tpm

    def initialize(self, baseline: EconomicStateEstimator):
        """Initialize Level A and shadow filters."""
        self._baseline = baseline

        level_a_branches = []
        priors = {"expansion": 0.50, "stagflation": 0.25, "contraction": 0.25}
        for regime_id, adjustments in LEVEL_A_ADJUSTMENTS.items():
            level_a_branches.append({
                "branch_id": regime_id,
                "name": regime_id.replace("_", " ").title(),
                "probability": priors[regime_id],
                "state_adjustments": adjustments,
            })
        self.level_a.initialize_branches(level_a_branches, baseline)

        for regime_id, adjustments in LEVEL_A_ADJUSTMENTS.items():
            shadow = EconomicStateEstimator()
            shadow.x = baseline.x.copy()
            shadow.P = baseline.P.copy()
            for factor, adj in adjustments.items():
                idx = FACTOR_INDEX.get(factor)
                if idx is not None:
                    shadow.x[idx] += adj
            self._shadows[regime_id] = shadow
            self._state_history[regime_id] = [shadow.x.copy()]
            self._innovations[regime_id] = []

        for stream in self._proxy_streams:
            self._obs_buffer[stream] = []

        self._initialized = True
        logger.info(
            "ShadowStateTracker initialized: Level A = IMM, "
            "Level B = evidence scorecards with %d proxy streams",
            len(self._proxy_streams),
        )

    def predict(self):
        if not self._initialized:
            return
        self.level_a.predict()
        for shadow in self._shadows.values():
            shadow.predict()

    def update(self, stream_key: str, value: float) -> IMMUpdate:
        if not self._initialized:
            return IMMUpdate(
                stream_key=stream_key, observed=value,
                branch_scores=[], triggers=[],
            )

        level_a_result = self.level_a.update(stream_key, value)
        self._step_count += 1

        # Update shadow filters and capture innovations
        for regime_id, shadow in self._shadows.items():
            result = shadow.update(stream_key, value)

            # Store state snapshot (ensure entry exists — load_state may have missed it)
            if regime_id not in self._state_history:
                self._state_history[regime_id] = []
            history = self._state_history[regime_id]
            history.append(shadow.x.copy())
            if len(history) > self._max_history:
                self._state_history[regime_id] = history[-self._max_history:]

            # Store innovation
            if result is not None:
                if regime_id not in self._innovations:
                    self._innovations[regime_id] = []
                innov_list = self._innovations[regime_id]
                innov_list.append({
                    "stream": stream_key,
                    "innov": result.innovation,
                    "innov_z": result.innovation_zscore,
                })
                if len(innov_list) > self.INNOV_WINDOW * 3:
                    self._innovations[regime_id] = innov_list[-(self.INNOV_WINDOW * 3):]

        # Buffer raw observation for proxy features
        if stream_key in self._proxy_streams:
            buf = self._obs_buffer.get(stream_key, [])
            buf.append(value)
            if len(buf) > self.OBS_BUFFER_LEN:
                self._obs_buffer[stream_key] = buf[-self.OBS_BUFFER_LEN:]
            else:
                self._obs_buffer[stream_key] = buf

        return level_a_result

    # ── Feature computation ──────────────────────────────────────────

    def _compute_slopes(self, regime_id: str) -> dict[str, np.ndarray]:
        """Factor slopes over short/medium/long windows + acceleration."""
        history = self._state_history.get(regime_id, [])
        n = len(history)
        current = history[-1] if history else np.zeros(N_FACTORS)

        slopes = {}
        for name, window in [
            ("short", self.SLOPE_SHORT),
            ("medium", self.SLOPE_MEDIUM),
            ("long", self.SLOPE_LONG),
        ]:
            if n > window:
                slopes[name] = (current - history[-window - 1]) / window
            else:
                slopes[name] = np.zeros(N_FACTORS)

        slopes["acceleration"] = slopes["short"] - slopes["medium"]
        return slopes

    def _compute_innovation_features(self, regime_id: str) -> dict[str, float]:
        """Rolling innovation summaries from shadow filter residuals."""
        innovs = self._innovations.get(regime_id, [])
        recent = innovs[-self.INNOV_WINDOW * 2:] if innovs else []

        # Map stream keys to factor groups for aggregation
        inflation_streams = {"US_CPI_YOY", "CORE_CPI", "PPI"}
        growth_streams = {"RETAIL_SALES", "HOUSING_STARTS", "SP500"}

        infl_surprise = 0.0
        growth_surprise = 0.0
        stress_sum = 0.0
        count = 0

        for item in recent:
            z = item.get("innov_z", 0.0)
            stream = item.get("stream", "")
            stress_sum += abs(z)
            count += 1
            if stream in inflation_streams:
                infl_surprise += z
            if stream in growth_streams:
                growth_surprise += z

        return {
            "inflation_surprise_roll": infl_surprise,
            "growth_surprise_roll": growth_surprise,
            "stress_index": stress_sum / max(count, 1),
        }

    def _compute_obs_proxies(self) -> dict[str, float]:
        """Observation-side proxy features from raw value buffers."""
        proxies = {}

        def _impulse(stream: str, lookback: int) -> float:
            buf = self._obs_buffer.get(stream, [])
            if len(buf) < lookback + 1:
                return 0.0
            return sum(buf[-lookback:]) - sum(buf[-(lookback * 2):-lookback]) if len(buf) >= lookback * 2 else sum(buf[-lookback:])

        def _last(stream: str) -> float:
            buf = self._obs_buffer.get(stream, [])
            return buf[-1] if buf else 0.0

        def _change(stream: str, lookback: int) -> float:
            buf = self._obs_buffer.get(stream, [])
            if len(buf) < lookback + 1:
                return 0.0
            return buf[-1] - buf[-lookback - 1]

        def _drawdown(stream: str, lookback: int) -> float:
            """Cumulative return over lookback (for log-return streams)."""
            buf = self._obs_buffer.get(stream, [])
            if len(buf) < lookback:
                return 0.0
            return sum(buf[-lookback:])

        # Oil impulse (cumulative log return)
        proxies["oil_impulse_20"] = _drawdown("OIL_PRICE", 20)
        proxies["oil_impulse_60"] = _drawdown("OIL_PRICE", 60)

        # PPI - CPI spread (cost-push vs demand-pull discriminator)
        ppi_last = _last("PPI")
        cpi_last = _last("US_CPI_YOY")
        proxies["ppi_cpi_spread"] = ppi_last - cpi_last

        # Core-headline gap
        core_last = _last("CORE_CPI")
        proxies["core_headline_gap"] = core_last - cpi_last

        # SP500 return / drawdown
        proxies["sp500_ret_20"] = _drawdown("SP500", 20)
        proxies["sp500_ret_60"] = _drawdown("SP500", 60)

        # Unemployment change
        proxies["unrate_change"] = _change("UNEMPLOYMENT_RATE", 3)

        # Claims change
        proxies["claims_change"] = _change("INITIAL_CLAIMS", 5)

        # Fed funds change
        proxies["fedfunds_change_short"] = _change("FED_FUNDS_RATE", 3)
        proxies["fedfunds_change_medium"] = _change("FED_FUNDS_RATE", 10)

        # Yield curve proxy (10Y - fed funds)
        proxies["yield_curve"] = _last("10Y_YIELD") - _last("FED_FUNDS_RATE")

        # Housing change
        proxies["housing_change"] = _change("HOME_PRICES", 3)

        # Sentiment level
        proxies["sentiment_level"] = _last("CONSUMER_CONFIDENCE")

        return proxies

    def _build_features(self, regime_id: str) -> dict:
        """Build full feature bundle for a parent regime."""
        shadow = self._shadows[regime_id]
        slopes = self._compute_slopes(regime_id)
        innovs = self._compute_innovation_features(regime_id)
        proxies = self._compute_obs_proxies()

        # Factor levels from shadow state
        levels = {f: float(shadow.x[i]) for f, i in FACTOR_INDEX.items()}

        # Factor slopes (medium-term, most useful)
        slope_med = {f: float(slopes["medium"][i]) for f, i in FACTOR_INDEX.items()}
        slope_short = {f: float(slopes["short"][i]) for f, i in FACTOR_INDEX.items()}
        accel = {f: float(slopes["acceleration"][i]) for f, i in FACTOR_INDEX.items()}

        return {
            "levels": levels,
            "slope_med": slope_med,
            "slope_short": slope_short,
            "accel": accel,
            "innovs": innovs,
            "proxies": proxies,
            "step_count": self._step_count,
        }

    # ── Evidence scorecards ──────────────────────────────────────────
    # Each returns a dict of {subtype_id: raw_evidence_score}.
    # Scores are NOT probabilities yet — softmax is applied afterward.
    #
    # Design rules:
    # - 3-6 positive terms, 2-4 negative terms, 1 distance anchor
    # - Coarse weight tiers: 0.5, 1.0, 1.5, 2.0
    # - Distance is ~20% of typical score magnitude
    # - Each term should be interpretable

    def _score_expansion(self, features: dict) -> dict[str, float]:
        """Score goldilocks / boom / disinflation."""
        lv = features["levels"]
        sl = features["slope_med"]
        ac = features["accel"]
        iv = features["innovs"]
        px = features["proxies"]

        # Slopes from Kalman state are very small (0.001-0.01 range).
        # Amplify by 100x so they contribute meaningfully to scoring.
        S = 100.0  # slope amplifier

        # ── Goldilocks: moderate + stable ──
        gold = 0.0
        growth = lv["growth_trend"]
        infl = lv["inflation_trend"]
        # Positive: moderate growth (sweet spot, broad band)
        gold += 1.5 * max(0, min(growth + 0.5, 1.0))
        gold -= 0.5 * max(0, growth - 1.0)
        # Positive: inflation low or benign
        gold += 1.5 * max(0, 1.0 - abs(infl))
        # Positive: low policy stance (not under heavy tightening or easing)
        gold += 1.0 * max(0, 0.5 - abs(lv["policy_stance"]))
        # Positive: stability = low absolute slopes (amplified)
        infl_slope_abs = abs(sl["inflation_trend"] * S)
        gold += 1.0 * max(0, 1.0 - infl_slope_abs)
        gold += 0.5 * max(0, 1.0 - abs(sl["growth_trend"] * S))
        # Positive: low stress
        gold += 0.5 * max(0, 1.5 - iv["stress_index"])
        # Positive: positive sentiment
        gold += 0.5 * max(0, lv["consumer_sentiment"])
        # Negative: overheating signals
        gold -= 1.0 * max(0, lv["labor_pressure"] - 0.5)
        gold -= 0.5 * max(0, lv["housing_momentum"] - 0.8)
        # Negative: very high inflation (that's not goldilocks)
        gold -= 1.0 * max(0, infl - 0.8)

        # ── Boom: overheating OR asset-led exuberance ──
        boom = 0.0
        # Route 1: cyclical overheating
        boom += 1.5 * max(0, growth - 0.3)
        boom += 1.0 * max(0, sl["growth_trend"] * S)
        boom += 1.0 * max(0, lv["labor_pressure"])
        boom += 1.0 * max(0, infl + sl["inflation_trend"] * S)
        boom += 1.0 * max(0, lv["housing_momentum"])
        # Bonus: multiple simultaneous overheating indicators
        heat_count = sum([
            1 for v in [growth, lv["labor_pressure"], lv["housing_momentum"]]
            if v > 0.2
        ])
        boom += 1.0 * max(0, heat_count - 1)  # bonus when 2+ factors overheating
        # Route 2: asset-led froth (capped to prevent domination)
        boom += 1.0 * min(1.5, max(0, px["sp500_ret_20"] * 1.5))
        boom += 0.5 * max(0, lv["financial_conditions"])
        boom += 0.5 * max(0, lv["consumer_sentiment"])
        # Negative: only penalize STRONG disinflationary pattern (not mild slope)
        boom -= 1.0 * max(0, -sl["inflation_trend"] * S - 0.5)
        boom -= 1.0 * max(0, -lv["financial_conditions"] - 0.3)
        boom -= 0.5 * max(0, -lv["consumer_sentiment"] - 0.3)

        # ── Disinflation: inflation falling under restrictive policy ──
        disinfl = 0.0
        infl_slope_amp = sl["inflation_trend"] * S
        # Positive: falling inflation trajectory — MUST exceed threshold
        # Small wiggles don't count; need sustained decline > 0.15 (amplified)
        disinfl += 1.5 * max(0, -infl_slope_amp - 0.15)
        disinfl += 0.5 * max(0, -ac["inflation_trend"] * S - 0.1)
        # PREREQUISITE: inflation should be elevated or recently was
        # Even at 0.1 it counts — you can still be disinflating from 0.5→0.1
        disinfl += 1.0 * max(0, infl + 0.1)
        # Positive: restrictive policy (CRITICAL for disinflation identity)
        disinfl += 1.5 * max(0, lv["policy_stance"])
        disinfl += 1.0 * max(0, px["fedfunds_change_medium"])
        # Positive: growth still positive (soft landing, not crash)
        disinfl += 0.5 * max(0, growth)
        # Positive: negative inflation surprises
        disinfl += 0.5 * max(0, -iv["inflation_surprise_roll"])
        # Negative: rising inflation
        disinfl -= 2.0 * max(0, infl_slope_amp)
        # Negative: growth collapse (contraction, not disinflation)
        disinfl -= 1.0 * max(0, -growth - 0.3)
        # Negative: easy policy = not disinflation
        disinfl -= 1.0 * max(0, -lv["policy_stance"])
        # Negative: overheating signals (that's boom, not disinflation)
        overheating = (
            max(0, lv["labor_pressure"] - 0.3) +
            max(0, lv["housing_momentum"] - 0.3) +
            max(0, growth - 0.3)
        )
        disinfl -= 1.5 * max(0, overheating - 0.3)

        # Prior biases: goldilocks is "default" so gets negative bias
        # to prevent it from winning by moderation alone
        gold -= 1.5  # goldilocks penalty — must truly be stable

        scores = {
            "expansion_goldilocks": gold,
            "expansion_boom": boom,
            "expansion_disinflation": disinfl,
        }
        return scores

    def _score_stagflation(self, features: dict) -> dict[str, float]:
        """Score cost_push / demand_pull."""
        lv = features["levels"]
        sl = features["slope_med"]
        iv = features["innovs"]
        px = features["proxies"]
        S = 100.0

        # ── Cost-push: supply-driven inflation, weak growth ──
        cp = 0.0
        cp += 1.5 * max(0, lv["commodity_pressure"])
        cp += 1.5 * min(2.5, max(0, px["oil_impulse_20"] * 1.5))
        cp += 1.5 * max(0, px["ppi_cpi_spread"])
        cp += 1.0 * max(0, lv["inflation_trend"])
        cp += 1.0 * max(0, -lv["growth_trend"])
        cp += 0.5 * max(0, -sl["growth_trend"] * S)
        cp += 0.5 * max(0, -lv["consumer_sentiment"])
        cp -= 1.5 * max(0, lv["growth_trend"] - 0.3)
        cp -= 1.0 * max(0, lv["consumer_sentiment"] - 0.3)
        cp -= 0.5 * max(0, lv["labor_pressure"] - 0.3)

        # ── Demand-pull: demand-driven inflation, strong growth ──
        dp = 0.0
        dp += 1.5 * max(0, lv["growth_trend"])
        dp += 1.0 * max(0, lv["labor_pressure"])
        dp += 1.0 * max(0, lv["consumer_sentiment"])
        dp += 1.0 * max(0, lv["inflation_trend"])
        dp += 0.5 * min(1.5, max(0, px["sp500_ret_20"] * 1.0))
        dp += 0.5 * max(0, -px["ppi_cpi_spread"])
        dp -= 1.0 * min(2.0, max(0, px["oil_impulse_20"] * 1.0))
        dp -= 1.0 * max(0, lv["commodity_pressure"] - 0.5)
        dp -= 0.5 * max(0, -lv["growth_trend"])

        scores = {
            "stagflation_cost_push": cp,
            "stagflation_demand_pull": dp,
        }
        return scores

    def _score_contraction(self, features: dict) -> dict[str, float]:
        """Score credit_crunch / demand_shock."""
        lv = features["levels"]
        sl = features["slope_med"]
        sl_s = features["slope_short"]
        iv = features["innovs"]
        px = features["proxies"]
        S = 100.0

        # Finance-leads-growth: is finance the DOMINANT weakness?
        # Positive = finance weaker than growth (credit crunch pattern)
        fin_dominance = abs(lv["financial_conditions"]) - abs(lv["growth_trend"])

        # Breadth of real-economy collapse
        real_breadth = (
            max(0, -lv["growth_trend"]) +
            max(0, -lv["consumer_sentiment"]) +
            max(0, -lv["labor_pressure"]) +
            max(0, -lv["housing_momentum"])
        )

        # ── Credit crunch: finance-first seizure ──
        cc = 0.0
        cc += 1.5 * max(0, -lv["financial_conditions"])
        cc += 1.0 * max(0, -sl["financial_conditions"] * S)
        # SP500 drawdown — scaled by finance dominance
        sp500_drawdown = min(2.5, max(0, -px["sp500_ret_60"] * 1.0))
        finance_is_dominant = 1.0 if fin_dominance > 0.0 else 0.3
        cc += 1.0 * sp500_drawdown * finance_is_dominant
        # Finance DOMINATES over real economy weakness
        cc += 1.5 * max(0, fin_dominance)
        cc += 0.5 * max(0, -px["yield_curve"])
        # Housing collapse (GFC signature — housing-led financial crisis)
        cc += 1.0 * max(0, -lv["housing_momentum"])
        cc += 2.0 * max(0, -sl["housing_momentum"] * S)  # slope = GFC had sustained decline
        # Negative: real economy collapse much worse than finance
        cc -= 0.5 * max(0, -fin_dominance - 0.3)

        # ── Demand shock: abrupt broad real-economy freeze ──
        ds = 0.0
        # Positive: breadth of real-economy collapse (THE key signal)
        ds += 2.0 * real_breadth
        # Positive: speed of collapse
        ds += 1.5 * max(0, -sl_s["growth_trend"] * S)
        ds += 1.0 * max(0, -sl_s["consumer_sentiment"] * S)
        ds += 0.5 * max(0, -sl_s["labor_pressure"] * S)
        # Positive: broad innovation stress
        ds += 0.5 * max(0, iv["stress_index"] - 1.0)
        # Positive: unemployment / claims spikes
        ds += 0.5 * min(1.5, max(0, px["unrate_change"] * 2.0))
        ds += 0.5 * min(1.5, max(0, px["claims_change"] * 2.0))
        # Positive: real economy DOMINATES finance (not finance-led)
        ds += 1.5 * max(0, -fin_dominance)
        # Negative: if finance dominance is clearly the pattern
        ds -= 0.5 * max(0, fin_dominance - 0.3)

        scores = {
            "contraction_credit_crunch": cc,
            "contraction_demand_shock": ds,
        }
        return scores

    def _score_sub_regimes(self, regime_id: str) -> dict[str, float]:
        """Score sub-regimes using evidence scorecards + softmax + optional sticky smoothing.

        Smoothing formula: p_smooth_t ∝ p_raw_t · (TPM^T @ p_smooth_{t-1})
        The transition prior nudges probabilities toward persistent states
        without overriding strong scorecard evidence.
        """
        features = self._build_features(regime_id)

        if regime_id == "expansion":
            raw_scores = self._score_expansion(features)
        elif regime_id == "stagflation":
            raw_scores = self._score_stagflation(features)
        elif regime_id == "contraction":
            raw_scores = self._score_contraction(features)
        else:
            return {}

        if not raw_scores:
            return {}

        # Softmax with family-specific temperature
        temp = self.TEMPERATURE.get(regime_id, 0.5)
        max_s = max(raw_scores.values())
        exp_scores = {
            k: np.exp((v - max_s) / temp)
            for k, v in raw_scores.items()
        }
        total = sum(exp_scores.values())
        if total < 1e-15:
            n = len(exp_scores)
            raw_probs = {k: 1.0 / n for k in exp_scores}
        else:
            raw_probs = {k: v / total for k, v in exp_scores.items()}

        # ── Sticky smoothing ──────────────────────────────────────────
        if not self._smoothing:
            return {k: round(v, 4) for k, v in raw_probs.items()}

        family_cfg = self.FAMILY_CONFIG.get(regime_id)
        if not family_cfg:
            return {k: round(v, 4) for k, v in raw_probs.items()}

        tpm = family_cfg["tpm"]
        subtypes = family_cfg["subtypes"]

        # Convert raw probs to ordered array matching subtype order
        p_raw = np.array([raw_probs.get(s, 1.0 / len(subtypes)) for s in subtypes])

        if regime_id in self._smooth_probs:
            # Transition prior: what the TPM predicts from last smoothed state
            p_prev = self._smooth_probs[regime_id]
            transition_prior = tpm.T @ p_prev  # column = "where we'd go from prev"

            # Element-wise multiply: raw evidence × transition prior
            p_smoothed = p_raw * transition_prior

            # Renormalize
            sm_total = p_smoothed.sum()
            if sm_total > 1e-15:
                p_smoothed = p_smoothed / sm_total
            else:
                p_smoothed = p_raw
        else:
            # First step: no prior, use raw
            p_smoothed = p_raw

        self._smooth_probs[regime_id] = p_smoothed.copy()

        return {s: round(float(p_smoothed[i]), 4) for i, s in enumerate(subtypes)}

    # ── Pipeline-compatible interface ──────────────────────────────

    @property
    def branches(self):
        """Passthrough to Level A branches for pipeline compatibility."""
        return self.level_a.branches

    @property
    def update_history(self):
        """Passthrough to Level A update history for pipeline compatibility."""
        return self.level_a.update_history

    def initialize_branches(self, branches_jsonb: list[dict], baseline: "EconomicStateEstimator"):
        """Pipeline-compatible init: delegates to initialize() after setting up Level A."""
        self.initialize(baseline)
        # Re-init Level A with custom branches if provided
        if branches_jsonb:
            self.level_a.initialize_branches(branches_jsonb, baseline)
            self._branches_initialized = True

    def bootstrap_from_spec(self, spec, estimator):
        """Passthrough to Level A for scenario bootstrapping."""
        self.level_a.bootstrap_from_spec(spec, estimator)

    def get_probabilities(self) -> dict[str, float]:
        return self.level_a.get_probabilities()

    def get_sub_probabilities(self, regime_id: str) -> dict[str, float]:
        return self._score_sub_regimes(regime_id)

    def get_joint_probabilities(self) -> dict[str, float]:
        result = {}
        level_a_probs = self.level_a.get_probabilities()
        for regime_id, parent_prob in level_a_probs.items():
            result[regime_id] = parent_prob
            sub_probs = self._score_sub_regimes(regime_id)
            for sub_id, sub_prob in sub_probs.items():
                result[sub_id] = round(parent_prob * sub_prob, 4)
        return result

    def get_branch_states(self) -> dict[str, dict]:
        result = self.level_a.get_branch_states()
        level_a_probs = self.level_a.get_probabilities()
        for regime_id, shadow in self._shadows.items():
            parent_prob = level_a_probs.get(regime_id, 0)
            sub_probs = self._score_sub_regimes(regime_id)
            slopes = self._compute_slopes(regime_id)
            for sub_id, sub_prob in sub_probs.items():
                result[sub_id] = {
                    "probability": sub_prob,
                    "joint_probability": round(parent_prob * sub_prob, 4),
                    "parent": regime_id,
                    "scoring": "evidence_scorecard",
                    "factors": {
                        f: round(float(shadow.x[i]), 4)
                        for f, i in FACTOR_INDEX.items()
                    },
                    "slopes_medium": {
                        f: round(float(slopes["medium"][i]), 4)
                        for f, i in FACTOR_INDEX.items()
                    },
                }
        return result

    def set_baseline(self, baseline: EconomicStateEstimator):
        self._baseline = baseline
        self.level_a.set_baseline(baseline)
        # Only reset shadow states if they haven't been loaded from persisted state.
        # load_state() populates _shadows with restored x/P — overwriting those
        # with baseline values destroys the converged shadow filter states and
        # causes wild probability swings on warm restart.
        for regime_id, adjustments in LEVEL_A_ADJUSTMENTS.items():
            if regime_id not in self._shadows:
                shadow = EconomicStateEstimator()
                shadow.x = baseline.x.copy()
                shadow.P = baseline.P.copy()
                for factor, adj in adjustments.items():
                    idx = FACTOR_INDEX.get(factor)
                    if idx is not None:
                        shadow.x[idx] += adj
                self._shadows[regime_id] = shadow

    def to_dict(self) -> dict:
        return {
            "type": "shadow_state",
            "level_a": self.level_a.to_dict(),
            "shadows": {
                rid: {"x": s.x.tolist(), "P": s.P.tolist()}
                for rid, s in self._shadows.items()
            },
            "state_history": {
                rid: [h.tolist() for h in hist[-self._max_history:]]
                for rid, hist in self._state_history.items()
            },
            "obs_buffer": {
                k: v[-self.OBS_BUFFER_LEN:] for k, v in self._obs_buffer.items() if v
            },
        }

    def load_state(self, state_dict: dict):
        # Handle legacy IMMBranchTracker state (no "type" or type != "shadow_state")
        state_type = state_dict.get("type", "")
        if state_type != "shadow_state":
            # Legacy format: pass entire dict to Level A
            self.level_a.load_state(state_dict)
            self._initialized = True
            return

        level_a_data = state_dict.get("level_a", {})
        self.level_a.load_state(level_a_data)

        for regime_id, shadow_data in state_dict.get("shadows", {}).items():
            if regime_id in self._shadows:
                self._shadows[regime_id].x = np.array(shadow_data["x"])
                self._shadows[regime_id].P = np.array(shadow_data["P"])
            else:
                shadow = EconomicStateEstimator()
                shadow.x = np.array(shadow_data["x"])
                shadow.P = np.array(shadow_data["P"])
                self._shadows[regime_id] = shadow

            # Ensure _state_history and _innovations exist for every shadow
            if regime_id not in self._state_history:
                self._state_history[regime_id] = []
            if regime_id not in self._innovations:
                self._innovations[regime_id] = []

        # Restore state_history from saved state (slope computation needs it)
        for regime_id, hist_list in state_dict.get("state_history", {}).items():
            self._state_history[regime_id] = [np.array(h) for h in hist_list]

        for stream, vals in state_dict.get("obs_buffer", {}).items():
            self._obs_buffer[stream] = vals

        self._initialized = True
