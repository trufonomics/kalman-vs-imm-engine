"""
TRUF Stream → Kalman Pipeline — Feeds real-time data into the economic state estimator.

Maps TRUF Network stream keys to Kalman filter observation keys,
fetches current values, normalizes them adaptively, and runs Kalman updates.

NORMALIZATION STRATEGY (Mar 24 2026):
  Static z-score normalization (calibrated March 6) broke for non-stationary series.
  CPI index levels ratchet up, oil prices regime-shift, BTC trends exponentially.
  Static mean/std → extreme z-scores → garbage Kalman state.

  New approach:
  1. TRANSFORM non-stationary series to stationary (pct_change for indices/levels)
  2. EWMA normalization for ALL series — adapts continuously, never stale
  3. SEED EWMA from March 6 static params → seamless migration, no cold start

  Each stream has a config: {transform, halflife_days, seed_mean, seed_std}.
  The AdaptiveNormalizer maintains running EWMA mean/variance per stream,
  persisted alongside Kalman state in JSONB.

Ref: Implementation Architecture doc (Level 1 section, Practical Implementation)
"""

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from heimdall.kalman_filter import (
    EconomicStateEstimator,
    StateUpdate,
    STREAM_LOADINGS,
)

logger = logging.getLogger(__name__)

# ── TRUF stream key → Kalman observation key mapping ──
# Maps KNOWN_STREAMS keys (truf_client.py) to STREAM_LOADINGS keys (kalman_filter.py)
# Only streams that have factor loadings defined are included.
TRUF_TO_KALMAN: dict[str, str] = {
    # ── Inflation (8 streams → 3 Kalman rows) ──
    # Only US-economy streams. International inflation needs its own factors.
    "US_INFLATION": "US_CPI_YOY",       # Truflation daily real-time CPI
    "BLS_CPI": "US_CPI_YOY",            # BLS headline CPI (monthly)
    "BLS_CPI_CORE": "CORE_CPI",         # BLS core CPI (monthly)
    "PCE_INDEX": "CORE_CPI",            # Fed's preferred measure (monthly)
    "PCE_CORE": "CORE_CPI",             # Core PCE (monthly)
    "TRUFLATION_PCE": "CORE_CPI",       # Truflation daily PCE proxy
    "PPI": "PPI",                       # Producer prices headline (monthly)
    "PPI_CORE": "PPI",                  # Core PPI (monthly)

    # ── Labor (7 streams → 3 Kalman rows) ──
    "UNEMPLOYMENT": "UNEMPLOYMENT_RATE",
    "INITIAL_CLAIMS": "INITIAL_CLAIMS",
    "CONTINUED_CLAIMS": "INITIAL_CLAIMS",
    "JOB_OPENINGS": "NONFARM_PAYROLLS",
    "JOB_HIRES": "NONFARM_PAYROLLS",
    "JOB_SEPARATIONS": "NONFARM_PAYROLLS",
    "WAGE_INFLATION": "NONFARM_PAYROLLS",

    # ── Housing (5 streams → 2 Kalman rows) ──
    "EXISTING_HOME_PRICE": "HOME_PRICES",
    "MORTGAGE_30YR": "HOUSING_STARTS",   # primary mortgage rate signal
    "FOR_SALE_INVENTORY": "HOUSING_STARTS",
    "MORTGAGE_DEBT": "HOME_PRICES",
    "RENTAL_VACANCY": "HOUSING_STARTS",

    # ── Retail / Consumer (4 streams → 2 Kalman rows) ──
    "TOTAL_RETAIL_SALES": "RETAIL_SALES",  # headline retail (monthly)
    "FOOD_SERVICE_SALES": "RETAIL_SALES",  # services spending (monthly)
    "CONSUMER_CONFIDENCE": "CONSUMER_CONFIDENCE",
    "BIG_MAC_US": "CONSUMER_CONFIDENCE",   # US purchasing power proxy

    # ── Commodities (4 streams → 3 Kalman rows) ──
    "CRUDE_OIL_BRENT": "OIL_PRICE",
    "NICKEL_FUTURES": "COPPER_PRICE",
    "EV_COMMODITY": "COPPER_PRICE",
    "SILICON_USD": "COMMODITY_PRESSURE_RAW",

    # ── Crypto (2 streams → 1 Kalman row) ──
    "BITCOIN": "BTC_USD",
    "TOTAL_MARKET_CAP": "BTC_USD",

    # ── Equities (1 stream → 1 Kalman row) ──
    "SP500": "SP500",
}


# ── Per-stream adaptive normalization config ─────────────────────────
#
# transform: How to make non-stationary series stationary before normalization.
#   "identity"   — Already stationary (rates, bounded percentages). Use raw value.
#   "pct_change" — Non-stationary level (price indices, nominal amounts).
#                  Computes (current - previous) / previous × 100 = % change.
#                  Requires a previous value; first observation after cold start
#                  uses seed_mean as the "previous" for a neutral initial z-score.
#
# halflife_days: EWMA halflife controlling how fast old observations are forgotten.
#   Shorter = more reactive to regime shifts but noisier.
#   Longer = smoother but slower to adapt.
#   Guidelines: macro rates (180d), labor/housing (120d), commodities (90d),
#               crypto/equities (60d).
#
# seed_mean, seed_std: Initial EWMA values from March 6 2026 calibration.
#   For "identity" streams: seed is in raw units (%, count, USD).
#   For "pct_change" streams: seed is in % change units (typical MoM change).
#   These ensure the system produces sensible output from the first observation
#   rather than returning neutral zeros during warm-up.
#
# WHY NOT just EWMA on raw levels for everything?
#   Monotonically increasing series (CPI index, retail sales) create a persistent
#   positive bias: EWMA mean always lags the actual value, so z-scores are always
#   slightly positive. For CPI (0.3 pts/month, halflife 180d), bias ≈ +1.3σ.
#   Converting to % change eliminates this because changes oscillate around zero.

@dataclass
class StreamConfig:
    """Configuration for adaptive normalization of a single TRUF stream."""
    transform: str        # "identity" or "pct_change"
    halflife_days: int    # EWMA halflife in days
    seed_mean: float      # Initial EWMA mean (from March 6 calibration)
    seed_std: float       # Initial EWMA std dev (from March 6 calibration)


# fmt: off
STREAM_CONFIG: dict[str, StreamConfig] = {
    # ── Inflation ──
    # US_INFLATION is Truflation's real-time YoY % → already stationary
    "US_INFLATION":   StreamConfig("identity",   180,  1.91,    0.45),
    # BLS/PCE/PPI are index LEVELS that ratchet up → pct_change
    # Seed mean/std are typical MoM % changes, not raw index levels
    "BLS_CPI":        StreamConfig("pct_change", 90,   0.20,    0.15),
    "BLS_CPI_CORE":   StreamConfig("pct_change", 90,   0.22,    0.12),
    "PCE_INDEX":      StreamConfig("pct_change", 90,   0.25,    0.15),
    "PCE_CORE":       StreamConfig("pct_change", 90,   0.25,    0.12),
    "TRUFLATION_PCE": StreamConfig("pct_change", 90,   0.12,    0.10),
    "PPI":            StreamConfig("pct_change", 90,   0.30,    0.25),
    "PPI_CORE":       StreamConfig("pct_change", 90,   0.30,    0.20),

    # ── Labor ──
    # Unemployment rate (%) → stationary
    "UNEMPLOYMENT":      StreamConfig("identity",   180,  4.30,      0.12),
    # Claims and JOLTS are bounded counts, quasi-stationary → identity with EWMA
    "INITIAL_CLAIMS":    StreamConfig("identity",   120,  224830.0,  12816.0),
    "CONTINUED_CLAIMS":  StreamConfig("identity",   120,  1906269.0, 43913.0),
    "JOB_OPENINGS":      StreamConfig("identity",   120,  7267.60,   342.97),
    "JOB_HIRES":         StreamConfig("identity",   120,  5326.60,   151.42),
    "JOB_SEPARATIONS":   StreamConfig("identity",   120,  5211.00,   86.44),
    # Wage "inflation" is actually avg hourly earnings in USD → trends up
    "WAGE_INFLATION":    StreamConfig("pct_change", 120,  0.30,      0.20),

    # ── Housing ──
    # Home prices in USD → trends up
    "EXISTING_HOME_PRICE": StreamConfig("pct_change", 120, 0.80,     0.60),
    # Mortgage rate (%) → stationary
    "MORTGAGE_30YR":       StreamConfig("identity",   180, 6.45,     0.29),
    # Inventory count → quasi-stationary, bounded
    "FOR_SALE_INVENTORY":  StreamConfig("identity",   120, 1259062.0, 117713.0),
    # Mortgage debt in trillions → trends up
    "MORTGAGE_DEBT":       StreamConfig("pct_change", 120, 0.40,     0.20),
    # Vacancy rate (%) → stationary
    "RENTAL_VACANCY":      StreamConfig("identity",   180, 7.10,     0.08),

    # ── Retail / Consumer ──
    # Retail sales in billions → trends up (nominal growth)
    "TOTAL_RETAIL_SALES":  StreamConfig("pct_change", 120, 0.30,     0.50),
    "FOOD_SERVICE_SALES":  StreamConfig("pct_change", 120, 0.35,     0.40),
    # Consumer confidence index → designed stationary (base ~100)
    "CONSUMER_CONFIDENCE": StreamConfig("identity",   180, 100.0,    15.0),
    # Big Mac price in USD → very slow trend, treat as quasi-stationary
    "BIG_MAC_US":          StreamConfig("identity",   180, 5.90,     0.16),

    # ── Commodities ──
    # Commodity prices fluctuate (not monotonically increasing) → identity + EWMA
    # EWMA adapts to regime shifts (e.g., oil $60→$90); shorter halflife
    "CRUDE_OIL_BRENT":    StreamConfig("identity",   90,  66.50,    4.02),
    "NICKEL_FUTURES":     StreamConfig("identity",   90,  15706.0,  1018.0),
    "EV_COMMODITY":       StreamConfig("identity",   120, 9490.0,   2516.0),
    "SILICON_USD":        StreamConfig("identity",   90,  786.29,   55.89),

    # ── Crypto ──
    # BTC and total market cap have strong long-term uptrend → identity + EWMA
    # Shorter halflife so EWMA adapts to bull/bear cycles
    "BITCOIN":            StreamConfig("identity",   60,  98465.0,  15205.0),
    "TOTAL_MARKET_CAP":   StreamConfig("identity",   60,  3.494e12, 5.239e11),

    # ── Equities ──
    # S&P has long-term uptrend but also drawdowns → identity + EWMA
    "SP500":              StreamConfig("identity",   90,  6372.04,  512.02),
}
# fmt: on


# Legacy Kalman-key normalization — used ONLY for denormalize_value() display purposes
# and as a last-resort fallback when no stream config exists.
NORMALIZATION: dict[str, dict[str, float]] = {
    "US_CPI_YOY": {"mean": 3.0, "std": 1.5},
    "CORE_CPI": {"mean": 3.5, "std": 1.0},
    "PPI": {"mean": 2.0, "std": 2.0},
    "NONFARM_PAYROLLS": {"mean": 200.0, "std": 100.0},
    "UNEMPLOYMENT_RATE": {"mean": 4.0, "std": 0.8},
    "INITIAL_CLAIMS": {"mean": 220.0, "std": 30.0},
    "HOUSING_STARTS": {"mean": 1400.0, "std": 200.0},
    "HOME_PRICES": {"mean": 400000.0, "std": 50000.0},
    "RETAIL_SALES": {"mean": 700.0, "std": 30.0},
    "CONSUMER_CONFIDENCE": {"mean": 100.0, "std": 15.0},
    "OIL_PRICE": {"mean": 75.0, "std": 15.0},
    "GOLD_PRICE": {"mean": 2000.0, "std": 200.0},
    "COPPER_PRICE": {"mean": 4.0, "std": 0.5},
    "FED_FUNDS_RATE": {"mean": 5.0, "std": 0.5},
    "10Y_YIELD": {"mean": 4.2, "std": 0.5},
    "SP500": {"mean": 5500.0, "std": 500.0},
    "BTC_USD": {"mean": 85000.0, "std": 15000.0},
    "COMMODITY_PRESSURE_RAW": {"mean": 50.0, "std": 20.0},
}


class AdaptiveNormalizer:
    """Exponentially weighted online normalizer for a single stream.

    Maintains running EWMA mean and variance, producing z-scores that
    adapt to non-stationary inputs without manual recalibration.

    EWMA update (Welford-style for numerical stability):
        delta = value - mean
        mean  += alpha * delta
        var   = (1 - alpha) * (var + alpha * delta^2)

    The halflife controls responsiveness:
        alpha = 1 - exp(-ln(2) / halflife_days)

    Seeded from March 6 calibration data so the system produces
    sensible z-scores from the very first observation.
    """

    # Minimum number of observations before EWMA stats are trusted.
    # Below this count, the seed values dominate.
    MIN_OBSERVATIONS = 3

    def __init__(self, halflife_days: int, seed_mean: float, seed_std: float):
        self.alpha = 1.0 - math.exp(-math.log(2) / max(halflife_days, 1))
        self._mean = seed_mean
        self._var = max(seed_std ** 2, 1e-10)
        self._count = 0
        # Store config for serialization
        self._halflife_days = halflife_days
        self._seed_mean = seed_mean
        self._seed_std = seed_std

    # Minimum std dev to prevent explosive z-scores when a series is
    # flat (e.g., fed funds rate held constant for months).
    _MIN_STD = 0.01

    def normalize(self, value: float) -> float:
        """Compute z-score from PRIOR stats, then update EWMA.

        Order matters (RiskMetrics convention):
          1. Compute z-score using OLD mean/variance (how surprising is this?)
          2. Update mean and variance with the new observation
        Computing z from the already-updated mean underestimates surprise
        because the mean has shifted toward the value.

        Variance update uses deviation from the OLD mean:
          var_new = (1-alpha) * var_old + alpha * (1-alpha) * delta^2
        This is the E[x^2] - E[x]^2 decomposition, numerically stable.
        """
        self._count += 1

        # Step 1: z-score from PRIOR statistics
        delta = value - self._mean
        std = max(math.sqrt(self._var), self._MIN_STD)
        z = delta / std

        # Step 2: update EWMA mean and variance
        self._mean += self.alpha * delta
        self._var = (1 - self.alpha) * (self._var + self.alpha * delta ** 2)

        return z

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return math.sqrt(max(self._var, 1e-10))

    def to_dict(self) -> dict:
        return {
            "mean": self._mean,
            "var": self._var,
            "count": self._count,
            "halflife_days": self._halflife_days,
            "seed_mean": self._seed_mean,
            "seed_std": self._seed_std,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptiveNormalizer":
        norm = cls(
            halflife_days=data.get("halflife_days", 120),
            seed_mean=data.get("seed_mean", 0.0),
            seed_std=data.get("seed_std", 1.0),
        )
        norm._mean = data.get("mean", norm._seed_mean)
        norm._var = data.get("var", norm._seed_std ** 2)
        norm._count = data.get("count", 0)
        return norm


def denormalize_value(kalman_key: str, normalized: float) -> float:
    """Convert a normalized Kalman value back to a raw scale.

    Uses the legacy NORMALIZATION dict for display purposes.
    """
    norm = NORMALIZATION.get(kalman_key)
    if not norm:
        return normalized
    return normalized * norm["std"] + norm["mean"]


class StreamPipeline:
    """Fetches TRUF streams and feeds them into the Kalman filter.

    Normalization is now adaptive: each stream has an EWMA normalizer
    that maintains running mean/variance, seeded from March 6 calibration.
    Non-stationary series (CPI indices, retail sales, etc.) are first
    transformed to % change, then normalized by EWMA.

    Usage:
        pipeline = StreamPipeline()
        pipeline.load_state(tree.kalman_state)  # restore from DB
        updates = await pipeline.run()           # fetch & update
        tree.kalman_state = pipeline.save_state() # persist back
    """

    # Minimum relative change to treat a value as "new".
    _VALUE_EPSILON = 1e-4
    _RAW_HISTORY_LIMIT = 128

    def __init__(self):
        self.estimator = EconomicStateEstimator()
        self._last_run: Optional[str] = None
        self._last_raw_values: dict[str, float] = {}
        self._raw_observation_history: list[dict] = []
        # Per-stream adaptive normalizers (created from STREAM_CONFIG)
        self._normalizers: dict[str, AdaptiveNormalizer] = {}
        self._init_normalizers()

    def _init_normalizers(self):
        """Create an AdaptiveNormalizer for each configured stream."""
        for truf_key, config in STREAM_CONFIG.items():
            self._normalizers[truf_key] = AdaptiveNormalizer(
                halflife_days=config.halflife_days,
                seed_mean=config.seed_mean,
                seed_std=config.seed_std,
            )

    def _get_config(self, truf_key: str) -> StreamConfig:
        """Get stream config, falling back to a sensible default."""
        config = STREAM_CONFIG.get(truf_key)
        if config:
            return config
        # Fallback: identity transform with moderate halflife
        return StreamConfig("identity", 120, 0.0, 1.0)

    def _get_normalizer(self, truf_key: str) -> AdaptiveNormalizer:
        """Get or create the EWMA normalizer for a stream."""
        if truf_key not in self._normalizers:
            config = self._get_config(truf_key)
            self._normalizers[truf_key] = AdaptiveNormalizer(
                halflife_days=config.halflife_days,
                seed_mean=config.seed_mean,
                seed_std=config.seed_std,
            )
        return self._normalizers[truf_key]

    def _transform_and_normalize(self, truf_key: str, raw_value: float) -> Optional[float]:
        """Apply stationarity transform + EWMA normalization.

        For "identity" streams: normalize raw value directly.
        For "pct_change" streams: compute % change from last value, then normalize.

        Returns None if the observation should be skipped (e.g., first pct_change
        observation with no prior value).
        """
        config = self._get_config(truf_key)
        normalizer = self._get_normalizer(truf_key)

        if config.transform == "pct_change":
            prev = self._last_raw_values.get(truf_key)
            if prev is None or prev == 0:
                # No previous value → can't compute change.
                # Store the raw value so next observation can compute change.
                # Don't feed this into the Kalman filter.
                return None
            pct_change = ((raw_value - prev) / abs(prev)) * 100.0
            return normalizer.normalize(pct_change)

        # "identity" — normalize raw value directly
        return normalizer.normalize(raw_value)

    def load_state(self, kalman_state: Optional[dict]):
        """Restore estimator and EWMA normalizers from JSONB."""
        if kalman_state:
            self.estimator.load_state(kalman_state)
            self._last_raw_values = kalman_state.get("_last_raw_values", {})
            self._raw_observation_history = [
                {
                    "timestamp": str(entry.get("timestamp", "")),
                    "values": {
                        str(key): float(value)
                        for key, value in (entry.get("values", {}) or {}).items()
                        if value is not None
                    },
                }
                for entry in kalman_state.get("_raw_observation_history", [])
                if isinstance(entry, dict) and isinstance(entry.get("values"), dict)
            ][-self._RAW_HISTORY_LIMIT:]
            # Restore EWMA normalizer state
            normalizer_state = kalman_state.get("_normalizers", {})
            for truf_key, norm_data in normalizer_state.items():
                self._normalizers[truf_key] = AdaptiveNormalizer.from_dict(norm_data)

    def save_state(self) -> dict:
        """Serialize estimator + EWMA normalizers for JSONB.

        Includes full innovation history for audit trail.
        """
        state = self.estimator.to_dict()
        state["last_pipeline_run"] = datetime.now(timezone.utc).isoformat()
        state["_last_raw_values"] = self._last_raw_values
        state["_raw_observation_history"] = self._raw_observation_history[-self._RAW_HISTORY_LIMIT:]
        # Persist EWMA normalizer state
        state["_normalizers"] = {
            key: norm.to_dict() for key, norm in self._normalizers.items()
        }
        return state

    def _record_raw_observations(
        self,
        raw_values: dict[str, float],
        timestamp: Optional[str] = None,
    ) -> None:
        """Persist a bounded audit trail of raw observations applied to the filter."""
        if not raw_values:
            return

        entry = {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "values": {str(key): float(value) for key, value in raw_values.items()},
        }
        self._raw_observation_history.append(entry)
        if len(self._raw_observation_history) > self._RAW_HISTORY_LIMIT:
            self._raw_observation_history = self._raw_observation_history[-self._RAW_HISTORY_LIMIT:]

    def _is_new_value(self, truf_key: str, raw_value: float) -> bool:
        """Check if a raw value is genuinely new (not a duplicate from last cycle)."""
        last = self._last_raw_values.get(truf_key)
        if last is None:
            return True
        if last == 0 and raw_value == 0:
            return False
        denom = max(abs(last), abs(raw_value), 1e-10)
        return abs(raw_value - last) / denom > self._VALUE_EPSILON

    async def run(self, stream_keys: Optional[list[str]] = None) -> list[StateUpdate]:
        """Fetch TRUF streams and run Kalman updates.

        Pipeline per stream:
          1. Fetch raw value from TRUF
          2. Check for mock data → skip
          3. Check for duplicate value → skip
          4. Apply stationarity transform (identity or pct_change)
          5. Normalize via EWMA → z-score
          6. Feed z-score into Kalman update

        Only runs predict+update when genuinely new data exists.
        """
        # NOTE: Requires TRUF client for live data fetching.
        # In standalone mode, use apply_raw_observations() instead of run().
        try:
            from app.services.truf_client import get_truf_client, KNOWN_STREAMS  # type: ignore
        except ImportError:
            raise ImportError(
                "StreamPipeline.run() requires the TRUF client. "
                "Use apply_raw_observations() for offline/standalone mode."
            )

        client = get_truf_client()
        updates: list[StateUpdate] = []

        keys_to_fetch = stream_keys or list(TRUF_TO_KALMAN.keys())

        new_observations: list[tuple[str, str, float, float]] = []
        skipped = 0
        transformed_skipped = 0

        # Build fetch tasks — resolve keys once, then fetch all streams in parallel
        fetch_items: list[tuple[str, str, dict]] = []
        for truf_key in keys_to_fetch:
            kalman_key = TRUF_TO_KALMAN.get(truf_key)
            if not kalman_key:
                continue
            stream_meta = KNOWN_STREAMS.get(truf_key)
            if not stream_meta:
                continue
            fetch_items.append((truf_key, kalman_key, stream_meta))

        async def _fetch_one(truf_key: str, meta: dict) -> tuple[str, dict | None]:
            try:
                record = await client.get_stream_record(
                    provider=meta["provider"],
                    stream_id=meta["streamId"],
                )
                return (truf_key, record)
            except Exception as e:
                logger.warning(f"Failed to fetch stream {truf_key}: {e}")
                return (truf_key, None)

        # Parallel fetch — all TRUF API calls run concurrently
        fetch_results = await asyncio.gather(
            *[_fetch_one(truf_key, meta) for truf_key, _, meta in fetch_items]
        )

        # Process results sequentially (cheap CPU work, no I/O)
        results_map = dict(fetch_results)
        for truf_key, kalman_key, _ in fetch_items:
            record = results_map.get(truf_key)
            if record is None:
                continue

            raw_value = float(record.get("value", 0))
            if record.get("is_mock"):
                continue  # Never feed mock data into the Kalman filter

            if not self._is_new_value(truf_key, raw_value):
                skipped += 1
                continue

            # Apply transform + adaptive normalization
            normalized = self._transform_and_normalize(truf_key, raw_value)

            # Store raw value BEFORE checking normalized result
            # (pct_change streams need the raw value stored for next cycle)
            self._last_raw_values[truf_key] = raw_value

            if normalized is None:
                # First observation for a pct_change stream — stored for next time
                transformed_skipped += 1
                continue

            new_observations.append((truf_key, kalman_key, raw_value, normalized))

        # Only predict + update if we have genuinely new data
        if new_observations:
            self.estimator.predict()

            for truf_key, kalman_key, raw_value, normalized in new_observations:
                update = self.estimator.update(kalman_key, normalized)
                if update:
                    updates.append(update)
                    logger.debug(
                        f"Kalman update: {truf_key}→{kalman_key} "
                        f"raw={raw_value:.4f} norm={normalized:.4f} "
                        f"innovation_z={update.innovation_zscore:.2f}"
                    )
            self._record_raw_observations({truf_key: raw_value for truf_key, _, raw_value, _ in new_observations})
        else:
            logger.info("No new observations — skipping predict+update cycle")

        self._last_run = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"Pipeline run complete: {len(updates)} updates, "
            f"{sum(1 for u in updates if u.is_anomalous)} anomalous"
            f"{f' ({skipped} unchanged)' if skipped else ''}"
            f"{f' ({transformed_skipped} first-obs pct_change)' if transformed_skipped else ''}"
        )

        return updates

    def apply_raw_observations(
        self,
        raw_values: dict[str, float],
        stream_keys: Optional[list[str]] = None,
        *,
        timestamp: Optional[str] = None,
        record_history: bool = True,
    ) -> list[StateUpdate]:
        """Replay raw stream values into the Kalman filter without fetching.

        This is used when a custom scenario tree needs to catch up to the
        latest engine readings from persisted warm state, rather than making
        another network-bound TRUF fetch cycle.
        """
        updates: list[StateUpdate] = []
        keys_to_apply = stream_keys or list(TRUF_TO_KALMAN.keys())
        new_observations: list[tuple[str, str, float, float]] = []

        skipped = 0
        transformed_skipped = 0

        for truf_key in keys_to_apply:
            kalman_key = TRUF_TO_KALMAN.get(truf_key)
            if not kalman_key or truf_key not in raw_values:
                continue

            raw_value = float(raw_values[truf_key])
            if not self._is_new_value(truf_key, raw_value):
                skipped += 1
                continue

            normalized = self._transform_and_normalize(truf_key, raw_value)
            self._last_raw_values[truf_key] = raw_value

            if normalized is None:
                transformed_skipped += 1
                continue

            new_observations.append((truf_key, kalman_key, raw_value, normalized))

        if new_observations:
            self.estimator.predict()
            for truf_key, kalman_key, raw_value, normalized in new_observations:
                update = self.estimator.update(kalman_key, normalized)
                if update:
                    updates.append(update)
                    logger.debug(
                        f"Kalman replay: {truf_key}→{kalman_key} "
                        f"raw={raw_value:.4f} norm={normalized:.4f} "
                        f"innovation_z={update.innovation_zscore:.2f}"
                    )
            if record_history:
                self._record_raw_observations(
                    {truf_key: raw_value for truf_key, _, raw_value, _ in new_observations},
                    timestamp=timestamp,
                )
        else:
            logger.info("No changed raw observations to replay into Kalman state")

        self._last_run = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"Replay complete: {len(updates)} updates"
            f"{f' ({skipped} unchanged)' if skipped else ''}"
            f"{f' ({transformed_skipped} first-obs pct_change)' if transformed_skipped else ''}"
        )
        return updates

    async def run_single(self, truf_key: str) -> Optional[StateUpdate]:
        """Fetch a single TRUF stream and run one Kalman update."""
        results = await self.run(stream_keys=[truf_key])
        return results[0] if results else None

    def get_factor_summary(self) -> dict[str, dict[str, float]]:
        """Get current factor estimates with uncertainty."""
        state = self.estimator.get_state()
        summary = {}
        for i, name in enumerate(self.estimator.factor_names if hasattr(self.estimator, 'factor_names') else []):
            value = float(state.mean[i])
            uncertainty = float(state.covariance[i, i] ** 0.5)
            summary[name] = {
                "value": value,
                "uncertainty": uncertainty,
                "z_score": value / uncertainty if uncertainty > 0 else 0.0,
            }
        return summary

    def get_anomalous_updates(self, threshold: float = 3.5) -> list[dict]:
        """Get recent anomalous innovations (z-score > threshold)."""
        return [
            u.to_dict()
            for u in self.estimator.recent_innovations
            if abs(u.innovation_zscore) > threshold
        ]

    def get_normalizer_diagnostics(self) -> dict[str, dict]:
        """Return current EWMA state for each stream — useful for debugging."""
        diagnostics = {}
        for truf_key, norm in self._normalizers.items():
            config = self._get_config(truf_key)
            diagnostics[truf_key] = {
                "transform": config.transform,
                "halflife_days": config.halflife_days,
                "ewma_mean": round(norm.mean, 4),
                "ewma_std": round(norm.std, 4),
                "observations": norm._count,
            }
        return diagnostics


async def run_pipeline_for_tree(
    tree_kalman_state: Optional[dict],
    stream_keys: Optional[list[str]] = None,
) -> tuple[dict, list[StateUpdate]]:
    """Convenience function: run the pipeline and return updated state + updates.

    Args:
        tree_kalman_state: Current kalman_state JSONB from RedwoodTree
        stream_keys: Optional list of TRUF keys to fetch

    Returns:
        (updated_kalman_state, list_of_updates) tuple.
        The kalman_state includes the full innovation history for audit trail.
    """
    pipeline = StreamPipeline()
    pipeline.load_state(tree_kalman_state)
    updates = await pipeline.run(stream_keys)
    return pipeline.save_state(), updates
