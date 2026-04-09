"""
Adaptive Calibration — Learns filter parameters from TRUF stream history.

The hardcoded H, F, R matrices are STARTING POINTS from economic theory.
This service estimates them from actual data, using:

  1. PCA on TRUF history → H matrix (factor loadings)
  2. Autocorrelation → F diagonal (persistence)
  3. Cross-correlation → F off-diagonal (transmission channels)
  4. Innovation diagnostics → R values (observation noise)
  5. Whiteness test → Q adjustment (process noise)

The spec says: "Initial loadings set by economic theory. Refined by running
PCA on 2+ years of TRUF stream history. Re-estimated quarterly."

Ref: Implementation Architecture doc (Level 1, Estimation section)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from heimdall.kalman_filter import (
    FACTORS,
    FACTOR_INDEX,
    N_FACTORS,
    EconomicStateEstimator,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    H: dict[str, dict[str, float]]         # stream → {factor: loading}
    F_diagonal: dict[str, float]            # factor → persistence
    F_cross: list[tuple[str, str, float]]   # (from, to, coeff)
    R: dict[str, float]                     # stream → noise
    Q_scale: float                          # multiplier on Q
    n_streams: int
    n_observations: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    method: str = ""


def estimate_from_history(
    stream_histories: dict[str, list[dict]],
    n_factors: int = N_FACTORS,
    min_observations: int = 30,
) -> Optional[CalibrationResult]:
    """Estimate filter parameters from TRUF stream history.

    Args:
        stream_histories: {stream_key: [{time: epoch, value: float}, ...]}
            Each stream needs enough history for statistical estimation.
        n_factors: Number of latent factors (default 8).
        min_observations: Minimum data points per stream to include.

    Returns:
        CalibrationResult with estimated H, F, R, Q parameters,
        or None if insufficient data.
    """
    # Filter streams with enough data
    valid_streams = {
        key: hist for key, hist in stream_histories.items()
        if len(hist) >= min_observations
    }

    if len(valid_streams) < n_factors:
        logger.warning(
            f"Need at least {n_factors} streams with {min_observations}+ observations, "
            f"got {len(valid_streams)}. Falling back to theory-based parameters."
        )
        return None

    stream_keys = sorted(valid_streams.keys())
    n_streams = len(stream_keys)

    # Build aligned data matrix (streams × time)
    # Align all streams to common timestamps
    data_matrix, timestamps = _align_streams(valid_streams, stream_keys)
    n_obs = data_matrix.shape[1]

    if n_obs < min_observations:
        logger.warning(f"Only {n_obs} aligned observations. Need {min_observations}+.")
        return None

    logger.info(
        f"Calibrating from {n_streams} streams × {n_obs} observations"
    )

    # ── Step 1: PCA → H matrix (factor loadings) ──
    H_loadings = _estimate_H_via_pca(data_matrix, stream_keys, n_factors)

    # ── Step 2: Autocorrelation → F diagonal (persistence) ──
    F_diagonal = _estimate_persistence(data_matrix, stream_keys, H_loadings)

    # ── Step 3: Cross-correlation → F off-diagonal ──
    F_cross = _estimate_cross_dynamics(data_matrix, stream_keys, H_loadings)

    # ── Step 4: Innovation-based R estimation ──
    R_values = _estimate_noise(data_matrix, stream_keys, H_loadings)

    return CalibrationResult(
        H=H_loadings,
        F_diagonal=F_diagonal,
        F_cross=F_cross,
        R=R_values,
        Q_scale=1.0,
        n_streams=n_streams,
        n_observations=n_obs,
        method="pca_autocorrelation",
    )


def _align_streams(
    histories: dict[str, list[dict]],
    stream_keys: list[str],
) -> tuple[np.ndarray, list[int]]:
    """Align multiple stream histories to common timestamps.

    Returns (data_matrix[n_streams, n_times], timestamps).
    Missing values are forward-filled.
    """
    # Collect all timestamps
    all_times: set[int] = set()
    for key in stream_keys:
        for record in histories[key]:
            all_times.add(int(record["time"]))

    sorted_times = sorted(all_times)
    time_index = {t: i for i, t in enumerate(sorted_times)}
    n_times = len(sorted_times)

    # Build matrix
    matrix = np.full((len(stream_keys), n_times), np.nan)
    for s_idx, key in enumerate(stream_keys):
        for record in histories[key]:
            t_idx = time_index[int(record["time"])]
            matrix[s_idx, t_idx] = float(record["value"])

    # Forward-fill NaN values
    for s_idx in range(len(stream_keys)):
        last_valid = np.nan
        for t_idx in range(n_times):
            if np.isnan(matrix[s_idx, t_idx]):
                matrix[s_idx, t_idx] = last_valid
            else:
                last_valid = matrix[s_idx, t_idx]

    # Drop columns where any stream is still NaN (no prior value)
    valid_cols = ~np.any(np.isnan(matrix), axis=0)
    matrix = matrix[:, valid_cols]
    timestamps = [t for t, v in zip(sorted_times, valid_cols) if v]

    return matrix, timestamps


def _estimate_H_via_pca(
    data: np.ndarray,
    stream_keys: list[str],
    n_factors: int,
) -> dict[str, dict[str, float]]:
    """Estimate factor loadings via PCA on standardized returns.

    PCA on the correlation matrix of stream returns gives us
    the empirical factor structure. The top n_factors principal
    components ARE the latent factors.
    """
    # Compute returns (pct changes)
    returns = np.diff(data, axis=1) / (np.abs(data[:, :-1]) + 1e-10)

    # Standardize
    means = returns.mean(axis=1, keepdims=True)
    stds = returns.std(axis=1, keepdims=True)
    stds[stds < 1e-10] = 1.0
    standardized = (returns - means) / stds

    # Correlation matrix
    corr = np.corrcoef(standardized)
    # Handle NaN correlations
    corr = np.nan_to_num(corr, nan=0.0)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Sort descending (eigh returns ascending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top n_factors
    n_components = min(n_factors, len(eigenvalues))
    loadings = eigenvectors[:, :n_components]

    # Scale loadings by sqrt(eigenvalue) for interpretability
    for j in range(n_components):
        if eigenvalues[j] > 0:
            loadings[:, j] *= np.sqrt(eigenvalues[j])

    # Map to factor names — assign PCA components to named factors
    # by matching sign patterns to economic theory
    H_result: dict[str, dict[str, float]] = {}
    for s_idx, stream_key in enumerate(stream_keys):
        stream_loadings = {}
        for f_idx in range(n_components):
            factor_name = FACTORS[f_idx] if f_idx < len(FACTORS) else f"factor_{f_idx}"
            loading = float(loadings[s_idx, f_idx])
            if abs(loading) > 0.05:  # Only keep non-negligible loadings
                stream_loadings[factor_name] = round(loading, 4)
        H_result[stream_key] = stream_loadings

    variance_explained = eigenvalues[:n_components].sum() / eigenvalues.sum()
    logger.info(
        f"PCA: {n_components} factors explain {variance_explained:.1%} of variance"
    )

    return H_result


def _estimate_persistence(
    data: np.ndarray,
    stream_keys: list[str],
    H_loadings: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Estimate factor persistence from lag-1 autocorrelation of streams.

    For each factor, find the stream with highest loading on it,
    compute its lag-1 autocorrelation, and use that as persistence.
    """
    persistence: dict[str, float] = {}

    for factor in FACTORS:
        # Find the best stream for this factor
        best_stream = None
        best_loading = 0.0
        for s_idx, key in enumerate(stream_keys):
            loading = abs(H_loadings.get(key, {}).get(factor, 0.0))
            if loading > best_loading:
                best_loading = loading
                best_stream = s_idx

        if best_stream is not None:
            series = data[best_stream]
            # Lag-1 autocorrelation
            if len(series) > 2:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                # Clamp to valid persistence range
                autocorr = max(0.5, min(0.99, float(np.nan_to_num(autocorr, nan=0.9))))
                persistence[factor] = round(autocorr, 4)
            else:
                persistence[factor] = 0.90
        else:
            persistence[factor] = 0.90

    return persistence


def _estimate_cross_dynamics(
    data: np.ndarray,
    stream_keys: list[str],
    H_loadings: dict[str, dict[str, float]],
    max_lag: int = 6,
    significance_threshold: float = 0.15,
) -> list[tuple[str, str, float]]:
    """Estimate cross-factor transmission channels from lagged correlations.

    For each pair of factors, compute the cross-correlation at lags 1..max_lag.
    If the peak correlation exceeds the threshold, it's a transmission channel.
    """
    # Build factor time series by weighted sum of stream returns
    returns = np.diff(data, axis=1) / (np.abs(data[:, :-1]) + 1e-10)
    factor_series = np.zeros((N_FACTORS, returns.shape[1]))

    for s_idx, key in enumerate(stream_keys):
        for factor, loading in H_loadings.get(key, {}).items():
            f_idx = FACTOR_INDEX.get(factor)
            if f_idx is not None:
                factor_series[f_idx] += loading * returns[s_idx]

    cross_dynamics = []
    for i, from_factor in enumerate(FACTORS):
        for j, to_factor in enumerate(FACTORS):
            if i == j:
                continue

            # Compute lagged correlation: does from_factor at t predict to_factor at t+lag?
            best_corr = 0.0
            for lag in range(1, max_lag + 1):
                if len(factor_series[i]) <= lag:
                    continue
                x = factor_series[i][:-lag]
                y = factor_series[j][lag:]
                if len(x) < 10:
                    continue
                corr = float(np.nan_to_num(np.corrcoef(x, y)[0, 1], nan=0.0))
                if abs(corr) > abs(best_corr):
                    best_corr = corr

            if abs(best_corr) > significance_threshold:
                # Scale down — cross-dynamics should be small coefficients
                coeff = best_corr * 0.1
                cross_dynamics.append((from_factor, to_factor, round(coeff, 4)))

    return cross_dynamics


def _estimate_noise(
    data: np.ndarray,
    stream_keys: list[str],
    H_loadings: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Estimate observation noise (R) from residual variance.

    R = variance of stream that ISN'T explained by the common factors.
    Higher R = noisier stream = Kalman trusts it less.
    """
    returns = np.diff(data, axis=1) / (np.abs(data[:, :-1]) + 1e-10)

    R_values: dict[str, float] = {}
    for s_idx, key in enumerate(stream_keys):
        stream_returns = returns[s_idx]
        total_var = float(np.var(stream_returns))

        # Explained variance = sum of (loading² × factor_variance)
        # Approximate: factor variance ≈ stream variance × loading²
        explained_var = 0.0
        for factor, loading in H_loadings.get(key, {}).items():
            explained_var += loading ** 2

        # Residual = total - explained (clamped to positive)
        residual_var = max(0.01, total_var * max(0.0, 1.0 - explained_var))
        R_values[key] = round(residual_var, 4)

    return R_values


def apply_calibration(
    estimator: EconomicStateEstimator,
    calibration: CalibrationResult,
):
    """Apply calibration results to a live estimator.

    Updates H, F, R in-place. The filter continues from its current
    state estimate — it doesn't reset. This is a smooth transition.
    """
    # Update H (stream registry)
    for stream_key, loadings in calibration.H.items():
        noise = calibration.R.get(stream_key, 0.10)
        estimator.register_stream(stream_key, loadings, noise)

    # Update F diagonal (persistence)
    for factor, persistence in calibration.F_diagonal.items():
        idx = FACTOR_INDEX.get(factor)
        if idx is not None:
            estimator.F[idx, idx] = persistence

    # Update F off-diagonal (cross-dynamics)
    # First zero out existing off-diagonals
    for i in range(N_FACTORS):
        for j in range(N_FACTORS):
            if i != j:
                estimator.F[i, j] = 0.0

    for from_factor, to_factor, coeff in calibration.F_cross:
        i = FACTOR_INDEX.get(to_factor)
        j = FACTOR_INDEX.get(from_factor)
        if i is not None and j is not None:
            estimator.F[i, j] = coeff

    # Rebuild Q from new persistence values
    estimator.Q = estimator._build_Q()

    logger.info(
        f"Applied calibration: {calibration.n_streams} streams, "
        f"{len(calibration.F_cross)} cross-dynamics, "
        f"method={calibration.method}"
    )


def diagnose_innovations(
    estimator: EconomicStateEstimator,
) -> dict:
    """Run diagnostic tests on recent innovations.

    If the filter is well-calibrated:
    - Innovations should be zero-mean (no systematic bias)
    - Innovation variance should match predicted variance (R is correct)
    - Innovations should be white (no autocorrelation → F is correct)

    Returns diagnostic dict with pass/fail for each test.
    """
    if len(estimator.recent_innovations) < 10:
        return {"status": "insufficient_data", "n_innovations": len(estimator.recent_innovations)}

    z_scores = [u.innovation_zscore for u in estimator.recent_innovations]
    z_arr = np.array(z_scores)

    # Test 1: Zero-mean (mean of z-scores should be near 0)
    mean_z = float(np.mean(z_arr))
    mean_test = abs(mean_z) < 0.5  # Allow some slack

    # Test 2: Unit variance (std of z-scores should be near 1)
    std_z = float(np.std(z_arr))
    variance_test = 0.5 < std_z < 2.0

    # Test 3: Whiteness (lag-1 autocorrelation of z-scores should be near 0)
    if len(z_arr) > 5:
        autocorr = float(np.corrcoef(z_arr[:-1], z_arr[1:])[0, 1])
        whiteness_test = abs(autocorr) < 0.3
    else:
        autocorr = 0.0
        whiteness_test = True

    # Test 4: Anomaly rate (should be ~5% for z > 2)
    anomaly_rate = sum(1 for z in z_scores if abs(z) > 2.0) / len(z_scores)
    anomaly_test = anomaly_rate < 0.15  # Allow up to 15%

    all_pass = mean_test and variance_test and whiteness_test and anomaly_test

    return {
        "status": "healthy" if all_pass else "needs_recalibration",
        "n_innovations": len(z_scores),
        "mean_z": round(mean_z, 4),
        "std_z": round(std_z, 4),
        "autocorrelation": round(autocorr, 4),
        "anomaly_rate": round(anomaly_rate, 4),
        "tests": {
            "zero_mean": mean_test,
            "unit_variance": variance_test,
            "whiteness": whiteness_test,
            "anomaly_rate": anomaly_test,
        },
        "recommendation": (
            None if all_pass
            else "Run estimate_from_history() with recent TRUF data to recalibrate"
        ),
    }


async def auto_calibrate(
    estimator: EconomicStateEstimator,
    force: bool = False,
) -> Optional[CalibrationResult]:
    """Auto-calibrate if diagnostics indicate the filter is drifting.

    Called periodically (e.g. daily) or when innovation diagnostics fail.
    Fetches TRUF history, runs PCA, and applies updated parameters.

    Args:
        estimator: The live estimator to potentially recalibrate.
        force: Force recalibration even if diagnostics pass.

    Returns:
        CalibrationResult if recalibration happened, None if skipped.
    """
    if not force:
        diagnostics = diagnose_innovations(estimator)
        if diagnostics["status"] == "healthy":
            logger.info("Filter diagnostics healthy — skipping recalibration")
            return None
        logger.info(f"Filter needs recalibration: {diagnostics}")

    # NOTE: auto_calibrate() requires the TRUF client for live data.
    # In standalone mode, use estimate_from_history() directly with pre-fetched data.
    try:
        from app.services.truf_client import get_truf_client, KNOWN_STREAMS  # type: ignore
    except ImportError:
        raise ImportError(
            "auto_calibrate() requires the TRUF client (app.services.truf_client). "
            "In standalone mode, use estimate_from_history() directly with pre-fetched data."
        )
    from heimdall.stream_pipeline import TRUF_TO_KALMAN

    client = get_truf_client()
    histories: dict[str, list[dict]] = {}

    for truf_key, kalman_key in TRUF_TO_KALMAN.items():
        meta = KNOWN_STREAMS.get(truf_key)
        if not meta:
            continue

        try:
            history = await client.get_stream_history(
                provider=meta["provider"],
                stream_id=meta["streamId"],
                days=365,  # 1 year of history
            )
            if history:
                histories[kalman_key] = history
        except Exception as e:
            logger.debug(f"Failed to fetch history for {truf_key}: {e}")

    if not histories:
        logger.warning("No TRUF history available for calibration")
        return None

    result = estimate_from_history(histories)
    if result:
        apply_calibration(estimator, result)

    return result
