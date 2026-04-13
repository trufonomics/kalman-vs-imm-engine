"""
EM Estimation of the Markov Transition Probability Matrix.

Every published MS-DFM (Hamilton 1989, Kim & Nelson 1999, Hashimzade
et al. 2024) estimates the TPM via MLE or EM. Hand-tuning the TPM is
a methodological vulnerability (though our experiments show the engine
is TPM-insensitive due to dominant observation likelihoods).

This module implements the Classification EM algorithm for TPM
estimation from IMM backtest output:

  E-step: Compute smoothed regime probabilities from the IMM's filtered
          probabilities using the forward-backward (Kim 1994) smoother.
  M-step: Estimate TPM[i,j] = sum of smoothed transition probabilities
          from regime i to j, normalized by total time in regime i.

NOTE: Hamilton (1989) originally used direct numerical MLE (BFGS),
not EM. The EM approach for Markov-switching was formalized by
Hamilton (1990) and Kim (1994). We use EM here for simplicity.

NOTE: The Kim smoother was derived for Hamilton-filtered probabilities.
Our filtered probabilities come from the IMM interaction step, which
mixes state estimates before prediction. This is a reasonable
approximation (Hashimzade et al. 2024 construct an IMM-matched
smoother), but is not exactly the standard algorithm.

Refs:
  Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle." Econometrica
      57(2), 357-384.
  Hamilton, J.D. (1990). "Analysis of Time Series Subject to Changes
      in Regime." J. Econometrics 45, 39-70.
  Kim, C.-J. (1994). "Dynamic Linear Models with Markov-Switching."
      J. Econometrics 60(1-2), 1-22.
  Dempster, A.P., Laird, N.M. & Rubin, D.B. (1977). "Maximum
      Likelihood from Incomplete Data via the EM Algorithm." JRSS B.
"""

import numpy as np


def kim_smoother(
    filtered_probs: np.ndarray,
    tpm: np.ndarray,
) -> np.ndarray:
    """Kim (1994) backward smoother for regime probabilities.

    Given filtered probabilities P(S_t | Y_1,...,Y_t) and the TPM,
    compute smoothed probabilities P(S_t | Y_1,...,Y_T) using the
    backward recursion.

    Args:
        filtered_probs: (T, K) array of filtered regime probabilities
        tpm: (K, K) transition probability matrix

    Returns:
        (T, K) array of smoothed regime probabilities
    """
    T, K = filtered_probs.shape
    smoothed = np.zeros_like(filtered_probs)
    smoothed[-1] = filtered_probs[-1]

    for t in range(T - 2, -1, -1):
        # Predicted probabilities at t+1 (from the forward pass)
        predicted = tpm.T @ filtered_probs[t]
        predicted = np.maximum(predicted, 1e-15)

        # Backward update
        for j in range(K):
            ratio_sum = 0.0
            for k in range(K):
                ratio_sum += tpm[j, k] * smoothed[t + 1, k] / predicted[k]
            smoothed[t, j] = filtered_probs[t, j] * ratio_sum

        # Renormalize
        total = smoothed[t].sum()
        if total > 1e-15:
            smoothed[t] /= total

    return smoothed


def em_estimate_tpm(
    filtered_probs: np.ndarray,
    tpm_init: np.ndarray,
    n_iterations: int = 20,
    min_transition: float = 0.005,
) -> tuple[np.ndarray, list[float]]:
    """Estimate TPM via EM algorithm.

    Iterates E-step (Kim smoother) and M-step (transition counting)
    until convergence.

    Args:
        filtered_probs: (T, K) filtered regime probabilities from IMM
        tpm_init: (K, K) initial TPM guess
        n_iterations: max EM iterations
        min_transition: floor for off-diagonal entries (regularization)

    Returns:
        (estimated_tpm, convergence_log)
    """
    T, K = filtered_probs.shape
    tpm = tpm_init.copy()
    convergence_log = []

    for iteration in range(n_iterations):
        # E-step: Kim smoother
        smoothed = kim_smoother(filtered_probs, tpm)

        # Compute smoothed joint probabilities P(S_t=j, S_{t-1}=i | Y)
        # Using the standard formula:
        # P(S_t=j, S_{t-1}=i | Y) = smoothed[t,j] * tpm[i,j] * filtered[t-1,i]
        #                            / predicted[t,j]
        transition_counts = np.zeros((K, K))

        for t in range(1, T):
            predicted = tpm.T @ filtered_probs[t - 1]
            predicted = np.maximum(predicted, 1e-15)

            for i in range(K):
                for j in range(K):
                    joint = (
                        smoothed[t, j]
                        * tpm[i, j]
                        * filtered_probs[t - 1, i]
                        / predicted[j]
                    )
                    transition_counts[i, j] += joint

        # M-step: normalize rows to get new TPM
        new_tpm = np.zeros((K, K))
        for i in range(K):
            row_sum = transition_counts[i].sum()
            if row_sum > 1e-15:
                new_tpm[i] = transition_counts[i] / row_sum
            else:
                new_tpm[i] = tpm[i]

        # Enforce minimum transition probabilities (regularization)
        for i in range(K):
            for j in range(K):
                if i != j:
                    new_tpm[i, j] = max(new_tpm[i, j], min_transition)
            # Renormalize row
            new_tpm[i] /= new_tpm[i].sum()

        # Check convergence
        delta = float(np.max(np.abs(new_tpm - tpm)))
        convergence_log.append(delta)
        tpm = new_tpm

        if delta < 1e-6:
            break

    return tpm, convergence_log


def estimate_tpm_from_backtest(
    daily_log: list[dict],
    branch_order: list[str],
    tpm_init: np.ndarray,
    n_iterations: int = 20,
) -> tuple[np.ndarray, list[str], list[float]]:
    """Convenience: estimate TPM from backtest daily_log output.

    Args:
        daily_log: list of {"date", "probabilities": {branch: prob}} dicts
        branch_order: ordered list of branch IDs matching TPM indices
        tpm_init: initial TPM guess
        n_iterations: max EM iterations

    Returns:
        (estimated_tpm, branch_order, convergence_log)
    """
    T = len(daily_log)
    K = len(branch_order)

    # Build filtered probability matrix
    filtered = np.zeros((T, K))
    for t, entry in enumerate(daily_log):
        probs = entry["probabilities"]
        for k, branch_id in enumerate(branch_order):
            filtered[t, k] = probs.get(branch_id, 1.0 / K)

    estimated_tpm, convergence = em_estimate_tpm(
        filtered, tpm_init, n_iterations,
    )

    return estimated_tpm, branch_order, convergence
