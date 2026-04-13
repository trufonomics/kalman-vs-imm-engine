#!/usr/bin/env python3
"""CLI for the Heimdall Kalman + IMM engine.

Run the engine, inspect state, feed observations, check regime probabilities.

Usage:
    python kalman_cli.py state                     # Show current state
    python kalman_cli.py predict                   # Run one prediction step
    python kalman_cli.py update CPI 0.032          # Feed an observation
    python kalman_cli.py regime                    # Show regime probabilities
    python kalman_cli.py history                   # Recent innovation history
    python kalman_cli.py simulate --steps 100      # Run synthetic simulation
    python kalman_cli.py streams                   # List registered streams
    python kalman_cli.py factor inflation_trend    # Single factor detail
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from heimdall.kalman_filter import (
    FACTORS,
    N_FACTORS,
    PERSISTENCE,
    STREAM_LOADINGS,
    EconomicStateEstimator,
)
from heimdall.imm_tracker import IMMBranchTracker

# State file for persistence between CLI invocations
STATE_FILE = Path(__file__).parent / ".kalman_state.json"


def load_engine() -> tuple[EconomicStateEstimator, IMMBranchTracker]:
    """Load engine from saved state or create fresh."""
    estimator = EconomicStateEstimator()
    tracker = IMMBranchTracker()

    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            saved = json.load(f)
        if "kalman" in saved:
            estimator.load_state(saved["kalman"])
        if "imm" in saved:
            tracker.load_state(saved["imm"])

    return estimator, tracker


def save_engine(estimator: EconomicStateEstimator, tracker: IMMBranchTracker):
    """Persist engine state between CLI invocations."""
    state = {
        "kalman": estimator.to_dict(),
        "imm": tracker.to_dict(),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def cmd_state(args):
    """Display current state estimate."""
    estimator, _ = load_engine()
    state = estimator.get_state()

    print("=" * 50)
    print("  ECONOMIC STATE ESTIMATE")
    print("=" * 50)
    print(f"  Updates processed: {estimator.update_count}")
    print()

    # Factor values with uncertainty
    for i, factor in enumerate(FACTORS):
        val = state.mean[i]
        unc = np.sqrt(state.covariance[i, i])
        bar = _bar(val, width=30)
        print(f"  {factor:25s}  {val:+.4f} +/- {unc:.4f}  {bar}")

    print()

    # Covariance condition number (proxy for geometry complexity)
    eigvals = np.linalg.eigvalsh(state.covariance)
    condition = eigvals[-1] / max(eigvals[0], 1e-10)
    print(f"  Covariance condition: {condition:.1f}")
    print(f"  Effective rank: {np.sum(eigvals > 0.001 * eigvals[-1])}/{N_FACTORS}")


def cmd_predict(args):
    """Run one prediction step (advance state by one day)."""
    estimator, tracker = load_engine()

    state_before = estimator.x.copy()
    estimator.predict()
    state_after = estimator.x.copy()

    delta = state_after - state_before
    print("Prediction step (1 day):")
    for i, factor in enumerate(FACTORS):
        if abs(delta[i]) > 1e-6:
            print(f"  {factor:25s}: {state_before[i]:+.4f} -> {state_after[i]:+.4f}  (delta: {delta[i]:+.6f})")

    save_engine(estimator, tracker)
    print("\nState saved.")


def cmd_update(args):
    """Feed an observation to the engine."""
    estimator, tracker = load_engine()

    stream_key = args.stream
    value = args.value

    # Try to match partial stream names
    if stream_key not in estimator.stream_registry:
        matches = [k for k in estimator.stream_registry if stream_key.upper() in k]
        if len(matches) == 1:
            stream_key = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous stream '{stream_key}'. Matches: {matches}")
            return
        else:
            print(f"Unknown stream '{stream_key}'. Use 'streams' command to list.")
            return

    estimator.predict()
    result = estimator.update(stream_key, value)

    if result:
        print(f"Updated with {stream_key} = {value}")
        print(f"  Innovation: {result.innovation:+.4f} (z-score: {result.innovation_zscore:+.2f})")
        if result.is_anomalous:
            print(f"  *** ANOMALOUS (|z| > 3.5) ***")

        # Show which factors moved most
        kg = np.array(result.kalman_gain)
        top_idx = np.argsort(np.abs(kg))[::-1][:3]
        print("  Top factor movements:")
        for idx in top_idx:
            print(f"    {FACTORS[idx]:25s}: gain={kg[idx]:+.4f}, shift={kg[idx] * result.innovation:+.6f}")

    save_engine(estimator, tracker)
    print("\nState saved.")


def cmd_regime(args):
    """Show IMM regime probabilities."""
    _, tracker = load_engine()
    probs = tracker.get_probabilities()

    print("=" * 50)
    print("  REGIME PROBABILITIES")
    print("=" * 50)

    for branch_id, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 40)
        print(f"  {branch_id:20s}  {prob:6.1%}  |{bar}")


def cmd_history(args):
    """Show recent innovation history."""
    estimator, _ = load_engine()

    n = min(args.n, len(estimator.recent_innovations))
    if n == 0:
        print("No innovation history. Feed some observations first.")
        return

    print(f"Last {n} innovations:")
    print(f"  {'Stream':20s} {'Observed':>10s} {'Predicted':>10s} {'z-score':>8s} {'Anomalous':>10s}")
    print("  " + "-" * 62)

    for update in estimator.recent_innovations[-n:]:
        anomalous = "***" if update.is_anomalous else ""
        print(f"  {update.stream_key:20s} {update.observed:10.4f} {update.predicted:10.4f} "
              f"{update.innovation_zscore:+8.2f} {anomalous:>10s}")


def cmd_streams(args):
    """List all registered streams."""
    estimator, _ = load_engine()

    print(f"{'Stream':25s} {'Noise (R)':>10s} {'Primary Factor':25s}")
    print("-" * 65)

    for stream_key, (H_row, R) in sorted(estimator.stream_registry.items()):
        primary_idx = np.argmax(np.abs(H_row))
        primary = FACTORS[primary_idx]
        print(f"{stream_key:25s} {R:10.4f} {primary:25s}")


def cmd_factor(args):
    """Show detailed info for a single factor."""
    estimator, _ = load_engine()

    factor = args.factor
    if factor not in FACTORS:
        matches = [f for f in FACTORS if factor.lower() in f.lower()]
        if len(matches) == 1:
            factor = matches[0]
        else:
            print(f"Unknown factor. Available: {FACTORS}")
            return

    idx = FACTORS.index(factor)
    val = estimator.x[idx]
    unc = np.sqrt(estimator.P[idx, idx])

    print(f"Factor: {factor}")
    print(f"  Value: {val:+.6f}")
    print(f"  Uncertainty (1σ): {unc:.6f}")
    print(f"  Persistence: {PERSISTENCE[factor]}")
    print(f"  Monthly decay: {PERSISTENCE[factor]**22:.3f}")

    # Cross-correlations from covariance
    print(f"\n  Cross-correlations (from P):")
    for j, other in enumerate(FACTORS):
        if j != idx:
            corr = estimator.P[idx, j] / np.sqrt(estimator.P[idx, idx] * estimator.P[j, j])
            if abs(corr) > 0.1:
                print(f"    {other:25s}: {corr:+.3f}")

    # Which streams observe this factor?
    print(f"\n  Observable via streams:")
    for stream_key, loadings in STREAM_LOADINGS.items():
        if factor in loadings:
            loading = loadings[factor]
            print(f"    {stream_key:25s}: loading={loading:+.2f}")


def cmd_simulate(args):
    """Run a quick synthetic simulation."""
    estimator, tracker = load_engine()
    rng = np.random.default_rng(args.seed)

    streams = list(estimator.stream_registry.keys())
    n_steps = args.steps

    print(f"Running {n_steps}-step simulation...")

    for t in range(n_steps):
        estimator.predict()
        stream_key = streams[t % len(streams)]
        H_row, R = estimator.stream_registry[stream_key]
        z = float(H_row @ estimator.x + rng.normal(0, np.sqrt(R)))
        estimator.update(stream_key, z)

    # Save before displaying state
    save_engine(estimator, tracker)

    print(f"Done. Final state:")
    cmd_state(args)


def cmd_reset(args):
    """Reset engine to fresh state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        print("Engine state reset.")
    else:
        print("No saved state found.")


def _bar(value: float, width: int = 30) -> str:
    """ASCII bar for factor values in [-1, 1] range."""
    clamped = max(-1.0, min(1.0, value))
    mid = width // 2
    pos = int((clamped + 1) / 2 * width)
    bar = [" "] * width
    bar[mid] = "|"
    if pos < mid:
        for i in range(max(0, pos), mid):
            bar[i] = "-"
    elif pos > mid:
        for i in range(mid + 1, min(width, pos + 1)):
            bar[i] = "+"
    return "[" + "".join(bar) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Heimdall Kalman + IMM Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("state", help="Show current state estimate")
    sub.add_parser("predict", help="Run one prediction step")

    up = sub.add_parser("update", help="Feed an observation")
    up.add_argument("stream", help="Stream key (e.g., US_CPI_YOY)")
    up.add_argument("value", type=float, help="Observed value")

    sub.add_parser("regime", help="Show regime probabilities")

    hist = sub.add_parser("history", help="Recent innovations")
    hist.add_argument("-n", type=int, default=10, help="Number of entries")

    sub.add_parser("streams", help="List registered streams")

    fac = sub.add_parser("factor", help="Detail for one factor")
    fac.add_argument("factor", help="Factor name (partial match ok)")

    sim = sub.add_parser("simulate", help="Run synthetic simulation")
    sim.add_argument("--steps", type=int, default=100)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--save", action="store_true")

    sub.add_parser("reset", help="Reset to fresh state")

    args = parser.parse_args()

    commands = {
        "state": cmd_state,
        "predict": cmd_predict,
        "update": cmd_update,
        "regime": cmd_regime,
        "history": cmd_history,
        "streams": cmd_streams,
        "factor": cmd_factor,
        "simulate": cmd_simulate,
        "reset": cmd_reset,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
