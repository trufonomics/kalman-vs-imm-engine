"""
C1 Tempering Validation + New Diagnostics.

Validates:
  1. C1 tempering ceiling (0.70) is active in imm_tracker.py
  2. Ferro-Fricker (2012) bias-corrected Brier for stagflation
  3. Berkowitz (2001) PIT LR test vs chi-squared histogram

Usage:
    cd /private/tmp/kalman-vs-imm-engine
    python backtests/c1_validation.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from heimdall.regime_diagnostics import run_full_diagnostics, berkowitz_pit_test
from backtests.diagnostic_comparison import run_engine
from backtests.full_history_backtest import fetch_data, REGIME_CHECKPOINTS


def main():
    os.environ.setdefault("TRUF_PRIVATE_KEY", "dummy")

    # Verify C1 is active
    from heimdall.imm_tracker import IMMBranchTracker
    t = IMMBranchTracker()
    assert hasattr(t, 'TEMPER_CEILING'), "TEMPER_CEILING not found"
    assert t.TEMPER_CEILING == 0.70, f"Expected 0.70, got {t.TEMPER_CEILING}"
    print(f"  C1 active: TEMPER_FLOOR={t.TEMPER_FLOOR}, TEMPER_CEILING={t.TEMPER_CEILING}")

    data_path = Path(__file__).parent.parent / "data" / "full_history_cache.csv"
    if data_path.exists():
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        data = fetch_data()

    print("\n--- Running Full Stack (P0+P2b+P4+P5+C1) ---")
    result = run_engine(
        data, "Full Stack + C1",
        use_regime_noise=True, use_state_tpm=True,
        use_cumulator=True, use_gas=True,
    )
    print(f"  {result.total_updates} updates, LL={result.total_log_likelihood:.0f}, "
          f"anomaly={result.anomaly_count}")

    # Full diagnostics (now includes Ferro-Fricker)
    diag = run_full_diagnostics(result.daily_log, REGIME_CHECKPOINTS)

    print("\n" + "=" * 90)
    print("  REGIME DIAGNOSTICS — FULL STACK + C1")
    print("=" * 90)

    # ROC/AUC
    print("\n  ROC/AUC (Berge & Jorda 2011):")
    for regime in ["contraction", "stagflation", "expansion"]:
        r = diag["roc"][regime]
        print(f"    {regime:<15s} AUC={r.auc:.4f}  optimal_threshold={r.optimal_threshold:.3f}")

    # Murphy (1973) Brier
    print("\n  Brier Score (Murphy 1973):")
    print(f"    {'Regime':<15s} {'Brier':>8s} {'Reliab':>8s} {'Resol':>8s} {'Uncert':>8s} {'BSS':>8s}")
    print("    " + "-" * 55)
    for regime in ["contraction", "stagflation"]:
        b = diag["brier"][regime]
        print(f"    {regime:<15s} {b.brier_score:>8.4f} {b.reliability:>8.4f} "
              f"{b.resolution:>8.4f} {b.uncertainty:>8.4f} {b.brier_skill_score:>8.4f}")

    # Ferro-Fricker (2012) bias-corrected Brier
    print("\n  Ferro-Fricker (2012) Bias-Corrected Brier:")
    print(f"    {'Regime':<15s} {'Reliab_BC':>10s} {'Reliab_raw':>10s} "
          f"{'Resol_BC':>10s} {'Resol_raw':>10s} {'Bias':>8s}")
    print("    " + "-" * 65)
    for regime in ["contraction", "stagflation"]:
        ff = diag["ferro_fricker"][regime]
        print(f"    {regime:<15s} {ff.reliability:>10.6f} {ff.reliability_uncorrected:>10.6f} "
              f"{ff.resolution:>10.6f} {ff.resolution_uncorrected:>10.6f} "
              f"{ff.bias_correction:>8.6f}")

    # Detection lag (dual thresholds)
    print("\n  Detection Lag (50% early warning / 80% Chauvet-Piger):")
    detected_50 = 0
    detected_80 = 0
    for dl in diag["detection_lag"]:
        lag50 = f"{dl.detection_lag_weeks:.1f}w" if dl.detection_lag_weeks is not None else "MISSED"
        lag80 = f"{dl.detection_lag_weeks_80:.1f}w" if dl.detection_lag_weeks_80 is not None else "MISSED"
        if dl.detection_lag_weeks is not None:
            detected_50 += 1
        if dl.detection_lag_weeks_80 is not None:
            detected_80 += 1
        print(f"    {lag50:>8s}/{lag80:>8s}  peak={dl.peak_probability:.3f}  {dl.event}")
    print(f"    Total: {detected_50}/11 @50%, {detected_80}/11 @80%")

    # Berkowitz PIT test (aggregate innovations across all streams)
    all_innovations = []
    all_variances = []
    for stream_key in result.innovation_log:
        innov = result.innovation_log[stream_key]
        var = result.variance_log.get(stream_key, [])
        if len(innov) == len(var):
            all_innovations.extend(innov)
            all_variances.extend(var)

    if all_innovations:
        print(f"\n  Berkowitz (2001) PIT LR Test ({len(all_innovations)} innovations):")
        bk = berkowitz_pit_test(all_innovations, all_variances)
        print(f"    LR stat = {bk.lr_statistic:.2f}, p = {bk.p_value:.4f}")
        print(f"    mu_hat = {bk.mu_hat:.4f}, sigma_hat = {bk.sigma_hat:.4f}, "
              f"rho_hat = {bk.rho_hat:.4f}")
        print(f"    Correct specification: {'YES' if bk.is_correct else 'NO (rejected)'}")

        # Also per-stream Berkowitz for key streams
        print("\n    Per-stream Berkowitz:")
        key_streams = ["US_CPI_YOY", "UNEMPLOYMENT_RATE", "SP500",
                       "INITIAL_CLAIMS", "10Y_YIELD", "CONSUMER_CONFIDENCE",
                       "FED_FUNDS_RATE", "OIL_PRICE"]
        for stream in key_streams:
            if stream in result.innovation_log and stream in result.variance_log:
                si = result.innovation_log[stream]
                sv = result.variance_log[stream]
                if len(si) >= 20 and len(si) == len(sv):
                    bks = berkowitz_pit_test(si, sv)
                    status = "OK" if bks.is_correct else "REJECT"
                    print(f"      {stream:<25s} LR={bks.lr_statistic:>7.2f}  "
                          f"p={bks.p_value:.4f}  mu={bks.mu_hat:+.3f}  "
                          f"sig={bks.sigma_hat:.3f}  rho={bks.rho_hat:+.3f}  [{status}]")
    else:
        print("\n  Berkowitz test: no innovations found in engine output")

    # Sharpness
    s = diag["sharpness"]
    print(f"\n  Sharpness (Gneiting et al. 2007):")
    print(f"    mean_max_prob={s.mean_max_prob:.3f}, >80%={s.frac_above_80:.1%}, "
          f">50%={s.frac_above_50:.1%}")


if __name__ == "__main__":
    main()
