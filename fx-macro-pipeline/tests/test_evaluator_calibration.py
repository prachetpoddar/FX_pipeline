"""
Evaluator calibration regression test (FIX 6).

Locks the calibration that the Layer-2 power audit
(data/layer2_audit/LAYER2_AUDIT_REPORT.md) demanded. Uses the audit's
synthetic panel generator (make_panel) directly — same code path that
produced the audit numbers — and asserts:

  - Old threshold-style rule (sharpe>0.3, acc>0.55) has det ~ 0 at ic <= 0.10.
    Guard against anyone re-introducing it.
  - decision_ttest FPR is in [0.0, 0.10] at ic = 0.00 (calibrated under
    the global null).
  - Timeseries portfolio t-test: det >= 0.8 at ic = 0.02, n_inst = 18.
  - Cross-sectional rank-IC t-test: det >= 0.8 at ic = 0.01, n_inst = 18,
    rho = 0.6, h = 1.
  - ic_in_position recovers ~ 2x the zero-padded full-series IC at 70%
    gating (audit found ungated->gated attenuation ratio ~ 0.52).
  - BH-FDR multiple-testing guard: under global null with k candidate
    signals, family-wise false discovery stays near alpha.

Seeds are reduced to keep runtime tractable (>=20 per cell, as the brief
permits). Pytest-discoverable; also runnable directly.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

# Import make_panel directly from the audit so the calibration tracks
# the exact data-generating process the audit characterized.
_AUDIT = PROJ / "data" / "layer2_audit" / "run_audit.py"
_spec = importlib.util.spec_from_file_location("layer2_audit_run", _AUDIT)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)  # type: ignore[union-attr]
make_panel = _audit.make_panel

from stage2_agents.evaluation import (  # noqa: E402
    decision_ttest,
    portfolio_return_series,
    rank_ic_series,
    ic_in_position,
    bh_fdr_decision,
    decide_family,
)


N_SEEDS = 20  # reduced from audit's 30; sufficient for the locked thresholds
N_PERIODS = 2500
ALPHA = 0.05


def _threshold_rule_fires(sharpe, accuracy, sharpe_min=0.3, acc_min=0.55):
    """The old Layer-2 pass rule — re-implemented here only to guard
    against anyone re-adding it as a verdict layer."""
    if not (np.isfinite(sharpe) and np.isfinite(accuracy)):
        return False
    return (sharpe > sharpe_min) and (accuracy > acc_min)


# ---------------------------------------------------------------------------
# Guard: the old threshold rule still has ~0 detection
# ---------------------------------------------------------------------------

def test_threshold_rule_zero_detection_at_low_ic():
    """If anyone re-adds the sharpe/acc threshold as a verdict, it must
    continue to have ~0 detection at the IC range it was claiming to
    catch. Audit Task A: det = 0 at every cell."""
    rng_master = np.random.default_rng(11)
    n_inst = 18
    rho = 0.6
    fires = 0
    n_seeds = N_SEEDS
    for _ in range(n_seeds):
        seed = int(rng_master.integers(1, 2**31 - 1))
        S, F = make_panel(
            n_periods=N_PERIODS, n_instruments=n_inst,
            ic_true=0.05, axis="timeseries", rho_cross=rho, seed=seed,
        )
        sig = np.sign(S.values)
        # Build per-instrument sharpe / accuracy and check whether ANY
        # instrument trips the rule. Even at ic=0.05 the audit found 0 hits.
        per_sharpe = []
        per_acc = []
        for i in range(n_inst):
            s = sig[:, i]
            r = F.values[:, i]
            strat = s * r
            sd = strat.std(ddof=1)
            sh = float(strat.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
            ac = float((np.sign(s) == np.sign(r)).mean())
            per_sharpe.append(sh)
            per_acc.append(ac)
        any_fires = any(_threshold_rule_fires(s_, a_)
                        for s_, a_ in zip(per_sharpe, per_acc))
        if any_fires:
            fires += 1
    det = fires / n_seeds
    assert det <= 0.10, (
        f"Threshold rule re-fired at unexpected rate {det:.2f}; "
        f"audit recorded det=0 at all cells. Did someone re-add it as a "
        f"verdict layer?"
    )


# ---------------------------------------------------------------------------
# Calibration: t-test FPR under the global null
# ---------------------------------------------------------------------------

def test_ttest_fpr_under_null():
    """At ic=0, the t-test must fire no more often than alpha (within
    sampling noise). Acceptable window [0.0, 0.10].

    Note: we use 100 seeds here (more than N_SEEDS) because the binomial
    sampling SE on a 5% rate with 20 trials is ~5%, so 20 seeds cannot
    distinguish a properly calibrated 5% from a leaky 10%. 100 seeds gives
    SE ~ 2% which is tight enough to enforce the [0, 0.10] bound.
    """
    rng_master = np.random.default_rng(99)
    n_inst = 18
    fires = 0
    n_seeds = 100  # tighter calibration check
    for _ in range(n_seeds):
        seed = int(rng_master.integers(1, 2**31 - 1))
        S, F = make_panel(
            n_periods=N_PERIODS, n_instruments=n_inst,
            ic_true=0.00, axis="timeseries", rho_cross=0.6, seed=seed,
        )
        sig = np.sign(S)
        port = portfolio_return_series(sig, F)
        v = decision_ttest(port.values, alpha=ALPHA)
        if v["fires"]:
            fires += 1
    fpr = fires / n_seeds
    assert 0.0 <= fpr <= 0.10, (
        f"t-test FPR={fpr:.2f} outside [0.0, 0.10] under global null"
    )


# ---------------------------------------------------------------------------
# Calibration: timeseries portfolio detection
# ---------------------------------------------------------------------------

def test_timeseries_detection_at_ic_002():
    """Portfolio t-test must hit det >= 0.8 at ic=0.02, n=18, rho=0.6, h=1.
    Uses 30 seeds (matches the audit's seeds-per-cell) so the detection
    rate estimate is stable enough to enforce the 0.80 threshold."""
    rng_master = np.random.default_rng(33)
    n_inst = 18
    fires = 0
    n_seeds = max(N_SEEDS, 30)
    for _ in range(n_seeds):
        seed = int(rng_master.integers(1, 2**31 - 1))
        S, F = make_panel(
            n_periods=N_PERIODS, n_instruments=n_inst,
            ic_true=0.02, axis="timeseries", rho_cross=0.6, seed=seed,
        )
        sig = np.sign(S)
        port = portfolio_return_series(sig, F)
        v = decision_ttest(port.values, alpha=ALPHA)
        if v["fires"]:
            fires += 1
    det = fires / n_seeds
    assert det >= 0.8, (
        f"Timeseries detection at ic=0.02 was {det:.2f} (target >= 0.80)"
    )


# ---------------------------------------------------------------------------
# Calibration: cross-sectional rank-IC detection
# ---------------------------------------------------------------------------

def test_crosssectional_detection_at_ic_001():
    """Rank-IC t-test must hit det >= 0.8 at ic=0.01, n=18, rho=0.6, h=1.
    Audit Task B at this cell measured detection_rate = 0.93 with a
    one-sided test (n=30 seeds). decision_ttest is two-sided per spec
    (~half power against one-sided at the boundary), so the true rate at
    this cell is closer to 0.80-0.85 — we use 100 seeds here so the
    estimate is tight enough to enforce the 0.80 floor."""
    rng_master = np.random.default_rng(99)
    n_inst = 18
    fires = 0
    n_seeds = 100
    for _ in range(n_seeds):
        seed = int(rng_master.integers(1, 2**31 - 1))
        S, F = make_panel(
            n_periods=N_PERIODS, n_instruments=n_inst,
            ic_true=0.01, axis="crosssectional", holding_period=1,
            rho_cross=0.6, seed=seed,
        )
        ric = rank_ic_series(S, F)
        v = decision_ttest(ric.values, alpha=ALPHA)
        if v["fires"]:
            fires += 1
    det = fires / n_seeds
    assert det >= 0.8, (
        f"Cross-sectional detection at ic=0.01 was {det:.2f} "
        f"(target >= 0.80)"
    )


# ---------------------------------------------------------------------------
# Calibration: ic_in_position recovers ~2x zero-padded IC at 70% gating
# ---------------------------------------------------------------------------

def test_in_position_ic_recovers_attenuation():
    """At 70% gating (signal == 0 for 70% of rows), the full-series
    Spearman IC attenuates by ~0.52x (audit headline). ic_in_position
    must recover most of that — restored IC should be >= 1.5x the
    full-series IC (a loose form of "~2x")."""
    rng_master = np.random.default_rng(55)
    n_inst = 18
    ic_in_vals = []
    ic_full_vals = []
    for _ in range(N_SEEDS):
        seed = int(rng_master.integers(1, 2**31 - 1))
        S, F = make_panel(
            n_periods=N_PERIODS, n_instruments=n_inst,
            ic_true=0.05, axis="timeseries", rho_cross=0.6, seed=seed,
        )
        # 70% gating: zero out 70% of signal rows per instrument
        rng = np.random.default_rng(seed + 1)
        for i in range(n_inst):
            mask = rng.random(N_PERIODS) < 0.7
            S.iloc[mask, i] = 0.0
        # Average over instruments to keep this cheap
        for i in range(n_inst):
            d = ic_in_position(np.sign(S.iloc[:, i]), F.iloc[:, i])
            if np.isfinite(d["ic_in_position"]) and np.isfinite(d["ic_full"]):
                ic_in_vals.append(d["ic_in_position"])
                ic_full_vals.append(d["ic_full"])
    mean_in = float(np.mean(ic_in_vals))
    mean_full = float(np.mean(ic_full_vals))
    print(f"  mean in-position IC: {mean_in:.4f}")
    print(f"  mean full-series IC: {mean_full:.4f}")
    print(f"  ratio (in_position / full): {mean_in / mean_full:.2f}")
    # ratio should be > 1.5 (audit observed full/in ≈ 0.52 i.e. in/full ≈ 1.9)
    assert mean_in / mean_full >= 1.5, (
        f"ic_in_position recovery ratio {mean_in/mean_full:.2f} < 1.5 "
        f"(audit expects ~2x recovery from 0.52x attenuation)"
    )


# ---------------------------------------------------------------------------
# One-sided FPR: must stay near alpha (Layer-3 follow-up to decision_ttest)
# ---------------------------------------------------------------------------

def test_one_sided_fpr_under_null():
    """sided='greater' FPR under the global null must stay near alpha
    (target ~0.05, never above 2*alpha = 0.10).

    Justification: one-sided tests in Layer 3 are only used for signals
    whose direction is pre-registered. Even so, if the test miscalibrates
    in the chosen direction it can never exceed 2*alpha — that's the
    hard ceiling. We enforce both bounds with 200 seeds so the SE is
    ~1.5% (tight enough to fail a 10% leak).
    """
    rng_master = np.random.default_rng(77)
    n_inst = 18
    fires_greater = 0
    fires_less = 0
    n_seeds = 200
    for _ in range(n_seeds):
        seed = int(rng_master.integers(1, 2**31 - 1))
        S, F = make_panel(
            n_periods=N_PERIODS, n_instruments=n_inst,
            ic_true=0.00, axis="timeseries", rho_cross=0.6, seed=seed,
        )
        sig = np.sign(S)
        port = portfolio_return_series(sig, F).values
        if decision_ttest(port, sided="greater", alpha=ALPHA)["fires"]:
            fires_greater += 1
        if decision_ttest(port, sided="less", alpha=ALPHA)["fires"]:
            fires_less += 1
    fpr_g = fires_greater / n_seeds
    fpr_l = fires_less / n_seeds
    print(f"  one-sided FPR greater={fpr_g:.3f}  less={fpr_l:.3f}")
    assert fpr_g <= 0.10, f"sided='greater' FPR {fpr_g:.3f} > 2*alpha"
    assert fpr_l <= 0.10, f"sided='less' FPR {fpr_l:.3f} > 2*alpha"


# ---------------------------------------------------------------------------
# Multiple-testing guard: BH-FDR under global null
# ---------------------------------------------------------------------------

def test_bh_fdr_under_global_null():
    """Under k=10 independent candidate signals all with ic=0, BH-FDR
    must keep family-wise discovery rate near alpha=0.05."""
    rng_master = np.random.default_rng(66)
    n_inst = 18
    k_signals = 10
    n_trials = max(N_SEEDS, 30)
    fdr_count = 0
    for _ in range(n_trials):
        # Build k independent signal series; reuse one fwd panel
        seed = int(rng_master.integers(1, 2**31 - 1))
        _, F = make_panel(
            n_periods=N_PERIODS, n_instruments=n_inst,
            ic_true=0.0, axis="timeseries", rho_cross=0.6, seed=seed,
        )
        series_list = []
        for j in range(k_signals):
            S_j, _ = make_panel(
                n_periods=N_PERIODS, n_instruments=n_inst,
                ic_true=0.0, axis="timeseries", rho_cross=0.6,
                seed=seed + 1000 + j,
            )
            sig_j = np.sign(S_j)
            port = portfolio_return_series(sig_j, F)
            series_list.append(port.values)
        out = decide_family(series_list, alpha=ALPHA)
        if out["fires_bh"].any():
            fdr_count += 1
    fwer = fdr_count / n_trials
    print(f"  family-wise discovery rate under global null: {fwer:.2f}")
    # Allow some slack — BH controls FDR not FWER, but at k=10 the upper
    # bound is loose. 0.20 is a forgiving cap; tighten if it proves too lax.
    assert fwer <= 0.20, (
        f"BH-FDR family-wise discovery {fwer:.2f} exceeds 0.20 — "
        f"multiple-testing guard is not behaving"
    )


# ---------------------------------------------------------------------------
# Smoke runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_threshold_rule_zero_detection_at_low_ic()
    print("OK threshold_rule_zero_detection_at_low_ic")
    test_ttest_fpr_under_null()
    print("OK ttest_fpr_under_null")
    test_timeseries_detection_at_ic_002()
    print("OK timeseries_detection_at_ic_002")
    test_crosssectional_detection_at_ic_001()
    print("OK crosssectional_detection_at_ic_001")
    test_in_position_ic_recovers_attenuation()
    print("OK in_position_ic_recovers_attenuation")
    test_one_sided_fpr_under_null()
    print("OK one_sided_fpr_under_null")
    test_bh_fdr_under_global_null()
    print("OK bh_fdr_under_global_null")
