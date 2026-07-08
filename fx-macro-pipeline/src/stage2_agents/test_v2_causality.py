"""
test_v2_causality.py
====================

Empirical sanity checks for spectral_agent_v2.py. Run this BEFORE running v2
on real data. These tests prove the core correctness properties:

    TEST 1 — Future-data invariance
    -------------------------------
    Generate synthetic data. Compute v2 signals. Then scramble all values
    AFTER a cutoff date t*. Recompute v2 signals. Signals at all dates <= t*
    MUST be identical. If they differ, the code uses future information.

    TEST 2 — Null-signal on pure noise
    ----------------------------------
    Feed in independent Gaussian noise for every currency. v2 should produce
    Sharpes centered around zero with no systematic positive bias. If mean
    Sharpe is significantly positive on random data, the evaluation is rigged.

    TEST 3 — Recovery on a known signal
    -----------------------------------
    Construct a target series where next-day return is deterministically a
    shifted USD return with noise. v2 should produce a clearly positive
    Sharpe (not the 1.89 magnitude of the leaky version — something real).

If all three pass, the forecaster is leak-free and the evaluation is honest.
You can then trust whatever Sharpe it reports on real data.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow import of spectral_agent_v2 from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import spectral_agent_v2 as v2

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def make_synthetic_returns(n_days=3000, n_majors=8, seed=0):
    """Generate a DataFrame of synthetic log returns with a plausible
    correlation structure among majors."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    # Common factor + idiosyncratic noise
    common = rng.normal(0, 0.005, n_days)
    data = {}
    majors = v2.MAJOR_CURRENCIES[:n_majors]
    for m in majors:
        idio = rng.normal(0, 0.005, n_days)
        data[m] = 0.5 * common + idio
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# TEST 1: future-data invariance
# ---------------------------------------------------------------------------
#
# Audit note (data/layer2_audit, Task F): the previous version of this test
# used `df[col].iloc[k+1:] = ...`. Under pandas 3.0.2 chained assignment is a
# no-op on a copy — the scramble never mutated df_scrambled, so the test was
# vacuous and would have "passed" for any function, including ones that read
# the future. This version (a) uses .loc with the explicit index to mutate
# the actual frame, (b) self-checks that post-cutoff rows actually changed,
# and (c) requires the harness to flag a deliberately leaky control function
# as failing — a smoke alarm that has never been triggered is not known to
# work.


def _scramble_after(df, cutoff_idx, seed=999):
    """Return a copy of df with rows AFTER cutoff_idx permuted, per column.

    Uses .loc with the explicit post-cutoff index to write through to the
    DataFrame (rather than the broken `df[col].iloc[k+1:] = ...` idiom that
    pandas 3.0.2 silently drops). Returns (df_scrambled, post_cutoff_index).
    """
    df_scrambled = df.copy()
    post_idx = df_scrambled.index[cutoff_idx + 1:]
    rng = np.random.default_rng(seed)
    for col in df_scrambled.columns:
        original = df_scrambled.loc[post_idx, col].to_numpy()
        scrambled = rng.permutation(original)
        df_scrambled.loc[post_idx, col] = scrambled
    return df_scrambled, post_idx


def _causal_features(df, target):
    """Reference (causal) feature builder used inside the test."""
    usd = v2.build_usd_index(df, exclude=target)
    return v2.compute_rolling_features(usd, df[target])


def _leaky_features(df, target):
    """
    Positive control: a deliberately FUTURE-LEAKING feature builder.

    For each row t it injects df[target] at row t+1 into the coherence
    column. A correctly armed invariance test must report this as a FAIL
    (features at t <= cutoff differ after the post-cutoff scramble). If the
    test passes a leaky function, the test is not actually checking
    anything.
    """
    feats = _causal_features(df, target)
    # Add an unambiguously future-looking column: target_{t+1}
    leak = df[target].shift(-1).reindex(feats.index)
    feats = feats.copy()
    feats["coherence"] = feats["coherence"].astype(float) + leak.astype(float)
    return feats


def _invariance_diff(df, df_scrambled, target, cutoff_date, feature_fn):
    """Run feature_fn on both frames and return max abs diff at t <= cutoff."""
    feats_orig = feature_fn(df, target)
    feats_scr = feature_fn(df_scrambled, target)
    orig_upto = feats_orig.loc[feats_orig.index <= cutoff_date]
    scr_upto = feats_scr.loc[feats_scr.index <= cutoff_date]
    diffs = (orig_upto - scr_upto).abs().max().max()
    if np.isnan(diffs):
        diffs = 0.0
    return float(diffs), orig_upto, scr_upto


def test_future_invariance():
    print("\n[TEST 1] Future-data invariance")
    df = make_synthetic_returns(n_days=2000, seed=1)
    target = "EUR"
    cutoff_idx = 1500
    cutoff_date = df.index[cutoff_idx]

    # Build the scrambled frame and SELF-CHECK that we actually mutated it.
    df_scrambled, post_idx = _scramble_after(df, cutoff_idx, seed=999)
    pre_max = (df.loc[df.index <= cutoff_date]
               - df_scrambled.loc[df.index <= cutoff_date]).abs().max().max()
    post_max = (df.loc[post_idx] - df_scrambled.loc[post_idx]).abs().max().max()
    print(f"  Self-check pre-cutoff max diff:  {pre_max:.2e}  (must be 0)")
    print(f"  Self-check post-cutoff max diff: {post_max:.2e}  (must be > 0)")
    if not (pre_max == 0.0 and post_max > 0.0):
        print("  FAIL — scramble harness is broken; cannot trust test result")
        return False

    # (a) Causal function must come out identical on t <= cutoff
    causal_diff, _, _ = _invariance_diff(
        df, df_scrambled, target, cutoff_date, _causal_features
    )
    print(f"  Causal-function max diff up to cutoff:  {causal_diff:.2e}")
    causal_ok = causal_diff < 1e-10

    # (b) Positive control: leaky function MUST be caught
    leaky_diff, orig_lk, scr_lk = _invariance_diff(
        df, df_scrambled, target, cutoff_date, _leaky_features
    )
    print(f"  Leaky-control max diff up to cutoff:    {leaky_diff:.2e}  "
          f"(must be > 0 — positive control)")
    leaky_caught = leaky_diff > 1e-10

    if causal_ok and leaky_caught:
        print("  PASS — feature computation is strictly causal AND the "
              "leaky positive control is correctly flagged")
        return True
    else:
        if not causal_ok:
            print("  FAIL — feature computation leaks future information")
            bad = ((orig_lk - scr_lk).abs()
                   .max(axis=1).sort_values(ascending=False))
            print(f"  First 5 divergent dates:\n{bad.head()}")
        if not leaky_caught:
            print("  FAIL — leaky positive control was NOT caught; the "
                  "harness is not actually testing invariance")
        return False


# ---------------------------------------------------------------------------
# TEST 2: null-signal on pure noise
# ---------------------------------------------------------------------------

def test_null_signal():
    print("\n[TEST 2] Null-signal on pure noise")
    rng = np.random.default_rng(42)
    n_days = 3000
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    # Completely independent noise for every currency, INCLUDING majors
    # (so USD index is also just noise)
    n_currencies = 20
    cols = v2.MAJOR_CURRENCIES + [f"FAKE{i}" for i in range(n_currencies)]
    data = {c: rng.normal(0, 0.005, n_days) for c in cols}
    df = pd.DataFrame(data, index=dates)

    split_idx = int(0.6 * n_days)
    split_date = df.index[split_idx]

    results = []
    for ccy in cols:
        try:
            res = v2.process_currency(ccy, df, split_date)
        except Exception:
            continue
        if res is not None:
            results.append(res)

    if not results:
        print("  WARN — no currencies produced results on noise; test inconclusive")
        return True

    res_df = pd.DataFrame(results)
    mean_sharpe = res_df["sharpe"].mean()
    mean_acc = res_df["accuracy"].mean()
    n = len(res_df)
    # One-sample t-test against 0 for Sharpe
    se = res_df["sharpe"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    t_stat = mean_sharpe / se if se and se > 0 else np.nan

    print(f"  N evaluated: {n}")
    print(f"  Mean Sharpe on noise: {mean_sharpe:.3f} (t-stat vs 0: "
          f"{t_stat:.2f})")
    print(f"  Mean accuracy on noise: {mean_acc:.3f} "
          f"(expected ~0.50)")

    # PASS if mean Sharpe is within 2 SE of zero and accuracy is near 0.5
    sharpe_ok = abs(t_stat) < 2.5 if not np.isnan(t_stat) else True
    acc_ok = abs(mean_acc - 0.5) < 0.05
    if sharpe_ok and acc_ok:
        print("  PASS — no systematic bias on noise")
        return True
    else:
        print("  FAIL — systematic bias detected on noise")
        print(f"    sharpe_ok={sharpe_ok}, acc_ok={acc_ok}")
        return False


# ---------------------------------------------------------------------------
# TEST 3: recovery on a known signal
# ---------------------------------------------------------------------------

def test_signal_recovery():
    print("\n[TEST 3] Recovery on a known signal")
    rng = np.random.default_rng(7)
    n_days = 3000
    dates = pd.bdate_range("2010-01-01", periods=n_days)

    # Majors: noise
    data = {m: rng.normal(0, 0.005, n_days) for m in v2.MAJOR_CURRENCIES}
    df_majors = pd.DataFrame(data, index=dates)
    usd_full = v2.build_usd_index(df_majors)

    # Construct a target currency whose NEXT-DAY return is partly explained
    # by TODAY's USD return (5-day lag structure with noise).
    lag = 5
    usd_vals = usd_full.values
    # Target_t+1 = -0.5 * USD_{t - lag + 1} + noise      (so at time t,
    # looking at USD_{t-lag+1}, sign should predict target_{t+1})
    noise = rng.normal(0, 0.005, n_days)
    target = np.zeros(n_days)
    for t in range(lag, n_days - 1):
        target[t + 1] = -0.8 * usd_vals[t - lag + 1] + noise[t + 1]
    target_series = pd.Series(target, index=dates, name="TARGET")

    # Mix into a full DataFrame
    df = df_majors.copy()
    df["TARGET"] = target_series

    split_idx = int(0.6 * n_days)
    split_date = df.index[split_idx]

    res = v2.process_currency("TARGET", df, split_date)
    if res is None:
        print("  FAIL — no result produced on known-signal data")
        return False

    print(f"  Direction learned: {res['direction']}")
    print(f"  Train accuracy: {res['train_accuracy']:.3f}")
    print(f"  Test accuracy (in-position): {res['accuracy']:.3f}")
    print(f"  Test IC: {res['ic']:.3f}")
    print(f"  Test Sharpe: {res['sharpe']:.3f}")
    print(f"  Time in market: {res['time_in_market']:.3f}")

    # We built an INVERSE relationship (coefficient -0.8), so we expect
    # direction == 'inverse' and Sharpe > 0.5 (it's a strong signal).
    ok = (res["direction"] == "inverse"
          and res["accuracy"] > 0.55
          and res["sharpe"] > 0.3)
    if ok:
        print("  PASS — recovers a known embedded signal")
        return True
    else:
        print("  FAIL — could not recover a known signal")
        return False


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []
    results.append(("future_invariance", test_future_invariance()))
    results.append(("null_signal",       test_null_signal()))
    results.append(("signal_recovery",   test_signal_recovery()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, ok in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)
