"""
No-look-ahead test for the Layer-3 weekly CARRY signal.

Pattern mirrors test_momentum_nolookahead and TEST 1 of v2 causality:
scramble the DAILY carry_features values AFTER a cutoff DATE (using the
correct .loc[index, col] = scrambled idiom — NOT chained iloc), rebuild
the weekly carry signal panel, and assert that the weekly signal at all
week-ends <= cutoff is bit-identical.

Includes a self-check that the post-cutoff rows actually mutated and a
positive control: a deliberately leaky variant of build_daily_carry_signal
that READS THE FUTURE must be flagged as DIFFERENT under the scramble.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.weekly_panel import (  # noqa: E402
    build_weekly_panel,
    load_raw_inputs,
)
from stage3_signals.carry import (  # noqa: E402
    load_carry_features,
    build_daily_carry_signal,
    to_weekly_point_in_time,
    currency_of,
)


def _scramble_after(features: pd.DataFrame, cutoff_date: pd.Timestamp,
                    seed: int = 99) -> tuple:
    """Mutate post-cutoff rows of every column via .loc — never chained iloc.
    Returns (scrambled_df, post_cutoff_index)."""
    out = features.copy()
    post_idx = out.index[out.index > cutoff_date]
    rng = np.random.default_rng(seed)
    for col in out.columns:
        vals = out.loc[post_idx, col].to_numpy()
        out.loc[post_idx, col] = rng.permutation(vals)
    return out, post_idx


def _leaky_daily_carry_signal(carry_features, pairs):
    """POSITIVE CONTROL: a deliberately FUTURE-LEAKING daily carry signal.
    It uses carry_signal_pct.shift(-30), i.e. the rate-diff 30 trading days
    AHEAD. A no-look-ahead test that's actually checking anything must
    flag THIS as DIFFERENT under the post-cutoff scramble."""
    out = pd.DataFrame(index=carry_features.index)
    for p in pairs:
        ccy = currency_of(p)
        col = ("carry_signal_pct", ccy)
        if col in carry_features.columns:
            out[p] = carry_features[col].shift(-30)
        else:
            out[p] = np.nan
    return out


def test_weekly_carry_signal_invariant_to_future_scramble():
    ohlc, lr = load_raw_inputs()
    base = build_weekly_panel(ohlc, lr)
    pairs = base["universe"]
    week_idx = base["weekly_returns"].index

    cf = load_carry_features()
    cutoff_date = pd.Timestamp("2018-01-05")  # mid-sample weekday
    cf_scr, post_idx = _scramble_after(cf, cutoff_date, seed=42)

    pre_diff = (cf.loc[cf.index <= cutoff_date]
                - cf_scr.loc[cf.index <= cutoff_date]).abs().max().max()
    post_diff = (cf.loc[post_idx] - cf_scr.loc[post_idx]).abs().max().max()
    assert pre_diff == 0.0, (
        f"self-check: pre-cutoff rows changed (max diff {pre_diff}); "
        "scramble harness broken."
    )
    assert post_diff > 0, (
        f"self-check: post-cutoff rows unchanged ({post_diff}); the "
        "chained-iloc bug from Layer-2 FIX 4 is back."
    )

    # CAUSAL: original build_daily_carry_signal
    daily_causal_orig = build_daily_carry_signal(cf, pairs)
    daily_causal_scr = build_daily_carry_signal(cf_scr, pairs)
    weekly_causal_orig = to_weekly_point_in_time(daily_causal_orig, week_idx)
    weekly_causal_scr = to_weekly_point_in_time(daily_causal_scr, week_idx)
    pre_me = weekly_causal_orig.index[weekly_causal_orig.index <= cutoff_date]
    a = weekly_causal_orig.loc[pre_me].fillna(-9e18)
    b = weekly_causal_scr.loc[pre_me].fillna(-9e18)
    assert (a == b).all().all(), (
        "Weekly carry signal at week-ends <= cutoff changed after future "
        "scramble — look-ahead leak in build_daily_carry_signal."
    )

    # LEAKY POSITIVE CONTROL: must show pre-cutoff divergence
    daily_leaky_orig = _leaky_daily_carry_signal(cf, pairs)
    daily_leaky_scr = _leaky_daily_carry_signal(cf_scr, pairs)
    weekly_leaky_orig = to_weekly_point_in_time(daily_leaky_orig, week_idx)
    weekly_leaky_scr = to_weekly_point_in_time(daily_leaky_scr, week_idx)
    al = weekly_leaky_orig.loc[pre_me]
    bl = weekly_leaky_scr.loc[pre_me]
    leak_diff = (al.fillna(-9e18) != bl.fillna(-9e18)).any().any()
    assert leak_diff, (
        "POSITIVE CONTROL FAILED: a deliberately future-leaking carry "
        "signal (shift(-30)) was NOT flagged by the scramble. The test "
        "is not actually checking causality."
    )


if __name__ == "__main__":
    test_weekly_carry_signal_invariant_to_future_scramble()
    print("OK weekly_carry_signal_invariant_to_future_scramble")
