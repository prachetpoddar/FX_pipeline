"""
No-look-ahead test for the Layer-3 monthly-momentum signal builder.

Scrambles all daily-oriented returns AFTER a cutoff month-end (using the
correct .loc[index, col] = scrambled idiom — chained-iloc is a no-op
under pandas 3.0.2; that's exactly the bug FIX 4 caught in Layer 2).
Then re-builds the monthly panel and asserts that signals AND ranks at
month-ends <= cutoff are bit-identical to the originals.

Includes a SELF-CHECK that the scramble actually mutated the post-cutoff
daily rows — same belt-and-braces pattern as test_v2_causality TEST 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from stage3_signals.momentum import (  # noqa: E402
    load_universe_returns,
    orient_returns,
    build_monthly_panel,
    cross_sectional_weights,
    ALL_FORMATIONS_MONTHS,
)


def _scramble_after(daily_df, cutoff_date, seed=12345):
    """Mutate post-cutoff rows in-place via .loc[idx, col] (NOT chained iloc).
    Returns the scrambled frame and the post-cutoff index for the self-check."""
    out = daily_df.copy()
    post_idx = out.index[out.index > cutoff_date]
    rng = np.random.default_rng(seed)
    for col in out.columns:
        vals = out.loc[post_idx, col].to_numpy()
        out.loc[post_idx, col] = rng.permutation(vals)
    return out, post_idx


def test_signals_invariant_to_future_scramble():
    raw = load_universe_returns()
    oriented = orient_returns(raw)

    # Pick a cutoff in the middle of the sample
    cutoff_pos = len(oriented) // 2
    cutoff_date = oriented.index[cutoff_pos]

    scrambled, post_idx = _scramble_after(oriented, cutoff_date, seed=12345)

    # SELF-CHECK: scramble actually mutated post-cutoff rows
    pre_diff = (oriented.loc[oriented.index <= cutoff_date]
                - scrambled.loc[oriented.index <= cutoff_date]).abs().max().max()
    post_diff = (oriented.loc[post_idx] - scrambled.loc[post_idx]).abs().max().max()
    assert pre_diff == 0.0, (
        f"Self-check failed: pre-cutoff rows changed (max diff {pre_diff}). "
        f"The .loc-based scramble should not touch them."
    )
    assert post_diff > 0, (
        f"Self-check failed: post-cutoff rows unchanged. The scramble was a "
        f"no-op (this is exactly the chained-iloc bug FIX 4 caught)."
    )

    # Build monthly panels from both
    panel_orig = build_monthly_panel(oriented, ALL_FORMATIONS_MONTHS)
    panel_scr = build_monthly_panel(scrambled, ALL_FORMATIONS_MONTHS)

    # For each formation: signals at month-ends <= cutoff_date must be
    # bit-identical (same goes for cross-sectional ranks built from them).
    pre_me = [m for m in panel_orig["signals"][3].index if m <= cutoff_date]
    assert len(pre_me) > 0, "no pre-cutoff month-ends found"

    for L in ALL_FORMATIONS_MONTHS:
        s_orig = panel_orig["signals"][L].loc[pre_me]
        s_scr = panel_scr["signals"][L].loc[pre_me]
        # Bit-identical (NaN-safe)
        eq = (s_orig.fillna(-9e18) == s_scr.fillna(-9e18)).all().all()
        assert eq, (
            f"L={L}: signals at month-ends <= cutoff differ after future "
            f"scramble (look-ahead leak)."
        )

        # Same for the tercile weights derived from the signal
        w_orig = cross_sectional_weights(s_orig)
        w_scr = cross_sectional_weights(s_scr)
        eq_w = (w_orig.fillna(-9e18) == w_scr.fillna(-9e18)).all().all()
        assert eq_w, (
            f"L={L}: tercile weights at month-ends <= cutoff differ after "
            f"future scramble."
        )


if __name__ == "__main__":
    test_signals_invariant_to_future_scramble()
    print("OK signals_invariant_to_future_scramble")
