"""
Cost-accounting tests for the Layer-3 momentum backtest.

  - Zero-spread => net == gross exactly.
  - Positive spread strictly reduces net (whenever there is any turnover).
  - A book that never changes has turnover == 0 across all periods after
    the initial build.
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
    backtest_formation,
    cross_sectional_weights,
    turnover_series,
    cost_series,
    gross_return_series,
    currency_to_pair_map,
    cost_per_pair_bps,
    ALL_FORMATIONS_MONTHS,
)


def _setup():
    raw = load_universe_returns()
    oriented = orient_returns(raw)
    panel = build_monthly_panel(oriented, ALL_FORMATIONS_MONTHS)
    sig = panel["signals"][3]
    fwd = panel["fwd_ret"]
    return sig, fwd, list(oriented.columns)


def test_zero_spread_net_equals_gross():
    sig, fwd, cols = _setup()
    zero_spread = {c: 0.0 for c in cols}
    bt = backtest_formation(sig, fwd, q=1.0 / 3.0, spread_bps=zero_spread)
    diff = (bt["gross"] - bt["net"]).abs().max()
    assert diff < 1e-12, f"Zero spread should give net==gross; max diff {diff}"


def test_positive_spread_strictly_reduces_net():
    sig, fwd, cols = _setup()
    pos_spread = {c: 50.0 for c in cols}  # 50 bps round-trip
    bt = backtest_formation(sig, fwd, q=1.0 / 3.0, spread_bps=pos_spread)
    # On any month with turnover > 0, net < gross strictly.
    turn = bt["turnover"]
    months_with_turnover = turn[turn > 0].index
    assert len(months_with_turnover) > 5, (
        "Need at least a few turnover months to test the property"
    )
    for m in months_with_turnover:
        assert bt["net"].loc[m] < bt["gross"].loc[m], (
            f"net should be strictly less than gross on month {m} with "
            f"turnover {turn.loc[m]:.4f}; got net={bt['net'].loc[m]:.5f} "
            f"gross={bt['gross'].loc[m]:.5f}"
        )


def test_static_book_has_zero_turnover_after_build():
    """A weights frame that never changes after period 0 must have
    turnover==0 from period 1 onward. The first period builds the book
    so its turnover equals |w_0|."""
    dates = pd.bdate_range("2020-01-01", periods=10, freq="ME")
    cols = ["A", "B", "C", "D", "E", "F"]
    # Fixed long-A,B,C / short-D,E,F book
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    W.loc[:, ["A", "B", "C"]] = 1.0 / 3
    W.loc[:, ["D", "E", "F"]] = -1.0 / 3
    turn = turnover_series(W)
    # First period turnover = sum |w_0| = 6 * (1/3) = 2.0
    assert abs(turn.iloc[0] - 2.0) < 1e-12, (
        f"first-period turnover should equal sum |w_0| = 2.0; got {turn.iloc[0]}"
    )
    # All other periods: weights unchanged -> turnover == 0
    assert (turn.iloc[1:].abs() < 1e-12).all(), (
        f"static book should have zero turnover after period 0; "
        f"got {turn.iloc[1:].values}"
    )

    spread = {c: 10.0 for c in cols}  # 10 bps
    cost = cost_series(W, spread)
    assert (cost.iloc[1:].abs() < 1e-12).all(), (
        "static book should have zero cost after period 0"
    )


def test_cost_proportional_to_spread():
    """Doubling the spread doubles the cost (per period)."""
    sig, fwd, cols = _setup()
    W = cross_sectional_weights(sig)
    s10 = {c: 10.0 for c in cols}
    s20 = {c: 20.0 for c in cols}
    c10 = cost_series(W, s10)
    c20 = cost_series(W, s20)
    # Pick months with nonzero c10 only
    nz = c10[c10 > 0].index
    assert len(nz) > 5
    ratio = (c20.loc[nz] / c10.loc[nz])
    assert (ratio - 2.0).abs().max() < 1e-12, (
        f"cost should double when spread doubles; got ratios "
        f"min={ratio.min()} max={ratio.max()}"
    )


if __name__ == "__main__":
    test_zero_spread_net_equals_gross(); print("OK zero_spread_net_equals_gross")
    test_positive_spread_strictly_reduces_net(); print("OK positive_spread_strictly_reduces_net")
    test_static_book_has_zero_turnover_after_build(); print("OK static_book_has_zero_turnover_after_build")
    test_cost_proportional_to_spread(); print("OK cost_proportional_to_spread")
