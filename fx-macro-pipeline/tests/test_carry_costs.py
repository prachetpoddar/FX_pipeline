"""
Cost-accounting tests for the weekly carry backtest.

Mirrors tests/test_momentum_costs.py — same patterns, applied to the carry
weights/cost path so the helpers themselves stay correct when used through
the carry driver.

  - Zero spread → net == gross exactly.
  - Positive spread strictly reduces net on weeks with turnover.
  - Costs are linear in the spread vector.

These tests don't re-test the cost machinery from first principles (that's
already locked by tests/test_momentum_costs.py). They verify the carry
driver wires it up correctly: weights from the carry signal, fwd from the
weekly panel, spreads pair-keyed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from stage3_signals.carry import build_weekly_carry_panel  # noqa: E402
from stage3_signals.run_carry_weekly import (  # noqa: E402
    backtest_carry_signal,
)
from stage3_signals.run_momentum_weekly import (  # noqa: E402
    pair_spread_bps,
)


def _setup():
    panel = build_weekly_carry_panel()
    sig = panel["signal_primary"]
    fwd = panel["fwd_ret"]
    pairs = panel["universe"]
    return sig, fwd, pairs


def test_zero_spread_net_equals_gross():
    sig, fwd, pairs = _setup()
    zero = {p: 0.0 for p in pairs}
    bt = backtest_carry_signal(sig, fwd, zero)
    diff = (bt["gross"] - bt["net"]).abs().max()
    assert diff < 1e-12, (
        f"Zero spread should give net == gross; max diff {diff}"
    )


def test_positive_spread_strictly_reduces_net_on_turnover_weeks():
    sig, fwd, pairs = _setup()
    pos = {p: 50.0 for p in pairs}
    bt = backtest_carry_signal(sig, fwd, pos)
    turn = bt["turnover"]
    weeks_w_turnover = turn[turn > 1e-12].index
    assert len(weeks_w_turnover) > 5, (
        "Need at least a few turnover weeks to test the property"
    )
    for w in weeks_w_turnover:
        assert bt["net"].loc[w] < bt["gross"].loc[w], (
            f"On week {w} with turnover {turn.loc[w]:.4f} net should be "
            f"strictly < gross; got net={bt['net'].loc[w]:.6f} "
            f"gross={bt['gross'].loc[w]:.6f}"
        )


def test_cost_linear_in_spread():
    """Doubling the spread doubles the per-week cost everywhere."""
    sig, fwd, pairs = _setup()
    s10 = {p: 10.0 for p in pairs}
    s20 = {p: 20.0 for p in pairs}
    bt10 = backtest_carry_signal(sig, fwd, s10)
    bt20 = backtest_carry_signal(sig, fwd, s20)
    c10 = bt10["cost"]
    c20 = bt20["cost"]
    nz = c10[c10 > 1e-15].index
    assert len(nz) > 5
    ratio = (c20.loc[nz] / c10.loc[nz])
    assert (ratio - 2.0).abs().max() < 1e-9, (
        f"cost should double when spread doubles; got min={ratio.min()} "
        f"max={ratio.max()}"
    )


def test_default_tier_spreads_resolve():
    """pair_spread_bps must produce a per-pair float for every pair in
    the carry universe and use the registered tier-keyed defaults."""
    sig, fwd, pairs = _setup()
    sp = pair_spread_bps(pairs)
    for p in pairs:
        assert p in sp and sp[p] > 0


if __name__ == "__main__":
    test_zero_spread_net_equals_gross(); print("OK zero_spread_net_equals_gross")
    test_positive_spread_strictly_reduces_net_on_turnover_weeks()
    print("OK positive_spread_strictly_reduces_net_on_turnover_weeks")
    test_cost_linear_in_spread(); print("OK cost_linear_in_spread")
    test_default_tier_spreads_resolve(); print("OK default_tier_spreads_resolve")
