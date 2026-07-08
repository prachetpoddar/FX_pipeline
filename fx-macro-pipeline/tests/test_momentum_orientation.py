"""
Orientation tests for Layer-3 momentum.

  - Oriented signs follow the universe (XXXUSD -> +1, USDXXX -> -1).
  - Equal-weight oriented mean tracks -USD index strongly.
  - POSITIVE CONTROL: a synthetic pair with a known sign mis-orients if the
    map is inverted — proves the check has teeth (would have caught a bug
    that flipped the convention).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.universe import UNIVERSE  # noqa: E402
from stage3_signals.momentum import (  # noqa: E402
    orientation_map,
    orient_returns,
    load_universe_returns,
)


def test_orientation_signs():
    """Spot-check signs against UNIVERSE."""
    pairs = list(load_universe_returns().columns)
    omap = orientation_map(pairs)
    assert omap["EURUSD"] == +1
    assert omap["GBPUSD"] == +1
    assert omap["AUDUSD"] == +1
    assert omap["NZDUSD"] == +1
    assert omap["USDJPY"] == -1
    assert omap["USDCHF"] == -1
    assert omap["USDCAD"] == -1


def test_oriented_correlations():
    raw = load_universe_returns()
    oriented = orient_returns(raw)
    eur_corr = raw["EURUSD"].corr(oriented["EUR"])
    jpy_corr = raw["USDJPY"].corr(oriented["JPY"])
    assert eur_corr > 0.999, f"EUR oriented corr {eur_corr:.4f} should be +1"
    assert jpy_corr < -0.999, f"JPY oriented corr {jpy_corr:.4f} should be -1"


def test_mean_oriented_tracks_negative_usd_index():
    raw = load_universe_returns()
    oriented = orient_returns(raw)
    # USD index from USDXXX pairs (where base==USD): positive = USD up
    usd_idx_pairs = [p for p in raw.columns if UNIVERSE[p]["base"] == "USD"]
    usd_idx = raw[usd_idx_pairs].mean(axis=1)
    mean_or = oriented.mean(axis=1)
    com = mean_or.dropna().index.intersection(usd_idx.dropna().index)
    corr = mean_or.loc[com].corr(usd_idx.loc[com])
    assert corr < -0.9, (
        f"Equal-weight oriented mean should track -USD index strongly; "
        f"got corr={corr:+.3f}"
    )


def test_positive_control_inverted_map_breaks_corr():
    """If we INVERT the orientation map (flip every sign), the corr with
    the USD index should change sign — proves the test is sensitive to
    the orientation convention, not just to any sign-coherent panel."""
    raw = load_universe_returns()
    # Hand-build an inverted oriented panel
    oriented_inv = pd.DataFrame(index=raw.index)
    omap = orientation_map(list(raw.columns))
    for col in raw.columns:
        sign = -omap[col]  # INVERTED
        meta = UNIVERSE[col]
        ccy = meta["base"] if omap[col] == +1 else meta["quote"]
        oriented_inv[ccy] = sign * raw[col]
    usd_idx_pairs = [p for p in raw.columns if UNIVERSE[p]["base"] == "USD"]
    usd_idx = raw[usd_idx_pairs].mean(axis=1)
    mean_inv = oriented_inv.mean(axis=1)
    com = mean_inv.dropna().index.intersection(usd_idx.dropna().index)
    corr_inv = mean_inv.loc[com].corr(usd_idx.loc[com])
    # Inverted map should now POSITIVELY correlate with USD index
    assert corr_inv > 0.9, (
        f"POSITIVE CONTROL FAILED: inverting the orientation map should "
        f"flip the sign of the corr-with-USD-index. Got corr_inv={corr_inv:+.3f}; "
        f"if this isn't strongly positive, the test isn't actually checking "
        f"the orientation convention."
    )


if __name__ == "__main__":
    test_orientation_signs(); print("OK orientation_signs")
    test_oriented_correlations(); print("OK oriented_correlations")
    test_mean_oriented_tracks_negative_usd_index(); print("OK mean_oriented_tracks_negative_usd_index")
    test_positive_control_inverted_map_breaks_corr(); print("OK positive_control_inverted_map_breaks_corr")
