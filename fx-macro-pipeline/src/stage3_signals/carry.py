"""
Layer-3 Factor 2: cross-sectional FX carry (weekly).

Read-only. Builds the weekly carry SIGNAL panel (primary = lagged short-rate
differential XXX-USD; secondary = CIP-forward-implied carry) aligned to the
SAME weekly return panel produced by src/stage1_collection_v2/weekly_panel.py.
Cross-sectional convention: signal value > 0 means the currency has HIGHER
yield than USD (high-yielder), so the carry hypothesis predicts that currency
to strengthen vs USD on average. Both signal columns are PAIR-keyed (matching
the weekly_returns panel) and orientation-consistent with the oriented weekly
returns.

Convention checks (verified at run time, asserted in tests):

  PRIMARY: signal[pair][t] = carry_signal_pct[currency(pair)][t]
    - currency(pair) is the non-USD leg per UNIVERSE.
    - carry_signal_pct is rate_diff_pct lagged by 1 trading day (already
      built that way in carry_features.parquet); using it directly is
      point-in-time and no-look-ahead.

  SECONDARY (CIP): signal[pair][t] =
    -orientation(pair) * cip_fwd_points_pair[pair][t] / spot_close[pair][t]
    - Sign derivation: for an XXXUSD quote, the forward (USD per XXX)
      satisfies F/S = (1+i_USD)/(1+i_XXX). If F<S (cip<0) then i_USD<i_XXX,
      so XXX is the HIGH yielder. orientation(XXXUSD)=+1, so
      -orientation*cip = -1*cip = positive → high yielder. ✓
    - For a USDXXX quote: F/S = (1+i_XXX)/(1+i_USD). F>S (cip>0) iff
      i_XXX>i_USD, so XXX is the high yielder. orientation(USDXXX)=-1,
      so -orientation*cip = +1*cip = positive → high yielder. ✓
    - /spot normalises the cross-pair magnitudes so the ranks are not
      dominated by the JPY/HUF price-level scale.

The basis (primary - secondary) is the cross-currency basis and reported
explicitly in the carry-vs-reversal overlap section of the report.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.universe import UNIVERSE  # noqa: E402
from stage1_collection_v2.weekly_panel import (  # noqa: E402
    build_weekly_panel,
    load_raw_inputs,
    week_label,
)
from stage3_signals.momentum import orientation_map  # noqa: E402

DEFAULT_CARRY_PARQUET = PROJ / "data" / "processed_v2" / "carry_features.parquet"
DEFAULT_CLOSE_PARQUET = PROJ / "data" / "processed_v2" / "fx_close_prices.parquet"

OUT_DIR = PROJ / "data" / "layer3_carry"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_carry_features(path: Path = DEFAULT_CARRY_PARQUET) -> pd.DataFrame:
    """MultiIndex columns [field, key]. field ∈
    {carry_signal_pct, cip_fwd_points_pair, rate_diff_pct}."""
    return pd.read_parquet(path)


def load_close_prices(path: Path = DEFAULT_CLOSE_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


def currency_of(pair: str, universe: Optional[dict] = None) -> str:
    """Return the non-USD currency code of an FX pair."""
    u = universe or UNIVERSE
    meta = u[pair]
    return meta["base"] if meta["quote"] == "USD" else meta["quote"]


# ---------------------------------------------------------------------------
# Daily carry signal panels (pair-keyed; pre-resample)
# ---------------------------------------------------------------------------

def build_daily_carry_signal(carry_features: pd.DataFrame,
                             pairs: Sequence[str],
                             universe: Optional[dict] = None) -> pd.DataFrame:
    """
    PRIMARY: daily series of carry_signal_pct[currency(pair)] for each pair.
    Column = pair code (so it joins to the weekly return panel which is
    pair-keyed). Value = rate differential currency-USD (already lagged
    one trading day in the source — point-in-time at the daily index).
    """
    out = pd.DataFrame(index=carry_features.index)
    for p in pairs:
        ccy = currency_of(p, universe)
        col = ("carry_signal_pct", ccy)
        if col in carry_features.columns:
            out[p] = carry_features[col]
        else:
            out[p] = np.nan
    return out


def build_daily_cip_signal(carry_features: pd.DataFrame,
                           close_prices: pd.DataFrame,
                           pairs: Sequence[str],
                           universe: Optional[dict] = None) -> pd.DataFrame:
    """
    SECONDARY: daily CIP-forward-implied carry per pair. Sign-flipped via
    -orientation(pair) (see module docstring) so high value = high yielder.
    Normalized by spot close to remove the quote-level scale across pairs
    (else USDHUF and USDCHF would not be cross-sectionally comparable).
    """
    u = universe or UNIVERSE
    omap = orientation_map(list(pairs), universe=u)
    out = pd.DataFrame(index=carry_features.index)
    for p in pairs:
        col = ("cip_fwd_points_pair", p)
        if col not in carry_features.columns:
            out[p] = np.nan
            continue
        cip = carry_features[col].astype(float)
        # Align spot to the same index; outer reindex
        if p in close_prices.columns:
            spot = close_prices[p].reindex(carry_features.index).astype(float)
        else:
            out[p] = np.nan
            continue
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = cip / spot.where(spot.abs() > 1e-12)
        # Lag by 1 day to match the primary's point-in-time convention
        # (carry_signal_pct is already lagged in source; we mirror that
        # here so the two signals are on the same information set).
        out[p] = -float(omap[p]) * norm.shift(1)
    return out


# ---------------------------------------------------------------------------
# Resample to Friday-NY week-end (point-in-time)
# ---------------------------------------------------------------------------

def to_weekly_point_in_time(daily_signal: pd.DataFrame,
                            week_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Take the LAST available value of `daily_signal` on or before each week
    end (Friday). Point-in-time: never reads ahead of the week-end date.
    """
    di = daily_signal.copy()
    di.index = pd.DatetimeIndex(di.index)
    out = pd.DataFrame(index=week_index, columns=di.columns, dtype=float)
    # Reindex forward-fill across all daily dates, then pick week-ends
    full_idx = di.index.union(week_index).sort_values()
    ff = di.reindex(full_idx).ffill()
    out = ff.reindex(week_index)
    return out


# ---------------------------------------------------------------------------
# End-to-end weekly carry panel
# ---------------------------------------------------------------------------

def build_weekly_carry_panel(
    carry_features: Optional[pd.DataFrame] = None,
    close_prices: Optional[pd.DataFrame] = None,
    universe: Optional[dict] = None,
) -> dict:
    """
    Build the weekly carry panel: (signal_primary, signal_secondary,
    fwd_ret) all on the SAME Friday-NY week index and the SAME live
    universe as the resampler.

    Returns
    -------
    dict with keys:
        weekly_returns   pd.DataFrame  (oriented weekly returns, kept-only)
        signal_primary   pd.DataFrame  (carry_signal_pct, pair-keyed)
        signal_secondary pd.DataFrame  (CIP-implied carry, pair-keyed)
        fwd_ret          pd.DataFrame  (signal aligned to next-week return)
        week_index_signal pd.DatetimeIndex (signals/fwd index; last week dropped)
        universe         list[str]
        floors, listing_dates, status, diag (from weekly_panel)
    """
    if carry_features is None:
        carry_features = load_carry_features()
    if close_prices is None:
        close_prices = load_close_prices()
    ohlc, lr = load_raw_inputs()
    base = build_weekly_panel(ohlc, lr, universe=universe)
    pairs = base["universe"]

    daily_primary = build_daily_carry_signal(carry_features, pairs, universe)
    daily_secondary = build_daily_cip_signal(carry_features, close_prices,
                                             pairs, universe)

    weekly_returns = base["weekly_returns"]
    week_idx = weekly_returns.index

    weekly_primary = to_weekly_point_in_time(daily_primary, week_idx)
    weekly_secondary = to_weekly_point_in_time(daily_secondary, week_idx)

    # Restrict signals to KEPT cells (NaN where return panel is NaN). The
    # signal is informationally fine outside kept cells but we won't trade
    # those positions, so excluding them keeps the cross-section honest.
    keep_mask = weekly_returns.notna()
    weekly_primary = weekly_primary.where(keep_mask)
    weekly_secondary = weekly_secondary.where(keep_mask)

    # Forward returns: signal at week w predicts return in w+1
    fwd = weekly_returns.shift(-1)
    # Drop the last row from signals + fwd (no fwd to realize)
    week_idx_signal = week_idx[:-1]
    signal_primary = weekly_primary.iloc[:-1]
    signal_secondary = weekly_secondary.iloc[:-1]
    fwd = fwd.iloc[:-1]

    return {
        "weekly_returns": weekly_returns,
        "signal_primary": signal_primary,
        "signal_secondary": signal_secondary,
        "fwd_ret": fwd,
        "week_index_signal": week_idx_signal,
        "universe": pairs,
        "floors": base["floors"],
        "listing_dates": base["listing_dates"],
        "status": base["status"],
        "diag": base["diag"],
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_weekly_carry_outputs(panel: dict,
                               out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Long-format for human auditing
    rows = []
    sig_p = panel["signal_primary"]
    sig_s = panel["signal_secondary"]
    fwd = panel["fwd_ret"]
    for w in sig_p.index:
        for c in sig_p.columns:
            rows.append({
                "week_end": w, "pair": c,
                "carry_primary_pct": sig_p.loc[w, c],
                "cip_secondary_norm": sig_s.loc[w, c],
                "fwd_ret_w_plus_1": fwd.loc[w, c],
            })
    pd.DataFrame(rows).to_parquet(out_dir / "weekly_carry_panel.parquet")


if __name__ == "__main__":
    panel = build_weekly_carry_panel()
    p, s = panel["signal_primary"], panel["signal_secondary"]
    print(f"signal_primary   shape: {p.shape}  non-NaN: {p.notna().sum().sum()}")
    print(f"signal_secondary shape: {s.shape}  non-NaN: {s.notna().sum().sum()}")
    print()
    print("Mean signal value per pair (primary, full sample):")
    print(p.mean().sort_values(ascending=False).to_string())
    print()
    print("Mean signal value per pair (CIP secondary, full sample):")
    print(s.mean().sort_values(ascending=False).to_string())
    write_weekly_carry_outputs(panel)
    print(f"\nWrote {OUT_DIR/'weekly_carry_panel.parquet'}")
