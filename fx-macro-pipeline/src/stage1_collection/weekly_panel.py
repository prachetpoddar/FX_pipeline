"""
weekly_panel.py
===============

SHARED Layer-1.5 weekly resampler. Read-only on data/processed_v2/.

  - Friday 17:00-NY close convention via pandas period 'W-FRI' on the
    existing daily index (already at 17:00-NY). Non-overlapping weeks.
  - Universe = intersection of fx_daily_ohlc.parquet pairs with
    fx_log_returns.parquet columns. AUTO-EXPANDS when A1/A2 land — never
    hardcode the pair list.
  - Per-pair, per-day BINARY active-day rule:
        active(day, pair) := num_bars[day, pair] >= FLOOR[pair]
        FLOOR[pair] := 0.5 * median(num_bars[:, pair])   (computed in-code)
  - A (week, pair) cell is KEPT iff it has >= MIN_ACTIVE_DAYS_PER_WEEK
    active days. Otherwise the weekly return is NaN (dropped from that
    week's cross-section).
  - ACTIVE-SINCE cohort: pairs with first available data after the
    universe start contribute only from their first listed week.
    Pre-listing cells are STATUS not_listed — DISTINCT from
    dropped_stale, so no audit can conflate "not present yet" with
    "data-quality drop".
  - Staleness diagnostic written alongside the panel for HUMAN AUDIT
    ONLY; it must never feed into a predictor or weight.

The weekly return per kept (week, pair) is the sum of daily oriented log
returns within the week (orientation map sourced from
src/stage3_signals/momentum.orientation_map — verified by
tests/test_momentum_orientation.py).

Outputs (when run as a script):
    data/weekly/weekly_returns.parquet           (T_weeks x N_pairs)
    data/weekly/weekly_staleness_diag.parquet    (long format)
    data/weekly/weekly_universe_status.parquet   (T_weeks x N_pairs of status)

Module API:
    build_weekly_panel(ohlc, log_returns) -> dict
        keys: weekly_returns, status, diag, floors, listing_dates, universe
    write_weekly_outputs(panel, out_dir)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.universe import UNIVERSE  # noqa: E402
from stage3_signals.momentum import orientation_map  # noqa: E402

DEFAULT_OHLC_PARQUET = PROJ / "data" / "processed_v2" / "fx_daily_ohlc.parquet"
DEFAULT_RETURNS_PARQUET = PROJ / "data" / "processed_v2" / "fx_log_returns.parquet"
DEFAULT_OUT_DIR = PROJ / "data" / "weekly"

WEEK_RULE = "W-FRI"               # Friday-anchored period
MIN_ACTIVE_DAYS_PER_WEEK = 3      # >= 3 active days required to KEEP a cell
FLOOR_FRACTION = 0.5              # active iff num_bars >= 0.5 * median(num_bars)

# Status tokens (kept short to fit cleanly in parquet)
STATUS_KEPT = "kept"
STATUS_DROPPED_STALE = "dropped_stale"
STATUS_NOT_LISTED = "not_listed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_floors(num_bars: pd.DataFrame) -> Dict[str, float]:
    """
    FLOOR[pair] = FLOOR_FRACTION * median(num_bars[:, pair]) over all days
    where num_bars is non-NaN. Returns a dict (no rounding — keep float
    precision; comparison is num_bars >= floor).
    """
    floors: Dict[str, float] = {}
    for pair in num_bars.columns:
        s = num_bars[pair].dropna()
        med = float(s.median()) if len(s) else float("nan")
        floors[pair] = FLOOR_FRACTION * med
    return floors


def first_listed_dates(num_bars: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    """First date a pair has any positive num_bars. Used to anchor
    active-since cohorts."""
    out: Dict[str, pd.Timestamp] = {}
    for pair in num_bars.columns:
        s = num_bars[pair].dropna()
        s = s[s > 0]
        if len(s):
            out[pair] = pd.Timestamp(s.index.min())
        else:
            out[pair] = pd.NaT
    return out


def week_label(daily_index: pd.DatetimeIndex) -> pd.Series:
    """Map each daily date to its Friday-ending week label (Timestamp at
    the Friday). Uses pandas Period 'W-FRI' then takes period end_time
    normalized to date."""
    p = pd.PeriodIndex(daily_index, freq=WEEK_RULE)
    # end_time is sunday for W-SUN etc; for W-FRI the period covers Sat..Fri
    # and end_time is the Friday at end-of-day. Normalize to the Friday date.
    fri = pd.DatetimeIndex(p.end_time.normalize())
    return pd.Series(fri, index=daily_index, name="week_end")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_weekly_panel(
    ohlc: pd.DataFrame,
    log_returns: pd.DataFrame,
    universe: Optional[dict] = None,
    min_active: int = MIN_ACTIVE_DAYS_PER_WEEK,
) -> dict:
    """
    Build the weekly oriented-return panel + per-cell status + staleness
    diagnostic.

    Parameters
    ----------
    ohlc : DataFrame
        Daily OHLC panel with MultiIndex columns [pair, field]. Must
        include 'num_bars' field.
    log_returns : DataFrame
        Daily log-returns panel (validated upstream).
    universe : dict, optional
        UNIVERSE dict (defaults to the package-level one). Only used for
        orientation; pair membership comes from the intersection of OHLC
        and log_returns columns.
    min_active : int
        Active-day floor per week.

    Returns
    -------
    dict with keys:
        weekly_returns     pd.DataFrame   (T_weeks x N_pairs)
        status             pd.DataFrame   (T_weeks x N_pairs) of status tokens
        diag               pd.DataFrame   long-format staleness diagnostic
        floors             dict[pair] -> float
        listing_dates      dict[pair] -> Timestamp
        universe           list[str] of active pairs
        week_index         pd.DatetimeIndex
    """
    u = universe or UNIVERSE
    # Universe = intersection (auto-expands when A1/A2 land)
    pairs = [c for c in log_returns.columns if (c, "num_bars") in ohlc.columns]
    if not pairs:
        raise RuntimeError("Empty weekly universe — OHLC and returns share no pairs.")

    # num_bars slice
    num_bars = ohlc.xs("num_bars", level="field", axis=1)[pairs]
    floors = compute_floors(num_bars)
    listing = first_listed_dates(num_bars)

    # Orientation: +1 if quote==USD, -1 if base==USD. Re-uses the same
    # map the momentum tests verify.
    omap = orientation_map(pairs, universe=u)

    # Build oriented daily log returns. Keep pair names (not currency)
    # so that the weekly panel can be joined to any other pair-keyed
    # frame downstream.
    oriented_daily = pd.DataFrame(index=log_returns.index)
    for p in pairs:
        oriented_daily[p] = float(omap[p]) * log_returns[p]

    # Align indexes between ohlc and log_returns (they should already be
    # identical per Surface check; assert).
    if not oriented_daily.index.equals(num_bars.index):
        # If they disagree, align to intersection.
        common = oriented_daily.index.intersection(num_bars.index)
        oriented_daily = oriented_daily.loc[common]
        num_bars = num_bars.loc[common]

    daily_index = oriented_daily.index
    week_lbl = week_label(daily_index)
    # Build active mask per (day, pair)
    floor_vec = pd.Series(floors).reindex(num_bars.columns)
    active_mask = num_bars.ge(floor_vec, axis=1).fillna(False)

    # Group daily rows by their week label and aggregate per pair.
    # Use groupby on the week label.
    grouped = oriented_daily.groupby(week_lbl.values)
    week_keys = pd.DatetimeIndex(sorted(week_lbl.unique()))

    n_active = num_bars.where(active_mask).groupby(week_lbl.values).count()
    # n_calendar_days_present = count of non-NaN num_bars rows in week
    n_cal = num_bars.groupby(week_lbl.values).count()
    weekly_ret_raw = oriented_daily.groupby(week_lbl.values).sum(min_count=1)

    # Build diagnostics: min, median, max num_bars per week + max stale run
    min_nb = num_bars.groupby(week_lbl.values).min()
    med_nb = num_bars.groupby(week_lbl.values).median()
    max_nb = num_bars.groupby(week_lbl.values).max()

    # zero_ret_fraction per (week, pair): share of daily oriented returns
    # in the week that are exactly 0.0 (proxy for intraweek staleness).
    zero_mask = (oriented_daily == 0.0)
    n_zero = zero_mask.groupby(week_lbl.values).sum()
    zero_frac = n_zero / n_cal.replace(0, np.nan)

    # max_stale_run within the week (consecutive identical daily returns).
    # Cheap-but-correct: compute per-pair per-week via groupby.apply on a
    # boolean series. Skipping pairs with no data in that week.
    def _max_stale_run(series: pd.Series) -> int:
        s = series.dropna()
        if len(s) < 2:
            return 0
        same = (s.diff() == 0).astype(int)
        if same.sum() == 0:
            return 0
        groups = (same == 0).cumsum()
        return int(same.groupby(groups).sum().max())

    max_runs = {}
    for pair in pairs:
        per = []
        for w, sub in oriented_daily[[pair]].groupby(week_lbl.values):
            per.append((w, _max_stale_run(sub[pair])))
        max_runs[pair] = pd.Series(dict(per))
    max_run_df = pd.DataFrame(max_runs).reindex(week_keys)

    # Status frame
    status = pd.DataFrame(STATUS_KEPT, index=week_keys, columns=pairs)
    listing_ts = pd.Series(listing).reindex(pairs)
    # Mark not_listed: week_end strictly before the first-listed date
    # snapped to that pair's first Friday week label.
    for p in pairs:
        first = listing_ts[p]
        if pd.isna(first):
            status[p] = STATUS_NOT_LISTED
            continue
        # First Friday-week that contains data for this pair
        first_week_p = week_lbl.loc[oriented_daily[p].first_valid_index()] \
            if oriented_daily[p].first_valid_index() is not None else None
        if first_week_p is not None:
            status.loc[status.index < first_week_p, p] = STATUS_NOT_LISTED

    # Mark dropped_stale: active days < min_active AND status currently kept
    drop_mask = (n_active.reindex(week_keys)[pairs].fillna(0)
                 < min_active)
    # Don't overwrite not_listed
    drop_mask = drop_mask & (status == STATUS_KEPT)
    status[drop_mask] = STATUS_DROPPED_STALE

    # Apply status to the weekly_returns frame: NaN if not kept
    weekly_returns = weekly_ret_raw.reindex(week_keys)[pairs].copy()
    weekly_returns[status != STATUS_KEPT] = np.nan

    # Long-format diagnostic
    diag_rows = []
    for p in pairs:
        diag_rows.append(pd.DataFrame({
            "week_end": week_keys,
            "pair": p,
            "n_active_days": n_active.reindex(week_keys)[p].fillna(0).astype(int).values,
            "n_calendar_days_present": n_cal.reindex(week_keys)[p].fillna(0).astype(int).values,
            "min_num_bars": min_nb.reindex(week_keys)[p].values,
            "median_num_bars": med_nb.reindex(week_keys)[p].values,
            "max_num_bars": max_nb.reindex(week_keys)[p].values,
            "max_stale_run": max_run_df[p].values if p in max_run_df.columns else np.nan,
            "zero_ret_fraction": zero_frac.reindex(week_keys)[p].fillna(np.nan).values,
            "status": status[p].values,
        }))
    diag = pd.concat(diag_rows, ignore_index=True)

    return {
        "weekly_returns": weekly_returns,
        "status": status,
        "diag": diag,
        "floors": floors,
        "listing_dates": {k: v for k, v in listing.items()},
        "universe": pairs,
        "week_index": week_keys,
        "orientation_map": omap,
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_raw_inputs(
    ohlc_path: Path = DEFAULT_OHLC_PARQUET,
    returns_path: Path = DEFAULT_RETURNS_PARQUET,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ohlc = pd.read_parquet(ohlc_path)
    log_returns = pd.read_parquet(returns_path)
    return ohlc, log_returns


def write_weekly_outputs(panel: dict, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel["weekly_returns"].to_parquet(out_dir / "weekly_returns.parquet")
    panel["diag"].to_parquet(out_dir / "weekly_staleness_diag.parquet")
    panel["status"].to_parquet(out_dir / "weekly_universe_status.parquet")


def main():
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    print(f"weekly_returns shape: {panel['weekly_returns'].shape}")
    print(f"universe (n={len(panel['universe'])}): {panel['universe']}")
    counts = panel["status"].apply(pd.value_counts).fillna(0).T \
        if False else panel["status"].apply(
            lambda c: c.value_counts(), axis=0).fillna(0).astype(int).T
    print(f"per-pair status counts:")
    print(counts)
    # Listing dates
    for p in panel["universe"]:
        print(f"  {p}: first listed {panel['listing_dates'][p].date() if not pd.isna(panel['listing_dates'][p]) else 'NaT'}; "
              f"floor num_bars = {panel['floors'][p]:.0f}")
    write_weekly_outputs(panel)
    print(f"\nWrote {DEFAULT_OUT_DIR}")


if __name__ == "__main__":
    main()
