"""
daily_aggregator.py
===================

Aggregates 1-minute HistData bars into daily OHLC bars using the
FX-standard 17:00 New York time daily close convention.

WHY 17:00 NY
------------
The global FX market is open 24×5 (Sunday 17:00 NY through Friday 17:00 NY).
Every retail FX broker uses 17:00 NY as the daily close. This is also
the close used by virtually all FX research papers and risk-management
systems. Using any other convention (e.g., 16:00 CET like ECB references,
or midnight UTC) creates an artificial day boundary that doesn't align
with how anyone actually trades or measures FX P&L.

WHAT 17:00 NY MEANS IN UTC
--------------------------
NY uses Eastern Time, which is UTC-5 (EST) or UTC-4 (EDT) depending on
daylight savings. So:
    Winter (EST): 17:00 NY = 22:00 UTC
    Summer (EDT): 17:00 NY = 21:00 UTC

We handle this by converting the HistData timestamps to America/New_York
timezone, then defining a daily session as 17:00 NY of day N-1 to 17:00 NY
of day N, with "day N" being the calendar date in New York of the *close*.

OHLC AGGREGATION
----------------
For each NY-day d:
    Open   = first 1-minute bar's Open in (d-1, 17:00 NY → d, 17:00 NY]
    High   = max High of all bars in window
    Low    = min Low of all bars in window
    Close  = last 1-minute bar's Close in window  (= price at d 17:00 NY)
    NumBars = count of 1-minute bars present (for liquidity diagnostics)

FX TRADING WEEK
---------------
Sun 17:00 NY → Mon 17:00 NY  → labelled as Monday
Mon 17:00 NY → Tue 17:00 NY  → Tuesday
...
Thu 17:00 NY → Fri 17:00 NY  → Friday
Fri 17:00 NY → Sun 17:00 NY  : market closed → no daily bar

So we expect ~5 daily bars per week. Weekend bars (Saturdays, Sundays in
NY date) will be silently absent.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_1min_csv(path):
    """
    Load a single HistData 1-minute CSV.
    Format: DateTime;Open;High;Low;Close;Volume (semicolon-separated)
    DateTime is YYYYMMDD HHMMSS in EST (NY).
    """
    df = pd.read_csv(
        path, sep=";", header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
        dtype={"open": float, "high": float, "low": float,
               "close": float, "volume": float},
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")
    # Localize to America/New_York (HistData ASCII files are EST per spec,
    # but they use America/New_York semantics — EST in winter, EDT in summer)
    df["datetime"] = df["datetime"].dt.tz_localize(
        "America/New_York", ambiguous="infer", nonexistent="shift_forward"
    )
    df = df.set_index("datetime").sort_index()
    return df


def aggregate_to_daily(minute_df):
    """
    Aggregate 1-minute bars to daily OHLC using a 17:00 NY close.

    Implementation: we label each minute with the "session_date" =
    NY-calendar-date of the 17:00 close that closes its session.
    A bar at Tue 14:00 NY belongs to Tue session (closes Tue 17:00).
    A bar at Tue 18:00 NY belongs to Wed session (closes Wed 17:00).
    A bar at Tue 17:00 NY exactly: by convention, this is the closing
    print of Tue session, so it belongs to Tue.
    """
    ny = minute_df.index.tz_convert("America/New_York")
    hour = ny.hour
    minute = ny.minute
    # Bars strictly after 17:00 belong to next session
    after_close = (hour > 17) | ((hour == 17) & (minute > 0))
    session_date = ny.normalize()
    session_date = session_date + pd.to_timedelta(after_close.astype(int), unit="D")
    session_date = session_date.tz_localize(None).normalize()

    g = minute_df.groupby(session_date)
    daily = pd.DataFrame({
        "open":     g["open"].first(),
        "high":     g["high"].max(),
        "low":      g["low"].min(),
        "close":    g["close"].last(),
        "num_bars": g["close"].count(),
    })
    daily.index.name = "date"

    # Drop weekend-labeled sessions. A bar exactly at Sun 17:00 NY would
    # be labelled Sunday by the groupby (since 17:00:00 is at the boundary,
    # not after it). In real markets that's the FX open tick. Either way,
    # Mon→Fri sessions are the valid ones; Sat and Sun labels are dropped.
    weekday = pd.to_datetime(daily.index).dayofweek
    weekend_mask = weekday >= 5    # 5=Sat, 6=Sun
    if weekend_mask.any():
        logger.warning(
            f"    Dropping {weekend_mask.sum()} weekend-labelled sessions "
            f"(Sat/Sun — unexpected for a 17:00-NY close convention)"
        )
        daily = daily[~weekend_mask]
    return daily


def aggregate_pair(pair_code, cache_dir, start_year, end_year):
    """
    Find all 1-minute CSVs for a pair, aggregate each to daily, concatenate.
    Returns DataFrame indexed by date with columns open/high/low/close/num_bars.
    """
    pair_dir = Path(cache_dir) / pair_code.upper()
    if not pair_dir.exists():
        logger.warning(f"  {pair_code}: no data directory at {pair_dir}")
        return None

    daily_frames = []
    for year in range(start_year, end_year + 1):
        path = pair_dir / f"DAT_ASCII_{pair_code.upper()}_M1_{year}.csv"
        if not path.exists() or path.stat().st_size < 1000:
            logger.debug(f"  {pair_code} {year}: missing or empty")
            continue
        try:
            minute_df = load_1min_csv(path)
        except Exception as e:
            logger.warning(f"  {pair_code} {year}: load failed — {e}")
            continue
        try:
            daily = aggregate_to_daily(minute_df)
            daily_frames.append(daily)
        except Exception as e:
            logger.warning(f"  {pair_code} {year}: aggregate failed — {e}")
            continue

    if not daily_frames:
        return None

    out = pd.concat(daily_frames).sort_index()
    # Drop any duplicates (year boundaries can overlap by one bar)
    out = out[~out.index.duplicated(keep="last")]
    logger.info(
        f"  {pair_code}: aggregated to {len(out)} daily bars "
        f"({out.index.min().date()} → {out.index.max().date()})"
    )
    return out


def aggregate_universe(universe_dict, cache_dir, start_year=2010,
                      end_year=None, output_path=None):
    """
    Aggregate all pairs in universe to daily. Returns a wide DataFrame
    with index=date and a MultiIndex column (pair, field) where field in
    {open, high, low, close, num_bars}.

    Also returns coverage_report: per pair, count of daily bars and date range.
    """
    if end_year is None:
        end_year = pd.Timestamp.today().year

    all_pairs = {}
    coverage = []
    for pair_code in universe_dict:
        logger.info(f"Aggregating {pair_code}...")
        daily = aggregate_pair(pair_code, cache_dir, start_year, end_year)
        if daily is None or len(daily) == 0:
            coverage.append({
                "pair": pair_code, "n_bars": 0,
                "first_date": None, "last_date": None,
            })
            continue
        all_pairs[pair_code] = daily
        coverage.append({
            "pair":       pair_code,
            "n_bars":     len(daily),
            "first_date": daily.index.min().date(),
            "last_date":  daily.index.max().date(),
        })

    if not all_pairs:
        raise RuntimeError("No pairs produced any daily data.")

    # Wide DataFrame: index=date, columns=(pair, field)
    wide = pd.concat(all_pairs, axis=1)
    wide.columns.names = ["pair", "field"]
    wide = wide.sort_index()

    coverage_df = pd.DataFrame(coverage).set_index("pair")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(output_path)
        logger.info(f"Wrote daily OHLC to {output_path}")

    return wide, coverage_df
