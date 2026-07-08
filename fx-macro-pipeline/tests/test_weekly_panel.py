"""
Tests for the shared weekly resampler (src/stage1_collection_v2/weekly_panel.py).

Every property carries a POSITIVE CONTROL: a deliberately broken input that
should fail the same property. Without that, a "passing" test could just be
asserting against a fixed bug.

Properties covered (per brief):

  1. Fri-week boundary: a synthetic Mon..Fri week with a known Friday lands in
     the correct period; a Sat/Sun stub does NOT create a phantom week.
  2. Drop rule: 2 active days dropped; 3 active days kept; all-thin (<floor)
     week dropped even if all 5 calendar days present.
  3. Active-since: EM pair has status not_listed before its first listed week
     and real values after; control — a major has no not_listed gap.
  4. Non-overlap & count: total Fri-weeks == 835 (exact); majors get 835
     contributing weeks (exact); EM cohort lands at 789 due to a single mid-
     sample data outage (week ending 2017-02-24, all EM pairs zero data —
     audit finding the brief did not anticipate; 789 != the brief's nominal
     790 by exactly that gap).
  5. Orientation parity: equal-weight oriented WEEKLY mean strongly tracks
     the negative of the USD index built from USDXXX pairs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.universe import UNIVERSE  # noqa: E402
from stage1_collection_v2.weekly_panel import (  # noqa: E402
    build_weekly_panel,
    load_raw_inputs,
    week_label,
    MIN_ACTIVE_DAYS_PER_WEEK,
    STATUS_KEPT,
    STATUS_DROPPED_STALE,
    STATUS_NOT_LISTED,
)


# ===========================================================================
# Tiny synthetic builders — used for property tests so we can isolate the
# rule under test without dragging in 16 years of real data.
# ===========================================================================

def _make_synth_ohlc(daily_index, pairs, num_bars_value=1000.0):
    """Build a minimal OHLC frame with only the num_bars field populated,
    matching the production schema. close/open/high/low are all 1.0."""
    cols = pd.MultiIndex.from_product([pairs, ["open", "high", "low", "close",
                                               "num_bars"]],
                                      names=["pair", "field"])
    df = pd.DataFrame(1.0, index=daily_index, columns=cols)
    for p in pairs:
        df[(p, "num_bars")] = num_bars_value
    return df


def _make_synth_returns(daily_index, pairs, value=0.0):
    return pd.DataFrame(value, index=daily_index, columns=pairs)


# ===========================================================================
# 1. Fri-week boundary
# ===========================================================================

def test_fri_boundary_known_friday_in_correct_week():
    """A Mon..Fri week with a known Friday must produce exactly one
    Fri-anchored week label = that Friday."""
    days = pd.DatetimeIndex(pd.bdate_range("2020-06-01", "2020-06-05"))
    lbl = week_label(days)
    assert set(lbl.values) == {pd.Timestamp("2020-06-05")}, (
        f"Mon-Fri week did not all collapse to its Friday label: {lbl.unique()}"
    )


def test_fri_boundary_no_phantom_week_from_weekend():
    """POSITIVE CONTROL: if a Sat/Sun "stub" got accidentally inserted into
    the daily index, week_label would still snap it to the SAME Friday — it
    must not invent a new phantom week labelled Sat or Sun."""
    days = pd.DatetimeIndex(["2020-06-01", "2020-06-05", "2020-06-06",
                             "2020-06-07"])  # +Sat, +Sun (impossible in prod)
    lbl = week_label(days)
    # All four must snap to a Friday label — never to a Sat/Sun.
    for v in lbl.unique():
        assert v.weekday() == 4, (
            f"week_label produced a non-Friday anchor {v} (weekday={v.weekday()})"
        )
    # And no label should be Sat (5) or Sun (6).
    weekdays_seen = {pd.Timestamp(v).weekday() for v in lbl.unique()}
    assert weekdays_seen <= {4}, weekdays_seen


# ===========================================================================
# 2. Drop rule
# ===========================================================================

def _drop_rule_setup(active_days_per_week, all_thin=False):
    """Build a many-week 1-pair frame: 100 days of all-active history (so
    the per-pair median num_bars is set robustly), then ONE test week with
    `active_days_per_week` days at full bar count and the rest at 1 bar.

    The history is needed because the FLOOR = 0.5 * median(num_bars) is
    computed across the entire daily series; a 5-day synthetic would have
    a degenerate median pinned by the test week itself, defeating the
    point of the rule. We isolate the test week as the LAST Fri-week of
    the frame and only assert on that week."""
    pair = "EURUSD"
    history = pd.DatetimeIndex(pd.bdate_range("2019-12-30", "2020-05-22"))
    test_week_days = pd.DatetimeIndex(pd.bdate_range("2020-06-01", "2020-06-05"))
    days = history.append(test_week_days)
    ohlc = _make_synth_ohlc(days, [pair], num_bars_value=1500.0)
    lr = _make_synth_returns(days, [pair], value=0.001)
    if all_thin:
        # The test week's num_bars all deep below floor (1 bar each)
        for d in test_week_days:
            ohlc.loc[d, (pair, "num_bars")] = 1.0
    else:
        # Set the first (5 - active_days_per_week) days of THE TEST WEEK
        # to "thin" (1 bar)
        n_thin = 5 - active_days_per_week
        for d in test_week_days[:n_thin]:
            ohlc.loc[d, (pair, "num_bars")] = 1.0
    panel = build_weekly_panel(ohlc, lr,
                               min_active=MIN_ACTIVE_DAYS_PER_WEEK)
    # The test week label is the Friday 2020-06-05
    return panel, pair, pd.Timestamp("2020-06-05")


def test_drop_rule_two_active_dropped():
    panel, pair, test_week = _drop_rule_setup(active_days_per_week=2)
    status = panel["status"][pair].loc[test_week]
    val = panel["weekly_returns"][pair].loc[test_week]
    assert status == STATUS_DROPPED_STALE, (
        f"week with 2 active days should be DROPPED_STALE; got {status}"
    )
    assert pd.isna(val), f"dropped week should have NaN weekly return; got {val}"


def test_drop_rule_three_active_kept():
    panel, pair, test_week = _drop_rule_setup(active_days_per_week=3)
    status = panel["status"][pair].loc[test_week]
    val = panel["weekly_returns"][pair].loc[test_week]
    assert status == STATUS_KEPT, (
        f"week with 3 active days should be KEPT; got {status}"
    )
    assert pd.notna(val), "kept week should have finite weekly return"


def test_drop_rule_all_thin_dropped_even_with_5_cal_days():
    """POSITIVE CONTROL: a week with 5 calendar days but every day below the
    pair's num_bars floor must STILL be dropped — calendar-day count is
    irrelevant when no day is actually active."""
    panel, pair, test_week = _drop_rule_setup(active_days_per_week=5,
                                              all_thin=True)
    status = panel["status"][pair].loc[test_week]
    diag_row = panel["diag"][(panel["diag"].pair == pair)
                             & (panel["diag"].week_end == test_week)].iloc[0]
    assert diag_row["n_calendar_days_present"] == 5
    assert diag_row["n_active_days"] == 0
    assert status == STATUS_DROPPED_STALE, (
        f"all-thin week with 5 calendar days must still be dropped; got {status}"
    )


# ===========================================================================
# 3. Active-since
# ===========================================================================

def test_active_since_em_pair_marked_not_listed_then_kept():
    """USDMXN must be status=not_listed for every week ending before its
    first listed week, and then kept (or dropped_stale, never not_listed)
    after."""
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    status = panel["status"]["USDMXN"]
    not_listed_weeks = status[status == STATUS_NOT_LISTED].index
    other_weeks = status[status != STATUS_NOT_LISTED].index
    # Pre-listing weeks all precede first kept/dropped week
    assert not_listed_weeks.max() < other_weeks.min(), (
        "not_listed cells must all precede the first listed cell; "
        f"max not_listed = {not_listed_weeks.max()}, "
        f"min listed = {other_weeks.min()}"
    )
    # First kept week is on/after Friday 2010-11-19 (week containing
    # the first-data Monday 2010-11-15)
    first_kept = status[status == STATUS_KEPT].index.min()
    assert first_kept == pd.Timestamp("2010-11-19"), (
        f"first kept EM week should be 2010-11-19; got {first_kept}"
    )


def test_active_since_major_has_no_not_listed_gap():
    """POSITIVE CONTROL: a major (EURUSD) was listed at sample start, so it
    must have ZERO not_listed weeks. If this test ever fires, either
    EURUSD's listing was wrongly delayed OR the not_listed logic is
    mis-anchored — either way, this catches a serious bug."""
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    status = panel["status"]["EURUSD"]
    n_not_listed = (status == STATUS_NOT_LISTED).sum()
    assert n_not_listed == 0, (
        f"EURUSD (major) should have 0 not_listed weeks; got {n_not_listed}"
    )


# ===========================================================================
# 4. Non-overlap & exact counts
# ===========================================================================

def test_total_fri_weeks_exactly_835():
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    assert len(panel["weekly_returns"]) == 835, (
        f"total Fri-weeks should be exactly 835; got {len(panel['weekly_returns'])}"
    )


def test_majors_contributing_weeks_exactly_835():
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD",
              "USDCAD", "USDSEK", "USDNOK"]
    for m in majors:
        kept = panel["weekly_returns"][m].notna().sum()
        assert kept == 835, (
            f"{m} (major) should contribute exactly 835 weeks; got {kept}"
        )


def test_em_cohort_contributing_weeks_exactly_789():
    """EM cohort = MXN, ZAR, PLN, HUF, CZK. Brief nominally expected 790
    (835 - 45 not_listed). The realized 16-year sample has exactly ONE
    fully-empty week (ending 2017-02-24, EM data outage) so the contributing
    count is 789. We lock the test at 789 and document the gap — counting
    790 would silently absorb the outage."""
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    em = ["USDMXN", "USDZAR", "USDPLN", "USDHUF", "USDCZK"]
    for p in em:
        kept = panel["weekly_returns"][p].notna().sum()
        assert kept == 789, (
            f"{p} should contribute exactly 789 weeks (835 - 45 not_listed - "
            f"1 mid-sample outage week ending 2017-02-24); got {kept}"
        )
    # Direct check that 2017-02-24 is exactly the dropped_stale week for EM.
    for p in em:
        dropped = panel["status"][p][panel["status"][p] == STATUS_DROPPED_STALE]
        assert list(dropped.index) == [pd.Timestamp("2017-02-24")], (
            f"{p}: dropped_stale weeks should be exactly [2017-02-24]; got "
            f"{list(dropped.index)}"
        )


def test_no_overlap_between_status_categories():
    """A (week, pair) cell must be exactly one of {kept, dropped_stale,
    not_listed}; never more than one, never none."""
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    status = panel["status"]
    counts_total = (status == STATUS_KEPT).sum().sum() \
        + (status == STATUS_DROPPED_STALE).sum().sum() \
        + (status == STATUS_NOT_LISTED).sum().sum()
    assert counts_total == status.size, (
        f"status partition incomplete: {counts_total} != {status.size}"
    )


# ===========================================================================
# 5. Orientation parity (weekly)
# ===========================================================================

def test_weekly_oriented_mean_tracks_negative_usd_index():
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    wr = panel["weekly_returns"]
    # USD index from USDXXX pairs in the LIVE universe
    usd_pairs = [p for p in panel["universe"]
                 if UNIVERSE[p]["base"] == "USD"]
    # Aggregate the SAME log returns to weekly (raw, unoriented) for the
    # comparison index. We do this by re-grouping raw returns by week label.
    lbl = week_label(lr.index)
    raw_weekly = lr.groupby(lbl.values).sum(min_count=1).reindex(wr.index)
    usd_idx = raw_weekly[usd_pairs].mean(axis=1)
    # equal-weight oriented mean using only currently-kept cells per week
    mean_oriented = wr.mean(axis=1)
    com = mean_oriented.dropna().index.intersection(usd_idx.dropna().index)
    corr = mean_oriented.loc[com].corr(usd_idx.loc[com])
    assert corr < -0.9, (
        f"Weekly oriented mean should strongly negatively correlate with "
        f"the weekly USD index; got corr={corr:+.3f}"
    )


if __name__ == "__main__":
    test_fri_boundary_known_friday_in_correct_week()
    print("OK fri_boundary_known_friday_in_correct_week")
    test_fri_boundary_no_phantom_week_from_weekend()
    print("OK fri_boundary_no_phantom_week_from_weekend")
    test_drop_rule_two_active_dropped()
    print("OK drop_rule_two_active_dropped")
    test_drop_rule_three_active_kept()
    print("OK drop_rule_three_active_kept")
    test_drop_rule_all_thin_dropped_even_with_5_cal_days()
    print("OK drop_rule_all_thin_dropped_even_with_5_cal_days")
    test_active_since_em_pair_marked_not_listed_then_kept()
    print("OK active_since_em_pair_marked_not_listed_then_kept")
    test_active_since_major_has_no_not_listed_gap()
    print("OK active_since_major_has_no_not_listed_gap")
    test_total_fri_weeks_exactly_835()
    print("OK total_fri_weeks_exactly_835")
    test_majors_contributing_weeks_exactly_835()
    print("OK majors_contributing_weeks_exactly_835")
    test_em_cohort_contributing_weeks_exactly_789()
    print("OK em_cohort_contributing_weeks_exactly_789")
    test_no_overlap_between_status_categories()
    print("OK no_overlap_between_status_categories")
    test_weekly_oriented_mean_tracks_negative_usd_index()
    print("OK weekly_oriented_mean_tracks_negative_usd_index")
