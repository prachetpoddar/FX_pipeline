import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "outputs"

OVERALL_SHIFT_THRESHOLD = 0.5    # points on 0–10 WB scale
SUBCOMPONENT_THRESHOLD  = 1.0    # points on 0–10 WB scale
PRE_POST_WINDOW_YEARS   = 5

SUBCOMPONENTS = [
    "size_of_government",
    "legal_system_property_rights",
    "sound_money",
    "freedom_to_trade",
    "regulation",
]

PATTERN_TAG_MAP = {
    "size_of_government":           "fiscal_consolidation",
    "legal_system_property_rights": "property_rights_reform",
    "sound_money":                  "monetary_stabilisation",
    "freedom_to_trade":             "trade_liberalisation",
    "regulation":                   "deregulation",
}


def detect_events(ief_df,
                  overall_threshold=OVERALL_SHIFT_THRESHOLD,
                  sub_threshold=SUBCOMPONENT_THRESHOLD):
    """
    Identify country-years where a significant governance/freedom
    shift occurred. Returns DataFrame of events.
    """
    events = []
    for country in ief_df["country"].unique():
        cdf = ief_df[ief_df["country"] == country].sort_values("year").copy()
        if len(cdf) < 3:
            continue

        cdf["overall_change"] = cdf["overall_score"].diff()
        iso3 = cdf["iso3"].iloc[0] if "iso3" in cdf.columns else None

        for sc in SUBCOMPONENTS:
            if sc in cdf.columns:
                cdf[f"{sc}_change"] = cdf[sc].diff()

        for _, row in cdf.iterrows():
            if pd.isna(row.get("overall_change")):
                continue

            overall_chg = row["overall_change"]
            year        = int(row["year"])

            sub_changes = {}
            for sc in SUBCOMPONENTS:
                chg_col = f"{sc}_change"
                if chg_col in row.index and not pd.isna(row[chg_col]):
                    sub_changes[sc] = row[chg_col]

            max_sub = max(abs(v) for v in sub_changes.values()) if sub_changes else 0
            if abs(overall_chg) < overall_threshold and max_sub < sub_threshold:
                continue

            primary_drivers = sorted(
                sub_changes.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]

            pattern_tags = list(set(
                PATTERN_TAG_MAP.get(sc, "other")
                for sc, v in primary_drivers if abs(v) >= 0.3
            ))

            events.append({
                "country":        country,
                "iso3":           iso3,
                "event_year":     year,
                "direction":      "improvement" if overall_chg >= 0 else "deterioration",
                "overall_score":  row["overall_score"],
                "overall_change": round(overall_chg, 4),
                "primary_drivers": "; ".join(
                    f"{sc}({v:+.2f})" for sc, v in primary_drivers
                ),
                "pattern_tags":   "; ".join(pattern_tags),
                "max_sub_change": round(max_sub, 4),
            })

    df = pd.DataFrame(events)
    logger.info(
        f"Detected {len(df)} events across "
        f"{df['country'].nunique() if len(df) else 0} countries"
    )
    return df


def extract_windows(ief_df, events_df, window=PRE_POST_WINDOW_YEARS):
    """Extract 5-year pre/post IEF trajectory for each event."""
    windows = []
    for _, event in events_df.iterrows():
        country    = event["country"]
        event_year = event["event_year"]

        cdf = ief_df[ief_df["country"] == country].sort_values("year")
        subset = cdf[
            (cdf["year"] >= event_year - window) &
            (cdf["year"] <= event_year + window)
        ].copy()

        if len(subset) < 3:
            continue

        subset["year_offset"]             = subset["year"] - event_year
        subset["event_year"]              = event_year
        subset["direction"]               = event["direction"]
        subset["pattern_tags"]            = event["pattern_tags"]
        subset["overall_change_at_event"] = event["overall_change"]
        windows.append(subset)

    if not windows:
        return pd.DataFrame()

    result = pd.concat(windows, ignore_index=True)
    logger.info(f"Extracted {len(result)} window observations")
    return result


def compute_leading_indicators(ief_df, events_df,
                                window=PRE_POST_WINDOW_YEARS):
    """
    For each improvement event, identify which sub-components
    started moving before the overall score shifted.
    Returns a summary DataFrame.
    """
    improvement_events = events_df[events_df["direction"] == "improvement"]
    if improvement_events.empty:
        return pd.DataFrame()

    lead_records = []
    for _, event in improvement_events.iterrows():
        country    = event["country"]
        event_year = event["event_year"]

        cdf = ief_df[ief_df["country"] == country].sort_values("year")
        pre = cdf[
            (cdf["year"] >= event_year - window) &
            (cdf["year"] < event_year)
        ].copy()

        if len(pre) < 2:
            continue

        for sc in SUBCOMPONENTS:
            if sc not in pre.columns:
                continue
            vals = pre[sc].dropna()
            if len(vals) < 2:
                continue
            recent = vals.iloc[-2:]
            if recent.iloc[1] > recent.iloc[0]:
                lead_records.append({
                    "country":      country,
                    "event_year":   event_year,
                    "subcomponent": sc,
                    "lead_years":   1,
                    "pattern_tag":  PATTERN_TAG_MAP.get(sc, "other"),
                })

    if not lead_records:
        return pd.DataFrame()

    leads = pd.DataFrame(lead_records)
    summary = leads.groupby("subcomponent").agg(
        avg_lead_years  = ("lead_years", "mean"),
        event_count     = ("event_year", "count"),
        reliability_pct = ("event_year",
                           lambda x: len(x) / len(improvement_events) * 100),
    ).round(2).sort_values("reliability_pct", ascending=False)

    logger.info(
        f"Leading indicator analysis: {len(summary)} sub-components evaluated"
    )
    return summary


def screen_current_watchlist(ief_df, leading_indicators_df,
                              screen_year=None, top_n_signals=2):
    """
    Screen for countries currently showing early warning signals
    of an impending improvement event.
    """
    if screen_year is None:
        screen_year = int(ief_df["year"].max())

    logger.info(f"Screening for early warning signals in {screen_year}...")

    if leading_indicators_df.empty:
        logger.warning("No leading indicators — skipping watchlist")
        return pd.DataFrame()

    high_rel = leading_indicators_df[
        leading_indicators_df["reliability_pct"] > 40
    ].index.tolist()

    if not high_rel:
        high_rel = leading_indicators_df.head(3).index.tolist()

    watchlist = []
    for country in ief_df["country"].unique():
        cdf = ief_df[ief_df["country"] == country].sort_values("year")
        recent = cdf[cdf["year"] >= screen_year - 2]
        if len(recent) < 2:
            continue

        signals_firing = []
        for sc in high_rel:
            if sc not in recent.columns:
                continue
            vals = recent[sc].dropna()
            if len(vals) >= 2 and vals.iloc[-1] > vals.iloc[-2]:
                change = vals.iloc[-1] - vals.iloc[-2]
                signals_firing.append(f"{sc}(+{change:.2f})")

        if len(signals_firing) >= top_n_signals:
            iso3 = cdf["iso3"].iloc[0] if "iso3" in cdf.columns else None
            cur  = cdf[cdf["year"] == screen_year]["overall_score"]
            watchlist.append({
                "country":             country,
                "iso3":                iso3,
                "screen_year":         screen_year,
                "signals_firing":      len(signals_firing),
                "signal_details":      "; ".join(signals_firing),
                "current_score":       float(cur.values[0]) if len(cur) else None,
                "projected_direction": "improvement",
            })

    result = pd.DataFrame(watchlist)
    if not result.empty:
        result = result.sort_values("signals_firing", ascending=False)
    logger.info(f"Watchlist: {len(result)} countries flagged")
    return result
