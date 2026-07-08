import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"


def build_event_catalogue(events_df, windows_df, market_returns=None):
    """
    Build the pattern library: for each event, store the full
    context including IEF trajectory, market response, and pattern tags.

    market_returns: optional DataFrame of equity index returns
                    indexed by (iso3, year) if available.
    Returns the enriched catalogue as a DataFrame.
    """
    if events_df.empty:
        logger.warning("No events to catalogue")
        return pd.DataFrame()

    catalogue = events_df.copy()

    # Attach pre/post IEF score summaries from windows
    if not windows_df.empty:
        for event_idx, event in catalogue.iterrows():
            country    = event["country"]
            event_year = event["event_year"]

            w = windows_df[
                (windows_df["country"] == country) &
                (windows_df["event_year"] == event_year)
            ]

            pre  = w[w["year_offset"] < 0]["overall_score"]
            post = w[w["year_offset"] > 0]["overall_score"]

            catalogue.loc[event_idx, "pre_window_avg"]    = round(pre.mean(), 2)  if len(pre)  else None
            catalogue.loc[event_idx, "post_window_avg"]   = round(post.mean(), 2) if len(post) else None
            catalogue.loc[event_idx, "pre_window_trend"]  = round(
                float(np.polyfit(range(len(pre)),  pre.values,  1)[0]), 3
            ) if len(pre) >= 2 else None
            catalogue.loc[event_idx, "post_window_trend"] = round(
                float(np.polyfit(range(len(post)), post.values, 1)[0]), 3
            ) if len(post) >= 2 else None

    logger.info(f"Event catalogue built: {len(catalogue)} entries")
    return catalogue


def find_historical_analogue(country, current_signals, catalogue_df,
                              leading_indicators_df):
    """
    Given a country's current firing signals, find the closest historical
    analogue in the event catalogue.

    Similarity is measured by overlap in pattern_tags.
    Returns the top 3 most similar historical events.
    """
    if catalogue_df.empty:
        return pd.DataFrame()

    # Extract pattern tags from current signals
    current_tags = set()
    for sig in current_signals:
        sc = sig.split("(")[0].strip()
        from ief_event_detector import PATTERN_TAG_MAP
        tag = PATTERN_TAG_MAP.get(sc)
        if tag:
            current_tags.add(tag)

    if not current_tags:
        return catalogue_df.head(3)

    # Score each historical event by tag overlap
    scores = []
    for _, event in catalogue_df[catalogue_df["direction"] == "improvement"].iterrows():
        event_tags = set(str(event.get("pattern_tags", "")).split("; "))
        overlap    = len(current_tags & event_tags)
        scores.append((overlap, event))

    scores.sort(key=lambda x: x[0], reverse=True)
    top3 = [e for _, e in scores[:3]]

    if not top3:
        return pd.DataFrame()

    result = pd.DataFrame(top3)
    result["tag_overlap_score"] = [s for s, _ in scores[:3]]
    return result[["country", "event_year", "overall_change",
                   "pattern_tags", "tag_overlap_score",
                   "post_window_avg", "post_window_trend"]].reset_index(drop=True)


def save_catalogue(catalogue_df, watchlist_df, leading_indicators_df):
    """Save all Stage 4 Module A outputs."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not catalogue_df.empty:
        catalogue_df.to_csv(OUTPUTS_DIR / "ief_event_catalogue.csv", index=False)
        logger.info(f"Saved ief_event_catalogue.csv ({len(catalogue_df)} events)")

    if not watchlist_df.empty:
        watchlist_df.to_csv(OUTPUTS_DIR / "ief_watchlist.csv", index=False)
        logger.info(f"Saved ief_watchlist.csv ({len(watchlist_df)} countries)")

    if not leading_indicators_df.empty:
        leading_indicators_df.to_csv(OUTPUTS_DIR / "ief_leading_indicators.csv")
        logger.info(f"Saved ief_leading_indicators.csv")
