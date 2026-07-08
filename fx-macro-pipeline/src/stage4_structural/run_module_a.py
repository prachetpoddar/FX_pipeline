import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ief_client import fetch_ief_historical_bulk
from ief_event_detector import (
    detect_events,
    extract_windows,
    compute_leading_indicators,
    screen_current_watchlist,
)
from event_catalogue import build_event_catalogue, save_catalogue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Stage 4 — Module A: IEF Event Detector ===")

    # 1. Fetch historical IEF data
    logger.info("Step 1: Fetching IEF historical data...")
    ief_df = fetch_ief_historical_bulk()
    logger.info(
        f"IEF data: {ief_df['country'].nunique()} countries, "
        f"years {int(ief_df['year'].min())}–{int(ief_df['year'].max())}"
    )

    # 2. Detect significant events
    logger.info("\nStep 2: Detecting significant IEF shift events...")
    events_df = detect_events(ief_df)
    logger.info(
        f"Events found: {len(events_df)} total  |  "
        f"improvements: {(events_df['direction']=='improvement').sum()}  |  "
        f"deteriorations: {(events_df['direction']=='deterioration').sum()}"
    )

    # 3. Extract 5-year pre/post windows
    logger.info("\nStep 3: Extracting 5-year event windows...")
    windows_df = extract_windows(ief_df, events_df)

    # 4. Compute leading indicators
    logger.info("\nStep 4: Computing leading indicator analysis...")
    leading_indicators = compute_leading_indicators(ief_df, events_df)
    if not leading_indicators.empty:
        logger.info("\nTop leading indicators for improvement events:")
        logger.info(f"\n{leading_indicators.head(6).to_string()}")

    # 5. Screen current watchlist
    logger.info("\nStep 5: Screening for current early warning signals...")
    watchlist_df = screen_current_watchlist(ief_df, leading_indicators)
    if not watchlist_df.empty:
        logger.info(f"\nWatchlist countries ({len(watchlist_df)}):")
        logger.info(f"\n{watchlist_df[['country','signals_firing','signal_details','current_score']].head(10).to_string()}")

    # 6. Build event catalogue
    logger.info("\nStep 6: Building event catalogue...")
    catalogue_df = build_event_catalogue(events_df, windows_df)

    # 7. Save all outputs
    logger.info("\nStep 7: Saving outputs...")
    save_catalogue(catalogue_df, watchlist_df, leading_indicators)

    logger.info("\n=== Module A complete ===")
    logger.info(f"Events catalogued:    {len(catalogue_df)}")
    logger.info(f"Watchlist countries:  {len(watchlist_df)}")
    logger.info(
        f"Leading indicators:   {len(leading_indicators)} sub-components analysed"
    )

    return catalogue_df, watchlist_df, leading_indicators


if __name__ == "__main__":
    main()
