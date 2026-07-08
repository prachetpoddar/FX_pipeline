"""
run_stage1_v2.py
================

Orchestrates the full Stage 1 v2 pipeline:
    1. Fetch 1-minute HistData bars for the tradable universe
    2. Aggregate to daily OHLC with 17:00 NY close convention
    3. Compute log returns + validate
    4. Write fx_log_returns.parquet (only pairs that pass validation)

Run order:
    conda activate fxpipeline
    pip install histdatacom    # one-time
    python run_stage1_v2.py

Output: data/processed_v2/fx_log_returns.parquet (and supporting files)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from universe          import UNIVERSE
from histdata_client   import HistDataClient
from daily_aggregator  import aggregate_universe
from preprocessor_v2   import run_preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# Paths — adjust for your machine if needed
PROJECT_ROOT     = Path(__file__).resolve().parents[2]
RAW_HISTDATA_DIR = PROJECT_ROOT / "data" / "raw" / "histdata_1min"
DAILY_PANEL_PATH = PROJECT_ROOT / "data" / "processed_v2" / "fx_daily_ohlc.parquet"
OUTPUT_DIR       = PROJECT_ROOT / "data" / "processed_v2"

START_YEAR = 2010
END_YEAR   = None    # None = current year


def main():
    logger.info("=== Stage 1 v2 ===")
    logger.info(f"Universe: {len(UNIVERSE)} pairs")
    logger.info(f"Years: {START_YEAR} → {END_YEAR or 'current'}")

    # ── Step 1: download ────────────────────────────────────────────────────
    logger.info("\n--- Step 1: HistData download ---")
    client = HistDataClient(cache_dir=RAW_HISTDATA_DIR)
    download_report = client.download_universe(
        UNIVERSE, start_year=START_YEAR, end_year=END_YEAR
    )
    download_report.to_csv(OUTPUT_DIR / "download_report.csv")
    logger.info(f"\nDownload report:\n{download_report}")
    # Identify pairs with at least some data
    ok_mask = download_report.isin(["ok", "cached"]).any(axis=1)
    available_pairs = download_report.index[ok_mask].tolist()
    logger.info(f"\nPairs with at least some data: {available_pairs}")
    if not available_pairs:
        raise RuntimeError("No pairs returned any data. Check histdatacom install.")

    # ── Step 2: aggregate to daily ──────────────────────────────────────────
    logger.info("\n--- Step 2: aggregate 1-min → daily OHLC ---")
    universe_avail = {p: UNIVERSE[p] for p in available_pairs}
    daily_panel, coverage_report = aggregate_universe(
        universe_avail,
        cache_dir=RAW_HISTDATA_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR or 2099,
        output_path=DAILY_PANEL_PATH,
    )
    coverage_report.to_csv(OUTPUT_DIR / "coverage_report.csv")
    logger.info(f"\nCoverage:\n{coverage_report}")

    # ── Step 3: preprocess + validate ───────────────────────────────────────
    logger.info("\n--- Step 3: log returns + validation ---")
    returns, validation = run_preprocessing(
        DAILY_PANEL_PATH, UNIVERSE, OUTPUT_DIR
    )

    logger.info(f"\n=== Stage 1 v2 complete ===")
    logger.info(f"Returns parquet: {OUTPUT_DIR / 'fx_log_returns.parquet'}")
    logger.info(f"Validation report: {OUTPUT_DIR / 'validation_report.csv'}")
    logger.info(f"Shape: {returns.shape}")


if __name__ == "__main__":
    main()
