import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
EFW_FILE      = RAW_DIR / "efw_data.xlsx"
SHEET_NAME    = "EFW Index 1970-2023"
HEADER_ROW    = 3   # 0-indexed row 3 = Excel row 4

# Exact column indices from the Fraser Institute EFW master index
COL_YEAR      = 0
COL_ISO       = 1
COL_COUNTRY   = 2
COL_OVERALL   = 3
COL_AREA1     = 21   # Size of Government
COL_AREA2     = 32   # Legal System & Property Rights (with gender adjustment)
COL_AREA3     = 42   # Sound Money
COL_AREA4     = 60   # Freedom to Trade Internationally
COL_AREA5     = 83   # Regulation

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


def fetch_ief_historical_bulk(force_download=False):
    """
    Load the Fraser Institute EFW dataset from the local Excel file.
    Returns a long-format DataFrame with columns:
      year, iso3, country, overall_score,
      size_of_government, legal_system_property_rights,
      sound_money, freedom_to_trade, regulation
    """
    cache_path = RAW_DIR / "efw_historical.parquet"
    if cache_path.exists() and not force_download:
        logger.info(f"Loading cached EFW data from {cache_path}")
        return pd.read_parquet(cache_path)

    if not EFW_FILE.exists():
        raise FileNotFoundError(
            f"EFW data file not found at {EFW_FILE}\n"
            f"Please download from fraserinstitute.org/economic-freedom/dataset "
            f"and save as data/raw/efw_data.xlsx"
        )

    logger.info(f"Reading EFW data from {EFW_FILE}...")
    raw = pd.read_excel(
        EFW_FILE,
        sheet_name=SHEET_NAME,
        header=HEADER_ROW,
        engine="openpyxl",
    )

    # Select and rename only the columns we need
    cols = {
        raw.columns[COL_YEAR]:    "year",
        raw.columns[COL_ISO]:     "iso3",
        raw.columns[COL_COUNTRY]: "country",
        raw.columns[COL_OVERALL]: "overall_score",
        raw.columns[COL_AREA1]:   "size_of_government",
        raw.columns[COL_AREA2]:   "legal_system_property_rights",
        raw.columns[COL_AREA3]:   "sound_money",
        raw.columns[COL_AREA4]:   "freedom_to_trade",
        raw.columns[COL_AREA5]:   "regulation",
    }
    df = raw[list(cols.keys())].rename(columns=cols).copy()

    # Clean up
    df = df[df["year"].notna() & df["country"].notna()].copy()
    df["year"]          = df["year"].astype(int)
    df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
    for sc in SUBCOMPONENTS:
        df[sc] = pd.to_numeric(df[sc], errors="coerce")

    df = df[df["overall_score"].notna()]
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    # Cache it
    df.to_parquet(cache_path)

    logger.info(
        f"EFW data loaded: {len(df)} rows, "
        f"{df['country'].nunique()} countries, "
        f"years {df['year'].min()}–{df['year'].max()}"
    )
    return df
