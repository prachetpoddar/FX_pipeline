import requests
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from io import StringIO

logger = logging.getLogger(__name__)

RAW_DIR     = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── SECTOR PAIR KNOWLEDGE BASE ────────────────────────────────────────
# Maps primary sector → list of inverse sectors with:
#   - name: human-readable inverse sector name
#   - mechanism: why it moves inverse
#   - elasticity: how strongly it responds (1=direct substitute, 0.5=partial)
#   - lag_months: typical lag before inverse sector responds
#   - wb_indicator: World Bank indicator for inverse sector if available

SECTOR_PAIRS = {
    "oil": {
        "description": "Crude oil and petroleum products",
        "wb_commodity": "CRUDE_OIL",
        "inverses": [
            {"name": "biofuels",         "mechanism": "direct energy substitute",          "elasticity": 0.85, "lag_months": 6,  "ticker_proxy": "BIOFUEL_INDEX"},
            {"name": "solar_energy",     "mechanism": "long-run energy substitute",        "elasticity": 0.70, "lag_months": 12, "ticker_proxy": "SOLAR_INDEX"},
            {"name": "wind_energy",      "mechanism": "long-run energy substitute",        "elasticity": 0.65, "lag_months": 12, "ticker_proxy": "WIND_INDEX"},
            {"name": "electric_vehicles","mechanism": "transport demand destruction",      "elasticity": 0.60, "lag_months": 9,  "ticker_proxy": "EV_INDEX"},
            {"name": "natural_gas",      "mechanism": "direct energy substitute short-run","elasticity": 0.75, "lag_months": 3,  "ticker_proxy": "NATGAS"},
        ]
    },
    "coal": {
        "description": "Thermal and coking coal",
        "wb_commodity": "COAL_AUS",
        "inverses": [
            {"name": "natural_gas",      "mechanism": "direct power generation substitute","elasticity": 0.80, "lag_months": 3,  "ticker_proxy": "NATGAS"},
            {"name": "nuclear_energy",   "mechanism": "baseload power substitute",         "elasticity": 0.55, "lag_months": 24, "ticker_proxy": "URANIUM"},
            {"name": "solar_energy",     "mechanism": "long-run power substitute",         "elasticity": 0.60, "lag_months": 12, "ticker_proxy": "SOLAR_INDEX"},
            {"name": "lng",              "mechanism": "clean energy policy substitute",    "elasticity": 0.70, "lag_months": 6,  "ticker_proxy": "LNG_INDEX"},
        ]
    },
    "natural_gas": {
        "description": "Natural gas and LNG",
        "wb_commodity": "NATGAS_EUR",
        "inverses": [
            {"name": "solar_energy",     "mechanism": "residential energy substitute",     "elasticity": 0.65, "lag_months": 12, "ticker_proxy": "SOLAR_INDEX"},
            {"name": "heat_pumps",       "mechanism": "heating substitute technology",     "elasticity": 0.70, "lag_months": 9,  "ticker_proxy": "HEATPUMP_INDEX"},
            {"name": "biomass",          "mechanism": "industrial heat substitute",        "elasticity": 0.55, "lag_months": 6,  "ticker_proxy": "BIOMASS_INDEX"},
        ]
    },
    "wheat": {
        "description": "Wheat and grain exports",
        "wb_commodity": "WHEAT_US",
        "inverses": [
            {"name": "alternative_grains","mechanism": "direct food substitute (rice, millet)","elasticity": 0.75, "lag_months": 3, "ticker_proxy": "RICE_INDEX"},
            {"name": "vertical_farming",  "mechanism": "supply-side substitute technology",    "elasticity": 0.45, "lag_months": 18,"ticker_proxy": "AGTECH_INDEX"},
            {"name": "insect_protein",    "mechanism": "alternative protein source",           "elasticity": 0.35, "lag_months": 12,"ticker_proxy": "ALTPRO_INDEX"},
            {"name": "cassava",           "mechanism": "staple food substitute in EM",         "elasticity": 0.60, "lag_months": 6, "ticker_proxy": "CASSAVA_PROXY"},
        ]
    },
    "corn": {
        "description": "Corn / maize exports",
        "wb_commodity": "MAIZE_US",
        "inverses": [
            {"name": "sorghum",          "mechanism": "direct feed and food substitute",   "elasticity": 0.70, "lag_months": 4,  "ticker_proxy": "SORGHUM_PROXY"},
            {"name": "alternative_grains","mechanism": "animal feed substitute",           "elasticity": 0.65, "lag_months": 3,  "ticker_proxy": "RICE_INDEX"},
            {"name": "lab_grown_meat",   "mechanism": "reduces feed demand long-run",      "elasticity": 0.30, "lag_months": 24, "ticker_proxy": "ALTPRO_INDEX"},
        ]
    },
    "soybeans": {
        "description": "Soybean and soy products",
        "wb_commodity": "SOYBEAN_US",
        "inverses": [
            {"name": "canola_rapeseed",  "mechanism": "direct oilseed substitute",        "elasticity": 0.75, "lag_months": 4,  "ticker_proxy": "CANOLA_PROXY"},
            {"name": "sunflower_oil",    "mechanism": "cooking oil substitute",           "elasticity": 0.70, "lag_months": 3,  "ticker_proxy": "SUNFLOWER_PROXY"},
            {"name": "pea_protein",      "mechanism": "plant protein substitute",         "elasticity": 0.55, "lag_months": 12, "ticker_proxy": "ALTPRO_INDEX"},
        ]
    },
    "copper": {
        "description": "Copper mining and exports",
        "wb_commodity": "COPPER_LME",
        "inverses": [
            {"name": "aluminium",        "mechanism": "construction and wiring substitute","elasticity": 0.65, "lag_months": 6,  "ticker_proxy": "ALUMINIUM_LME"},
            {"name": "fibre_optic",      "mechanism": "telecoms cable substitute",        "elasticity": 0.80, "lag_months": 9,  "ticker_proxy": "FIBEROPTIC_PROXY"},
            {"name": "wireless_tech",    "mechanism": "eliminates copper wiring demand",  "elasticity": 0.55, "lag_months": 18, "ticker_proxy": "WIRELESS_INDEX"},
        ]
    },
    "gold": {
        "description": "Gold mining and exports",
        "wb_commodity": "GOLD_LME",
        "inverses": [
            {"name": "risk_assets",      "mechanism": "gold/risk asset inverse correlation","elasticity": 0.85, "lag_months": 1,  "ticker_proxy": "EQUITY_INDEX"},
            {"name": "cryptocurrencies", "mechanism": "digital store of value substitute", "elasticity": 0.60, "lag_months": 3,  "ticker_proxy": "CRYPTO_INDEX"},
            {"name": "real_estate",      "mechanism": "alternative inflation hedge",       "elasticity": 0.50, "lag_months": 6,  "ticker_proxy": "REALESTATE_INDEX"},
        ]
    },
    "coffee": {
        "description": "Coffee bean exports",
        "wb_commodity": "COFFEE_OTHER",
        "inverses": [
            {"name": "tea",              "mechanism": "direct beverage substitute",        "elasticity": 0.75, "lag_months": 3,  "ticker_proxy": "TEA_PROXY"},
            {"name": "energy_drinks",    "mechanism": "caffeine delivery substitute",      "elasticity": 0.60, "lag_months": 6,  "ticker_proxy": "BEVERAGE_INDEX"},
            {"name": "chicory",          "mechanism": "coffee extender / substitute",      "elasticity": 0.55, "lag_months": 4,  "ticker_proxy": "CHICORY_PROXY"},
        ]
    },
    "tourism": {
        "description": "International tourism revenue",
        "wb_commodity": None,
        "inverses": [
            {"name": "domestic_leisure", "mechanism": "staycation substitution",           "elasticity": 0.70, "lag_months": 2,  "ticker_proxy": "DOMESTIC_LEISURE"},
            {"name": "virtual_reality",  "mechanism": "digital travel substitute long-run","elasticity": 0.35, "lag_months": 24, "ticker_proxy": "VR_INDEX"},
            {"name": "short_haul_rail",  "mechanism": "replaces short-haul flights",      "elasticity": 0.65, "lag_months": 6,  "ticker_proxy": "RAIL_INDEX"},
        ]
    },
    "manufacturing": {
        "description": "Heavy manufacturing exports",
        "wb_commodity": None,
        "inverses": [
            {"name": "3d_printing",      "mechanism": "distributed manufacturing substitute","elasticity": 0.45, "lag_months": 18, "ticker_proxy": "ADDMFG_INDEX"},
            {"name": "automation",       "mechanism": "labour cost arbitrage eroded",       "elasticity": 0.55, "lag_months": 12, "ticker_proxy": "ROBOT_INDEX"},
            {"name": "nearshoring",      "mechanism": "supply chain relocation",            "elasticity": 0.60, "lag_months": 12, "ticker_proxy": "LOGISTICS_INDEX"},
        ]
    },
    "palm_oil": {
        "description": "Palm oil production and export",
        "wb_commodity": "PALM_OIL",
        "inverses": [
            {"name": "sunflower_oil",    "mechanism": "cooking oil substitute",           "elasticity": 0.80, "lag_months": 3,  "ticker_proxy": "SUNFLOWER_PROXY"},
            {"name": "canola_rapeseed",  "mechanism": "edible oil substitute",            "elasticity": 0.75, "lag_months": 4,  "ticker_proxy": "CANOLA_PROXY"},
            {"name": "algae_oil",        "mechanism": "sustainable oil substitute",       "elasticity": 0.40, "lag_months": 18, "ticker_proxy": "ALGAE_PROXY"},
        ]
    },
    "rubber": {
        "description": "Natural rubber exports",
        "wb_commodity": "RUBBER_SGP",
        "inverses": [
            {"name": "synthetic_rubber", "mechanism": "petroleum-derived substitute",     "elasticity": 0.85, "lag_months": 3,  "ticker_proxy": "SYNTHRUBBER_PROXY"},
            {"name": "recycled_rubber",  "mechanism": "circular economy substitute",      "elasticity": 0.55, "lag_months": 9,  "ticker_proxy": "RECYCLING_INDEX"},
        ]
    },
    "cotton": {
        "description": "Cotton fibre exports",
        "wb_commodity": "COTTON_A_IDX",
        "inverses": [
            {"name": "polyester",        "mechanism": "synthetic fibre substitute",       "elasticity": 0.80, "lag_months": 3,  "ticker_proxy": "POLYESTER_PROXY"},
            {"name": "hemp_linen",       "mechanism": "sustainable fibre substitute",     "elasticity": 0.55, "lag_months": 6,  "ticker_proxy": "HEMP_PROXY"},
            {"name": "recycled_textiles","mechanism": "circular fashion substitute",      "elasticity": 0.45, "lag_months": 12, "ticker_proxy": "RECYCLTEX_PROXY"},
        ]
    },
}

# Country → primary sector mapping
# Based on World Bank commodity export dependency data
COUNTRY_PRIMARY_SECTORS = {
    "BRA": {"sector": "soybeans",      "export_share_pct": 14.0, "secondary": "oil"},
    "ARG": {"sector": "soybeans",      "export_share_pct": 18.0, "secondary": "corn"},
    "USA": {"sector": "corn",          "export_share_pct": 8.0,  "secondary": "soybeans"},
    "UKR": {"sector": "wheat",         "export_share_pct": 12.0, "secondary": "corn"},
    "AUS": {"sector": "wheat",         "export_share_pct": 6.0,  "secondary": "coal"},
    "IDN": {"sector": "palm_oil",      "export_share_pct": 11.0, "secondary": "coal"},
    "MYS": {"sector": "palm_oil",      "export_share_pct": 8.0,  "secondary": "oil"},
    "ETH": {"sector": "coffee",        "export_share_pct": 28.0, "secondary": "manufacturing"},
    "VNM": {"sector": "coffee",        "export_share_pct": 10.0, "secondary": "manufacturing"},
    "COL": {"sector": "coffee",        "export_share_pct": 7.0,  "secondary": "oil"},
    "SAU": {"sector": "oil",           "export_share_pct": 70.0, "secondary": "natural_gas"},
    "RUS": {"sector": "oil",           "export_share_pct": 45.0, "secondary": "natural_gas"},
    "NGA": {"sector": "oil",           "export_share_pct": 60.0, "secondary": "manufacturing"},
    "NOR": {"sector": "oil",           "export_share_pct": 40.0, "secondary": "natural_gas"},
    "ARE": {"sector": "oil",           "export_share_pct": 35.0, "secondary": "natural_gas"},
    "KWT": {"sector": "oil",           "export_share_pct": 80.0, "secondary": "natural_gas"},
    "IRN": {"sector": "oil",           "export_share_pct": 55.0, "secondary": "natural_gas"},
    "IRQ": {"sector": "oil",           "export_share_pct": 85.0, "secondary": "natural_gas"},
    "VEN": {"sector": "oil",           "export_share_pct": 75.0, "secondary": "gold"},
    "KAZ": {"sector": "oil",           "export_share_pct": 50.0, "secondary": "copper"},
    "AZE": {"sector": "oil",           "export_share_pct": 65.0, "secondary": "natural_gas"},
    "CHL": {"sector": "copper",        "export_share_pct": 45.0, "secondary": "gold"},
    "PER": {"sector": "copper",        "export_share_pct": 25.0, "secondary": "gold"},
    "ZMB": {"sector": "copper",        "export_share_pct": 70.0, "secondary": "manufacturing"},
    "COD": {"sector": "copper",        "export_share_pct": 55.0, "secondary": "gold"},
    "ZAF": {"sector": "gold",          "export_share_pct": 12.0, "secondary": "coal"},
    "GHA": {"sector": "gold",          "export_share_pct": 35.0, "secondary": "oil"},
    "THA": {"sector": "rubber",        "export_share_pct": 6.0,  "secondary": "manufacturing"},
    "CHN": {"sector": "manufacturing", "export_share_pct": 18.0, "secondary": "coal"},
    "DEU": {"sector": "manufacturing", "export_share_pct": 20.0, "secondary": "manufacturing"},
    "JPN": {"sector": "manufacturing", "export_share_pct": 22.0, "secondary": "manufacturing"},
    "TUR": {"sector": "manufacturing", "export_share_pct": 15.0, "secondary": "tourism"},
    "GRC": {"sector": "tourism",       "export_share_pct": 20.0, "secondary": "manufacturing"},
    "ESP": {"sector": "tourism",       "export_share_pct": 12.0, "secondary": "manufacturing"},
    "THA": {"sector": "tourism",       "export_share_pct": 11.0, "secondary": "rubber"},
    "MAR": {"sector": "tourism",       "export_share_pct": 10.0, "secondary": "manufacturing"},
    "UZB": {"sector": "cotton",        "export_share_pct": 15.0, "secondary": "natural_gas"},
    "PAK": {"sector": "cotton",        "export_share_pct": 22.0, "secondary": "manufacturing"},
    "IND": {"sector": "cotton",        "export_share_pct": 8.0,  "secondary": "manufacturing"},
    "AUS": {"sector": "coal",          "export_share_pct": 14.0, "secondary": "wheat"},
    "ZAF": {"sector": "coal",          "export_share_pct": 8.0,  "secondary": "gold"},
    "MOZ": {"sector": "coal",          "export_share_pct": 25.0, "secondary": "natural_gas"},
}

# World Bank Pink Sheet commodity price codes
# Available at: https://www.worldbank.org/en/research/commodity-markets
WB_PINK_SHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570"
    "-0350012021/related/CMO-Historical-Data-Monthly.xlsx"
)

# Fallback: alternative Pink Sheet URL
WB_PINK_SHEET_ALT = (
    "https://www.worldbank.org/en/research/commodity-markets"
)

# Z-score thresholds for signal generation
Z_HIGH_THRESHOLD  =  2.0   # unnatural high — sell major, consider inverse
Z_LOW_THRESHOLD   = -2.0   # unnatural low  — potential major recovery
Z_MODERATE_HIGH   =  1.5   # approaching high — watch
Z_MODERATE_LOW    = -1.5   # approaching low  — watch
ROLLING_WINDOW    = 120    # months (10 years) for z-score baseline


def fetch_wb_pink_sheet(force_download=False):
    """
    Download World Bank Pink Sheet commodity price data.
    Returns DataFrame with monthly prices for 70+ commodities.
    """
    cache_path = RAW_DIR / "wb_pink_sheet.parquet"
    if cache_path.exists() and not force_download:
        logger.info("Loading cached Pink Sheet data")
        return pd.read_parquet(cache_path)

    logger.info("Downloading World Bank Pink Sheet commodity prices...")

    for url in [WB_PINK_SHEET_URL]:
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.content) > 10000:
                # Pink Sheet is an Excel file
                from io import BytesIO
                xl = pd.read_excel(
                    BytesIO(resp.content),
                    sheet_name="Monthly Prices",
                    skiprows=4,
                    engine="openpyxl",
                )
                df = _parse_pink_sheet(xl)
                if df is not None and len(df) > 10:
                    df.to_parquet(cache_path)
                    logger.info(f"Pink Sheet saved: {df.shape}")
                    return df
        except Exception as e:
            logger.warning(f"Pink Sheet download failed: {e}")

    # Fallback: build synthetic price series from known commodity APIs
    logger.warning("Pink Sheet unavailable — building from World Bank commodity API")
    return _fetch_wb_commodity_api()


def _parse_pink_sheet(xl):
    """Parse the World Bank Pink Sheet Excel file."""
    try:
        # Row 0 is units ($/bbl etc.) — drop it
        xl = xl.iloc[1:].copy()

        # Date column is 'Unnamed: 0', format is '1960M01'
        xl = xl.rename(columns={"Unnamed: 0": "date"})
        xl = xl.dropna(subset=["date"])

        # Parse 'YYYYMXX' format → datetime
        def parse_wb_date(s):
            try:
                s = str(s).strip()
                year  = int(s[:4])
                month = int(s[5:])
                return pd.Timestamp(year=year, month=month, day=1)
            except Exception:
                return pd.NaT

        xl["date"] = xl["date"].apply(parse_wb_date)
        xl = xl.dropna(subset=["date"])
        xl = xl.set_index("date")
        xl = xl.apply(pd.to_numeric, errors="coerce")
        xl = xl.sort_index()

        # Rename columns to our commodity codes
        col_map = {
            "Crude oil, average":   "OIL_WTI",
            "Crude oil, WTI":       "OIL_WTI",
            "Crude oil, Brent":     "OIL_BRENT",
            "Coal, Australian":     "COAL_AUS",
            "Natural gas, US":      "NATGAS_EUR",
            "Natural gas, Europe":  "NATGAS_EUR",
            "Cocoa":                "COCOA",
            "Coffee, Arabica":      "COFFEE_OTHER",
            "Coffee, Robusta":      "COFFEE_OTHER",
            "Tea, avg 3 auctions":  "TEA_PROXY",
            "Wheat, US SRW":        "WHEAT_US",
            "Wheat, US HRW":        "WHEAT_US",
            "Maize":                "MAIZE_US",
            "Soybeans":             "SOYBEAN_US",
            "Palm oil":             "PALM_OIL",
            "Soybean oil":          "SOYBEAN_OIL",
            "Groundnut oil":        "GROUNDNUT_OIL",
            "Sunflower oil":        "SUNFLOWER_PROXY",
            "Coconut oil":          "COCONUT_OIL",
            "Rapeseed oil":         "CANOLA_PROXY",
            "Cotton, A Index":      "COTTON_A_IDX",
            "Rubber, TSR20":        "RUBBER_SGP",
            "Copper":               "COPPER_LME",
            "Aluminum":             "ALUMINIUM_LME",
            "Gold":                 "GOLD_LME",
            "Silver":               "SILVER",
            "Platinum":             "PLATINUM",
            "Tin":                  "TIN",
            "Nickel":               "NICKEL",
            "Zinc":                 "ZINC",
            "Lead":                 "LEAD",
            "Iron ore, cfr spot":   "IRON_ORE",
            "Uranium":              "URANIUM",
        }
        # Apply renames for columns that exist
        xl = xl.rename(columns={k: v for k, v in col_map.items() if k in xl.columns})

        # For duplicate target columns, keep the first
        xl = xl.loc[:, ~xl.columns.duplicated()]

        logger.info(
            f"Pink Sheet parsed: {len(xl)} months "
            f"({xl.index[0].date()} → {xl.index[-1].date()}), "
            f"{len(xl.columns)} commodities"
        )
        return xl

    except Exception as e:
        logger.warning(f"Pink Sheet parse error: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None


def _fetch_wb_commodity_api():
    """
    Fallback: fetch commodity prices from World Bank API.
    Returns DataFrame indexed by date with commodity columns.
    """
    # World Bank commodity price indicators
    wb_commodities = {
        "PCOALAUUSD":   "COAL_AUS",
        "PCOALAUUSDM":  "COAL_AUS",
        "POILWTIUSDM":  "OIL_WTI",
        "PNGASEUUSDM":  "NATGAS_EUR",
        "PCOFIBROBUSD": "COFFEE_OTHER",
        "PWHEAMTUSD":   "WHEAT_US",
        "PMAIZMTUSD":   "MAIZE_US",
        "PSOYBUSDM":    "SOYBEAN_US",
        "PCOPPUSDM":    "COPPER_LME",
        "PGOLDUSDM":    "GOLD_LME",
        "PALMUSDM":     "PALM_OIL",
        "PCOTLUSDM":    "COTTON_A_IDX",
        "PRUBBUSDM":    "RUBBER_SGP",
    }

    frames = []
    for wb_code, our_name in wb_commodities.items():
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{wb_code}"
            f"?format=json&per_page=1000&mrv=300"
        )
        try:
            resp = requests.get(url, timeout=20)
            data = resp.json()
            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                continue
            records = []
            for entry in data[1]:
                if entry.get("value") and entry.get("date"):
                    records.append({
                        "date":   entry["date"],
                        our_name: float(entry["value"]),
                    })
            if records:
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"], format="%Y", errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date")
                frames.append(df)
                logger.info(f"  Fetched {our_name}: {len(df)} observations")
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"  Failed {wb_code}: {e}")

    if not frames:
        logger.warning("No commodity data retrieved from any source")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index()
    return combined


def compute_zscore(series, window=ROLLING_WINDOW):
    """
    Compute rolling z-score for a price series.
    z = (current - rolling_mean) / rolling_std
    Uses expanding window until enough data is available.
    """
    rolling_mean = series.rolling(window=window, min_periods=24).mean()
    rolling_std  = series.rolling(window=window, min_periods=24).std()
    zscore = (series - rolling_mean) / (rolling_std + 1e-10)
    return zscore


def classify_signal(zscore_val):
    """Classify a z-score into an action signal."""
    if pd.isna(zscore_val):
        return "insufficient_data"
    if zscore_val >= Z_HIGH_THRESHOLD:
        return "MAJOR_HIGH_SELL_INVERSE_BUY"
    if zscore_val >= Z_MODERATE_HIGH:
        return "APPROACHING_HIGH_WATCH"
    if zscore_val <= Z_LOW_THRESHOLD:
        return "MAJOR_LOW_INVERSE_SELL"
    if zscore_val <= Z_MODERATE_LOW:
        return "APPROACHING_LOW_WATCH"
    return "NEUTRAL"


def generate_alert_message(country, sector_name, zscore, signal, inverses):
    """Generate a human-readable alert for a country-sector signal."""
    sector_info = SECTOR_PAIRS.get(sector_name, {})
    direction   = "high" if zscore > 0 else "low"
    abs_z       = abs(zscore)

    if signal == "MAJOR_HIGH_SELL_INVERSE_BUY":
        action = "ALERT: Consider reducing exposure to major sector"
        inverse_action = "OPPORTUNITY: Inverse sectors likely to benefit"
        urgency = "HIGH" if abs_z >= 3.0 else "MODERATE"
    elif signal == "APPROACHING_HIGH_WATCH":
        action = "WATCH: Major sector approaching historically high levels"
        inverse_action = "MONITOR: Begin tracking inverse sector entry points"
        urgency = "LOW"
    elif signal == "MAJOR_LOW_INVERSE_SELL":
        action = "ALERT: Major sector at historically low levels — possible recovery"
        inverse_action = "CAUTION: Reduce inverse sector exposure — major may rebound"
        urgency = "HIGH" if abs_z >= 3.0 else "MODERATE"
    elif signal == "APPROACHING_LOW_WATCH":
        action = "WATCH: Major sector approaching historically low levels"
        inverse_action = "MONITOR: Inverse sector may face headwinds"
        urgency = "LOW"
    else:
        return None

    top_inverses = sorted(inverses, key=lambda x: x["elasticity"], reverse=True)[:3]
    inv_str = ", ".join(
        f"{inv['name']} (elasticity {inv['elasticity']}, "
        f"~{inv['lag_months']}mo lag)"
        for inv in top_inverses
    )

    return {
        "country":        country,
        "major_sector":   sector_name,
        "sector_desc":    sector_info.get("description", sector_name),
        "zscore":         round(zscore, 3),
        "signal":         signal,
        "urgency":        urgency,
        "action":         action,
        "inverse_sectors": inv_str,
        "inverse_action": inverse_action,
        "top_inverse":    top_inverses[0]["name"] if top_inverses else None,
        "top_elasticity": top_inverses[0]["elasticity"] if top_inverses else None,
        "top_lag_months": top_inverses[0]["lag_months"] if top_inverses else None,
    }


def run_module_d(force_download=False, save=True):
    logger.info("=== Stage 4 Module D v2: Inverse Sector Monitor ===")

    # Step 1: Fetch commodity price data
    logger.info("Step 1: Fetching commodity price data...")
    prices = fetch_wb_pink_sheet(force_download=force_download)

    if prices.empty:
        logger.warning("No price data available — generating framework output only")
        alerts = _generate_framework_output()
    else:
        alerts = _run_full_analysis(prices)

    # Step 2: Always generate the sector pair reference table
    logger.info("Step 2: Building sector pair reference table...")
    pair_table = build_sector_pair_table()

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        pair_table.to_csv(OUTPUTS_DIR / "sector_pairs.csv", index=False)
        logger.info(f"Saved sector_pairs.csv ({len(pair_table)} pairs)")

        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_df.to_csv(OUTPUTS_DIR / "sector_alerts.csv", index=False)
            logger.info(f"Saved sector_alerts.csv ({len(alerts_df)} alerts)")
            logger.info("\n" + "="*60)
            logger.info("ACTIVE SECTOR ALERTS:")
            logger.info("="*60)
            for alert in alerts:
                if alert["urgency"] in ["HIGH", "MODERATE"]:
                    logger.info(
                        f"\n[{alert['urgency']}] {alert['country']} — "
                        f"{alert['major_sector'].upper()}"
                    )
                    logger.info(f"  Z-score: {alert['zscore']:+.2f} | "
                                f"Signal: {alert['signal']}")
                    logger.info(f"  {alert['action']}")
                    logger.info(f"  {alert['inverse_action']}")
                    logger.info(f"  Top inverse: {alert['top_inverse']} "
                                f"(elasticity {alert['top_elasticity']}, "
                                f"{alert['top_lag_months']}mo lag)")
        else:
            logger.info("No active alerts generated")
            pd.DataFrame().to_csv(OUTPUTS_DIR / "sector_alerts.csv", index=False)

    return alerts, pair_table


def _run_full_analysis(prices):
    """Run z-score analysis when price data is available."""
    logger.info("Running z-score analysis on commodity prices...")
    alerts = []
    processed_sectors = set()

    for iso3, info in COUNTRY_PRIMARY_SECTORS.items():
        sector_name = info["sector"]
        if sector_name in processed_sectors:
            continue

        sector_data = SECTOR_PAIRS.get(sector_name)
        if not sector_data:
            continue

        wb_code = sector_data.get("wb_commodity")
        if not wb_code:
            logger.info(f"Skipping {sector_name} — no commodity price proxy available")
            processed_sectors.add(sector_name)
            continue
        if wb_code not in prices.columns:
            # Try partial match
            matching = [c for c in prices.columns
                        if any(part in c.upper()
                               for part in wb_code.upper().split("_")[:2])]
            if not matching:
                logger.warning(f"No price data found for {sector_name} ({wb_code})")
                continue
            wb_code = matching[0]

        price_series = prices[wb_code].dropna()
        if len(price_series) < 24:
            continue

        zscore_series = compute_zscore(price_series)
        latest_zscore = float(zscore_series.iloc[-1])
        signal        = classify_signal(latest_zscore)

        if signal == "NEUTRAL" or signal == "insufficient_data":
            processed_sectors.add(sector_name)
            continue

        # Find all countries with this primary sector
        countries_with_sector = [
            iso for iso, data in COUNTRY_PRIMARY_SECTORS.items()
            if data["sector"] == sector_name
        ]

        for iso3_country in countries_with_sector:
            alert = generate_alert_message(
                country     = iso3_country,
                sector_name = sector_name,
                zscore      = latest_zscore,
                signal      = signal,
                inverses    = sector_data["inverses"],
            )
            if alert:
                alert["export_share_pct"] = COUNTRY_PRIMARY_SECTORS[iso3_country].get(
                    "export_share_pct", 0
                )
                alert["latest_price"] = round(float(price_series.iloc[-1]), 2)
                alert["price_date"]   = str(price_series.index[-1].date())
                alerts.append(alert)

        processed_sectors.add(sector_name)

    alerts.sort(key=lambda x: (
        0 if x["urgency"] == "HIGH" else 1 if x["urgency"] == "MODERATE" else 2,
        -abs(x["zscore"])
    ))

    logger.info(f"Analysis complete: {len(alerts)} active alerts generated")
    return alerts


def _generate_framework_output():
    """
    Generate framework-only output when no price data is available.
    Returns structural alerts based on sector pair knowledge base only.
    """
    logger.info("Generating framework output (no live price data)...")
    alerts = []
    for iso3, info in COUNTRY_PRIMARY_SECTORS.items():
        sector_name = info["sector"]
        sector_data = SECTOR_PAIRS.get(sector_name, {})
        inverses    = sector_data.get("inverses", [])
        if not inverses:
            continue
        alerts.append({
            "country":         iso3,
            "major_sector":    sector_name,
            "sector_desc":     sector_data.get("description", sector_name),
            "zscore":          None,
            "signal":          "NO_PRICE_DATA",
            "urgency":         "INFO",
            "action":          "Price data unavailable — sector pair framework loaded",
            "inverse_sectors": ", ".join(
                f"{inv['name']} (e={inv['elasticity']})"
                for inv in inverses[:3]
            ),
            "inverse_action":  "Load Pink Sheet data for live signals",
            "top_inverse":     inverses[0]["name"] if inverses else None,
            "top_elasticity":  inverses[0]["elasticity"] if inverses else None,
            "top_lag_months":  inverses[0]["lag_months"] if inverses else None,
            "export_share_pct": info.get("export_share_pct", 0),
            "latest_price":    None,
            "price_date":      None,
        })
    return alerts


def build_sector_pair_table():
    """
    Build a flat reference table of all sector pairs.
    One row per (major_sector, inverse_sector) pair.
    Includes all countries with that major sector.
    """
    rows = []
    for sector_name, sector_data in SECTOR_PAIRS.items():
        countries = [
            iso for iso, info in COUNTRY_PRIMARY_SECTORS.items()
            if info["sector"] == sector_name
        ]
        for inv in sector_data["inverses"]:
            rows.append({
                "major_sector":       sector_name,
                "major_description":  sector_data["description"],
                "wb_commodity_code":  sector_data.get("wb_commodity"),
                "countries":          ", ".join(countries),
                "inverse_sector":     inv["name"],
                "mechanism":          inv["mechanism"],
                "elasticity":         inv["elasticity"],
                "lag_months":         inv["lag_months"],
                "ticker_proxy":       inv["ticker_proxy"],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    )
    run_module_d()
