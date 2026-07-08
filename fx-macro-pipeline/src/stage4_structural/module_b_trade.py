import requests
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# World Bank indicators
WB_INDICATORS = {
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",   # Current account balance % GDP
    "NE.EXP.GNFS.ZS":    "exports_pct_gdp",            # Exports of goods and services % GDP
    "NE.IMP.GNFS.ZS":    "imports_pct_gdp",            # Imports of goods and services % GDP
    "NY.GDP.MKTP.KD.ZG":  "gdp_growth",                # GDP growth annual %
    "FP.CPI.TOTL.ZG":    "inflation",                  # Inflation CPI %
}

# Minimum years of data required
MIN_YEARS = 10


def fetch_wb_indicator(indicator_code, col_name, start_year=2000, end_year=2024):
    """Fetch a single World Bank indicator for all countries."""
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
        f"?format=json&per_page=20000&date={start_year}:{end_year}"
    )
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            logger.warning(f"No data returned for {indicator_code}")
            return pd.DataFrame()

        records = []
        for entry in data[1]:
            if entry.get("value") is None:
                continue
            records.append({
                "iso3":    entry["countryiso3code"],
                "country": entry["country"]["value"],
                "year":    int(entry["date"]),
                col_name:  float(entry["value"]),
            })

        df = pd.DataFrame(records)
        df = df[df["iso3"].str.len() == 3]
        logger.info(f"Fetched {indicator_code}: {len(df)} observations")
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch {indicator_code}: {e}")
        return pd.DataFrame()


def fetch_all_trade_data(force_download=False):
    """
    Fetch all trade and macro indicators from World Bank.
    Returns wide DataFrame: rows=(iso3, year), columns=indicators.
    """
    cache_path = RAW_DIR / "wb_trade_data.parquet"
    if cache_path.exists() and not force_download:
        logger.info("Loading cached World Bank trade data")
        return pd.read_parquet(cache_path)

    logger.info("Fetching World Bank trade and macro data...")
    frames = []
    for code, name in WB_INDICATORS.items():
        df = fetch_wb_indicator(code, name)
        if not df.empty:
            frames.append(df.set_index(["iso3", "country", "year"]))
        time.sleep(0.3)

    if not frames:
        raise RuntimeError("Could not fetch any World Bank data")

    combined = pd.concat(frames, axis=1).reset_index()
    combined = combined.sort_values(["iso3", "year"])

    combined.to_parquet(cache_path)
    logger.info(f"World Bank data saved: {combined.shape}")
    return combined


def compute_trade_metrics(trade_df):
    """
    Compute per-country trade balance metrics:
      - trade_balance_pct_gdp: exports - imports as % GDP
      - surplus_years_pct: % of years with positive trade balance
      - mean_current_account: average current account % GDP
      - trade_trend: linear trend of current account over sample
      - trade_volatility: std of annual current account changes
    """
    results = []
    for (iso3, country), grp in trade_df.groupby(["iso3", "country"]):
        grp = grp.sort_values("year")

        # Trade balance
        if "exports_pct_gdp" in grp and "imports_pct_gdp" in grp:
            grp["trade_balance"] = grp["exports_pct_gdp"] - grp["imports_pct_gdp"]
        elif "current_account_pct_gdp" in grp:
            grp["trade_balance"] = grp["current_account_pct_gdp"]
        else:
            continue

        tb = grp["trade_balance"].dropna()
        ca = grp["current_account_pct_gdp"].dropna() if "current_account_pct_gdp" in grp else tb

        if len(tb) < MIN_YEARS:
            continue

        # Linear trend
        x = np.arange(len(ca))
        trend = float(np.polyfit(x, ca.values, 1)[0]) if len(ca) >= 2 else 0.0

        results.append({
            "iso3":                   iso3,
            "country":                country,
            "mean_trade_balance":     round(float(tb.mean()), 3),
            "surplus_years_pct":      round(float((tb > 0).mean() * 100), 1),
            "mean_current_account":   round(float(ca.mean()), 3),
            "current_account_trend":  round(trend, 4),
            "trade_volatility":       round(float(tb.std()), 3),
            "mean_gdp_growth":        round(float(grp["gdp_growth"].dropna().mean()), 3)
                                      if "gdp_growth" in grp else None,
            "mean_inflation":         round(float(grp["inflation"].dropna().mean()), 3)
                                      if "inflation" in grp else None,
            "data_years":             len(tb),
        })

    df = pd.DataFrame(results)
    logger.info(f"Trade metrics computed for {len(df)} countries")
    return df


def score_trade_strength(metrics_df):
    """
    Score each country on trade balance strength (0-1 normalised).
    Components:
      - mean_current_account (40%): persistent surplus = strong
      - surplus_years_pct (30%): consistency of surplus
      - current_account_trend (20%): improving trend
      - trade_volatility inverse (10%): lower volatility = better
    """
    df = metrics_df.copy()

    def norm(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        n = (series - mn) / (mx - mn)
        return 1 - n if invert else n

    df["n_ca"]   = norm(df["mean_current_account"])
    df["n_surp"] = norm(df["surplus_years_pct"])
    df["n_trend"]= norm(df["current_account_trend"])
    df["n_vol"]  = norm(df["trade_volatility"], invert=True)

    df["trade_strength_score"] = (
        0.40 * df["n_ca"] +
        0.30 * df["n_surp"] +
        0.20 * df["n_trend"] +
        0.10 * df["n_vol"]
    ).round(4)

    # Grade
    def grade(s):
        if s >= 0.75: return "surplus_strong"
        if s >= 0.55: return "surplus_moderate"
        if s >= 0.35: return "balanced"
        return "deficit"

    df["trade_grade"] = df["trade_strength_score"].apply(grade)
    df = df.sort_values("trade_strength_score", ascending=False)

    logger.info(
        f"Trade scores computed: "
        f"{(df['trade_grade']=='surplus_strong').sum()} strong surplus, "
        f"{(df['trade_grade']=='deficit').sum()} deficit countries"
    )
    return df


def run_module_b(force_download=False, save=True):
    logger.info("=== Stage 4 Module B: Trade Balance Correlator ===")

    trade_raw     = fetch_all_trade_data(force_download=force_download)
    trade_metrics = compute_trade_metrics(trade_raw)
    trade_scores  = score_trade_strength(trade_metrics)

    logger.info("\nTop 10 by trade strength:")
    logger.info(f"\n{trade_scores[['country','mean_current_account','surplus_years_pct','trade_strength_score','trade_grade']].head(10).to_string()}")

    logger.info("\nBottom 5 (persistent deficit):")
    logger.info(f"\n{trade_scores[['country','mean_current_account','surplus_years_pct','trade_strength_score','trade_grade']].tail(5).to_string()}")

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        trade_scores.to_csv(OUTPUTS_DIR / "trade_balance_scores.csv", index=False)
        logger.info(f"Saved trade_balance_scores.csv ({len(trade_scores)} countries)")

    return trade_scores


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    )
    run_module_b()
