import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "outputs"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Weight of the structural score in the final composite
# Stage 3 composite score is weighted by this multiplier
STRUCTURAL_WEIGHT = 0.25   # structural layer contributes 25% to final score

# Watchlist signal boost / penalty
WATCHLIST_IMPROVEMENT_BOOST    =  0.15
WATCHLIST_DETERIORATION_PENALTY = -0.20

# EFW score thresholds for structural quality tiers
EFW_TIER_MAP = [
    (7.5, "strong",   1.20),  # score >= 7.5 → multiplier 1.20
    (6.5, "good",     1.10),  # score >= 6.5 → multiplier 1.10
    (5.5, "moderate", 1.00),  # score >= 5.5 → multiplier 1.00
    (4.5, "weak",     0.90),  # score >= 4.5 → multiplier 0.90
    (0.0, "poor",     0.75),  # score <  4.5 → multiplier 0.75
]


def load_stage3_composite():
    """Load Stage 3 comparison report (currency-level composite scores)."""
    path = OUTPUTS_DIR / "comparison_report.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 3 comparison report not found at {path}. "
            f"Run Stage 2 first."
        )
    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded Stage 3 composite: {len(df)} currencies")
    return df


def load_module_a_outputs():
    """Load all Module A outputs."""
    catalogue = pd.read_csv(OUTPUTS_DIR / "ief_event_catalogue.csv")
    watchlist = pd.read_csv(OUTPUTS_DIR / "ief_watchlist.csv")
    leading   = pd.read_csv(
        OUTPUTS_DIR / "ief_leading_indicators.csv", index_col=0
    )
    logger.info(
        f"Module A outputs loaded: "
        f"{len(catalogue)} events, "
        f"{len(watchlist)} watchlist countries"
    )
    return catalogue, watchlist, leading


def load_efw_current():
    """Load the most recent EFW score per country."""
    path = Path(__file__).resolve().parents[2] / "data" / "raw" / "efw_historical.parquet"
    efw  = pd.read_parquet(path)
    # Get most recent year per country
    latest = (
        efw.sort_values("year")
           .groupby(["country", "iso3"])
           .last()
           .reset_index()[["country", "iso3", "year", "overall_score"]]
    )
    latest.columns = ["country", "iso3", "efw_year", "efw_score"]
    logger.info(
        f"EFW current scores: {len(latest)} countries, "
        f"most recent year: {int(latest['efw_year'].max())}"
    )
    return latest


def build_fx_to_iso3_map():
    """
    Map Frankfurter FX currency codes to ISO3 country codes.
    Used to join Stage 3 (currency-level) with Module A (country-level).
    Note: some currencies cover multiple countries (EUR) or
    don't map 1:1. We use the primary country association.
    """
    return {
        "AED": "ARE", "AFN": "AFG", "ALL": "ALB", "AMD": "ARM",
        "ANG": "ANT", "AOA": "AGO", "ARS": "ARG", "AUD": "AUS",
        "AZN": "AZE", "BAM": "BIH", "BBD": "BRB", "BDT": "BGD",
        "BGN": "BGR", "BHD": "BHR", "BIF": "BDI", "BMD": "BMU",
        "BND": "BRN", "BOB": "BOL", "BRL": "BRA", "BSD": "BHS",
        "BTN": "BTN", "BWP": "BWA", "BYN": "BLR", "BZD": "BLZ",
        "CAD": "CAN", "CDF": "COD", "CHF": "CHE", "CLP": "CHL",
        "CNH": "CHN", "CNY": "CHN", "COP": "COL", "CRC": "CRI",
        "CUP": "CUB", "CVE": "CPV", "CZK": "CZE", "DJF": "DJI",
        "DKK": "DNK", "DOP": "DOM", "DZD": "DZA", "EGP": "EGY",
        "ETB": "ETH", "EUR": "EUR", "FJD": "FJI", "GBP": "GBR",
        "GEL": "GEO", "GHS": "GHA", "GMD": "GMB", "GNF": "GIN",
        "GTQ": "GTM", "GYD": "GUY", "HKD": "HKG", "HNL": "HND",
        "HTG": "HTI", "HUF": "HUN", "IDR": "IDN", "ILS": "ISR",
        "INR": "IND", "IQD": "IRQ", "IRR": "IRN", "ISK": "ISL",
        "JMD": "JAM", "JOD": "JOR", "JPY": "JPN", "KES": "KEN",
        "KGS": "KGZ", "KHR": "KHM", "KMF": "COM", "KRW": "KOR",
        "KWD": "KWT", "KYD": "CYM", "KZT": "KAZ", "LAK": "LAO",
        "LBP": "LBN", "LKR": "LKA", "LRD": "LBR", "LSL": "LSO",
        "LYD": "LBY", "MAD": "MAR", "MDL": "MDA", "MGA": "MDG",
        "MKD": "MKD", "MMK": "MMR", "MNT": "MNG", "MOP": "MAC",
        "MRU": "MRT", "MUR": "MUS", "MVR": "MDV", "MWK": "MWI",
        "MXN": "MEX", "MYR": "MYS", "MZN": "MOZ", "NAD": "NAM",
        "NGN": "NGA", "NIO": "NIC", "NOK": "NOR", "NPR": "NPL",
        "NZD": "NZL", "OMR": "OMN", "PAB": "PAN", "PEN": "PER",
        "PGK": "PNG", "PHP": "PHL", "PKR": "PAK", "PLN": "POL",
        "PYG": "PRY", "QAR": "QAT", "RON": "ROU", "RSD": "SRB",
        "RUB": "RUS", "RWF": "RWA", "SAR": "SAU", "SBD": "SLB",
        "SCR": "SYC", "SDG": "SDN", "SEK": "SWE", "SGD": "SGP",
        "SHP": "SHN", "SOS": "SOM", "SRD": "SUR", "SSP": "SSD",
        "STN": "STP", "SVC": "SLV", "SYP": "SYR", "SZL": "SWZ",
        "THB": "THA", "TJS": "TJK", "TMT": "TKM", "TND": "TUN",
        "TOP": "TON", "TRY": "TUR", "TTD": "TTO", "TWD": "TWN",
        "TZS": "TZA", "UAH": "UKR", "UGX": "UGA", "USD": "USA",
        "UYU": "URY", "UZS": "UZB", "VES": "VEN", "VND": "VNM",
        "VUV": "VUT", "WST": "WSM", "XAF": "CMR", "XCD": "ATG",
        "XOF": "SEN", "XPF": "PYF", "YER": "YEM", "ZAR": "ZAF",
        "ZMW": "ZMB", "ZWG": "ZWE",
    }


def get_efw_tier(score):
    """Return (tier_label, multiplier) for a given EFW score."""
    if pd.isna(score):
        return "unknown", 1.0
    for threshold, label, multiplier in EFW_TIER_MAP:
        if score >= threshold:
            return label, multiplier
    return "poor", 0.75


def compute_structural_adjustment(
    currency,
    iso3,
    efw_current,
    watchlist_df,
    catalogue_df,
):
    """
    Compute the structural adjustment for a single currency.

    Returns a dict with:
      - efw_score: most recent EFW score
      - efw_tier: quality tier label
      - efw_multiplier: score multiplier from tier
      - watchlist_flag: improvement / deterioration / none
      - watchlist_boost: numeric adjustment from watchlist
      - recent_events: count of events in last 10 years
      - event_direction_bias: net direction of recent events
      - structural_adjustment: final combined adjustment
    """
    result = {
        "currency":             currency,
        "iso3":                 iso3,
        "efw_score":            None,
        "efw_tier":             "unknown",
        "efw_multiplier":       1.0,
        "watchlist_flag":       "none",
        "watchlist_boost":      0.0,
        "recent_events":        0,
        "event_direction_bias": 0.0,
        "structural_adjustment": 0.0,
    }

    if iso3 is None:
        return result

    # EFW current score
    match = efw_current[efw_current["iso3"] == iso3]
    if len(match):
        score = float(match["efw_score"].values[0])
        tier, mult = get_efw_tier(score)
        result["efw_score"]      = round(score, 3)
        result["efw_tier"]       = tier
        result["efw_multiplier"] = mult

    # Watchlist status
    wl_match = watchlist_df[watchlist_df["iso3"] == iso3]
    if len(wl_match):
        direction = str(wl_match["projected_direction"].values[0])
        result["watchlist_flag"] = direction
        result["watchlist_boost"] = (
            WATCHLIST_IMPROVEMENT_BOOST
            if direction == "improvement"
            else WATCHLIST_DETERIORATION_PENALTY
        )

    # Recent event history (last 10 years from most recent EFW data = 2023)
    recent_cutoff = 2013
    recent_events = catalogue_df[
        (catalogue_df["iso3"] == iso3) &
        (catalogue_df["event_year"] >= recent_cutoff)
    ]
    result["recent_events"] = len(recent_events)

    if len(recent_events):
        improvements   = (recent_events["direction"] == "improvement").sum()
        deteriorations = (recent_events["direction"] == "deterioration").sum()
        # Bias: +1 = all improvements, -1 = all deteriorations
        total = improvements + deteriorations
        result["event_direction_bias"] = round(
            (improvements - deteriorations) / total, 3
        )

    # Final structural adjustment:
    # = (efw_multiplier - 1.0)     ← base tier effect
    # + watchlist_boost             ← early warning signal
    # + 0.05 * direction_bias       ← recent event history nudge
    result["structural_adjustment"] = round(
        (result["efw_multiplier"] - 1.0)
        + result["watchlist_boost"]
        + 0.05 * result["event_direction_bias"],
        4,
    )

    return result


def integrate(save=True):
    """
    Main integration function.
    Loads Stage 3 composite scores and Module A outputs,
    computes structural adjustments per currency, and
    produces an enhanced composite score.
    """
    logger.info("=== Stage 4 Module A → Stage 3 Integration ===")

    # Load all inputs
    stage3      = load_stage3_composite()
    catalogue, watchlist, leading = load_module_a_outputs()
    efw_current = load_efw_current()
    fx_iso3_map = build_fx_to_iso3_map()

    # Compute structural adjustment per currency
    adjustments = []
    for currency in stage3.index:
        iso3 = fx_iso3_map.get(currency)
        adj  = compute_structural_adjustment(
            currency, iso3, efw_current, watchlist, catalogue
        )
        adjustments.append(adj)

    adj_df = pd.DataFrame(adjustments).set_index("currency")

    # Merge with Stage 3
    combined = stage3.join(adj_df, how="left")

    # Recompute composite score incorporating structural layer
    # Original composite (0–1 normalised) + structural_adjustment weighted
    # by STRUCTURAL_WEIGHT
    def enhanced_composite(row):
        base = row.get("spectral_sharpe", 0)
        if pd.isna(base):
            base = 0
        # Normalise Sharpe to [0,1] over [-2, +5]
        base_n = max(0, min(1, (base + 2) / 7))

        acc = row.get("spectral_dir_accuracy", 0.5)
        if pd.isna(acc):
            acc = 0.5

        mcc = row.get("spectral_mcc", 0)
        if pd.isna(mcc):
            mcc = 0
        mcc_n = (mcc + 1) / 2

        sig = 1 if row.get("significant", False) else 0

        # Original Stage 3 score (same formula as dashboard)
        stage3_score = 0.40 * base_n + 0.25 * acc + 0.25 * mcc_n + 0.10 * sig

        # Structural adjustment
        struct_adj = row.get("structural_adjustment", 0.0)
        if pd.isna(struct_adj):
            struct_adj = 0.0

        # Final: blend stage3 score with structural adjustment
        final = stage3_score * (1 - STRUCTURAL_WEIGHT) + (
            stage3_score + struct_adj
        ) * STRUCTURAL_WEIGHT
        return round(final, 4)

    combined["stage3_composite"]   = combined.apply(
        lambda r: round(
            0.40 * max(0, min(1, (r.get("spectral_sharpe", 0) + 2) / 7))
            + 0.25 * (r.get("spectral_dir_accuracy", 0.5) or 0.5)
            + 0.25 * ((r.get("spectral_mcc", 0) or 0) + 1) / 2
            + 0.10 * (1 if r.get("significant", False) else 0),
            4,
        ),
        axis=1,
    )
    combined["enhanced_composite"] = combined.apply(enhanced_composite, axis=1)
    combined["composite_delta"]    = (
        combined["enhanced_composite"] - combined["stage3_composite"]
    ).round(4)

    # Grade the enhanced composite
    def grade(score):
        if score >= 0.72: return "S"
        if score >= 0.62: return "A"
        if score >= 0.52: return "B"
        if score >= 0.42: return "C"
        return "D"

    combined["enhanced_grade"] = combined["enhanced_composite"].apply(grade)

    # Summary stats
    logger.info(f"\n{'='*50}")
    logger.info("Integration summary:")
    logger.info(f"  Currencies integrated:     {len(combined)}")
    logger.info(
        f"  Currencies with EFW data:  "
        f"{combined['efw_score'].notna().sum()}"
    )
    logger.info(
        f"  Currencies on watchlist:   "
        f"{(combined['watchlist_flag'] == 'improvement').sum()}"
    )
    logger.info(
        f"  Mean composite delta:      "
        f"{combined['composite_delta'].mean():.4f}"
    )
    logger.info(
        f"  Biggest upgrades:\n"
        f"{combined.nlargest(5,'composite_delta')[['enhanced_composite','efw_tier','watchlist_flag','composite_delta']].to_string()}"
    )
    logger.info(
        f"  Biggest downgrades:\n"
        f"{combined.nsmallest(5,'composite_delta')[['enhanced_composite','efw_tier','watchlist_flag','composite_delta']].to_string()}"
    )

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUTS_DIR / "enhanced_composite.csv")
        logger.info(
            f"\nSaved → data/outputs/enhanced_composite.csv"
        )

    return combined


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    )
    integrate()
