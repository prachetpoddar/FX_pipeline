"""
universe.py
===========

Defines the tradable currency-pair universe for Stage 1 v2.

This is intentionally narrow (~18 pairs) compared to the original 170 because
the project goal is *real trading*, and most of the original Frankfurter
universe was not tradable. Every pair here is offered by at least one major
retail FX broker (OANDA, IG, Interactive Brokers).

PAIR NAMING CONVENTION
----------------------
We use the conventional FX market quote:
    EURUSD  = USD per 1 EUR  (EUR is base, USD is quote)
    USDJPY  = JPY per 1 USD  (USD is base)
Always quoted such that the base is the "stronger" or more commonly-traded
currency.  This matters because all returns we compute are "% change in
quote per unit base" — flipping the quote flips the sign of returns.

TIERS
-----
Tier 1 = G10 majors, highly liquid, tight retail spreads (1-5 bps round-trip)
Tier 2 = liquid EM, retail-tradable, spreads 10-30 bps
Tier 3 = less liquid EM and pegged currencies; some are NDF-only and
         may not have HistData coverage. Included for research; we'll
         see what survives the data acquisition step.
"""

UNIVERSE = {
    # ── Tier 1: G10 majors ────────────────────────────────────────────────
    "EURUSD": {"tier": 1, "base": "EUR", "quote": "USD",
               "histdata_pair": "eurusd"},
    "GBPUSD": {"tier": 1, "base": "GBP", "quote": "USD",
               "histdata_pair": "gbpusd"},
    "USDJPY": {"tier": 1, "base": "USD", "quote": "JPY",
               "histdata_pair": "usdjpy"},
    "USDCHF": {"tier": 1, "base": "USD", "quote": "CHF",
               "histdata_pair": "usdchf"},
    "AUDUSD": {"tier": 1, "base": "AUD", "quote": "USD",
               "histdata_pair": "audusd"},
    "NZDUSD": {"tier": 1, "base": "NZD", "quote": "USD",
               "histdata_pair": "nzdusd"},
    "USDCAD": {"tier": 1, "base": "USD", "quote": "CAD",
               "histdata_pair": "usdcad"},
    "USDSEK": {"tier": 1, "base": "USD", "quote": "SEK",
               "histdata_pair": "usdsek"},
    "USDNOK": {"tier": 1, "base": "USD", "quote": "NOK",
               "histdata_pair": "usdnok"},

    # ── Tier 2: liquid EM ─────────────────────────────────────────────────
    "USDMXN": {"tier": 2, "base": "USD", "quote": "MXN",
               "histdata_pair": "usdmxn"},
    "USDZAR": {"tier": 2, "base": "USD", "quote": "ZAR",
               "histdata_pair": "usdzar"},
    "USDTRY": {"tier": 2, "base": "USD", "quote": "TRY",
               "histdata_pair": "usdtry"},
    "USDPLN": {"tier": 2, "base": "USD", "quote": "PLN",
               "histdata_pair": "usdpln"},
    "USDHUF": {"tier": 2, "base": "USD", "quote": "HUF",
               "histdata_pair": "usdhuf"},
    "USDCZK": {"tier": 2, "base": "USD", "quote": "CZK",
               "histdata_pair": "usdczk"},
    "USDSGD": {"tier": 2, "base": "USD", "quote": "SGD",
               "histdata_pair": "usdsgd"},

    # ── Tier 3: less liquid / pegged / NDF-only ───────────────────────────
    # HistData coverage uncertain for these — fetcher will report which
    # actually succeed.
    "USDHKD": {"tier": 3, "base": "USD", "quote": "HKD",
               "histdata_pair": "usdhkd",
               "notes": "PEGGED to USD via HKMA band; signal expected to be ~0"},
    "USDILS": {"tier": 3, "base": "USD", "quote": "ILS",
               "histdata_pair": "usdils"},
}


def get_pairs_by_tier(tier=None):
    """Return list of pair codes, optionally filtered by tier."""
    if tier is None:
        return list(UNIVERSE.keys())
    return [k for k, v in UNIVERSE.items() if v["tier"] == tier]


def is_pegged(pair_code):
    """True if pair is known to be pegged (so expect no real return signal)."""
    return "PEGGED" in UNIVERSE.get(pair_code, {}).get("notes", "")
