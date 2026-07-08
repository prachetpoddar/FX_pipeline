import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from module_b_trade      import run_module_b
from module_c_volatility import run_module_c
from module_d_sectors    import run_module_d
from integrate_module_a  import integrate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"


def run_all():
    logger.info("=" * 60)
    logger.info("=== Stage 4: Full Structural Analysis ===")
    logger.info("=" * 60)

    # Module B
    logger.info("\n--- Module B: Trade Balance ---")
    trade_scores = run_module_b()

    # Module C
    logger.info("\n--- Module C: Policy Volatility ---")
    volatility_scores, base_models = run_module_c()

    # Module D
    logger.info("\n--- Module D: Sector Export Monitor ---")
    sector_alerts, sector_scores = run_module_d()

    # Final integration with Stage 3
    logger.info("\n--- Final Integration: Stage 3 + All Module A-D Scores ---")
    enhanced = integrate_all(trade_scores, volatility_scores, sector_scores)

    logger.info("\n" + "=" * 60)
    logger.info("=== Stage 4 complete ===")
    logger.info(f"Outputs saved to: {OUTPUTS_DIR}")
    return enhanced


def integrate_all(trade_df, volatility_df, sector_df):
    """
    Extend the Module A integration to also incorporate
    Module B (trade) and C (volatility) scores.
    Module D v2 produces alerts rather than per-country scores
    so it feeds the dashboard directly via sector_alerts.csv.
    """
    import pandas as pd
    import numpy as np

    # Load existing enhanced composite from Module A integration
    comp_path = OUTPUTS_DIR / "enhanced_composite.csv"
    if not comp_path.exists():
        logger.warning("enhanced_composite.csv not found — running Module A integration first")
        from integrate_module_a import integrate
        enhanced = integrate(save=True)
    else:
        enhanced = pd.read_csv(comp_path, index_col=0)

    from integrate_module_a import build_fx_to_iso3_map
    fx_map = build_fx_to_iso3_map()

    # Module B scores indexed by ISO3
    if not trade_df.empty and "iso3" in trade_df.columns:
        trade_idx = trade_df.set_index("iso3")[
            ["trade_strength_score", "trade_grade",
             "mean_current_account", "surplus_years_pct"]
        ]
    else:
        trade_idx = pd.DataFrame()

    # Module C scores indexed by ISO3
    if not volatility_df.empty and "iso3" in volatility_df.columns:
        vol_idx = volatility_df.set_index("iso3")[
            ["stability_score", "classification",
             "policy_volatility", "post_crisis_deviation"]
        ]
    else:
        vol_idx = pd.DataFrame()

    # Merge into enhanced composite
    for currency in enhanced.index:
        iso3 = fx_map.get(currency)
        if not iso3:
            continue
        if not trade_idx.empty and iso3 in trade_idx.index:
            for col in trade_idx.columns:
                enhanced.loc[currency, col] = trade_idx.loc[iso3, col]
        if not vol_idx.empty and iso3 in vol_idx.index:
            for col in vol_idx.columns:
                enhanced.loc[currency, col] = vol_idx.loc[iso3, col]

    # Recompute full composite with trade and stability adjustments
    def final_composite(row):
        base = float(row.get("enhanced_composite", 0) or 0)
        ts = row.get("trade_strength_score")
        trade_adj = float((ts - 0.5) * 0.10) if ts and not np.isnan(float(ts)) else 0.0
        ss = row.get("stability_score")
        stab_adj = float((ss - 0.5) * 0.10) if ss and not np.isnan(float(ss)) else 0.0
        return round(base + trade_adj + stab_adj, 4)

    enhanced["full_composite"] = enhanced.apply(final_composite, axis=1)

    def full_grade(s):
        if s >= 0.72: return "S"
        if s >= 0.62: return "A"
        if s >= 0.52: return "B"
        if s >= 0.42: return "C"
        return "D"

    enhanced["full_grade"] = enhanced["full_composite"].apply(full_grade)
    enhanced.to_csv(OUTPUTS_DIR / "enhanced_composite.csv")
    logger.info("Updated enhanced_composite.csv with Module B and C scores")

    top10 = enhanced.nlargest(10, "full_composite")[
        ["full_composite", "full_grade", "efw_tier",
         "trade_grade", "classification", "watchlist_flag"]
    ]
    logger.info(f"\nTop 10 currencies by full composite:\n{top10.to_string()}")
    return enhanced

    # Recompute final composite incorporating all four modules
    def final_composite(row):
        base = float(row.get("enhanced_composite", 0) or 0)

        # Trade bonus/penalty (-0.05 to +0.05)
        ts = row.get("trade_strength_score")
        trade_adj = float((ts - 0.5) * 0.10) if ts and not np.isnan(float(ts)) else 0.0

        # Stability bonus/penalty (-0.05 to +0.05)
        ss = row.get("stability_score")
        stab_adj = float((ss - 0.5) * 0.10) if ss and not np.isnan(float(ss)) else 0.0

        # Sector health bonus (-0.03 to +0.03)
        sh = row.get("sector_health_score")
        sect_adj = float((sh - 0.5) * 0.06) if sh and not np.isnan(float(sh)) else 0.0

        return round(base + trade_adj + stab_adj + sect_adj, 4)

    enhanced["full_composite"] = enhanced.apply(final_composite, axis=1)

    def full_grade(s):
        if s >= 0.72: return "S"
        if s >= 0.62: return "A"
        if s >= 0.52: return "B"
        if s >= 0.42: return "C"
        return "D"

    enhanced["full_grade"] = enhanced["full_composite"].apply(full_grade)

    enhanced.to_csv(OUTPUTS_DIR / "enhanced_composite.csv")
    logger.info(f"Updated enhanced_composite.csv with all module scores")

    top10 = enhanced.nlargest(10, "full_composite")[
        ["full_composite", "full_grade", "efw_tier",
         "trade_grade", "classification", "watchlist_flag"]
    ]
    logger.info(f"\nTop 10 currencies by full composite:\n{top10.to_string()}")

    return enhanced


if __name__ == "__main__":
    run_all()
