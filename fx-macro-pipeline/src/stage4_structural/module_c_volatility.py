import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)

RAW_DIR     = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"

# Rolling window for volatility computation (years)
VOLATILITY_WINDOW = 5
# Clean period for base model
CLEAN_PERIOD_START = 2013
CLEAN_PERIOD_END   = 2019
# Minimum history required
MIN_YEARS = 8


def load_efw_data():
    path = RAW_DIR / "efw_historical.parquet"
    return pd.read_parquet(path)


def compute_policy_volatility(efw_df):
    """
    For each country compute:
      - policy_volatility: std of year-on-year EFW score changes (full period)
      - clean_period_score: mean EFW score during 2013-2019 base period
      - clean_period_volatility: volatility during 2013-2019 only
      - post_clean_drift: mean score change per year since 2019
      - regime_change_count: number of years with >THRESHOLD change
      - reform_consistency: % of years moving in same direction as overall trend
      - classification: stable_reformer / volatile / stagnant / deteriorating
    """
    CHANGE_THRESHOLD = 0.3  # points on 0-10 scale = significant annual shift

    results = []
    for (country, iso3), grp in efw_df.groupby(["country", "iso3"]):
        grp = grp.sort_values("year").copy()
        grp["yoy_change"] = grp["overall_score"].diff()
        valid = grp[grp["yoy_change"].notna()]

        if len(valid) < MIN_YEARS:
            continue

        # Full period volatility
        full_vol = float(valid["yoy_change"].std())

        # Clean period
        clean = grp[
            (grp["year"] >= CLEAN_PERIOD_START) &
            (grp["year"] <= CLEAN_PERIOD_END)
        ]
        clean_score = float(clean["overall_score"].mean()) if len(clean) >= 2 else None
        clean_vol   = float(clean["yoy_change"].std()) if len(clean) >= 3 else None

        # Post-clean drift (2020 onwards)
        post = grp[grp["year"] > CLEAN_PERIOD_END]
        if len(post) >= 2:
            x = np.arange(len(post))
            post_drift = float(np.polyfit(x, post["overall_score"].values, 1)[0])
        else:
            post_drift = 0.0

        # Regime change count
        regime_changes = int((valid["yoy_change"].abs() >= CHANGE_THRESHOLD).sum())

        # Reform consistency: % of changes in dominant direction
        positive_changes = (valid["yoy_change"] > 0).sum()
        negative_changes = (valid["yoy_change"] < 0).sum()
        dominant = max(positive_changes, negative_changes)
        consistency = float(dominant / len(valid)) if len(valid) > 0 else 0.5

        # Overall trend direction
        x_full = np.arange(len(grp.dropna(subset=["overall_score"])))
        y_full = grp.dropna(subset=["overall_score"])["overall_score"].values
        overall_trend = float(np.polyfit(x_full, y_full, 1)[0]) if len(x_full) >= 2 else 0.0

        # Classification
        if full_vol < 0.15 and overall_trend >= 0:
            classification = "stable_reformer"
        elif full_vol < 0.15 and overall_trend < 0:
            classification = "stagnant"
        elif full_vol >= 0.15 and overall_trend >= 0:
            classification = "volatile_improving"
        else:
            classification = "volatile_deteriorating"

        # Current score
        current_score = float(grp["overall_score"].iloc[-1]) if len(grp) > 0 else None
        current_year  = int(grp["year"].iloc[-1]) if len(grp) > 0 else None

        results.append({
            "iso3":                  iso3,
            "country":               country,
            "current_efw_score":     round(current_score, 3) if current_score else None,
            "current_year":          current_year,
            "policy_volatility":     round(full_vol, 4),
            "clean_period_score":    round(clean_score, 3) if clean_score else None,
            "clean_period_volatility": round(clean_vol, 4) if clean_vol else None,
            "post_clean_drift":      round(post_drift, 4),
            "overall_trend":         round(overall_trend, 4),
            "regime_change_count":   regime_changes,
            "reform_consistency":    round(consistency, 3),
            "classification":        classification,
            "data_years":            len(valid),
        })

    df = pd.DataFrame(results)

    # Distribution summary
    for cls in ["stable_reformer","stagnant","volatile_improving","volatile_deteriorating"]:
        n = (df["classification"] == cls).sum()
        logger.info(f"  {cls}: {n} countries")

    logger.info(f"Policy volatility computed for {len(df)} countries")
    return df


def score_policy_stability(volatility_df):
    """
    Convert volatility metrics into a stability score (0-1).
    Higher = more stable and consistent reform trajectory.
    """
    df = volatility_df.copy()

    def norm(s, invert=False):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        n = (s - mn) / (mx - mn)
        return 1 - n if invert else n

    df["n_vol"]   = norm(df["policy_volatility"], invert=True)
    df["n_cons"]  = norm(df["reform_consistency"])
    df["n_trend"] = norm(df["overall_trend"])
    df["n_reg"]   = norm(df["regime_change_count"], invert=True)

    df["stability_score"] = (
        0.35 * df["n_vol"] +
        0.30 * df["n_cons"] +
        0.25 * df["n_trend"] +
        0.10 * df["n_reg"]
    ).round(4)

    df = df.sort_values("stability_score", ascending=False)
    logger.info(f"Stability scores computed: mean={df['stability_score'].mean():.3f}")
    return df


def build_base_model(efw_df):
    """
    Fit a simple linear base model on the clean period (2013-2019)
    for each country using major stable economies.
    Returns a dict {iso3: {'slope': float, 'intercept': float, 'r2': float}}
    """
    clean = efw_df[
        (efw_df["year"] >= CLEAN_PERIOD_START) &
        (efw_df["year"] <= CLEAN_PERIOD_END)
    ].copy()

    base_models = {}
    for (country, iso3), grp in clean.groupby(["country", "iso3"]):
        grp = grp.sort_values("year").dropna(subset=["overall_score"])
        if len(grp) < 4:
            continue
        x = grp["year"].values - CLEAN_PERIOD_START
        y = grp["overall_score"].values
        slope, intercept, r, _, _ = stats.linregress(x, y)
        base_models[iso3] = {
            "slope":     round(slope, 5),
            "intercept": round(intercept, 3),
            "r2":        round(r**2, 4),
            "country":   country,
        }

    logger.info(f"Base models fitted for {len(base_models)} countries")
    return base_models


def compute_post_crisis_deviation(efw_df, base_models):
    """
    For each country, measure how far the post-2019 EFW trajectory
    deviates from the clean-period base model.
    Positive deviation = better than expected (structural improvement)
    Negative deviation = worse than expected (structural deterioration)
    """
    post = efw_df[efw_df["year"] > CLEAN_PERIOD_END].copy()
    deviations = []

    for (country, iso3), grp in post.groupby(["country", "iso3"]):
        if iso3 not in base_models:
            continue
        model = base_models[iso3]
        grp = grp.sort_values("year").dropna(subset=["overall_score"])

        predicted = model["intercept"] + model["slope"] * (
            grp["year"].values - CLEAN_PERIOD_START
        )
        actual    = grp["overall_score"].values
        deviation = float(np.mean(actual - predicted))

        deviations.append({
            "iso3":                  iso3,
            "country":               country,
            "base_model_r2":         model["r2"],
            "post_crisis_deviation": round(deviation, 4),
            "deviation_direction":   "above" if deviation > 0 else "below",
        })

    df = pd.DataFrame(deviations)
    logger.info(
        f"Post-crisis deviations: "
        f"{(df['post_crisis_deviation'] > 0).sum()} above baseline, "
        f"{(df['post_crisis_deviation'] <= 0).sum()} below baseline"
    )
    return df


def run_module_c(save=True):
    logger.info("=== Stage 4 Module C: Policy Volatility Scorer ===")

    efw_df = load_efw_data()

    # Compute volatility metrics
    logger.info("Computing policy volatility metrics...")
    volatility_df = compute_policy_volatility(efw_df)
    stability_df  = score_policy_stability(volatility_df)

    logger.info("\nTop 10 most stable reformers:")
    logger.info(f"\n{stability_df[['country','current_efw_score','policy_volatility','classification','stability_score']].head(10).to_string()}")

    # Build clean-period base model
    logger.info("\nFitting clean-period base models (2013-2019)...")
    base_models = build_base_model(efw_df)

    # Compute post-crisis deviations
    logger.info("Computing post-crisis deviations...")
    deviations_df = compute_post_crisis_deviation(efw_df, base_models)

    # Merge stability + deviations
    combined = stability_df.merge(
        deviations_df[["iso3","base_model_r2","post_crisis_deviation","deviation_direction"]],
        on="iso3", how="left"
    )

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUTS_DIR / "policy_volatility_scores.csv", index=False)
        # Save base model summary
        base_df = pd.DataFrame(base_models).T
        base_df.to_csv(OUTPUTS_DIR / "base_models.csv")
        logger.info(f"Saved policy_volatility_scores.csv and base_models.csv")

    return combined, base_models


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    )
    run_module_c()
