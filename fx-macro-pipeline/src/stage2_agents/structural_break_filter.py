import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "outputs"
RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"

# Known structural break periods — dates where global shocks
# distort the underlying currency-USD relationship
STRUCTURAL_BREAKS = [
    {"name": "GFC",        "start": "2008-09-01", "end": "2009-06-30"},
    {"name": "EUR_crisis", "start": "2011-06-01", "end": "2012-12-31"},
    {"name": "COVID",      "start": "2020-02-01", "end": "2021-06-30"},
    {"name": "Ukraine_war","start": "2022-02-01", "end": "2022-12-31"},
]

# Clean base period
CLEAN_START = "2013-01-01"
CLEAN_END   = "2019-12-31"


def load_log_returns():
    return pd.read_parquet(PROCESSED_DIR / "fx_log_returns.parquet")


def load_base_model_scores():
    """Load Module C post-crisis deviation scores."""
    path = OUTPUTS_DIR / "policy_volatility_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_break_dummies(date_index):
    """
    Build a DataFrame of dummy variables for each structural break period.
    Rows = dates, columns = break names. Value = 1 during break, 0 otherwise.
    """
    dummies = pd.DataFrame(index=date_index)
    for brk in STRUCTURAL_BREAKS:
        col = f"dummy_{brk['name']}"
        dummies[col] = 0
        mask = (date_index >= brk["start"]) & (date_index <= brk["end"])
        dummies.loc[mask, col] = 1
    return dummies


def get_clean_period_mask(date_index):
    """Return boolean mask for the clean base period."""
    return (date_index >= CLEAN_START) & (date_index <= CLEAN_END)


def fit_clean_period_model(series, date_index):
    """
    Fit a simple AR(1) model on the clean period only.
    Returns (intercept, ar1_coef, residual_std) on clean period.
    """
    clean_mask = get_clean_period_mask(date_index)
    clean_series = series[clean_mask].dropna()

    if len(clean_series) < 60:
        return None

    y = clean_series.values[1:]
    x = clean_series.values[:-1]

    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None

    slope, intercept, r, _, se = stats.linregress(x, y)


    residuals = y - (intercept + slope * x)
    residual_std = float(np.std(residuals))

    return {
        "intercept":    intercept,
        "ar1_coef":     slope,
        "r2":           r**2,
        "residual_std": residual_std,
        "n_obs":        len(clean_series),
    }


def compute_break_adjusted_returns(log_returns):
    """
    For each currency, compute break-period-adjusted log returns by:
    1. Fitting an AR(1) model on the clean period
    2. Computing residuals for the full series
    3. Flagging observations during structural break periods
    4. Returning original returns with a break_flag column

    Returns dict: {currency: {'returns': series, 'break_flags': series,
                               'clean_model': dict, 'break_impact': dict}}
    """
    dummies = build_break_dummies(log_returns.index)
    results = {}

    for currency in log_returns.columns:
        series = log_returns[currency].dropna()

        # Fit clean period model
        model = fit_clean_period_model(series, series.index)
        if model is None:
            continue

        # Compute residuals for full series
        y = series.values[1:]
        x = series.values[:-1]
        idx = series.index[1:]

        predicted = model["intercept"] + model["ar1_coef"] * x
        residuals = pd.Series(y - predicted, index=idx)

        # Measure break impact: mean absolute residual during each break
        # vs mean absolute residual during clean period
        clean_mask = get_clean_period_mask(idx)
        clean_mad  = float(residuals[clean_mask].abs().mean())

        break_impacts = {}
        for brk in STRUCTURAL_BREAKS:
            brk_mask = (idx >= brk["start"]) & (idx <= brk["end"])
            if brk_mask.sum() > 0:
                brk_mad = float(residuals[brk_mask].abs().mean())
                break_impacts[brk["name"]] = round(brk_mad / (clean_mad + 1e-10), 3)

        # Break flag: 1 if any dummy is active
        break_flag = dummies.reindex(series.index).fillna(0).max(axis=1)

        results[currency] = {
            "returns":       series,
            "break_flags":   break_flag,
            "residuals":     residuals,
            "clean_model":   model,
            "break_impacts": break_impacts,
        }

    logger.info(
        f"Break-adjusted models fitted for {len(results)} currencies"
    )
    return results


def get_clean_only_returns(log_returns):
    """
    Return a version of log_returns with structural break periods
    replaced by NaN. Used for fitting models on clean data only.
    """
    clean = log_returns.copy()
    for brk in STRUCTURAL_BREAKS:
        mask = (log_returns.index >= brk["start"]) & \
               (log_returns.index <= brk["end"])
        clean.loc[mask] = np.nan
    logger.info(
        f"Clean-only returns: {clean.notna().sum().mean():.0f} avg "
        f"valid obs per currency (vs {len(log_returns)} total)"
    )
    return clean


def summarise_break_sensitivity(break_adjusted_results):
    """
    Summarise which currencies are most sensitive to structural breaks.
    Returns DataFrame sorted by COVID sensitivity (most disrupted first).
    """
    records = []
    for currency, data in break_adjusted_results.items():
        row = {"currency": currency}
        row.update({
            f"impact_{k}": v
            for k, v in data["break_impacts"].items()
        })
        row["clean_model_r2"] = data["clean_model"]["r2"]
        row["clean_residual_std"] = data["clean_model"]["residual_std"]
        records.append(row)

    df = pd.DataFrame(records).set_index("currency")

    if "impact_COVID" in df.columns:
        df = df.sort_values("impact_COVID", ascending=False)

    logger.info(
        f"Break sensitivity summary: {len(df)} currencies\n"
        f"Most COVID-sensitive: {df.index[:5].tolist()}"
    )
    return df


def run_structural_break_filter(save=True):
    """Main entry point for structural break analysis."""
    logger.info("=== Structural Break Filter ===")

    log_returns = load_log_returns()

    logger.info("Fitting clean-period AR(1) models and computing break impacts...")
    break_results = compute_break_adjusted_returns(log_returns)

    logger.info("Computing clean-only return series...")
    clean_returns = get_clean_only_returns(log_returns)

    logger.info("Summarising break sensitivity...")
    sensitivity = summarise_break_sensitivity(break_results)

    logger.info(f"\nTop 10 most COVID-sensitive currencies:")
    if "impact_COVID" in sensitivity.columns:
        impact_cols = [c for c in sensitivity.columns if c.startswith('impact_')]
        display_cols = impact_cols[:3] + ['clean_model_r2']
        display_cols = [c for c in display_cols if c in sensitivity.columns]
        logger.info(f"\n{sensitivity[display_cols].head(10).to_string()}")

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        sensitivity.to_csv(OUTPUTS_DIR / "break_sensitivity.csv")
        clean_returns.to_parquet(PROCESSED_DIR / "fx_log_returns_clean.parquet")
        logger.info(
            "Saved break_sensitivity.csv and fx_log_returns_clean.parquet"
        )

    return break_results, clean_returns, sensitivity


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    )
    run_structural_break_filter()
