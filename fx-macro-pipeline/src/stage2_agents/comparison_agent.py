import numpy as np
import pandas as pd
import logging
from sklearn.metrics import matthews_corrcoef
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"


def load_predictions():
    spectral_preds = pd.read_parquet(OUTPUTS_DIR / "spectral_predictions.parquet")
    granger_preds  = pd.read_parquet(OUTPUTS_DIR / "granger_predictions.parquet")
    return spectral_preds, granger_preds


def load_actuals(test_log_returns):
    """Convert log returns to binary direction: +1 if positive, -1 if negative."""
    return np.sign(test_log_returns).replace(0, np.nan)


def directional_accuracy(predictions, actuals):
    """
    Compute directional accuracy per currency.
    predictions: DataFrame of +1/-1 signals
    actuals:     DataFrame of +1/-1 actual directions
    Returns Series of accuracy scores indexed by currency.
    """
    scores = {}
    common_cols = predictions.columns.intersection(actuals.columns)
    for col in common_cols:
        pred = predictions[col].dropna()
        act  = actuals[col].dropna()
        aligned = pred.index.intersection(act.index)
        if len(aligned) < 10:
            continue
        p = pred.loc[aligned]
        a = act.loc[aligned]
        scores[col] = float((p == a).mean())
    return pd.Series(scores, name="directional_accuracy")


def mcc_score(predictions, actuals):
    """
    Compute Matthews Correlation Coefficient per currency.
    More informative than accuracy for imbalanced signals.
    """
    scores = {}
    common_cols = predictions.columns.intersection(actuals.columns)
    for col in common_cols:
        pred = predictions[col].dropna()
        act  = actuals[col].dropna()
        aligned = pred.index.intersection(act.index)
        if len(aligned) < 10:
            continue
        p = pred.loc[aligned].values
        a = act.loc[aligned].values
        # MCC requires labels — map +1/-1 to 1/0
        p_bin = (p > 0).astype(int)
        a_bin = (a > 0).astype(int)
        try:
            scores[col] = float(matthews_corrcoef(a_bin, p_bin))
        except Exception:
            scores[col] = np.nan
    return pd.Series(scores, name="mcc")


def implied_sharpe(predictions, actual_returns):
    """
    Compute the Sharpe ratio of a hypothetical strategy that goes long/short
    based on the signal, scaled by actual log returns.
    """
    scores = {}
    common_cols = predictions.columns.intersection(actual_returns.columns)
    for col in common_cols:
        pred    = predictions[col].dropna()
        returns = actual_returns[col].dropna()
        aligned = pred.index.intersection(returns.index)
        if len(aligned) < 10:
            continue
        strategy_returns = pred.loc[aligned] * returns.loc[aligned]
        mean_r = strategy_returns.mean()
        std_r  = strategy_returns.std()
        if std_r == 0:
            continue
        # Annualise: daily data, ~252 trading days
        sharpe = float((mean_r / std_r) * np.sqrt(252))
        scores[col] = round(sharpe, 4)
    return pd.Series(scores, name="implied_sharpe")


def build_report(predictions, actuals, actual_returns, model_name):
    """Build a combined evaluation DataFrame for one model."""
    da   = directional_accuracy(predictions, actuals)
    mcc  = mcc_score(predictions, actuals)
    shp  = implied_sharpe(predictions, actual_returns)
    report = pd.concat([da, mcc, shp], axis=1)
    report.columns = [
        f"{model_name}_dir_accuracy",
        f"{model_name}_mcc",
        f"{model_name}_sharpe",
    ]
    return report


def compare(spectral_preds, granger_preds, test_log_returns):
    """
    Run full comparison between spectral and Granger predictions.
    Returns combined report and a winner summary.
    """
    actuals        = load_actuals(test_log_returns)
    spectral_report = build_report(spectral_preds, actuals, test_log_returns, "spectral")
    granger_report  = build_report(granger_preds,  actuals, test_log_returns, "granger")

    # Merge on common currencies
    common = spectral_report.index.intersection(granger_report.index)
    combined = pd.concat(
        [spectral_report.loc[common], granger_report.loc[common]], axis=1
    )

    # Determine winner per currency per metric
    combined["dir_winner"] = np.where(
        combined["spectral_dir_accuracy"] >= combined["granger_dir_accuracy"],
        "spectral", "granger"
    )
    combined["mcc_winner"] = np.where(
        combined["spectral_mcc"] >= combined["granger_mcc"],
        "spectral", "granger"
    )
    combined["sharpe_winner"] = np.where(
        combined["spectral_sharpe"] >= combined["granger_sharpe"],
        "spectral", "granger"
    )

    # Aggregate summary
    summary = {
        "spectral_mean_accuracy": combined["spectral_dir_accuracy"].mean(),
        "granger_mean_accuracy":  combined["granger_dir_accuracy"].mean(),
        "spectral_mean_mcc":      combined["spectral_mcc"].mean(),
        "granger_mean_mcc":       combined["granger_mcc"].mean(),
        "spectral_mean_sharpe":   combined["spectral_sharpe"].mean(),
        "granger_mean_sharpe":    combined["granger_sharpe"].mean(),
        "spectral_dir_wins":      (combined["dir_winner"] == "spectral").sum(),
        "granger_dir_wins":       (combined["dir_winner"] == "granger").sum(),
        "spectral_mcc_wins":      (combined["mcc_winner"] == "spectral").sum(),
        "granger_mcc_wins":       (combined["mcc_winner"] == "granger").sum(),
        "spectral_sharpe_wins":   (combined["sharpe_winner"] == "spectral").sum(),
        "granger_sharpe_wins":    (combined["sharpe_winner"] == "granger").sum(),
    }

    logger.info("=== Comparison Summary ===")
    for k, v in summary.items():
        logger.info(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")

    # Overall recommendation
    spectral_wins = summary["spectral_dir_wins"] + summary["spectral_mcc_wins"] + summary["spectral_sharpe_wins"]
    granger_wins  = summary["granger_dir_wins"]  + summary["granger_mcc_wins"]  + summary["granger_sharpe_wins"]
    recommendation = "spectral" if spectral_wins >= granger_wins else "granger"
    logger.info(f"  Recommended model: {recommendation.upper()}")
    summary["recommendation"] = recommendation

    return combined, summary


def run(test_log_returns, save=True):
    logger.info("=== Comparison Agent ===")
    spectral_preds, granger_preds = load_predictions()
    combined, summary = compare(spectral_preds, granger_preds, test_log_returns)

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUTS_DIR / "comparison_report.csv")
        pd.Series(summary).to_csv(OUTPUTS_DIR / "comparison_summary.csv")
        logger.info("Saved comparison_report.csv and comparison_summary.csv")

    return combined, summary
