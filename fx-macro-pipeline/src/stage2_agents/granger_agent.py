import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
from pathlib import Path

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "outputs"
TRAIN_FRAC    = 0.6
MAX_LAG       = 10    # maximum lag order to test
FDR_ALPHA     = 0.05  # significance level after BH correction


def load_data():
    log_returns   = pd.read_parquet(PROCESSED_DIR / "fx_log_returns.parquet")
    regime_labels = pd.read_parquet(PROCESSED_DIR / "regime_labels.parquet")
    return log_returns, regime_labels


def train_test_split(log_returns):
    n = len(log_returns)
    split_idx = int(n * TRAIN_FRAC)
    train = log_returns.iloc[:split_idx]
    test  = log_returns.iloc[split_idx:]
    return train, test


def build_usd_index(log_returns, major_currencies=None):
    if major_currencies is None:
        major_currencies = ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK"]
    available = [c for c in major_currencies if c in log_returns.columns]
    return log_returns[available].mean(axis=1)


def select_lag_order(bivariate_df, max_lag=MAX_LAG):
    """
    Fit a VAR model and select lag order via AIC.
    bivariate_df: DataFrame with exactly 2 columns [usd_index, target_currency]
    """
    clean = bivariate_df.dropna()
    if len(clean) < max_lag * 4:
        return 1
    try:
        model = VAR(clean)
        result = model.select_order(maxlags=max_lag)
        lag = result.aic
        # select_order returns a dict of {lag: aic_value} — pick the min
        if isinstance(lag, dict):
            lag = min(lag, key=lag.get)
        return max(1, int(lag))
    except Exception:
        return 2


def run_granger_test(usd_series, target_series, max_lag=MAX_LAG):
    """
    Test whether usd_series Granger-causes target_series.
    Returns the best p-value across lag orders 1..max_lag,
    and the lag at which it occurs.
    """
    combined = pd.DataFrame({
        "usd":    usd_series,
        "target": target_series,
    }).dropna()

    if len(combined) < max_lag * 4:
        return None, None, None

    try:
        # grangercausalitytests returns {lag: [test_results, ...]}
        # We use the F-test (ssr_ftest) p-value
        results = grangercausalitytests(
            combined[["target", "usd"]],  # note: [effect, cause] order
            maxlag=max_lag,
            verbose=False,
        )
        pvals = {
            lag: res[0]["ssr_ftest"][1]
            for lag, res in results.items()
        }
        best_lag = min(pvals, key=pvals.get)
        best_pval = pvals[best_lag]
        return best_pval, best_lag, pvals

    except Exception as e:
        logger.warning(f"Granger test failed: {e}")
        return None, None, None


def compute_impulse_response(usd_series, target_series, lag, steps=20):
    """
    Fit a bivariate VAR at the given lag order and extract the
    impulse response of target to a unit shock in usd.
    Returns array of length `steps`.
    """
    combined = pd.DataFrame({
        "usd":    usd_series,
        "target": target_series,
    }).dropna()

    try:
        model  = VAR(combined)
        fitted = model.fit(lag)
        irf    = fitted.irf(steps)
        # irf.irfs shape: (steps+1, n_vars, n_vars)
        # [step, response_var_idx, shock_var_idx]
        # usd=0, target=1 -> response of target (1) to shock in usd (0)
        response = irf.irfs[:, 1, 0]
        return response
    except Exception as e:
        logger.warning(f"IRF failed: {e}")
        return None


def run_granger_agent(train):
    """
    For each currency, run Granger causality test (USD → currency).
    Apply Benjamini-Hochberg FDR correction across all tests.
    For significant currencies, compute impulse response functions.

    Returns summary DataFrame with p-values, significance flags,
    IRF direction, and directional signal.
    """
    usd_index = build_usd_index(train)
    raw_results = []

    currencies = [c for c in train.columns if c != "USD"]
    logger.info(f"Running Granger tests for {len(currencies)} currencies...")

    for i, col in enumerate(currencies):
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(currencies)}")

        series = train[col].dropna()
        common = usd_index.index.intersection(series.index)
        if len(common) < 60:
            continue

        pval, best_lag, all_pvals = run_granger_test(
            usd_index.loc[common],
            series.loc[common],
            max_lag=MAX_LAG,
        )
        if pval is None:
            continue

        raw_results.append({
            "currency": col,
            "best_pval": pval,
            "best_lag":  best_lag,
        })

    if not raw_results:
        raise RuntimeError("No Granger results computed.")

    df = pd.DataFrame(raw_results).set_index("currency")

    # Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = multipletests(
        df["best_pval"].values,
        alpha=FDR_ALPHA,
        method="fdr_bh",
    )
    df["pval_corrected"] = pvals_corrected
    df["significant"]    = reject

    logger.info(
        f"BH correction: {reject.sum()}/{len(reject)} currencies "
        f"significant at FDR={FDR_ALPHA}"
    )

    # Compute IRF for significant currencies
    irf_direction = {}
    irf_peak_days = {}

    significant = df[df["significant"]].index.tolist()
    logger.info(f"Computing IRFs for {len(significant)} significant currencies...")

    for col in significant:
        series = train[col].dropna()
        common = usd_index.index.intersection(series.index)
        lag    = int(df.loc[col, "best_lag"])
        irf    = compute_impulse_response(
            usd_index.loc[common],
            series.loc[common],
            lag=lag,
            steps=20,
        )
        if irf is not None:
            # Direction: sign of cumulative IRF response
            cumulative = float(np.sum(irf))
            irf_direction[col] = "inverse" if cumulative < 0 else "aligned"
            # Peak response day
            irf_peak_days[col] = int(np.argmax(np.abs(irf)))

    df["irf_direction"] = df.index.map(irf_direction)
    df["irf_peak_day"]  = df.index.map(irf_peak_days)

    df = df.sort_values("pval_corrected")
    logger.info("Granger agent complete.")
    return df


def generate_directional_signals(granger_results, test, usd_index_full):
    """
    Generate binary directional predictions on the test set.
    Only uses currencies that passed BH correction.
    """
    predictions = {}
    usd_test = usd_index_full.loc[test.index].dropna()
    significant = granger_results[granger_results["significant"]]

    for currency, row in significant.iterrows():
        if currency not in test.columns:
            continue
        lag        = int(row["best_lag"])
        usd_lagged = usd_test.shift(lag)
        usd_signal = np.sign(usd_lagged.dropna())
        direction  = row.get("irf_direction", "aligned")
        if direction == "inverse":
            predictions[currency] = -usd_signal
        else:
            predictions[currency] = usd_signal

    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.reindex(usd_test.index)
    return pred_df


def run(save=True):
    logger.info("=== Granger / VAR Agent ===")
    log_returns, _ = load_data()
    train, test    = train_test_split(log_returns)

    granger_results = run_granger_agent(train)

    usd_index_full  = build_usd_index(log_returns)
    predictions     = generate_directional_signals(granger_results, test, usd_index_full)

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        granger_results.to_csv(OUTPUTS_DIR / "granger_results.csv")
        predictions.to_parquet(OUTPUTS_DIR / "granger_predictions.parquet")
        logger.info("Saved granger_results.csv and granger_predictions.parquet")

    return granger_results, predictions, test


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    )
    run()
