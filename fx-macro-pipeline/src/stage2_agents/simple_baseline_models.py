"""
simple_baseline_models.py
=========================

Two-parameter baseline forecasters for daily FX returns. Built to test
whether simple linear models achieve meaningful directional skill before
investing in spectral/coherence complexity.

DESIGN
------
Each model has the form:
    r_{t+1} = a * feature_1(t) + b * feature_2(t) + epsilon
where feature_1 and feature_2 are constructed strictly from data <= t,
and (a, b) are fit by OLS on training data only.

MODEL FAMILIES
--------------
    lag_k          : a*USD_{t-k+1} + b*r_t              k in {1,2,5,10}
    ma_k           : a*MA_k(USD)_t + b*r_t              k in {5,20,60}
    mean_revert_k  : a*(r_t - MA_k(r)_t) + b*USD_{t-1}  k in {20,60}
    ar_usd         : a*r_t + b*USD_{t-1}

Total: 10 model variants per currency.

LEAK CONTROLS (same as spectral_agent_v2)
-----------------------------------------
    - USD index is leave-one-out for majors
    - All features at time t use only data <= t
    - Target is r_{t+1} (next-day return); we shift target by -1
    - OLS fits on training set only
    - Test-set predictions use frozen (a, b) from training

MODEL SELECTION
---------------
For each currency:
    1. Fit all 10 variants on training data
    2. Pick the variant with best TRAINING IC (this is the choice that
       gets to "see" training data only)
    3. Report TEST metrics for that variant — this is the honest number
       under cross-validation discipline
    4. Also report median across variants — robust against cherry-picking

METRICS
-------
Same as v2: accuracy, IC, Sharpe, turnover, time-in-market (=1 for these
models since they always have a position).

OUTPUTS
-------
    baseline_results.csv          — per-currency winning model + test metrics
    baseline_all_variants.csv     — all 10 metrics per currency (for audit)
    baseline_predictions.parquet  — winning-model signals on test set
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"

TRAIN_FRAC = 0.6
MAJOR_CURRENCIES = ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK"]
MIN_OBS_FOR_CURRENCY = 500   # enough for any MA window + train/test
MAX_MA_WINDOW = 60           # largest moving-average window we use


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    return pd.read_parquet(PROCESSED_DIR / "fx_log_returns.parquet")


def train_test_split(log_returns):
    n = len(log_returns)
    split_idx = int(n * TRAIN_FRAC)
    split_date = log_returns.index[split_idx]
    return log_returns.iloc[:split_idx], log_returns.iloc[split_idx:], split_date


def build_usd_index(log_returns, exclude=None):
    available = [c for c in MAJOR_CURRENCIES if c in log_returns.columns]
    if exclude in available:
        available = [c for c in available if c != exclude]
    if len(available) < 4:
        raise ValueError(f"Too few majors after LOO: {available}")
    return log_returns[available].mean(axis=1)


# ---------------------------------------------------------------------------
# Feature builders — all strictly causal (use only data <= t)
# ---------------------------------------------------------------------------

def feature_lag_usd(usd, k):
    """USD return k days ago: USD_{t-k+1}. shift(k-1) so feature at row t is
    the USD value k-1 days BEFORE t.  For k=1 this is the same-day USD return
    (still causal w.r.t. the t->t+1 target)."""
    return usd.shift(k - 1)


def feature_ma_usd(usd, k):
    """Moving average of USD over the last k days, ending at t (inclusive)."""
    return usd.rolling(window=k, min_periods=k).mean()


def feature_ma_target(target, k):
    """Moving average of target over the last k days, ending at t (inclusive)."""
    return target.rolling(window=k, min_periods=k).mean()


def feature_target_lag(target):
    """target_t — used as the b-feature in several models."""
    return target


# ---------------------------------------------------------------------------
# Model variant definitions
# ---------------------------------------------------------------------------

def build_features_for_variant(variant, usd, target):
    """
    Return (f1, f2) — two pd.Series aligned on target's index, each strictly
    causal (computed from data up to and including t).
    """
    if variant.startswith("lag_"):
        k = int(variant.split("_")[1])
        f1 = feature_lag_usd(usd, k)
        f2 = feature_target_lag(target)
    elif variant.startswith("ma_"):
        k = int(variant.split("_")[1])
        f1 = feature_ma_usd(usd, k)
        f2 = feature_target_lag(target)
    elif variant.startswith("mean_revert_"):
        k = int(variant.split("_")[2])
        f1 = target - feature_ma_target(target, k)
        f2 = feature_lag_usd(usd, 1)
    elif variant == "ar_usd":
        f1 = feature_target_lag(target)
        f2 = feature_lag_usd(usd, 1)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return f1, f2


VARIANTS = (
    ["lag_1", "lag_2", "lag_5", "lag_10"]
    + ["ma_5", "ma_20", "ma_60"]
    + ["mean_revert_20", "mean_revert_60"]
    + ["ar_usd"]
)


# ---------------------------------------------------------------------------
# OLS fit + prediction
# ---------------------------------------------------------------------------

def fit_ols_2param(f1, f2, target_next):
    """
    Fit r_{t+1} = a*f1(t) + b*f2(t) by OLS on rows where all three are
    finite. Returns (a, b) or (nan, nan) if not enough data.
    """
    df = pd.DataFrame({"f1": f1, "f2": f2, "y": target_next}).dropna()
    if len(df) < 60:
        return np.nan, np.nan
    X = df[["f1", "f2"]].to_numpy()
    y = df["y"].to_numpy()
    try:
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(coefs[0]), float(coefs[1])
    except Exception:
        return np.nan, np.nan


def predict(f1, f2, a, b):
    """Compute predicted r_{t+1} = a*f1(t) + b*f2(t) at every row."""
    return a * f1 + b * f2


# ---------------------------------------------------------------------------
# Metrics (matches v2's signature)
# ---------------------------------------------------------------------------

def evaluate_predictions(predictions, target_next):
    """
    predictions: pd.Series of real-valued forecasts indexed by date
    target_next: pd.Series of realized r_{t+1} indexed by date
    """
    common = predictions.index.intersection(target_next.index)
    p = predictions.loc[common]
    y = target_next.loc[common]
    valid = p.notna() & y.notna()
    p = p[valid]
    y = y[valid]
    if len(p) < 30:
        return None

    # Directional signal: sign of prediction
    signal = np.sign(p)
    strategy_ret = signal * y

    accuracy = float((np.sign(p) == np.sign(y)).mean())
    # IC: rank correlation between predicted and realized returns
    try:
        ic = float(p.rank().corr(y.rank()))
    except Exception:
        ic = np.nan

    mean_r = float(strategy_ret.mean())
    std_r = float(strategy_ret.std())
    sharpe = float((mean_r / std_r) * np.sqrt(252)) if std_r > 0 else np.nan

    # Turnover: fraction of days where sign changes
    turnover = float((signal.diff().abs() > 0).mean())
    time_in_market = 1.0   # baseline models always have a position

    return {
        "accuracy": accuracy,
        "ic": ic,
        "sharpe": sharpe,
        "turnover": turnover,
        "time_in_market": time_in_market,
        "n_predictions": int(len(p)),
    }


# ---------------------------------------------------------------------------
# Per-currency pipeline
# ---------------------------------------------------------------------------

def process_currency(currency, log_returns, split_date):
    """
    Returns:
        winner_row: dict with selected model + its test metrics
        all_variants: list of dicts, one per variant, with both train + test metrics
        signal_winner: pd.Series of the winning model's test-set signal
    """
    if currency not in log_returns.columns:
        return None, None, None
    target = log_returns[currency].dropna()
    if len(target) < MIN_OBS_FOR_CURRENCY:
        return None, None, None

    usd = build_usd_index(log_returns, exclude=currency)

    # Align on a common index
    common = target.index.intersection(usd.index)
    target = target.loc[common]
    usd = usd.loc[common]

    # Target shifted by -1: target_next at row t = target_{t+1}
    target_next = target.shift(-1)

    # Train/test masks
    train_mask = target.index < split_date
    test_mask = target.index >= split_date

    variant_records = []
    train_ic_by_variant = {}
    test_predictions_by_variant = {}

    for variant in VARIANTS:
        try:
            f1, f2 = build_features_for_variant(variant, usd, target)
        except Exception as e:
            logger.warning(f"  {currency}/{variant} feature build failed: {e}")
            continue

        # Fit on training data only
        f1_train = f1[train_mask]
        f2_train = f2[train_mask]
        y_train = target_next[train_mask]
        a, b = fit_ols_2param(f1_train, f2_train, y_train)
        if np.isnan(a) or np.isnan(b):
            continue

        # Predict on full series, then evaluate train and test separately
        preds_all = predict(f1, f2, a, b)
        preds_train = preds_all[train_mask]
        preds_test = preds_all[test_mask]

        train_metrics = evaluate_predictions(preds_train, target_next[train_mask])
        test_metrics = evaluate_predictions(preds_test, target_next[test_mask])
        if train_metrics is None or test_metrics is None:
            continue

        rec = {
            "currency": currency,
            "variant": variant,
            "a": a,
            "b": b,
            "train_accuracy": train_metrics["accuracy"],
            "train_ic":       train_metrics["ic"],
            "train_sharpe":   train_metrics["sharpe"],
            "test_accuracy":  test_metrics["accuracy"],
            "test_ic":        test_metrics["ic"],
            "test_sharpe":    test_metrics["sharpe"],
            "test_turnover":  test_metrics["turnover"],
            "n_test":         test_metrics["n_predictions"],
        }
        variant_records.append(rec)
        train_ic_by_variant[variant] = train_metrics["ic"]
        test_predictions_by_variant[variant] = preds_test

    if not variant_records:
        return None, None, None

    # Model selection: pick the variant with best TRAINING IC.
    # If multiple variants tie or all are nan, fall back to first variant.
    if train_ic_by_variant:
        best_variant = max(
            train_ic_by_variant,
            key=lambda v: (
                train_ic_by_variant[v] if not np.isnan(train_ic_by_variant[v])
                else -np.inf
            ),
        )
    else:
        best_variant = variant_records[0]["variant"]

    winner = next(r for r in variant_records if r["variant"] == best_variant)
    winner_signal = np.sign(test_predictions_by_variant[best_variant])

    # Also compute robust aggregates across all variants
    test_ics = [r["test_ic"] for r in variant_records if not np.isnan(r["test_ic"])]
    test_sharpes = [r["test_sharpe"] for r in variant_records if not np.isnan(r["test_sharpe"])]
    test_accs = [r["test_accuracy"] for r in variant_records if not np.isnan(r["test_accuracy"])]
    winner["median_test_ic_across_variants"] = float(np.median(test_ics)) if test_ics else np.nan
    winner["median_test_sharpe_across_variants"] = float(np.median(test_sharpes)) if test_sharpes else np.nan
    winner["median_test_accuracy_across_variants"] = float(np.median(test_accs)) if test_accs else np.nan
    winner["n_variants_fit"] = len(variant_records)

    return winner, variant_records, winner_signal


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(save=True, currencies=None):
    logger.info("=== Simple Baseline Models (two-parameter OLS) ===")
    logger.info(f"Variants: {VARIANTS}")

    log_returns = load_data()
    train, test, split_date = train_test_split(log_returns)
    logger.info(
        f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} rows)"
    )
    logger.info(
        f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} rows)"
    )

    if currencies is None:
        currencies = [c for c in log_returns.columns if c != "USD"]

    logger.info(f"Processing {len(currencies)} currencies...")

    winners = []
    all_variant_rows = []
    all_signals = {}

    for i, ccy in enumerate(currencies):
        if i % 25 == 0:
            logger.info(f"  Progress: {i}/{len(currencies)}")
        try:
            w, variants, sig = process_currency(ccy, log_returns, split_date)
        except Exception as e:
            logger.warning(f"  {ccy} failed: {e}")
            continue
        if w is None:
            continue
        winners.append(w)
        all_variant_rows.extend(variants)
        if sig is not None:
            all_signals[ccy] = sig

    if not winners:
        raise RuntimeError("No currencies produced results.")

    winners_df = pd.DataFrame(winners).set_index("currency")
    winners_df = winners_df.sort_values("test_sharpe", ascending=False)

    variants_df = pd.DataFrame(all_variant_rows)

    # Aggregate summary
    logger.info("\n=== Baseline aggregate results ===")
    logger.info(f"Currencies evaluated: {len(winners_df)}")
    logger.info(
        f"WINNER-MODEL test accuracy   mean={winners_df['test_accuracy'].mean():.4f}  "
        f"median={winners_df['test_accuracy'].median():.4f}"
    )
    logger.info(
        f"WINNER-MODEL test IC         mean={winners_df['test_ic'].mean():.4f}  "
        f"median={winners_df['test_ic'].median():.4f}"
    )
    logger.info(
        f"WINNER-MODEL test Sharpe     mean={winners_df['test_sharpe'].mean():.4f}  "
        f"median={winners_df['test_sharpe'].median():.4f}"
    )
    logger.info(
        f"MEDIAN-OF-VARIANTS test acc   mean={winners_df['median_test_accuracy_across_variants'].mean():.4f}"
    )
    logger.info(
        f"MEDIAN-OF-VARIANTS test IC    mean={winners_df['median_test_ic_across_variants'].mean():.4f}"
    )
    logger.info(
        f"Pct of currencies with winner test Sharpe > 0: "
        f"{(winners_df['test_sharpe'] > 0).mean():.4f}"
    )

    # Which variant wins most often?
    variant_counts = winners_df["variant"].value_counts()
    logger.info("\nVariant win counts (which model gets picked by training IC):")
    for v, n in variant_counts.items():
        logger.info(f"  {v}: {n}")

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        winners_df.to_csv(OUTPUTS_DIR / "baseline_results.csv")
        variants_df.to_csv(OUTPUTS_DIR / "baseline_all_variants.csv", index=False)
        if all_signals:
            signals_df = pd.DataFrame(all_signals)
            signals_df.to_parquet(OUTPUTS_DIR / "baseline_predictions.parquet")
        logger.info("Saved: baseline_results.csv, baseline_all_variants.csv, "
                    "baseline_predictions.parquet")

    return winners_df, variants_df, all_signals


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    )
    run()
