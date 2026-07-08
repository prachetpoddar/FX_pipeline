"""
spectral_agent_v2.py
====================

A rolling-window, strictly-causal spectral forecaster for daily FX returns.

WHAT THIS DOES DIFFERENTLY FROM v1
----------------------------------
The original spectral_agent.py had a structural look-ahead bug: at test time
it used sign(usd_return_at_t) as the "prediction" for currency returns at the
SAME time t. That is not a forecast — it is contemporaneous correlation, and
the high Sharpe (1.89) was an artifact of this leak, amplified by the USD
index including the currencies it was being used to predict.

v2 fixes this by:
    1. Computing cross-PSD in a ROLLING window ending at time t (strictly
       causal — nothing from t+1 or later leaks in)
    2. Using the USD return from t-phase_lag_days to t (not t+1) to form the
       prediction for the t-to-t+1 return
    3. Building the USD index leave-one-out (for the 8 majors, each gets
       a version of the index that excludes itself)
    4. Gating signals by rolling coherence vs a per-currency threshold
       (computed on training data only) — low-coherence periods produce
       no position, not a random one
    5. Determining sign convention (aligned vs inverse) from training data
       only, then freezing it for the test set

METRICS
-------
Reports per-currency and aggregate:
    - Directional accuracy (hit rate, conditional on being in-position)
    - Information coefficient (rank corr between signal and next-day return)
    - Signal-weighted Sharpe (annualized)
    - Turnover (fraction of days with position changes)
    - Time-in-market (fraction of days with non-zero position)

All metrics are computed on signal × next-day return (strict t+1 forecast).
No transaction costs applied here — that is Layer 4 of the audit and
happens downstream.

INTERMEDIATE ARTIFACTS
----------------------
Saves the rolling coherence series, phase-lag series, and per-currency
threshold so you can inspect WHY each currency performed the way it did.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import csd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs"

# ---- Hyperparameters (set once, documented) --------------------------------
TRAIN_FRAC = 0.6
WINDOW = 256                     # rolling window length in trading days (~1yr)
NPERSEG = 128                    # Welch segment length within each window
COHERENCE_PCTILE = 70            # per-currency gate: fire only when rolling
                                 # coherence is above this percentile on train
MAJOR_CURRENCIES = ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK"]
MIN_OBS_FOR_CURRENCY = WINDOW + 252   # need a full window + ~1yr test space


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_data():
    log_returns = pd.read_parquet(PROCESSED_DIR / "fx_log_returns.parquet")
    return log_returns


def train_test_split(log_returns):
    n = len(log_returns)
    split_idx = int(n * TRAIN_FRAC)
    split_date = log_returns.index[split_idx]
    train = log_returns.iloc[:split_idx]
    test = log_returns.iloc[split_idx:]
    logger.info(
        f"Train: {train.index[0].date()} → {train.index[-1].date()} "
        f"({len(train)} rows)"
    )
    logger.info(
        f"Test:  {test.index[0].date()} → {test.index[-1].date()} "
        f"({len(test)} rows)"
    )
    return train, test, split_date


# ---------------------------------------------------------------------------
# USD index construction (leave-one-out for majors)
# ---------------------------------------------------------------------------

def build_usd_index(log_returns, exclude=None):
    """
    Build an equal-weighted USD strength index from majors.
    If `exclude` is one of the majors, drop it (leave-one-out).
    Rising index = stronger USD.
    """
    available = [c for c in MAJOR_CURRENCIES if c in log_returns.columns]
    if exclude in available:
        available = [c for c in available if c != exclude]
    if len(available) < 4:
        raise ValueError(f"Too few majors available after LOO: {available}")
    return log_returns[available].mean(axis=1)


# ---------------------------------------------------------------------------
# Single-window spectral feature extraction
# ---------------------------------------------------------------------------

def spectral_features_one_window(x, y, nperseg=NPERSEG):
    """
    Given two aligned numpy arrays of length W (USD index and target currency
    returns, both ending at the same time t), compute:
        - coherence at dominant cross-PSD frequency (normalized)
        - dominant period in days
        - phase lag in days (positive = target lags USD)

    Returns None if degenerate.
    """
    if len(x) < nperseg or len(y) < nperseg:
        return None
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None

    try:
        freqs, Sxy = csd(x, y, fs=1.0, nperseg=nperseg, window="hann")
    except Exception:
        return None

    magnitude = np.abs(Sxy)
    phase = np.angle(Sxy)

    # Exclude DC component
    if len(magnitude) < 2:
        return None
    idx = int(np.argmax(magnitude[1:]) + 1)
    dom_freq = float(freqs[idx])
    if dom_freq <= 0:
        return None

    # Normalized coherence: peak / mean (excluding DC)
    mean_mag = float(np.mean(magnitude[1:]) + 1e-10)
    coherence = float(magnitude[idx] / mean_mag)

    period_days = 1.0 / dom_freq
    phase_lag_days = float(phase[idx] / (2 * np.pi * dom_freq))

    return {
        "coherence": coherence,
        "dominant_period": period_days,
        "phase_lag": phase_lag_days,
    }


# ---------------------------------------------------------------------------
# Rolling feature extraction for one currency
# ---------------------------------------------------------------------------

def compute_rolling_features(usd_series, target_series, window=WINDOW,
                             nperseg=NPERSEG):
    """
    Slide a window of length `window` across the full common-index series.
    At each time t, compute spectral features using ONLY data from
    t-window+1 to t (strict causal).

    Returns a DataFrame indexed by date, columns:
        coherence, dominant_period, phase_lag
    """
    common = usd_series.index.intersection(target_series.index)
    x_full = usd_series.loc[common].values
    y_full = target_series.loc[common].values
    dates = common

    n = len(common)
    if n < window + 1:
        return None

    coherence = np.full(n, np.nan)
    period = np.full(n, np.nan)
    phase_lag = np.full(n, np.nan)

    # Compute at each t where we have a full window ending at t (inclusive).
    # Feature at index t is based on indices [t-window+1 .. t], so first
    # valid t is window-1.
    for t in range(window - 1, n):
        x_win = x_full[t - window + 1 : t + 1]
        y_win = y_full[t - window + 1 : t + 1]

        # Skip windows that contain NaNs
        if np.any(np.isnan(x_win)) or np.any(np.isnan(y_win)):
            continue

        feats = spectral_features_one_window(x_win, y_win, nperseg=nperseg)
        if feats is None:
            continue

        coherence[t] = feats["coherence"]
        period[t] = feats["dominant_period"]
        phase_lag[t] = feats["phase_lag"]

    return pd.DataFrame(
        {"coherence": coherence, "dominant_period": period,
         "phase_lag": phase_lag},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Sign-convention calibration (training data only)
# ---------------------------------------------------------------------------

def calibrate_direction_and_threshold(features_df, usd_series, target_series,
                                      split_date):
    """
    Using ONLY data before split_date:
      1. Compute the per-currency coherence threshold (e.g., 70th pctile)
      2. For time points above threshold, determine whether the signal
         convention (aligned vs inverse) has better directional accuracy
         on training data

    Returns dict with keys:
        threshold, direction ('aligned'|'inverse'), n_calibration_points,
        train_accuracy
    """
    # Training-only features
    train_feats = features_df.loc[features_df.index < split_date].dropna()
    if len(train_feats) < 100:
        return None

    threshold = float(np.percentile(train_feats["coherence"], COHERENCE_PCTILE))

    # Above-threshold training points
    gated = train_feats[train_feats["coherence"] >= threshold]
    if len(gated) < 50:
        return None

    # At each such t, the candidate signal is sign(USD at t - phase_lag).
    # We need target return at t+1.
    # Build lookup tables once.
    common = usd_series.index.intersection(target_series.index)
    usd_arr = usd_series.loc[common].to_numpy()
    tgt_arr = target_series.loc[common].to_numpy()
    date_to_idx = {d: i for i, d in enumerate(common)}

    agree_aligned = 0
    agree_inverse = 0
    n_valid = 0

    for t_date, row in gated.iterrows():
        if t_date not in date_to_idx:
            continue
        i = date_to_idx[t_date]
        # Need t+1 for target return; phase_lag days back for USD signal
        lag_days = int(round(row["phase_lag"]))
        # Clamp lag to reasonable range
        if lag_days < 0 or lag_days > WINDOW // 2:
            continue
        if i + 1 >= len(common):
            continue
        if i - lag_days < 0:
            continue

        usd_signal = np.sign(usd_arr[i - lag_days])
        tgt_next = tgt_arr[i + 1]
        if usd_signal == 0 or np.isnan(tgt_next):
            continue

        if np.sign(tgt_next) == usd_signal:
            agree_aligned += 1
        else:
            agree_inverse += 1
        n_valid += 1

    if n_valid < 30:
        return None

    aligned_acc = agree_aligned / n_valid
    inverse_acc = agree_inverse / n_valid

    if aligned_acc >= inverse_acc:
        direction = "aligned"
        train_accuracy = aligned_acc
    else:
        direction = "inverse"
        train_accuracy = inverse_acc

    return {
        "threshold": threshold,
        "direction": direction,
        "train_accuracy": float(train_accuracy),
        "n_calibration_points": int(n_valid),
    }


# ---------------------------------------------------------------------------
# Signal generation on test set (strictly causal)
# ---------------------------------------------------------------------------

def generate_test_signals(features_df, usd_series, calibration, split_date,
                          test_index):
    """
    At each test-set date t:
      - If coherence_t < threshold -> signal = 0 (flat)
      - Else -> signal = direction_sign * sign(USD return at t - phase_lag)
    Uses only data from times <= t (features are causal by construction;
    USD return used is from times <= t).

    Returns a pd.Series of signals indexed on test dates (value in {-1, 0, +1}).
    """
    if calibration is None:
        return None

    threshold = calibration["threshold"]
    direction_sign = 1 if calibration["direction"] == "aligned" else -1

    # USD lookup
    usd_arr = usd_series.to_numpy()
    usd_dates = {d: i for i, d in enumerate(usd_series.index)}

    signals = pd.Series(0.0, index=test_index)

    for t_date in test_index:
        if t_date not in features_df.index:
            continue
        row = features_df.loc[t_date]
        if pd.isna(row["coherence"]) or pd.isna(row["phase_lag"]):
            continue
        if row["coherence"] < threshold:
            continue
        lag_days = int(round(row["phase_lag"]))
        if lag_days < 0 or lag_days > WINDOW // 2:
            continue
        if t_date not in usd_dates:
            continue
        i = usd_dates[t_date]
        if i - lag_days < 0:
            continue
        usd_val = usd_arr[i - lag_days]
        if np.isnan(usd_val):
            continue
        usd_sign = np.sign(usd_val)
        if usd_sign == 0:
            continue
        signals.loc[t_date] = direction_sign * usd_sign

    return signals


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_signal(signal, target_returns):
    """
    Evaluate a signal against next-day target returns.
    signal at time t predicts return from t to t+1, so we shift target
    returns by -1 (i.e., return_at_t_plus_1 at row t).
    """
    # Align
    common = signal.index.intersection(target_returns.index)
    s = signal.loc[common]
    # Next-day return: return at t+1 placed at row t
    r_next = target_returns.shift(-1).loc[common]

    # Drop last row (no t+1 return)
    valid = s.notna() & r_next.notna()
    s = s[valid]
    r_next = r_next[valid]

    if len(s) == 0:
        return None

    # Fraction of time in-position
    in_position = (s != 0)
    time_in_market = float(in_position.mean())

    # Turnover: fraction of days where signal changes
    turnover = float((s.diff().abs() > 0).mean())

    # Restrict metrics to in-position periods for accuracy/IC/Sharpe
    s_pos = s[in_position]
    r_pos = r_next[in_position]

    if len(s_pos) < 10:
        return {
            "time_in_market": time_in_market,
            "turnover": turnover,
            "n_positions": int(in_position.sum()),
            "accuracy": np.nan,
            "ic": np.nan,
            "sharpe": np.nan,
        }

    # Directional accuracy among in-position days
    accuracy = float((np.sign(s_pos) == np.sign(r_pos)).mean())

    # Information coefficient: Spearman rank corr between signal and next return
    # (Signal is {-1, +1} in-position; effectively this is point-biserial
    # rank corr, but we compute across ALL days including zeros for fairness.)
    try:
        ic = float(pd.Series(s.values).rank().corr(
            pd.Series(r_next.values).rank()
        ))
    except Exception:
        ic = np.nan

    # Strategy returns: signal * next-day return
    strat = s * r_next
    mean_r = float(strat.mean())
    std_r = float(strat.std())
    sharpe = float((mean_r / std_r) * np.sqrt(252)) if std_r > 0 else np.nan

    return {
        "time_in_market": time_in_market,
        "turnover": turnover,
        "n_positions": int(in_position.sum()),
        "accuracy": accuracy,
        "ic": ic,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Main driver — per-currency pipeline
# ---------------------------------------------------------------------------

def process_currency(currency, log_returns, split_date, save_intermediate=False):
    """
    End-to-end pipeline for one currency. Returns a dict of results plus
    optionally the intermediate feature DataFrame.
    """
    if currency not in log_returns.columns:
        return None
    target_series = log_returns[currency].dropna()
    if len(target_series) < MIN_OBS_FOR_CURRENCY:
        return None

    # Build USD index leave-one-out
    usd_series = build_usd_index(log_returns, exclude=currency)

    # Rolling features
    features = compute_rolling_features(usd_series, target_series)
    if features is None or features["coherence"].notna().sum() < 100:
        return None

    # Calibrate threshold + direction on training data only
    calibration = calibrate_direction_and_threshold(
        features, usd_series, target_series, split_date
    )
    if calibration is None:
        return None

    # Generate signals on test set
    test_index = target_series.loc[target_series.index >= split_date].index
    signal = generate_test_signals(
        features, usd_series, calibration, split_date, test_index
    )
    if signal is None:
        return None

    # Evaluate
    metrics = evaluate_signal(signal, target_series)
    if metrics is None:
        return None

    result = {
        "currency": currency,
        **calibration,
        **metrics,
    }
    if save_intermediate:
        result["_features"] = features
        result["_signal"] = signal
    return result


def run(save=True, currencies=None):
    logger.info("=== Spectral Agent v2 (causal, rolling, gated) ===")
    logger.info(
        f"Hyperparams: window={WINDOW}, nperseg={NPERSEG}, "
        f"coherence_pctile={COHERENCE_PCTILE}, train_frac={TRAIN_FRAC}"
    )

    log_returns = load_data()
    train, test, split_date = train_test_split(log_returns)

    if currencies is None:
        currencies = [c for c in log_returns.columns if c != "USD"]

    logger.info(f"Processing {len(currencies)} currencies...")

    rows = []
    all_signals = {}
    all_features = {}

    for i, ccy in enumerate(currencies):
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(currencies)}")
        try:
            res = process_currency(
                ccy, log_returns, split_date, save_intermediate=True
            )
        except Exception as e:
            logger.warning(f"  {ccy} failed: {e}")
            continue
        if res is None:
            continue
        feats = res.pop("_features", None)
        signal = res.pop("_signal", None)
        rows.append(res)
        if signal is not None:
            all_signals[ccy] = signal
        if feats is not None:
            all_features[ccy] = feats

    if not rows:
        raise RuntimeError("No currencies produced results.")

    results_df = pd.DataFrame(rows).set_index("currency")
    results_df = results_df.sort_values("sharpe", ascending=False)

    # Aggregate summary
    logger.info("\n=== v2 Aggregate Results ===")
    logger.info(f"Currencies evaluated: {len(results_df)}")
    logger.info(f"Mean directional accuracy (in-position): "
                f"{results_df['accuracy'].mean():.4f}")
    logger.info(f"Median directional accuracy (in-position): "
                f"{results_df['accuracy'].median():.4f}")
    logger.info(f"Mean IC: {results_df['ic'].mean():.4f}")
    logger.info(f"Mean Sharpe: {results_df['sharpe'].mean():.4f}")
    logger.info(f"Median Sharpe: {results_df['sharpe'].median():.4f}")
    logger.info(f"Mean time-in-market: "
                f"{results_df['time_in_market'].mean():.4f}")
    logger.info(f"Mean turnover: {results_df['turnover'].mean():.4f}")
    logger.info(f"Pct with positive Sharpe: "
                f"{(results_df['sharpe'] > 0).mean():.4f}")

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(OUTPUTS_DIR / "spectral_v2_results.csv")

        # Save signals as a wide DataFrame (one column per currency)
        if all_signals:
            signals_df = pd.DataFrame(all_signals)
            signals_df.to_parquet(OUTPUTS_DIR / "spectral_v2_predictions.parquet")

        # Save per-currency coherence series (stacked long format to keep
        # file size manageable)
        feat_rows = []
        for ccy, f in all_features.items():
            f2 = f.copy()
            f2["currency"] = ccy
            f2 = f2.reset_index().rename(columns={"index": "date"})
            feat_rows.append(f2)
        if feat_rows:
            features_long = pd.concat(feat_rows, ignore_index=True)
            features_long.to_parquet(
                OUTPUTS_DIR / "spectral_v2_features.parquet"
            )

        logger.info("Saved: spectral_v2_results.csv, "
                    "spectral_v2_predictions.parquet, "
                    "spectral_v2_features.parquet")

    return results_df, all_signals, all_features


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    )
    run()
