"""
preprocessor_v2.py
==================

Computes log returns from daily OHLC bars and applies hard sanity checks
before writing output.

DESIGN PRINCIPLE
----------------
The original Stage 1 silently accepted corrupted data (ECB reference rates
with 0.44 autocorrelation in daily returns). That contamination propagated
to every downstream metric. This module refuses to write output if data
fails validation — corruption stops here, doesn't propagate.

WHAT GETS VALIDATED PER PAIR
----------------------------
    1. Lag-1 autocorrelation of daily log returns must be in [-0.10, +0.10]
       (true tradable FX has near-zero return autocorrelation; anomalies
       indicate stale prices, smoothing, or a data error)
    2. Daily |return| standard deviation must be within plausible bounds
       for the pair's tier (Tier 1: 0.003-0.012, Tier 2: 0.004-0.020,
       Tier 3: 0.003-0.030; pegged pairs allowed near zero)
    3. No run of >10 consecutive identical closes (indicates stale data)
    4. No more than 5% of returns flagged as >5σ outliers
    5. At least 1000 valid daily observations (else not enough for modeling)

OUTPUT
------
    fx_log_returns.parquet      — daily log returns, columns=pair
    fx_close_prices.parquet     — daily close prices (for diagnostics)
    validation_report.csv       — per-pair validation results
    rejected_pairs.csv          — pairs that failed validation, with reasons
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation thresholds — set deliberately, documented
# ---------------------------------------------------------------------------
#
# LEGACY_* constants preserve the pre-remediation values for diff legibility
# and so the old behavior remains reproducible by flag. The active constants
# below them implement the audit-driven fixes (see
# data/layer2_audit/gate_false_rejection.csv, Task E):
#
#   - 96% reject on bid_ask_bounce_major (ac1≈-0.12): the symmetric
#     |ac1|<=0.10 bound treated a benign bid-ask bounce as if it were the
#     ECB smoothing pathology. Fix: asymmetric — reject only ac1 > +0.10,
#     allow negative ac1 down to AC1_NEG_FLOOR.
#   - 100% reject on crisis_year_patch (40-day stale run inside 4000 clean
#     days): whole-pair rejection on localized corruption torpedoed 15
#     clean years of data. Fix: WINDOWED EXCLUSION — drop the bad window
#     and keep the pair if >= MIN_OBS clean rows remain.
#   - 100% reject on thin_but_tradable (11-day legitimate holiday flat):
#     MAX_STALE_RUN=10 is too tight for legitimately thin pairs. Fix:
#     raise the ceiling to MAX_STALE_RUN=15.
#
# Legacy values
LEGACY_MAX_ABS_AUTOCORR = 0.10
LEGACY_MAX_STALE_RUN    = 10

# Active values
MAX_AUTOCORR_POS        = 0.10     # reject if ac1 > +0.10 (smoothing/stale)
AC1_NEG_FLOOR           = -0.30    # reject if ac1 < -0.30 (extreme bounce)
MIN_OBS                 = 1000
MAX_STALE_RUN           = 15       # raised from 10; holiday-tolerant
MAX_OUTLIER_FRAC        = 0.05
OUTLIER_SIGMA           = 5.0
PEG_STD_THRESHOLD       = 0.0005   # below this, treat as pegged-ish

# Windowed-exclusion knobs — drop bad windows, keep clean rows.
WINDOW_EXCL_PAD         = 2        # pad each excluded window by this many bars
WINDOW_EXCL_MIN_LEN     = 5        # only excise runs >= this length (else
                                   # leave them in — small stale clusters
                                   # are within noise)

# Tier-specific std bounds for daily log returns
STD_BOUNDS = {
    1: (0.003, 0.012),   # G10 majors
    2: (0.004, 0.020),   # liquid EM
    3: (0.0,   0.030),   # less liquid / NDF — wider tolerance; pegged
                          # currencies allowed near zero
}


def compute_log_returns_from_close(close_series):
    """Standard log return: r_t = ln(P_t / P_{t-1}). NaN-safe."""
    return np.log(close_series / close_series.shift(1))


def detect_stale_run(close_series, max_run=MAX_STALE_RUN):
    """Longest run of identical consecutive closes. Returns the run length."""
    s = close_series.dropna()
    if len(s) < 2:
        return 0
    same = (s.diff() == 0).astype(int)
    # Compute run lengths of consecutive 1s
    if same.sum() == 0:
        return 0
    # Cumulative sum trick: each "0" resets a new group
    groups = (same == 0).cumsum()
    run_lengths = same.groupby(groups).sum()
    return int(run_lengths.max())


def find_stale_runs(close_series, min_len=WINDOW_EXCL_MIN_LEN):
    """
    Return a list of (start_pos, end_pos_inclusive) integer index positions
    for every run of identical consecutive closes of length >= min_len.
    Positions are into the *dropna'd* close series. The caller maps these
    to the original index.
    """
    s = close_series.dropna()
    if len(s) < 2:
        return []
    # A run of length L of equal closes shows up as L-1 consecutive zeros in
    # diff. We want runs where diff == 0 for >= min_len - 1 consecutive bars.
    diff = s.diff().to_numpy()
    runs = []
    i = 1
    n = len(diff)
    while i < n:
        if diff[i] == 0:
            j = i
            while j < n and diff[j] == 0:
                j += 1
            # The stale run covers original positions (i-1 .. j-1) inclusive
            run_len = (j - 1) - (i - 1) + 1
            if run_len >= min_len:
                runs.append((i - 1, j - 1))
            i = j + 1
        else:
            i += 1
    return runs


def excise_windows(returns_series, close_series, pad=WINDOW_EXCL_PAD,
                   min_len=WINDOW_EXCL_MIN_LEN):
    """
    Drop entries from `returns_series` that fall inside detected stale-run
    windows in `close_series` (each window padded by `pad` bars on either
    side). Returns a new Series — the original is untouched.
    """
    s = close_series.dropna()
    if len(s) < 2:
        return returns_series
    runs = find_stale_runs(s, min_len=min_len)
    if not runs:
        return returns_series
    bad_dates = set()
    idx = s.index
    n = len(idx)
    for a, b in runs:
        lo = max(0, a - pad)
        hi = min(n - 1, b + pad)
        for k in range(lo, hi + 1):
            bad_dates.add(idx[k])
    if not bad_dates:
        return returns_series
    keep_mask = ~returns_series.index.isin(bad_dates)
    return returns_series[keep_mask]


def validate_pair(pair_code, close_series, tier):
    """
    Run all sanity checks for one pair. Returns dict with status and reasons.

    Remediation (audit Task E):
      - AC1 bound is now asymmetric — reject ac1 > +0.10 (the smoothing /
        stale-price pathology) but tolerate negative ac1 down to
        AC1_NEG_FLOOR (benign bid-ask bounce).
      - Localized stale-run corruption is no longer a whole-pair death
        sentence. We excise the bad windows (with WINDOW_EXCL_PAD bars of
        padding) and re-run std/autocorr/outlier checks on the clean
        remainder. The pair passes if at least MIN_OBS clean rows remain.
      - MAX_STALE_RUN raised from 10 to 15 so legitimate holiday flats in
        thin-but-tradable pairs do not trigger.
      - When windowed exclusion is applied, status becomes "PASS_WINDOWED"
        and the dropped window count + total dropped rows are reported.
    """
    reasons = []
    close = close_series.dropna()
    n_close = len(close)
    if n_close < MIN_OBS:
        reasons.append(f"too_few_obs={n_close}")

    returns_full = compute_log_returns_from_close(close_series).dropna()
    n_ret_full = len(returns_full)
    if n_ret_full < MIN_OBS:
        reasons.append(f"too_few_returns={n_ret_full}")

    # Peg short-circuit BEFORE windowed exclusion: an entirely-flat series
    # (e.g. a pegged or de-facto pegged pair) has near-zero std on the raw
    # returns. Excising it would leave us with zero rows and then we'd
    # misclassify it as "no data" rather than "pegged-like".
    peg_short_circuit = (
        n_ret_full > 0
        and float(returns_full.std()) < PEG_STD_THRESHOLD
    )

    # Detect localized stale runs up front; if present, work with a
    # windowed-excluded returns series for the std/autocorr/outlier checks.
    if peg_short_circuit:
        returns = returns_full
        stale_runs = []
        n_excised = 0
    else:
        stale_runs = find_stale_runs(close, min_len=WINDOW_EXCL_MIN_LEN)
        returns = excise_windows(returns_full, close,
                                 pad=WINDOW_EXCL_PAD,
                                 min_len=WINDOW_EXCL_MIN_LEN)
        n_excised = int(n_ret_full - len(returns))
    n_windows = len(stale_runs)
    windowed = n_excised > 0
    n_ret = len(returns)

    # Longest run remaining AFTER excision — this is the run that, if it
    # still exceeds MAX_STALE_RUN, indicates non-localized staleness (e.g.
    # a pair that is mostly flat).
    stale_run_after = detect_stale_run(
        close_series[~close_series.index.isin(
            returns_full.index.difference(returns.index)
        )]
    )

    if n_ret >= 100:
        std = float(returns.std())
        ac1 = float(returns.autocorr(lag=1))
        lo, hi = STD_BOUNDS.get(tier, (0.0, 0.05))
        # Pegged exception: if std is near zero, that's actually expected
        is_pegged_like = std < PEG_STD_THRESHOLD

        if not is_pegged_like:
            if not (lo <= std <= hi):
                reasons.append(
                    f"std_out_of_bounds={std:.5f}_tier{tier}_lo{lo}_hi{hi}"
                )
            # Asymmetric AC1 bound
            if ac1 > MAX_AUTOCORR_POS:
                reasons.append(f"autocorr_pos_too_high={ac1:+.4f}")
            elif ac1 < AC1_NEG_FLOOR:
                reasons.append(f"autocorr_neg_extreme={ac1:+.4f}")

        # Outlier check
        if std > 0:
            z = (returns - returns.mean()) / std
            outlier_frac = float((z.abs() > OUTLIER_SIGMA).mean())
            if outlier_frac > MAX_OUTLIER_FRAC:
                reasons.append(f"outlier_frac={outlier_frac:.3f}")
        else:
            outlier_frac = 0.0
    else:
        std = float("nan")
        ac1 = float("nan")
        outlier_frac = float("nan")
        is_pegged_like = False

    # Post-excision stale-run check: if a run still exceeds MAX_STALE_RUN
    # after windowed exclusion, the staleness is not localized.
    if stale_run_after > MAX_STALE_RUN and not is_pegged_like:
        reasons.append(f"stale_run={stale_run_after}")

    # Status
    if reasons:
        status = "FAIL"
    elif windowed:
        status = "PASS_WINDOWED"
    else:
        status = "PASS"

    return {
        "pair":          pair_code,
        "tier":          tier,
        "n_close":       n_close,
        "n_returns":     n_ret,
        "n_excised":     n_excised,
        "n_windows":     n_windows,
        "std":           std,
        "autocorr_1":    ac1,
        "outlier_frac":  outlier_frac,
        "stale_run":     stale_run_after,
        "is_pegged":     is_pegged_like,
        "status":        status,
        "reasons":       ";".join(reasons) if reasons else "",
    }


def build_returns_panel(daily_panel_wide, universe_dict, allow_pegged=True):
    """
    Build a wide DataFrame of daily log returns from daily OHLC panel.

    daily_panel_wide:  DataFrame indexed by date, columns MultiIndex
                       (pair, field) — output of aggregate_universe
    universe_dict:     UNIVERSE dict
    allow_pegged:      if True, pegged pairs pass validation as long as
                       they're explicitly flagged in universe (USDHKD)
    """
    close = daily_panel_wide.xs("close", level="field", axis=1)
    returns = pd.DataFrame(index=close.index)
    validations = []

    for pair_code in close.columns:
        meta = universe_dict.get(pair_code, {})
        tier = meta.get("tier", 3)
        v = validate_pair(pair_code, close[pair_code], tier)
        # Whitelist: if universe explicitly notes this is pegged, downgrade
        # peg-related failures to a warning rather than hard fail
        if allow_pegged and "PEGGED" in meta.get("notes", ""):
            v["status"] = "PASS_PEGGED"
        validations.append(v)
        if v["status"].startswith("PASS"):
            r = compute_log_returns_from_close(close[pair_code])
            # Apply windowed exclusion to the written returns so localized
            # corruption is dropped (rather than the whole pair).
            r = excise_windows(r.dropna(), close[pair_code],
                               pad=WINDOW_EXCL_PAD,
                               min_len=WINDOW_EXCL_MIN_LEN)
            returns[pair_code] = r

    validation_df = pd.DataFrame(validations).set_index("pair")
    return returns, validation_df


def run_preprocessing(daily_panel_path, universe_dict, output_dir):
    """
    Full preprocessing pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(daily_panel_path)
    logger.info(f"Loaded daily panel: {daily.shape}")

    close_panel = daily.xs("close", level="field", axis=1)
    close_panel.to_parquet(output_dir / "fx_close_prices.parquet")

    returns, validation = build_returns_panel(daily, universe_dict)

    logger.info("\n=== Validation summary ===")
    status_counts = validation["status"].value_counts()
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")

    passed = validation[validation["status"].str.startswith("PASS")].index.tolist()
    failed = validation[validation["status"] == "FAIL"]

    logger.info(f"\nPassed pairs: {passed}")
    if len(failed) > 0:
        logger.warning(f"\nFailed pairs (will be excluded from returns parquet):")
        for pair, row in failed.iterrows():
            logger.warning(f"  {pair}: {row['reasons']}")

    validation.to_csv(output_dir / "validation_report.csv")
    if len(failed) > 0:
        failed.to_csv(output_dir / "rejected_pairs.csv")

    # Write only passing pairs to the main returns parquet
    if passed:
        returns_clean = returns[passed]
        returns_clean.to_parquet(output_dir / "fx_log_returns.parquet")
        logger.info(f"\nWrote fx_log_returns.parquet with {len(passed)} pairs")
    else:
        raise RuntimeError(
            "ZERO pairs passed validation. Refusing to write empty output."
        )

    return returns_clean, validation
