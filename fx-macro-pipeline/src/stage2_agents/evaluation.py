"""
evaluation.py
=============

Layer-2 calibrated evaluation primitives.

Replaces the old Sharpe-threshold / accuracy-threshold decision rule
(see spectral_agent_v2.evaluate_signal) which the Layer-2 power audit
(data/layer2_audit/LAYER2_AUDIT_REPORT.md, Task A) found to have detection
rate = 0 at all IC levels and breadth conditions surveyed. That rule is
not present in this module by design. Diagnostics (sharpe, accuracy,
in-position IC, turnover, time-in-market) remain DESCRIPTIVE; the verdict
is a t-test on the per-period strategy return / rank-IC series.

Public API (verdict-bearing):

    decision_ttest(series, alpha=0.05)
        Two-sided t-test of series mean vs 0 with HAC (Newey-West) SE if
        statsmodels is installed; plain t otherwise. Returns
        (t_stat, p_value, fires, used_hac).

    evaluate_timeseries(signal, target_returns, alpha=0.05)
        Per-instrument timeseries evaluation. Returns descriptive
        diagnostics + the t-test verdict on the strategy return series.

    portfolio_return_series(signal_panel, fwd_panel)
        Equal-weight signal*fwd across instruments per period -> 1 series.

    evaluate_portfolio(signal_panel, fwd_panel, alpha=0.05)
        Portfolio (mean across instruments per period) t-test verdict.

    rank_ic_series(signal_panel, fwd_panel)
        Per-period Spearman rank correlation across instruments.

    long_short_return(signal_panel, fwd_panel, q=0.2,
                      periods_per_year=252)
        Top-q minus bottom-q forward returns per period, plus IR.

    evaluate_crosssectional(signal_panel, fwd_panel, alpha=0.05)
        Cross-sectional rank-IC t-test verdict + LS-IR.

    ic_in_position(signal, fwd_return)
        IC computed only over rows where signal != 0; also returns
        time_in_market. (Per the audit, full-series IC attenuates ~0.52x
        under 70% gating; in-position is the honest number.)

Calibration targets (locked by tests/test_evaluator_calibration.py):
  - decision_ttest FPR in [0, 0.10] at ic=0
  - timeseries portfolio t-test: det >= 0.8 at ic=0.02, n=18
  - cross-sectional rank-IC t-test: det >= 0.8 at ic=0.01, n=18, rho=0.6, h=1
  - in-position IC ~ 2x zero-padded IC at 70% gating
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    _HAVE_SM = True
except Exception:  # pragma: no cover
    sm = None
    _HAVE_SM = False


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def ic_in_position(signal, fwd_return):
    """
    Spearman IC restricted to rows where signal != 0.

    Parameters
    ----------
    signal : pd.Series
        Signal values (often {-1, 0, +1}). NaNs allowed.
    fwd_return : pd.Series
        Forward returns aligned to signal index. NaNs allowed.

    Returns
    -------
    dict with keys: ic_in_position, ic_full, time_in_market, n_in_position

    Notes
    -----
    The audit found that computing IC on the full series (including
    zero-signal rows) attenuates the realized IC by a factor of ~0.52
    under 70% gating. This function exposes both for comparability but
    the in-position figure is the honest one.
    """
    s = pd.Series(signal).astype(float)
    r = pd.Series(fwd_return).astype(float)
    s, r = s.align(r, join="inner")
    valid = s.notna() & r.notna()
    s = s[valid]
    r = r[valid]
    if len(s) == 0:
        return {
            "ic_in_position": np.nan,
            "ic_full": np.nan,
            "time_in_market": 0.0,
            "n_in_position": 0,
        }
    in_pos = (s != 0)
    tim = float(in_pos.mean())
    n_in = int(in_pos.sum())
    if n_in >= 10 and s[in_pos].std(ddof=1) > 0 and r[in_pos].std(ddof=1) > 0:
        ic_in, _ = stats.spearmanr(s[in_pos].values, r[in_pos].values)
        ic_in = float(ic_in)
    else:
        ic_in = np.nan
    if s.std(ddof=1) > 0 and r.std(ddof=1) > 0:
        ic_full, _ = stats.spearmanr(s.values, r.values)
        ic_full = float(ic_full)
    else:
        ic_full = np.nan
    return {
        "ic_in_position": ic_in,
        "ic_full": ic_full,
        "time_in_market": tim,
        "n_in_position": n_in,
    }


# ---------------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------------

def portfolio_return_series(signal_panel, fwd_panel):
    """
    Equal-weight signal*fwd across instruments per period.

    Parameters
    ----------
    signal_panel : pd.DataFrame  (T x N)
    fwd_panel : pd.DataFrame     (T x N) — forward returns aligned to
                                 signal_panel (signal at row t predicts
                                 fwd_panel at row t).

    Returns
    -------
    pd.Series of length T (rows with all-NaN dropped).
    """
    S = signal_panel.values.astype(float)
    F = fwd_panel.values.astype(float)
    contrib = S * F
    # nanmean ignores instruments missing on a given period
    with np.errstate(invalid="ignore"):
        port = np.nanmean(contrib, axis=1)
    out = pd.Series(port, index=signal_panel.index)
    return out.dropna()


def rank_ic_series(signal_panel, fwd_panel):
    """
    Per-period Spearman rank correlation across instruments.

    Vectorized along the time axis using scipy.stats.rankdata.
    Mirrors the reference implementation in data/layer2_audit/run_audit.py
    so calibration carries over.
    """
    common_idx = signal_panel.index.intersection(fwd_panel.index)
    common_cols = signal_panel.columns.intersection(fwd_panel.columns)
    S = signal_panel.loc[common_idx, common_cols].to_numpy(dtype=float)
    F = fwd_panel.loc[common_idx, common_cols].to_numpy(dtype=float)
    # Drop rows that are all-NaN on either side
    row_ok = (~np.isnan(S).all(axis=1)) & (~np.isnan(F).all(axis=1))
    S = S[row_ok]
    F = F[row_ok]
    idx = common_idx[row_ok]
    if S.shape[0] == 0 or S.shape[1] < 3:
        return pd.Series([], dtype=float)
    s_ranks = stats.rankdata(S, axis=1)
    f_ranks = stats.rankdata(F, axis=1)
    sr_c = s_ranks - s_ranks.mean(axis=1, keepdims=True)
    fr_c = f_ranks - f_ranks.mean(axis=1, keepdims=True)
    num = (sr_c * fr_c).sum(axis=1)
    den = np.sqrt((sr_c ** 2).sum(axis=1)) * np.sqrt((fr_c ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_t = num / np.where(den > 0, den, np.nan)
    return pd.Series(per_t, index=idx).dropna()


def long_short_return(signal_panel, fwd_panel, q=0.2, periods_per_year=252):
    """
    Top-quintile minus bottom-quintile forward return per period.

    Returns
    -------
    dict with keys: series (pd.Series), mean, std, IR (annualized).
    """
    common_idx = signal_panel.index.intersection(fwd_panel.index)
    common_cols = signal_panel.columns.intersection(fwd_panel.columns)
    S = signal_panel.loc[common_idx, common_cols].to_numpy(dtype=float)
    F = fwd_panel.loc[common_idx, common_cols].to_numpy(dtype=float)
    n_inst = S.shape[1]
    n_q = max(1, int(round(q * n_inst)))
    # Rows with NaN in S or F masked: replace NaNs with 0 in S only changes
    # ordering minimally; safer to drop bad rows.
    row_ok = np.isfinite(S).all(axis=1) & np.isfinite(F).all(axis=1)
    S = S[row_ok]
    F = F[row_ok]
    idx = common_idx[row_ok]
    if S.shape[0] == 0:
        return {"series": pd.Series([], dtype=float), "mean": np.nan,
                "std": np.nan, "IR": np.nan, "periods_per_year": periods_per_year}
    order = np.argsort(S, axis=1)
    bot = np.take_along_axis(F, order[:, :n_q], axis=1).mean(axis=1)
    top = np.take_along_axis(F, order[:, -n_q:], axis=1).mean(axis=1)
    ls = top - bot
    series = pd.Series(ls, index=idx)
    sd = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    mn = float(series.mean()) if len(series) else np.nan
    ir = float(mn / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else np.nan
    return {"series": series, "mean": mn, "std": sd, "IR": ir,
            "periods_per_year": periods_per_year}


# ---------------------------------------------------------------------------
# Verdict — t-test decision layer
# ---------------------------------------------------------------------------

def _newey_west_se(x, maxlags=None):
    """Newey-West HAC SE for the sample mean of x. Returns (mean, se)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    if maxlags is None:
        # Default Newey-West rule of thumb
        maxlags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        maxlags = max(0, maxlags)
    mean = float(x.mean())
    if _HAVE_SM:
        X = np.ones((n, 1))
        ols = sm.OLS(x, X).fit(cov_type="HAC",
                               cov_kwds={"maxlags": maxlags})
        se = float(np.sqrt(ols.cov_params()[0, 0]))
        return mean, se
    # Plain SE fallback
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else float("nan")
    return mean, se


def decision_ttest(series, alpha=0.05, use_hac=True, sided="two"):
    """
    t-test on a per-period series vs 0.

    Parameters
    ----------
    series : array-like
        Per-period statistic (e.g. portfolio return, rank-IC).
    alpha : float
        Significance level (interpreted in the chosen `sided` direction).
    use_hac : bool
        If True and statsmodels is available, use Newey-West HAC SE.
    sided : {"two", "greater", "less"}
        - "two" (default): two-sided test. Preserves the calibration
          tests in tests/test_evaluator_calibration.py.
        - "greater": one-sided test of mean > 0. Use only for a signal
          whose direction was PRE-REGISTERED before looking at data.
        - "less": one-sided test of mean < 0.
        One-sided p = two_sided_p / 2 if the sign matches the registered
        direction, else 1 - that.

    Returns
    -------
    dict with keys: t_stat, p_value, fires, used_hac, n, sided.
    """
    if sided not in ("two", "greater", "less"):
        raise ValueError(f"sided must be one of 'two','greater','less'; got {sided!r}")
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 30:
        return {"t_stat": float("nan"), "p_value": float("nan"),
                "fires": False, "used_hac": False, "n": int(n),
                "sided": sided}
    if x.std(ddof=1) == 0:
        return {"t_stat": float("nan"), "p_value": float("nan"),
                "fires": False, "used_hac": False, "n": int(n),
                "sided": sided}
    used_hac = bool(use_hac and _HAVE_SM)
    if used_hac:
        mean, se = _newey_west_se(x)
        if not np.isfinite(se) or se <= 0:
            t = float("nan")
            p_two = float("nan")
        else:
            t = mean / se
            # Two-sided p from large-sample normal approx (HAC is asymptotic)
            p_two = float(2.0 * (1.0 - stats.norm.cdf(abs(t))))
    else:
        t_stat, p_two = stats.ttest_1samp(x, 0.0)
        t = float(t_stat)
        p_two = float(p_two)

    if sided == "two" or not np.isfinite(p_two):
        p = p_two
    elif sided == "greater":
        # p_one = p_two/2 if t > 0 else 1 - p_two/2
        p = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
    else:  # "less"
        p = p_two / 2.0 if t < 0 else 1.0 - p_two / 2.0

    fires = bool(np.isfinite(p) and p < alpha)
    return {"t_stat": float(t), "p_value": float(p),
            "fires": fires, "used_hac": used_hac, "n": int(n),
            "sided": sided}


# ---------------------------------------------------------------------------
# Top-level verdict functions
# ---------------------------------------------------------------------------

def evaluate_timeseries(signal, target_returns, alpha=0.05):
    """
    Per-instrument timeseries evaluator. Diagnostics-plus-verdict.

    `target_returns` follows the same convention as
    spectral_agent_v2.evaluate_signal: target_returns at row t is the
    contemporaneous return; we shift -1 internally so r_next at row t is
    the next-day return. Signal at row t multiplies r_next at row t.

    Returns
    -------
    dict
        time_in_market, turnover, n_positions, accuracy, ic_in_position,
        ic_full, sharpe, ttest_t, ttest_p, fires (bool), used_hac.
    """
    common = signal.index.intersection(target_returns.index)
    s = pd.Series(signal).loc[common].astype(float)
    r_next = pd.Series(target_returns).shift(-1).loc[common].astype(float)
    valid = s.notna() & r_next.notna()
    s = s[valid]
    r_next = r_next[valid]

    if len(s) == 0:
        return None

    in_pos = (s != 0)
    tim = float(in_pos.mean())
    turnover = float((s.diff().abs() > 0).mean())
    n_pos = int(in_pos.sum())

    diag = ic_in_position(s, r_next)
    if n_pos >= 10:
        accuracy = float((np.sign(s[in_pos]) == np.sign(r_next[in_pos])).mean())
    else:
        accuracy = float("nan")

    strat = (s * r_next).dropna()
    mean_r = float(strat.mean()) if len(strat) else float("nan")
    std_r = float(strat.std(ddof=1)) if len(strat) > 1 else float("nan")
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r and std_r > 0 else float("nan")

    verdict = decision_ttest(strat.values, alpha=alpha)

    return {
        "time_in_market": tim,
        "turnover": turnover,
        "n_positions": n_pos,
        "accuracy": accuracy,
        "ic_in_position": diag["ic_in_position"],
        "ic_full": diag["ic_full"],
        "sharpe": sharpe,
        "ttest_t": verdict["t_stat"],
        "ttest_p": verdict["p_value"],
        "fires": verdict["fires"],
        "used_hac": verdict["used_hac"],
        "n": verdict["n"],
    }


def evaluate_portfolio(signal_panel, fwd_panel, alpha=0.05):
    """Portfolio (mean-across-instruments) t-test verdict + diagnostics."""
    port = portfolio_return_series(signal_panel, fwd_panel)
    verdict = decision_ttest(port.values, alpha=alpha)
    mean_r = float(port.mean()) if len(port) else float("nan")
    std_r = float(port.std(ddof=1)) if len(port) > 1 else float("nan")
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r and std_r > 0 else float("nan")
    return {
        "portfolio_mean": mean_r,
        "portfolio_std": std_r,
        "portfolio_sharpe": sharpe,
        "ttest_t": verdict["t_stat"],
        "ttest_p": verdict["p_value"],
        "fires": verdict["fires"],
        "used_hac": verdict["used_hac"],
        "n": verdict["n"],
        "series": port,
    }


def evaluate_crosssectional(signal_panel, fwd_panel, alpha=0.05,
                            q=0.2, periods_per_year=252):
    """Cross-sectional rank-IC t-test + LS-IR."""
    ric = rank_ic_series(signal_panel, fwd_panel)
    ls = long_short_return(signal_panel, fwd_panel, q=q,
                           periods_per_year=periods_per_year)
    verdict = decision_ttest(ric.values, alpha=alpha)
    return {
        "mean_rank_ic": float(ric.mean()) if len(ric) else float("nan"),
        "rank_ic_series": ric,
        "long_short_IR": ls["IR"],
        "long_short_series": ls["series"],
        "ttest_t": verdict["t_stat"],
        "ttest_p": verdict["p_value"],
        "fires": verdict["fires"],
        "used_hac": verdict["used_hac"],
        "n": verdict["n"],
    }


# ---------------------------------------------------------------------------
# Walk-forward — POOL, don't vote (audit Task D)
# ---------------------------------------------------------------------------

def walk_forward_pooled(signal_fn, signal_panel, fwd_panel, n_folds=5,
                        scheme="expanding", alpha=0.05, axis="timeseries",
                        q=0.2, periods_per_year=252):
    """
    Walk-forward evaluation that concatenates OOS per-period series across
    folds and runs ONE t-test on the pooled series.

    Per-fold Sharpe / IC are reported as a stability DIAGNOSTIC only —
    never as a fold vote. Audit Task D showed that on a genuine ic=0.03
    timeseries signal, per-fold Sharpe ranged [-1.84, 2.43] and a
    majority-vote-of-folds rule mode-failed; pooling detected the same
    signal cleanly. Voting across noisy folds is itself a false-negative
    generator.

    Parameters
    ----------
    signal_fn : callable
        signal_fn(train_signal, train_fwd, test_signal, test_fwd) -> oos_signal
        Returns the out-of-sample signal panel (DataFrame aligned to
        test_fwd's index/columns) to be evaluated on test_fwd. If the
        signal does not need fitting, pass `lambda ts, tf, xs, xf: xs`.
    signal_panel : pd.DataFrame
    fwd_panel : pd.DataFrame
    n_folds : int
    scheme : "expanding" or "rolling"
    axis : "timeseries" (portfolio_return_series) or
           "crosssectional" (rank_ic_series)
    """
    if axis not in ("timeseries", "crosssectional"):
        raise ValueError(axis)
    common_idx = signal_panel.index.intersection(fwd_panel.index)
    S = signal_panel.loc[common_idx]
    F = fwd_panel.loc[common_idx]
    T = len(S)
    if T < 4 * n_folds:
        raise ValueError(f"Too few periods ({T}) for {n_folds} folds")

    # Build fold boundaries
    fold_size = T // (n_folds + 1)
    pooled = []
    per_fold = []

    for k in range(n_folds):
        test_start = (k + 1) * fold_size
        test_end = (k + 2) * fold_size if k < n_folds - 1 else T
        if scheme == "expanding":
            train_start = 0
        elif scheme == "rolling":
            train_start = max(0, test_start - fold_size)
        else:
            raise ValueError(scheme)
        train_S = S.iloc[train_start:test_start]
        train_F = F.iloc[train_start:test_start]
        test_S_in = S.iloc[test_start:test_end]
        test_F = F.iloc[test_start:test_end]
        oos_signal = signal_fn(train_S, train_F, test_S_in, test_F)

        if axis == "timeseries":
            series = portfolio_return_series(oos_signal, test_F)
        else:
            series = rank_ic_series(oos_signal, test_F)
        pooled.append(series)

        n = len(series)
        if n > 1 and series.std(ddof=1) > 0:
            fold_mean = float(series.mean())
            fold_std = float(series.std(ddof=1))
            fold_sharpe = (fold_mean / fold_std
                          * np.sqrt(periods_per_year if axis == "timeseries"
                                    else 1.0))
        else:
            fold_mean = float("nan")
            fold_sharpe = float("nan")
        per_fold.append({"fold": k, "n": n, "mean": fold_mean,
                         "fold_sharpe_or_ic": fold_sharpe,
                         "test_start": str(S.index[test_start]),
                         "test_end": str(S.index[test_end - 1])})

    pooled_series = pd.concat(pooled) if pooled else pd.Series([], dtype=float)
    verdict = decision_ttest(pooled_series.values, alpha=alpha)

    fold_df = pd.DataFrame(per_fold)
    if len(fold_df):
        dispersion = {
            "min": float(fold_df["fold_sharpe_or_ic"].min()),
            "median": float(fold_df["fold_sharpe_or_ic"].median()),
            "max": float(fold_df["fold_sharpe_or_ic"].max()),
        }
    else:
        dispersion = {"min": float("nan"), "median": float("nan"),
                      "max": float("nan")}

    return {
        "pooled_series": pooled_series,
        "verdict": verdict,  # THE verdict
        "fires": verdict["fires"],
        "per_fold": fold_df,
        "dispersion": dispersion,
        "axis": axis,
        "scheme": scheme,
        "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# report_windows — always report full + clean-subperiod metrics
# ---------------------------------------------------------------------------

def report_windows(signal_panel, fwd_panel, clean_mask=None, alpha=0.05,
                   axis="timeseries"):
    """
    Always emit BOTH full-test and clean-subperiod metrics.

    Parameters
    ----------
    clean_mask : pd.Series[bool] or None
        Boolean mask aligned to signal_panel.index marking the
        clean (non-flagged) sub-period. If None, the clean window is
        identical to the full window.
    axis : "timeseries" | "crosssectional"
    """
    if axis == "timeseries":
        fn = evaluate_portfolio
    elif axis == "crosssectional":
        fn = evaluate_crosssectional
    else:
        raise ValueError(axis)

    full = fn(signal_panel, fwd_panel, alpha=alpha)
    if clean_mask is None:
        clean = full
        clean_idx = signal_panel.index
    else:
        clean_idx = signal_panel.index[clean_mask.reindex(signal_panel.index).fillna(False).values]
        clean = fn(signal_panel.loc[clean_idx], fwd_panel.loc[clean_idx],
                   alpha=alpha)

    return {
        "full": full,
        "clean": clean,
        "n_full": len(signal_panel),
        "n_clean": len(clean_idx),
    }


# ---------------------------------------------------------------------------
# Multiple-testing — BH-FDR guard for families of signals
# ---------------------------------------------------------------------------

def bh_fdr_decision(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array-like of float
    alpha : float — target FDR.

    Returns
    -------
    np.ndarray[bool] of same length, True where the corresponding signal
    is declared significant under BH at level alpha.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    if not passed.any():
        cutoff = -1
    else:
        cutoff = np.max(np.where(passed)[0])
    fires = np.zeros(m, dtype=bool)
    if cutoff >= 0:
        fires[order[: cutoff + 1]] = True
    return fires


def decide_family(series_list, alpha=0.05, use_hac=True):
    """
    Run decision_ttest on each series and apply BH-FDR across the family.

    Returns
    -------
    dict with: t_stats, p_values, fires_marginal, fires_bh.
    """
    t_stats = []
    p_values = []
    fires_marginal = []
    for s in series_list:
        v = decision_ttest(s, alpha=alpha, use_hac=use_hac)
        t_stats.append(v["t_stat"])
        p_values.append(v["p_value"])
        fires_marginal.append(v["fires"])
    p_arr = np.array(p_values, dtype=float)
    # NaNs in p_values get treated as 1.0 for BH (cannot reject)
    p_for_bh = np.where(np.isfinite(p_arr), p_arr, 1.0)
    fires_bh = bh_fdr_decision(p_for_bh, alpha=alpha)
    return {
        "t_stats": np.array(t_stats),
        "p_values": p_arr,
        "fires_marginal": np.array(fires_marginal),
        "fires_bh": fires_bh,
    }
