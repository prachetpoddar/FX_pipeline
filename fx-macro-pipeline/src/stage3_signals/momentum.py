"""
Layer-3 Factor 1: cross-sectional FX momentum.

Reads fx_log_returns.parquet (read-only). Builds an orientation-corrected
daily return panel, a monthly signal/forward panel for L in {1,3,6,12} months,
and a dollar-neutral tercile long-short portfolio with transaction costs.

All verdict decisions go through src/stage2_agents/evaluation.py
(decision_ttest, rank_ic_series, long_short_return, ic_in_position,
walk_forward_pooled, report_windows, decide_family). Nothing here
reimplements a parallel metric.

Module is read-only on data stores; outputs land under
data/layer3_momentum/.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.universe import UNIVERSE  # noqa: E402
from stage2_agents.evaluation import (  # noqa: E402
    decision_ttest,
    rank_ic_series,
    long_short_return,
    ic_in_position,
    walk_forward_pooled,
    report_windows,
    decide_family,
)


DEFAULT_FX_PARQUET = PROJ / "data" / "processed_v2" / "fx_log_returns.parquet"
OUT_DIR = PROJ / "data" / "layer3_momentum"

# Tier-keyed default round-trip spreads (bps). Conservative retail figures.
DEFAULT_SPREAD_BPS_BY_TIER: Dict[int, float] = {1: 3.0, 2: 20.0, 3: 30.0}

# Pre-registered primary formation. NEVER mutate this without re-doing Part 3.
PRIMARY_FORMATION_MONTHS: int = 3
SECONDARY_FORMATIONS_MONTHS: tuple = (1, 6, 12)
ALL_FORMATIONS_MONTHS: tuple = (1, 3, 6, 12)

# High-vol months to optionally exclude in the "clean" report_windows view.
# Months (YYYY-MM); inclusive list, conservative.
CLEAN_EXCLUDE_MONTHS: tuple = (
    "2020-02", "2020-03", "2020-04",  # COVID
    "2022-02", "2022-03",             # Russia/Ukraine
)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def orientation_map(pairs: Sequence[str],
                    universe: Optional[dict] = None) -> Dict[str, int]:
    """
    Build an orientation map: sign = +1 if quote=="USD" (XXXUSD, like
    EURUSD) so an oriented return > 0 means XXX strengthened vs USD;
    sign = -1 if base=="USD" (USDXXX, like USDJPY) so we flip raw return.

    Raises if a pair is missing from `universe` or has neither base nor
    quote == "USD" (we don't know how to orient cross pairs without an
    explicit registration).
    """
    u = universe or UNIVERSE
    out = {}
    for p in pairs:
        meta = u.get(p)
        if meta is None:
            raise KeyError(f"{p}: not in universe — cannot orient")
        base = meta.get("base")
        quote = meta.get("quote")
        if quote == "USD" and base != "USD":
            out[p] = +1
        elif base == "USD" and quote != "USD":
            out[p] = -1
        else:
            raise ValueError(
                f"{p}: cross-pair (base={base}, quote={quote}) — orientation "
                f"not defined without explicit anchor"
            )
    return out


def orient_returns(log_returns: pd.DataFrame,
                   omap: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Multiply each column by its orientation sign. Output column name is
    the currency strengthened-vs-USD (so a +1 oriented EURUSD return =
    EUR strengthened; a -1 oriented USDJPY return = JPY strengthened)."""
    if omap is None:
        omap = orientation_map(list(log_returns.columns))
    out = pd.DataFrame(index=log_returns.index)
    for col in log_returns.columns:
        sign = omap[col]
        # Currency name = base if sign==+1 (XXXUSD), quote if sign==-1 (USDXXX)
        meta = UNIVERSE[col]
        ccy = meta["base"] if sign == +1 else meta["quote"]
        out[ccy] = sign * log_returns[col]
    return out


# ---------------------------------------------------------------------------
# Monthly panel
# ---------------------------------------------------------------------------

def month_end_index(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last business date observed within each (year, month)."""
    g = pd.Series(daily_index, index=daily_index).groupby(
        [daily_index.year, daily_index.month]
    ).max()
    return pd.DatetimeIndex(g.values).sort_values()


def build_monthly_panel(oriented_daily: pd.DataFrame,
                        formations_months: Sequence[int] = ALL_FORMATIONS_MONTHS
                        ) -> dict:
    """
    Build month-end signal panels and forward-return panel.

    Convention:
      - month-end m = last business day with data within calendar month.
      - signal_L[m, X] = cumulative oriented log return over the L months
        ENDING AT month-end m, strictly using data <= m.
      - fwd_ret[m, X] = sum of daily oriented log returns over the
        FOLLOWING calendar month (strictly > m, < next month-end m+1).
        The last month-end is dropped (no future month to realize).
      - NO skip-month: FX shows 1-month continuation (registered).

    Returns
    -------
    dict with keys:
        month_ends      -> DatetimeIndex of all month-ends (T)
        signals         -> {L: DataFrame[T-1 x N]}  (last m dropped to align with fwd)
        fwd_ret         -> DataFrame[T-1 x N]
        universe        -> list[str] of currency columns
    """
    daily = oriented_daily.copy()
    daily.index = pd.DatetimeIndex(daily.index)
    me = month_end_index(daily.index)
    # Cumulative oriented log return (per currency). NaN-safe via fillna(0)
    # since accumulation of NaN would propagate; in practice the universe
    # is rectangular after universe-level filtering.
    cum = daily.fillna(0.0).cumsum()
    cum_at_me = cum.reindex(me, method="ffill")

    signals: Dict[int, pd.DataFrame] = {}
    # Drop the last month-end from the signal panel because there's no
    # forward month to realize against it.
    me_signal = me[:-1]
    for L in formations_months:
        # Signal at month-end m = cum(m) - cum(m - L months)
        # If we don't have L months of history, NaN.
        shifted = cum_at_me.shift(L)
        sig_full = cum_at_me - shifted
        sig = sig_full.reindex(me_signal)
        signals[L] = sig

    # Forward return for month m = cum(m+1) - cum(m), summed over the days
    # in the next month. Equivalent to: cum_at_me.shift(-1) - cum_at_me.
    fwd_full = cum_at_me.shift(-1) - cum_at_me
    fwd = fwd_full.reindex(me_signal)

    return {
        "month_ends": me,
        "signals": signals,
        "fwd_ret": fwd,
        "universe": list(daily.columns),
    }


# ---------------------------------------------------------------------------
# Cross-sectional tercile portfolio with costs
# ---------------------------------------------------------------------------

def cross_sectional_weights(signal: pd.DataFrame, q: float = 1.0 / 3.0
                            ) -> pd.DataFrame:
    """
    Dollar-neutral tercile weights from a cross-sectional signal.

    For each row m:
      - n = number of non-NaN columns
      - n_q = max(1, round(q * n))
      - long the top n_q with weight +1/n_q
      - short the bottom n_q with weight -1/n_q
      - middle currencies get 0
    """
    W = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for m, row in signal.iterrows():
        finite = row.dropna()
        n = len(finite)
        if n < 3:
            continue
        n_q = max(1, int(round(q * n)))
        ranked = finite.sort_values()
        bot = ranked.iloc[:n_q].index
        top = ranked.iloc[-n_q:].index
        W.loc[m, top] = 1.0 / n_q
        W.loc[m, bot] = -1.0 / n_q
    return W


def cost_per_pair_bps(pair_currency_map: Dict[str, str],
                      spread_bps_by_tier: Optional[Dict[int, float]] = None,
                      universe: Optional[dict] = None) -> Dict[str, float]:
    """
    Per-CURRENCY round-trip spread (bps) keyed by the underlying tradable
    pair's tier. Each currency maps to its USDXXX or XXXUSD pair.
    """
    u = universe or UNIVERSE
    spreads = spread_bps_by_tier or DEFAULT_SPREAD_BPS_BY_TIER
    out: Dict[str, float] = {}
    for ccy, pair in pair_currency_map.items():
        tier = u[pair]["tier"]
        out[ccy] = float(spreads.get(tier, max(spreads.values())))
    return out


def currency_to_pair_map(universe: Optional[dict] = None,
                         pairs: Optional[Sequence[str]] = None
                         ) -> Dict[str, str]:
    """Map currency code -> its tradable pair in our universe."""
    u = universe or UNIVERSE
    pair_iter = pairs if pairs is not None else u.keys()
    out: Dict[str, str] = {}
    for p in pair_iter:
        meta = u[p]
        ccy = meta["base"] if meta["quote"] == "USD" else meta["quote"]
        out[ccy] = p
    return out


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    """Sum of |Δw_i| per period (entry+exit halved per side)."""
    dW = weights.diff()
    # First row: building the book from cash = |w_0|
    dW.iloc[0] = weights.iloc[0]
    return dW.abs().sum(axis=1)


def cost_series(weights: pd.DataFrame, spread_bps: Dict[str, float]) -> pd.Series:
    """
    Per-period cost = sum_i |Δw_i| * (spread_i / 2) (in decimal, not bps).
    Half-spread per side; full round-trip on a complete entry-and-exit.
    """
    dW = weights.diff()
    dW.iloc[0] = weights.iloc[0]
    # spread vector in decimal half-spread
    spread_vec = pd.Series(spread_bps).reindex(weights.columns).fillna(
        max(spread_bps.values()) if spread_bps else 0.0
    )
    half = spread_vec / 2.0 / 1e4  # bps -> decimal, half-spread
    cost = (dW.abs() * half).sum(axis=1)
    return cost


def gross_return_series(weights: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Σ_i w_i * fwd_ret_i per period."""
    aligned = fwd_ret.reindex(columns=weights.columns)
    return (weights * aligned).sum(axis=1, min_count=1)


def annualized_ir(series: pd.Series, periods_per_year: int = 12) -> float:
    s = series.dropna()
    if len(s) < 2 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(series: pd.Series) -> float:
    """Max drawdown of cumulative-sum series (log returns)."""
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    cum = s.cumsum()
    dd = cum - cum.cummax()
    return float(dd.min())


def hit_rate(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    return float((s > 0).mean())


# ---------------------------------------------------------------------------
# Per-formation backtest
# ---------------------------------------------------------------------------

def backtest_formation(signal: pd.DataFrame, fwd_ret: pd.DataFrame,
                       q: float = 1.0 / 3.0,
                       spread_bps: Optional[Dict[str, float]] = None,
                       ) -> dict:
    """
    Run a single-formation tercile L/S backtest with costs.

    Returns a dict including:
      weights, gross, cost, net, turnover, rank_ic_series, dollar_factor
      metrics — gross & net IR, mean rank IC, in-position IC, hit rate,
      turnover, max drawdown, dollar-beta, dollar-R2.
    """
    weights = cross_sectional_weights(signal, q=q)
    gross = gross_return_series(weights, fwd_ret)
    if spread_bps is None:
        spread_bps = {c: 0.0 for c in weights.columns}
    cost = cost_series(weights, spread_bps)
    net = gross - cost
    turn = turnover_series(weights)

    # rank IC across instruments per month
    ric = rank_ic_series(signal, fwd_ret)
    diag = ic_in_position(weights.stack(future_stack=True),
                          fwd_ret.reindex_like(weights).stack(future_stack=True))

    # Dollar-neutrality diagnostic: regress net on equal-weight oriented fwd
    dollar = fwd_ret.mean(axis=1).reindex(net.index)
    common = net.dropna().index.intersection(dollar.dropna().index)
    y = net.loc[common].values
    x = dollar.loc[common].values
    if len(common) > 2 and x.std(ddof=1) > 0 and y.std(ddof=1) > 0:
        beta, alpha_intercept = np.polyfit(x, y, 1)
        y_pred = alpha_intercept + beta * x
        ss_res = float(((y - y_pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        beta = float("nan")
        r2 = float("nan")

    return {
        "weights": weights,
        "gross": gross,
        "cost": cost,
        "net": net,
        "turnover": turn,
        "rank_ic_series": ric,
        "dollar_series": dollar,
        "metrics": {
            "n_months": int(len(net.dropna())),
            "gross_ir": annualized_ir(gross),
            "net_ir": annualized_ir(net),
            "mean_rank_ic": float(ric.mean()) if len(ric) else float("nan"),
            "ic_in_position_pooled": diag["ic_in_position"],
            "hit_rate_net": hit_rate(net),
            "mean_turnover": float(turn.mean()) if len(turn) else float("nan"),
            "max_drawdown_net": max_drawdown(net),
            "dollar_beta": float(beta) if np.isfinite(beta) else float("nan"),
            "dollar_r2": float(r2) if np.isfinite(r2) else float("nan"),
        },
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def load_universe_returns(parquet_path: Path = DEFAULT_FX_PARQUET
                          ) -> pd.DataFrame:
    """Read the cleaned daily log-return panel, drop currencies that fall
    outside the registered universe, and forward an empty DataFrame if the
    universe is empty (defensive only)."""
    df = pd.read_parquet(parquet_path)
    # Use only columns present in UNIVERSE (defensive — should be all)
    cols = [c for c in df.columns if c in UNIVERSE]
    return df[cols]
