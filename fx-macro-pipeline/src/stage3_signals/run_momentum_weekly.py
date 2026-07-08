"""
Layer-3 Factor 1 — cross-sectional FX momentum, WEEKLY resolution.

End-to-end weekly re-verdict per the brief:
  - PART 1 input: data/weekly/weekly_returns.parquet (built by
    src/stage1_collection_v2/weekly_panel.py). Read-only.
  - PART 2 floor: src/stage3_signals/detection_floor_weekly.py
    headline = 0.020 at n=15 (rho=0.6, 789 weeks).
  - PART 3 verdict here: primary L=8w (registered BEFORE looking),
    secondaries {2,4,13,26}w (pre-registered robustness, BH-corrected),
    1-week non-overlapping holding period, terciles, dollar-neutral, costs
    Tier1 3bps / Tier2 20bps round-trip.

Writes data/layer3_momentum/MOMENTUM_WEEKLY_REPORT.md and per-formation
return series under data/layer3_momentum/per_formation_weekly_L*.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.universe import UNIVERSE  # noqa: E402
from stage1_collection_v2.weekly_panel import (  # noqa: E402
    build_weekly_panel,
    load_raw_inputs,
    STATUS_KEPT,
    STATUS_NOT_LISTED,
    STATUS_DROPPED_STALE,
)
from stage2_agents.evaluation import (  # noqa: E402
    decision_ttest,
    rank_ic_series,
    walk_forward_pooled,
    bh_fdr_decision,
)
from stage3_signals.momentum import (  # noqa: E402
    cross_sectional_weights,
    turnover_series,
    cost_series,
    gross_return_series,
    annualized_ir,
    max_drawdown,
    hit_rate,
    DEFAULT_SPREAD_BPS_BY_TIER,
)
from stage3_signals.detection_floor_weekly import main as run_floor_weekly

OUT = PROJ / "data" / "layer3_momentum"
OUT.mkdir(parents=True, exist_ok=True)

# Pre-registered formations (weeks). Fixed BEFORE looking at any data.
PRIMARY_FORMATION_WEEKS = 8
SECONDARY_FORMATIONS_WEEKS = (2, 4, 13, 26)
ALL_FORMATIONS_WEEKS = tuple(sorted({PRIMARY_FORMATION_WEEKS,
                                     *SECONDARY_FORMATIONS_WEEKS}))

# Clean-subperiod exclusions (calendar months at weekly resolution).
# COVID Q1 2020 + Russia/Ukraine Q1 2022. Conservative.
CLEAN_EXCLUDE_MONTHS = (
    "2020-02", "2020-03", "2020-04",
    "2022-02", "2022-03",
)

PERIODS_PER_YEAR = 52


# ---------------------------------------------------------------------------
# Build signal + fwd panels at weekly resolution
# ---------------------------------------------------------------------------

def build_weekly_signal_panels(weekly_returns: pd.DataFrame,
                               formations_weeks=ALL_FORMATIONS_WEEKS) -> dict:
    """
    Signal_L[w, X] = sum of weekly oriented returns over the L weeks
    ENDING AT week w (inclusive), strictly using data <= w.
    fwd_ret[w, X] = weekly return in week w+1 (the held week).

    NaNs in the underlying weekly_returns (dropped_stale, not_listed) are
    treated as MISSING — the signal at w skips them within its rolling sum,
    so a pair that has fewer than L valid weekly returns in its lookback
    window gets NaN signal at w (and is excluded from that week's
    cross-section, which is the correct behaviour). fwd_ret is NaN if the
    pair isn't kept in week w+1.
    """
    wr = weekly_returns.copy()
    wr.index = pd.DatetimeIndex(wr.index)
    weeks = wr.index

    # min_periods=L requires L valid observations to emit a signal; else NaN
    signals: Dict[int, pd.DataFrame] = {}
    for L in formations_weeks:
        sig = wr.rolling(window=L, min_periods=L).sum()
        # Last week has no fwd to realize against; drop it
        signals[L] = sig.iloc[:-1]

    fwd = wr.shift(-1).iloc[:-1]  # week w+1 return aligned to row w

    return {
        "weeks": weeks,
        "weeks_signal": weeks[:-1],
        "signals": signals,
        "fwd_ret": fwd,
        "universe": list(wr.columns),
    }


# ---------------------------------------------------------------------------
# Tier-keyed spread mapping keyed by PAIR (the weekly panel is pair-indexed)
# ---------------------------------------------------------------------------

def pair_spread_bps(pairs, spread_by_tier=None,
                    universe=None) -> Dict[str, float]:
    """Return {pair: round-trip spread bps} for every column in the weekly
    panel. The weekly panel uses pair codes as columns (orientation already
    baked into the returns), so we key spreads by pair, not by currency."""
    u = universe or UNIVERSE
    sp = spread_by_tier or DEFAULT_SPREAD_BPS_BY_TIER
    out: Dict[str, float] = {}
    for p in pairs:
        tier = u[p]["tier"]
        out[p] = float(sp.get(tier, max(sp.values())))
    return out


# ---------------------------------------------------------------------------
# Per-formation backtest (weekly variant of momentum.backtest_formation)
# ---------------------------------------------------------------------------

def backtest_weekly_formation(signal: pd.DataFrame, fwd_ret: pd.DataFrame,
                              q: float = 1.0 / 3.0,
                              spread_bps: Dict[str, float] = None,
                              periods_per_year: int = PERIODS_PER_YEAR
                              ) -> dict:
    weights = cross_sectional_weights(signal, q=q)
    gross = gross_return_series(weights, fwd_ret)
    if spread_bps is None:
        spread_bps = {c: 0.0 for c in weights.columns}
    cost = cost_series(weights, spread_bps)
    net = gross - cost
    turn = turnover_series(weights)
    ric = rank_ic_series(signal, fwd_ret)

    # Dollar-neutrality diagnostic
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
        beta = float("nan"); r2 = float("nan")

    n_obs = int(len(net.dropna()))
    return {
        "weights": weights, "gross": gross, "cost": cost, "net": net,
        "turnover": turn, "rank_ic_series": ric,
        "metrics": {
            "n_weeks": n_obs,
            "gross_ir": annualized_ir(gross, periods_per_year),
            "net_ir": annualized_ir(net, periods_per_year),
            "mean_rank_ic": float(ric.mean()) if len(ric) else float("nan"),
            "hit_rate_net": hit_rate(net),
            "mean_turnover": float(turn.mean()) if len(turn) else float("nan"),
            "max_drawdown_net": max_drawdown(net),
            "dollar_beta": float(beta) if np.isfinite(beta) else float("nan"),
            "dollar_r2": float(r2) if np.isfinite(r2) else float("nan"),
        },
    }


# ---------------------------------------------------------------------------
# Windowed slicing helpers
# ---------------------------------------------------------------------------

def make_clean_mask(index: pd.DatetimeIndex, exclude_yyyymm) -> pd.Series:
    keep = pd.Series(True, index=index)
    for ym in exclude_yyyymm:
        yyyy, mm = ym.split("-")
        keep[(index.year == int(yyyy)) & (index.month == int(mm))] = False
    return keep


def make_n15_only_mask(weekly_returns: pd.DataFrame) -> pd.Series:
    """A week is in the 'n=15-only span' iff every column in the panel is
    non-NaN (kept) that week. Excludes both the early n=10 listing gap and
    any mid-sample multi-pair drops (e.g. 2017-02-24)."""
    return weekly_returns.notna().all(axis=1)


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------

def run(alpha=0.05, n_floor_seeds=60):
    print("[Weekly momentum] loading weekly panel ...")
    ohlc, lr = load_raw_inputs()
    panel = build_weekly_panel(ohlc, lr)
    wr = panel["weekly_returns"]
    pairs = panel["universe"]
    print(f"  weekly_returns shape: {wr.shape}; universe: {pairs}")
    # Range of effective universe size per week
    per_week_n = wr.notna().sum(axis=1)
    print(f"  per-week breadth: min={per_week_n.min()} median={per_week_n.median()} "
          f"max={per_week_n.max()}")

    print("\n[PART 2 recall] weekly detection floor")
    live_floor, early_floor = run_floor_weekly()

    print("\n[PART 3] Per-formation backtest at weekly resolution")
    spread = pair_spread_bps(pairs)
    sp_panels = build_weekly_signal_panels(wr, ALL_FORMATIONS_WEEKS)
    fwd = sp_panels["fwd_ret"]
    results: Dict[int, dict] = {}
    summary_rows = []
    for L in ALL_FORMATIONS_WEEKS:
        sig = sp_panels["signals"][L]
        bt = backtest_weekly_formation(sig, fwd, q=1.0 / 3.0,
                                       spread_bps=spread,
                                       periods_per_year=PERIODS_PER_YEAR)
        results[L] = bt
        m = bt["metrics"]
        marker = " *PRIMARY*" if L == PRIMARY_FORMATION_WEEKS else ""
        print(f"  L={L:>2}w{marker}  n_weeks={m['n_weeks']:4d}  "
              f"gross_IR={m['gross_ir']:+.2f}  net_IR={m['net_ir']:+.2f}  "
              f"rank_IC={m['mean_rank_ic']:+.4f}  "
              f"turnover={m['mean_turnover']:.2f}  "
              f"dollar_beta={m['dollar_beta']:+.3f} R2={m['dollar_r2']:.3f}")
        summary_rows.append({"formation_weeks": L, **m})
        per = pd.DataFrame({
            "gross": bt["gross"], "cost": bt["cost"], "net": bt["net"],
            "turnover": bt["turnover"],
        })
        per.to_csv(OUT / f"per_formation_weekly_L{L}.csv")

    pd.DataFrame(summary_rows).to_csv(OUT / "factor_backtest_weekly.csv",
                                       index=False)

    # ---------------- Primary verdict ----------------
    print("\n  PRIMARY VERDICT (L=8w):")
    prim = results[PRIMARY_FORMATION_WEEKS]
    net_prim = prim["net"].dropna()
    gross_prim = prim["gross"].dropna()
    ric_prim = prim["rank_ic_series"].dropna()

    # Pooled walk-forward on NET LS series (treat as 1-instrument "panel")
    ls_panel = pd.DataFrame({"LS": net_prim})
    sig_panel = pd.DataFrame({"LS": np.ones_like(net_prim.values)},
                             index=net_prim.index)
    def passthrough(train_S, train_F, test_S, test_F):
        return test_S
    wf = walk_forward_pooled(
        passthrough, sig_panel, ls_panel, n_folds=5, scheme="expanding",
        alpha=alpha, axis="timeseries", periods_per_year=PERIODS_PER_YEAR,
    )
    pooled = wf["pooled_series"].dropna()
    verdict_pooled = decision_ttest(pooled.values, alpha=alpha, sided="greater")
    verdict_full_net = decision_ttest(net_prim.values, alpha=alpha,
                                      sided="greater")
    verdict_full_ric = decision_ttest(ric_prim.values, alpha=alpha,
                                      sided="greater")
    print(f"    pooled OOS one-sided  t={verdict_pooled['t_stat']:+.2f}  "
          f"p={verdict_pooled['p_value']:.4f}  fires={verdict_pooled['fires']}")
    print(f"    full-sample net       t={verdict_full_net['t_stat']:+.2f}  "
          f"p={verdict_full_net['p_value']:.4f}  fires={verdict_full_net['fires']}")
    print(f"    full-sample rank-IC   t={verdict_full_ric['t_stat']:+.2f}  "
          f"p={verdict_full_ric['p_value']:.4f}  fires={verdict_full_ric['fires']}")
    print(f"    fold dispersion (Sharpe):  min={wf['dispersion']['min']:+.2f}  "
          f"median={wf['dispersion']['median']:+.2f}  "
          f"max={wf['dispersion']['max']:+.2f}")

    # ---------------- report_windows: full vs clean vs n=15-only ----------
    print("\n  report_windows views:")
    clean_mask = make_clean_mask(net_prim.index, CLEAN_EXCLUDE_MONTHS)
    n15_only_mask_full = make_n15_only_mask(wr)
    # Align n15 mask to net_prim's index
    n15_mask = n15_only_mask_full.reindex(net_prim.index).fillna(False)

    def _slice(net, ric, mask):
        idx = net.index[mask.values] if isinstance(mask, pd.Series) \
            else net.index[mask]
        ns = net.loc[idx]
        rs = ric.loc[ric.index.intersection(idx)]
        ir = annualized_ir(ns, PERIODS_PER_YEAR)
        return ir, float(rs.mean()) if len(rs) else float("nan"), len(ns)

    full_ir = annualized_ir(net_prim, PERIODS_PER_YEAR)
    full_ric_mean = float(ric_prim.mean())
    full_n = len(net_prim)
    clean_ir, clean_ric, clean_n = _slice(net_prim, ric_prim, clean_mask)
    n15_ir, n15_ric, n15_n = _slice(net_prim, ric_prim, n15_mask)
    print(f"    full           n_weeks={full_n}  net IR={full_ir:+.2f}  "
          f"rank_IC={full_ric_mean:+.4f}")
    print(f"    clean          n_weeks={clean_n}  net IR={clean_ir:+.2f}  "
          f"rank_IC={clean_ric:+.4f}  (excl COVID + Ukraine months)")
    print(f"    n=15 only span n_weeks={n15_n}  net IR={n15_ir:+.2f}  "
          f"rank_IC={n15_ric:+.4f}  (excl early n=10 + EM outage)")

    # ---------------- Secondary (BH-FDR) ----------------
    print(f"\n  SECONDARY (BH-FDR across L in {SECONDARY_FORMATIONS_WEEKS}):")
    sec_series, sec_labels, sec_p = [], [], []
    for L in SECONDARY_FORMATIONS_WEEKS:
        s = results[L]["net"].dropna().values
        v = decision_ttest(s, alpha=alpha, sided="greater")
        sec_series.append(s); sec_labels.append(f"L{L}w")
        sec_p.append(v["p_value"])
    sec_bh = bh_fdr_decision(sec_p, alpha=alpha)
    for lab, p, fires in zip(sec_labels, sec_p, sec_bh):
        print(f"    {lab}: one-sided p={p:.4f}  BH-fires={bool(fires)}")

    # ---------------- GO/NO-GO outcome ----------------
    primary_net_ic = float(ric_prim.mean())
    primary_gross_ic = float(rank_ic_series(
        sp_panels["signals"][PRIMARY_FORMATION_WEEKS], fwd
    ).mean())
    print(f"\n  WEEKLY GO/NO-GO floor (Part 2, n=15, rho=0.6): "
          f"net rank-IC > {live_floor:.3f}")
    print(f"  Measured primary gross rank-IC = {primary_gross_ic:+.4f}")
    print(f"  Measured primary net rank-IC   = {primary_net_ic:+.4f}")

    if primary_net_ic > live_floor and verdict_pooled["fires"]:
        outcome = "(i) GO  — net IC > floor AND pooled OOS t-test fires"
    elif primary_gross_ic > live_floor and primary_net_ic <= live_floor:
        outcome = ("(ii) HALT — gross IC > floor but NET < floor; signal real "
                   "but eaten by costs (report cost wedge)")
    elif primary_gross_ic <= live_floor:
        if primary_gross_ic < 0:
            outcome = (f"(iii) NO-GO on momentum — gross IC ({primary_gross_ic:+.4f}) "
                       f"< floor ({live_floor:.3f}). NB: IC is NEGATIVE, consistent "
                       "with cross-sectional REVERSAL rather than momentum; magnitude "
                       "still below the floor so reversal is also not confirmable")
        else:
            outcome = ("(iii) NO-GO — gross IC < floor; not detectable at this "
                       "breadth, stop and reconsider universe/horizon")
    else:
        outcome = ("(iv) AMBIGUOUS — net IC above floor but t-test doesn't fire "
                   "(signal small or HAC-SE too inflated)")
    print(f"  OUTCOME: {outcome}")

    # ---------------- write report ----------------
    _write_report(
        out_path=OUT / "MOMENTUM_WEEKLY_REPORT.md",
        pairs=pairs, per_week_n=per_week_n,
        live_floor=live_floor, early_floor=early_floor,
        summary_rows=summary_rows,
        verdict_pooled=verdict_pooled,
        verdict_full_net=verdict_full_net,
        verdict_full_ric=verdict_full_ric,
        wf_dispersion=wf["dispersion"],
        full_n=full_n, full_ir=full_ir, full_ric=full_ric_mean,
        clean_n=clean_n, clean_ir=clean_ir, clean_ric=clean_ric,
        n15_n=n15_n, n15_ir=n15_ir, n15_ric=n15_ric,
        sec_labels=sec_labels, sec_p=sec_p, sec_bh=sec_bh,
        primary_net_ic=primary_net_ic, primary_gross_ic=primary_gross_ic,
        outcome=outcome,
        spread_by_tier=DEFAULT_SPREAD_BPS_BY_TIER,
    )
    print(f"\n  Wrote {OUT/'MOMENTUM_WEEKLY_REPORT.md'}")
    return {
        "outcome": outcome,
        "verdict_pooled": verdict_pooled,
        "primary_net_ic": primary_net_ic,
        "primary_gross_ic": primary_gross_ic,
        "live_floor": live_floor,
        "summary_rows": summary_rows,
    }


def _write_report(out_path, pairs, per_week_n, live_floor, early_floor,
                  summary_rows, verdict_pooled, verdict_full_net,
                  verdict_full_ric, wf_dispersion,
                  full_n, full_ir, full_ric, clean_n, clean_ir, clean_ric,
                  n15_n, n15_ir, n15_ric, sec_labels, sec_p, sec_bh,
                  primary_net_ic, primary_gross_ic, outcome, spread_by_tier):
    L = []
    L.append("# Layer-3 Factor 1 — Weekly cross-sectional FX momentum")
    L.append("")
    L.append("Branch: `weekly-resampler`. Read-only on `data/processed_v2/` "
             "and `data/weekly/`. All verdicts via "
             "`src/stage2_agents/evaluation.py`.")
    L.append("")
    L.append(f"Universe (n={len(pairs)}): {', '.join(pairs)}. "
             "Per-week breadth: "
             f"min={per_week_n.min()}, median={per_week_n.median():.0f}, "
             f"max={per_week_n.max()}.")
    L.append("")

    L.append("## Pre-registration")
    L.append("")
    L.append(f"- Primary formation: **L = {PRIMARY_FORMATION_WEEKS} weeks** "
             "(registered before any backtest).")
    L.append(f"- Secondary formations (pre-registered, BH-corrected, "
             f"robustness only): {SECONDARY_FORMATIONS_WEEKS}.")
    L.append("- Holding: 1 week non-overlapping (matches the floor's h=5).")
    L.append("- Decision: `decision_ttest(sided='greater')` via "
             "`walk_forward_pooled` (5 folds, expanding, pool OOS — never "
             "vote folds).")
    L.append("- Costs (tier-keyed, round-trip bps): "
             f"`{json.dumps(spread_by_tier)}`. "
             "Cost per period = Σ |Δw_i| · (spread_i / 2).")
    L.append("")

    L.append("## Part 2 — Weekly detection floor (recap)")
    L.append("")
    L.append(f"- At n=15, rho=0.6, ~789 weeks: **net rank-IC > {live_floor:.3f}** "
             "for detection rate ≥ 0.80, `sided='greater'`.")
    L.append(f"- At n=10 (early-2010 listing gap + 2017-02-24 EM-outage week): "
             f"floor = {early_floor:.3f}.")
    L.append("- vs the monthly floor of 0.050 — weekly resolution buys ~2.5× "
             "more data so the floor drops by roughly the expected √4≈2 factor.")
    L.append("")
    L.append("See `detection_floor_weekly.csv` and "
             "`detection_floor_weekly_summary.csv`.")
    L.append("")

    L.append("## Part 3 — Per-formation backtest (terciles, dollar-neutral)")
    L.append("")
    L.append("| L (weeks) | n_weeks | gross IR | net IR | mean rank IC | "
             "hit rate (net) | turnover | max DD (net) | dollar β | dollar R² |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        L.append(
            f"| {row['formation_weeks']} | {row['n_weeks']} | "
            f"{row['gross_ir']:+.2f} | {row['net_ir']:+.2f} | "
            f"{row['mean_rank_ic']:+.4f} | {row['hit_rate_net']:.2f} | "
            f"{row['mean_turnover']:.2f} | {row['max_drawdown_net']:+.3f} | "
            f"{row['dollar_beta']:+.3f} | {row['dollar_r2']:.3f} |"
        )
    L.append("")

    L.append(f"### Primary verdict (L={PRIMARY_FORMATION_WEEKS}w)")
    L.append("")
    L.append(f"- Pooled OOS one-sided: t = {verdict_pooled['t_stat']:+.2f}, "
             f"p = {verdict_pooled['p_value']:.4f}, "
             f"**fires = {verdict_pooled['fires']}**")
    L.append(f"- Full-sample net one-sided: t = {verdict_full_net['t_stat']:+.2f}, "
             f"p = {verdict_full_net['p_value']:.4f}")
    L.append(f"- Full-sample rank-IC one-sided: t = {verdict_full_ric['t_stat']:+.2f}, "
             f"p = {verdict_full_ric['p_value']:.4f}")
    L.append(f"- Fold dispersion (Sharpe, diagnostic only — never the verdict): "
             f"min = {wf_dispersion['min']:+.2f}, "
             f"median = {wf_dispersion['median']:+.2f}, "
             f"max = {wf_dispersion['max']:+.2f}")
    L.append("")

    L.append("### report_windows: full vs clean-subperiod vs n=15-only span")
    L.append("")
    L.append("| view | n_weeks | net IR | mean rank IC |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| full                                       | {full_n} | "
             f"{full_ir:+.2f} | {full_ric:+.4f} |")
    L.append(f"| clean (excl COVID Q1 2020 + Ukraine Q1 2022) | {clean_n} | "
             f"{clean_ir:+.2f} | {clean_ric:+.4f} |")
    L.append(f"| n=15 only span (excl early n=10 + EM outage) | {n15_n} | "
             f"{n15_ir:+.2f} | {n15_ric:+.4f} |")
    L.append("")
    L.append("Reading: if any of these views is materially different from the "
             "full view, it tells us whether the signal is regime- or "
             "breadth-dependent. Identical → robust to that cut.")
    L.append("")

    L.append("### Secondary (BH-FDR, robustness only — does NOT override primary)")
    L.append("")
    L.append("| L | one-sided p | BH-fires |")
    L.append("|---|---:|---:|")
    for lab, p, fires in zip(sec_labels, sec_p, sec_bh):
        L.append(f"| {lab} | {p:.4f} | {bool(fires)} |")
    L.append("")

    L.append("## GO/NO-GO")
    L.append("")
    L.append(f"- Floor (Part 2, n=15): net rank-IC > **{live_floor:.3f}**")
    L.append(f"- Measured primary gross rank-IC: **{primary_gross_ic:+.4f}**")
    L.append(f"- Measured primary net rank-IC:   **{primary_net_ic:+.4f}**")
    L.append("")
    L.append(f"**Outcome: {outcome}**")
    L.append("")
    L.append("Outcomes spelled out (per brief):")
    L.append("- (i) net > floor AND pooled OOS t-test fires → real, proceed.")
    L.append("- (ii) gross > floor but NET < floor → signal real but eaten by "
             "costs.")
    L.append("- (iii) gross < floor → not detectable at this breadth; if gross "
             "IC is negative the data leans towards cross-sectional reversal "
             "(noted explicitly).")
    L.append("- (iv) net > floor but t-test doesn't fire → signal small or "
             "HAC-SE too inflated.")
    L.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    run()
