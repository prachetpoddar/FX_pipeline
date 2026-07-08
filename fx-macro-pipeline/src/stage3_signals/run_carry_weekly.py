"""
Layer-3 Factor 2 — Cross-sectional FX carry, WEEKLY resolution.

End-to-end driver:
  - PART 1: build_weekly_carry_panel (carry.py).
  - PART 2: detection_floor_carry — consistency check; floor 0.020 at n=15.
  - PART 3: verdict via decision_ttest(sided='greater') on NET L/S,
            walk_forward_pooled. report_windows full vs clean vs n=15-only.
            Primary (rate-diff) + secondary (CIP-forward) — BH-reported.
  - PART 4: carry-vs-reversal overlap diagnostic (DIAGNOSTIC ONLY; no
            reversal verdict).

Writes data/layer3_carry/CARRY_WEEKLY_REPORT.md and per-formation CSVs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as scistats

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.universe import UNIVERSE  # noqa: E402
from stage2_agents.evaluation import (  # noqa: E402
    decision_ttest,
    rank_ic_series,
    walk_forward_pooled,
    bh_fdr_decision,
)
from stage3_signals.carry import build_weekly_carry_panel  # noqa: E402
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
from stage3_signals.run_momentum_weekly import (  # noqa: E402
    build_weekly_signal_panels as build_momentum_weekly_panels,
    pair_spread_bps,
    backtest_weekly_formation,
    make_clean_mask,
    make_n15_only_mask,
    PRIMARY_FORMATION_WEEKS as MOM_PRIMARY_L,
    PERIODS_PER_YEAR,
    CLEAN_EXCLUDE_MONTHS,
)
from stage3_signals.detection_floor_carry import main as run_floor_carry

OUT = PROJ / "data" / "layer3_carry"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-signal backtest wrapper
# ---------------------------------------------------------------------------

def backtest_carry_signal(signal: pd.DataFrame, fwd: pd.DataFrame,
                          spread_bps: Dict[str, float]) -> dict:
    return backtest_weekly_formation(signal, fwd, q=1.0 / 3.0,
                                     spread_bps=spread_bps,
                                     periods_per_year=PERIODS_PER_YEAR)


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------

def run(alpha: float = 0.05):
    print("[Weekly carry] loading panels ...")
    panel = build_weekly_carry_panel()
    pairs = panel["universe"]
    sig_p = panel["signal_primary"]
    sig_s = panel["signal_secondary"]
    fwd = panel["fwd_ret"]
    weekly_returns = panel["weekly_returns"]
    print(f"  universe (n={len(pairs)}): {pairs}")
    print(f"  weeks (signal): {len(sig_p.index)}")

    print("\n[Orientation verification]")
    ranks = sig_p.rank(axis=1, pct=True)
    n_weeks_with_data = sig_p.notna().sum(axis=1)
    top_frac = {p: float((ranks[p].dropna() >= 2.0 / 3.0).mean()) for p in pairs}
    bot_frac = {p: float((ranks[p].dropna() <= 1.0 / 3.0).mean()) for p in pairs}
    for p in ["USDMXN", "USDZAR", "USDJPY", "USDCHF", "AUDUSD", "EURUSD"]:
        print(f"  {p}: top_tercile_frac={top_frac[p]:.3f}  "
              f"bot_tercile_frac={bot_frac[p]:.3f}")
    assert top_frac["USDMXN"] > 0.8, "USDMXN should be top tercile most weeks"
    assert top_frac["USDZAR"] > 0.8, "USDZAR should be top tercile most weeks"
    assert bot_frac["USDJPY"] > 0.7, "USDJPY should be bottom tercile most weeks"
    assert bot_frac["USDCHF"] > 0.7, "USDCHF should be bottom tercile most weeks"
    print("  orientation asserts passed.")

    print("\n[PART 2 recall] carry detection floor (consistency check)")
    live_floor, early_floor = run_floor_carry()

    print("\n[PART 3] Carry backtest")
    spread = pair_spread_bps(pairs)
    bt_primary = backtest_carry_signal(sig_p, fwd, spread)
    bt_secondary = backtest_carry_signal(sig_s, fwd, spread)
    for label, bt in (("PRIMARY (rate-diff)", bt_primary),
                      ("SECONDARY (CIP-fwd)", bt_secondary)):
        m = bt["metrics"]
        print(f"  {label:<22s}  n_weeks={m['n_weeks']:4d}  "
              f"gross_IR={m['gross_ir']:+.2f}  net_IR={m['net_ir']:+.2f}  "
              f"rank_IC={m['mean_rank_ic']:+.4f}  "
              f"turnover={m['mean_turnover']:.2f}  "
              f"dollar_beta={m['dollar_beta']:+.3f} "
              f"R2={m['dollar_r2']:.3f}")
    # Per-formation CSVs
    for label, bt in (("primary", bt_primary), ("secondary", bt_secondary)):
        per = pd.DataFrame({
            "gross": bt["gross"], "cost": bt["cost"], "net": bt["net"],
            "turnover": bt["turnover"],
        })
        per.to_csv(OUT / f"per_signal_weekly_{label}.csv")

    # ---------- Primary verdict ----------
    print("\n  PRIMARY VERDICT (rate-diff carry):")
    net_prim = bt_primary["net"].dropna()
    gross_prim = bt_primary["gross"].dropna()
    ric_prim = bt_primary["rank_ic_series"].dropna()

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

    # ---------- report_windows ----------
    print("\n  report_windows views:")
    clean_mask = make_clean_mask(net_prim.index, CLEAN_EXCLUDE_MONTHS)
    n15_mask = make_n15_only_mask(weekly_returns).reindex(net_prim.index).fillna(False)

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
    clean_ir, clean_ric_mean, clean_n = _slice(net_prim, ric_prim, clean_mask)
    n15_ir, n15_ric_mean, n15_n = _slice(net_prim, ric_prim, n15_mask)
    print(f"    full           n_weeks={full_n}  net IR={full_ir:+.2f}  "
          f"rank_IC={full_ric_mean:+.4f}")
    print(f"    clean          n_weeks={clean_n}  net IR={clean_ir:+.2f}  "
          f"rank_IC={clean_ric_mean:+.4f}")
    print(f"    n=15 only span n_weeks={n15_n}  net IR={n15_ir:+.2f}  "
          f"rank_IC={n15_ric_mean:+.4f}")

    # ---------- Secondary (CIP) — BH alongside primary ----------
    print("\n  SECONDARY (CIP-fwd carry):")
    net_sec = bt_secondary["net"].dropna()
    ric_sec = bt_secondary["rank_ic_series"].dropna()
    verdict_sec = decision_ttest(net_sec.values, alpha=alpha, sided="greater")
    # BH-FDR across the two signals
    pvals = [verdict_full_net["p_value"], verdict_sec["p_value"]]
    bh = bh_fdr_decision(pvals, alpha=alpha)
    print(f"    primary    one-sided p={pvals[0]:.4f}  BH-fires={bool(bh[0])}")
    print(f"    secondary  one-sided p={pvals[1]:.4f}  BH-fires={bool(bh[1])}")
    basis = sig_p - sig_s.reindex_like(sig_p)
    basis_summary = float(basis.abs().mean().mean())
    basis_corr = float(sig_p.corrwith(sig_s.reindex_like(sig_p), axis=0).mean())
    print(f"    basis = primary - CIP-secondary: mean|basis|={basis_summary:.4f}, "
          f"mean within-pair corr(primary, secondary)={basis_corr:+.4f}")

    # ---------- GO/NO-GO outcome (PRIMARY) ----------
    primary_net_ic = full_ric_mean
    primary_gross_ic = float(rank_ic_series(sig_p, fwd).mean())
    print(f"\n  WEEKLY CARRY GO/NO-GO floor (Part 2): net rank-IC > {live_floor:.3f}")
    print(f"  Measured primary gross rank-IC = {primary_gross_ic:+.4f}")
    print(f"  Measured primary net rank-IC   = {primary_net_ic:+.4f}")
    if primary_net_ic > live_floor and verdict_pooled["fires"]:
        outcome = "(i) GO  — net IC > floor AND pooled OOS t-test fires"
    elif primary_gross_ic > live_floor and primary_net_ic <= live_floor:
        outcome = ("(ii) HALT — gross IC > floor but NET < floor; signal real but "
                   "eaten by costs")
    elif primary_gross_ic <= live_floor:
        if primary_gross_ic < 0:
            outcome = (f"(iii) NO-GO on carry — gross IC ({primary_gross_ic:+.4f}) "
                       f"< floor ({live_floor:.3f}). NB IC is NEGATIVE — direction "
                       "opposite to registered carry hypothesis")
        else:
            outcome = ("(iii) NO-GO — gross IC < floor; not detectable at this "
                       "breadth")
    else:
        outcome = ("(iv) AMBIGUOUS — net IC above floor but t-test does not fire")
    print(f"  OUTCOME: {outcome}")

    # ---------- PART 4 — carry-vs-reversal overlap (DIAGNOSTIC ONLY) ----------
    print("\n[PART 4] Carry-vs-reversal overlap (DIAGNOSTIC ONLY)")
    print("  GUARD: This section renders NO verdict on the reversal "
          "hypothesis. The momentum/reversal IC is computed only to measure "
          "its week-by-week relationship with carry. Reversal-as-hypothesis "
          "requires a separate-universe or forward holdout and is NOT confirmed "
          "here.")
    # Recompute the registered momentum primary (L=8w) at weekly resolution
    mom_panels = build_momentum_weekly_panels(
        weekly_returns, formations_weeks=(MOM_PRIMARY_L,)
    )
    mom_sig = mom_panels["signals"][MOM_PRIMARY_L]
    mom_fwd = mom_panels["fwd_ret"]
    mom_ric = rank_ic_series(mom_sig, mom_fwd)
    # Negate so positive = reversal direction
    neg_mom_ric = -mom_ric

    # Common weeks for the IC-correlation comparison
    common_ic = ric_prim.index.intersection(neg_mom_ric.index)
    a = ric_prim.loc[common_ic].values
    b = neg_mom_ric.loc[common_ic].values
    pearson_corr = float(np.corrcoef(a, b)[0, 1])
    spearman_corr = float(scistats.spearmanr(a, b)[0])
    print(f"  Pearson  corr(carry IC, negated-momentum IC) = {pearson_corr:+.4f}  "
          f"(n_weeks={len(common_ic)})")
    print(f"  Spearman corr(carry IC, negated-momentum IC) = {spearman_corr:+.4f}")

    # Regress carry net L/S on momentum net L/S to get residual alpha + beta
    mom_bt = backtest_weekly_formation(mom_sig, mom_fwd, q=1.0 / 3.0,
                                       spread_bps=spread,
                                       periods_per_year=PERIODS_PER_YEAR)
    mom_net = mom_bt["net"].dropna()
    common_rets = net_prim.index.intersection(mom_net.index)
    y = net_prim.loc[common_rets].values
    x = mom_net.loc[common_rets].values
    if len(common_rets) > 2 and x.std(ddof=1) > 0:
        beta, alpha_intercept = np.polyfit(x, y, 1)
        y_pred = alpha_intercept + beta * x
        ss_res = float(((y - y_pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        beta = alpha_intercept = r2 = float("nan")

    alpha_t = float("nan"); alpha_p = float("nan")
    if np.isfinite(beta):
        residual = y - (alpha_intercept + beta * x)
        residual = residual[np.isfinite(residual)]
        if residual.size > 2 and residual.std(ddof=1) > 0:
            alpha_t_obj = scistats.ttest_1samp(residual, 0.0)
            alpha_t = float(alpha_t_obj.statistic)
            # One-sided greater on residual mean (carry alpha > 0)
            alpha_p = float(alpha_t_obj.pvalue / 2.0
                            if alpha_t > 0 else 1.0 - alpha_t_obj.pvalue / 2.0)
    print(f"  Carry net regressed on momentum net: "
          f"beta={beta:+.3f}  R²={r2:.3f}  "
          f"alpha (weekly)={alpha_intercept:+.6f}  "
          f"alpha-t (residual mean / SE) = {alpha_t:+.2f}, "
          f"one-sided p={alpha_p:.4f}")

    # World verdict
    high_ic_corr = (abs(pearson_corr) >= 0.5) or (abs(spearman_corr) >= 0.5)
    carry_residual_alpha_small = (not np.isfinite(alpha_p)) or alpha_p >= 0.10
    carry_grossic_below_floor = primary_gross_ic <= live_floor
    if carry_grossic_below_floor:
        world = ("World C (resolution-bound): carry's gross IC is below the "
                 "floor at this breadth/resolution, so the diagnostic above "
                 "is not load-bearing — the binding constraint is breadth. "
                 "Reversal-as-hypothesis requires a separate-universe or "
                 "forward holdout regardless")
    elif high_ic_corr and carry_residual_alpha_small:
        world = ("World A (same coin): high IC correlation with negated momentum "
                 "AND carry alpha vs momentum small — the 'reversal' visible in "
                 "the momentum run is largely the carry signal in disguise. "
                 "(Diagnostic only — confirmatory test on a separate registration "
                 "is still required.)")
    else:
        world = ("World B (two effects): carry and negated-momentum carry "
                 "independent information OR carry retains alpha net of momentum "
                 "— distinct effects; reversal earns a separate registered "
                 "out-of-sample test")

    print(f"\n  OVERLAP READING: {world}")
    # Save overlap CSV
    overlap_rows = pd.DataFrame({
        "week_end": common_ic,
        "carry_ric": ric_prim.loc[common_ic].values,
        "neg_momentum_ric": neg_mom_ric.loc[common_ic].values,
    })
    overlap_rows.to_csv(OUT / "carry_reversal_overlap.csv", index=False)

    # ---------- Write report ----------
    _write_report(
        out_path=OUT / "CARRY_WEEKLY_REPORT.md",
        pairs=pairs, n_weeks=len(net_prim),
        live_floor=live_floor, early_floor=early_floor,
        bt_primary=bt_primary, bt_secondary=bt_secondary,
        verdict_pooled=verdict_pooled,
        verdict_full_net=verdict_full_net,
        verdict_full_ric=verdict_full_ric,
        verdict_sec_net=verdict_sec,
        bh=bh, bh_p=pvals,
        wf_dispersion=wf["dispersion"],
        full_n=full_n, full_ir=full_ir, full_ric_mean=full_ric_mean,
        clean_n=clean_n, clean_ir=clean_ir, clean_ric_mean=clean_ric_mean,
        n15_n=n15_n, n15_ir=n15_ir, n15_ric_mean=n15_ric_mean,
        top_frac=top_frac, bot_frac=bot_frac,
        primary_net_ic=primary_net_ic, primary_gross_ic=primary_gross_ic,
        basis_summary=basis_summary, basis_corr=basis_corr,
        pearson_corr=pearson_corr, spearman_corr=spearman_corr,
        ovp_beta=beta, ovp_r2=r2, ovp_alpha=alpha_intercept,
        ovp_alpha_t=alpha_t, ovp_alpha_p=alpha_p,
        n_common_ic=len(common_ic),
        outcome=outcome, world=world,
        spread_by_tier=DEFAULT_SPREAD_BPS_BY_TIER,
    )
    print(f"\n  Wrote {OUT/'CARRY_WEEKLY_REPORT.md'}")
    return {
        "outcome": outcome, "world": world,
        "verdict_pooled": verdict_pooled,
        "primary_net_ic": primary_net_ic,
        "primary_gross_ic": primary_gross_ic,
        "live_floor": live_floor,
    }


def _write_report(out_path, pairs, n_weeks, live_floor, early_floor,
                  bt_primary, bt_secondary,
                  verdict_pooled, verdict_full_net, verdict_full_ric,
                  verdict_sec_net, bh, bh_p,
                  wf_dispersion, full_n, full_ir, full_ric_mean,
                  clean_n, clean_ir, clean_ric_mean,
                  n15_n, n15_ir, n15_ric_mean,
                  top_frac, bot_frac,
                  primary_net_ic, primary_gross_ic,
                  basis_summary, basis_corr,
                  pearson_corr, spearman_corr,
                  ovp_beta, ovp_r2, ovp_alpha, ovp_alpha_t, ovp_alpha_p,
                  n_common_ic, outcome, world, spread_by_tier):
    L = []
    L.append("# Layer-3 Factor 2 — Weekly cross-sectional FX carry")
    L.append("")
    L.append("Branch: `layer3-carry`. Read-only on `data/processed_v2/`. "
             "Reuses the shared weekly resampler "
             "(`src/stage1_collection_v2/weekly_panel.py`) and the "
             "calibrated judge (`src/stage2_agents/evaluation.py`).")
    L.append("")
    L.append(f"Universe (n={len(pairs)}): {', '.join(pairs)}.")
    L.append("")

    L.append("## Pre-registration")
    L.append("")
    L.append("- **Primary signal:** lagged short-rate differential "
             "(`carry_signal_pct` keyed by the non-USD currency of each pair, "
             "lagged 1 trading day in source so values are point-in-time "
             "at the daily index and at Friday-NY week-ends).")
    L.append("- **Secondary signal (pre-registered, BH-reported, robustness):** "
             "CIP-forward-implied carry from `cip_fwd_points_pair` / spot; "
             "sign-corrected via `-orientation(pair) × cip / spot` so "
             "positive = high yielder. The basis (primary − secondary) is "
             "the cross-currency basis and reported below.")
    L.append("- **Sign:** registered POSITIVE (high-rate currencies "
             "outperform — the carry premium). `sided='greater'`.")
    L.append("- **Holding:** 1 week non-overlapping (matches the floor's h=5). "
             "Terciles over live pairs; dollar-neutral L/S; tier-keyed costs "
             f"`{json.dumps(spread_by_tier)}` round-trip bps; cost per week "
             "= Σ |Δw_i| · (spread_i / 2).")
    L.append("- **Decision:** `decision_ttest(sided='greater')` on the NET "
             "L/S weekly series via `walk_forward_pooled` "
             "(5 expanding folds, pool OOS — never vote folds).")
    L.append("")

    L.append("## Part 1 — Carry panel + orientation verification")
    L.append("")
    L.append("Top-tercile and bottom-tercile placement (fraction of weeks "
             "the pair ranks in that tercile by primary carry signal):")
    L.append("")
    L.append("| pair | top tercile frac | bottom tercile frac |")
    L.append("|---|---:|---:|")
    for p in ["USDMXN", "USDZAR", "USDHUF", "USDPLN", "NZDUSD", "AUDUSD",
              "USDNOK", "USDCZK", "USDHKD", "USDCAD", "GBPUSD", "USDSEK",
              "EURUSD", "USDJPY", "USDCHF"]:
        if p in top_frac:
            L.append(f"| {p} | {top_frac[p]:.3f} | {bot_frac[p]:.3f} |")
    L.append("")
    L.append("USDMXN, USDZAR rank top tercile in essentially every week "
             "(MXN, ZAR high-yielders); USDCHF, USDJPY rank bottom tercile "
             "in essentially every week (CHF, JPY low-yielders). Orientation "
             "is verified.")
    L.append("")

    L.append("## Part 2 — Carry detection floor")
    L.append("")
    L.append(f"Consistency check, NOT a fresh measurement. The carry panel "
             f"uses the SAME weekly grid (Friday-NY, non-overlapping h=5) "
             f"and SAME live universe (n=15) as the weekly momentum run, "
             f"so the floor MUST be identical:")
    L.append("")
    L.append(f"- n=15, rho=0.6, ~789 weeks: **rank-IC > {live_floor:.3f}**")
    L.append(f"- n=10 (early span and EM-outage week): floor "
             f"{early_floor:.3f}")
    L.append("")

    L.append("## Part 3 — Carry verdict")
    L.append("")
    m_p = bt_primary["metrics"]; m_s = bt_secondary["metrics"]
    L.append("| signal | n_weeks | gross IR | net IR | mean rank IC | "
             "hit rate (net) | turnover | max DD (net) | dollar β | R² |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    L.append(f"| primary (rate-diff) | {m_p['n_weeks']} | "
             f"{m_p['gross_ir']:+.2f} | {m_p['net_ir']:+.2f} | "
             f"{m_p['mean_rank_ic']:+.4f} | {m_p['hit_rate_net']:.2f} | "
             f"{m_p['mean_turnover']:.2f} | {m_p['max_drawdown_net']:+.3f} | "
             f"{m_p['dollar_beta']:+.3f} | {m_p['dollar_r2']:.3f} |")
    L.append(f"| secondary (CIP-fwd)  | {m_s['n_weeks']} | "
             f"{m_s['gross_ir']:+.2f} | {m_s['net_ir']:+.2f} | "
             f"{m_s['mean_rank_ic']:+.4f} | {m_s['hit_rate_net']:.2f} | "
             f"{m_s['mean_turnover']:.2f} | {m_s['max_drawdown_net']:+.3f} | "
             f"{m_s['dollar_beta']:+.3f} | {m_s['dollar_r2']:.3f} |")
    L.append("")
    L.append(f"Carry's per-week turnover ({m_p['mean_turnover']:.2f}) is "
             f"sharply lower than weekly momentum's (~1.0–2.0) — the rate "
             "differential changes slowly, so net IR is much closer to gross "
             "IR.")
    L.append("")

    L.append("### Primary verdict (rate-diff carry)")
    L.append("")
    L.append(f"- Pooled OOS one-sided: t = {verdict_pooled['t_stat']:+.2f}, "
             f"p = {verdict_pooled['p_value']:.4f}, "
             f"**fires = {verdict_pooled['fires']}**")
    L.append(f"- Full-sample net one-sided: t = {verdict_full_net['t_stat']:+.2f}, "
             f"p = {verdict_full_net['p_value']:.4f}")
    L.append(f"- Full-sample rank-IC one-sided: t = {verdict_full_ric['t_stat']:+.2f}, "
             f"p = {verdict_full_ric['p_value']:.4f}")
    L.append(f"- Fold dispersion (Sharpe, diagnostic only): "
             f"min = {wf_dispersion['min']:+.2f}, "
             f"median = {wf_dispersion['median']:+.2f}, "
             f"max = {wf_dispersion['max']:+.2f}")
    L.append("")

    L.append("### report_windows: full vs clean vs n=15-only")
    L.append("")
    L.append("| view | n_weeks | net IR | mean rank IC |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| full                                       | {full_n} | "
             f"{full_ir:+.2f} | {full_ric_mean:+.4f} |")
    L.append(f"| clean (excl COVID Q1 2020 + Ukraine Q1 2022) | {clean_n} | "
             f"{clean_ir:+.2f} | {clean_ric_mean:+.4f} |")
    L.append(f"| n=15 only span (excl early n=10 + EM outage) | {n15_n} | "
             f"{n15_ir:+.2f} | {n15_ric_mean:+.4f} |")
    L.append("")

    L.append("### Secondary (CIP-fwd, BH-corrected)")
    L.append("")
    L.append("| signal | one-sided p | BH-fires |")
    L.append("|---|---:|---:|")
    L.append(f"| primary (rate-diff) | {bh_p[0]:.4f} | {bool(bh[0])} |")
    L.append(f"| secondary (CIP-fwd) | {bh_p[1]:.4f} | {bool(bh[1])} |")
    L.append("")
    L.append(f"- Within-pair correlation between primary and secondary "
             f"(mean over pairs): {basis_corr:+.4f}.")
    L.append(f"- Mean absolute basis |primary − secondary| (in primary's "
             f"% units): {basis_summary:.4f}.")
    L.append("")

    L.append("## GO/NO-GO (primary)")
    L.append("")
    L.append(f"- Floor: net rank-IC > **{live_floor:.3f}**")
    L.append(f"- Measured primary gross rank-IC: **{primary_gross_ic:+.4f}**")
    L.append(f"- Measured primary net rank-IC:   **{primary_net_ic:+.4f}**")
    L.append("")
    L.append(f"**Outcome: {outcome}**")
    L.append("")
    L.append("Outcomes spelled out:")
    L.append("- (i) net > floor AND pooled OOS t-test fires → real, proceed.")
    L.append("- (ii) gross > floor but NET < floor → signal real but eaten "
             "by costs.")
    L.append("- (iii) gross < floor → not detectable at this breadth.")
    L.append("- (iv) net > floor but t-test doesn't fire → signal small or "
             "HAC-SE too inflated.")
    L.append("")

    L.append("## Part 4 — Carry-vs-reversal overlap (DIAGNOSTIC ONLY)")
    L.append("")
    L.append("> **GUARD:** This section renders **no verdict on the reversal "
             "hypothesis**. The momentum / negated-momentum IC series is "
             "computed only to measure its week-by-week relationship with "
             "carry. Reversal-as-hypothesis requires a separate-universe or "
             "forward holdout and is NOT confirmed here.")
    L.append("")
    L.append(f"Common weeks: **{n_common_ic}**. Correlations between the "
             f"weekly CARRY rank-IC series and the NEGATED weekly MOMENTUM "
             f"rank-IC series (L=8w, the registered momentum primary):")
    L.append("")
    L.append(f"- Pearson  ρ = **{pearson_corr:+.4f}**")
    L.append(f"- Spearman ρ = **{spearman_corr:+.4f}**")
    L.append("")
    L.append("Regression of weekly CARRY NET L/S returns on weekly MOMENTUM "
             "NET L/S returns:")
    L.append("")
    L.append(f"- β = **{ovp_beta:+.3f}**   R² = {ovp_r2:.3f}")
    L.append(f"- Weekly intercept (carry alpha vs momentum) = {ovp_alpha:+.6f}")
    L.append(f"- Residual alpha-t = {ovp_alpha_t:+.2f}, one-sided "
             f"p = {ovp_alpha_p:.4f}")
    L.append("")
    L.append("Interpretation table:")
    L.append("")
    L.append("- **World A (same coin):** high IC correlation with negated "
             "momentum AND small carry alpha vs momentum → \"reversal\" was "
             "carry.")
    L.append("- **World B (two effects):** low/zero IC correlation OR carry "
             "retains alpha net of momentum → distinct effects; reversal "
             "earns a separate registered out-of-sample test.")
    L.append("- **World C (resolution-bound):** carry also NO-GO at floor → "
             "constraint is breadth/resolution; reversal requires a separate "
             "test regardless.")
    L.append("")
    L.append(f"**Reading: {world}**")
    L.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    run()
