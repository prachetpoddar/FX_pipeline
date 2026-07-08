"""
Layer-3 momentum — end-to-end driver.

  - PART 1: orient daily returns, build monthly panel.
  - PART 2: per-formation backtest (gross + net + dollar-beta).
  - PART 3: primary verdict (L=3) via walk_forward_pooled on NET; secondary
            BH-FDR across {1,6,12}; report_windows full vs clean-subperiod.
  - Writes data/layer3_momentum/{monthly_panel.parquet, factor_backtest.csv,
            per_formation_<L>_returns.csv, MOMENTUM_REPORT.md}.

Read-only on data/processed_v2/. No data-store writes.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.universe import UNIVERSE  # noqa: E402
from stage2_agents.evaluation import (  # noqa: E402
    decision_ttest,
    rank_ic_series,
    walk_forward_pooled,
    report_windows,
    decide_family,
    portfolio_return_series,
)
from stage3_signals.momentum import (  # noqa: E402
    load_universe_returns,
    orient_returns,
    orientation_map,
    build_monthly_panel,
    backtest_formation,
    currency_to_pair_map,
    cost_per_pair_bps,
    ALL_FORMATIONS_MONTHS,
    PRIMARY_FORMATION_MONTHS,
    SECONDARY_FORMATIONS_MONTHS,
    CLEAN_EXCLUDE_MONTHS,
    DEFAULT_SPREAD_BPS_BY_TIER,
)
from stage3_signals.detection_floor import main as run_floor

OUT = PROJ / "data" / "layer3_momentum"
OUT.mkdir(parents=True, exist_ok=True)


def make_clean_mask(index: pd.DatetimeIndex, exclude_yyyymm) -> pd.Series:
    keep = pd.Series(True, index=index)
    for m in exclude_yyyymm:
        yyyy, mm = m.split("-")
        mask = (index.year == int(yyyy)) & (index.month == int(mm))
        keep[mask] = False
    return keep


def run(parquet_path=None, alpha=0.05):
    raw = load_universe_returns(
        parquet_path if parquet_path else None
    ) if parquet_path else load_universe_returns()
    pairs = list(raw.columns)
    n_live = len(pairs)
    print(f"[Layer 3 momentum] live universe size = {n_live}: {pairs}")

    # --- PART 0 ---
    print("\n[PART 0] Detection floor at operating conditions")
    live_floor, bh_floor = run_floor(live_n_inst=n_live)

    # --- PART 1 ---
    print("\n[PART 1] Orientation + monthly panel")
    omap = orientation_map(pairs)
    oriented = orient_returns(raw)
    # Sanity asserts — Part 1 verification gates everything else
    eur_corr = raw["EURUSD"].corr(oriented["EUR"])
    jpy_corr = raw["USDJPY"].corr(oriented["JPY"])
    usd_idx_pairs = [p for p in pairs if UNIVERSE[p]["base"] == "USD"]
    usd_idx = raw[usd_idx_pairs].mean(axis=1)
    mean_or = oriented.mean(axis=1)
    com = mean_or.dropna().index.intersection(usd_idx.dropna().index)
    usd_corr = mean_or.loc[com].corr(usd_idx.loc[com])
    print(f"  corr(EURUSD raw, EUR oriented) = {eur_corr:+.4f}  (expect +1)")
    print(f"  corr(USDJPY raw, JPY oriented) = {jpy_corr:+.4f}  (expect -1)")
    print(f"  corr(mean oriented, USD idx)   = {usd_corr:+.4f}  (expect strongly negative)")
    assert eur_corr > 0.999, "EUR orientation broken"
    assert jpy_corr < -0.999, "JPY orientation broken"
    assert usd_corr < -0.9, "mean oriented should track -USD"

    panel = build_monthly_panel(oriented, ALL_FORMATIONS_MONTHS)
    print(f"  month-ends: {len(panel['month_ends'])}; "
          f"fwd_ret shape: {panel['fwd_ret'].shape}")
    # Write as long-format parquet (m, currency, L1, L3, L6, L12, fwd_ret)
    rows = []
    for m in panel["fwd_ret"].index:
        for c in panel["fwd_ret"].columns:
            row = {"month_end": m, "currency": c,
                   "fwd_ret": panel["fwd_ret"].loc[m, c]}
            for L, sig in panel["signals"].items():
                row[f"signal_L{L}"] = sig.loc[m, c]
            rows.append(row)
    pd.DataFrame(rows).to_parquet(OUT / "monthly_panel.parquet")

    # --- PART 2 ---
    print("\n[PART 2] Per-formation backtest (gross + net + dollar-beta)")
    c2p = currency_to_pair_map(pairs=pairs)
    spread_by_currency = cost_per_pair_bps(c2p)
    results = {}
    summary_rows = []
    for L in ALL_FORMATIONS_MONTHS:
        sig = panel["signals"][L]
        fwd = panel["fwd_ret"]
        bt = backtest_formation(sig, fwd, q=1.0 / 3.0,
                                spread_bps=spread_by_currency)
        results[L] = bt
        m = bt["metrics"]
        marker = " *PRIMARY*" if L == PRIMARY_FORMATION_MONTHS else ""
        print(f"  L={L:>2}m{marker}  "
              f"gross_IR={m['gross_ir']:+.2f}  "
              f"net_IR={m['net_ir']:+.2f}  "
              f"rank_IC={m['mean_rank_ic']:+.4f}  "
              f"turnover={m['mean_turnover']:.2f}  "
              f"dollar_beta={m['dollar_beta']:+.3f} "
              f"R2={m['dollar_r2']:.3f}")
        summary_rows.append({"formation_months": L, **m})
        # Per-formation monthly series
        per = pd.DataFrame({
            "gross": bt["gross"],
            "cost": bt["cost"],
            "net": bt["net"],
            "turnover": bt["turnover"],
        })
        per.to_csv(OUT / f"per_formation_L{L}_returns.csv")

    pd.DataFrame(summary_rows).to_csv(OUT / "factor_backtest.csv", index=False)

    # --- PART 3 ---
    print("\n[PART 3] Verdict through the calibrated judge")
    primary = results[PRIMARY_FORMATION_MONTHS]
    net_primary = primary["net"].dropna()
    gross_primary = primary["gross"].dropna()
    ric_primary = primary["rank_ic_series"].dropna()

    # Walk-forward pooled on the NET LS return series (timeseries axis).
    # signal_fn returns oos_signal panel = test_S unchanged; we are
    # evaluating the realized portfolio return series directly via a single
    # "instrument" portfolio. To use walk_forward_pooled in timeseries mode
    # we wrap the LS series as a one-column panel.
    ls_panel = pd.DataFrame({"LS": net_primary})
    sig_panel = pd.DataFrame({"LS": np.ones_like(net_primary.values)},
                             index=net_primary.index)
    def passthrough(train_S, train_F, test_S, test_F):
        return test_S
    wf = walk_forward_pooled(
        passthrough, sig_panel, ls_panel,
        n_folds=5, scheme="expanding",
        alpha=alpha, axis="timeseries", periods_per_year=12,
    )
    # walk_forward_pooled internally calls decision_ttest two-sided; we need
    # one-sided 'greater' on the same pooled series for the primary verdict.
    pooled = wf["pooled_series"].dropna()
    verdict_pooled_one = decision_ttest(pooled.values, alpha=alpha,
                                        sided="greater")
    # Also a one-sided test on the in-sample net series (descriptive)
    verdict_full_one = decision_ttest(net_primary.values, alpha=alpha,
                                      sided="greater")
    # And on rank-IC series
    verdict_ric_one = decision_ttest(ric_primary.values, alpha=alpha,
                                     sided="greater")

    print(f"  primary (L={PRIMARY_FORMATION_MONTHS}m):")
    print(f"    pooled WF OOS one-sided t={verdict_pooled_one['t_stat']:+.2f} "
          f"p={verdict_pooled_one['p_value']:.4f} fires={verdict_pooled_one['fires']}")
    print(f"    full-sample net  one-sided t={verdict_full_one['t_stat']:+.2f} "
          f"p={verdict_full_one['p_value']:.4f} fires={verdict_full_one['fires']}")
    print(f"    full-sample rank-IC one-sided t={verdict_ric_one['t_stat']:+.2f} "
          f"p={verdict_ric_one['p_value']:.4f} fires={verdict_ric_one['fires']}")
    print(f"    fold dispersion (sharpe): min={wf['dispersion']['min']:+.2f} "
          f"median={wf['dispersion']['median']:+.2f} "
          f"max={wf['dispersion']['max']:+.2f}")

    # Clean-subperiod report
    clean_mask = make_clean_mask(net_primary.index, CLEAN_EXCLUDE_MONTHS)
    ric_full_series = ric_primary
    clean_idx = net_primary.index[clean_mask.reindex(net_primary.index).fillna(False).values]
    net_clean = net_primary.loc[net_primary.index.intersection(clean_idx)]
    ric_clean = ric_primary.loc[ric_primary.index.intersection(clean_idx)]
    n_excl = len(net_primary) - len(net_clean)
    print(f"  clean-subperiod report (excluding {n_excl} months "
          f"in {CLEAN_EXCLUDE_MONTHS}):")
    full_ir = float(net_primary.mean() / net_primary.std(ddof=1) * np.sqrt(12))
    clean_ir = float(net_clean.mean() / net_clean.std(ddof=1) * np.sqrt(12)) \
        if net_clean.std(ddof=1) > 0 else float("nan")
    full_ric = float(ric_full_series.mean())
    clean_ric = float(ric_clean.mean())
    print(f"    full   IR={full_ir:+.2f}  mean rank_IC={full_ric:+.4f}  "
          f"(n_months={len(net_primary)})")
    print(f"    clean  IR={clean_ir:+.2f}  mean rank_IC={clean_ric:+.4f}  "
          f"(n_months={len(net_clean)})")

    # Secondary BH-FDR across {1, 6, 12}
    print(f"  secondary (BH-FDR across L in {SECONDARY_FORMATIONS_MONTHS}):")
    sec_series = []
    sec_labels = []
    for L in SECONDARY_FORMATIONS_MONTHS:
        s = results[L]["net"].dropna().values
        sec_series.append(s)
        sec_labels.append(f"L{L}")
    sec_fam = decide_family(sec_series, alpha=alpha, use_hac=True)
    # decide_family is two-sided; re-derive one-sided 'greater' p-values
    sec_one_p = []
    for s in sec_series:
        v = decision_ttest(s, alpha=alpha, sided="greater")
        sec_one_p.append(v["p_value"])
    from stage2_agents.evaluation import bh_fdr_decision
    sec_bh = bh_fdr_decision(sec_one_p, alpha=alpha)
    for lab, p, fires in zip(sec_labels, sec_one_p, sec_bh):
        print(f"    {lab}: one-sided p={p:.4f}  BH-fires={bool(fires)}")

    # GO/NO-GO outcome
    primary_net_ic = float(ric_primary.mean())
    primary_gross_ic = float(rank_ic_series(panel["signals"][PRIMARY_FORMATION_MONTHS],
                                            panel["fwd_ret"]).mean())
    print()
    print(f"  GO/NO-GO THRESHOLD (Part 0): net rank-IC must exceed "
          f"{live_floor:.3f} at n={n_live}")
    print(f"  Measured primary net rank-IC   = {primary_net_ic:+.4f}")
    print(f"  Measured primary gross rank-IC = {primary_gross_ic:+.4f}")
    if primary_net_ic > live_floor and verdict_pooled_one["fires"]:
        outcome = "(i) GO  — net IC > floor AND pooled OOS t-test fires"
    elif primary_gross_ic > live_floor and primary_net_ic <= live_floor:
        outcome = ("(ii) HALT — gross IC > floor but NET < floor; signal "
                   "real but eaten by costs (report cost wedge)")
    elif primary_gross_ic <= live_floor:
        # Sub-case: negative IC observed (cross-sectional REVERSAL evidence
        # rather than absence of signal). Report it explicitly — does not
        # change the verdict on the registered momentum hypothesis, but it
        # is information for Factor-2 design.
        if primary_gross_ic < 0:
            outcome = (f"(iii) NO-GO on momentum — gross IC ({primary_gross_ic:+.4f}) "
                       f"< floor ({live_floor:.3f}). NB: IC is NEGATIVE, consistent "
                       "with cross-sectional REVERSAL rather than momentum; the "
                       "magnitude is still below the floor so the reversal signal "
                       "is also not confirmable from this sample alone")
        else:
            outcome = ("(iii) NO-GO — gross IC < floor; not detectable at "
                       "this breadth, stop and reconsider universe/horizon")
    else:
        outcome = ("(iv) AMBIGUOUS — net IC above floor but t-test does "
                   "not fire (signal small or HAC-SE too inflated)")
    print(f"  OUTCOME: {outcome}")

    # Write MOMENTUM_REPORT.md
    write_report(
        out_path=OUT / "MOMENTUM_REPORT.md",
        pairs=pairs,
        omap=omap,
        eur_corr=eur_corr, jpy_corr=jpy_corr, usd_corr=usd_corr,
        live_floor=live_floor, bh_floor=bh_floor,
        summary_rows=summary_rows,
        primary_net_ic=primary_net_ic,
        primary_gross_ic=primary_gross_ic,
        verdict_pooled_one=verdict_pooled_one,
        verdict_full_one=verdict_full_one,
        verdict_ric_one=verdict_ric_one,
        wf_dispersion=wf["dispersion"],
        full_ir=full_ir, clean_ir=clean_ir,
        full_ric=full_ric, clean_ric=clean_ric,
        n_full=len(net_primary), n_clean=len(net_clean),
        sec_labels=sec_labels, sec_p=sec_one_p, sec_bh=sec_bh,
        outcome=outcome,
        spread_by_tier=DEFAULT_SPREAD_BPS_BY_TIER,
    )
    print(f"\n  Wrote {OUT/'MOMENTUM_REPORT.md'}")
    return {
        "outcome": outcome,
        "verdict_pooled_one": verdict_pooled_one,
        "primary_net_ic": primary_net_ic,
        "primary_gross_ic": primary_gross_ic,
        "live_floor": live_floor,
    }


def write_report(out_path, pairs, omap, eur_corr, jpy_corr, usd_corr,
                 live_floor, bh_floor, summary_rows,
                 primary_net_ic, primary_gross_ic,
                 verdict_pooled_one, verdict_full_one, verdict_ric_one,
                 wf_dispersion, full_ir, clean_ir, full_ric, clean_ric,
                 n_full, n_clean, sec_labels, sec_p, sec_bh, outcome,
                 spread_by_tier):
    L = []
    L.append("# Layer-3 Factor 1 — Cross-sectional FX momentum")
    L.append("")
    L.append("Branch: `layer3-momentum`. Read-only on `data/processed_v2/`. "
             "All verdicts via `src/stage2_agents/evaluation.py`.")
    L.append("")
    L.append(f"Live universe (n={len(pairs)}): {', '.join(pairs)}.  "
             "USDTRY absent (held-for-review).")
    L.append("")

    L.append("## Part 0 — Detection floor at operating conditions")
    L.append("")
    L.append(f"At the live universe size and a monthly rebalance "
             f"(h=21 trading days, ~119 non-overlapping monthly obs over "
             f"the 10-year sample), `rank_ic_series` + `decision_ttest "
             f"sided='greater'` requires **rank-IC > {live_floor:.3f}** for "
             f"detection rate >= 0.80 at rho=0.6 (the conservative case).")
    L.append("")
    L.append(f"BH-FDR-corrected floor across K=4 formations: "
             f"~{bh_floor if np.isfinite(bh_floor) else 'undetectable in grid'}. "
             f"Used for context only — the primary test is a single "
             f"pre-registered formation so no multiplicity penalty applies.")
    L.append("")
    L.append("See `detection_floor.csv` and `detection_floor_summary.csv`.")
    L.append("")

    L.append("## Part 1 — Orientation verification")
    L.append("")
    L.append(f"- corr(EURUSD raw, EUR oriented) = **{eur_corr:+.4f}** "
             "(expect +1)")
    L.append(f"- corr(USDJPY raw, JPY oriented) = **{jpy_corr:+.4f}** "
             "(expect -1)")
    L.append(f"- corr(mean oriented, USD index from USDXXX pairs) = "
             f"**{usd_corr:+.4f}** (expect strongly negative)")
    L.append("")
    L.append("Orientation map (sign applied to raw return):")
    L.append("")
    L.append("| pair | sign | strengthens-vs-USD currency |")
    L.append("|---|---:|---|")
    for p, s in omap.items():
        meta = UNIVERSE[p]
        ccy = meta["base"] if s == +1 else meta["quote"]
        L.append(f"| {p} | {s:+d} | {ccy} |")
    L.append("")

    L.append("## Part 2 — Per-formation backtest (terciles, dollar-neutral)")
    L.append("")
    L.append(f"Transaction costs: tier-keyed round-trip spreads (bps): "
             f"{json.dumps(spread_by_tier)}. "
             "Cost per period = sum |Δw_i| * (spread_i / 2).")
    L.append("")
    L.append("| L | n | gross IR | net IR | mean rank IC | hit rate (net) | "
             "turnover | max DD (net) | dollar β | dollar R² |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        L.append(
            f"| {row['formation_months']} | {row['n_months']} | "
            f"{row['gross_ir']:+.2f} | {row['net_ir']:+.2f} | "
            f"{row['mean_rank_ic']:+.4f} | {row['hit_rate_net']:.2f} | "
            f"{row['mean_turnover']:.2f} | {row['max_drawdown_net']:+.3f} | "
            f"{row['dollar_beta']:+.3f} | {row['dollar_r2']:.3f} |"
        )
    L.append("")

    L.append("## Part 3 — Verdict through the calibrated judge")
    L.append("")
    L.append(f"**Primary (pre-registered, single test):** "
             f"L={PRIMARY_FORMATION_MONTHS} months, "
             "walk_forward_pooled on NET LS monthly returns "
             "(expanding, 5 folds), one-sided `sided='greater'`.")
    L.append("")
    L.append(f"- Pooled OOS one-sided: t={verdict_pooled_one['t_stat']:+.2f}, "
             f"p={verdict_pooled_one['p_value']:.4f}, "
             f"fires={verdict_pooled_one['fires']}")
    L.append(f"- Full-sample net one-sided: t={verdict_full_one['t_stat']:+.2f}, "
             f"p={verdict_full_one['p_value']:.4f}, "
             f"fires={verdict_full_one['fires']}")
    L.append(f"- Full-sample rank-IC one-sided: t={verdict_ric_one['t_stat']:+.2f}, "
             f"p={verdict_ric_one['p_value']:.4f}, "
             f"fires={verdict_ric_one['fires']}")
    L.append(f"- Fold dispersion (Sharpe): "
             f"min={wf_dispersion['min']:+.2f}, "
             f"median={wf_dispersion['median']:+.2f}, "
             f"max={wf_dispersion['max']:+.2f} "
             "(stability diagnostic only — fold votes are NEVER the verdict)")
    L.append("")
    L.append("**Clean-subperiod (report_windows):**")
    L.append("")
    L.append(f"| view | n_months | net IR | mean rank IC |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| full       | {n_full}  | {full_ir:+.2f} | {full_ric:+.4f} |")
    L.append(f"| clean      | {n_clean} | {clean_ir:+.2f} | {clean_ric:+.4f} |")
    L.append("")
    L.append(f"Clean excludes: {', '.join(CLEAN_EXCLUDE_MONTHS)} "
             "(COVID + Russia/Ukraine).")
    L.append("")
    L.append(f"**Secondary (BH-FDR across L in "
             f"{SECONDARY_FORMATIONS_MONTHS}, robustness only):**")
    L.append("")
    L.append("| L | one-sided p | BH-fires |")
    L.append("|---|---:|---:|")
    for lab, p, fires in zip(sec_labels, sec_p, sec_bh):
        L.append(f"| {lab} | {p:.4f} | {bool(fires)} |")
    L.append("")

    L.append("## GO/NO-GO")
    L.append("")
    L.append(f"- Floor (Part 0): net rank-IC > **{live_floor:.3f}**")
    L.append(f"- Measured primary gross rank-IC: **{primary_gross_ic:+.4f}**")
    L.append(f"- Measured primary net rank-IC:   **{primary_net_ic:+.4f}**")
    L.append("")
    L.append(f"**Outcome: {outcome}**")
    L.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    run()
