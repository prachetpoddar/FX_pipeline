# Layer-3 Factor 1 — Weekly cross-sectional FX momentum

Branch: `weekly-resampler`. Read-only on `data/processed_v2/` and `data/weekly/`. All verdicts via `src/stage2_agents/evaluation.py`.

Universe (n=15): EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD, USDSEK, USDNOK, USDMXN, USDZAR, USDPLN, USDHUF, USDCZK, USDHKD. Per-week breadth: min=9, median=15, max=15.

## Pre-registration

- Primary formation: **L = 8 weeks** (registered before any backtest).
- Secondary formations (pre-registered, BH-corrected, robustness only): (2, 4, 13, 26).
- Holding: 1 week non-overlapping (matches the floor's h=5).
- Decision: `decision_ttest(sided='greater')` via `walk_forward_pooled` (5 folds, expanding, pool OOS — never vote folds).
- Costs (tier-keyed, round-trip bps): `{"1": 3.0, "2": 20.0, "3": 30.0}`. Cost per period = Σ |Δw_i| · (spread_i / 2).

## Part 2 — Weekly detection floor (recap)

- At n=15, rho=0.6, ~789 weeks: **net rank-IC > 0.020** for detection rate ≥ 0.80, `sided='greater'`.
- At n=10 (early-2010 listing gap + 2017-02-24 EM-outage week): floor = 0.030.
- vs the monthly floor of 0.050 — weekly resolution buys ~2.5× more data so the floor drops by roughly the expected √4≈2 factor.

See `detection_floor_weekly.csv` and `detection_floor_weekly_summary.csv`.

## Part 3 — Per-formation backtest (terciles, dollar-neutral)

| L (weeks) | n_weeks | gross IR | net IR | mean rank IC | hit rate (net) | turnover | max DD (net) | dollar β | dollar R² |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 834 | -0.56 | -1.30 | -0.0331 | 0.41 | 1.89 | -1.462 | -0.136 | 0.023 |
| 4 | 834 | -0.62 | -1.17 | -0.0380 | 0.42 | 1.36 | -1.253 | -0.128 | 0.023 |
| 8 | 834 | -0.46 | -0.83 | -0.0280 | 0.45 | 0.99 | -0.969 | -0.153 | 0.029 |
| 13 | 834 | -0.56 | -0.87 | -0.0288 | 0.44 | 0.84 | -1.005 | -0.159 | 0.032 |
| 26 | 834 | -0.32 | -0.54 | -0.0127 | 0.45 | 0.61 | -0.651 | -0.189 | 0.045 |

### Primary verdict (L=8w)

- Pooled OOS one-sided: t = -3.84, p = 0.9999, **fires = False**
- Full-sample net one-sided: t = -3.76, p = 0.9999
- Full-sample rank-IC one-sided: t = -2.23, p = 0.9871
- Fold dispersion (Sharpe, diagnostic only — never the verdict): min = -1.62, median = -0.77, max = -0.67

### report_windows: full vs clean-subperiod vs n=15-only span

| view | n_weeks | net IR | mean rank IC |
|---|---:|---:|---:|
| full                                       | 834 | -0.83 | -0.0280 |
| clean (excl COVID Q1 2020 + Ukraine Q1 2022) | 814 | -0.95 | -0.0279 |
| n=15 only span (excl early n=10 + EM outage) | 751 | -0.81 | -0.0280 |

Reading: if any of these views is materially different from the full view, it tells us whether the signal is regime- or breadth-dependent. Identical → robust to that cut.

### Secondary (BH-FDR, robustness only — does NOT override primary)

| L | one-sided p | BH-fires |
|---|---:|---:|
| L2w | 1.0000 | False |
| L4w | 1.0000 | False |
| L13w | 1.0000 | False |
| L26w | 0.9909 | False |

## GO/NO-GO

- Floor (Part 2, n=15): net rank-IC > **0.020**
- Measured primary gross rank-IC: **-0.0280**
- Measured primary net rank-IC:   **-0.0280**

**Outcome: (iii) NO-GO on momentum — gross IC (-0.0280) < floor (0.020). NB: IC is NEGATIVE, consistent with cross-sectional REVERSAL rather than momentum; magnitude still below the floor so reversal is also not confirmable**

Outcomes spelled out (per brief):
- (i) net > floor AND pooled OOS t-test fires → real, proceed.
- (ii) gross > floor but NET < floor → signal real but eaten by costs.
- (iii) gross < floor → not detectable at this breadth; if gross IC is negative the data leans towards cross-sectional reversal (noted explicitly).
- (iv) net > floor but t-test doesn't fire → signal small or HAC-SE too inflated.
