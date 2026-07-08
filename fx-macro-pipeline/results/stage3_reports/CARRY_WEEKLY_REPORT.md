# Layer-3 Factor 2 — Weekly cross-sectional FX carry

Branch: `layer3-carry`. Read-only on `data/processed_v2/`. Reuses the shared weekly resampler (`src/stage1_collection_v2/weekly_panel.py`) and the calibrated judge (`src/stage2_agents/evaluation.py`).

Universe (n=15): EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD, USDSEK, USDNOK, USDMXN, USDZAR, USDPLN, USDHUF, USDCZK, USDHKD.

## Pre-registration

- **Primary signal:** lagged short-rate differential (`carry_signal_pct` keyed by the non-USD currency of each pair, lagged 1 trading day in source so values are point-in-time at the daily index and at Friday-NY week-ends).
- **Secondary signal (pre-registered, BH-reported, robustness):** CIP-forward-implied carry from `cip_fwd_points_pair` / spot; sign-corrected via `-orientation(pair) × cip / spot` so positive = high yielder. The basis (primary − secondary) is the cross-currency basis and reported below.
- **Sign:** registered POSITIVE (high-rate currencies outperform — the carry premium). `sided='greater'`.
- **Holding:** 1 week non-overlapping (matches the floor's h=5). Terciles over live pairs; dollar-neutral L/S; tier-keyed costs `{"1": 3.0, "2": 20.0, "3": 30.0}` round-trip bps; cost per week = Σ |Δw_i| · (spread_i / 2).
- **Decision:** `decision_ttest(sided='greater')` on the NET L/S weekly series via `walk_forward_pooled` (5 expanding folds, pool OOS — never vote folds).

## Part 1 — Carry panel + orientation verification

Top-tercile and bottom-tercile placement (fraction of weeks the pair ranks in that tercile by primary carry signal):

| pair | top tercile frac | bottom tercile frac |
|---|---:|---:|
| USDMXN | 1.000 | 0.000 |
| USDZAR | 1.000 | 0.000 |
| USDHUF | 0.697 | 0.203 |
| USDPLN | 0.859 | 0.000 |
| NZDUSD | 0.658 | 0.000 |
| AUDUSD | 0.582 | 0.219 |
| USDNOK | 0.338 | 0.011 |
| USDCZK | 0.364 | 0.258 |
| USDHKD | 0.508 | 0.233 |
| USDCAD | 0.020 | 0.211 |
| GBPUSD | 0.031 | 0.147 |
| USDSEK | 0.000 | 0.765 |
| EURUSD | 0.034 | 0.865 |
| USDJPY | 0.000 | 0.966 |
| USDCHF | 0.000 | 1.000 |

USDMXN, USDZAR rank top tercile in essentially every week (MXN, ZAR high-yielders); USDCHF, USDJPY rank bottom tercile in essentially every week (CHF, JPY low-yielders). Orientation is verified.

## Part 2 — Carry detection floor

Consistency check, NOT a fresh measurement. The carry panel uses the SAME weekly grid (Friday-NY, non-overlapping h=5) and SAME live universe (n=15) as the weekly momentum run, so the floor MUST be identical:

- n=15, rho=0.6, ~789 weeks: **rank-IC > 0.020**
- n=10 (early span and EM-outage week): floor 0.030

## Part 3 — Carry verdict

| signal | n_weeks | gross IR | net IR | mean rank IC | hit rate (net) | turnover | max DD (net) | dollar β | R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary (rate-diff) | 834 | -0.22 | -0.24 | +0.0182 | 0.51 | 0.05 | -0.425 | +0.378 | 0.146 |
| secondary (CIP-fwd)  | 834 | -0.21 | -0.24 | +0.0182 | 0.51 | 0.05 | -0.425 | +0.379 | 0.149 |

Carry's per-week turnover (0.05) is sharply lower than weekly momentum's (~1.0–2.0) — the rate differential changes slowly, so net IR is much closer to gross IR.

### Primary verdict (rate-diff carry)

- Pooled OOS one-sided: t = -0.71, p = 0.7618, **fires = False**
- Full-sample net one-sided: t = -1.05, p = 0.8524
- Full-sample rank-IC one-sided: t = +1.38, p = 0.0831
- Fold dispersion (Sharpe, diagnostic only): min = -0.64, median = -0.07, max = +0.53

### report_windows: full vs clean vs n=15-only

| view | n_weeks | net IR | mean rank IC |
|---|---:|---:|---:|
| full                                       | 834 | -0.24 | +0.0182 |
| clean (excl COVID Q1 2020 + Ukraine Q1 2022) | 814 | -0.12 | +0.0212 |
| n=15 only span (excl early n=10 + EM outage) | 751 | -0.22 | +0.0182 |

### Secondary (CIP-fwd, BH-corrected)

| signal | one-sided p | BH-fires |
|---|---:|---:|
| primary (rate-diff) | 0.8524 | False |
| secondary (CIP-fwd) | 0.8475 | False |

- Within-pair correlation between primary and secondary (mean over pairs): +1.0000.
- Mean absolute basis |primary − secondary| (in primary's % units): 1.7632.

## GO/NO-GO (primary)

- Floor: net rank-IC > **0.020**
- Measured primary gross rank-IC: **+0.0182**
- Measured primary net rank-IC:   **+0.0182**

**Outcome: (iii) NO-GO — gross IC < floor; not detectable at this breadth**

Outcomes spelled out:
- (i) net > floor AND pooled OOS t-test fires → real, proceed.
- (ii) gross > floor but NET < floor → signal real but eaten by costs.
- (iii) gross < floor → not detectable at this breadth.
- (iv) net > floor but t-test doesn't fire → signal small or HAC-SE too inflated.

## Part 4 — Carry-vs-reversal overlap (DIAGNOSTIC ONLY)

> **GUARD:** This section renders **no verdict on the reversal hypothesis**. The momentum / negated-momentum IC series is computed only to measure its week-by-week relationship with carry. Reversal-as-hypothesis requires a separate-universe or forward holdout and is NOT confirmed here.

Common weeks: **672**. Correlations between the weekly CARRY rank-IC series and the NEGATED weekly MOMENTUM rank-IC series (L=8w, the registered momentum primary):

- Pearson  ρ = **+0.1226**
- Spearman ρ = **+0.1170**

Regression of weekly CARRY NET L/S returns on weekly MOMENTUM NET L/S returns:

- β = **-0.305**   R² = 0.077
- Weekly intercept (carry alpha vs momentum) = -0.000680
- Residual alpha-t = -0.00, one-sided p = 0.5000

Interpretation table:

- **World A (same coin):** high IC correlation with negated momentum AND small carry alpha vs momentum → "reversal" was carry.
- **World B (two effects):** low/zero IC correlation OR carry retains alpha net of momentum → distinct effects; reversal earns a separate registered out-of-sample test.
- **World C (resolution-bound):** carry also NO-GO at floor → constraint is breadth/resolution; reversal requires a separate test regardless.

**Reading: World C (resolution-bound): carry's gross IC is below the floor at this breadth/resolution, so the diagnostic above is not load-bearing — the binding constraint is breadth. Reversal-as-hypothesis requires a separate-universe or forward holdout regardless**
