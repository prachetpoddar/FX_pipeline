# Layer-2 Remediation — Changes

Branch: `layer2-remediation`
Driver: `data/layer2_audit/LAYER2_AUDIT_REPORT.md` (the power audit; six tasks A–F)

The audit characterized the existing Layer-2 evaluator as a false-negative
machine: detection rate = 0 at every IC level surveyed under the
sharpe>0.3, accuracy>0.55 pass rule (Task A); a missing cross-sectional
primitive (Task B); breadth collapse via cross-sectional correlation
(Task C); fold-vote walk-forward generating its own false negatives
(Task D); a validation gate that false-rejected 96–100% on three
benign-but-real cases (Task E); and a future-invariance test that was
vacuous under pandas 3.0.2 (Task F).

The fixes below are scoped exactly to those audit findings.

---

## FIX 1 — Replace the pass rule with a t-test decision layer

**File:** `src/stage2_agents/evaluation.py` (new)
**Audit ref:** Task A (det = 0 at every cell of the sharpe>0.3/acc>0.55 rule)

- New `decision_ttest(series, alpha=0.05)`: two-sided t-test of a
  per-period series vs 0, Newey-West HAC SE when `statsmodels` is
  installed (logged in the result), plain SE otherwise.
- New `evaluate_timeseries(...)`: returns the descriptive diagnostics
  the old `evaluate_signal` returned (accuracy, sharpe,
  `ic_in_position`, turnover, `time_in_market`) PLUS the t-test verdict.
  The verdict is the t-test; the diagnostics are descriptive only.
- New `portfolio_return_series(...)` and `evaluate_portfolio(...)`.
- New `ic_in_position(signal, fwd_return)`: IC computed only over rows
  where `signal != 0`. The audit found full-series IC attenuates ~0.52x
  under 70% gating; this returns the honest in-position figure (plus
  the full-series figure for comparability).

**Old behavior:** sharpe>0.3 AND accuracy>0.55 → fire. Detection rate at
every cell = 0 (Task A). This rule is NOT present anywhere in the new
module; it has been removed, not tuned.

**New behavior:** verdict is `decision_ttest` on the portfolio strategy
return series. Calibration locked by `tests/test_evaluator_calibration.py`.

## FIX 2 — Cross-sectional evaluator (net-new primary metric)

**File:** `src/stage2_agents/evaluation.py` (in the same module)
**Audit ref:** Task B (the primitive was absent from the pipeline)

- `rank_ic_series(signal_panel, fwd_panel)`: vectorized per-period
  Spearman rank correlation across instruments (matches the reference
  implementation in `data/layer2_audit/run_audit.py`).
- `long_short_return(signal_panel, fwd_panel, q=0.2, periods_per_year=252)`:
  top-q minus bottom-q forward return per period plus annualized IR.
  `periods_per_year` is required, explicit, no defaulting based on
  inferred frequency.
- `evaluate_crosssectional(...)`: verdict is `decision_ttest` on the
  rank-IC series, plus the LS-IR as a descriptive number.

**Old behavior:** none — the cross-sectional axis did not exist.
**New behavior:** the cross-sectional rank-IC verdict is Layer 3's
primary metric. Calibration locked: det >= 0.8 at ic=0.01, n_inst=18,
rho=0.6, h=1.

## FIX 3 — Walk-forward: POOL, don't vote

**File:** `src/stage2_agents/evaluation.py` (`walk_forward_pooled`,
`report_windows`)
**Audit ref:** Task D (per-fold Sharpe range [-1.84, +2.43] on a genuine
ic=0.03 signal; fold-vote mode-failed; pooled view detected it)

- `walk_forward_pooled(signal_fn, signal_panel, fwd_panel, n_folds=5,
  scheme=...)`: emits per-fold OUT-OF-SAMPLE per-period series, then
  CONCATENATES them and runs ONE `decision_ttest` on the full pooled
  series. That is the verdict.
- Per-fold Sharpe/IC are returned as a `dispersion` diagnostic only —
  documented as a stability indicator, NEVER as a vote.
- `report_windows(signal_panel, fwd_panel, clean_mask=...)`: always
  emits BOTH the full-test and clean-subperiod metrics beside the
  pooled walk-forward verdict, so a single break or vol regime cannot
  silently dominate the headline.

**Old behavior:** there was no pooled walk-forward; per-fold metrics
could be combined however the consumer chose (often a majority vote
across folds).
**New behavior:** verdict = one t-test on the pooled OOS series; fold
dispersion is a side-channel diagnostic.

## FIX 4 — Repair and arm the future-invariance test

**File:** `src/stage2_agents/test_v2_causality.py` (TEST 1)
**Audit ref:** Task F (pandas 3.0.2: `df[col].iloc[k+1:] = ...` is a
no-op under CoW; the test was vacuous and would have "passed" for any
function, including ones that read the future)

- Replaced `df[col].iloc[cutoff_idx+1:] = scrambled` with
  `df.loc[post_idx, col] = scrambled`. The scramble now actually
  mutates `df_scrambled`.
- Added an explicit self-check: max-abs diff at `t <= cutoff_date` must
  be 0 AND max-abs diff at `t > cutoff_date` must be > 0 BEFORE
  trusting any invariance result downstream. If the scramble harness
  itself is broken, the test fails fast and explicitly.
- Added a positive control: `_leaky_features` is a deliberately
  future-leaking feature builder (it injects `df[target].shift(-1)`).
  TEST 1 now requires (a) the causal function to be identical up to
  the cutoff AND (b) the leaky control to be flagged as DIVERGENT.
  A smoke alarm that has never been triggered is not known to work.

**Old behavior:** TEST 1 "passed" any function because the scramble was
a no-op.
**New behavior:** TEST 1 passes only if the harness is verifiably
mutating post-cutoff rows AND the causal function is invariant AND the
leaky positive control is correctly flagged.

## FIX 5 — Gate remediation in `preprocessor_v2.validate_pair`

**File:** `src/stage1_collection_v2/preprocessor_v2.py`
**Audit ref:** Task E (bid_ask_bounce 96% rejected, crisis_year_patch
and thin_but_tradable both 100% rejected — Layer 2 was getting an
artificially clean / artificially shrunken universe)

- Legacy constants preserved as `LEGACY_MAX_ABS_AUTOCORR = 0.10` and
  `LEGACY_MAX_STALE_RUN = 10` so the diff is legible and the old
  behavior is reproducible by flag.
- **Asymmetric AC1 bound (a):** old `abs(ac1) > 0.10` → new
  `ac1 > MAX_AUTOCORR_POS (+0.10)` for the smoothing pathology, and
  `ac1 < AC1_NEG_FLOOR (-0.30)` for extreme bid-ask bounce. A benign
  -0.12 bid-ask bounce now passes.
- **Windowed exclusion (b):** new `find_stale_runs` + `excise_windows`.
  A 40-day stale patch inside 15 clean years is now excised (with
  WINDOW_EXCL_PAD=2 bars of padding) rather than torpedoing the whole
  pair. Status `PASS_WINDOWED` reports `n_excised` and `n_windows`.
  `build_returns_panel` applies the same excision when writing the
  returns parquet.
- **Stale-run ceiling raised (c):** `MAX_STALE_RUN = 15` (was 10) so an
  11-day legitimate holiday flat in a thin-but-tradable pair no longer
  trips the gate. The post-excision longest-run check still catches
  pairs that are stale enough to be non-localized.
- **Peg short-circuit preserved:** an entirely-flat series (std <
  `PEG_STD_THRESHOLD`) is now detected BEFORE excision so a true peg
  still surfaces as `is_pegged=True` rather than being excised down to
  zero rows.

**Old behavior on the six audit cases (reject_rate):**

| case                   | old  | new  |
|------------------------|-----:|-----:|
| clean_major            | 0.00 | 0.00 |
| bid_ask_bounce_major   | 0.96 | 0.00 |
| high_vol_EM            | 0.00 | 0.00 |
| crisis_year_patch      | 1.00 | 0.00 |
| thin_but_tradable      | 1.00 | 0.00 |
| de_facto_peg_no_note   | 0.00 | 0.00 |

Locked by `tests/test_gate_remediation.py` (which replays the same
case construction the audit used).

## FIX 6 — Regression test that locks the calibration

**File:** `tests/test_evaluator_calibration.py` (new)
**Audit ref:** Task A + Task B + the in-position-IC headline (Task A)

Imports `make_panel` directly from `data/layer2_audit/run_audit.py`
(does not re-implement the generator). Asserts, at reduced seeds:

- `threshold_rule_zero_detection_at_low_ic`: old rule must remain
  effectively dead at ic=0.05 — guard against re-introduction.
- `ttest_fpr_under_null` (100 seeds): FPR in [0.0, 0.10] at ic=0.
- `timeseries_detection_at_ic_002` (30 seeds): det >= 0.8 at ic=0.02,
  n=18, rho=0.6.
- `crosssectional_detection_at_ic_001` (100 seeds): det >= 0.8 at
  ic=0.01, n=18, rho=0.6, h=1.
- `in_position_ic_recovers_attenuation` (20 seeds): in-position IC /
  full-series IC ratio >= 1.5 (audit measured ~1.9 from 0.52x
  attenuation).
- `bh_fdr_under_global_null` (30 seeds, k=10 candidates): BH-FDR
  family-wise discovery <= 0.20 — multiple-testing guard.

---

## How to reproduce / run

```bash
# Calibration regression
python tests/test_evaluator_calibration.py

# Gate remediation (Task E case replay)
python tests/test_gate_remediation.py

# Stage-1 unit tests (now includes PASS_WINDOWED path)
python src/stage1_collection_v2/test_stage1_v2_synthetic.py

# Stage-2 causality (TEST 1 now armed; TEST 3 is pre-existing failure
# unrelated to this remediation — see signal_recovery in
# spectral_agent_v2)
python src/stage2_agents/test_v2_causality.py
```

---

## Scope boundaries

- Did NOT touch `data/processed_v2/`, `data/raw/`, `data/processed/`,
  any Stage 1 run scripts, or read-lock anything that the v2 download
  may have been writing to.
- Did NOT delete or edit the audit CSVs under `data/layer2_audit/`.
- Did NOT delete the old `evaluate_signal` in
  `spectral_agent_v2.py` — left in place so callers that still
  reference it keep working; new code should use
  `evaluation.evaluate_timeseries` / `evaluate_portfolio` /
  `evaluate_crosssectional`.
- Kept `LEGACY_MAX_ABS_AUTOCORR` and `LEGACY_MAX_STALE_RUN` as named
  constants in `preprocessor_v2.py` so the old behavior is recoverable
  by flag.

The branch is intentionally NOT merged to main — leave for review.

---

# Layer-3 Factor 1 — Cross-sectional FX momentum (branch: `layer3-momentum`)

(Merged into main as the foundation for `weekly-resampler`. Verdict
recorded here; full report in `data/layer3_momentum/MOMENTUM_REPORT.md`.)

- Extended `decision_ttest` with `sided` parameter ("two" default,
  "greater", "less"). Calibration tests pinned: one-sided FPR ≤ 2·alpha
  under the null.
- New `src/stage3_signals/momentum.py`: orientation map (XXXUSD: +1,
  USDXXX: -1), monthly signal/forward panel, tercile dollar-neutral
  weights with tier-keyed cost accounting.
- Verified orientation: corr(EURUSD raw, EUR oriented) = +1.0000;
  corr(USDJPY raw, JPY oriented) = -1.0000; corr(mean oriented, USD idx)
  = -0.9872.
- Monthly floor at n=15, rho=0.6, h=21 (sided="greater"): **rank-IC > 0.050**.
- Primary L=3m verdict: pooled OOS one-sided t = -2.13, p = 0.98,
  fires = False. Primary net rank-IC = **-0.0427**.
- Outcome: **(iii) NO-GO on momentum** — gross IC < floor in absolute
  value AND in the wrong direction; reversal magnitude ≈ floor but not
  confirmable from this sample.

---

# Weekly resampler — Layer-1.5 (branch: `weekly-resampler`)

Driver: monthly resolution gave only 191 observations and a floor of
0.050; the cross-sectional Spearman test needs ~4× more data to detect
the reversal-magnitude IC the monthly run found. Move to weekly.

## Part 1 — Shared resampler  (`src/stage1_collection_v2/weekly_panel.py`)

- Universe = intersection of `fx_daily_ohlc.parquet` pairs with
  `fx_log_returns.parquet` columns. AUTO-EXPANDS when A1/A2 land — no
  hardcoded pair list.
- Friday 17:00-NY close via pandas period `W-FRI` on the existing daily
  index. Non-overlapping weeks. Total: **835 Fri-weeks**.
- Per-pair active-day floor pinned to **`FLOOR[pair] = 0.5 × median(num_bars[:, pair])`**
  computed in-code from the real data. Realised values (recomputed every
  run, so they self-update):

  | pair | median | floor | pair | median | floor |
  |---|---:|---:|---|---:|---:|
  | USDHKD | 855 | 428 | NZDUSD | 1429 | 714 |
  | USDTRY | 1213 | 606 | USDCHF | 1430 | 715 |
  | USDHUF | 1320 | 660 | AUDUSD | 1431 | 716 |
  | USDZAR | 1324 | 662 | USDSEK | 1433 | 716 |
  | USDMXN | 1370 | 685 | USDJPY | 1434 | 717 |
  | USDSGD | 1371 | 686 | GBPUSD | 1435 | 718 |
  | USDCZK | 1408 | 704 | EURUSD | 1436 | 718 |
  | USDPLN | 1412 | 706 |  |  |  |
  | USDCAD | 1426 | 713 |  |  |  |
  | USDNOK | 1428 | 714 |  |  |  |

- A `(week, pair)` cell is **kept** iff it has ≥ 3 active days. Else
  `dropped_stale`. Pre-listing cells = `not_listed`, distinct from
  dropped_stale.
- **Single clean n=10 → n=15 step:** majors + USDHKD listed from
  2010-01-04, first kept week 2010-01-08. EM cohort {MXN, ZAR, PLN, HUF,
  CZK} listed 2010-11-15, first kept week 2010-11-19.
- **Audit finding the brief did not anticipate:** there is exactly ONE
  mid-sample multi-pair drop week — Friday 2017-02-24, where all five EM
  pairs have zero data (real outage). So EM contributing-weeks = **789**,
  not the brief's nominal 790. The `dropped_stale` status surfaces this
  cleanly; the test asserts exact 789 with a comment explaining the gap.
- Staleness diagnostic is written alongside the panel for HUMAN audit
  only — it never feeds a weight or signal.

## Part 2 — Weekly detection floor (`detection_floor_weekly.py`)

Re-measured at non-overlapping h=5, sided="greater", >=60 seeds:

| n | rho | n_weeks | min detectable rank-IC |
|---:|---:|---:|---:|
| 10 | 0.3 | 835 | 0.030 |
| 10 | 0.6 | 835 | 0.030 |
| 10 | 0.6 | 789 | 0.030 |
| 15 | 0.3 | 835 | 0.030 |
| 15 | 0.6 | 835 | **0.020** |
| 15 | 0.6 | 789 | **0.020** |

**WEEKLY GO/NO-GO floor: rank-IC > 0.020 at n=15** (and 0.030 for the
n=10 spans — applies to both the early 2010 listing gap AND the
2017-02-24 EM-outage week). Drop from the monthly 0.050 is roughly the
expected √(835/191) ≈ 2.1× factor.

## Part 3 — Weekly momentum re-verdict (`run_momentum_weekly.py`)

Pre-registered: primary **L = 8 weeks**; secondaries {2, 4, 13, 26}w;
1-week non-overlapping holding; terciles; dollar-neutral; tier costs
(1: 3 bps, 2: 20 bps round-trip).

Per-formation table (sample n_weeks ≈ 834):

| L | gross IR | net IR | rank IC | turnover | dollar β |
|---:|---:|---:|---:|---:|---:|
| 2  | -0.56 | -1.30 | -0.0331 | 1.89 | -0.14 |
| 4  | -0.62 | -1.17 | -0.0380 | 1.36 | -0.13 |
| **8** | **-0.46** | **-0.83** | **-0.0280** | **0.99** | **-0.15** |
| 13 | -0.56 | -0.87 | -0.0288 | 0.84 | -0.16 |
| 26 | -0.32 | -0.54 | -0.0127 | 0.61 | -0.19 |

Primary (L=8w):
- Pooled OOS one-sided: t = **-3.84**, p = 0.9999, fires = False.
- Full-sample net one-sided: t = -3.76, p = 0.9999.
- Full-sample rank-IC one-sided: t = -2.23, p = 0.9871.
- Fold dispersion (Sharpe): min = -1.62, median = -0.77, max = -0.67
  (every fold consistently negative; no fold rescues momentum).

`report_windows` views (NET IR, mean rank-IC):

| view | n_weeks | net IR | rank IC |
|---|---:|---:|---:|
| full | 834 | -0.83 | -0.0280 |
| clean (excl COVID + Ukraine) | 814 | -0.95 | -0.0279 |
| n=15 only span (excl early n=10 + EM outage) | 751 | -0.81 | -0.0280 |

All three views are essentially identical → the negative signal is NOT
a regime artifact and NOT a breadth artifact. Robust across cuts.

Secondary (BH-FDR across {2, 4, 13, 26}w): every one-sided p ≥ 0.99,
all BH-fires = False.

**WEEKLY OUTCOME: (iii) NO-GO on momentum.** Gross rank-IC = -0.028 vs
floor +0.020. NB |IC| > floor, consistent with cross-sectional REVERSAL;
the registered hypothesis was momentum so the verdict is NO-GO. The
reversal direction is now visible at weekly resolution where it was
invisible at monthly:

| frequency | floor | primary rank-IC | pooled t (greater) |
|---|---:|---:|---:|
| monthly (L=3m, n=15) | +0.050 | -0.043 | -2.13 |
| weekly  (L=8w, n=15) | +0.020 | -0.028 | **-3.84** |

At weekly resolution we have a sharply negative, regime-robust, fold-
robust reading on the registered momentum direction. The judge worked
twice: registered hypothesis evaluated, rejected, no positive result
manufactured at either frequency. The weekly view tightens the evidence
without changing the verdict.

The branch is NOT merged.

---

# Layer-3 Factor 2 — Cross-sectional FX carry (branch: `layer3-carry`)

Carry is the registered cross-currency test the judge hasn't seen, AND the
method-control for any later "reversal" hypothesis arising from the
weekly-momentum negative-IC reading.

## Pre-registration (fixed before any IC was computed)

- **Primary signal:** lagged short-rate differential
  (`carry_features.parquet → carry_signal_pct`, keyed by the non-USD
  currency, already lagged 1 trading day in source).
- **Secondary signal (pre-registered, BH-reported, robustness):**
  CIP-forward-implied carry. Sign-corrected via `-orientation(pair) × cip
  / spot` so positive = high yielder.
- **Sign:** registered POSITIVE; `sided='greater'`.
- **Holding:** 1 week non-overlapping; terciles; dollar-neutral L/S;
  tier costs Tier1=3bps, Tier2=20bps round-trip.
- **Decision:** `decision_ttest(sided='greater')` on the NET L/S series
  via `walk_forward_pooled` (5 expanding folds, pool OOS).

## Part 1 — Carry panel (`src/stage3_signals/carry.py`)

Built on top of `build_weekly_panel` so universe / week index / drop rule
are shared with momentum.

Top/bottom-tercile placement (orientation verification, asserted in
`tests/test_carry_orientation.py`):

| pair | top tercile frac | bot tercile frac |
|---|---:|---:|
| USDMXN | 1.000 | 0.000 |
| USDZAR | 1.000 | 0.000 |
| USDCHF | 0.000 | 1.000 |
| USDJPY | 0.000 | 0.966 |
| EURUSD | 0.034 | 0.865 |
| AUDUSD | 0.582 | 0.219 |

Within-pair Spearman(primary, secondary) across weeks ≈ **+1.00** — the
two signals agree on cross-sectional rank order, confirming the CIP sign
correction.

## Part 2 — Detection floor (consistency check)

Same weekly grid, same universe, same `h=5` non-overlap as weekly
momentum. Reuses `data/layer3_momentum/detection_floor_weekly_summary.csv`.

**Floor at n=15, rho=0.6: rank-IC > 0.020** — identical to the momentum
floor, as expected.

## Part 3 — Verdict

| signal | n_weeks | gross IR | net IR | mean rank IC | hit rate | turnover | dollar β | dollar R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary (rate-diff) | 834 | -0.22 | -0.24 | **+0.0182** | 0.49 | 0.05 | +0.378 | 0.146 |
| secondary (CIP-fwd) | 834 | -0.21 | -0.24 | +0.0182 | 0.49 | 0.05 | +0.379 | 0.149 |

Per-week turnover is **0.05** (vs 0.99 for momentum L=8w) — the rate
differential changes glacially, so net IR is essentially gross IR. Costs
are not the binding constraint.

Primary (L=registered-rate-diff, 1-week hold):
- Pooled OOS one-sided: t = **-0.71**, p = 0.76, fires = **False**.
- Full-sample net one-sided: t = -1.05, p = 0.85.
- Full-sample rank-IC one-sided: t = **+1.38**, p = 0.083 — direction
  correct, just below the 5% one-sided threshold and right at the floor.
- Fold dispersion (Sharpe): min -0.64, median -0.07, max +0.53 — wide,
  no fold clearly positive.

`report_windows` (all near-identical):

| view | n_weeks | net IR | mean rank IC |
|---|---:|---:|---:|
| full | 834 | -0.24 | +0.0182 |
| clean (excl COVID + Ukraine) | 814 | -0.12 | +0.0212 |
| n=15 only (excl early n=10 + EM outage) | 751 | -0.22 | +0.0182 |

Secondary CIP, BH-corrected: both signals one-sided p ≈ 0.85, BH = False.
Basis (mean within-pair corr primary↔secondary) = +1.00.

**Note on dollar-neutrality:** carry's net L/S series has a meaningful
dollar β of **+0.38 (R² ≈ 0.15)** — substantially larger than weekly
momentum's β (−0.15). This is a known carry-portfolio property: a long-EM
/ short-low-yielder book is implicitly long-USD-funded-EM, which loads on
the dollar factor. Reported but not corrected: pre-registered weights are
the verdict-bearing object.

**WEEKLY CARRY OUTCOME: (iii) NO-GO.** Gross rank-IC = +0.0182 vs floor
+0.020. Direction is correct (carry premium positive) but the magnitude
falls just under the breadth-and-resolution floor.

## Part 4 — Carry-vs-reversal overlap (DIAGNOSTIC ONLY)

> **GUARD:** This section renders no verdict on the reversal hypothesis.
> The (negated) weekly momentum rank-IC is used only to measure its
> week-by-week relationship with carry. Reversal-as-hypothesis requires
> a separate-universe or forward holdout.

Common weeks where both ICs exist: **672**.

- Pearson(carry rank-IC, **negated** momentum rank-IC) = **+0.123**
- Spearman = **+0.117**

Regression of weekly CARRY NET L/S returns on weekly MOMENTUM NET L/S
returns:
- β = **−0.305**   R² = 0.077
- Weekly intercept (carry alpha vs momentum) = −0.000680
- Residual one-sided alpha-p ≈ 0.50

**Reading: World C (resolution-bound).** Carry's gross IC is just under
the floor at this breadth/resolution, so the overlap diagnostic above is
not load-bearing. The binding constraint is breadth — the same constraint
that NO-GO'd momentum. Reversal-as-hypothesis requires a
separate-universe or forward holdout regardless of what this diagnostic
shows.

## Standing comparison of the two registered factors

| factor | direction | floor | gross rank-IC | pooled OOS t | net IR | turnover | outcome |
|---|---|---:|---:|---:|---:|---:|---|
| momentum L=8w (registered +) | wrong sign | 0.020 | -0.028 | **-3.84** | -0.83 | 0.99 | NO-GO (reversal-shaped) |
| carry primary (registered +) | right sign | 0.020 | +0.018 | -0.71 | -0.24 | 0.05 | NO-GO (just under floor) |

Carry is direction-correct but breadth-constrained. Momentum is the
opposite sign of its registration. The two are not "the same coin" in
the simple sense — IC correlation only +0.12, returns regression β = −0.3
with R² = 0.08. They overlap modestly; neither passes at this breadth.

The branch is NOT merged.
