# Layer-3 BREADTH — curated crosses, three registered re-verdicts

Branch: `layer3-breadth`. Read-only on `data/processed_v2/` and `data/layer3_*`. Reuses `weekly_panel.py`, `carry.py`, `evaluation.py`, `run_momentum_weekly.py`. No reimplemented metrics; the verdict machinery is the same calibrated judge used since Layer 2 remediation.

**Pre-registration (frozen before any IC was computed; no tuning):**
- Universe = 15 USD pairs + crosses by RULE (both legs in G10 ex-USD AND both legs in the live USD-pair universe). Enumeration: 36 crosses, n_expanded = 51.
- Cross orientation/codes: BASE = higher-ranked currency in the frozen list EUR > GBP > AUD > NZD > CAD > CHF > NOK > SEK > JPY. Oriented cross return = BASE_oriented − QUOTE_oriented (log-additive identity from the leg USD pairs; verified at run time).
- Cross cost (registered, conservative): round-trip = sum of BOTH legs' USD-pair tier spreads (synthesis cost, no direct-cross saving).
- Cross drop rule: a cross is KEPT in a week iff BOTH legs are KEPT that week.
- Carry signal: registered POSITIVE; raw tercile L/S (NOT dollar-neutral — OOS dollar-neutral lost on `layer3-carry`; raw is the winning construction).
- Momentum: L = 8 weeks, registered POSITIVE.
- Reversal: L = 8 weeks, registered NEGATIVE, CROSSES-ONLY subset (instruments the reversal reading never saw).
- Holding: 1 week non-overlapping. Decision: `decision_ttest(sided=...)` on NET L/S via `walk_forward_pooled` (5 expanding folds, pool OOS — never vote folds).

## Part 1 — Expanded universe

- n_usd = 15;  n_cross = 36;  n_expanded = **51**.
- Crosses: EURGBP, EURAUD, EURNZD, EURCAD, EURCHF, EURNOK, EURSEK, EURJPY, GBPAUD, GBPNZD, GBPCAD, GBPCHF, GBPNOK, GBPSEK, GBPJPY, AUDNZD, AUDCAD, AUDCHF, AUDNOK, AUDSEK, AUDJPY, NZDCAD, NZDCHF, NZDNOK, NZDSEK, NZDJPY, CADCHF, CADNOK, CADSEK, CADJPY, CHFNOK, CHFSEK, CHFJPY, NOKSEK, NOKJPY, SEKJPY.
- Per-week breadth (expanded): min = 45,  median = 51,  max = 51.

## Part 2 — Detection floor at expanded breadth

Same harness as the weekly floor (non-overlapping h=5, sided='greater', ≥60 seeds). Measured rho on the live expanded panel:

- USD-only mean off-diag corr = +0.483
- expanded mean off-diag corr = +0.062  (the curated crosses are largely orthogonal to USD pairs — this is what 'independent breadth' looks like in the data)

Floors (snapped to closest grid rho):

- USD-only (n=15, rho≈0.6, 789w): rank-IC > **0.020** (consistent with prior weekly runs)
- EXPANDED (n=51, rho≈0.05, 789w): rank-IC > **0.015**

See `detection_floor_breadth.csv` and `detection_floor_breadth_summary.csv`.

## Part 3 — Verdicts

### Carry (headline) — registered POSITIVE, full expanded universe

| metric | value |
|---|---:|
| n_weeks | 834 |
| gross IR | -0.30 |
| net IR | -0.31 |
| mean rank IC | **+0.0113** |
| turnover | 0.03 |
| dollar β | +1.508 |
| dollar R² | 0.242 |

- Pooled OOS one-sided greater: t = -1.57,  p = 0.9419,  **fires = False**
- Full rank-IC one-sided greater: t = +0.84,  p = 0.2002
- Fold dispersion (Sharpe): min = -0.65,  median = -0.44,  max = -0.06

`report_windows` (full / clean / n=15-only):

| view | n_weeks | net IR | mean rank IC |
|---|---:|---:|---:|
| full | 834 | -0.31 | +0.0113 |
| clean | 814 | -0.27 | +0.0130 |
| n15_only | 751 | -0.30 | +0.0113 |

**OUTCOME: (iii) NO-GO — gross IC (+0.0113) magnitude < floor (0.015)**

### Momentum (registered POSITIVE) — full expanded universe, L=8w

| metric | value |
|---|---:|
| n_weeks | 834 |
| gross IR | -0.37 |
| net IR | -0.55 |
| mean rank IC | **-0.0328** |
| turnover | 0.93 |
| dollar β | -0.610 |

- Pooled OOS one-sided greater: t = -2.56,  p = 0.9948,  **fires = False**
- Full rank-IC one-sided greater: t = -2.46,  p = 0.9931

`report_windows`:

| view | n_weeks | net IR | mean rank IC |
|---|---:|---:|---:|
| full | 834 | -0.55 | -0.0328 |
| clean | 814 | -0.62 | -0.0323 |
| n15_only | 751 | -0.52 | -0.0328 |

**OUTCOME: (iii) NO-GO — gross IC (-0.0328) wrong direction**

### Reversal (registered NEGATIVE) — CROSSES-ONLY subset, L=8w

Semi-OOS: the 36 crosses are instruments the prior reversal reading never saw. Reported separately; **NOT** the same test as the momentum-positive verdict and must not be conflated.

| metric | value |
|---|---:|
| n_weeks | 834 |
| gross IR | -0.44 |
| net IR | -0.59 |
| mean rank IC | **-0.0322** |
| turnover | 0.95 |
| dollar β | -0.268 |

- Pooled OOS one-sided LESS: t = -2.62,  p = 0.0043,  **fires = True**
- Full rank-IC one-sided LESS: t = -2.43,  p = 0.0075

`report_windows`:

| view | n_weeks | net IR | mean rank IC |
|---|---:|---:|---:|
| full | 834 | -0.59 | -0.0322 |
| clean | 814 | -0.67 | -0.0323 |
| n15_only | 751 | -0.51 | -0.0323 |

**OUTCOME: (i) GO — net IC beyond floor (in registered direction) AND pooled t-test fires**

### Headline comparison: n=15 prior vs n_expanded now

| factor | n=15 rank-IC | n=15 floor | n=15 outcome | n_expanded rank-IC | n_expanded floor | n_expanded outcome |
|---|---:|---:|---|---:|---:|---|
| carry (raw, sided='greater') | +0.0182 | 0.020 | NO-GO | +0.0113 | 0.015 | (iii) NO-GO |
| momentum (sided='greater') | −0.028 | 0.020 | NO-GO | -0.0328 | 0.015 | (iii) NO-GO |
| reversal (sided='less', crosses-only) | n/a | n/a | (prior was descriptive only) | -0.0322 | 0.015 | (i) GO |

## Part 4 — APPENDIX A: corrected momentum spectrum (NO VERDICT)

> **GUARD:** Descriptive. No hypothesis registered, no verdict. The prior appendix measured the integrated rolling-sum signal's PSD, which is near-random-walk and 1/f²-dominated. This appendix measures (a) the DIFFERENCED L=8w signal and (b) the RAW weekly oriented returns. The shuffle null is phase-randomization (Fourier shuffle), preserving the PSD envelope so a peak counts only if it exceeds what the envelope would produce.

Per-series pooled summary (median across all instruments):

| series | median dom. period (wks) | median peak / null95 | median ACF first-zero lag |
|---|---:|---:|---:|
| diff_signal (L=8w differenced) | 5.3 | 1.14 | 1.0 |
| raw_returns (weekly oriented) | 5.4 | 1.17 | 1.0 |

Interpretation guidance (informational — no verdict):
- peak/null95 ≫ 1 (e.g. > 2) with a consistent dominant period across instruments → a frequency-band lane is worth registering post-breadth, with that band pre-specified.
- peak/null95 ≈ 1 → the apparent peak is within phase-shuffle null; broadband; drop the lane.

See `momentum_spectrum_corrected.csv` for per-pair values.

## Part 5 — APPENDIX B: PCA dimensionality (NO VERDICT)

> **GUARD:** Descriptive. Validates whether the curated crosses bought INDEPENDENT breadth. No PCA-derived signal is built — PCA-as-signal is a separate registered future step, not this run.

| universe | n | T | PC1 var share | PC1 same-sign loadings | N_eff (participation) | PCs above MP null | MP λ⁺ |
|---|---:|---:|---:|---:|---:|---:|---:|
| USD_only | 15 | 752 | 0.548 | True | 3.08 | 2 | 1.30 |
| expanded | 51 | 752 | 0.263 | False | 6.91 | 9 | 1.59 |

Headline reading (descriptive):
- N_eff went from **3.08** (USD-only) → **6.91** (expanded). Ratio = **2.24×**.
- PC1 variance share moved 0.548 → 0.263. PC1 same-sign loadings: USD True, expanded False. (Same-sign loadings on the 15-USD panel is the expected dollar mode; on the expanded panel the dollar mode is no longer the dominant single-direction shock.)
- Eigenvalues above the Marchenko-Pastur noise floor: USD = 2, expanded = 9. This is the measured count of 'real' PCs (above iid-null) — anything beyond is estimation noise on a weekly sample.

See `pca_dimensionality.csv`.
