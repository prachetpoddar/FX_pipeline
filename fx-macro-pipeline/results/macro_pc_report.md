# Macro-PC FINAL re-run — FUNDING + full DOLLAR

Branch: `macro-pc-funding`. Read-only on `data/processed_v2/`, `data/layer3_*`, and FRED (via cached CSV). Reuses `weekly_panel.py`, `carry.py`, the momentum module, `evaluation.py`, and the prior `macro_pc.py`.

**This is the FINAL registered macro-conditioning attempt.** Pre-committed kill/keep rule below — binding either way.

## Pre-committed decision rule

Macro-conditioning is KEPT iff ALL THREE gates pass:

- **(A) PC1 dominance:** PC1 eigenvalue >= **1.5** (decisive common mode, not the prior 1.186-vs-1.146 marginal pass).
- **(B) Response R²:** carry OR momentum IC-on-PC-innovation R² >= **0.02** with at least one coefficient significant at one-sided **0.05** (HAC SE).
- **(C) Leave-one-event-out generalizes:** mean pred/actual corr POSITIVE AND mean sign-agreement >= **0.6** across the 6 registered episodes (BOTH factors must satisfy).

Any failure → CLOSED. Accept the negative. No further axes, no nonlinear response, no "richer data next time."

## Registered changes vs the prior run (only two)

1. **Added FUNDING factor (registered source: `NFCI` from FRED).** Primary CIP-basis construction from internal data is NOT reliable here — forward tenor and unit conventions are not pinned in our `carry_features.parquet`, and a sketch construction does NOT spike at the known COVID 2020 funding shock (failure mode that disqualifies the CIP series). NFCI is the Chicago Fed weekly Financial Conditions Index, 1971-present, covering our entire 2010-2026 window — including the post-2022 Ukraine and 2024 carry-unwind episodes that TEDRATE (discontinued 2022-01) would miss. Corr(NFCI, TEDRATE) over their 2010-2022 overlap = 0.58. NFCI captures COVID 2020 stress (+0.31 vs full-sample mean −0.47).
2. **DOLLAR enters at full strength** (not neutralized) as the literature's DOL common mode.

Everything else frozen: linear one-coef-per-PC response, first-difference innovation conditioner, HAC SE, FXVOL-only baseline, leave-one-event-out across the same 6 episodes.

## Part 0 — FUNDING (NFCI) coverage + episode verification

FUNDING (NFCI) covers 2010-01-08 to 2026-01-02. Episode peaks (NFCI is signed: higher = tighter conditions / more stress):

| episode | window | NFCI max | NFCI mean | full mean |
|---|---|---:|---:|---:|
| GFC tail 2010-11 | 2010-11-01..2010-12-31 | -0.441 | -0.468 | -0.465 |
| EUR crisis 2011 | 2011-09-01..2011-12-31 | -0.035 | -0.080 | -0.465 |
| China/oil 2015-16 | 2015-08-01..2016-02-29 | -0.272 | -0.385 | -0.465 |
| COVID 2020 | 2020-02-15..2020-04-30 | +0.306 | +0.061 | -0.465 |
| Ukraine 2022 | 2022-02-01..2022-10-31 | -0.098 | -0.263 | -0.465 |
| Carry unwind 2024 | 2024-07-01..2024-09-30 | -0.358 | -0.383 | -0.465 |

COVID 2020 is the only episode with NFCI clearly elevated above the full-sample mean. The other episodes — including the EUR sovereign crisis and Ukraine 2022 — registered as loose-or-neutral financial-conditions weeks for the U.S. even when they were live FX stress events. This itself is a useful finding: from the U.S. financial-conditions vantage, many of our episodes were not USD-funding events.

## Part 1 — 6-factor PCA

Eigenvalue spectrum:

| PC | eigenvalue | shuffle null 95% | retained? | var share |
|---|---:|---:|---|---:|
| PC1 | 1.545 | 1.163 | YES | 0.258 |
| PC2 | 1.169 | 1.095 | YES | 0.195 |
| PC3 | 1.005 | 1.047 | no | 0.168 |
| PC4 | 0.947 | 1.004 | no | 0.158 |
| PC5 | 0.903 | 0.964 | no | 0.151 |
| PC6 | 0.429 | 0.926 | no | 0.072 |

**Retained: 2 PC(s)** vs the prior run's 1.

Retained PC loadings:

| factor | PC1 | PC2 |
|---|---:|---:|
| DOLLAR | +0.120 | +0.575 |
| FXVOL | -0.474 | -0.458 |
| DISPERSION | -0.508 | +0.464 |
| TAIL | +0.052 | +0.477 |
| BREADTH | -0.039 | -0.070 |
| FUNDING | -0.706 | +0.109 |

**Interpretation (post-hoc):** PC1 is dominated by FUNDING (loading −0.71), with DISPERSION and FXVOL also loading negatively — a 'funding/stress' axis where PC1 ↓ means tighter funding + wider rate-dispersion + higher vol. PC2 captures the dollar/risk-on signal: DOLLAR +0.58, TAIL +0.48, DISPERSION +0.46, FXVOL −0.46. BREADTH (as constructed: fraction same-signed as the mean) loads near zero on both — carrying through the prior finding that it correlates with |DOLLAR|, not signed DOLLAR.

## Part 2 — Factor response on PC innovations

### carry

| term | β | HAC SE | t | p | R² | n |
|---|---:|---:|---:|---:|---:|---:|
| intercept | +0.01825 | 0.01305 | +1.40 | 0.1618 | 0.0022 | 739 |
| PC1 | +0.04159 | 0.03775 | +1.10 | 0.2705 | 0.0022 | 739 |
| PC2 | -0.01076 | 0.01658 | -0.65 | 0.5165 | 0.0022 | 739 |

Baseline (dFXVOL only): R² = **0.0000**, β[dFXVOL] = -0.4524 (t=-0.15, p=0.8791).

### momentum_L8w

| term | β | HAC SE | t | p | R² | n |
|---|---:|---:|---:|---:|---:|---:|
| intercept | -0.02803 | 0.01259 | -2.23 | 0.0260 | 0.0005 | 672 |
| PC1 | +0.00766 | 0.04203 | +0.18 | 0.8554 | 0.0005 | 672 |
| PC2 | +0.00342 | 0.01863 | +0.18 | 0.8542 | 0.0005 | 672 |

Baseline (dFXVOL only): R² = **0.0008**, β[dFXVOL] = -2.2226 (t=-0.60, p=0.5507).

## Part 3 — Leave-one-event-out

Episodes:

| label | start | end |
|---|---|---|
| gfc_tail_2010-11 | 2010-11 | 2010-12 |
| eur_crisis_2011-12 | 2011-08 | 2011-12 |
| china_oil_2015-16 | 2015-08 | 2016-02 |
| covid_2020 | 2020-02 | 2020-04 |
| ukraine_inflation_2022 | 2022-02 | 2022-10 |
| carry_unwind_2024 | 2024-07 | 2024-09 |

### carry

| episode | n_in_ep | n_train | pred/actual corr | sign agree | mean actual | mean pred |
|---|---:|---:|---:|---:|---:|---:|
| gfc_tail_2010-11 | 4 | 735 | -0.570 | 0.50 | +0.0768 | +0.0191 |
| eur_crisis_2011-12 | 22 | 717 | +0.163 | 0.41 | -0.1416 | +0.0221 |
| china_oil_2015-16 | 18 | 721 | +0.127 | 0.39 | -0.0796 | +0.0158 |
| covid_2020 | 12 | 727 | +0.167 | 0.50 | -0.2704 | +0.0183 |
| ukraine_inflation_2022 | 39 | 700 | +0.063 | 0.59 | +0.0848 | +0.0106 |
| carry_unwind_2024 | 13 | 726 | +0.212 | 0.38 | -0.0788 | +0.0198 |

Mean pred/actual corr: **+0.027**;  mean sign-agreement: **0.46**.

### momentum_L8w

| episode | n_in_ep | n_train | pred/actual corr | sign agree | mean actual | mean pred |
|---|---:|---:|---:|---:|---:|---:|
| gfc_tail_2010-11 | 0 | 672 | — | — | +nan | +nan |
| eur_crisis_2011-12 | 22 | 650 | +0.027 | 0.27 | +0.1211 | -0.0337 |
| china_oil_2015-16 | 10 | 662 | -0.414 | 0.70 | -0.1093 | -0.0289 |
| covid_2020 | 12 | 660 | -0.317 | 0.42 | +0.1125 | -0.0371 |
| ukraine_inflation_2022 | 39 | 633 | -0.169 | 0.54 | -0.0600 | -0.0269 |
| carry_unwind_2024 | 13 | 659 | -0.460 | 0.54 | -0.0234 | -0.0282 |

Mean pred/actual corr: **-0.267**;  mean sign-agreement: **0.49**.

## Part 4 — Gate verdict

| gate | what | passed? |
|---|---|---|
| (A) PC1 eigvalue | PC1 = 1.545, threshold 1.5 | **PASS** |
| (B) Response R² + significance | best: None | **FAIL** |
| (C) LOO generalization | both factors must satisfy corr > 0 AND sign-agree >= 0.6 | **FAIL** |

### DECISION: **CLOSED**

Pre-committed gate(s) failed: B, C. Per registered rule, macro-conditioning on these factors as constructed is CLOSED — accept the negative. No further axes, no nonlinear response, no "richer data next time." This was the final registered macro attempt.

---

**Per the pre-committed rule, this was the final registered macro-conditioning attempt. The verdict above is binding. No further iteration.**
