# HEADLINE NUMBERS

- **MDI (timeseries, threshold rule)** at n_inst=18, rho=0.6: smallest ic_true with detection_rate >= 0.8 = not reached at ic<=0.10
- **MDI (timeseries, portfolio t-test)** at n_inst=18, rho=0.6: smallest ic_true with detection_rate >= 0.8 = 0.02
- **threshold-vs-ttest gap @ ic=0.03,n=18,rho=0.6**: det_threshold=0.00, det_ttest=1.00, excess false-negative rate of pass rule ≈ 1.00
- **IC-over-zeros attenuation ratio (70% gating)**: mean = 0.520 (IC computed over zero-padded signal is biased toward 0)
- **cross-sectional primitive (NET-NEW; not in pipeline)** MDI at n_inst=18,rho=0.6,h=1: ic_true with det>=0.8 = 0.01
- **breadth-collapse**: at ic_true=0.025,n=18, portfolio Sharpe rho=0 → 1.30, rho=0.6 → 1.36 (ratio 1.05x if sh0>0)
- **window sensitivity**: fixed_60_40_full_test: sharpe=0.59, dec=FAIL_THRESH+TTEST_FAIL | clean_subperiod: sharpe=1.99, dec=FAIL_THRESH+TTEST_PASS | walk_forward_5fold_summary: sharpe=0.84, dec=mode=FAIL_THRESH+TTEST_FAIL, sharpe_range=[-1.84,2.43]
- **gate worst false-rejection cases**: crisis_year_patch: 1.00 (stale_run); thin_but_tradable: 1.00 (stale_run); bid_ask_bounce_major: 0.96 (autocorr_too_high)
- **v2 future-invariance test integrity**: VACUOUS — scramble is a no-op under installed pandas

# LAYER 2 POWER AUDIT — FINDINGS

- master seed: 20260620
- default seeds per cell: 30
- pass rule (existing Layer-2): sharpe>0.3 AND accuracy>0.55
- alpha for t-test reference decisions: 0.05
- statsmodels available: True

## Generator validation

- axis=timeseries: realized mean per-instrument IC = 0.0499 (target 0.05);  realized mean off-diag corr = 0.3975 (target 0.40)
- axis=crosssectional: realized mean per-instrument IC = 0.0499 (target 0.05);  realized mean off-diag corr = 0.3975 (target 0.40)

## Task A — timeseries detection surface (real evaluators)

- resuming from existing CSVs: detection_timeseries.csv rows=54, detection_timeseries_gated.csv rows=10

### Headline slice at FX-realistic conditions (n_inst=18, rho=0.6)

| ic_true | mean_ic_evalsig | mean_ic_evalpred | mean_sharpe | det_threshold | det_ttest |
|---:|---:|---:|---:|---:|---:|
| 0.00 | -0.0003 | -0.0009 | -0.0059 | 0.00 | 0.00 |
| 0.01 | 0.0064 | 0.0081 | 0.1055 | 0.00 | 0.57 |
| 0.02 | 0.0165 | 0.0195 | 0.2684 | 0.00 | 0.97 |
| 0.03 | 0.0243 | 0.0297 | 0.3933 | 0.00 | 1.00 |
| 0.05 | 0.0374 | 0.0462 | 0.6074 | 0.00 | 1.00 |
| 0.10 | 0.0793 | 0.0964 | 1.2937 | 0.00 | 1.00 |

### IC-over-zeros (gating attenuation)

| ic_true | rho | mean_ic_ungated | mean_ic_gated | attenuation_ratio |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0 | -0.0010 | -0.0006 | 0.558 |
| 0.00 | 0.6 | -0.0001 | -0.0003 | 2.691 |
| 0.02 | 0.0 | 0.0163 | 0.0079 | 0.487 |
| 0.02 | 0.6 | 0.0164 | 0.0083 | 0.504 |
| 0.03 | 0.0 | 0.0228 | 0.0110 | 0.481 |
| 0.03 | 0.6 | 0.0237 | 0.0129 | 0.546 |
| 0.05 | 0.0 | 0.0408 | 0.0217 | 0.531 |
| 0.05 | 0.6 | 0.0412 | 0.0219 | 0.531 |
| 0.10 | 0.0 | 0.0768 | 0.0408 | 0.530 |
| 0.10 | 0.6 | 0.0776 | 0.0426 | 0.549 |

## Task B — cross-sectional detection surface (net-new primitive)

- This primitive is NOT present in the current pipeline (the existing
  Layer-2 only does per-instrument timeseries metrics).

- wrote: detection_crosssectional.csv  rows=90

### Headline slice at FX-realistic conditions (n_inst=18, rho=0.6, h=1)
| ic_true | mean_rank_ic | mean_IR | det_rate |
|---:|---:|---:|---:|
| 0.00 | 0.0019 | 0.099 | 0.07 |
| 0.01 | 0.0140 | 0.886 | 0.93 |
| 0.02 | 0.0292 | 1.799 | 1.00 |
| 0.03 | 0.0436 | 2.621 | 1.00 |
| 0.05 | 0.0712 | 4.369 | 1.00 |

## Task C — breadth-collapse isolation

- wrote: breadth_collapse.csv  rows=10

### Curve

| axis | rho_cross | N_eff | mean_port_sharpe | detection_rate |
|---|---:|---:|---:|---:|
| timeseries | 0.00 | 18.00 | 1.297 | 0.00 |
| timeseries | 0.20 | 4.09 | 1.404 | 0.00 |
| timeseries | 0.40 | 2.31 | 1.266 | 0.00 |
| timeseries | 0.60 | 1.61 | 1.358 | 0.00 |
| timeseries | 0.80 | 1.23 | 1.432 | 0.00 |
| crosssectional | 0.00 | 18.00 | 1.347 | 0.00 |
| crosssectional | 0.20 | 4.09 | 1.292 | 0.00 |
| crosssectional | 0.40 | 2.31 | 1.388 | 0.00 |
| crosssectional | 0.60 | 1.61 | 1.480 | 0.00 |
| crosssectional | 0.80 | 1.23 | 1.324 | 0.00 |

**Interpretation note**: in this construction the *signal* is drawn iid across
instruments, so per-instrument strategy returns are uncorrelated even when the
*returns* are highly correlated, and portfolio Sharpe stays near 1.3 as N_eff
collapses. This isolates one thing — *return collinearity alone does not null a
signal that varies independently across instruments*. The realistic
dollar-factor failure mode is a *shared* signal (one USD-direction call for all
crosses); under that construction the breadth collapse is real and the curve
would slope sharply downward. The audit shows the existing threshold rule
(det_rate=0 everywhere) is the binding constraint regardless of which breadth
story applies.

## Task D — window-regime sensitivity (same signal, three verdicts)

- wrote: window_sensitivity.csv  rows=8

### Three verdicts on the same injected signal

| regime | measured_sharpe | measured_ic | decision |
|---|---:|---:|---|
| fixed_60_40_full_test | 0.592 | 0.0179 | FAIL_THRESH+TTEST_FAIL |
| clean_subperiod | 1.994 | 0.0299 | FAIL_THRESH+TTEST_PASS |
| walk_forward_5fold_summary | 0.837 | 0.0198 | mode=FAIL_THRESH+TTEST_FAIL, sharpe_range=[-1.84,2.43] |
| wf_fold0 | 2.428 | 0.0308 | FAIL_THRESH+TTEST_PASS |
| wf_fold1 | 2.073 | 0.0228 | FAIL_THRESH+TTEST_PASS |
| wf_fold2 | 0.712 | 0.0161 | FAIL_THRESH+TTEST_FAIL |
| wf_fold3 | -1.844 | 0.0029 | FAIL_THRESH+TTEST_FAIL |
| wf_fold4 | 0.813 | 0.0264 | FAIL_THRESH+TTEST_FAIL |

## Task E — validation-gate false-rejection rate

- wrote: gate_false_rejection.csv  rows=6

| case | tier | n_seeds | reject_rate | dominant_reason | breakdown |
|---|---:|---:|---:|---|---|
| clean_major | 1 | 50 | 0.00 |  |  |
| bid_ask_bounce_major | 1 | 50 | 0.96 | autocorr_too_high | autocorr_too_high=48 |
| high_vol_EM | 2 | 50 | 0.00 |  |  |
| crisis_year_patch | 1 | 50 | 1.00 | stale_run | stale_run=50 |
| thin_but_tradable | 1 | 50 | 1.00 | stale_run | stale_run=50 |
| de_facto_peg_no_note | 1 | 50 | 0.00 |  |  |

## Task F — future-invariance test integrity (vacuity check)

- pandas.__version__ = 3.0.2
- copy-on-write: option flag value = True (pandas 3.0.2)

- v2 scramble idiom (`df[col].iloc[k+1:] = ...`): post-cutoff equals original = True
  - max abs change in after-cutoff region: 0.000000
- baseline scramble idiom (`df.loc[idx, c] = ...`): post-cutoff equals original = False
  - max abs change in after-cutoff region: 4.371829
- warnings emitted during v2 idiom:
  - ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained assignment.
Such chained assignment never works 
  - ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained assignment.
Such chained assignment never works 
  - ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained assignment.
Such chained assignment never works 

**Verdict: v2 future-invariance test is VACUOUS under installed pandas.**
If the scramble does not actually mutate post-cutoff rows, the v2
TEST 1 cannot fail and provides no causality guarantee.


## CSV artifacts

- data/layer2_audit/detection_timeseries.csv
- data/layer2_audit/detection_timeseries_gated.csv
- data/layer2_audit/detection_crosssectional.csv
- data/layer2_audit/breadth_collapse.csv
- data/layer2_audit/window_sensitivity.csv
- data/layer2_audit/gate_false_rejection.csv
