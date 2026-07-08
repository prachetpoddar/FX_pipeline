# HEADLINE NUMBERS

- **weekend rows in fx_raw.parquet**: 1700
- **pegged columns (>95% zero diffs)**: 10
- **EUR ac1**: all=0.2438 (handoff claim ~+0.44)   weekday-only=0.3420
- **CHF ac1**: all=-0.4744 (handoff claim ~+0.49)   weekday-only=0.0184
- **EUR std**: 0.0033 (handoff claim ~0.0033, real-tradable ~0.0060)
- **majors std vs real-tradable benchmark**: EUR: 0.0033 vs 0.0060; GBP: 0.0035 vs 0.0060; JPY: 0.0054 vs 0.0060; CHF: 0.0209 vs 0.0065; AUD: 0.0041 vs 0.0070; CAD: 0.0061 vs 0.0050; SEK: 0.1006 vs 0.0075; NOK: 0.0048 vs 0.0080
- **Task 3 leak**: mean sharpe_leaky=2.4770 vs mean sharpe_lagged=0.6784 (handoff: leaky ~1.89, lagged ~0; gap = the leak)
- **Task 4 counts**: count(acc<0.50 AND sharpe>0)=90;  of those, MAR_correct>MAR_wrong=78
- **EUR worked example**: variant=ma_5  test_acc=0.5938  test_sharpe=2.7356  MAR_correct=0.002100  MAR_wrong=0.001777  sharpe(k=-1,0,+1)=(12.601, 2.736, 0.391)  acc(k=-1,0,+1)=(0.956, 0.594, 0.509)

# FX PIPELINE — FORENSICS REPORT

Read-only diagnostic. Verifies three empirical claims from session_handoff_2.

## Task 1 — Raw provenance & weekend audit

- file: `data/raw/fx_raw.parquet`
- shape: (5947, 176)
- index dtype: datetime64[us]
- index min/max: 2010-01-01 00:00:00 / 2026-04-13 00:00:00
- rows with Sat/Sun index (weekday>=5): **1700**
- EUR on 2010-01-04: **0.696190** (handoff claim ~1.4364)
- columns with >95% zero daily diffs (pegged): **10**
  - list (col, frac_zero_diffs):
    - AED: 1.0000
    - ANG: 1.0000
    - BHD: 1.0000
    - BMD: 1.0000
    - JOD: 1.0000
    - OMR: 1.0000
    - QAR: 1.0000
    - SAR: 1.0000
    - TMM: 1.0000
    - ZWD: 0.9923

## Task 2 — Return autocorrelation & dispersion audit

- file: `data/processed/fx_log_returns.parquet`  shape=(5946, 170)
- index min/max: 2010-01-02 00:00:00 / 2026-04-13 00:00:00
- weekend rows in the returns file: **1700**
- wrote: `data/forensics/forensics_returns_audit.csv`  rows=170

### Majors focus

| currency | n_obs | n_weekend_rows | frac_zero | std | std_real_bench | ac1_all | ac1_weekday | ac1_nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EUR | 5946 | 1700 | 0.0061 | 0.003322 | 0.0060 | 0.2438 | 0.3420 | 0.2437 |
| GBP | 5946 | 1700 | 0.0072 | 0.003525 | 0.0060 | 0.2205 | 0.3549 | 0.2205 |
| JPY | 5946 | 1700 | 0.0209 | 0.005430 | 0.0060 | -0.1341 | 0.2007 | -0.1344 |
| CHF | 5946 | 1700 | 0.0074 | 0.020948 | 0.0065 | -0.4744 | 0.0184 | -0.4744 |
| AUD | 5946 | 1700 | 0.0138 | 0.004104 | 0.0070 | 0.3005 | 0.3437 | 0.3003 |
| CAD | 5946 | 1700 | 0.0232 | 0.006058 | 0.0050 | -0.3171 | 0.1355 | -0.3174 |
| SEK | 5946 | 1700 | 0.0600 | 0.100568 | 0.0075 | -0.0589 | -0.0855 | -0.3782 |
| NOK | 5946 | 1700 | 0.0582 | 0.004833 | 0.0080 | 0.2216 | 0.3483 | 0.2017 |

## Task 3 — Leak isolation (1.89 Sharpe)

- USD index built equal-weight from: ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'SEK', 'NOK'] (v1 logic, NOT LOO)
- temporal split: split_idx=3567  train [2010-01-02..2019-10-08]  test [2019-10-09..2026-04-13]
- wrote: `data/forensics/forensics_leak_isolation.csv`  rows=160

- **mean sharpe_leaky** across 160 currencies: **2.4770**
- **mean sharpe_lagged**: **0.6784**
- median sharpe_leaky: 0.3955
- median sharpe_lagged: 0.2468
- mean acc_leaky: 0.4220   mean acc_lagged: 0.3758
- gap (leaky - lagged) mean sharpe: **1.7986**

### Majors row-by-row

| currency | sharpe_leaky | sharpe_lagged | acc_leaky | acc_lagged |
|---|---:|---:|---:|---:|
| EUR | 9.9423 | 2.8141 | 0.8226 | 0.6018 |
| GBP | 8.9907 | 2.3446 | 0.7860 | 0.5736 |
| JPY | -5.6998 | -1.4418 | 0.2917 | 0.4386 |
| CHF | 10.1739 | 2.5355 | 0.7961 | 0.5660 |
| AUD | 11.6810 | 2.6436 | 0.8150 | 0.5757 |
| CAD | 9.5136 | 2.6412 | 0.7322 | 0.5715 |
| SEK | 12.1958 | 3.0886 | 0.8012 | 0.5862 |
| NOK | 9.5939 | 1.9860 | 0.7915 | 0.5597 |

## Task 4 — Accuracy/Sharpe reconciliation

- split_date=2019-10-09  train_rows=3567  test_rows=2379
- running bm.process_currency() on 170 currencies
- wrote: `data/forensics/forensics_accuracy_sharpe.csv`  rows=151
- winner test_accuracy mean=0.4207  median=0.4050
- winner test_sharpe   mean=1.4017  median=1.3092
- count(test_accuracy<0.50 AND test_sharpe>0): **90**
- of those, mean_abs_ret_correct > mean_abs_ret_wrong: **78**

### Interpretation rules (stated, not editorialized)
- asymmetric payoffs => acc<0.50 with sharpe>0, k=0 already maximizes sharpe, and mean_abs_ret_correct > mean_abs_ret_wrong
- off-by-one misalignment => a k=±1 shift sharply raises accuracy toward >0.50 and/or flips the sharpe relative to k=0

### Majors row-by-row

| ccy | variant | acc | sharpe | MAR_correct | MAR_wrong | sharpe_-1 | sharpe_0 | sharpe_+1 | acc_-1 | acc_0 | acc_+1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EUR | ma_5 | 0.5938 | 2.7356 | 0.002100 | 0.001777 | 12.601 | 2.736 | 0.391 | 0.956 | 0.594 | 0.509 |
| GBP | lag_2 | 0.5879 | 2.7255 | 0.002463 | 0.002012 | 12.194 | 2.725 | 0.607 | 0.989 | 0.588 | 0.511 |
| JPY | ma_20 | 0.4050 | -1.8009 | 0.002454 | 0.002595 | -7.458 | -1.801 | 0.436 | 0.257 | 0.405 | 0.488 |
| CHF | ma_5 | 0.4041 | -3.5394 | 0.001760 | 0.002332 | -13.724 | -3.539 | -0.161 | 0.108 | 0.404 | 0.493 |
| AUD | ma_5 | 0.5887 | 3.0423 | 0.003002 | 0.002534 | 15.172 | 3.042 | 0.414 | 0.945 | 0.589 | 0.509 |
| CAD | ma_60 | 0.3991 | -2.8706 | 0.001563 | 0.001833 | -14.649 | -2.871 | -0.677 | 0.067 | 0.399 | 0.474 |
| SEK | ma_5 | 0.5210 | 1.2158 | 0.002918 | 0.002715 | 4.455 | 1.216 | -0.115 | 0.601 | 0.521 | 0.493 |
| NOK | ma_5 | 0.5900 | 2.8884 | 0.003656 | 0.003034 | 11.974 | 2.888 | 0.199 | 0.925 | 0.590 | 0.499 |

## CSV artifacts

- data/forensics/forensics_returns_audit.csv
- data/forensics/forensics_leak_isolation.csv
- data/forensics/forensics_accuracy_sharpe.csv
