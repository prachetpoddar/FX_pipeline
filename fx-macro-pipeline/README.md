# fx-macro-pipeline

A pre-registered quantitative research program across FX, macro, and event-driven signals. **Six independent research lanes; six honest negatives.** The result isn't a strategy — it's a characterization of *where the resolving power of disciplined non-institutional quant research runs out*, and one clean case where a signal fails for a reason more data can't fix.

The negatives are the point, and they're trustworthy because the discipline that produced them was never relaxed to produce something more flattering.

---

## The finding in one paragraph

Every lane was pre-registered: signs, horizons, and pass/fail floors frozen *before* any statistic touched returns, with positive controls, honest-band (raw / in-sample / out-of-sample) reporting, and walk-forward-pooled evaluation under Benjamini–Hochberg correction. Five of six lanes closed on **breadth** — too few *independent, clean* observations at non-institutional data access to resolve effects that were often real and direction-correct (there are three US presidential elections in a decade; there is no fourth, and no data purchase makes one). The sixth closed on a different and cleaner constraint — **signal structure**: with clean, sufficient, free data, the target's forecast errors carried no mechanism-predictable structure at all. That sixth lane is what makes the breadth finding non-circular: the methodology reaches a clean verdict when breadth permits, and still refuses to manufacture a positive.

**Full argument:** [`docs/PROGRAM_SYNTHESIS.md`](docs/PROGRAM_SYNTHESIS.md)

---

## What's here

```
src/
  stage1_collection/   1-min HistData ticks → daily OHLC → weekly panel (17:00 NY close); universe registration + validation
  stage2_agents/       spectral (STFT/CPSD) + Granger/VAR causality + structural-break filter + BH-FDR evaluation harness
  stage3_signals/      cross-sectional momentum & carry factors; dollar-neutral portfolios; detection-floor + no-lookahead + orientation tests
  stage4_structural/   macro overlays — event detection (EFW/IEF), trade-balance scoring, policy-volatility, commodity inverse-sector
tests/                 pytest suite: orientation, no-lookahead, cost accounting, evaluator calibration, weekly panel
results/               curated evidence — dashboard, per-currency spectral figures, factor & macro reports
docs/                  PROGRAM_SYNTHESIS.md (the payoff), CHANGES.md (audit log), forensics & audit reports, cross-domain transfer note
data/demo/             small Frankfurter-derived panel so Stage 2 runs on a fresh clone
```

Real `statsmodels` / `scipy` implementations, not stubs: STFT + cross-power spectral density, VAR + Granger with Benjamini–Hochberg FDR, structural-break windowing, CIP-forward carry with orientation and no-lookahead testing, detection-floor calibration.

## The methodology, briefly

- **Pre-registration** — sign / horizon / floor / verdict frozen before measurement; one-sided tests only for theory-given signs; no post-hoc sign flips.
- **Honest band** — raw / in-sample-ceiling / OOS-truth reported together, so an in-sample number can't masquerade as a result. (Caught a carry construction whose in-sample IC was 58% illusory.)
- **Independence-unit discipline** — significance sized to independent units (currencies, regimes, episodes, events), never to raw observation count.
- **Causal-cleanliness gates** — returns-independent validation (stationarity, timestamp causality, coverage) must pass *before* returns are touched; a gate failure halts the lane instead of prompting a tweak.
- **Kill rules over rescues** — a disappointing result is read under a pre-committed rule, never used to motivate a new construction. Expanding data or altering a spec after seeing a result is the cardinal sin.

## Quick start

```bash
pip install -r requirements.txt
pytest tests/                                   # synthetic-data tests, no external calls
python src/stage2_agents/run_stage2.py          # runs against the demo panel in data/demo/
```

Stage 1 (fresh HistData download) and parts of Stage 4 (World Bank / Fraser at runtime) need network access and are not required for the demo. See [`docs/PROGRAM_SYNTHESIS.md`](docs/PROGRAM_SYNTHESIS.md) for the full lane-by-lane account.

## A note on scope

This is research, not investment advice, and it deliberately ships no live strategy. Data sources with redistribution restrictions (HistData 1-minute ticks, Fraser EFW) are excluded; the code that consumes them is included. The demo panel is Frankfurter-derived (public, redistributable).

## License

MIT — see [`LICENSE`](LICENSE).
