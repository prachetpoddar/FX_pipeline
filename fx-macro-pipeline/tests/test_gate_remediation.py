"""
Gate remediation regression test (FIX 5).

Re-runs the audit Task E cases (data/layer2_audit/run_audit.py task_E) against
the patched preprocessor_v2.validate_pair and asserts reject_rate ~ 0 on all
six cases. Uses the same MA(1) construction the audit used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from stage1_collection_v2.preprocessor_v2 import validate_pair  # noqa: E402


N_SEEDS = 50
MASTER_SEED = 20260620 + 4  # match audit seeding


def _benign_ma1_returns(n, std, target_ac1, rng):
    a = float(target_ac1)
    if abs(a) < 1e-6:
        theta = 0.0
    else:
        disc = max(0.0, 1.0 - 4 * a * a)
        theta = (1.0 - np.sqrt(disc)) / (2.0 * a)
    e = rng.normal(0, 1.0, n + 1)
    r = e[1:] + theta * e[:-1]
    r = r * (std / r.std(ddof=0))
    return r


def _build_close_from_returns(rets, start=1.20):
    return start * np.exp(np.cumsum(rets))


CASES = [
    ("clean_major",            0.006, None,   4000, 1, dict()),
    ("bid_ask_bounce_major",   0.006, -0.12,  4000, 1, dict()),
    ("high_vol_EM",            0.018, None,   4000, 2, dict()),
    ("crisis_year_patch",      0.006, None,   4000, 1, dict(crisis_patch=True)),
    ("thin_but_tradable",      0.006, None,   4000, 1, dict(holiday_flat=True)),
    ("de_facto_peg_no_note",   0.0003, None,  4000, 1, dict()),
]


def _reject_rate(case):
    name, std, ac1_t, n, tier, extra = case
    rng_master = np.random.default_rng(MASTER_SEED)
    fails = 0
    for _ in range(N_SEEDS):
        seed = int(rng_master.integers(1, 2**31 - 1))
        rng = np.random.default_rng(seed)
        if ac1_t is not None:
            rets = _benign_ma1_returns(n, std, ac1_t, rng)
        else:
            rets = rng.normal(0, std, n)
        if extra.get("crisis_patch"):
            patch_idx = n // 2
            rets[patch_idx:patch_idx + 40] = 0.0
        if extra.get("holiday_flat"):
            rets[200:211] = 0.0
        close = _build_close_from_returns(rets)
        dates = pd.bdate_range("2010-01-01", periods=len(close))
        cs = pd.Series(close, index=dates)
        res = validate_pair(f"{name}_{_}", cs, tier)
        if not res["status"].startswith("PASS"):
            fails += 1
    return fails / N_SEEDS


def test_all_six_cases_pass():
    results = {c[0]: _reject_rate(c) for c in CASES}
    print("\nreject_rate per case (after remediation):")
    for k, v in results.items():
        print(f"  {k:30s}  {v:.2f}")
    # Target: ~0 reject_rate on all six.
    for name, rr in results.items():
        assert rr <= 0.05, f"{name} reject_rate {rr:.2f} > 0.05"


if __name__ == "__main__":
    test_all_six_cases_pass()
    print("\nALL SIX CASES PASS")
