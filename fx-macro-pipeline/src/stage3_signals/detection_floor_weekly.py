"""
PART 2 — Weekly detection floor for cross-sectional rank-IC.

Re-measures the floor at the REAL weekly operating conditions; do NOT
extrapolate from the monthly floor. Uses the audit's make_panel harness
with non-overlapping holding period h=5 (so the floor cell matches the
resampler's non-overlap; otherwise the floor lies).

Grid (per brief):
  n_instruments in {10, 15}
  rho_cross in {0.3, 0.6}
  n_periods ~ 835 (and ~789 for the n=15 listed span)
  ic_grid {0.005, 0.01, 0.015, 0.02, 0.03, 0.05}
  >=60 seeds
  sided="greater", target det >= 0.8
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from stage2_agents.evaluation import decision_ttest, rank_ic_series  # noqa: E402

_AUDIT = PROJ / "data" / "layer2_audit" / "run_audit.py"
_spec = importlib.util.spec_from_file_location("layer2_audit_run", _AUDIT)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)  # type: ignore[union-attr]
make_panel = _audit.make_panel

OUT = PROJ / "data" / "layer3_momentum"
OUT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 20260620 + 200
SIDED = "greater"
TARGET = 0.8


def detection_rate(ic_true, n_inst, rho_cross, n_periods, h, n_seeds):
    """Detection rate at the cell — non-overlapping holding-period h
    (must equal the resampler's non-overlap to keep the floor honest)."""
    rng_master = np.random.default_rng(
        MASTER_SEED + int(1e5 * ic_true) + 1000 * n_inst
        + int(100 * rho_cross) + h + n_periods
    )
    fires = 0
    for _ in range(n_seeds):
        seed = int(rng_master.integers(1, 2**31 - 1))
        # n_periods here is the AUDIT-DAILY period count; non-overlapping
        # weekly windows at stride h=5 → we end up with n_periods/h points.
        # To get a target of ~835 weekly observations we use n_periods=835*h
        # = 4175.
        T = n_periods * h
        S, F = make_panel(
            n_periods=T, n_instruments=n_inst,
            ic_true=ic_true, axis="crosssectional",
            holding_period=h, rho_cross=rho_cross, seed=seed,
        )
        idx = np.arange(0, T, h)
        ric = rank_ic_series(S.iloc[idx], F.iloc[idx])
        v = decision_ttest(ric.values, alpha=0.05, sided=SIDED)
        if v["fires"]:
            fires += 1
    return fires / n_seeds


def measure(n_inst_grid=(10, 15), rho_grid=(0.3, 0.6),
            n_periods_grid=(835, 789),
            ic_grid=(0.005, 0.01, 0.015, 0.02, 0.03, 0.05),
            h=5, n_seeds=60):
    rows = []
    print(f"sweeping {len(n_inst_grid)*len(rho_grid)*len(n_periods_grid)*len(ic_grid)} "
          f"cells (n_seeds={n_seeds}, h={h}, sided={SIDED})")
    for n_inst in n_inst_grid:
        for rho in rho_grid:
            for n_periods in n_periods_grid:
                for ic in ic_grid:
                    rate = detection_rate(ic, n_inst, rho, n_periods, h, n_seeds)
                    print(f"  n={n_inst} rho={rho} n_periods={n_periods} "
                          f"ic={ic:.3f}  det={rate:.2f}")
                    rows.append({
                        "n_instruments": n_inst,
                        "rho_cross": rho,
                        "n_weeks": n_periods,
                        "ic_true": ic,
                        "holding_period_daily": h,
                        "n_seeds": n_seeds,
                        "sided": SIDED,
                        "detection_rate": rate,
                    })
    df = pd.DataFrame(rows)

    summary = []
    for n_inst in n_inst_grid:
        for rho in rho_grid:
            for n_periods in n_periods_grid:
                sub = df[(df.n_instruments == n_inst)
                         & (df.rho_cross == rho)
                         & (df.n_weeks == n_periods)
                         & (df.detection_rate >= TARGET)]
                floor = float(sub["ic_true"].min()) if len(sub) else float("nan")
                summary.append({
                    "n_instruments": n_inst, "rho_cross": rho,
                    "n_weeks": n_periods,
                    "min_detectable_ic": floor,
                })
    summary_df = pd.DataFrame(summary)
    return df, summary_df


def main():
    df, summary = measure()
    df.to_csv(OUT / "detection_floor_weekly.csv", index=False)
    summary.to_csv(OUT / "detection_floor_weekly_summary.csv", index=False)

    print("\nFloor table (n, rho, n_weeks, min_detectable_ic):")
    print(summary.to_string(index=False))

    # Conservative headline: n=15 (live), rho=0.6, n_weeks=789 (listed span,
    # post-EM-listing); EARLY n=10 span uses the n=10 row.
    live = summary[(summary.n_instruments == 15)
                   & (summary.rho_cross == 0.6)
                   & (summary.n_weeks == 789)]
    early = summary[(summary.n_instruments == 10)
                    & (summary.rho_cross == 0.6)
                    & (summary.n_weeks == 835)]
    live_floor = float(live["min_detectable_ic"].iloc[0]) if len(live) else float("nan")
    early_floor = float(early["min_detectable_ic"].iloc[0]) if len(early) else float("nan")

    msg = (
        f"\nWEEKLY GO/NO-GO: net rank-IC must exceed {live_floor:.3f} at n=15 "
        f"(and {early_floor:.3f} for the n=10 early span).\n"
        f"NB: n=10 applies to early-2010 (pre-EM listing, weeks 1..45) AND to "
        f"the 2017-02-24 mid-sample EM-outage week and any other "
        f"multi-pair drop weeks — not solely to the listing gap."
    )
    print(msg)
    with open(OUT / "PART2_WEEKLY_FLOOR.txt", "w") as f:
        f.write(msg + "\n")
        f.write("\nFloor table:\n")
        f.write(summary.to_string(index=False) + "\n")
    return live_floor, early_floor


if __name__ == "__main__":
    main()
