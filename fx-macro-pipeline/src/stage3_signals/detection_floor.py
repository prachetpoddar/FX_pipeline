"""
PART 0b — Cross-sectional rank-IC detection floor at THIS factor's
operating conditions. Read-only. Writes data/layer3_momentum/detection_floor.csv.

Operating conditions (Layer-3 momentum monthly tercile L/S):
  - holding_period h = 21 trading days (~ 1 month rebalance)
  - n_instruments: live universe size (probe {12, 15, 16} for sensitivity)
  - rho_cross: {0.3, 0.6} (FX cross-currency factor structure)
  - sided='greater' (registered positive; momentum premium hypothesis)

Confirmatory test in Part 3 is a SINGLE pre-registered formation (L=3) so the
floor is reported unmodified. The secondary BH-FDR floor across K=4 formations
is also reported for context only — it does not gate the confirmatory test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

from stage2_agents.evaluation import (  # noqa: E402
    decision_ttest,
    rank_ic_series,
    bh_fdr_decision,
)

_AUDIT = PROJ / "data" / "layer2_audit" / "run_audit.py"
_spec = importlib.util.spec_from_file_location("layer2_audit_run", _AUDIT)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)  # type: ignore[union-attr]
make_panel = _audit.make_panel

OUT = PROJ / "data" / "layer3_momentum"
OUT.mkdir(parents=True, exist_ok=True)
MASTER_SEED = 20260620 + 100


def detection_rate_at(ic_true, n_inst, rho_cross, h, n_seeds, sided="greater"):
    """Detection rate of decision_ttest on rank_ic_series at the given
    operating-condition cell.

    Uses NON-overlapping windows at stride h to mirror the audit's
    holding-period convention for h>1. n_periods=2500 (~10 years daily)
    is the audit's standard cell size."""
    n_periods = 2500
    rng_master = np.random.default_rng(MASTER_SEED + int(1000 * ic_true)
                                       + 10 * n_inst + int(10 * rho_cross)
                                       + h)
    fires = 0
    for _ in range(n_seeds):
        seed = int(rng_master.integers(1, 2**31 - 1))
        S, F = make_panel(
            n_periods=n_periods, n_instruments=n_inst,
            ic_true=ic_true, axis="crosssectional",
            holding_period=h, rho_cross=rho_cross, seed=seed,
        )
        # Non-overlapping windows when h>1
        idx = np.arange(0, n_periods, h)
        Sm = S.iloc[idx]
        Fm = F.iloc[idx]
        ric = rank_ic_series(Sm, Fm)
        v = decision_ttest(ric.values, alpha=0.05, sided=sided)
        if v["fires"]:
            fires += 1
    return fires / n_seeds


def measure_floor(n_inst_grid=(12, 15, 16), rho_grid=(0.3, 0.6),
                  ic_grid=(0.005, 0.01, 0.015, 0.02, 0.03,
                           0.04, 0.05, 0.07, 0.10),
                  h=21, n_seeds=60, target=0.8, sided="greater"):
    """Sweep and return a DataFrame; per (n, rho) the smallest ic_true
    with detection_rate >= target."""
    rows = []
    print(f"Sweeping {len(n_inst_grid)*len(rho_grid)*len(ic_grid)} cells "
          f"({n_seeds} seeds each, h={h}, sided={sided})...")
    for n_inst in n_inst_grid:
        for rho in rho_grid:
            for ic in ic_grid:
                rate = detection_rate_at(ic, n_inst, rho, h, n_seeds, sided)
                print(f"  n={n_inst} rho={rho} ic={ic:.3f}  det={rate:.2f}")
                rows.append({
                    "n_instruments": n_inst, "rho_cross": rho,
                    "ic_true": ic, "holding_period": h,
                    "n_seeds": n_seeds, "sided": sided,
                    "detection_rate": rate,
                })
    df = pd.DataFrame(rows)

    # Smallest ic_true with detection_rate >= target per (n, rho)
    floors = []
    for n_inst in n_inst_grid:
        for rho in rho_grid:
            sub = df[(df["n_instruments"] == n_inst)
                     & (df["rho_cross"] == rho)
                     & (df["detection_rate"] >= target)]
            if len(sub):
                floor = float(sub["ic_true"].min())
            else:
                floor = float("nan")  # not detectable in the swept grid
            floors.append({"n_instruments": n_inst, "rho_cross": rho,
                           "min_detectable_ic": floor})
    floors_df = pd.DataFrame(floors)
    return df, floors_df


def measure_floor_bh(n_inst, rho, h, n_seeds, k_signals=4, target=0.8,
                     sided="greater"):
    """BH-FDR-corrected detection floor across k_signals simultaneous true-null +
    one-true tests. For each seed we generate k_signals-1 null cross-sectional
    panels and 1 true-signal panel; we pass all four rank-IC series through
    decide_family with BH-FDR at alpha=0.05; we count detections on the
    true-signal slot.

    Reports, for the same ic grid, the BH-corrected detection rate."""
    ic_grid = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06)
    n_periods = 2500
    out_rows = []
    for ic_true in ic_grid:
        rng_master = np.random.default_rng(MASTER_SEED + 50000
                                           + int(10000 * ic_true))
        bh_fires_true_slot = 0
        for _ in range(n_seeds):
            seed = int(rng_master.integers(1, 2**31 - 1))
            series_list = []
            for j in range(k_signals):
                ic_j = ic_true if j == 0 else 0.0
                S, F = make_panel(
                    n_periods=n_periods, n_instruments=n_inst,
                    ic_true=ic_j, axis="crosssectional",
                    holding_period=h, rho_cross=rho,
                    seed=seed + 7 * j + 1,
                )
                idx = np.arange(0, n_periods, h)
                ric = rank_ic_series(S.iloc[idx], F.iloc[idx]).values
                series_list.append(ric)
            from stage2_agents.evaluation import decide_family
            v = decide_family(series_list, alpha=0.05, use_hac=True)
            # Convert two-sided p to one-sided 'greater' before BH
            # (decide_family runs two-sided; we re-derive one-sided here)
            ones = []
            for s in series_list:
                vv = decision_ttest(s, alpha=0.05, sided=sided)
                ones.append(vv["p_value"])
            from stage2_agents.evaluation import bh_fdr_decision
            fires = bh_fdr_decision(ones, alpha=0.05)
            if fires[0]:
                bh_fires_true_slot += 1
        rate = bh_fires_true_slot / n_seeds
        print(f"  BH (k={k_signals}) n={n_inst} rho={rho} ic={ic_true:.3f}  det={rate:.2f}")
        out_rows.append({"n_instruments": n_inst, "rho_cross": rho,
                         "ic_true": ic_true, "k_signals": k_signals,
                         "detection_rate_bh": rate})
    df = pd.DataFrame(out_rows)
    sub = df[df["detection_rate_bh"] >= target]
    floor_bh = float(sub["ic_true"].min()) if len(sub) else float("nan")
    return df, floor_bh


def main(live_n_inst: int = 15):
    print(f"PART 0b — operating-condition detection floor "
          f"(live universe size = {live_n_inst})")
    df, floors = measure_floor(n_seeds=60)
    df_path = OUT / "detection_floor.csv"
    df.to_csv(df_path, index=False)
    floors_path = OUT / "detection_floor_summary.csv"
    floors.to_csv(floors_path, index=False)

    # Live floor — interpolate to live universe size (use the closest n
    # in the grid that is also rho=0.6, the conservative case)
    print("\nFloor by (n, rho), smallest ic with det>=0.80:")
    print(floors.to_string(index=False))

    # Headline number: use the closest n_inst >= live (conservative), rho=0.6.
    target_rho = 0.6
    sub = floors[floors["rho_cross"] == target_rho].copy()
    sub["dist"] = (sub["n_instruments"] - live_n_inst).abs()
    sub = sub.sort_values(["dist", "n_instruments"]).reset_index(drop=True)
    live_floor = float(sub.iloc[0]["min_detectable_ic"])

    # Multiplicity context: BH across K=4 formations
    print(f"\nBH-corrected floor probe at n={live_n_inst}, rho={target_rho}, "
          f"K=4 formations:")
    bh_df, floor_bh = measure_floor_bh(
        n_inst=live_n_inst, rho=target_rho, h=21, n_seeds=60, k_signals=4,
    )
    bh_df.to_csv(OUT / "detection_floor_bh_k4.csv", index=False)
    print(f"BH floor (k=4): min_detectable_ic = {floor_bh}")

    msg = (
        f"\nGO/NO-GO THRESHOLD: net rank-IC must exceed {live_floor:.3f} "
        f"at n={live_n_inst} (single pre-registered formation, sided='greater').\n"
        f"For context (NOT the gating number): if we used BH-FDR across K=4 "
        f"formations the inflated floor at the same cell would be "
        f"~{floor_bh if np.isfinite(floor_bh) else 'undetectable in grid'}."
    )
    print(msg)

    with open(OUT / "PART0_FLOOR.txt", "w") as f:
        f.write(msg + "\n")
        f.write("\nFloor table (n, rho, min_detectable_ic):\n")
        f.write(floors.to_string(index=False) + "\n")
    return live_floor, floor_bh


if __name__ == "__main__":
    n_live = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(n_live)
