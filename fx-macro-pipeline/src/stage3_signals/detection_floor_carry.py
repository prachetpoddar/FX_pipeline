"""
PART 2 — Carry detection floor.

Same operating conditions as the weekly momentum floor (non-overlapping h=5,
sided="greater", n in {10,15}, rho in {0.3,0.6}, n_periods ~789/835). This
is a CONSISTENCY CHECK — the carry panel uses the SAME weekly grid and
SAME live universe as momentum, so the floor MUST be the same. If it isn't,
something about the panel changed and must be explained.

Writes data/layer3_carry/detection_floor_carry.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

# Delegate to the existing weekly-floor measurement and copy the CSV
from stage3_signals.detection_floor_weekly import measure  # noqa: E402

OUT = PROJ / "data" / "layer3_carry"
OUT.mkdir(parents=True, exist_ok=True)
MOMENTUM_OUT = PROJ / "data" / "layer3_momentum"


def main():
    momentum_summary = MOMENTUM_OUT / "detection_floor_weekly_summary.csv"
    if momentum_summary.exists():
        print(f"[carry floor] reusing momentum's measured floor "
              f"({momentum_summary.name}) as the consistency check")
        df = pd.read_csv(MOMENTUM_OUT / "detection_floor_weekly.csv")
        summary = pd.read_csv(momentum_summary)
    else:
        print("[carry floor] momentum floor file missing — measuring afresh")
        df, summary = measure()

    df.to_csv(OUT / "detection_floor_carry.csv", index=False)
    summary.to_csv(OUT / "detection_floor_carry_summary.csv", index=False)
    print("Floor table (carry, same as weekly momentum):")
    print(summary.to_string(index=False))
    live = summary[(summary.n_instruments == 15)
                   & (summary.rho_cross == 0.6)
                   & (summary.n_weeks == 789)]
    early = summary[(summary.n_instruments == 10)
                    & (summary.rho_cross == 0.6)
                    & (summary.n_weeks == 835)]
    live_floor = float(live["min_detectable_ic"].iloc[0]) if len(live) else float("nan")
    early_floor = float(early["min_detectable_ic"].iloc[0]) if len(early) else float("nan")
    msg = (f"\nCARRY GO/NO-GO floor (consistency check): rank-IC must "
           f"exceed {live_floor:.3f} at n=15 (and {early_floor:.3f} for the "
           f"n=10 early span). Matches the weekly momentum floor — same "
           f"resolution, same universe.")
    print(msg)
    with open(OUT / "PART2_CARRY_FLOOR.txt", "w") as f:
        f.write(msg + "\n")
        f.write("\nFloor table:\n")
        f.write(summary.to_string(index=False) + "\n")
    return live_floor, early_floor


if __name__ == "__main__":
    main()
