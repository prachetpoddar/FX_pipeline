"""
Orientation tests for Layer-3 carry.

  - Known high-yielder pairs (USDMXN, USDZAR) rank in the top tercile
    > 90% of weeks.
  - Known low-yielder pairs (USDJPY, USDCHF) rank in the bottom tercile
    > 90% of weeks.
  - POSITIVE CONTROL: inverting the orientation (negating the primary
    signal) flips top↔bottom tercile placement. If the test passes both
    the real and inverted maps, it isn't actually checking orientation.
  - The CIP secondary signal has the SAME within-pair rank ordering as
    the rate-diff primary (they should agree on direction even if scaled
    differently).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from stage3_signals.carry import build_weekly_carry_panel  # noqa: E402


def _tercile_fracs(signal: pd.DataFrame, pair: str):
    ranks = signal.rank(axis=1, pct=True)
    s = ranks[pair].dropna()
    top = float((s >= 2.0 / 3.0).mean())
    bot = float((s <= 1.0 / 3.0).mean())
    return top, bot


def test_high_yielder_top_tercile():
    panel = build_weekly_carry_panel()
    sig = panel["signal_primary"]
    for p in ["USDMXN", "USDZAR"]:
        top, bot = _tercile_fracs(sig, p)
        assert top >= 0.90, (
            f"{p} (known high yielder) ranked top tercile only {top:.3f} "
            "of weeks; expected >= 0.90. Orientation likely inverted."
        )
        assert bot <= 0.05, f"{p} should never be bottom tercile; got {bot:.3f}"


def test_low_yielder_bottom_tercile():
    panel = build_weekly_carry_panel()
    sig = panel["signal_primary"]
    for p in ["USDJPY", "USDCHF"]:
        top, bot = _tercile_fracs(sig, p)
        assert bot >= 0.90, (
            f"{p} (known low yielder) ranked bottom tercile only "
            f"{bot:.3f} of weeks; expected >= 0.90"
        )
        assert top <= 0.05, f"{p} should never be top tercile; got {top:.3f}"


def test_positive_control_inverted_signal_flips_tercile():
    """If we NEGATE the primary signal, USDMXN should now rank BOTTOM
    tercile most weeks. If this test ever passes both the real and the
    inverted map, the test wasn't actually sensitive to orientation."""
    panel = build_weekly_carry_panel()
    sig = -panel["signal_primary"]
    top_mxn, bot_mxn = _tercile_fracs(sig, "USDMXN")
    top_jpy, bot_jpy = _tercile_fracs(sig, "USDJPY")
    assert bot_mxn >= 0.90, (
        f"POSITIVE CONTROL FAILED: under an inverted carry signal, "
        f"USDMXN should rank bottom tercile most weeks; got {bot_mxn:.3f}"
    )
    assert top_jpy >= 0.90, (
        f"POSITIVE CONTROL FAILED: under an inverted signal USDJPY should "
        f"rank top tercile most weeks; got {top_jpy:.3f}"
    )


def test_primary_and_cip_secondary_agree_on_ranks():
    """Within each week's cross-section, the two signals must produce
    the same rank order (mean Spearman across weeks > 0.99). They differ
    only in scale: rate_diff_pct vs cip-fwd-pts/spot."""
    panel = build_weekly_carry_panel()
    p = panel["signal_primary"]
    s = panel["signal_secondary"]
    common = p.index.intersection(s.index)
    p = p.loc[common]; s = s.loc[common]
    # Per-week Spearman across pairs
    from scipy import stats as scistats
    rhos = []
    for w in common:
        a = p.loc[w].dropna()
        b = s.loc[w].dropna()
        common_pairs = a.index.intersection(b.index)
        if len(common_pairs) < 3:
            continue
        rho = scistats.spearmanr(a.loc[common_pairs], b.loc[common_pairs])[0]
        if np.isfinite(rho):
            rhos.append(rho)
    mean_rho = float(np.mean(rhos))
    assert mean_rho > 0.99, (
        f"Primary and CIP-secondary should agree on cross-sectional rank order "
        f"(mean Spearman {mean_rho:.4f} expected > 0.99). If they don't, the "
        "CIP sign convention is wrong."
    )


if __name__ == "__main__":
    test_high_yielder_top_tercile(); print("OK high_yielder_top_tercile")
    test_low_yielder_bottom_tercile(); print("OK low_yielder_bottom_tercile")
    test_positive_control_inverted_signal_flips_tercile()
    print("OK positive_control_inverted_signal_flips_tercile")
    test_primary_and_cip_secondary_agree_on_ranks()
    print("OK primary_and_cip_secondary_agree_on_ranks")
