import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "outputs"
PLOTS_DIR     = Path(__file__).resolve().parents[2] / "data" / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.6


def load_all():
    log_returns      = pd.read_parquet(PROCESSED_DIR / "fx_log_returns.parquet")
    spectral_results = pd.read_csv(OUTPUTS_DIR / "spectral_results.csv", index_col=0)
    spectral_preds   = pd.read_parquet(OUTPUTS_DIR / "spectral_predictions.parquet")
    granger_results  = pd.read_csv(OUTPUTS_DIR / "granger_results.csv", index_col=0)
    comparison       = pd.read_csv(OUTPUTS_DIR / "comparison_report.csv", index_col=0)
    return log_returns, spectral_results, spectral_preds, granger_results, comparison


def get_split(log_returns):
    n = len(log_returns)
    split_idx = int(n * TRAIN_FRAC)
    return log_returns.index[split_idx]


def build_usd_index(log_returns):
    majors = ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK"]
    available = [c for c in majors if c in log_returns.columns]
    return log_returns[available].mean(axis=1)


def plot_currency(
    currency,
    log_returns,
    spectral_preds,
    spectral_results,
    comparison,
    usd_index,
    split_date,
):
    """
    Plot a 3-panel chart for one currency:
      Panel 1: Cumulative log return of currency vs USD index (full period)
      Panel 2: Spectral signal vs actual direction on test set
      Panel 3: Cumulative strategy return on test set
    """
    if currency not in log_returns.columns:
        print(f"{currency} not found in log returns.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        f"{currency} — Spectral Model Analysis",
        fontsize=14, fontweight="bold", y=0.98
    )

    ret   = log_returns[currency].dropna()
    usd   = usd_index.reindex(ret.index).dropna()
    train = ret[ret.index < split_date]
    test  = ret[ret.index >= split_date]

    # ── Panel 1: Cumulative returns ──────────────────────────────────────
    ax1 = axes[0]
    cumret_full = ret.cumsum()
    cumusd_full = (-usd).cumsum()  # invert: rising = USD weakening

    ax1.plot(cumret_full.index, cumret_full.values,
             color="#2563EB", linewidth=1.2, label=f"{currency} cumulative return")
    ax1.plot(cumusd_full.index, cumusd_full.values,
             color="#DC2626", linewidth=1.0, alpha=0.7, linestyle="--",
             label="USD weakness index")
    ax1.axvline(split_date, color="gray", linestyle=":", linewidth=1.0, label="Train/test split")
    ax1.axhline(0, color="black", linewidth=0.4)
    ax1.set_ylabel("Cumulative log return")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.set_title("Full period: currency vs USD weakness", fontsize=10)

    # ── Panel 2: Signal vs actual direction (test set) ───────────────────
    ax2 = axes[1]
    if currency in spectral_preds.columns:
        pred   = spectral_preds[currency].dropna()
        actual = np.sign(test.reindex(pred.index).dropna())
        common = pred.index.intersection(actual.index)
        pred   = pred.loc[common]
        actual = actual.loc[common]

        correct   = pred == actual
        incorrect = pred != actual

        ax2.scatter(
            actual.index[correct], actual.values[correct],
            color="#16A34A", s=2, alpha=0.6, label="Correct prediction"
        )
        ax2.scatter(
            actual.index[incorrect], actual.values[incorrect],
            color="#DC2626", s=2, alpha=0.4, label="Incorrect prediction"
        )
        ax2.plot(pred.index, pred.values * 0.5,
                 color="#7C3AED", linewidth=0.8, alpha=0.5, label="Signal (scaled)")
        accuracy = correct.mean()
        ax2.set_title(
            f"Test set: predicted vs actual direction  |  accuracy = {accuracy:.1%}",
            fontsize=10
        )
    else:
        ax2.set_title("No spectral predictions for this currency", fontsize=10)

    ax2.set_ylabel("Direction (+1 / -1)")
    ax2.axhline(0, color="black", linewidth=0.4)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # ── Panel 3: Cumulative strategy return (test set) ───────────────────
    ax3 = axes[2]
    if currency in spectral_preds.columns:
        strategy_ret = pred * test.reindex(pred.index).dropna()
        strategy_ret = strategy_ret.loc[common]
        cum_strategy = strategy_ret.cumsum()
        cum_buyhold  = test.reindex(common).cumsum()

        ax3.plot(cum_strategy.index, cum_strategy.values,
                 color="#7C3AED", linewidth=1.2, label="Strategy (signal × return)")
        ax3.plot(cum_buyhold.index, cum_buyhold.values,
                 color="#2563EB", linewidth=1.0, alpha=0.6,
                 linestyle="--", label="Buy and hold")
        ax3.axhline(0, color="black", linewidth=0.4)

        # Annotate Sharpe if available
        if currency in comparison.index:
            sharpe = comparison.loc[currency, "spectral_sharpe"]
            ax3.set_title(
                f"Test set: cumulative strategy return  |  Sharpe = {sharpe:.2f}",
                fontsize=10
            )
        else:
            ax3.set_title("Test set: cumulative strategy return", fontsize=10)

        ax3.set_ylabel("Cumulative log return")
        ax3.legend(fontsize=8, loc="upper left")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax3.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    outpath = PLOTS_DIR / f"{currency}_spectral.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def plot_top_currencies(n_best=6, n_worst=3):
    """
    Auto-select the top N currencies by spectral Sharpe and plot them,
    plus the N worst for comparison.
    """
    log_returns, spectral_results, spectral_preds, granger_results, comparison = load_all()
    split_date  = get_split(log_returns)
    usd_index   = build_usd_index(log_returns)

    # Rank by spectral Sharpe
    ranked = comparison["spectral_sharpe"].dropna().sort_values(ascending=False)
    top    = ranked.head(n_best).index.tolist()
    worst  = ranked.tail(n_worst).index.tolist()
    targets = top + worst

    print(f"\nTop {n_best} by Sharpe: {top}")
    print(f"Bottom {n_worst} by Sharpe: {worst}\n")

    for ccy in targets:
        plot_currency(
            ccy,
            log_returns,
            spectral_preds,
            spectral_results,
            comparison,
            usd_index,
            split_date,
        )


def plot_specific(currencies):
    """Plot a specific list of currency codes."""
    log_returns, spectral_results, spectral_preds, granger_results, comparison = load_all()
    split_date = get_split(log_returns)
    usd_index  = build_usd_index(log_returns)

    for ccy in currencies:
        plot_currency(
            ccy,
            log_returns,
            spectral_preds,
            spectral_results,
            comparison,
            usd_index,
            split_date,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # e.g. python plot_results.py EUR JPY BRL
        currencies = [c.upper() for c in sys.argv[1:]]
        print(f"Plotting: {currencies}")
        plot_specific(currencies)
    else:
        # Default: plot top 6 + bottom 3 by Sharpe
        plot_top_currencies(n_best=6, n_worst=3)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
