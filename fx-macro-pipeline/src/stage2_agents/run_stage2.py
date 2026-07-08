import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral_agent import run as run_spectral
from granger_agent  import run as run_granger
from comparison_agent import run as run_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Stage 2: Spectral and Causal Analysis ===")

    # Agent 1 — Spectral
    logger.info("\n--- Agent 1: Spectral ---")
    spectral_results, spectral_preds, test, split_date = run_spectral(save=True)

    # Agent 2 — Granger / VAR
    logger.info("\n--- Agent 2: Granger / VAR ---")
    granger_results, granger_preds, _ = run_granger(save=True)

    # Agent 3 — Comparison
    logger.info("\n--- Agent 3: Comparison ---")
    combined, summary = run_comparison(test_log_returns=test, save=True)

    logger.info("\n=== Stage 2 complete ===")
    logger.info(f"Recommended model: {summary['recommendation'].upper()}")
    logger.info(
        f"Spectral — mean accuracy: {summary['spectral_mean_accuracy']:.4f}, "
        f"mean Sharpe: {summary['spectral_mean_sharpe']:.4f}"
    )
    logger.info(
        f"Granger  — mean accuracy: {summary['granger_mean_accuracy']:.4f}, "
        f"mean Sharpe: {summary['granger_mean_sharpe']:.4f}"
    )


if __name__ == "__main__":
    main()
