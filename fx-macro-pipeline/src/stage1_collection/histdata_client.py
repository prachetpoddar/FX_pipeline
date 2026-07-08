"""
histdata_client.py
==================

Programmatic fetcher for 1-minute FX bars from HistData.com via the
`histdatacom` third-party package. We wrap it rather than calling its CLI
directly so we can:

    - Catch failures per pair-year (HistData is occasionally flaky;
      some pair-year combinations simply don't exist)
    - Track which downloads succeeded and which didn't
    - Cache aggressively — never re-download a pair-year we already have
    - Surface coverage gaps as data, not as silent missing files

INSTALL
-------
On the target machine, before first use:
    conda activate fxpipeline
    pip install histdatacom

USAGE
-----
    from histdata_client import HistDataClient
    client = HistDataClient(cache_dir="data/raw/histdata_1min")
    report = client.download_universe(start_year=2010, end_year=2026)
    # report is a DataFrame: rows=pair, columns=year, value="ok"/"fail"/"missing"

The fetcher does NOT aggregate to daily — that's a separate step in
daily_aggregator.py. We keep the raw 1-minute files so re-aggregation
under different daily-close conventions is cheap.

HistData's 1-minute archive format per pair-year:
    DAT_ASCII_<PAIR>_M1_<YEAR>.csv
    DateTime;Open;High;Low;Close;Volume       (semicolon-separated)
    20120201 000000;1.306600;1.306600;1.306560;1.306560;0
    ...
    timestamps are EST (NYC) time per HistData spec.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class HistDataClient:
    def __init__(self, cache_dir, validate_install=True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if validate_install:
            self._validate_histdatacom_installed()

    @staticmethod
    def _validate_histdatacom_installed():
        try:
            import histdatacom  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "histdatacom is not installed. Run: pip install histdatacom"
            )

    def _expected_filename(self, pair_code, year):
        # histdatacom's downloaded filename pattern
        return self.cache_dir / pair_code.upper() / f"DAT_ASCII_{pair_code.upper()}_M1_{year}.csv"

    def _has_data(self, pair_code, year):
        path = self._expected_filename(pair_code, year)
        return path.exists() and path.stat().st_size > 1000

    def fetch_pair_year(self, pair_code, year):
        """
        Download 1-minute bars for one pair and one year.
        Returns: 'cached' | 'ok' | 'fail' | 'missing' (= server returned 404)

        NOTE: this calls histdatacom as a subprocess so we get isolation per
        request and don't have its internal state leak into ours.
        """
        if self._has_data(pair_code, year):
            return "cached"

        pair_lower = pair_code.lower()
        pair_upper = pair_code.upper()
        target = self._expected_filename(pair_code, year)
        target.parent.mkdir(parents=True, exist_ok=True)

        # histdatacom v0.78 CLI: -D download, -X extract zip, --data-directory dir.
        # It places extracted CSVs at:
        #     <cwd>/<data-directory>/ASCII/M1/<pair_lower>/<year>/DAT_ASCII_<PAIR>_M1_<year>.csv
        # so we let it download, then move the file to the location the rest of
        # the pipeline expects (cache_dir/<PAIR>/DAT_ASCII_<PAIR>_M1_<year>.csv).
        cmd = [
            sys.executable, "-m", "histdatacom",
            "-p", pair_lower,
            "-t", "1-minute-bar-quotes",
            "-s", f"{year}-01",
            "-e", f"{year}-12",
            "-f", "ascii",
            "-D",
            "-X",
            "--data-directory", str(self.cache_dir),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=str(self.cache_dir.parent),
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"  {pair_code} {year}: timeout")
            return "fail"

        if result.returncode != 0:
            logger.debug(f"  {pair_code} {year} stderr: {result.stderr[:500]}")
            if "404" in result.stderr or "not found" in result.stderr.lower():
                return "missing"
            return "fail"

        # Locate the actual downloaded CSV (histdatacom's path layout) and
        # relocate to the cache-dir layout the aggregator expects.
        filename = f"DAT_ASCII_{pair_upper}_M1_{year}.csv"
        candidates = [p for p in self.cache_dir.parent.rglob(filename)
                      if p.resolve() != target.resolve()]
        if not candidates:
            # subprocess succeeded but produced nothing -> server has no data
            return "missing"
        src = candidates[0]
        try:
            src.replace(target)
        except OSError:
            import shutil
            shutil.copyfile(src, target)
            src.unlink(missing_ok=True)

        if self._has_data(pair_code, year):
            return "ok"
        return "fail"

    def download_universe(self, universe_dict, start_year=2010, end_year=None):
        """
        Iterate over (pair, year) pairs and download. Returns a DataFrame
        with rows=pair, columns=year, values = status string.
        """
        if end_year is None:
            end_year = pd.Timestamp.today().year

        statuses = {}
        for i, (pair_code, meta) in enumerate(universe_dict.items()):
            logger.info(f"[{i+1}/{len(universe_dict)}] {pair_code}")
            pair_status = {}
            for year in range(start_year, end_year + 1):
                status = self.fetch_pair_year(pair_code, year)
                pair_status[year] = status
                logger.info(f"    {year}: {status}")
            statuses[pair_code] = pair_status

        report = pd.DataFrame(statuses).T
        report.index.name = "pair"
        return report
