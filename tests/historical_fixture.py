"""Deterministic source-shaped data for offline Python/Rust integration checks."""

import sys
from pathlib import Path

import pandas as pd

from qflab.market_data import freeze_snapshot, schedule

START = "2025-07-01"
END = "2025-07-10"


def source_frame(prices, start=START, end=END):
    sessions = schedule(start, end)
    dates = pd.DatetimeIndex([r["session_date"] for r in sessions], tz="America/New_York")
    rows = [
        {
            "Open": p - 0.5,
            "High": p + 1,
            "Low": p - 1,
            "Close": p,
            "Adj Close": p * 0.987654321,
            "Volume": 1_600_000_000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        }
        for p in prices
    ]
    return pd.DataFrame(rows, index=dates)


def make_fixture(path):
    return freeze_snapshot(
        Path(path),
        {
            "ALPHA": source_frame([100, 100, 200, 200, 100, 100]),
            "BETA": source_frame([50, 50, 50, 50, 100, 100]),
        },
        START,
        END,
        source={"provider": "fabricated offline test fixture; not Yahoo data"},
    )


if __name__ == "__main__":
    make_fixture(sys.argv[1])
