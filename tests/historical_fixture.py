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


def make_monthly_fixture(path, mode="monthly"):
    start = "2025-01-01" if mode == "year" else "2025-03-28"
    end = (
        "2026-01-01"
        if mode == "year"
        else ("2025-04-01" if mode == "monthly_terminal" else "2025-04-04")
    )
    calendar = schedule(start, end)
    count = len(calendar)
    alpha = (
        [100 if r["session_date"] < "2025-01-31" else 200 for r in calendar]
        if mode == "year"
        else [100] + [200] * (count - 1)
    )
    beta = [50] * count
    if mode == "monthly_future":
        alpha[3:] = [900] * (count - 3)
        beta[3:] = [10] * (count - 3)
    raw = {"ALPHA": source_frame(alpha, start, end), "BETA": source_frame(beta, start, end)}
    # Exact cents simplify the independent hand ledger for this timing fixture.
    for frame in raw.values():
        frame["Adj Close"] = frame["Close"]
    return freeze_snapshot(
        Path(path),
        raw,
        start,
        end,
        source={"provider": "fabricated monthly timing fixture; not Yahoo data"},
    )


def make_optimizer_fixture(path, future=False):
    start, end = "2024-12-27", "2025-03-05"
    calendar = schedule(start, end)
    raw = {}
    for j in range(8):
        prices = []
        price = 100 + 3 * j
        for session in calendar:
            day = session["session_date"]
            fast = (j < 3) if day < "2025-02-01" else (3 <= j < 6)
            price += 0.5 if fast else 0.02
            prices.append(round(price * (1.3 if future and day >= "2025-02-10" else 1), 2))
        raw[f"A{j}"] = source_frame(prices, start, end)
    return freeze_snapshot(
        Path(path),
        raw,
        start,
        end,
        source={"provider": "fabricated causal optimizer fixture; not Yahoo data"},
    )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2].startswith("optimizer"):
        make_optimizer_fixture(sys.argv[1], future=sys.argv[2] == "optimizer_future")
    elif len(sys.argv) > 2:
        make_monthly_fixture(sys.argv[1], sys.argv[2])
    else:
        make_fixture(sys.argv[1])
