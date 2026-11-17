"""Daily closes from a backtest run's frozen snapshot, read without pandas.

The research-track studies that need a price history (quantum kernels, generative models) use
exactly the snapshot the paired backtest ran on, so they see the same data and nothing later.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_closes(run_directory: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Returns (session dates, symbols, closes[session, symbol])."""
    snapshot = Path(run_directory) / "snapshot"
    files = sorted(p for p in snapshot.glob("*.csv") if p.name != "calendar.csv")
    if not files:
        raise FileNotFoundError(f"No price files in {snapshot}")
    symbols, columns, dates = [], [], None
    for path in files:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        sessions = [row["session_date"] for row in rows]
        if dates is None:
            dates = sessions
        elif sessions != dates:
            raise ValueError(f"{path.name} has different sessions from the other files")
        symbols.append(path.stem)
        columns.append([float(row["close"]) for row in rows])
    return dates, symbols, np.array(columns).T


def basket_log_returns(closes: np.ndarray) -> np.ndarray:
    """Daily log return of an equal-weight, daily-rebalanced basket of all symbols."""
    simple = closes[1:] / closes[:-1] - 1
    return np.log1p(simple.mean(axis=1))
