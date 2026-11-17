"""Financial metrics derived from a saved run's ledger files, with their conventions stated.

Inputs are the backtester's own outputs (``equity.csv`` rows and ``fills.json``); nothing here
re-simulates trading.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

CONVENTIONS = {
    "returns": "simple daily returns of account equity (cash + units x close)",
    "annualization": "252 sessions per year for volatility, Sharpe and Sortino",
    "risk_free_rate": "0; the minimum acceptable return for Sortino is also 0",
    "cagr": "(ending / starting equity) ^ (365.25 / calendar days) - 1",
    "max_drawdown": "largest peak-to-trough decline of daily equity, as a negative fraction",
    "turnover": "sum of absolute traded notional / mean daily equity over the period",
    "gross_of_fees": "ending equity plus fees paid; ignores what the fees would have earned",
}


def financial_metrics(equity_rows: list[dict], fills: list[dict], starting_cash: float) -> dict:
    equity = np.array([float(row["equity"]) for row in equity_rows])
    if len(equity) < 2:
        raise ValueError("Metrics need at least two equity observations")
    returns = equity[1:] / equity[:-1] - 1
    spread = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside = math.sqrt(float((np.minimum(returns, 0.0) ** 2).mean()))
    first = datetime.fromisoformat(equity_rows[0]["timestamp"])
    last = datetime.fromisoformat(equity_rows[-1]["timestamp"])
    days = (last - first).total_seconds() / 86400
    notional = sum(float(f["units"]) * float(f["price"]) for f in fills)
    fees = sum(float(f["commission"]) for f in fills)
    peak = np.maximum.accumulate(equity)
    return {
        "starting_equity": float(starting_cash),
        "ending_equity": float(equity[-1]),
        "total_return": float(equity[-1] / equity[0] - 1),
        "cagr": float((equity[-1] / equity[0]) ** (365.25 / days) - 1) if days > 0 else None,
        "volatility": spread * math.sqrt(252),
        "sharpe": float(returns.mean() / spread * math.sqrt(252)) if spread > 0 else None,
        "sortino": float(returns.mean() / downside * math.sqrt(252)) if downside > 0 else None,
        "max_drawdown": float((equity / peak - 1).min()),
        "fees": fees,
        "traded_notional": notional,
        "turnover": notional / float(equity.mean()),
        "fill_count": len(fills),
        "ending_equity_gross_of_fees": float(equity[-1]) + fees,
        "sessions": len(equity),
        "first_session": equity_rows[0]["timestamp"],
        "last_session": equity_rows[-1]["timestamp"],
        "conventions": CONVENTIONS,
    }


def reconcile_cash(equity_rows: list[dict], fills: list[dict], starting_cash: float) -> dict:
    """Rebuild daily cash from the fills alone and compare it with the recorded cash.

    A self-financing ledger changes cash only through trades and fees. Trade notionals are
    rounded to cents, as the engine does.
    """
    worst = 0.0
    for row in equity_rows:
        cash = starting_cash
        for fill in fills:
            if fill["timestamp"] <= row["timestamp"]:
                notional = round(float(fill["units"]) * float(fill["price"]), 2)
                cash -= notional if fill["side"] == "BUY" else -notional
                cash -= float(fill["commission"])
        worst = max(worst, abs(cash - float(row["cash"])))
    return {
        "reconciles": worst < 0.005 * max(1, len(fills)),
        "largest_difference": worst,
        "sessions_checked": len(equity_rows),
        "fills_used": len(fills),
    }
