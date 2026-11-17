"""Past-only mean/variance selection using one stdin JSON request; opens no data files."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal, localcontext

SCALE = 10**12


def dec(value) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("Financial inputs must be decimal strings")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("Non-finite financial input")
    return result


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() != timedelta(0):
        raise ValueError("Timestamps must explicitly use UTC")
    return result


def energy(model: dict, selected: tuple[str, ...] | list[str]) -> int:
    chosen = set(selected)
    return (
        int(model["constant"])
        + sum(int(model["linear"][s]) for s in chosen)
        + sum(
            int(p["coefficient"])
            for p in model["pairs"]
            if p["left"] in chosen and p["right"] in chosen
        )
    )


def solve_request(request: dict, request_hash: str) -> dict:
    with localcontext() as context:
        context.prec = 50
        return _solve(request, request_hash)


def _solve(request: dict, request_hash: str) -> dict:
    if (
        set(request)
        != {
            "schema_version",
            "data_cutoff",
            "symbols",
            "cash",
            "holdings",
            "equity",
            "history",
            "settings",
        }
        or request["schema_version"] != 1
    ):
        raise ValueError("Unsupported optimizer request schema")
    symbols = request["symbols"]
    if not 2 <= len(symbols) <= 16 or symbols != sorted(set(symbols)):
        raise ValueError("Supply 2–16 distinct, sorted instrument IDs")
    if any(not re.fullmatch(r"[A-Z0-9]{1,20}\.SIM", s) for s in symbols):
        raise ValueError("Unsupported instrument ID")
    settings = request["settings"]
    if set(settings) != {
        "select_count",
        "lookback_returns",
        "forecast_horizon_sessions",
        "risk_aversion",
        "turnover_penalty",
        "cash_reserve",
    }:
        raise ValueError("Unsupported model settings")
    k, lookback, horizon = (
        settings[name] for name in ["select_count", "lookback_returns", "forecast_horizon_sessions"]
    )
    if (
        any(type(n) is not int for n in [k, lookback, horizon])
        or not 1 <= k <= len(symbols)
        or not 2 <= lookback <= 2520
        or not 1 <= horizon <= 252
    ):
        raise ValueError("Invalid cardinality, lookback, or forecast horizon")
    risk, turnover, reserve = (
        dec(settings[name]) for name in ["risk_aversion", "turnover_penalty", "cash_reserve"]
    )
    if not 0 <= risk <= 1000 or not 0 <= turnover <= 1 or not 0 <= reserve < 1:
        raise ValueError("Invalid risk, turnover, or reserve parameter")
    cutoff = timestamp(request["data_cutoff"])
    history = request["history"]
    if len(history) != lookback + 1:
        raise ValueError("Request must contain exactly lookback_returns + 1 closes")
    previous = None
    prices = []
    for row in history:
        if set(row) != {"timestamp", "closes"} or set(row["closes"]) != set(symbols):
            raise ValueError("Incomplete price cross-section")
        when = timestamp(row["timestamp"])
        if when > cutoff or (previous is not None and when <= previous):
            raise ValueError("Future, duplicate, or out-of-order observation")
        previous = when
        values = [dec(row["closes"][s]) for s in symbols]
        if any(p <= 0 for p in values):
            raise ValueError("Prices must be positive")
        prices.append(values)
    if previous != cutoff:
        raise ValueError("Last observation must match data_cutoff")
    if set(request["holdings"]) != set(symbols):
        raise ValueError("Holdings/universe mismatch")
    holdings = [dec(request["holdings"][s]) for s in symbols]
    cash, equity = dec(request["cash"]), dec(request["equity"])
    if cash < 0 or equity <= 0 or any(q < 0 or q != q.to_integral_value() for q in holdings):
        raise ValueError("Invalid long-only portfolio state")
    if cash + sum(q * p for q, p in zip(holdings, prices[-1], strict=True)) != equity:
        raise ValueError("Cash, holdings, prices and equity disagree")
    returns = [
        [b / a - 1 for a, b in zip(prices[t - 1], prices[t], strict=True)]
        for t in range(1, len(prices))
    ]
    means = [sum(row[i] for row in returns) / lookback for i in range(len(symbols))]
    covariance = [
        [
            sum((row[i] - means[i]) * (row[j] - means[j]) for row in returns) / (lookback - 1)
            for j in range(len(symbols))
        ]
        for i in range(len(symbols))
    ]
    mu = [m * horizon for m in means]
    sigma = [[c * horizon for c in row] for row in covariance]
    weight = (1 - reserve) / k
    previous_membership = [int(q > 0) for q in holdings]

    def coefficient(value: Decimal) -> str:
        return str(int((value * SCALE).to_integral_value(rounding=ROUND_HALF_EVEN)))

    # Financial objective = risk * w'Σw - μ'w + turnover/K * membership changes.
    # Cardinality is enforced by enumeration; no artificial QUBO penalty is needed here.
    model = {
        "schema_version": 1,
        "scale": str(SCALE),
        "symbols": symbols,
        "select_count": k,
        "constant": coefficient(turnover / k * sum(previous_membership)),
        "linear": {
            s: coefficient(
                risk * weight**2 * sigma[i][i]
                - weight * mu[i]
                + turnover / k * (1 - 2 * previous_membership[i])
            )
            for i, s in enumerate(symbols)
        },
        "pairs": [
            {
                "left": symbols[i],
                "right": symbols[j],
                "coefficient": coefficient(2 * risk * weight**2 * sigma[i][j]),
            }
            for i in range(len(symbols))
            for j in range(i + 1, len(symbols))
        ],
    }
    candidates = [
        (energy(model, selection), selection) for selection in itertools.combinations(symbols, k)
    ]
    best_energy, selected = min(candidates)  # Lexicographic tie-break; no randomness.
    unit = weight.quantize(Decimal("0.000000000001"), rounding=ROUND_DOWN)
    weights = {s: "0" for s in symbols}
    for i, symbol in enumerate(selected):
        weights[symbol] = str(unit if i < k - 1 else 1 - reserve - unit * (k - 1))
    return {
        "schema_version": 1,
        "method": "exact_enumeration",
        "status": "feasible",
        "data_cutoff": request["data_cutoff"],
        "request_sha256": request_hash,
        "selected": list(selected),
        "target_weights": weights,
        "objective_integer": str(best_energy),
        "canonical_model": model,
        "diagnostics": {
            "evaluated_portfolios": len(candidates),
            "return_observations": lookback,
            "first_observation": history[0]["timestamp"],
            "last_observation": history[-1]["timestamp"],
            "forecast_horizon_sessions": horizon,
            "mean_returns": {s: str(mu[i]) for i, s in enumerate(symbols)},
            "covariance": [[str(c) for c in row] for row in sigma],
            "previous_membership": dict(zip(symbols, previous_membership, strict=True)),
            "model": "sample daily simple-return mean and unbiased covariance, both scaled "
            "linearly by horizon; membership-change penalty is a turnover proxy",
            "quantization": "each polynomial coefficient rounded to 1e-12; exact minimum "
            "refers to this discrete financial objective",
        },
    }


def main() -> None:
    raw = sys.stdin.buffer.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError("Optimizer request exceeds 1 MiB")
    result = solve_request(json.loads(raw), hashlib.sha256(raw).hexdigest())
    sys.stdout.write(json.dumps(result, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
