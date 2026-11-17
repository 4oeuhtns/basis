import copy
import hashlib
import itertools
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from qflab.optimizer_worker import SCALE, energy, solve_request


def request():
    symbols = [f"A{i}.SIM" for i in range(8)]
    prices = [100.0] * 8
    history = []
    for t, day in enumerate(["02", "03", "06", "07"]):
        if t:
            prices = [
                p * (1 + (i - 3) * 0.002 + [0.003, -0.001, 0.002][t - 1] * (i + 1) / 8)
                for i, p in enumerate(prices)
            ]
        history.append(
            {
                "timestamp": f"2025-01-{day}T21:00:00+00:00",
                "closes": dict(zip(symbols, map(str, prices), strict=True)),
            }
        )
    return {
        "schema_version": 1,
        "data_cutoff": history[-1]["timestamp"],
        "symbols": symbols,
        "cash": "10000",
        "holdings": dict.fromkeys(symbols, "0"),
        "equity": "10000",
        "history": history,
        "settings": {
            "select_count": 3,
            "lookback_returns": 3,
            "forecast_horizon_sessions": 21,
            "risk_aversion": "3",
            "turnover_penalty": "0.001",
            "cash_reserve": "0.10",
        },
    }


def test_exact_solver_and_all_binary_energies_agree_with_independent_numpy_arithmetic():
    data = request()
    result = solve_request(data, "a" * 64)
    symbols = data["symbols"]
    p = np.array([[float(row["closes"][s]) for s in symbols] for row in data["history"]])
    returns = p[1:] / p[:-1] - 1
    mu, covariance = returns.mean(axis=0) * 21, np.cov(returns, rowvar=False, ddof=1) * 21

    def financial(selected):
        x = np.array([int(s in selected) for s in symbols])
        w = x * 0.3
        return 3 * (w @ covariance @ w) - mu @ w + 0.001 / 3 * x.sum()

    # Every bitstring checks the financial polynomial, independently of cardinality enforcement.
    for bits in itertools.product([0, 1], repeat=8):
        selected = [s for s, bit in zip(symbols, bits, strict=True) if bit]
        assert energy(result["canonical_model"], selected) / SCALE == pytest.approx(
            financial(selected), abs=1e-10
        )
    reference = min(itertools.combinations(symbols, 3), key=financial)
    assert result["selected"] == list(reference)
    assert int(result["objective_integer"]) == energy(result["canonical_model"], reference)
    assert result["diagnostics"]["evaluated_portfolios"] == 56
    assert sum(map(Decimal, result["target_weights"].values())) == Decimal("0.9")
    assert sum(Decimal(w) > 0 for w in result["target_weights"].values()) == 3
    np.testing.assert_allclose(
        np.array(result["diagnostics"]["covariance"], dtype=float),
        covariance,
        rtol=1e-10,
        atol=1e-14,
    )


def test_flat_forecasts_use_actual_membership_and_deterministic_ties():
    data = request()
    for row in data["history"]:
        row["closes"] = dict.fromkeys(data["symbols"], "100")
    assert solve_request(data, "a" * 64)["selected"] == data["symbols"][:3]
    selected = [data["symbols"][0], *data["symbols"][-2:]]
    for symbol in selected:
        data["holdings"][symbol] = "10"
    data["cash"] = "7000"
    assert solve_request(data, "a" * 64)["selected"] == selected


@pytest.mark.parametrize("bad", ["future", "duplicate", "cash", "cardinality", "nan"])
def test_worker_rejects_inconsistent_or_future_inputs(bad):
    data = request()
    if bad == "future":
        data["history"][-1]["timestamp"] = "2025-01-08T21:00:00+00:00"
    elif bad == "duplicate":
        data["history"][0]["timestamp"] = data["history"][1]["timestamp"]
    elif bad == "cash":
        data["cash"] = "9999"
    elif bad == "cardinality":
        data["settings"]["select_count"] = 9
    else:
        data["history"][0]["closes"][data["symbols"][0]] = "NaN"
    with pytest.raises(ValueError):
        solve_request(data, "a" * 64)


def test_stdin_protocol_hashes_exact_request_and_needs_no_dataset_path(tmp_path):
    script = Path(__file__).resolve().parents[1] / "qflab" / "optimizer_worker.py"
    data = request()
    raw = json.dumps(data).encode()
    (tmp_path / "future_prices.csv").write_text("future information must not be read")
    response = subprocess.run(
        [sys.executable, "-I", str(script)],
        input=raw,
        capture_output=True,
        cwd=tmp_path,
        check=True,
    )
    result = json.loads(response.stdout)
    assert result["request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result == solve_request(copy.deepcopy(data), hashlib.sha256(raw).hexdigest())
