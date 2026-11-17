"""Experiment layer: financial metrics, the classical heuristic, risk forecasts, and run records."""

import json
import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from test_optimizer import request as optimizer_request  # noqa: E402

from qflab.experiments.metrics import financial_metrics, reconcile_cash  # noqa: E402
from qflab.methods.classical_portfolio import swap_search  # noqa: E402
from qflab.optimizer_worker import energy, solve_request  # noqa: E402


def equity_rows(values, start="2025-01-02"):
    from datetime import date, timedelta

    day = date.fromisoformat(start)
    rows = []
    for value in values:
        rows.append({"timestamp": f"{day}T21:00:00+00:00", "equity": value, "cash": value})
        day += timedelta(days=1)
    return rows


def test_metrics_match_hand_calculations_and_state_their_conventions():
    values = [100.0, 110.0, 99.0, 108.9]
    result = financial_metrics(equity_rows(values), fills=[], starting_cash=100.0)
    returns = np.array([0.10, -0.10, 0.10])
    assert result["total_return"] == pytest.approx(0.089)
    assert result["max_drawdown"] == pytest.approx(-0.10)
    assert result["volatility"] == pytest.approx(returns.std(ddof=1) * math.sqrt(252))
    assert result["sharpe"] == pytest.approx(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
    downside = math.sqrt((np.minimum(returns, 0) ** 2).mean())
    assert result["sortino"] == pytest.approx(returns.mean() / downside * math.sqrt(252))
    assert result["cagr"] == pytest.approx(1.089 ** (365.25 / 3) - 1)
    assert "252" in result["conventions"]["annualization"]
    flat = financial_metrics(equity_rows([100.0, 100.0, 100.0]), fills=[], starting_cash=100.0)
    assert flat["sharpe"] is None and flat["sortino"] is None and flat["max_drawdown"] == 0.0


def test_turnover_fees_and_gross_equity_come_from_the_saved_fills():
    fills = [
        {
            "timestamp": "2025-01-03T21:00:00+00:00",
            "side": "BUY",
            "units": "5",
            "price": "10",
            "commission": "0.05",
            "instrument_id": "A.SIM",
        },
        {
            "timestamp": "2025-01-04T21:00:00+00:00",
            "side": "SELL",
            "units": "2",
            "price": "12",
            "commission": "0.02",
            "instrument_id": "A.SIM",
        },
    ]
    result = financial_metrics(equity_rows([100.0, 99.95, 109.93]), fills, starting_cash=100.0)
    assert result["fees"] == pytest.approx(0.07)
    assert result["traded_notional"] == pytest.approx(74.0)
    assert result["turnover"] == pytest.approx(74.0 / np.mean([100.0, 99.95, 109.93]))
    assert result["ending_equity_gross_of_fees"] == pytest.approx(110.0)
    assert result["fill_count"] == 2


def test_cash_reconciliation_detects_a_ledger_that_is_not_self_financing():
    fills = [
        {
            "timestamp": "2025-01-03T21:00:00+00:00",
            "side": "BUY",
            "units": "5",
            "price": "10.004",
            "commission": "0.05",
            "instrument_id": "A.SIM",
        }
    ]
    rows = [
        {"timestamp": "2025-01-02T21:00:00+00:00", "cash": 100.0, "equity": 100.0},
        {"timestamp": "2025-01-03T21:00:00+00:00", "cash": 49.93, "equity": 99.95},
    ]
    assert reconcile_cash(rows, fills, 100.0)["reconciles"] is True  # notional rounds to cents
    rows[1]["cash"] = 60.0
    report = reconcile_cash(rows, fills, 100.0)
    assert report["reconciles"] is False and report["largest_difference"] == pytest.approx(10.07)


def test_swap_search_is_feasible_seeded_and_respects_its_budget():
    reference = solve_request(optimizer_request(), "a" * 64)
    model = reference["canonical_model"]
    optimum = int(reference["objective_integer"])
    first = swap_search(model, seed=3, max_evaluations=200)
    assert first == swap_search(model, seed=3, max_evaluations=200)
    assert len(first["selected"]) == 3 and first["selected"] == sorted(first["selected"])
    assert int(first["objective_integer"]) == energy(model, first["selected"]) >= optimum
    assert first["status"] == "local_optimum" and first["evaluations"] <= 200
    tiny = swap_search(model, seed=3, max_evaluations=4)
    assert tiny["status"] == "budget_exhausted" and tiny["evaluations"] == 4
    assert int(tiny["objective_integer"]) >= int(first["objective_integer"])
    # A local optimum has no improving single swap.
    chosen = set(first["selected"])
    for out in chosen:
        for incoming in set(model["symbols"]) - chosen:
            assert energy(model, sorted(chosen - {out} | {incoming})) >= int(
                first["objective_integer"]
            )


def test_portfolio_moments_use_target_weights_and_the_solver_forecast():
    from qflab.experiments.risk_forecast import portfolio_moments

    reference = solve_request(optimizer_request(), "a" * 64)
    mean, std = portfolio_moments(reference)
    symbols = reference["canonical_model"]["symbols"]
    w = np.array([float(reference["target_weights"][s]) for s in symbols])
    mu = np.array([float(reference["diagnostics"]["mean_returns"][s]) for s in symbols])
    sigma = np.array(reference["diagnostics"]["covariance"], dtype=float)
    assert mean == pytest.approx(float(w @ mu))
    assert std == pytest.approx(math.sqrt(float(w @ sigma @ w)))
    assert w.sum() == pytest.approx(0.9)  # the cash reserve earns nothing in this model


def test_risk_lab_report_separates_targets_and_records_every_method(tmp_path):
    pytest.importorskip("qflab.backends.cudaq_backend", exc_type=ImportError)
    from qflab.experiments.risk_lab import run_risk_lab

    config = {
        "call": {"spot": 100.0, "strike": 105.0, "rate": 0.03, "volatility": 0.2, "maturity": 1.0},
        "truncation_std": 3.0,
        "confidence": 0.95,
        "reference_qubits": 3,
        "seeds": 4,
        "iqae": {"epsilons": [0.05, 0.02], "shots": 100},
        "monte_carlo_samples": [100, 1000],
        "sobol": {"samples_per_replicate": [16, 64], "replicates": 4},
        "grid_qubits": [2, 3],
        "strikes": [100.0, 110.0],
        "noise": {"qubits": 1, "two_qubit_depolarizing": [0.0, 0.05], "epsilon": 0.05, "seeds": 3},
    }
    report = run_risk_lab(config, target="qpp-cpu", seed=5)
    json.dumps(report, allow_nan=False)  # the whole record is plain JSON
    reference = report["reference"]
    assert reference["model_error"] == pytest.approx(
        reference["exact_discrete"] - reference["analytical_continuous"]
    )
    assert all(row["matches"] for row in report["circuit_verification"])
    by_method = {}
    for row in report["benchmark"]:
        by_method.setdefault(row["method"], []).append(row)
    assert set(by_method) == {"monte_carlo", "iqae", "sobol_qmc"}
    assert {r["target"] for r in by_method["iqae"]} == {"discrete"}
    assert {r["target"] for r in by_method["sobol_qmc"]} == {"continuous"}
    for row in report["benchmark"]:
        assert row["seeds"] == 4 and len(row["points"]) == 4
        assert 0 <= row["coverage"] <= 1 and row["median_work"] > 0
        assert row["evidence_label"]
    iqae_rows = sorted(by_method["iqae"], key=lambda r: r["budget"], reverse=True)
    assert iqae_rows[0]["median_work"] < iqae_rows[1]["median_work"]  # tighter costs more
    assert iqae_rows[1]["deepest_circuit"]["qubits"] == 4
    assert [row["scenario_qubits"] for row in report["grid_sweep"]] == [2, 3]
    assert report["grid_sweep"][1]["state_preparation"]["rotations"] == 15
    assert len(report["strike_sweep"]) == 2 and len(report["noise_study"]) == 2
    assert report["noise_study"][0]["two_qubit_depolarizing"] == 0.0


def test_committed_lab_config_runs_every_section():
    from pathlib import Path

    from qflab.lab import SECTIONS
    from qflab.ui.tearsheet import SECTIONS as SHOWN

    config = json.loads((Path(__file__).parents[1] / "configs" / "lab.json").read_text())
    assert [name for name in SECTIONS if name in config] == SECTIONS
    assert list(SHOWN) == SECTIONS  # every section has a place on the tearsheet
