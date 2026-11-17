import copy
import hashlib
import importlib.util
import itertools
import json
import math

import pytest

from qflab.optimizer_worker import energy
from qflab.qaoa import (
    classiq_problem,
    compile_qubo,
    load_experiment,
    prepare,
    run,
    summarize_samples,
    verify_qubo,
)


def financial_model(n=8, k=3):
    symbols = [f"A{i}.SIM" for i in range(n)]
    return {
        "schema_version": 1,
        "scale": "1000000000000",
        "symbols": symbols,
        "select_count": k,
        "constant": "17",
        "linear": {s: str((i - 3) * 7) for i, s in enumerate(symbols)},
        "pairs": [
            {"left": symbols[i], "right": symbols[j], "coefficient": str((i + j) % 5 - 2)}
            for i in range(n)
            for j in range(i + 1, n)
        ],
    }


@pytest.mark.parametrize(("n", "k"), [(2, 1), (3, 2), (8, 3), (8, 8)])
def test_qubo_matches_independent_financial_formula_and_enforces_cardinality(n, k):
    model = financial_model(n, k)
    original = copy.deepcopy(model)
    qubo = compile_qubo(model)
    valid, invalid = [], []
    for bits in itertools.product((0, 1), repeat=n):
        financial = (
            17
            + sum((i - 3) * 7 * bits[i] for i in range(n))
            + sum(((i + j) % 5 - 2) * bits[i] * bits[j] for i in range(n) for j in range(i + 1, n))
        )
        selected = [s for s, bit in zip(model["symbols"], bits, strict=True) if bit]
        actual = energy(qubo, selected)
        assert actual == financial + int(qubo["penalty_integer"]) * (sum(bits) - k) ** 2
        (valid if sum(bits) == k else invalid).append(actual)
    assert min(invalid) > max(valid)
    assert verify_qubo(model, qubo)["feasible_states"] == math.comb(n, k)
    assert model == original
    qubo["linear"][model["symbols"][0]] = "0"
    with pytest.raises(ValueError, match="translation"):
        verify_qubo(model, qubo)


def replay_record():
    symbols = ["A.SIM", "B.SIM", "C.SIM"]
    history = [
        {"timestamp": f"2025-01-{day}T21:00:00+00:00", "closes": dict.fromkeys(symbols, "100")}
        for day in ["02", "03", "06"]
    ]
    request = {
        "schema_version": 1,
        "data_cutoff": history[-1]["timestamp"],
        "symbols": symbols,
        "cash": "10000",
        "equity": "10000",
        "holdings": dict.fromkeys(symbols, "0"),
        "history": history,
        "settings": {
            "select_count": 2,
            "lookback_returns": 2,
            "forecast_horizon_sessions": 21,
            "risk_aversion": "3",
            "turnover_penalty": "0.001",
            "cash_reserve": "0.10",
        },
    }
    raw = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return {"request": request, "request_sha256": hashlib.sha256(raw).hexdigest()}


def prepared(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps([replay_record()]))
    output = tmp_path / "experiment"
    prepare(replay, 0, output)
    return replay, output


def test_preparation_freezes_one_past_only_request_and_detects_mutation(tmp_path):
    replay, output = prepared(tmp_path)
    reference, _ = load_experiment(output)
    assert reference["selected"] == ["A.SIM", "B.SIM"]
    request = json.loads((output / "request.json").read_text())
    assert request == replay_record()["request"]
    assert all(row["timestamp"] <= request["data_cutoff"] for row in request["history"])
    with pytest.raises(FileExistsError):
        prepare(replay, 0, output)
    with pytest.raises(ValueError, match="index"):
        prepare(replay, -1, tmp_path / "negative")
    (output / "qubo.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        load_experiment(output)
    record = replay_record()
    record["request"]["cash"] = "9999"
    replay.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="hash"):
        prepare(replay, 0, tmp_path / "bad_hash")


def test_sample_decoding_uses_each_outcome_and_reports_missing_feasible_solution():
    model = financial_model(3, 2)
    symbols = model["symbols"]
    optimum, best = min((energy(model, s), s) for s in itertools.combinations(symbols, 2))
    reference = {
        "canonical_model": model,
        "objective_integer": str(optimum),
        "selected": list(best),
    }
    bits = [int(s in best) for s in symbols]
    bad = {"solution": {"x": [0, 0, 0]}, "probability": 0.75, "cost": -999999}
    good = {"solution": {"x": bits}, "probability": 0.25, "cost": 999999}
    result = summarize_samples(reference, [bad, good])
    assert result["best_selected"] == list(best)
    assert result["objective_gap_integer"] == "0"
    assert result["feasible_probability"] == result["optimal_probability"] == 0.25
    bad["probability"] = 1
    result = summarize_samples(reference, [bad])
    assert result["status"] == "no_feasible_sample"
    assert result["best_selected"] is None
    assert result["objective_gap_integer"] is None
    with pytest.raises(ValueError, match="probabilities"):
        summarize_samples(reference, [good])


@pytest.mark.skipif(
    importlib.util.find_spec("classiq") is None, reason="optional quantum environment"
)
def test_actual_classiq_cost_and_offline_export_need_no_synthesis(tmp_path, monkeypatch):
    from classiq.applications.combinatorial_optimization import CombinatorialProblem

    _, output = prepared(tmp_path)
    model = financial_model()
    qubo = compile_qubo(model)
    problem = classiq_problem(qubo, 1)  # Checks all 256 actual SDK energies internally.
    assert problem.num_layers_ == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline build must never synthesize or execute")

    monkeypatch.setattr(CombinatorialProblem, "get_qprog", forbidden)
    attempt = run(output, layers=1, shots=1024, maxiter=20, build_only=True)
    assert (attempt / "portfolio.qmod").exists()
    assert json.loads((attempt / "run.json").read_text())["status"] == "built_offline"
