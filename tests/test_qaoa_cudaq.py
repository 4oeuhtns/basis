"""QAOA on CUDA-Q: Ising translation, circuit conventions, the solver loop, and the worker."""

import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from test_optimizer import request as optimizer_request  # noqa: E402
from test_qaoa import financial_model  # noqa: E402

from qflab.methods.qaoa_solver import (  # noqa: E402
    interpolate,
    ising_from_qubo,
    solve_qaoa,
    validate_solver,
)
from qflab.optimizer_worker import energy, solve_request  # noqa: E402
from qflab.qaoa import compile_qubo, verify_qubo  # noqa: E402

SETTINGS = {
    "method": "qaoa_cudaq",
    "layers": 1,
    "shots": 512,
    "final_shots": 2048,
    "grid": 6,
    "maxiter": 25,
    "seed": 7,
    "penalty_scale": "1",
    "target": "qpp-cpu",
}


def test_reduced_penalty_is_explicit_and_the_default_still_dominates():
    model = financial_model()
    full, reduced = compile_qubo(model), compile_qubo(model, penalty_scale="0.01")
    assert int(reduced["penalty_integer"]) < int(full["penalty_integer"])
    assert int(reduced["penalty_integer"]) == math.ceil(int(full["penalty_integer"]) * 0.01)
    assert verify_qubo(model, full)["all_infeasible_states_are_worse"] is True
    with pytest.raises(ValueError, match="insufficient"):
        verify_qubo(model, reduced)
    audit = verify_qubo(model, reduced, require_dominance=False)
    assert audit["all_infeasible_states_are_worse"] is False
    assert audit["qubo_and_ising_match"] is True
    with pytest.raises(ValueError):
        compile_qubo(model, penalty_scale="0")


@pytest.mark.parametrize(("n", "k"), [(3, 2), (8, 3)])
def test_ising_energy_equals_normalized_qubo_energy_for_every_bitstring(n, k):
    model = financial_model(n, k)
    qubo = compile_qubo(model)
    constant, field, couplings = ising_from_qubo(qubo)
    divisor = int(qubo["normalization_integer"])
    for bits in itertools.product((0, 1), repeat=n):
        z = [1 - 2 * b for b in bits]
        ising = (
            constant
            + sum(h * zi for h, zi in zip(field, z, strict=True))
            + sum(j * z[a] * z[b] for a, b, j in couplings)
        )
        chosen = [s for s, b in zip(model["symbols"], bits, strict=True) if b]
        assert ising == pytest.approx(energy(qubo, chosen) / divisor, abs=1e-9)


def test_interpolation_keeps_the_schedule_shape_when_adding_a_layer():
    assert interpolate([0.4, 0.6]) == pytest.approx([0.4, 0.4, 0.6, 0.6])
    deeper = interpolate([0.2, 0.4, 0.8, 0.6])
    assert deeper == pytest.approx([0.2, 0.3, 0.4, 0.8, 0.7, 0.6])


def test_solver_settings_are_validated():
    assert validate_solver(dict(SETTINGS)) == SETTINGS
    for key, bad in [("layers", 0), ("shots", 10), ("method", "other"), ("penalty_scale", 1.0)]:
        with pytest.raises(ValueError):
            validate_solver({**SETTINGS, key: bad})
    with pytest.raises(ValueError):
        validate_solver({**SETTINGS, "extra": 1})


# ---------------------------------------------------------------- CUDA-Q circuit checks
cudaq_backend = pytest.importorskip("qflab.backends.cudaq_backend", exc_type=ImportError)


def reference_distribution(field, couplings, gamma, beta):
    """Independent one-layer QAOA simulation; index bit i (least significant first) = qubit i."""
    n = len(field)
    index = np.arange(2**n)
    z = 1 - 2 * ((index[:, None] >> np.arange(n)) & 1)
    cost = z @ np.array(field) + sum(j * z[:, a] * z[:, b] for a, b, j in couplings)
    state = np.exp(-1j * gamma * cost) / math.sqrt(2**n)
    for qubit in range(n):
        partner = state[index ^ (1 << qubit)]
        state = math.cos(beta) * state - 1j * math.sin(beta) * partner
    return np.abs(state) ** 2


def test_circuit_matches_an_independent_simulation_and_the_sample_bit_order():
    field = [0.31, -0.42, 0.17]
    couplings = [(0, 1, 0.25), (0, 2, -0.4), (1, 2, 0.1)]
    sampler = cudaq_backend.QaoaSampler(field, couplings, layers=1, seed=3)
    assert np.allclose(sampler.probabilities([0.0, 0.0]), 1 / 8, atol=1e-12)
    gamma, beta = 0.7, 0.45
    expected = reference_distribution(field, couplings, gamma, beta)
    assert np.allclose(sampler.probabilities([gamma, beta]), expected, atol=1e-9)
    assert expected.max() - expected.min() > 0.1  # a non-trivial distribution
    shots = 40_000
    counts = sampler([gamma, beta], shots)
    assert sum(counts.values()) == shots
    for bits, count in counts.items():
        index = sum(int(b) << i for i, b in enumerate(bits))  # sampled strings: qubit 0 first
        assert count / shots == pytest.approx(expected[index], abs=0.012)
    resources = sampler.compiled_resources()
    assert resources["qubits"] == 3
    assert resources["multi_qubit_gates"] == 2 * len(couplings)


def test_solver_returns_an_independently_scored_feasible_selection_and_counts_all_work():
    reference = solve_request(optimizer_request(), "a" * 64)
    result = solve_qaoa(reference, SETTINGS, cudaq_backend.QaoaSampler, decision_seed=11)
    again = solve_qaoa(reference, SETTINGS, cudaq_backend.QaoaSampler, decision_seed=11)
    assert result["counts"] == again["counts"] and result["parameters"] == again["parameters"]
    model = reference["canonical_model"]
    assert result["status"] == "feasible"
    assert len(result["selected"]) == 3 and set(result["selected"]) <= set(model["symbols"])
    assert int(result["objective_integer"]) == energy(model, result["selected"])
    assert int(result["objective_gap_integer"]) >= 0
    assert result["exact_reference"]["selected"] == reference["selected"]
    work = result["resources"]
    assert work["objective_evaluations"] == len(result["trace"]) <= 6 * 6 + 25
    assert sum(stage["evaluations"] for stage in result["stages"]) == len(result["trace"])
    assert work["total_shots"] == work["objective_evaluations"] * 512 + 2048
    assert sum(result["counts"].values()) == 2048
    assert work["qubits"] == 8 and work["two_qubit_gates"] == 2 * 28
    # Sampled frequencies and the privileged exact distribution describe the same circuit.
    exact = result["exact_distribution"]
    assert result["feasible_probability"] == pytest.approx(exact["feasible_probability"], abs=0.05)
    assert result["uniform_baseline"]["feasible_probability"] == pytest.approx(56 / 256)
    # The optimizer starts from parameters it can only improve on.
    assert min(result["trace"]) <= result["trace"][0]


def test_no_feasible_sample_is_reported_without_inventing_a_portfolio():
    reference = solve_request(optimizer_request(), "a" * 64)

    class OnlyEmpty:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, parameters, shots):
            return {"00000000": shots}

        def probabilities(self, parameters):
            p = np.zeros(256)
            p[0] = 1.0
            return p

        def compiled_resources(self):
            return {"qubits": 8, "gates": 0, "multi_qubit_gates": 0, "depth": 0}

    result = solve_qaoa(reference, SETTINGS, OnlyEmpty, decision_seed=1)
    assert result["status"] == "no_feasible_sample"
    assert result["selected"] is None and result["objective_gap_integer"] is None
    assert result["feasible_probability"] == 0.0


def run_worker(payload: dict):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    worker = Path(__file__).parents[1] / "qflab" / "qaoa_worker.py"
    done = subprocess.run(
        [sys.executable, "-I", str(worker)], input=raw, capture_output=True, timeout=120
    )
    return raw, done


def test_worker_speaks_the_backtester_protocol_from_stdin_to_stdout(tmp_path):
    payload = {**optimizer_request(), "solver": SETTINGS}
    raw, done = run_worker(payload)
    assert done.returncode == 0, done.stderr.decode()
    response = json.loads(done.stdout)
    assert response["schema_version"] == 1 and response["method"] == "qaoa_cudaq"
    assert response["status"] == "feasible"
    assert response["request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert response["data_cutoff"] == payload["data_cutoff"]
    weights = response["target_weights"]
    assert list(weights) == payload["symbols"]
    assert [s for s, w in weights.items() if w != "0"] == response["selected"]
    assert sum(map(float, weights.values())) == pytest.approx(0.9)
    exact = solve_request(optimizer_request(), "a" * 64)
    assert response["canonical_model"] == exact["canonical_model"]
    assert response["diagnostics"]["exact_reference"]["selected"] == exact["selected"]
    assert response["diagnostics"]["mean_returns"] == exact["diagnostics"]["mean_returns"]
    assert len(done.stdout) < 200_000
    # Same bytes in, same answer out: the seed is derived from the request itself.
    assert json.loads(run_worker(payload)[1].stdout)["selected"] == response["selected"]


def test_worker_rejects_requests_without_valid_solver_settings():
    assert run_worker(optimizer_request())[1].returncode != 0
    bad = {**optimizer_request(), "solver": {**SETTINGS, "layers": 99}}
    assert run_worker(bad)[1].returncode != 0
