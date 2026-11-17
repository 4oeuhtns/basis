"""Iterative amplitude estimation: the SDK-free algorithm, then the CUDA-Q circuit behind it."""

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from qflab.methods.iqae import IdealSampler, clopper_pearson, find_next_k, iqae  # noqa: E402
from qflab.problems.expectation import ExpectationProblem, lognormal_call  # noqa: E402

CALL = {"spot": 100.0, "strike": 105.0, "rate": 0.03, "volatility": 0.2, "maturity": 1.0}


def random_problem(qubits, seed, zeros=False):
    rng = np.random.default_rng(seed)
    p = rng.random(2**qubits)
    if zeros:
        p[rng.integers(0, len(p), size=max(1, len(p) // 3))] = 0.0
    g = rng.random(2**qubits)
    g[0], g[-1] = 0.0, 1.0
    return ExpectationProblem("random", np.arange(len(p), dtype=float), p / p.sum(), g, 1.0, 1.0)


def test_clopper_pearson_edges_and_coverage():
    assert clopper_pearson(0, 100, 0.05)[0] == 0.0
    assert clopper_pearson(100, 100, 0.05)[1] == 1.0
    low, high = clopper_pearson(30, 100, 0.05)
    assert low < 0.3 < high and high - low < 0.2
    assert clopper_pearson(0, 100, 0.05)[1] == pytest.approx(1 - 0.025 ** (1 / 100))


def test_first_round_measures_the_bare_preparation_and_later_rounds_grow():
    assert find_next_k(0, True, (0.0, 0.25), 2.0) == (0, True)
    k, _ = find_next_k(0, True, (0.10, 0.11), 2.0)
    assert k >= 1


@pytest.mark.parametrize("amplitude", [0.0, 0.003, 0.2, 0.5, 0.83, 1.0])
def test_interval_meets_the_precision_target_and_covers_the_truth(amplitude):
    epsilon, alpha, misses = 0.005, 0.05, 0
    for seed in range(60):
        result = iqae(IdealSampler(amplitude, seed), epsilon=epsilon, alpha=alpha, shots=100)
        assert result["status"] == "ok"
        assert (result["ci_high"] - result["ci_low"]) / 2 <= epsilon + 1e-12
        assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
        misses += not (result["ci_low"] - 1e-12 <= amplitude <= result["ci_high"] + 1e-12)
        rounds = result["rounds"]
        resources = result["resources"]
        assert resources["grover_calls"] == sum(r["shots"] * r["k"] for r in rounds)
        assert resources["state_preparation_calls"] == sum(
            r["shots"] * (2 * r["k"] + 1) for r in rounds
        )
        assert resources["shots"] == sum(r["shots"] for r in rounds)
        assert resources["max_grover_power"] == max(r["k"] for r in rounds)
    assert misses <= 6  # alpha = 5% over 60 seeds; the union bound makes misses rare


def test_same_seed_reproduces_and_work_grows_roughly_like_one_over_epsilon():
    first = iqae(IdealSampler(0.2, 3), epsilon=0.01, alpha=0.05, shots=100)
    again = iqae(IdealSampler(0.2, 3), epsilon=0.01, alpha=0.05, shots=100)
    assert first == again

    def work(epsilon):
        runs = [
            iqae(IdealSampler(0.2, seed), epsilon=epsilon, alpha=0.05, shots=100)
            for seed in range(20)
        ]
        return float(np.median([r["resources"]["state_preparation_calls"] for r in runs]))

    ratio = work(0.0005) / work(0.005)
    assert 4 < ratio < 40  # Monte Carlo would need 100 times the samples


def test_a_work_budget_stops_with_an_explicit_status_and_a_valid_interval():
    result = iqae(
        IdealSampler(0.2, 1), epsilon=1e-5, alpha=0.05, shots=100, max_state_preparation_calls=5000
    )
    assert result["status"] == "budget_exhausted"
    assert result["resources"]["state_preparation_calls"] <= 5000
    assert result["ci_low"] <= 0.2 <= result["ci_high"]
    with pytest.raises(ValueError):
        iqae(IdealSampler(0.2, 1), epsilon=0.6, alpha=0.05, shots=100)


# ---------------------------------------------------------------- CUDA-Q circuit checks
cudaq_backend = pytest.importorskip("qflab.backends.cudaq_backend", exc_type=ImportError)


@pytest.mark.parametrize(("qubits", "zeros"), [(1, False), (2, True), (3, False), (5, True)])
def test_state_preparation_loads_probabilities_and_payoff_exactly(qubits, zeros):
    problem = random_problem(qubits, seed=qubits, zeros=zeros)
    sampler = cudaq_backend.AmplitudeSampler(problem, seed=1)
    state = sampler.statevector(0)
    size = 2**qubits
    # Scenario j lives on qubits 0..n-1 (qubit 0 = least significant bit); the indicator is last.
    zero_branch = np.sqrt(problem.probabilities * (1 - problem.scaled_payoff))
    one_branch = np.sqrt(problem.probabilities * problem.scaled_payoff)
    assert np.allclose(state[:size], zero_branch, atol=1e-9)
    assert np.allclose(state[size:], one_branch, atol=1e-9)
    assert sampler.indicator_probability(0) == pytest.approx(problem.exact_amplitude, abs=1e-9)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_grover_powers_rotate_the_indicator_probability_as_predicted(k):
    problem = lognormal_call(**CALL, num_qubits=3, truncation_std=3.0)
    sampler = cudaq_backend.AmplitudeSampler(problem, seed=1)
    theta = math.asin(math.sqrt(problem.exact_amplitude))
    assert sampler.indicator_probability(k) == pytest.approx(
        math.sin((2 * k + 1) * theta) ** 2, abs=1e-9
    )


def test_reported_gate_counts_match_the_compiled_circuit():
    problem = lognormal_call(**CALL, num_qubits=3, truncation_std=3.0)
    sampler = cudaq_backend.AmplitudeSampler(problem, seed=1)
    for k in (0, 2):
        ours, compiled = sampler.resources(k), sampler.compiled_resources(k)
        assert ours["qubits"] == compiled["qubits"] == 4
        assert ours["gates"] == compiled["gates"]
        assert (
            ours["two_qubit_gates"] + ours["multi_controlled_gates"]
            == compiled["multi_qubit_gates"]
        )
    # Dense loading is visible: one rotation per scenario per level, about 2 * 2^n for A.
    assert sampler.resources(0)["rotations"] == 2 ** (3 + 1) - 1


def test_circuit_sampling_drives_iqae_to_the_exact_discrete_answer():
    problem = lognormal_call(**CALL, num_qubits=3, truncation_std=3.0)
    results = [
        iqae(cudaq_backend.AmplitudeSampler(problem, seed=seed), 0.01, 0.05, shots=100)
        for seed in (1, 1, 2)
    ]
    assert results[0] == results[1] != results[2]
    for result in results:
        assert result["ci_low"] <= problem.exact_amplitude <= result["ci_high"]


def test_unavailable_backend_is_an_explicit_state():
    problem = lognormal_call(**CALL, num_qubits=2, truncation_std=3.0)
    with pytest.raises(cudaq_backend.BackendUnavailable):
        cudaq_backend.AmplitudeSampler(problem, seed=1, target="no-such-target")


def test_noise_moves_the_estimate_away_from_the_ideal_answer():
    problem = lognormal_call(**CALL, num_qubits=2, truncation_std=3.0)
    ideal = cudaq_backend.AmplitudeSampler(problem, seed=4, target="density-matrix-cpu")
    noisy = cudaq_backend.AmplitudeSampler(
        problem, seed=4, target="density-matrix-cpu", noise={"two_qubit_depolarizing": 0.05}
    )
    theta, shots = math.asin(math.sqrt(problem.exact_amplitude)), 4000

    def ideal_probability(power):
        return math.sin((2 * power + 1) * theta) ** 2

    # Use the Grover power whose ideal outcome is farthest from a coin flip.
    k = max(range(3, 9), key=lambda power: abs(ideal_probability(power) - 0.5))
    expected = ideal_probability(k)
    assert abs(expected - 0.5) > 0.3
    assert abs(ideal(k, shots) / shots - expected) < 0.04
    # Depolarizing noise pulls the indicator toward 1/2, away from the ideal value.
    assert abs(noisy(k, shots) / shots - 0.5) < abs(expected - 0.5) - 0.1
