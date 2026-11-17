"""Risk-lab references: the discrete model, its analytical target, and classical estimators."""

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from qflab.methods.classical import (  # noqa: E402
    black_scholes_call,
    monte_carlo_discrete,
    normal_loss_probability,
    sobol_call,
)
from qflab.problems.expectation import gaussian_loss, lognormal_call  # noqa: E402

CALL = {"spot": 100.0, "strike": 105.0, "rate": 0.03, "volatility": 0.2, "maturity": 1.0}


def test_black_scholes_matches_a_published_value_and_put_call_parity():
    # Hull, "Options, Futures and Other Derivatives": S=42, K=40, r=10%, sigma=20%, T=0.5.
    assert black_scholes_call(42, 40, 0.10, 0.20, 0.5) == pytest.approx(4.759, abs=5e-4)
    call = black_scholes_call(**CALL)
    d1 = (math.log(100 / 105) + (0.03 + 0.02) * 1.0) / 0.2
    put = 105 * math.exp(-0.03) * 0.5 * math.erfc(d1 / math.sqrt(2) - 0.2 / math.sqrt(2)) - (
        100 * 0.5 * math.erfc(d1 / math.sqrt(2))
    )
    assert call - put == pytest.approx(100 - 105 * math.exp(-0.03), abs=1e-9)


@pytest.mark.parametrize("qubits", [2, 3, 5, 7])
def test_discrete_call_is_a_normalized_reversible_bounded_problem(qubits):
    problem = lognormal_call(**CALL, num_qubits=qubits, truncation_std=3.0)
    assert len(problem.grid) == len(problem.probabilities) == 2**qubits
    assert problem.probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    assert (problem.probabilities > 0).all() and (np.diff(problem.grid) > 0).all()
    assert problem.scaled_payoff.min() >= 0 and problem.scaled_payoff.max() == pytest.approx(1.0)
    # Scaling is reversible: amplitude -> value -> amplitude.
    direct = math.exp(-0.03) * float(problem.probabilities @ np.maximum(problem.grid - 105.0, 0.0))
    assert problem.exact_value == pytest.approx(direct, rel=1e-12)
    assert problem.to_amplitude(problem.to_value(0.3)) == pytest.approx(0.3)
    assert 0 < problem.exact_amplitude < 1
    # The dropped tail mass is recorded, not silently renormalized away.
    assert problem.meta["truncated_tail_mass"] == pytest.approx(math.erfc(3 / math.sqrt(2)))


def test_discretization_error_shrinks_and_is_reported_separately_from_the_reference():
    analytical = black_scholes_call(**CALL)

    def error(qubits, truncation):
        problem = lognormal_call(**CALL, num_qubits=qubits, truncation_std=truncation)
        return abs(problem.exact_value - analytical)

    # With a wide grid the bin width dominates, and refining the grid reduces the error.
    wide = [error(q, 6.0) for q in (3, 5, 7, 9)]
    assert wide == sorted(wide, reverse=True) and wide[-1] < 1e-3 < wide[0]
    # With a narrow grid, refining cannot remove the dropped tail: a truncation floor remains.
    assert error(10, 2.0) > 50 * error(10, 6.0)


def test_monte_carlo_targets_the_discrete_model_and_reports_work():
    problem = lognormal_call(**CALL, num_qubits=4, truncation_std=3.0)
    first = monte_carlo_discrete(problem, samples=4000, seed=11, confidence=0.95)
    again = monte_carlo_discrete(problem, samples=4000, seed=11, confidence=0.95)
    first.pop("elapsed_seconds"), again.pop("elapsed_seconds")
    assert first == again and first["target"] == "discrete"
    assert first["resources"]["samples"] == 4000
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]
    hits = 0
    for seed in range(200):
        result = monte_carlo_discrete(problem, samples=2000, seed=seed, confidence=0.95)
        hits += result["ci_low"] <= problem.exact_amplitude <= result["ci_high"]
    assert 180 <= hits <= 200  # nominal 95% coverage on 200 seeds


def test_sobol_targets_the_continuous_model():
    result = sobol_call(**CALL, samples_per_replicate=1024, replicates=16, seed=5)
    assert result["target"] == "continuous"
    assert result["resources"]["samples"] == 1024 * 16
    assert result["estimate_value"] == pytest.approx(black_scholes_call(**CALL), abs=0.02)
    with pytest.raises(ValueError, match="power of two"):
        sobol_call(**CALL, samples_per_replicate=1000, replicates=4, seed=5)


def test_gaussian_loss_probability_problem():
    analytical = normal_loss_probability(mean=0.01, std=0.04, threshold=0.05)
    assert analytical == pytest.approx(0.5 * math.erfc(0.06 / 0.04 / math.sqrt(2)))
    problem = gaussian_loss(mean=0.01, std=0.04, threshold=0.05, num_qubits=6, truncation_std=4.0)
    assert set(np.unique(problem.scaled_payoff)) <= {0.0, 1.0}
    assert problem.payoff_scale == 1.0 and problem.post_factor == 1.0
    assert problem.exact_value == pytest.approx(analytical, abs=0.01)
    # A threshold beyond the grid gives a legitimate zero-probability problem.
    empty = gaussian_loss(mean=0.01, std=0.04, threshold=0.9, num_qubits=3, truncation_std=3.0)
    assert empty.exact_amplitude == 0.0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        lognormal_call(**{**CALL, "volatility": 0.0}, num_qubits=3, truncation_std=3.0)
    with pytest.raises(ValueError):
        lognormal_call(**CALL, num_qubits=0, truncation_std=3.0)
    with pytest.raises(ValueError):
        gaussian_loss(mean=0.0, std=-1.0, threshold=0.05, num_qubits=3, truncation_std=3.0)
