"""Research-track algorithms (build plan sections 12 to 14), checked on the NumPy reference
runner so that they are verified independently of the quantum SDK."""

import itertools
import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from test_qaoa import financial_model  # noqa: E402

from qflab.circuit import Circuit, NumpyRunner, simulate  # noqa: E402
from qflab.methods.gas import (  # noqa: E402
    grover_adaptive_search,
    grover_circuit,
    quantization_audit,
    quantize,
    quantized_value,
    value_circuit,
)
from qflab.methods.generative import train_born_machine  # noqa: E402
from qflab.methods.iqae import IdealSampler  # noqa: E402
from qflab.methods.kernels import (  # noqa: E402
    exact_kernel,
    fit_ridge,
    predict,
    rbf_kernel,
    sampled_kernel,
)
from qflab.methods.linear_solvers import (  # noqa: E402
    estimate_inverse_quadratic,
    hhl_program,
    qsvt_program,
    recover_vector,
    solution_state,
)
from qflab.methods.mcmc import (  # noqa: E402
    QuantumProposal,
    autocorrelation_time,
    gibbs_matrix,
    local_proposal,
    metropolis_chain,
    metropolis_matrix,
    parallel_tempering,
    sector_model,
    spectral_gap,
    total_variation,
    uniform_move,
    uniform_proposal,
)
from qflab.methods.mlae import mlae, power_schedule  # noqa: E402
from qflab.methods.qaoa_circuits import xy_qaoa_circuit  # noqa: E402
from qflab.methods.qsp import odd_chebyshev_fit, r_response, reciprocal_target, solve_phases  # noqa: E402
from qflab.methods.relaxation import (  # noqa: E402
    encode,
    objective_terms,
    solve_relaxation,
    sparsify,
    verify_relaxation,
)
from qflab.methods.subspace import (  # noqa: E402
    clustered_covariance,
    diagonal_selection,
    krylov_circuits,
    localized_covariance,
    market_covariance,
    project,
    spectrum_reference,
)
from qflab.problems.linear_systems import (  # noqa: E402
    describe,
    geometric_spectrum,
    minimum_variance,
    spd_with_spectrum,
)


def k_hot(n, k):
    return [[int(i in combo) for i in range(n)] for combo in itertools.combinations(range(n), k)]


def unit_scale_model(n, k):
    """``financial_model`` with scale 1, so objective values are ordinary-sized numbers."""
    return {**financial_model(n, k), "scale": "1"}


# --------------------------------------------------------------------------------- MLAE
def test_mlae_is_accurate_covers_and_counts_work_like_iqae():
    amplitude = 0.2143
    runs = [mlae(IdealSampler(amplitude, s), power_schedule(7), 100, 0.05) for s in range(200)]
    coverage = np.mean([r["ci_low"] <= amplitude <= r["ci_high"] for r in runs])
    assert coverage >= 0.9
    assert np.median([abs(r["estimate"] - amplitude) for r in runs]) < 3e-3
    assert power_schedule(4) == [0, 1, 2, 4]
    assert runs[0]["resources"]["work"] == 100 * sum(2 * k + 1 for k in power_schedule(7))
    assert runs[0]["resources"]["max_grover_power"] == 32


def test_mlae_handles_amplitudes_at_the_boundary():
    for amplitude in (0.0, 1.0):
        found = mlae(IdealSampler(amplitude, 1), power_schedule(5), 50, 0.05)
        assert found["ci_low"] <= amplitude <= found["ci_high"]
        assert abs(found["estimate"] - amplitude) < 1e-3


# ------------------------------------------------------------------ Grover adaptive search
def test_quantized_objective_of_every_k_hot_string_fits_the_register():
    model = financial_model(8, 3)
    rounded = quantize(model, 6)
    values = [quantized_value(rounded, bits) for bits in k_hot(8, 3)]
    assert max(values) - min(values) <= 2**5 - 1


def test_value_circuit_writes_the_rounded_objective_into_the_register():
    model, n, m = financial_model(5, 2), 5, 5
    rounded = quantize(model, m)
    for bits in k_hot(n, 2):
        circuit = Circuit(n + m)
        for i, bit in enumerate(bits):
            if bit:
                circuit.x(i)
        state = simulate(circuit.extend(value_circuit(rounded, n, -3)))
        index = int(np.argmax(np.abs(state)))
        assert abs(state[index]) == pytest.approx(1.0)
        assert index >> n == (quantized_value(rounded, bits) - 3) % 2**m


def test_one_grover_rotation_amplifies_exactly_the_strings_below_the_threshold():
    model, n, k, m = financial_model(5, 2), 5, 2, 5
    rounded = quantize(model, m)
    strings = k_hot(n, k)
    values = sorted({quantized_value(rounded, bits) for bits in strings})
    threshold = values[1]  # marks the strings at the minimum value
    marked = [b for b in strings if quantized_value(rounded, b) < threshold]
    state = simulate(grover_circuit(rounded, n, k, threshold, 1))
    probabilities = np.abs(state) ** 2
    assert probabilities[2**n :].sum() < 1e-12  # the value register is uncomputed
    theta = math.asin(math.sqrt(len(marked) / len(strings)))
    found = sum(probabilities[sum(b << i for i, b in enumerate(bits))] for bits in marked)
    assert found == pytest.approx(math.sin(3 * theta) ** 2)


def test_grover_adaptive_search_finds_the_rounded_optimum_and_reports_its_costs():
    model = financial_model(5, 2)
    found = grover_adaptive_search(model, NumpyRunner(1), 5, seed=1)
    rounded = quantize(model, 5)
    assert found["quantized_value"] == min(quantized_value(rounded, b) for b in k_hot(5, 2))
    assert len(found["selected"]) == 2
    assert found["oracle_calls"] <= 400 and found["circuit_executions"] == len(found["history"])
    audit = quantization_audit(model, range(5, 9))
    assert all(row["fits_register"] for row in audit)


# ------------------------------------------------------------------- linear-system solvers
def test_qsp_phases_reproduce_an_odd_bounded_reciprocal():
    function, scale = reciprocal_target(5.0, 1e-3)
    solved = solve_phases(odd_chebyshev_fit(function, 1e-3 * scale / 2))
    assert solved["residual"] < 1e-10 and solved["degree"] % 2 == 1
    x = np.linspace(0.2, 1.0, 200)
    assert np.max(np.abs(r_response(solved["phases"], x).real * x / scale - 1)) < 2e-3
    assert np.max(np.abs(function(np.linspace(-1, 1, 2001)))) <= 0.8 + 1e-9


def test_hhl_is_exact_when_the_eigenvalues_sit_on_the_clock_grid():
    eigenvalues = [k / 16 for k in (3, 6, 10, 15)]
    matrix, rhs = spd_with_spectrum(eigenvalues, 4), np.ones(4)
    reference = describe(matrix, rhs)
    program = hhl_program(
        matrix, rhs, 4, (min(eigenvalues), max(eigenvalues)), evolution_time=2 * math.pi
    )
    state, probability = solution_state(program, NumpyRunner(0))
    assert abs(np.vdot(state, reference["solution_state"])) ** 2 == pytest.approx(1.0, abs=1e-10)
    ideal = (program.norm_factor * reference["solution_norm"] / 2) ** 2
    assert probability == pytest.approx(ideal, rel=1e-9)
    with pytest.raises(ValueError, match="wraps"):
        hhl_program(matrix, rhs, 4, (0.1, 1.0), evolution_time=7.0)


def test_trotterized_hhl_converges_to_the_exact_evolution():
    matrix, rhs = spd_with_spectrum(geometric_spectrum(2, 4), 1), np.ones(2)
    reference = describe(matrix, rhs)
    bounds = (reference["eigenvalue_min"], reference["eigenvalue_max"])
    exact_state, _ = solution_state(hhl_program(matrix, rhs, 5, bounds), NumpyRunner(0))
    infidelity = []
    for steps in (1, 4):
        program = hhl_program(matrix, rhs, 5, bounds, evolution="trotter", trotter_steps=steps)
        state, _ = solution_state(program, NumpyRunner(0))
        infidelity.append(1 - abs(np.vdot(state, exact_state)) ** 2)
    assert infidelity[1] < infidelity[0] and infidelity[1] < 1e-3


def test_qsvt_solver_matches_the_classical_solution_and_its_success_rate():
    matrix = spd_with_spectrum(geometric_spectrum(4, 6), 2)
    reference = describe(matrix, np.ones(4))
    bounds = (reference["eigenvalue_min"], reference["eigenvalue_max"])
    program = qsvt_program(matrix, np.ones(4), bounds, 1e-3)
    state, probability = solution_state(program, NumpyRunner(0))
    assert abs(np.vdot(state, reference["solution_state"])) ** 2 > 1 - 1e-5
    ideal = (program.norm_factor * reference["solution_norm"] / 2) ** 2
    assert probability == pytest.approx(ideal, rel=0.01)  # 1/x is approximated to 1e-3
    assert program.info["block_encoding_error"] < 1e-12
    assert program.info["block_encoding_calls"] == program.info["polynomial_degree"]


def test_readout_recovers_signed_solutions_from_counts_alone():
    for seed in range(50):
        matrix = spd_with_spectrum(geometric_spectrum(4, 8), seed)
        reference = describe(matrix, np.ones(4))
        if (reference["solution"] < 0).any():
            break
    else:
        pytest.fail("no test system with a negative solution entry")
    bounds = (reference["eigenvalue_min"], reference["eigenvalue_max"])
    program = qsvt_program(matrix, np.ones(4), bounds, 1e-3)
    found = recover_vector(program, NumpyRunner(1), 40_000)
    assert found["settings"] == 3 and found["shots"] == 120_000
    truth = reference["solution_state"]
    assert (
        min(np.linalg.norm(found["vector"] - truth), np.linalg.norm(found["vector"] + truth)) < 0.06
    )
    weights = minimum_variance(matrix, found["vector"])["weights"]
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.sign(weights) == np.sign(minimum_variance(matrix)["weights"]))
    scalar = estimate_inverse_quadratic(program, NumpyRunner(2), 200_000)
    assert scalar["value"] == pytest.approx(float(np.ones(4) @ reference["solution"]), rel=0.05)


def test_minimum_variance_weights_sum_to_one_and_cannot_be_improved():
    covariance = spd_with_spectrum([0.2, 0.5, 1.0, 3.0], 7)
    found = minimum_variance(covariance)
    assert found["weights"].sum() == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        step = rng.normal(size=4)
        step -= step.mean()  # keeps the budget
        moved = found["weights"] + 0.05 * step
        assert moved @ covariance @ moved >= found["variance"] - 1e-15


# ---------------------------------------------------------------- quantum-assisted MCMC
def test_quantum_proposal_is_symmetric_so_metropolis_keeps_the_target():
    model = sector_model([0, 0, 1, 1], 0.25, 0.6, 0.15, 2.0, 3)
    proposal = QuantumProposal(model, NumpyRunner(0), 0)
    matrix = proposal.matrix([proposal.draw_settings() for _ in range(3)])
    assert np.allclose(matrix, matrix.T, atol=1e-12) and np.allclose(matrix.sum(axis=1), 1)
    pi = model.distribution()
    for transition in (
        metropolis_matrix(model, matrix),
        metropolis_matrix(model, local_proposal(4)),
        metropolis_matrix(model, uniform_proposal(4)),
        gibbs_matrix(model),
    ):
        assert np.allclose(pi @ transition, pi, atol=1e-12)
        assert 0 < spectral_gap(transition, pi) <= 1


def test_chains_sample_the_target_and_report_their_work():
    model = sector_model([0, 0, 1, 1], 0.25, 0.6, 0.15, 1.0, 3)
    pi = model.distribution()
    rng = np.random.default_rng(0)
    chain = metropolis_chain(model, uniform_move(4, rng), 20_000, 0, rng)
    assert total_variation(chain, pi) < 0.05
    tempered, work = parallel_tempering(model, [0.25, 0.5, 1.0], 2_000, 15, rng)
    assert work == 6_000 and len(tempered) == 2_000
    proposal = QuantumProposal(model, NumpyRunner(1), 1)
    quantum = metropolis_chain(model, proposal, 200, 0, rng)
    assert proposal.executions == 200 and quantum.max() < 16
    assert 0.5 < autocorrelation_time(rng.normal(size=5_000)) < 2


# ----------------------------------------------------------- sample-based diagonalization
def test_subspace_projection_is_exact_on_all_coordinates_and_bounded_below_that():
    covariance = clustered_covariance(16, 4, 2.0, 1)
    reference = spectrum_reference(covariance)
    full = project(covariance, range(16), reference)
    assert abs(full["eigenvalue_error"]) < 1e-12 and full["vector_error"] < 1e-12
    for m in (2, 4, 8):
        found = project(covariance, diagonal_selection(covariance, m), reference)
        assert found["vector_error"] >= 1 - reference["best_possible_overlap"][m - 1] - 1e-9
        assert found["eigenvalue_error"] >= -1e-12  # a sub-block cannot exceed the top eigenvalue
    market = spectrum_reference(market_covariance(16, 0.4, 1))
    assert market["best_possible_overlap"][3] == pytest.approx(4 / 16, abs=0.05)


def test_krylov_circuits_prepare_the_evolved_trial_state():
    linalg = pytest.importorskip("scipy.linalg")
    covariance = localized_covariance(8, 2, 3.0, 1)
    hamiltonian = -covariance / np.linalg.norm(covariance)
    start = np.ones(8) / math.sqrt(8)
    ideal = krylov_circuits(covariance, "uniform", [0.0, 1.5])
    assert np.allclose(simulate(ideal[0]), start)
    target = linalg.expm(-1.5j * hamiltonian) @ start
    assert np.allclose(simulate(ideal[1]), target)
    trotter = krylov_circuits(covariance, "uniform", [1.5], evolution="trotter", trotter_steps=4)
    assert abs(np.vdot(simulate(trotter[0]), target)) ** 2 > 0.999


# ---------------------------------------------------------------- quantum relaxation
def test_relaxation_reproduces_the_sparsified_objective_and_bounds_it_from_below():
    model = unit_scale_model(8, 3)
    linear, pairs = objective_terms(model)
    for density in (0.0, 0.25, 0.5, 1.0):
        kept = sparsify(pairs, density)
        placement = encode(8, kept)
        assert all(placement[a][0] != placement[b][0] for a, b in kept)
        audit = verify_relaxation(linear, kept, 0.1, placement, 3)
        assert audit["encoded_states_reproduce_objective"] and audit["is_lower_bound"]
    # A complete interaction graph leaves nothing to share: one qubit per variable, Z axis,
    # which is the ordinary diagonal encoding.
    complete = {pair: 1.0 for pair in itertools.combinations(range(8), 2)}
    dense = encode(8, complete)
    assert len({q for q, _ in dense}) == 8 and all(axis == "Z" for _, axis in dense)
    assert len({q for q, _ in encode(8, {})}) == 3  # no interactions: three variables per qubit


def test_relaxation_solver_returns_k_funds_and_counts_its_shots():
    model = unit_scale_model(6, 2)
    found = solve_relaxation(model, NumpyRunner(0), 0.25, 100, 8, [0.0, 1.0])
    assert len(found["selected"]) == 2
    assert found["total_shots"] == 100 * found["circuit_executions"]
    assert found["needed_repair"] == (found["selected_before_repair"] != 2)


# ----------------------------------------------------------- kernels and generative models
def test_fidelity_kernel_is_a_valid_kernel_and_sampling_estimates_it():
    data = np.random.default_rng(0).uniform(0, math.pi, (6, 3))
    runner = NumpyRunner(0)
    kernel = exact_kernel(runner, data, data, 2)
    assert np.allclose(np.diag(kernel), 1) and np.allclose(kernel, kernel.T)
    assert np.linalg.eigvalsh(kernel).min() > -1e-10
    sampled, circuits = sampled_kernel(runner, data, data, 2, 4_000)
    assert circuits == 15 and np.abs(sampled - kernel).max() < 0.05
    assert np.allclose(exact_kernel(runner, data[:2], data, 2), kernel[:2])


def test_kernel_ridge_classifier_fits_a_separable_problem():
    points = np.random.default_rng(1).normal(size=(40, 2))
    labels = np.where(points[:, 0] > 0, 1, -1)
    coefficients = fit_ridge(rbf_kernel(points, points, 1.0), labels, 0.01)
    assert (predict(rbf_kernel(points, points, 1.0), coefficients) == labels).all()


def test_born_machine_learns_a_small_distribution():
    target = np.array([0.4, 0.1, 0.1, 0.4])
    fit = train_born_machine(NumpyRunner(0), target, 2, 2, seed=0, iterations=200)
    assert 0.5 * np.abs(fit["probabilities"] - target).sum() < 0.02
    assert fit["circuit_evaluations"] > 0


# -------------------------------------------------------------------- QAOA variants
def test_xy_mixer_circuit_never_leaves_the_k_hot_subspace():
    rng = np.random.default_rng(2)
    field = list(rng.normal(size=6))
    couplings = [(a, b, float(rng.normal())) for a, b in itertools.combinations(range(6), 2)]
    probabilities = (
        np.abs(simulate(xy_qaoa_circuit(field, couplings, 2, [0.3, 0.7, 0.4, 0.2]))) ** 2
    )
    weight_two = [bin(i).count("1") == 2 for i in range(64)]
    assert probabilities[weight_two].sum() == pytest.approx(1.0, abs=1e-12)
    assert probabilities.max() < 0.99  # the mixer actually moved amplitude


def test_qaoa_solver_rejects_unknown_mixers_and_misshapen_warm_starts():
    from test_optimizer import request as optimizer_request
    from test_qaoa_cudaq import SETTINGS

    from qflab.methods.qaoa_solver import solve_qaoa
    from qflab.optimizer_worker import solve_request

    reference = solve_request(optimizer_request(), "a" * 64)
    with pytest.raises(ValueError, match="mixer"):
        solve_qaoa(reference, SETTINGS, None, 1, mixer="ring")
    with pytest.raises(ValueError, match="initial_parameters"):
        solve_qaoa(reference, SETTINGS, None, 1, initial_parameters=[0.1])


def test_power_selection_finds_the_leading_factor_coordinates():
    from qflab.methods.subspace import power_selection

    covariance = clustered_covariance(16, 4, 2.0, 1)
    reference = spectrum_reference(covariance)
    chosen = power_selection(covariance, 4, 8)
    top = set(np.argsort(-np.abs(reference["vector"]))[:4].tolist())
    assert set(chosen) == top
