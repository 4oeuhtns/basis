"""The shared circuit layer: constructions against dense linear algebra, and the CUDA-Q runner
against the independent NumPy simulator."""

import math

import pytest

np = pytest.importorskip("numpy")
linalg = pytest.importorskip("scipy.linalg")

from qflab.circuit import (  # noqa: E402
    OPS,
    Circuit,
    NumpyRunner,
    bitstring,
    dicke_state,
    load_amplitudes,
    pauli_decompose,
    pauli_matrix,
    pauli_rotation,
    qft,
    simulate,
    trotter_evolution,
)


def basis_circuit(qubits, index, unitary=None):
    circuit = Circuit(qubits, unitary)
    for q in range(qubits):
        if index >> q & 1:
            circuit.x(q)
    return circuit


def random_unitary(width, rng):
    size = 2**width
    m = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return linalg.expm(1j * (m + m.conj().T))


def random_circuit(qubits, width, seed, length=40):
    """Every gate kind, with random operands; ``width`` > 0 adds a dense unitary."""
    rng = np.random.default_rng(seed)
    circuit = Circuit(qubits, random_unitary(width, rng) if width else None)
    for q in range(qubits):
        circuit.ry(float(rng.normal()), q)
    names = [k for k in OPS if width or k not in ("u", "cu", "v", "cv")]
    for _ in range(length):
        name = str(rng.choice(names))
        a, b, c = (int(v) for v in rng.permutation(qubits)[:3])
        angle = float(rng.normal())
        if name in ("u", "v"):
            circuit.add(name, int(rng.integers(0, qubits - width + 1)))
        elif name in ("cu", "cv"):
            start = int(rng.integers(0, qubits - width))
            outside = [q for q in range(qubits) if not start <= q < start + width]
            circuit.add(name, start, -1, int(rng.choice(outside)))
        elif name == "mcz":
            first = int(rng.integers(0, qubits - 2))
            stop = int(rng.integers(first + 1, qubits - 1))
            circuit.mcz(first, stop, int(rng.integers(stop, qubits)))
        elif name in ("ccry", "ccr1"):
            circuit.add(name, a, b, c, angle)
        elif name in ("cx", "cz", "cry", "crz", "cr1"):
            circuit.add(name, a, b, angle=angle)
        else:
            circuit.add(name, a, angle=angle)
    return circuit


# ------------------------------------------------------------------- SDK-free constructions
def test_qft_maps_basis_states_to_fourier_phases_and_inverts():
    for m in (1, 2, 3):
        size = 2**m
        for k in range(size):
            circuit = qft(basis_circuit(m, k), list(range(m)))
            expected = np.exp(2j * np.pi * k * np.arange(size) / size) / math.sqrt(size)
            assert np.allclose(simulate(circuit), expected)
            qft(circuit, list(range(m)), inverse=True)
            assert abs(simulate(circuit)[k]) == pytest.approx(1.0)


@pytest.mark.parametrize("word", ["X", "ZY", "XYZ", "IZI", "YIX"])
def test_pauli_rotation_is_the_matrix_exponential_with_and_without_control(word):
    n = len(word)
    rng = np.random.default_rng(n)
    prepare = Circuit(n + 1)
    for q in range(n + 1):
        prepare.ry(float(rng.normal()), q).rz(float(rng.normal()), q)
    prepare.cx(0, n)
    start = simulate(prepare)
    plain = Circuit(n + 1).extend(prepare)
    pauli_rotation(plain, 0.37, list(range(n)), word)
    full = np.kron(np.eye(2), pauli_matrix(word))
    assert np.allclose(simulate(plain), linalg.expm(1j * 0.37 * full) @ start)
    controlled = Circuit(n + 1).extend(prepare)
    pauli_rotation(controlled, 0.37, list(range(n)), word, control=n)
    block = linalg.block_diag(np.eye(2**n), linalg.expm(1j * 0.37 * pauli_matrix(word)))
    assert np.allclose(simulate(controlled), block @ start)


def test_controlled_identity_word_is_a_phase_on_the_control():
    circuit = Circuit(2).h(1)
    pauli_rotation(circuit, 0.5, [0], "I", control=1)
    assert np.allclose(simulate(circuit), [1 / math.sqrt(2), 0, np.exp(0.5j) / math.sqrt(2), 0])


def test_trotter_is_second_order_and_symmetric_for_a_real_symmetric_hamiltonian():
    rng = np.random.default_rng(3)
    h = rng.normal(size=(8, 8))
    h = h + h.T
    terms = pauli_decompose(h)
    assert np.allclose(sum(c * pauli_matrix(w) for w, c in terms), h)
    # Uncontrolled rotations drop the identity term's global phase; compare without it.
    exact = linalg.expm(1j * 0.3 * (h - np.trace(h) / 8 * np.eye(8)))
    errors = []
    for steps in (1, 2, 4):
        columns = [
            simulate(trotter_evolution(basis_circuit(3, k), terms, 0.3, steps, [0, 1, 2]))
            for k in range(8)
        ]
        product = np.array(columns).T
        assert np.allclose(product, product.T, atol=1e-10)  # palindromic steps keep U^T = U
        errors.append(float(np.abs(product - exact).max()))
    assert errors[2] < errors[1] < errors[0]
    assert errors[1] / errors[2] > 3  # about 4x per doubling of the step count


@pytest.mark.parametrize(("n", "k"), [(3, 1), (4, 2), (5, 3), (6, 1), (8, 3)])
def test_dicke_state_is_the_uniform_superposition_of_k_hot_strings(n, k):
    circuit = dicke_state(Circuit(n), list(range(n)), k)
    state = simulate(circuit)
    expected = np.array([float(bin(i).count("1") == k) for i in range(2**n)])
    assert np.allclose(state, expected / np.linalg.norm(expected))
    assert len(circuit.ops) < 10 * n * k  # O(n k) gates, not a dense load


def test_load_amplitudes_prepares_the_normalized_vector_including_zeros():
    rng = np.random.default_rng(4)
    amplitudes = rng.uniform(size=8)
    amplitudes[[1, 6]] = 0.0
    state = simulate(load_amplitudes(Circuit(3), [0, 1, 2], amplitudes))
    assert np.allclose(state, amplitudes / np.linalg.norm(amplitudes))


@pytest.mark.parametrize("width", [0, 2])
def test_inverse_undoes_every_gate_kind(width):
    circuit = random_circuit(5, width, seed=width)
    circuit.extend(circuit.inverse())
    assert abs(simulate(circuit)[0]) == pytest.approx(1.0)


def test_numpy_runner_samples_the_born_rule():
    circuit = Circuit(2).ry(1.1, 0).cx(0, 1)
    counts = NumpyRunner(1).sample(circuit, 40_000)
    assert set(counts) <= {"00", "11"}
    assert counts["11"] / 40_000 == pytest.approx(math.sin(0.55) ** 2, abs=0.01)
    assert bitstring(1, 3) == "100"  # qubit 0 first


# ------------------------------------------------------------------------ CUDA-Q execution
def cudaq_runner(seed, **options):
    programs = pytest.importorskip("qflab.backends.cudaq_programs", exc_type=ImportError)
    return programs.ProgramRunner(seed, **options)


def test_cudaq_runner_matches_the_reference_simulator_for_every_gate_and_unitary_width():
    runner = cudaq_runner(1)
    circuits = [random_circuit(6, width, seed=10 + width) for width in (0, 1, 2, 3, 4, 5)]
    # Compile every kernel first, then run in the opposite order: a custom operation's matrix
    # is bound at compile time, so a stale binding would show up here.
    for circuit in circuits:
        runner.resources(circuit)
    for circuit in reversed(circuits):
        assert np.allclose(runner.state(circuit), simulate(circuit), atol=1e-9)


def test_statevector_sampling_matches_cudaq_native_sampling():
    circuit = random_circuit(5, 2, seed=3)
    exact = np.abs(simulate(circuit)) ** 2
    for native in (False, True):
        counts = cudaq_runner(2, native_sampling=native).sample(circuit, 20_000)
        empirical = np.array([counts.get(bitstring(i, 5), 0) / 20_000 for i in range(32)])
        assert 0.5 * np.abs(empirical - exact).sum() < 0.04, native


def test_cached_distribution_follows_a_circuit_that_was_extended():
    runner = cudaq_runner(3)
    circuit = Circuit(2)
    assert runner.sample(circuit, 100) == {"00": 100}
    circuit.x(1)
    assert runner.sample(circuit, 100) == {"01": 100}


def test_cudaq_resources_report_depth_and_construction_counts():
    circuit = Circuit(3).h(0).cx(0, 1).cx(1, 2)
    found = cudaq_runner(4).resources(circuit)
    assert found["gates"] == 3 and found["multi_qubit_gates"] == 2 and found["depth"] == 3
    assert found["gate_counts"] == {"cx": 2, "h": 1}
