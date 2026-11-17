"""Quantum linear-system solvers (HHL and a QSVT-based solver), independent of any quantum SDK.

Both build a ``SolveProgram``: a circuit, the qubits holding the solution, and the measurement
outcomes that herald success. Given ``A`` (symmetric positive definite) and ``b``, a successful
run leaves the system register in a state proportional to ``A^-1 b``. It does **not** return
the entries of ``x``; ``recover_vector`` is the separate, costed measurement procedure that
turns the state into numbers.

What is classical preprocessing here, and disclosed in every result:

* HHL, ``evolution="exact"``: ``exp(i A t)`` is formed from a dense eigendecomposition and
  loaded as one unitary. ``evolution="trotter"`` replaces it with Pauli rotations.
* QSVT: the block encoding ``[[A/alpha, S], [S, -A/alpha]]`` is formed densely, and the phase
  factors come from a classical least-squares solve.
* Both are told an interval ``[low, high]`` containing the spectrum of ``A``.

Neither solver is given ``A^-1`` or the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from qflab.circuit import (
    Circuit,
    load_amplitudes,
    pauli_decompose,
    qft,
    trotter_evolution,
    uniformly_controlled_ry,
)
from qflab.methods.qsp import odd_chebyshev_fit, reciprocal_target, solve_phases


@dataclass
class SolveProgram:
    method: str
    circuit: Circuit
    system: list[int]
    success: dict[int, int]  # qubit -> value that heralds a solution
    norm_factor: float  # ideal success probability = (norm_factor * ||A^-1 b_hat||)^2
    info: dict = field(default_factory=dict)


def _prepare_rhs(circuit: Circuit, system: list[int], rhs: np.ndarray) -> str:
    rhs = np.asarray(rhs, dtype=float)
    if len(rhs) != 2 ** len(system):
        raise ValueError("Right-hand side length must be 2^n")
    if np.allclose(rhs, rhs[0]) and rhs[0] > 0:
        for q in system:
            circuit.h(q)
        return "uniform vector: one Hadamard per system qubit"
    if (rhs < 0).any():
        raise ValueError("Only non-negative right-hand sides are supported")
    load_amplitudes(circuit, system, rhs)
    return "dense amplitude load: 2^n - 1 rotations"


def _spectrum_check(matrix: np.ndarray, low: float, high: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    size = matrix.shape[0]
    if matrix.shape != (size, size) or size & (size - 1) or size < 2:
        raise ValueError("Matrix must be 2^n x 2^n")
    if not np.allclose(matrix, matrix.T, atol=1e-12 * max(1.0, np.abs(matrix).max())):
        raise ValueError("Matrix must be symmetric")
    if not 0 < low < high:
        raise ValueError("Need 0 < low < high")
    return matrix


def hhl_program(
    matrix: np.ndarray,
    rhs: np.ndarray,
    clock_qubits: int,
    bounds: tuple[float, float],
    evolution: str = "exact",
    trotter_steps: int = 1,
    evolution_time: float | None = None,
) -> SolveProgram:
    """HHL: phase estimation of ``exp(i A t)``, eigenvalue-controlled rotation, uncompute.

    ``bounds = (low, high)`` must contain the spectrum of ``A``. By default ``t`` maps ``high``
    to the largest clock value. Layout: system, then clock (least significant bit first), then
    the rotation ancilla.
    """
    low, high = bounds
    matrix = _spectrum_check(matrix, low, high)
    if not 1 <= clock_qubits <= 10 or evolution not in ("exact", "trotter"):
        raise ValueError("clock_qubits must be 1..10 and evolution 'exact' or 'trotter'")
    n = matrix.shape[0].bit_length() - 1
    system = list(range(n))
    clock = list(range(n, n + clock_qubits))
    ancilla = n + clock_qubits
    levels = 2**clock_qubits
    time = evolution_time or 2 * math.pi * (1 - 1 / levels) / high
    if high * time >= 2 * math.pi:
        raise ValueError("evolution_time wraps the largest eigenvalue past a full turn")

    forward = Circuit(ancilla + 1)
    for q in clock:
        forward.h(q)
    terms: list = []
    if evolution == "exact":
        values, vectors = np.linalg.eigh(matrix)
        forward.set_unitary((vectors * np.exp(1j * values * time)) @ vectors.T)
        for j, q in enumerate(clock):
            for _ in range(2**j):
                forward.add("cu", 0, -1, q)
    else:
        terms = pauli_decompose(matrix)
        for j, q in enumerate(clock):
            trotter_evolution(forward, terms, time * 2**j, trotter_steps * 2**j, system, control=q)
    qft(forward, clock, inverse=True)

    # Clock value k stands for the eigenvalue 2 pi k / (t * 2^c).
    unit = 2 * math.pi / (time * levels)
    smallest = max(1, math.floor(low / unit))
    constant = smallest * unit
    angles = np.zeros(levels)
    for k in range(1, levels):
        angles[k] = 2 * math.asin(min(1.0, constant / (k * unit)))

    circuit = Circuit(ancilla + 1)
    preparation = _prepare_rhs(circuit, system, rhs)
    circuit.extend(forward)
    for angle, target, control in uniformly_controlled_ry(angles, clock, ancilla):
        circuit.ry(angle, target)
        if control >= 0:
            circuit.cx(control, target)
    circuit.extend(forward.inverse())
    return SolveProgram(
        method="hhl",
        circuit=circuit,
        system=system,
        success={ancilla: 1, **dict.fromkeys(clock, 0)},
        norm_factor=constant,
        info={
            "clock_qubits": clock_qubits,
            "ancilla_qubits": clock_qubits + 1,
            "evolution": evolution,
            "evolution_time": time,
            "eigenvalue_resolution": unit,
            "rotation_constant": constant,
            "bounds": [low, high],
            "rhs_preparation": preparation,
            "trotter_steps_per_unit_power": trotter_steps if evolution == "trotter" else None,
            "pauli_terms": len(terms) if evolution == "trotter" else None,
            "classical_preprocessing": "dense eigendecomposition of A to form exp(iAt)"
            if evolution == "exact"
            else "Pauli decomposition of A (4^n traces)",
        },
    )


def qsvt_program(
    matrix: np.ndarray, rhs: np.ndarray, bounds: tuple[float, float], epsilon: float
) -> SolveProgram:
    """QSVT linear solver: an odd polynomial close to 1/x applied to a block encoding of A.

    ``bounds = (low, high)`` must contain the spectrum; ``high`` is the block-encoding
    normalization alpha and ``high / low`` the declared condition-number bound. ``epsilon`` is
    the relative accuracy of the reciprocal polynomial on the declared interval. Layout:
    system, then the block-encoding ancilla, then the qubit that selects the real part.
    """
    low, high = bounds
    matrix = _spectrum_check(matrix, low, high)
    n = matrix.shape[0].bit_length() - 1
    system = list(range(n))
    block, real = n, n + 1
    alpha, kappa = high, high / low
    function, scale = reciprocal_target(kappa, epsilon)
    coefficients = odd_chebyshev_fit(function, epsilon * scale / 2)
    solved = solve_phases(coefficients)
    phases, degree = solved["phases"], solved["degree"]

    values, vectors = np.linalg.eigh(matrix / alpha)
    values = np.clip(values, -1.0, 1.0)
    scaled = (vectors * values) @ vectors.T
    complement = (vectors * np.sqrt(1 - values**2)) @ vectors.T
    encoding = np.block([[scaled, complement], [complement, -scaled]])

    circuit = Circuit(n + 2, encoding)
    preparation = _prepare_rhs(circuit, system, rhs)
    circuit.h(real)

    def phase(angle: float) -> None:
        # e^{i angle Z} on the block ancilla, with the sign flipped when ``real`` is 1. The two
        # branches implement P and its complex conjugate; their sum is the real polynomial.
        circuit.cx(real, block).rz(-2 * angle, block).cx(real, block)

    phase(phases[degree])
    for j in range(degree - 1, -1, -1):
        circuit.add("u", 0)
        phase(phases[j])
    circuit.h(real)
    return SolveProgram(
        method="qsvt",
        circuit=circuit,
        system=system,
        success={block: 0, real: 0},
        norm_factor=scale * alpha,
        info={
            "ancilla_qubits": 2,
            "polynomial_degree": degree,
            "block_encoding_calls": degree,
            "block_encoding_normalization": alpha,
            "condition_number_bound": kappa,
            "polynomial_epsilon": epsilon,
            "polynomial_scale": scale,
            "phase_residual": solved["residual"],
            "phase_solve_seconds": solved["solve_seconds"],
            "bounds": [low, high],
            "rhs_preparation": preparation,
            "block_encoding_error": float(
                np.max(np.abs(encoding[: 2**n, : 2**n] - matrix / alpha))
            ),
            "classical_preprocessing": "dense block encoding from an eigendecomposition of A; "
            "Chebyshev fit of the reciprocal; least-squares phase solve",
        },
    )


# ------------------------------------------------------------------------------ analysis
def _success_index(program: SolveProgram) -> int:
    return sum(value << qubit for qubit, value in program.success.items())


def solution_state(program: SolveProgram, runner) -> tuple[np.ndarray, float]:
    """Heralded solution state and success probability from the full statevector.

    Privileged simulator access, for verification only; a device cannot provide it.
    """
    state = runner.state(program.circuit)
    size = 2 ** len(program.system)
    base = _success_index(program)
    vector = np.array([state[base + j] for j in range(size)])
    probability = float(np.vdot(vector, vector).real)
    if probability <= 0:
        raise ValueError("The success branch has zero probability")
    return vector / math.sqrt(probability), probability


def sample_solution(
    program: SolveProgram, runner, hadamards: list[int], shots: int
) -> tuple[np.ndarray, int]:
    """Counts over the system register among heralded shots, after optional Hadamards."""
    circuit = Circuit(program.circuit.qubits)
    circuit.extend(program.circuit)
    for q in hadamards:
        circuit.h(q)
    size = 2 ** len(program.system)
    counts = np.zeros(size, dtype=int)
    for bits, count in runner.sample(circuit, shots).items():
        if all(int(bits[q]) == value for q, value in program.success.items()):
            counts[sum(int(bits[q]) << j for j, q in enumerate(program.system))] += count
    return counts, int(counts.sum())


def recover_vector(program: SolveProgram, runner, shots: int) -> dict:
    """Estimate the real solution vector, up to one overall sign, from measurements alone.

    ``n + 1`` measurement settings, ``shots`` preparations each (failed heralds included):

    1. computational basis: the squared magnitudes;
    2. for each system qubit, a Hadamard on that qubit first: the outcome difference of a pair
       of indices that differ in that bit is ``2 x_i x_j``, which fixes their relative sign.

    Signs are propagated from the largest entry along the most reliable pairs.
    """
    n = len(program.system)
    size = 2**n
    counts, heralded = sample_solution(program, runner, [], shots)
    successes = [heralded]
    if heralded == 0:
        return {"status": "no_success", "shots": shots, "successes": 0, "settings": 1}
    magnitude = np.sqrt(counts / heralded)
    product = np.zeros((size, size))
    for j, q in enumerate(program.system):
        counts, heralded = sample_solution(program, runner, [q], shots)
        successes.append(heralded)
        if heralded == 0:
            continue
        for index in range(size):
            if not index >> j & 1:
                pair = index | 1 << j
                product[index, pair] = product[pair, index] = (
                    (counts[index] - counts[pair]) / heralded / 2
                )
    # Prim's algorithm on |x_i x_j|: follow the strongest evidence first.
    sign = np.zeros(size)
    start = int(magnitude.argmax())
    sign[start] = 1.0
    reached = {start}
    while len(reached) < size:
        weight, source, target = max(
            (abs(product[i, i ^ (1 << j)]), i, i ^ (1 << j))
            for i in reached
            for j in range(n)
            if i ^ (1 << j) not in reached
        )
        sign[target] = sign[source] * (1.0 if product[source, target] >= 0 else -1.0)
        reached.add(target)
    return {
        "status": "ok",
        "vector": sign * magnitude,
        "settings": n + 1,
        "shots": shots * (n + 1),
        "successes": int(sum(successes)),
        "success_rate": float(sum(successes) / (shots * (n + 1))),
    }


def estimate_inverse_quadratic(program: SolveProgram, runner, shots: int) -> dict:
    """Estimate ``1^T A^-1 1`` for the uniform right-hand side without reading out ``x``.

    One setting: Hadamards on every system qubit. The heralding rate gives the solution norm
    and the all-zeros rate gives the overlap with the uniform vector.
    """
    size = 2 ** len(program.system)
    counts, heralded = sample_solution(program, runner, program.system, shots)
    if heralded == 0 or counts[0] == 0:
        return {"status": "no_success", "shots": shots, "successes": heralded}
    overlap, rate = counts[0] / heralded, heralded / shots
    value = size * math.sqrt(overlap * rate) / program.norm_factor
    return {
        "status": "ok",
        "value": float(value),
        "shots": shots,
        "successes": heralded,
        "settings": 1,
    }
