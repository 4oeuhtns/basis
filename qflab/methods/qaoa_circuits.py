"""Constraint-preserving QAOA circuit: Dicke initial state and a ring XY mixer.

The penalty version of QAOA starts in the uniform superposition of all bitstrings and uses a
penalty to discourage the wrong number of selected funds. This variant never leaves the
feasible subspace: it starts in the equal superposition of all bitstrings with exactly K ones
and mixes with ``XX + YY`` terms, which only move a one from a qubit to its neighbour
(Hadfield et al., "From the quantum approximate optimization algorithm to a quantum
alternating operator ansatz", Algorithms 12, 34 (2019)).

The ring mixer is applied as even pairs then odd pairs. That is a Trotter split of the ring
Hamiltonian, and every factor conserves the number of ones exactly.
"""

from __future__ import annotations

from qflab.circuit import Circuit, dicke_state, pauli_rotation


def xy_qaoa_circuit(
    field: list[float],
    couplings: list[tuple[int, int, float]],
    select_count: int,
    parameters: list[float],
) -> Circuit:
    n = len(field)
    layers = len(parameters) // 2
    if len(parameters) != 2 * layers or layers < 1:
        raise ValueError("Expected one gamma and one beta per layer")
    circuit = Circuit(n)
    dicke_state(circuit, list(range(n)), select_count)
    ring = [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, (i + 1) % n) for i in range(1, n, 2)]
    for layer in range(layers):
        gamma, beta = parameters[layer], parameters[layers + layer]
        for a, b, strength in couplings:
            circuit.cx(a, b).rz(2 * gamma * strength, b).cx(a, b)
        for i in range(n):
            circuit.rz(2 * gamma * field[i], i)
        for a, b in ring:
            pauli_rotation(circuit, -beta / 2, [a, b], "XX")
            pauli_rotation(circuit, -beta / 2, [a, b], "YY")
    return circuit
