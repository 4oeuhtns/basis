"""SQOA-style portfolio selection through a quantum relaxation (plan section 13.1, R0).

A proposed adaptation, not a published portfolio method. It follows the structure of
Sampling-based Quantum Optimization with Quantum Relaxation (reported for MaxCut):

1. **Sparsify.** Keep only the largest pair terms of the financial objective. This changes the
   problem; the changed problem and its approximation error are reported separately.
2. **Relax.** Colour the interaction graph and pack up to three non-interacting variables on
   one qubit, each on its own Pauli axis (a quantum random access code). With ``k`` variables
   on a qubit, the spin ``z_i`` becomes ``sqrt(k) * P_i``. The relaxed Hamiltonian is no
   longer diagonal.
3. **Sample and diagonalize.** Measure time-evolved states in the computational basis, keep the
   most frequent configurations, and diagonalize the relaxed Hamiltonian in their span.
4. **Round.** Read each variable's magnetization ``<P_i>`` from the subspace ground state and
   select the K variables that lean most toward "selected". That top-K step is a classical
   repair and is reported as such.

Cardinality is not in the Hamiltonian: an all-pairs penalty would make every variable interact
with every other and remove the compression. A linear term (chemical potential) is tuned by
bisection so that the unrounded solution selects about K funds; every bisection step repeats
the quantum sampling and is counted.

With all pairs kept, every variable needs its own qubit, the encoding is the ordinary diagonal
one, and evolution under a diagonal Hamiltonian cannot change measurement probabilities: the
method then only returns its best sample.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from qflab.circuit import Circuit, pauli_matrix, trotter_evolution
from qflab.optimizer_worker import energy


def objective_terms(model: dict) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Linear and pair coefficients of the financial objective in objective units."""
    scale = int(model["scale"])
    symbols = model["symbols"]
    position = {s: i for i, s in enumerate(symbols)}
    linear = np.array([int(model["linear"][s]) / scale for s in symbols])
    pairs = {
        (position[p["left"]], position[p["right"]]): int(p["coefficient"]) / scale
        for p in model["pairs"]
        if int(p["coefficient"])
    }
    return linear, pairs


def sparsify(pairs: dict, density: float) -> dict:
    """Keep the ``density`` fraction of pair terms with the largest magnitude."""
    keep = max(0, min(len(pairs), round(density * len(pairs))))
    ranked = sorted(pairs.items(), key=lambda item: (-abs(item[1]), item[0]))
    return dict(ranked[:keep])


def evaluate(linear: np.ndarray, pairs: dict, bits) -> float:
    return float(
        sum(linear[i] for i in range(len(linear)) if bits[i])
        + sum(v for (a, b), v in pairs.items() if bits[a] and bits[b])
    )


def encode(n: int, pairs: dict) -> list[tuple[int, str]]:
    """Assign each variable a (qubit, Pauli axis); interacting variables never share a qubit."""
    neighbours = {i: set() for i in range(n)}
    for a, b in pairs:
        neighbours[a].add(b)
        neighbours[b].add(a)
    colour: dict[int, int] = {}
    for i in sorted(range(n), key=lambda v: (-len(neighbours[v]), v)):
        used = {colour[j] for j in neighbours[i] if j in colour}
        colour[i] = next(c for c in itertools.count() if c not in used)
    placement: dict[int, tuple[int, str]] = {}
    qubit = 0
    for c in sorted(set(colour.values())):
        members = [i for i in range(n) if colour[i] == c]
        for start in range(0, len(members), 3):
            for axis, i in zip("ZXY", members[start : start + 3], strict=False):
                placement[i] = (qubit, axis)
            qubit += 1
    return [placement[i] for i in range(n)]


def relaxed_hamiltonian(
    linear: np.ndarray, pairs: dict, potential: float, placement: list[tuple[int, str]]
) -> tuple[list[tuple[str, float]], float, list[float]]:
    """Pauli terms of the relaxed objective ``F_sparse(x) + potential * sum(x)``.

    Returns ``(terms, constant, factors)`` where ``factors[i]`` is ``sqrt(k)`` for the qubit
    of variable ``i``. Substitution: x = (1 - z) / 2, then z_i -> factors[i] * P_i.
    """
    n = len(linear)
    qubits = 1 + max(q for q, _ in placement)
    load = [sum(1 for q, _ in placement if q == target) for target in range(qubits)]
    factors = [math.sqrt(load[q]) for q, _ in placement]
    field = -(linear + potential) / 2
    constant = float((linear + potential).sum() / 2)
    terms: dict[str, float] = {}

    def word(entries: list[tuple[int, str]]) -> str:
        letters = ["I"] * qubits
        for q, axis in entries:
            letters[q] = axis
        return "".join(letters)

    for (a, b), value in pairs.items():
        constant += value / 4
        field[a] -= value / 4
        field[b] -= value / 4
        key = word([placement[a], placement[b]])
        terms[key] = terms.get(key, 0.0) + value / 4 * factors[a] * factors[b]
    for i in range(n):
        key = word([placement[i]])
        terms[key] = terms.get(key, 0.0) + field[i] * factors[i]
    return [(w, c) for w, c in terms.items() if c != 0.0], constant, factors


def dense(terms: list[tuple[str, float]], qubits: int) -> np.ndarray:
    matrix = np.zeros((2**qubits, 2**qubits), dtype=complex)
    for word, coefficient in terms:
        matrix += coefficient * pauli_matrix(word)
    return matrix


def encoded_state(bits, placement: list[tuple[int, str]]) -> np.ndarray:
    """The product state that encodes a bitstring: its Pauli expectations are z_i / sqrt(k)."""
    qubits = 1 + max(q for q, _ in placement)
    state = np.ones(1, dtype=complex)
    for target in range(qubits):
        members = [(axis, 1 - 2 * bits[i]) for i, (q, axis) in enumerate(placement) if q == target]
        bloch = sum(z * pauli_matrix(axis) for axis, z in members) / math.sqrt(len(members))
        values, vectors = np.linalg.eigh(bloch)
        state = np.kron(vectors[:, -1], state)  # +1 eigenvector; qubit 0 least significant
    return state


def verify_relaxation(linear, pairs, potential, placement, select_count: int) -> dict:
    """Check the relaxation against the objective it was derived from, for every bitstring."""
    n = len(linear)
    terms, constant, _ = relaxed_hamiltonian(linear, pairs, potential, placement)
    qubits = 1 + max(q for q, _ in placement)
    matrix = dense(terms, qubits)
    worst = 0.0
    best = math.inf
    for bits in itertools.product((0, 1), repeat=n):
        state = encoded_state(bits, placement)
        relaxed = float(np.real(np.vdot(state, matrix @ state))) + constant
        original = evaluate(linear, pairs, bits) + potential * sum(bits)
        worst = max(worst, abs(relaxed - original))
        best = min(best, original)
    ground = float(np.linalg.eigvalsh(matrix)[0]) + constant
    return {
        "bitstrings_checked": 2**n,
        "largest_mismatch": worst,
        "encoded_states_reproduce_objective": worst < 1e-9,
        "relaxed_ground_energy": ground,
        "best_binary_value": best,
        "relaxation_gap": best - ground,
        "is_lower_bound": ground <= best + 1e-9,
        "select_count": select_count,
    }


def magnetizations(vector: np.ndarray, placement: list[tuple[int, str]]) -> np.ndarray:
    qubits = 1 + max(q for q, _ in placement)
    out = np.zeros(len(placement))
    for i, (q, axis) in enumerate(placement):
        word = "".join(axis if j == q else "I" for j in range(qubits))
        out[i] = float(np.real(np.vdot(vector, pauli_matrix(word) @ vector)))
    return out


def solve_relaxation(
    model: dict,
    runner,
    density: float,
    shots: int,
    subspace_size: int,
    times: list[float],
    trotter_steps: int = 2,
    bisection_steps: int = 8,
) -> dict:
    """Run the four steps and score the recovered portfolio on the original objective."""
    symbols, k = model["symbols"], model["select_count"]
    n = len(symbols)
    linear, all_pairs = objective_terms(model)
    pairs = sparsify(all_pairs, density)
    placement = encode(n, pairs)
    qubits = 1 + max(q for q, _ in placement)
    total_shots = executions = gates = 0

    def attempt(potential: float) -> dict:
        nonlocal total_shots, executions, gates
        terms, _, _ = relaxed_hamiltonian(linear, pairs, potential, placement)
        scale = max(1e-12, math.sqrt(sum(c * c for _, c in terms)))
        counts: dict[int, int] = {}
        for time in times:
            circuit = Circuit(qubits)
            for q in range(qubits):
                circuit.h(q)
            if time:
                steps = max(1, round(trotter_steps * time))
                trotter_evolution(circuit, terms, -time / scale, steps, list(range(qubits)))
            for bits, count in runner.sample(circuit, shots).items():
                index = sum(int(b) << i for i, b in enumerate(bits))
                counts[index] = counts.get(index, 0) + count
            total_shots += shots
            executions += 1
            gates = max(gates, len(circuit.ops))
        chosen = sorted(
            index
            for index, _ in sorted(counts.items(), key=lambda t: (-t[1], t[0]))[:subspace_size]
        )
        matrix = dense(terms, qubits)[np.ix_(chosen, chosen)]
        values, vectors = np.linalg.eigh(matrix)
        ground = np.zeros(2**qubits, dtype=complex)
        ground[chosen] = vectors[:, 0]
        lean = magnetizations(ground, placement)  # negative leans toward "selected"
        return {
            "potential": potential,
            "subspace": len(chosen),
            "energy": float(values[0]),
            "magnetization": lean,
            "selected_before_repair": int((lean < 0).sum()),
        }

    # Bisection on the chemical potential: larger potential selects fewer funds.
    span = float(np.abs(linear).sum() + sum(abs(v) for v in pairs.values())) or 1.0
    low, high = -span, span
    best = attempt(0.0)
    for _ in range(bisection_steps):
        if best["selected_before_repair"] == k:
            break
        if best["selected_before_repair"] > k:
            low = best["potential"]
        else:
            high = best["potential"]
        trial = attempt((low + high) / 2)
        if abs(trial["selected_before_repair"] - k) <= abs(best["selected_before_repair"] - k):
            best = trial
    order = np.argsort(best["magnetization"], kind="stable")
    chosen_bits = [0] * n
    for i in order[:k]:
        chosen_bits[int(i)] = 1
    selected = [s for s, b in zip(symbols, chosen_bits, strict=True) if b]
    return {
        "selected": selected,
        "objective_integer": str(energy(model, selected)),
        "sparse_objective": evaluate(linear, pairs, chosen_bits),
        "selected_before_repair": best["selected_before_repair"],
        "needed_repair": best["selected_before_repair"] != k,
        "potential": best["potential"],
        "qubits": qubits,
        "pairs_kept": len(pairs),
        "pairs_total": len(all_pairs),
        "subspace": best["subspace"],
        "total_shots": total_shots,
        "circuit_executions": executions,
        "largest_circuit_gates": gates,
        "placement": [[q, axis] for q, axis in placement],
        "diagonal_encoding": all(axis == "Z" for _, axis in placement),
    }
