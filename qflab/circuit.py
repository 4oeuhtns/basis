"""A small, SDK-free circuit description shared by the research-track algorithms.

Algorithms build a ``Circuit`` (a list of gates); a *runner* executes it. The production
runner is ``qflab.backends.cudaq_programs.ProgramRunner`` (CUDA-Q). ``NumpyRunner`` here is an
independent reference simulator used by the tests to check the CUDA-Q execution, and to test
algorithms where no quantum SDK is installed.

Conventions (identical to the CUDA-Q backend):

* statevector index bit ``i`` is qubit ``i`` (qubit 0 is the least significant bit);
* sampled bitstrings list qubit 0 first;
* ``rz(t) = exp(-i t Z / 2)``, likewise ``rx`` and ``ry``; ``r1(t) = diag(1, exp(i t))``.

A circuit may carry one dense unitary ``U`` (and uses its inverse as ``v``) acting on a
contiguous block of qubits, for steps that are synthesized classically (exact Hamiltonian
evolution, a dense block encoding). Its use is always disclosed as classical preprocessing.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

# name -> (code, self-inverse, has angle)
OPS = {
    "h": (0, True, False),
    "x": (1, True, False),
    "z": (2, True, False),
    "rx": (3, False, True),
    "ry": (4, False, True),
    "rz": (5, False, True),
    "r1": (6, False, True),
    "cx": (7, True, False),
    "cz": (8, True, False),
    "cry": (9, False, True),
    "crz": (10, False, True),
    "cr1": (11, False, True),
    "ccry": (12, False, True),
    "ccr1": (13, False, True),
    "mcz": (14, True, False),
    "u": (15, False, False),
    "cu": (16, False, False),
    "v": (17, False, False),
    "cv": (18, False, False),
}
_SWAP = {"u": "v", "v": "u", "cu": "cv", "cv": "cu"}
MAX_UNITARY_QUBITS = 5


class Circuit:
    """Gate list. Operand roles:

    * one-qubit gates: ``a`` is the target;
    * ``cx cz cry crz cr1``: ``a`` controls ``b``;
    * ``ccry ccr1``: ``a`` and ``b`` control ``c``;
    * ``mcz``: qubits ``a .. b-1`` control a Z on ``c``;
    * ``u v``: the dense unitary (or its inverse) on qubits ``a .. a+m-1``; ``cu cv`` add the
      control qubit ``c``.
    """

    def __init__(self, qubits: int, unitary: np.ndarray | None = None) -> None:
        self.qubits = int(qubits)
        self.ops: list[tuple[str, int, int, int, float]] = []
        self.unitary = None
        if unitary is not None:
            self.set_unitary(unitary)

    def set_unitary(self, unitary: np.ndarray) -> None:
        unitary = np.asarray(unitary, dtype=complex)
        size = unitary.shape[0]
        width = size.bit_length() - 1
        if unitary.shape != (size, size) or 2**width != size or not 1 <= width:
            raise ValueError("The dense unitary must be 2^m x 2^m")
        if width > MAX_UNITARY_QUBITS:
            raise ValueError(f"Dense unitaries are limited to {MAX_UNITARY_QUBITS} qubits")
        if not np.allclose(unitary @ unitary.conj().T, np.eye(size), atol=1e-9):
            raise ValueError("Matrix is not unitary")
        self.unitary = unitary

    @property
    def unitary_qubits(self) -> int:
        return 0 if self.unitary is None else self.unitary.shape[0].bit_length() - 1

    def add(self, name: str, a: int, b: int = -1, c: int = -1, angle: float = 0.0) -> Circuit:
        if name not in OPS:
            raise ValueError(f"Unknown gate {name}")
        if name in _SWAP and self.unitary is None:
            raise ValueError("Set the dense unitary before using it")
        self.ops.append((name, int(a), int(b), int(c), float(angle)))
        return self

    # Short builders keep algorithm code readable.
    def h(self, q):
        return self.add("h", q)

    def x(self, q):
        return self.add("x", q)

    def z(self, q):
        return self.add("z", q)

    def rx(self, t, q):
        return self.add("rx", q, angle=t)

    def ry(self, t, q):
        return self.add("ry", q, angle=t)

    def rz(self, t, q):
        return self.add("rz", q, angle=t)

    def r1(self, t, q):
        return self.add("r1", q, angle=t)

    def cx(self, control, target):
        return self.add("cx", control, target)

    def cz(self, control, target):
        return self.add("cz", control, target)

    def cry(self, t, control, target):
        return self.add("cry", control, target, angle=t)

    def crz(self, t, control, target):
        return self.add("crz", control, target, angle=t)

    def cr1(self, t, control, target):
        return self.add("cr1", control, target, angle=t)

    def ccry(self, t, c1, c2, target):
        return self.add("ccry", c1, c2, target, angle=t)

    def ccr1(self, t, c1, c2, target):
        return self.add("ccr1", c1, c2, target, angle=t)

    def mcz(self, first, stop, target):
        return self.add("mcz", first, stop, target)

    def swap(self, a, b):
        return self.cx(a, b).cx(b, a).cx(a, b)

    def extend(self, other: Circuit) -> Circuit:
        if other.unitary is not None:
            if self.unitary is not None and other.unitary is not self.unitary:
                raise ValueError("A circuit carries at most one dense unitary")
            self.unitary = other.unitary
        self.ops.extend(other.ops)
        return self

    def inverse(self) -> Circuit:
        out = Circuit(self.qubits)
        out.unitary = self.unitary
        for name, a, b, c, angle in reversed(self.ops):
            out.ops.append((_SWAP.get(name, name), a, b, c, -angle if OPS[name][2] else 0.0))
        return out

    def counts(self) -> dict:
        """Logical gate counts straight from the construction."""
        tally = Counter(name for name, *_ in self.ops)
        multi = sum(v for k, v in tally.items() if k not in ("h", "x", "z", "rx", "ry", "rz", "r1"))
        return {
            "qubits": self.qubits,
            "gates": len(self.ops),
            "multi_qubit_gates": multi,
            "dense_unitary_calls": sum(tally[k] for k in _SWAP),
            "gate_counts": dict(sorted(tally.items())),
        }

    def arrays(self) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
        """Parallel lists in the form the CUDA-Q interpreter kernel takes."""
        width = self.unitary_qubits
        codes = []
        for name, *_ in self.ops:
            code = OPS[name][0]
            # Dense-unitary codes are spread out by arity: 20 + 4 * (m - 1) + {u, cu, v, cv}.
            codes.append(20 + 4 * (width - 1) + (code - 15) if code >= 15 else code)
        return (
            codes,
            [op[1] for op in self.ops],
            [op[2] for op in self.ops],
            [op[3] for op in self.ops],
            [op[4] for op in self.ops],
        )


# ------------------------------------------------------------------ reusable constructions
def qft(circuit: Circuit, qubits: list[int], inverse: bool = False) -> Circuit:
    """Quantum Fourier transform |k> -> sum_y exp(2 pi i k y / M)|y> / sqrt(M).

    ``qubits[j]`` holds bit ``j`` of the integer on both sides (swaps included).
    """
    m = len(qubits)
    block = Circuit(circuit.qubits)
    for j in range(m - 1, -1, -1):
        block.h(qubits[j])
        for i in range(j - 1, -1, -1):
            block.cr1(math.pi / 2 ** (j - i), qubits[i], qubits[j])
    for j in range(m // 2):
        block.swap(qubits[j], qubits[m - 1 - j])
    return circuit.extend(block.inverse() if inverse else block)


def pauli_rotation(
    circuit: Circuit, theta: float, qubits: list[int], word: str, control: int | None = None
) -> Circuit:
    """``exp(i theta P)`` for the Pauli string ``word`` (character ``j`` acts on ``qubits[j]``).

    With ``control`` the rotation happens only when that qubit is 1; an all-identity word then
    becomes a phase on the control.
    """
    active = [(q, p) for q, p in zip(qubits, word, strict=True) if p != "I"]
    if not active:
        if control is not None:
            circuit.r1(theta, control)
        return circuit
    for q, p in active:
        if p == "X":
            circuit.h(q)
        elif p == "Y":
            circuit.r1(-math.pi / 2, q).h(q)
    last = active[-1][0]
    for q, _ in active[:-1]:
        circuit.cx(q, last)
    if control is None:
        circuit.rz(-2 * theta, last)
    else:
        circuit.crz(-2 * theta, control, last)
    for q, _ in reversed(active[:-1]):
        circuit.cx(q, last)
    for q, p in active:
        if p == "X":
            circuit.h(q)
        elif p == "Y":
            circuit.h(q).r1(math.pi / 2, q)
    return circuit


_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0, -1.0]),
}


def pauli_matrix(word: str) -> np.ndarray:
    """Dense matrix of a Pauli string; character ``j`` acts on qubit ``j`` (index bit ``j``)."""
    out = np.eye(1)
    for p in word:
        out = np.kron(_PAULI[p], out)
    return out


def pauli_decompose(matrix: np.ndarray, tolerance: float = 1e-12) -> list[tuple[str, float]]:
    """Real coefficients ``c_P`` with ``matrix = sum c_P P`` for a Hermitian matrix."""
    size = matrix.shape[0]
    n = size.bit_length() - 1
    if matrix.shape != (size, size) or 2**n != size:
        raise ValueError("Matrix must be 2^n x 2^n")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
        raise ValueError("Matrix must be Hermitian")
    terms = []
    for index in range(4**n):
        word = "".join("IXYZ"[index >> (2 * j) & 3] for j in range(n))
        coefficient = float(np.real(np.trace(pauli_matrix(word) @ matrix))) / size
        if abs(coefficient) > tolerance:
            terms.append((word, coefficient))
    return terms


def trotter_evolution(
    circuit: Circuit,
    terms: list[tuple[str, float]],
    time: float,
    steps: int,
    qubits: list[int],
    control: int | None = None,
) -> Circuit:
    """Second-order (symmetric) Trotter approximation of ``exp(i time * sum c_P P)``.

    Each step is the palindrome ``e^{iA/2} e^{iB/2} ... e^{iB/2} e^{iA/2}``, so the product stays
    a symmetric matrix when the Hamiltonian is real symmetric.
    """
    if steps < 1:
        raise ValueError("Trotter needs at least one step")
    slice_time = time / steps
    for _ in range(steps):
        for word, coefficient in terms:
            pauli_rotation(circuit, coefficient * slice_time / 2, qubits, word, control)
        for word, coefficient in reversed(terms):
            pauli_rotation(circuit, coefficient * slice_time / 2, qubits, word, control)
    return circuit


def dicke_state(circuit: Circuit, qubits: list[int], k: int) -> Circuit:
    """Equal superposition of all bitstrings with exactly ``k`` ones (Baertschi-Eidenbenz).

    Starts from |0...0>. Uses O(n k) gates and no ancilla qubits, in contrast to the 2^n
    rotations of a dense amplitude load.
    """
    n = len(qubits)
    if not 0 < k < n:
        raise ValueError("Need 0 < k < n")
    for q in qubits[n - k :]:
        circuit.x(q)

    def split(last: int, weight: int) -> None:
        # Acts on positions (1-based) up to ``last``; moves weight between neighbours.
        top = qubits[last - 1]
        circuit.cx(qubits[last - 2], top)
        circuit.cry(2 * math.acos(math.sqrt(1 / last)), top, qubits[last - 2])
        circuit.cx(qubits[last - 2], top)
        for i in range(2, weight + 1):
            low = qubits[last - i - 1]
            circuit.cx(low, top)
            circuit.ccry(2 * math.acos(math.sqrt(i / last)), top, qubits[last - i], low)
            circuit.cx(low, top)

    for last in range(n, k, -1):
        split(last, k)
    for last in range(k, 1, -1):
        split(last, last - 1)
    return circuit


def uniformly_controlled_ry(
    alphas: np.ndarray, controls: list[int], target: int
) -> list[tuple[float, int, int]]:
    """Rotate ``target`` by ``alphas[b]`` when the control qubits hold the bit pattern ``b``.

    Returns a gate list of ``(angle, target, cnot_control)``: an RY, then a CNOT from
    ``cnot_control`` (or no CNOT when it is -1). This is the standard Gray-code construction:
    2^m rotations and 2^m CNOTs for m controls, with no ancilla qubits.
    """
    m = len(controls)
    if len(alphas) != 2**m:
        raise ValueError("Need one angle per control pattern")
    if m == 0:
        return [(float(alphas[0]), target, -1)]
    gates = []
    patterns = np.arange(2**m)
    for step in range(2**m):
        gray = step ^ (step >> 1)
        # bitwise_count returns uint8; convert before the subtraction or it wraps to 255.
        signs = 1.0 - 2.0 * (np.bitwise_count(patterns & gray) & 1).astype(float)
        angle = float(signs @ alphas) / 2**m
        # The CNOT control is the Gray-code bit that flips before the next rotation.
        bit = m - 1 if step == 2**m - 1 else ((step + 1) & -(step + 1)).bit_length() - 1
        gates.append((angle, target, controls[bit]))
    return gates


def load_amplitudes(circuit: Circuit, qubits: list[int], amplitudes: np.ndarray) -> Circuit:
    """Prepare ``sum_j amplitudes[j] |j>`` from |0...0> for non-negative real amplitudes.

    Dense loading: 2^n - 1 rotations for n qubits. ``qubits[j]`` holds bit ``j`` of the index.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    n = len(qubits)
    if len(amplitudes) != 2**n or (amplitudes < 0).any():
        raise ValueError("Need 2^n non-negative amplitudes")
    probabilities = amplitudes**2 / float(amplitudes @ amplitudes)
    for level in range(n):
        marginal = probabilities.reshape(-1, 2 ** (level + 1)).sum(axis=0)
        zero, one = marginal[: 2**level], marginal[2**level :]
        alphas = 2 * np.arctan2(np.sqrt(one), np.sqrt(zero))
        for angle, target, control in uniformly_controlled_ry(
            alphas, qubits[:level], qubits[level]
        ):
            circuit.ry(angle, target)
            if control >= 0:
                circuit.cx(control, target)
    return circuit


# ---------------------------------------------------------------------- reference simulator
def _gate(name: str, angle: float) -> np.ndarray:
    half = angle / 2
    if name == "h":
        return np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    if name in ("x", "cx"):
        return _PAULI["X"]
    if name in ("z", "cz", "mcz"):
        return _PAULI["Z"]
    if name == "rx":
        return np.array(
            [[math.cos(half), -1j * math.sin(half)], [-1j * math.sin(half), math.cos(half)]]
        )
    if name in ("ry", "cry", "ccry"):
        return np.array([[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]])
    if name in ("rz", "crz"):
        return np.diag([np.exp(-1j * half), np.exp(1j * half)])
    if name in ("r1", "cr1", "ccr1"):
        return np.diag([1.0, np.exp(1j * angle)])
    raise ValueError(name)


def _apply(state: np.ndarray, n: int, matrix: np.ndarray, targets: list[int], controls=()) -> None:
    """In place: ``matrix`` (index bit j = targets[j]) on ``targets`` where all controls are 1."""
    psi = state.reshape([2] * n)
    index: list = [slice(None)] * n
    for control in controls:
        index[n - 1 - control] = 1
    sub = psi[tuple(index)]
    remaining = [q for q in range(n - 1, -1, -1) if q not in controls]
    m = len(targets)
    axes = [remaining.index(targets[j]) for j in range(m - 1, -1, -1)]
    moved = np.tensordot(matrix.reshape([2] * (2 * m)), sub, axes=(list(range(m, 2 * m)), axes))
    psi[tuple(index)] = np.moveaxis(moved, list(range(m)), axes)


def simulate(circuit: Circuit, initial: np.ndarray | None = None) -> np.ndarray:
    n = circuit.qubits
    if initial is None:
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0
    else:
        state = np.array(initial, dtype=complex)
    width = circuit.unitary_qubits
    for name, a, b, c, angle in circuit.ops:
        if name in ("u", "cu", "v", "cv"):
            matrix = circuit.unitary if name in ("u", "cu") else circuit.unitary.conj().T
            controls = (c,) if name in ("cu", "cv") else ()
            _apply(state, n, matrix, list(range(a, a + width)), controls)
        elif name == "mcz":
            _apply(state, n, _gate(name, angle), [c], tuple(range(a, b)))
        elif name in ("ccry", "ccr1"):
            _apply(state, n, _gate(name, angle), [c], (a, b))
        elif name in ("cx", "cz", "cry", "crz", "cr1"):
            _apply(state, n, _gate(name, angle), [b], (a,))
        else:
            _apply(state, n, _gate(name, angle), [a])
    return state


def bitstring(index: int, qubits: int) -> str:
    """Bitstring of a statevector index, qubit 0 first."""
    return "".join("1" if index >> q & 1 else "0" for q in range(qubits))


class NumpyRunner:
    """Reference runner with the same interface as the CUDA-Q ``ProgramRunner``."""

    name = "numpy reference simulator"

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def state(self, circuit: Circuit) -> np.ndarray:
        return simulate(circuit)

    def sample(self, circuit: Circuit, shots: int) -> dict[str, int]:
        probabilities = np.abs(simulate(circuit)) ** 2
        draws = self._rng.multinomial(shots, probabilities / probabilities.sum())
        return {bitstring(int(i), circuit.qubits): int(draws[i]) for i in np.flatnonzero(draws)}

    def resources(self, circuit: Circuit) -> dict:
        return circuit.counts()
