"""Run ``qflab.circuit.Circuit`` programs on CUDA-Q.

One interpreter kernel executes any gate list, so the research-track algorithms (HHL, QSVT,
Grover adaptive search, sampling methods, kernels, generative models) all run on the same
simulator targets as the core IQAE and QAOA circuits.

A circuit's dense unitary is registered as a CUDA-Q custom operation. CUDA-Q binds a custom
operation's matrix when a kernel is compiled, so kernels are built per unitary by a factory
and compiled immediately.
"""

from __future__ import annotations

import hashlib
import warnings
import weakref

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cudaq

from qflab.backends.cudaq_backend import _compiled, select_target
from qflab.circuit import MAX_UNITARY_QUBITS, Circuit, bitstring


def _register(width: int, unitary: np.ndarray | None) -> None:
    for m in range(1, MAX_UNITARY_QUBITS + 1):
        matrix = unitary if m == width else np.eye(2**m, dtype=complex)
        cudaq.register_operation(f"qfl_u{m}", matrix)
        cudaq.register_operation(f"qfl_v{m}", matrix.conj().T)


def _build(unitary: np.ndarray | None):
    """Interpreter kernels bound to one dense unitary (or to none)."""
    width = 0 if unitary is None else unitary.shape[0].bit_length() - 1
    _register(width, unitary)

    @cudaq.kernel
    def run(
        q: cudaq.qview,
        codes: list[int],
        a: list[int],
        b: list[int],
        c: list[int],
        t: list[float],
    ):
        for i in range(len(codes)):
            code = codes[i]
            if code == 0:
                h(q[a[i]])
            elif code == 1:
                x(q[a[i]])
            elif code == 2:
                z(q[a[i]])
            elif code == 3:
                rx(t[i], q[a[i]])
            elif code == 4:
                ry(t[i], q[a[i]])
            elif code == 5:
                rz(t[i], q[a[i]])
            elif code == 6:
                r1(t[i], q[a[i]])
            elif code == 7:
                x.ctrl(q[a[i]], q[b[i]])
            elif code == 8:
                z.ctrl(q[a[i]], q[b[i]])
            elif code == 9:
                ry.ctrl(t[i], q[a[i]], q[b[i]])
            elif code == 10:
                rz.ctrl(t[i], q[a[i]], q[b[i]])
            elif code == 11:
                r1.ctrl(t[i], q[a[i]], q[b[i]])
            elif code == 12:
                ry.ctrl(t[i], q[a[i]], q[b[i]], q[c[i]])
            elif code == 13:
                r1.ctrl(t[i], q[a[i]], q[b[i]], q[c[i]])
            elif code == 14:
                z.ctrl(q[a[i] : b[i]], q[c[i]])
            # Dense unitary: the matrix's index bit j is qubit a + j, so qubits are passed
            # most-significant first.
            elif code == 20:
                qfl_u1(q[a[i]])
            elif code == 21:
                qfl_u1.ctrl(q[c[i]], q[a[i]])
            elif code == 22:
                qfl_v1(q[a[i]])
            elif code == 23:
                qfl_v1.ctrl(q[c[i]], q[a[i]])
            elif code == 24:
                qfl_u2(q[a[i] + 1], q[a[i]])
            elif code == 25:
                qfl_u2.ctrl(q[c[i]], q[a[i] + 1], q[a[i]])
            elif code == 26:
                qfl_v2(q[a[i] + 1], q[a[i]])
            elif code == 27:
                qfl_v2.ctrl(q[c[i]], q[a[i] + 1], q[a[i]])
            elif code == 28:
                qfl_u3(q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 29:
                qfl_u3.ctrl(q[c[i]], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 30:
                qfl_v3(q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 31:
                qfl_v3.ctrl(q[c[i]], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 32:
                qfl_u4(q[a[i] + 3], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 33:
                qfl_u4.ctrl(q[c[i]], q[a[i] + 3], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 34:
                qfl_v4(q[a[i] + 3], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 35:
                qfl_v4.ctrl(q[c[i]], q[a[i] + 3], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 36:
                qfl_u5(q[a[i] + 4], q[a[i] + 3], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 37:
                qfl_u5.ctrl(q[c[i]], q[a[i] + 4], q[a[i] + 3], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 38:
                qfl_v5(q[a[i] + 4], q[a[i] + 3], q[a[i] + 2], q[a[i] + 1], q[a[i]])
            elif code == 39:
                qfl_v5.ctrl(q[c[i]], q[a[i] + 4], q[a[i] + 3], q[a[i] + 2], q[a[i] + 1], q[a[i]])

    @cudaq.kernel
    def state(
        count: int, codes: list[int], a: list[int], b: list[int], c: list[int], t: list[float]
    ):
        q = cudaq.qvector(count)
        run(q, codes, a, b, c, t)

    @cudaq.kernel
    def sample(
        count: int, codes: list[int], a: list[int], b: list[int], c: list[int], t: list[float]
    ):
        q = cudaq.qvector(count)
        run(q, codes, a, b, c, t)
        mz(q)

    # Force compilation now, while this unitary is the registered one.
    probe = (max(1, width + 1), [0], [0], [0], [0], [0.0])
    cudaq.get_state(state, *probe)
    cudaq.sample(sample, *probe, shots_count=1)
    return state, sample


class ProgramRunner:
    """Executes circuits on a CUDA-Q simulator target; the interface the algorithms see."""

    def __init__(self, seed: int, target: str = "qpp-cpu", native_sampling: bool = False) -> None:
        select_target(target)
        self.target = target
        # ``cudaq.sample`` on the interpreter kernel costs about 0.1 s per call regardless of
        # circuit size, against about 0.01 s for ``cudaq.get_state``. Algorithms here make
        # thousands of one-shot calls, so by default shots are drawn from the probabilities of
        # the CUDA-Q statevector, which is the same law the simulator's own sampler draws from.
        # ``native_sampling=True`` uses ``cudaq.sample`` instead; a test compares the two.
        self.native_sampling = native_sampling or target != "qpp-cpu"
        self.name = f"CUDA-Q {cudaq.__version__} ({target})"
        self._seeds = np.random.default_rng(seed)
        self._kernels: dict[str, tuple] = {}
        self._distributions: dict[int, tuple] = {}

    def reseed(self, seed: int) -> None:
        """Restart the shot stream; compiled kernels and cached distributions are kept."""
        self._seeds = np.random.default_rng(seed)

    def _for(self, circuit: Circuit) -> tuple:
        key = (
            "none"
            if circuit.unitary is None
            else hashlib.sha1(np.ascontiguousarray(circuit.unitary).tobytes()).hexdigest()
        )
        if key not in self._kernels:
            if len(self._kernels) > 64:
                self._kernels.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._kernels[key] = _build(circuit.unitary)
        return self._kernels[key]

    def state(self, circuit: Circuit) -> np.ndarray:
        """Privileged simulator access: the full statevector. A device cannot provide it."""
        select_target("qpp-cpu")
        kernel, _ = self._for(circuit)
        return np.array(cudaq.get_state(kernel, circuit.qubits, *circuit.arrays()))

    def sample(self, circuit: Circuit, shots: int) -> dict[str, int]:
        """Measure every qubit ``shots`` times; bitstrings list qubit 0 first."""
        if not self.native_sampling:
            # Re-sampling the same circuit object does not need a second simulation.
            cached = self._distributions.get(id(circuit))
            if cached and cached[0]() is circuit and cached[1] == len(circuit.ops):
                probabilities = cached[2]
            else:
                probabilities = np.abs(self.state(circuit)) ** 2
                if len(self._distributions) > 256:
                    self._distributions.clear()
                self._distributions[id(circuit)] = (
                    weakref.ref(circuit),
                    len(circuit.ops),
                    probabilities,
                )
            draws = self._seeds.multinomial(shots, probabilities / probabilities.sum())
            return {bitstring(int(i), circuit.qubits): int(draws[i]) for i in np.flatnonzero(draws)}
        select_target(self.target)
        _, kernel = self._for(circuit)
        cudaq.set_random_seed(int(self._seeds.integers(1, 2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cudaq.sample(kernel, circuit.qubits, *circuit.arrays(), shots_count=shots)
        return {bits: int(count) for bits, count in result.items()}

    def resources(self, circuit: Circuit) -> dict:
        """Gate counts and depth of the circuit CUDA-Q builds, plus the construction's own."""
        select_target("qpp-cpu")
        kernel, _ = self._for(circuit)
        compiled = _compiled(kernel, circuit.qubits, *circuit.arrays())
        return {
            **circuit.counts(),
            "depth": compiled["depth"],
            "multi_qubit_depth": compiled["multi_qubit_depth"],
            "compiled_gates": compiled["gates"],
        }


class XyQaoaSampler:
    """Constraint-preserving QAOA; same interface as ``cudaq_backend.QaoaSampler``."""

    def __init__(self, field, couplings, layers, seed, target="qpp-cpu", select_count=1) -> None:
        from qflab.methods.qaoa_circuits import xy_qaoa_circuit

        self._build = lambda parameters: xy_qaoa_circuit(
            [float(v) for v in field],
            [(int(a), int(b), float(j)) for a, b, j in couplings],
            select_count,
            [float(v) for v in parameters],
        )
        self.layers = layers
        self._runner = ProgramRunner(seed, target)

    def __call__(self, parameters, shots: int) -> dict[str, int]:
        return self._runner.sample(self._build(parameters), shots)

    def probabilities(self, parameters) -> np.ndarray:
        return np.abs(self._runner.state(self._build(parameters))) ** 2

    def compiled_resources(self) -> dict:
        found = self._runner.resources(self._build([0.1] * (2 * self.layers)))
        return {**found, "gates": found["compiled_gates"]}
