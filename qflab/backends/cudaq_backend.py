"""CUDA-Q execution adapter: the only module that knows about kernels and simulator targets.

Two circuit families are built here.

* Amplitude estimation: a state preparation ``A`` that loads a discrete distribution and a
  bounded payoff, and the Grover operator ``Q = A S0 A^-1 S_chi`` built from it.
* QAOA: alternating cost and mixer layers for an Ising cost function.

Conventions, checked by the tests against the simulator's statevector:

* Scenario ``j`` is stored on qubits ``0..n-1`` with qubit 0 as the least significant bit.
* The indicator (payoff) qubit is the last qubit, index ``n``.
* CUDA-Q statevector index ``i`` has qubit 0 as its least significant bit; sampled bitstrings
  list qubit 0 first.

Every rotation and CNOT of ``A`` is an explicit gate. Loading a dense table costs about
``2 * 2^n`` rotations, and the resource report shows that cost instead of hiding it.
"""

from __future__ import annotations

import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cudaq

from qflab.problems.expectation import ExpectationProblem

CUDAQ_VERSION = cudaq.__version__


class BackendUnavailable(RuntimeError):
    """The requested simulator or device is not installed on this machine."""


def select_target(name: str) -> None:
    if not cudaq.has_target(name):
        raise BackendUnavailable(
            f"CUDA-Q target '{name}' is not available here; installed simulators include "
            "qpp-cpu and density-matrix-cpu (nvidia needs an NVIDIA GPU)"
        )
    if cudaq.get_target().name != name:
        cudaq.set_target(name)


def _noise_model(noise: dict | None):
    """Depolarizing noise after gates. Gates with two or more controls are left noiseless."""
    if not noise:
        return None
    allowed = {"single_qubit_depolarizing", "two_qubit_depolarizing"}
    if set(noise) - allowed:
        raise ValueError(f"Unsupported noise settings: {sorted(set(noise) - allowed)}")
    model = cudaq.NoiseModel()
    one = float(noise.get("single_qubit_depolarizing", 0.0))
    two = float(noise.get("two_qubit_depolarizing", 0.0))
    if one > 0:
        for gate in ("ry", "rx", "rz", "x", "z", "h"):
            model.add_all_qubit_channel(gate, cudaq.DepolarizationChannel(one))
    if two > 0:
        for gate in ("x", "z"):
            model.add_all_qubit_channel(gate, cudaq.Depolarization2(two), num_controls=1)
    return model


# --------------------------------------------------------------------- state preparation
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


def amplitude_program(problem: ExpectationProblem) -> list[tuple[float, int, int]]:
    """Gate list for ``A|0> = sum_j sqrt(p_j)|j>(sqrt(1-g_j)|0> + sqrt(g_j)|1>)``."""
    n = problem.num_qubits
    gates: list[tuple[float, int, int]] = []
    for level in range(n):
        # Probability of each pattern of qubits 0..level, summed over the higher qubits.
        marginal = problem.probabilities.reshape(-1, 2 ** (level + 1)).sum(axis=0)
        zero, one = marginal[: 2**level], marginal[2**level :]
        alphas = 2 * np.arctan2(np.sqrt(one), np.sqrt(zero))
        gates += uniformly_controlled_ry(alphas, list(range(level)), level)
    payoff = np.clip(problem.scaled_payoff, 0.0, 1.0)
    gates += uniformly_controlled_ry(2 * np.arcsin(np.sqrt(payoff)), list(range(n)), n)
    return gates


@cudaq.kernel
def _apply(q: cudaq.qview, angles: list[float], targets: list[int], controls: list[int]):
    for i in range(len(angles)):
        ry(angles[i], q[targets[i]])
        if controls[i] >= 0:
            x.ctrl(q[controls[i]], q[targets[i]])


@cudaq.kernel
def _apply_inverse(q: cudaq.qview, angles: list[float], targets: list[int], controls: list[int]):
    for j in range(len(angles)):
        i = len(angles) - 1 - j
        if controls[i] >= 0:
            x.ctrl(q[controls[i]], q[targets[i]])
        ry(-angles[i], q[targets[i]])


@cudaq.kernel
def _grover(q: cudaq.qview, k: int, angles: list[float], targets: list[int], controls: list[int]):
    count = q.size()
    _apply(q, angles, targets, controls)
    for _ in range(k):
        z(q[count - 1])  # S_chi: flip the sign of "indicator = 1"
        _apply_inverse(q, angles, targets, controls)
        x(q)  # S_0: flip the sign of |0...0>
        z.ctrl(q[0 : count - 1], q[count - 1])
        x(q)
        _apply(q, angles, targets, controls)


@cudaq.kernel
def _amplitude_state(
    count: int, k: int, angles: list[float], targets: list[int], controls: list[int]
):
    q = cudaq.qvector(count)
    _grover(q, k, angles, targets, controls)


@cudaq.kernel
def _amplitude_sample(
    count: int, k: int, angles: list[float], targets: list[int], controls: list[int]
):
    q = cudaq.qvector(count)
    _grover(q, k, angles, targets, controls)
    mz(q[count - 1])


def _compiled(kernel, *args) -> dict:
    """Gate counts of the circuit CUDA-Q actually builds, excluding measurements."""
    resources = cudaq.estimate_resources(kernel, *args)
    counts = {k: v for k, v in resources.to_dict().items() if not k.startswith("m")}
    return {
        "qubits": resources.num_qubits,
        "gates": sum(counts.values()),
        "multi_qubit_gates": resources.multi_qubit_gate_count,
        "depth": resources.depth,
        "multi_qubit_depth": resources.multi_qubit_depth,
        "gate_counts": counts,
    }


class AmplitudeSampler:
    """Runs ``Q^k A|0>`` on a CUDA-Q simulator and counts indicator = 1 outcomes."""

    def __init__(
        self,
        problem: ExpectationProblem,
        seed: int,
        target: str = "qpp-cpu",
        noise: dict | None = None,
    ) -> None:
        select_target(target)
        if noise and target != "density-matrix-cpu":
            raise ValueError("Noise needs the density-matrix-cpu target")
        self.problem, self.target = problem, target
        self.noise = _noise_model(noise)
        self.count = problem.num_qubits + 1
        program = amplitude_program(problem)
        self.angles = [g[0] for g in program]
        self.targets = [g[1] for g in program]
        self.controls = [g[2] for g in program]
        # One fresh simulator seed per call keeps rounds independent and runs reproducible.
        self._seeds = np.random.default_rng(seed)

    def _args(self, k: int):
        return self.count, k, self.angles, self.targets, self.controls

    def __call__(self, k: int, shots: int) -> int:
        select_target(self.target)
        cudaq.set_random_seed(int(self._seeds.integers(1, 2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.noise is None:
                result = cudaq.sample(_amplitude_sample, *self._args(k), shots_count=shots)
            else:
                result = cudaq.sample(
                    _amplitude_sample, *self._args(k), shots_count=shots, noise_model=self.noise
                )
        return int(result.count("1"))

    def statevector(self, k: int) -> np.ndarray:
        """Privileged simulator access, for verification only; a device cannot provide it."""
        select_target("qpp-cpu")
        state = np.array(cudaq.get_state(_amplitude_state, *self._args(k)))
        return state.real if np.allclose(state.imag, 0, atol=1e-12) else state

    def indicator_probability(self, k: int) -> float:
        state = self.statevector(k)
        return float(np.sum(np.abs(state[len(state) // 2 :]) ** 2))

    def resources(self, k: int) -> dict:
        """Gate counts from the construction itself (logical gates, all-to-all connectivity)."""
        n = self.count - 1
        rotations = len(self.angles)
        cnots = sum(c >= 0 for c in self.controls)
        per_iteration = 2 * (rotations + cnots) + 2 * self.count + 2
        reflection_is_two_qubit = n == 1
        return {
            "qubits": self.count,
            "scenario_qubits": n,
            "ancilla_qubits": 0,
            "grover_power": k,
            "rotations": rotations,
            "cnots_in_state_preparation": cnots,
            "gates": rotations + cnots + k * per_iteration,
            "two_qubit_gates": cnots + k * (2 * cnots + reflection_is_two_qubit),
            "multi_controlled_gates": 0 if reflection_is_two_qubit else k,
            "largest_gate_controls": n if k else min(n, 1),
            "basis": "ry, x, z, cx, and one multi-controlled z per Grover step; "
            "all-to-all connectivity; multi-controlled gates not decomposed",
        }

    def compiled_resources(self, k: int) -> dict:
        select_target("qpp-cpu")
        return _compiled(_amplitude_state, *self._args(k))


# ---------------------------------------------------------------------------------- QAOA
@cudaq.kernel
def _qaoa(
    count: int,
    layers: int,
    parameters: list[float],
    field: list[float],
    left: list[int],
    right: list[int],
    coupling: list[float],
):
    q = cudaq.qvector(count)
    h(q)
    for layer in range(layers):
        gamma = parameters[layer]
        beta = parameters[layers + layer]
        for t in range(len(coupling)):
            x.ctrl(q[left[t]], q[right[t]])
            rz(2.0 * gamma * coupling[t], q[right[t]])
            x.ctrl(q[left[t]], q[right[t]])
        for i in range(count):
            rz(2.0 * gamma * field[i], q[i])
        for i in range(count):
            rx(2.0 * beta, q[i])


class QaoaSampler:
    """Samples bitstrings from a QAOA circuit for ``E(z) = sum h_i z_i + sum J_ij z_i z_j``.

    ``z_i = +1`` means bit 0 (not selected) and ``z_i = -1`` means bit 1 (selected). Returned
    bitstrings list qubit 0 first, matching the order of ``field``.
    """

    def __init__(
        self,
        field: list[float],
        couplings: list[tuple[int, int, float]],
        layers: int,
        seed: int,
        target: str = "qpp-cpu",
    ) -> None:
        select_target(target)
        if layers < 1:
            raise ValueError("QAOA needs at least one layer")
        self.target, self.layers, self.count = target, layers, len(field)
        self.field = [float(v) for v in field]
        self.left = [int(a) for a, _, _ in couplings]
        self.right = [int(b) for _, b, _ in couplings]
        self.coupling = [float(j) for _, _, j in couplings]
        self._seeds = np.random.default_rng(seed)

    def _args(self, parameters):
        values = [float(v) for v in parameters]
        if len(values) != 2 * self.layers:
            raise ValueError("Expected one gamma and one beta per layer")
        return self.count, self.layers, values, self.field, self.left, self.right, self.coupling

    def __call__(self, parameters, shots: int) -> dict[str, int]:
        select_target(self.target)
        cudaq.set_random_seed(int(self._seeds.integers(1, 2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cudaq.sample(_qaoa, *self._args(parameters), shots_count=shots)
        return {bits: int(count) for bits, count in result.items()}

    def probabilities(self, parameters) -> np.ndarray:
        """Privileged exact distribution; index bit i (least significant first) is qubit i."""
        select_target("qpp-cpu")
        return np.abs(np.array(cudaq.get_state(_qaoa, *self._args(parameters)))) ** 2

    def compiled_resources(self) -> dict:
        select_target("qpp-cpu")
        return _compiled(_qaoa, *self._args([0.1] * (2 * self.layers)))


def environment() -> dict:
    return {
        "cudaq": CUDAQ_VERSION,
        "default_target": cudaq.get_target().name,
        "gpu_target_available": bool(cudaq.has_target("nvidia")),
        "numpy": np.__version__,
    }
