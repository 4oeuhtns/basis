"""Quantum fidelity kernels and a small kernel classifier (plan section 14, item 6).

The feature map is the ZZ map of Havlicek et al., "Supervised learning with quantum-enhanced
feature spaces", Nature 567, 209 (2019). The kernel is ``K(x, y) = |<phi(y)|phi(x)>|^2``.

* ``exact_kernel`` reads feature states from the simulator (privileged access).
* ``sampled_kernel`` runs one circuit ``U(y)^dagger U(x)`` per pair and counts all-zero
  outcomes: this is what a device would do, and its cost is one circuit per pair of examples.

The classifier is regularized least squares on +/-1 labels (kernel ridge classification), the
same for quantum and classical kernels, so only the kernel differs.
"""

from __future__ import annotations

import math

import numpy as np

from qflab.circuit import Circuit


def feature_circuit(x: np.ndarray, repetitions: int) -> Circuit:
    n = len(x)
    circuit = Circuit(n)
    for _ in range(repetitions):
        for q in range(n):
            circuit.h(q).rz(2 * float(x[q]), q)
        for a in range(n):
            for b in range(a + 1, n):
                angle = 2 * (math.pi - float(x[a])) * (math.pi - float(x[b]))
                circuit.cx(a, b).rz(angle, b).cx(a, b)
    return circuit


def exact_kernel(runner, left: np.ndarray, right: np.ndarray, repetitions: int) -> np.ndarray:
    states_left = np.array([runner.state(feature_circuit(x, repetitions)) for x in left])
    states_right = (
        states_left
        if right is left
        else np.array([runner.state(feature_circuit(x, repetitions)) for x in right])
    )
    return np.abs(states_left.conj() @ states_right.T) ** 2


def sampled_kernel(
    runner, left: np.ndarray, right: np.ndarray, repetitions: int, shots: int
) -> tuple[np.ndarray, int]:
    """Kernel estimated from measurement counts. Returns (matrix, circuits run)."""
    symmetric = right is left
    kernel = np.ones((len(left), len(right)))
    circuits = 0
    zeros = "0" * left.shape[1]
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            if symmetric and j <= i:
                continue
            circuit = feature_circuit(x, repetitions)
            circuit.extend(feature_circuit(y, repetitions).inverse())
            kernel[i, j] = runner.sample(circuit, shots).get(zeros, 0) / shots
            circuits += 1
            if symmetric:
                kernel[j, i] = kernel[i, j]
    return kernel, circuits


def rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    distance = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * distance)


def polynomial_kernel(left: np.ndarray, right: np.ndarray, degree: int) -> np.ndarray:
    return (1 + left @ right.T / left.shape[1]) ** degree


def fit_ridge(kernel: np.ndarray, labels: np.ndarray, ridge: float) -> np.ndarray:
    return np.linalg.solve(kernel + ridge * np.eye(len(kernel)), labels.astype(float))


def predict(cross_kernel: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """``cross_kernel[i, j] = K(test_i, train_j)``; returns +/-1."""
    return np.where(cross_kernel @ coefficients >= 0, 1, -1)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    share = successes / trials
    centre = (share + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials**2)) / (1 + z * z / trials)
    )
    return centre - half, centre + half
