"""A quantum circuit Born machine (QCBM) for a binned return distribution (plan 14, item 8).

Reference: Benedetti et al., "A generative modeling approach for benchmarking and training
shallow quantum circuits", npj Quantum Information 5, 45 (2019).

The model is a parameterized circuit on ``n`` qubits; measuring it gives one of ``2^n`` bins.
Training minimizes the cross-entropy between the training histogram and the circuit's exact
output probabilities, which are read from the simulator (privileged access: a device would
estimate them from shots, and the report states what that would cost).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from qflab.circuit import Circuit


def born_circuit(parameters: np.ndarray, qubits: int, layers: int) -> Circuit:
    """``layers`` blocks of RY, RZ on every qubit and a ring of CNOTs, then a final RY layer."""
    if len(parameters) != parameter_count(qubits, layers):
        raise ValueError("Wrong number of parameters")
    circuit = Circuit(qubits)
    values = iter(float(v) for v in parameters)
    for _ in range(layers):
        for q in range(qubits):
            circuit.ry(next(values), q).rz(next(values), q)
        for q in range(qubits):
            circuit.cx(q, (q + 1) % qubits)
    for q in range(qubits):
        circuit.ry(next(values), q)
    return circuit


def parameter_count(qubits: int, layers: int) -> int:
    return qubits * (2 * layers + 1)


def train_born_machine(
    runner, target: np.ndarray, qubits: int, layers: int, seed: int, iterations: int
) -> dict:
    """Fit the circuit's output distribution to ``target`` (a histogram over 2^qubits bins)."""
    if len(target) != 2**qubits or not math.isclose(float(target.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("target must be a probability vector over 2^qubits bins")
    evaluations = 0
    history: list[float] = []

    def probabilities(parameters: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        return np.abs(runner.state(born_circuit(parameters, qubits, layers))) ** 2

    def loss(parameters: np.ndarray) -> float:
        model = np.clip(probabilities(parameters), 1e-12, None)
        return float(-(target * np.log(model)).sum())

    rng = np.random.default_rng(seed)
    start = rng.uniform(-math.pi, math.pi, parameter_count(qubits, layers))
    found = minimize(
        loss,
        start,
        method="L-BFGS-B",
        callback=lambda p: history.append(loss(p)),
        options={"maxiter": iterations},
    )
    return {
        "parameters": found.x,
        "probabilities": probabilities(found.x),
        "train_cross_entropy": float(found.fun),
        "circuit_evaluations": evaluations,
        "loss_history": history,
        "optimizer_iterations": int(found.nit),
    }
