"""Sample-based subspace diagonalization for covariance spectra (plan section 13.3, R2).

The task: the largest eigenvalue and leading risk-factor direction of an N x N covariance
matrix, N = 2^n. Assets are the computational basis states of an n-qubit register, and
``H = -Sigma / ||Sigma||_F`` so that the leading factor is the ground state.

* **SQD** samples asset indices from a trial state, keeps the ``m`` most frequent, and
  diagonalizes the m x m sub-block of Sigma classically.
* **SKQD** takes its samples from ``exp(-i H t_k)|trial>`` for increasing times ``t_k``
  (a Krylov sequence), with either an ideal evolution (dense unitary: classical preprocessing)
  or a declared Trotter approximation built from the Pauli decomposition of H.

What comes out is an ordinary m-sparse classical vector, so there is no tomography cost; the
classical side needs the m^2 covariance entries of the chosen assets.

A subspace of m coordinates can capture at most the m largest squared entries of the true
eigenvector. For a uniform market factor that is m / N, however the samples are drawn.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigsh

from qflab.circuit import Circuit, load_amplitudes, pauli_decompose, trotter_evolution


# ---------------------------------------------------------------- covariance families
def localized_covariance(size: int, block: int, gap: float, seed: int) -> np.ndarray:
    """A few strongly co-moving assets dominate: the leading eigenvector is concentrated."""
    rng = np.random.default_rng(seed)
    vols = 0.5 + 0.2 * rng.uniform(0, 1, size)
    vols[:block] *= math.sqrt(gap * 2.0)
    correlation = np.full((size, size), 0.05)
    correlation[:block, :block] = 0.85
    np.fill_diagonal(correlation, 1.0)
    return correlation * np.outer(vols, vols)


def market_covariance(size: int, correlation: float, seed: int) -> np.ndarray:
    """One market factor loads on every asset: the leading eigenvector is delocalized."""
    rng = np.random.default_rng(seed)
    vols = 1.0 + 0.1 * rng.uniform(-1, 1, size)
    matrix = np.full((size, size), correlation)
    np.fill_diagonal(matrix, 1.0)
    return matrix * np.outer(vols, vols)


def clustered_covariance(size: int, clusters: int, gap: float, seed: int) -> np.ndarray:
    """Sector blocks; the first sector is scaled by ``gap`` to control the spectral gap."""
    rng = np.random.default_rng(seed)
    per = size // clusters
    vols = 1.0 + 0.1 * rng.uniform(-1, 1, size)
    vols[:per] *= math.sqrt(gap)
    correlation = np.full((size, size), 0.02)
    for c in range(clusters):
        correlation[c * per : (c + 1) * per, c * per : (c + 1) * per] = 0.6
    np.fill_diagonal(correlation, 1.0)
    return correlation * np.outer(vols, vols)


def pad_to_power_of_two(matrix: np.ndarray) -> np.ndarray:
    size = len(matrix)
    full = 1 << (size - 1).bit_length()
    padded = np.zeros((full, full))
    padded[:size, :size] = matrix
    return padded


# ------------------------------------------------------------------------ references
def spectrum_reference(covariance: np.ndarray) -> dict:
    values, vectors = np.linalg.eigh(covariance)
    lead = vectors[:, -1] * np.sign(vectors[:, -1].sum() or 1.0)
    weight = np.sort(lead**2)[::-1].cumsum()
    return {
        "eigenvalue": float(values[-1]),
        "second_eigenvalue": float(values[-2]),
        "gap_ratio": float(values[-1] / values[-2]),
        "vector": lead,
        "explained_variance": float(values[-1] / values.sum()),
        "coordinates_for_90_percent": int(np.searchsorted(weight, 0.9) + 1),
        "coordinates_for_99_percent": int(np.searchsorted(weight, 0.99) + 1),
        "best_possible_overlap": [float(v) for v in weight],
    }


def lanczos_reference(covariance: np.ndarray) -> dict:
    """Matrix-free Lanczos for the leading pair, counting matrix-vector products."""
    calls = 0

    def product(vector):
        nonlocal calls
        calls += 1
        return covariance @ vector

    size = len(covariance)
    operator = LinearOperator((size, size), matvec=product, dtype=float)
    values, _ = eigsh(operator, k=1, which="LA", v0=np.ones(size) / math.sqrt(size), tol=1e-10)
    return {"eigenvalue": float(values[0]), "matrix_vector_products": calls}


def randomized_reference(covariance: np.ndarray, seed: int, power_iterations: int = 4) -> dict:
    """Randomized range finder with a few power iterations (rank 1, oversampling 4)."""
    rng = np.random.default_rng(seed)
    probe = rng.normal(size=(len(covariance), 5))
    products = 5
    for _ in range(power_iterations):
        probe, _ = np.linalg.qr(covariance @ probe)
        products += 5
    small = probe.T @ covariance @ probe
    return {
        "eigenvalue": float(np.linalg.eigvalsh(small)[-1]),
        "matrix_vector_products": products + 5,
    }


def project(covariance: np.ndarray, indices: list[int], reference: dict) -> dict:
    """Diagonalize the sub-block on ``indices`` and score it against the true leading pair."""
    indices = sorted(set(int(i) for i in indices))
    block = covariance[np.ix_(indices, indices)]
    values, vectors = np.linalg.eigh(block)
    vector = np.zeros(len(covariance))
    vector[indices] = vectors[:, -1]
    overlap = float((vector @ reference["vector"]) ** 2)
    return {
        "subspace_size": len(indices),
        "eigenvalue": float(values[-1]),
        "eigenvalue_error": float(1 - values[-1] / reference["eigenvalue"]),
        "vector_error": 1 - overlap,
        "vector": vector,
    }


def diagonal_selection(covariance: np.ndarray, size: int) -> list[int]:
    """Classical competitor for choosing the sub-block: the ``size`` largest variances."""
    return [int(i) for i in np.argsort(np.diag(covariance))[::-1][:size]]


def power_selection(covariance: np.ndarray, size: int, steps: int) -> list[int]:
    """Classical Krylov competitor: the ``size`` largest entries of ``Sigma^steps 1``.

    Uses the same start (uniform) and the same number of applications of the covariance as an
    SKQD run with ``steps + 1`` Krylov states, but no quantum sampling.
    """
    vector = np.ones(len(covariance))
    for _ in range(steps):
        vector = covariance @ vector
        vector /= np.linalg.norm(vector)
    return [int(i) for i in np.argsort(-np.abs(vector), kind="stable")[:size]]


def downstream_error(covariance: np.ndarray, eigenvalue: float, vector: np.ndarray) -> float:
    """Relative error of the equal-weight portfolio variance under a one-factor model built
    from the estimate (factor part plus the true residual diagonal)."""
    size = len(covariance)
    weights = np.ones(size) / size
    factor = eigenvalue * np.outer(vector, vector)
    model = factor + np.diag(np.clip(np.diag(covariance) - np.diag(factor), 0, None))
    return float(abs(weights @ model @ weights / (weights @ covariance @ weights) - 1))


# ------------------------------------------------------------------------- samplers
def trial_circuit(covariance: np.ndarray, initial: str) -> Circuit:
    size = len(covariance)
    n = size.bit_length() - 1
    circuit = Circuit(n)
    if initial == "uniform":
        for q in range(n):
            circuit.h(q)
    elif initial == "variance":
        # Amplitudes proportional to each asset's volatility: O(N) classical information.
        load_amplitudes(circuit, list(range(n)), np.sqrt(np.diag(covariance)))
    else:
        raise ValueError("initial must be 'uniform' or 'variance'")
    return circuit


def krylov_circuits(
    covariance: np.ndarray,
    initial: str,
    times: list[float],
    evolution: str = "ideal",
    trotter_steps: int = 2,
) -> list[Circuit]:
    """One circuit per evolution time for ``exp(-i H t)|trial>``, H = -Sigma / ||Sigma||_F."""
    size = len(covariance)
    n = size.bit_length() - 1
    hamiltonian = -covariance / np.linalg.norm(covariance)
    terms = pauli_decompose(hamiltonian) if evolution == "trotter" else []
    circuits = []
    for time in times:
        circuit = trial_circuit(covariance, initial)
        if time != 0 and evolution == "ideal":
            circuit.set_unitary(expm(-1j * time * hamiltonian))
            circuit.add("u", 0)
        elif time != 0:
            steps = max(1, round(trotter_steps * abs(time)))
            trotter_evolution(circuit, terms, -time, steps, list(range(n)))
        circuits.append(circuit)
    return circuits


def sample_indices(circuits: list[Circuit], runner, shots: int) -> dict[int, int]:
    """Pool the measured asset indices of all circuits; ``shots`` per circuit."""
    pooled: dict[int, int] = {}
    for circuit in circuits:
        for bits, count in runner.sample(circuit, shots).items():
            index = sum(int(b) << i for i, b in enumerate(bits))
            pooled[index] = pooled.get(index, 0) + count
    return pooled


def most_frequent(counts: dict[int, int], size: int) -> list[int]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [index for index, _ in ranked[:size]]
