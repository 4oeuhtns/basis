"""Small symmetric positive-definite linear systems and the portfolio task built on them.

Plan section 12. The financial task is the budget-constrained minimum-variance portfolio with
unrestricted signs: for ``A = Sigma + delta * I``,

    minimize w^T A w subject to 1^T w = 1,   solved by   A x = 1,  w = x / (1^T x).

Weights can be negative. This task has no long-only, cardinality or turnover constraint and is
kept separate from the QAOA selection problem. Regularization (``delta``) changes the
portfolio model, so every method is given the same regularized matrix.

Everything here is classical: the reference solution uses a Cholesky solve, never an explicit
inverse.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def spd_with_spectrum(eigenvalues: list[float], seed: int) -> np.ndarray:
    """A symmetric positive-definite matrix with exactly these eigenvalues, in a random basis."""
    values = np.asarray(eigenvalues, dtype=float)
    size = len(values)
    if size < 2 or size & (size - 1) or (values <= 0).any():
        raise ValueError("Need a power-of-two number of positive eigenvalues")
    basis, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(size, size)))
    matrix = basis @ np.diag(values) @ basis.T
    return (matrix + matrix.T) / 2


def geometric_spectrum(size: int, kappa: float) -> list[float]:
    """Eigenvalues from 1/kappa to 1, evenly spaced on a log scale."""
    return [float(v) for v in np.geomspace(1 / kappa, 1.0, size)]


def regularized_covariance(
    covariance: np.ndarray, delta_fraction: float
) -> tuple[np.ndarray, float]:
    """``Sigma + delta I`` with ``delta = delta_fraction * mean variance``; returns (A, delta)."""
    covariance = np.asarray(covariance, dtype=float)
    delta = delta_fraction * float(np.mean(np.diag(covariance)))
    return covariance + delta * np.eye(len(covariance)), delta


def describe(matrix: np.ndarray, rhs: np.ndarray) -> dict:
    """Spectrum, condition number and the classical reference solution of ``A x = b``."""
    matrix, rhs = np.asarray(matrix, dtype=float), np.asarray(rhs, dtype=float)
    if not np.allclose(matrix, matrix.T, atol=1e-12 * np.abs(matrix).max()):
        raise ValueError("Matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0:
        raise ValueError("Matrix must be positive definite")
    solution = cho_solve(cho_factor(matrix), rhs)
    return {
        "size": len(rhs),
        "qubits": int(math.log2(len(rhs))),
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "solution": solution,
        "solution_state": solution / np.linalg.norm(solution),
        "solution_norm": float(np.linalg.norm(solution)),
        "residual": float(np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)),
    }


def minimum_variance(matrix: np.ndarray, solution: np.ndarray | None = None) -> dict:
    """Budget-constrained minimum-variance weights from a (possibly unnormalized) solution.

    Any nonzero overall scale of ``solution`` cancels in ``w = x / (1^T x)``.
    """
    matrix = np.asarray(matrix, dtype=float)
    ones = np.ones(len(matrix))
    if solution is None:
        solution = cho_solve(cho_factor(matrix), ones)
    total = float(ones @ solution)
    if abs(total) < 1e-12:
        raise ValueError("Solution sums to zero; weights are undefined")
    weights = np.asarray(solution, dtype=float) / total
    return {
        "weights": weights,
        "variance": float(weights @ matrix @ weights),
        "sum": float(weights.sum()),
        "negative_weights": int((weights < -1e-12).sum()),
    }
