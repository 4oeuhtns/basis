"""Quantum signal processing: a bounded odd polynomial for 1/x and its phase factors.

Used by the QSVT linear solver. Everything here is classical preprocessing, and its cost
(polynomial degree, solve time, residual) is reported with the solver's results.

Conventions. For a scalar ``x`` in [-1, 1] let ``R(x) = [[x, s], [s, -x]]`` with
``s = sqrt(1 - x^2)``: this is how any block encoding acts on the two-dimensional subspace of a
singular value ``x``. With phases ``phi_0 .. phi_d`` the sequence

    M(x) = e^{i phi_0 Z} R(x) e^{i phi_1 Z} R(x) ... R(x) e^{i phi_d Z}

has ``Re M(x)[0, 0] = f(x)`` for the target polynomial ``f``. The phases are found in the
equivalent ``W(x) = [[x, i s], [i s, x]]`` convention with symmetric phase factors by a
least-squares solve at Chebyshev nodes (Dong, Meng, Whaley and Lin, Phys. Rev. A 103, 042419
(2021)), then converted.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numpy.polynomial import chebyshev
from scipy.optimize import least_squares


def reciprocal_target(kappa: float, epsilon: float, peak: float = 0.8):
    """An odd, bounded stand-in for 1/x: ``scale * (1 - exp(-(b x)^2)) / x``.

    On ``|x| >= 1/kappa`` it equals ``scale / x`` to relative error ``epsilon``; near zero it
    bends smoothly through the origin so that it stays below ``peak`` everywhere.
    """
    if kappa < 1 or not 0 < epsilon < 0.5 or not 0 < peak < 1:
        raise ValueError("Need kappa >= 1, 0 < epsilon < 0.5 and 0 < peak < 1")
    width = kappa * math.sqrt(math.log(1 / epsilon))
    # max over x of (1 - exp(-(b x)^2)) / x is 0.6382 * b.
    scale = peak / (0.6382 * width)

    def function(x):
        x = np.asarray(x, dtype=float)
        safe = np.where(np.abs(x) < 1e-12, 1.0, x)
        return np.where(
            np.abs(x) < 1e-12, scale * width**2 * x, scale * -np.expm1(-((width * x) ** 2)) / safe
        )

    return function, scale


def odd_chebyshev_fit(function, tolerance: float, max_degree: int = 999) -> np.ndarray:
    """Chebyshev coefficients of the lowest odd degree reproducing ``function`` on [-1, 1]."""
    nodes = 2 * max_degree + 2
    full = chebyshev.chebinterpolate(function, nodes)
    full[::2] = 0.0  # the target is odd
    grid = np.cos(np.linspace(0, math.pi, 4001))
    exact = function(grid)
    for degree in range(1, max_degree + 1, 2):
        if np.max(np.abs(chebyshev.chebval(grid, full[: degree + 1]) - exact)) < tolerance:
            return full[: degree + 1]
    raise ValueError(f"No odd polynomial of degree <= {max_degree} reaches tolerance {tolerance}")


def _w_response(phases: np.ndarray, x: np.ndarray) -> np.ndarray:
    """<0| e^{i p0 Z} W e^{i p1 Z} ... W e^{i pd Z} |0> for every x (W convention)."""
    s = np.sqrt(np.clip(1 - x**2, 0, None))
    # Track the row vector <0| M as two complex components per x.
    top = np.exp(1j * phases[0]) * np.ones_like(x, dtype=complex)
    bottom = np.zeros_like(top)
    for phase in phases[1:]:
        top, bottom = top * x + bottom * 1j * s, top * 1j * s + bottom * x
        top, bottom = top * np.exp(1j * phase), bottom * np.exp(-1j * phase)
    return top


def r_response(phases: np.ndarray, x: np.ndarray) -> np.ndarray:
    """<0| e^{i p0 Z} R e^{i p1 Z} ... R e^{i pd Z} |0> for every x (R convention)."""
    s = np.sqrt(np.clip(1 - x**2, 0, None))
    top = np.exp(1j * phases[0]) * np.ones_like(x, dtype=complex)
    bottom = np.zeros_like(top)
    for phase in phases[1:]:
        top, bottom = top * x + bottom * s, top * s - bottom * x
        top, bottom = top * np.exp(1j * phase), bottom * np.exp(-1j * phase)
    return top


def solve_phases(coefficients: np.ndarray) -> dict:
    """Phase factors (R convention) whose real response is the given odd Chebyshev series."""
    degree = len(coefficients) - 1
    if degree % 2 != 1 or np.any(np.abs(coefficients[::2]) > 1e-14):
        raise ValueError("Only odd polynomials are supported")
    started = time.perf_counter()
    half = (degree + 1) // 2
    nodes = np.cos((2 * np.arange(1, half + 1) - 1) * math.pi / (4 * half))
    target = chebyshev.chebval(nodes, coefficients)
    if np.max(np.abs(target)) >= 1:
        raise ValueError("The target polynomial must stay strictly inside (-1, 1)")

    def full(reduced: np.ndarray) -> np.ndarray:
        # Symmetric phases: phi_j = phi_{d-j}. The reduced vector holds the second half.
        phases = np.concatenate([reduced[::-1], reduced])
        phases[0] += math.pi / 4
        phases[-1] += math.pi / 4
        return phases

    def residual(reduced: np.ndarray) -> np.ndarray:
        return _w_response(full(reduced), nodes).real - target

    found = least_squares(residual, np.zeros(half), method="lm", xtol=1e-15, ftol=1e-15)
    w_phases = full(found.x)
    # Convert: W = -i e^{-i pi/4 Z} R e^{i 3 pi/4 Z}, and fold the (-i)^d factor into phi_0.
    phases = w_phases.copy()
    phases[0] += -math.pi / 4 - math.pi * degree / 2
    phases[1:-1] += math.pi / 2
    phases[-1] += 3 * math.pi / 4
    grid = np.cos(np.linspace(0, math.pi, 2001))
    error = float(
        np.max(np.abs(r_response(phases, grid).real - chebyshev.chebval(grid, coefficients)))
    )
    return {
        "phases": phases,
        "degree": degree,
        "residual": error,
        "solve_seconds": time.perf_counter() - started,
        "function_evaluations": int(found.nfev),
    }
