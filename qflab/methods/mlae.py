"""Maximum-likelihood amplitude estimation (MLAE), independent of any quantum SDK.

Reference: Suzuki, Uno, Raymond, Tanaka, Onodera and Yamamoto, "Amplitude estimation without
phase estimation", Quantum Information Processing 19, 75 (2020).

Unlike IQAE, the Grover powers are fixed in advance: ``0, 1, 2, 4, ..., 2^(K-2)``. Each power
is run for ``shots`` shots, and one likelihood over all of them is maximized classically. It
takes the same ``sampler(k, shots)`` as IQAE and counts work the same way: one shot at power
``k`` is ``2k + 1`` applications of A or its inverse.

The interval is a likelihood-ratio interval. Its coverage is asymptotic, not guaranteed, and
the likelihood is multi-modal, so the reported interval is the hull of every accepted region.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


def power_schedule(count: int) -> list[int]:
    """Exponentially incremental schedule with ``count`` circuits."""
    if count < 1:
        raise ValueError("Need at least one circuit")
    return [0] + [2**j for j in range(count - 1)]


def mlae(sampler, powers: list[int], shots: int, alpha: float) -> dict:
    if not powers or shots < 1 or not 0 < alpha < 1:
        raise ValueError("Need at least one power, shots >= 1 and 0 < alpha < 1")
    ones = []
    for k in powers:
        hits = int(sampler(k, shots))
        if not 0 <= hits <= shots:
            raise ValueError("Sampler returned an impossible count")
        ones.append(hits)
    scale = 2 * np.array(powers) + 1
    hits = np.array(ones, dtype=float)

    def log_likelihood(theta):
        angle = np.multiply.outer(np.atleast_1d(theta), scale)
        p = np.clip(np.sin(angle) ** 2, 1e-300, 1 - 1e-16)
        return (hits * np.log(p) + (shots - hits) * np.log1p(-p)).sum(axis=1)

    # Scan finely enough to resolve the fastest oscillation, then polish the best cell.
    points = 64 * int(scale.max()) + 1
    grid = np.linspace(0.0, math.pi / 2, points)
    values = log_likelihood(grid)
    best = int(values.argmax())
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, points - 1)]
    found = minimize_scalar(
        lambda t: -float(log_likelihood(t)[0]),
        bounds=(low, high),
        method="bounded",
        options={"xatol": 1e-13},
    )
    theta = float(found.x) if -found.fun >= values[best] else float(grid[best])
    peak = max(float(-found.fun), float(values[best]))
    accepted = grid[2 * (peak - values) <= chi2.ppf(1 - alpha, 1)]
    theta_low, theta_high = min(theta, accepted.min()), max(theta, accepted.max())
    step = grid[1] - grid[0]
    theta_low, theta_high = max(0.0, theta_low - step), min(math.pi / 2, theta_high + step)
    estimate = math.sin(theta) ** 2
    preparation = int(shots * scale.sum())
    return {
        "method": "mlae",
        "target": "discrete",
        "status": "ok",
        "confidence": 1 - alpha,
        "estimate": estimate,
        "ci_low": math.sin(theta_low) ** 2,
        "ci_high": math.sin(theta_high) ** 2,
        "interval": "likelihood ratio (asymptotic), hull of accepted regions",
        "rounds": [
            {"k": int(k), "shots": shots, "ones": int(h)} for k, h in zip(powers, ones, strict=True)
        ],
        "fisher_information": float(
            shots * (scale**2).sum() / max(estimate * (1 - estimate), 1e-300)
        ),
        "resources": {
            "shots": shots * len(powers),
            "rounds": len(powers),
            "grover_calls": int(shots * sum(powers)),
            "state_preparation_calls": preparation,
            "max_grover_power": int(max(powers)),
            "work": preparation,
            "work_unit": "applications of A or its inverse",
        },
    }
