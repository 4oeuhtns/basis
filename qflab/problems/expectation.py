"""Canonical expectation problems shared by every estimator.

One problem is a finite grid of scenarios with probabilities ``p[j]`` and a payoff scaled into
``g[j]`` in [0, 1]. Every method estimates the same number, the *amplitude*

    a = sum_j p[j] * g[j]

and converts it back to money (or a probability) with ``to_value``. Classical and quantum
estimators receive this object; none of them rebuilds the grid on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ExpectationProblem:
    name: str
    grid: np.ndarray  # scenario outcome, e.g. terminal price or portfolio return
    probabilities: np.ndarray  # sums to one
    scaled_payoff: np.ndarray  # g[j] in [0, 1]
    payoff_scale: float  # raw payoff = payoff_scale * g
    post_factor: float  # e.g. the discount factor
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        size = len(self.grid)
        if size < 2 or size & (size - 1):
            raise ValueError("Scenario count must be a power of two, at least 2")
        if len(self.probabilities) != size or len(self.scaled_payoff) != size:
            raise ValueError("Grid, probabilities and payoff must have equal length")
        if (self.probabilities < 0).any() or not math.isclose(
            float(self.probabilities.sum()), 1.0, abs_tol=1e-12
        ):
            raise ValueError("Probabilities must be non-negative and sum to one")
        if self.scaled_payoff.min() < 0 or self.scaled_payoff.max() > 1 + 1e-12:
            raise ValueError("Scaled payoff must lie in [0, 1]")
        if not (self.payoff_scale > 0 and self.post_factor > 0):
            raise ValueError("Scales must be positive")

    @property
    def num_qubits(self) -> int:
        return int(math.log2(len(self.grid)))

    @property
    def exact_amplitude(self) -> float:
        """The exact discrete target every estimator of this problem is aiming at."""
        return float(self.probabilities @ self.scaled_payoff)

    @property
    def exact_value(self) -> float:
        return self.to_value(self.exact_amplitude)

    def to_value(self, amplitude: float) -> float:
        return self.post_factor * self.payoff_scale * amplitude

    def to_amplitude(self, value: float) -> float:
        return value / (self.post_factor * self.payoff_scale)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "scenario_qubits": self.num_qubits,
            "scenarios": len(self.grid),
            "payoff_scale": self.payoff_scale,
            "post_factor": self.post_factor,
            "exact_amplitude": self.exact_amplitude,
            "exact_value": self.exact_value,
            **self.meta,
        }


def _standard_normal_bins(num_qubits: int, truncation_std: float):
    """Equal-width bins in a standard normal variable, with exact bin masses.

    Mass outside ``[-truncation_std, truncation_std]`` is dropped and the rest renormalized.
    The dropped amount is returned so that reports can show it.
    """
    if type(num_qubits) is not int or not 1 <= num_qubits <= 12:
        raise ValueError("num_qubits must be an integer from 1 to 12")
    if not 0.5 <= truncation_std <= 8:
        raise ValueError("truncation_std must be between 0.5 and 8")
    edges = np.linspace(-truncation_std, truncation_std, 2**num_qubits + 1)
    mass = np.diff(norm.cdf(edges))
    kept = float(mass.sum())
    midpoints = (edges[:-1] + edges[1:]) / 2
    return midpoints, mass / kept, 1.0 - kept


def lognormal_call(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    num_qubits: int,
    truncation_std: float,
) -> ExpectationProblem:
    """European call under a risk-neutral lognormal terminal price, on a truncated grid.

    ln S_T is normal with mean ``ln(spot) + (rate - volatility^2/2) * maturity`` and standard
    deviation ``volatility * sqrt(maturity)``. Each bin is represented by the price at its
    midpoint in log space.
    """
    if not (spot > 0 and strike > 0 and volatility > 0 and maturity > 0):
        raise ValueError("spot, strike, volatility and maturity must be positive")
    z, probabilities, tail = _standard_normal_bins(num_qubits, truncation_std)
    log_mean = math.log(spot) + (rate - volatility**2 / 2) * maturity
    log_std = volatility * math.sqrt(maturity)
    grid = np.exp(log_mean + log_std * z)
    payoff = np.maximum(grid - strike, 0.0)
    scale = float(payoff.max()) or 1.0
    return ExpectationProblem(
        name="european_call",
        grid=grid,
        probabilities=probabilities,
        scaled_payoff=payoff / scale,
        payoff_scale=scale,
        post_factor=math.exp(-rate * maturity),
        meta={
            "model": "risk-neutral lognormal terminal price",
            "discretization": "equal-width bins in log price; exact bin mass; midpoint price",
            "truncation_std": truncation_std,
            "truncated_tail_mass": tail,
            "parameters": {
                "spot": spot,
                "strike": strike,
                "rate": rate,
                "volatility": volatility,
                "maturity": maturity,
            },
        },
    )


def gaussian_loss(
    mean: float, std: float, threshold: float, num_qubits: int, truncation_std: float
) -> ExpectationProblem:
    """Probability that a normally distributed return falls below ``-threshold``.

    The payoff is the plain event indicator evaluated at each bin's midpoint return, so the
    bin that straddles the threshold contributes a visible discretization error.
    """
    if not (std > 0 and threshold >= 0):
        raise ValueError("std must be positive and threshold non-negative")
    z, probabilities, tail = _standard_normal_bins(num_qubits, truncation_std)
    grid = mean + std * z
    return ExpectationProblem(
        name="loss_probability",
        grid=grid,
        probabilities=probabilities,
        scaled_payoff=(grid < -threshold).astype(float),
        payoff_scale=1.0,
        post_factor=1.0,
        meta={
            "model": "scalar normal portfolio return (physical-measure forecast)",
            "discretization": "equal-width bins in return; exact bin mass; midpoint indicator",
            "truncation_std": truncation_std,
            "truncated_tail_mass": tail,
            "parameters": {"mean": mean, "std": std, "threshold": threshold},
        },
    )
