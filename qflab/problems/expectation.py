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


def _scaled(payoff: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(payoff.max()) or 1.0
    return payoff / scale, scale


def two_factor_loss(
    weights: tuple[float, float],
    vols: tuple[float, float],
    correlation: float,
    option_strike: float,
    option_notional: float,
    threshold: float,
    qubits_per_factor: int,
    truncation_std: float,
    payoff: str = "indicator",
) -> ExpectationProblem:
    """Tail risk of a two-asset book with a short call: correlated factors, nonlinear exposure.

    Horizon returns ``r_i = vols[i] * z_i`` with correlated standard normals. The book holds
    ``weights`` in the two assets and is short ``option_notional`` of a call on asset 1 struck
    at return ``option_strike``, so its profit is

        pnl = w1 r1 + w2 r2 - option_notional * max(r1 - option_strike, 0).

    ``payoff="indicator"`` targets P(pnl < -threshold); ``"excess"`` targets the expected loss
    beyond the threshold, E[max(-pnl - threshold, 0)]. Scenario index = i + 2^n * j for bin
    ``i`` of factor 1 and bin ``j`` of factor 2.
    """
    if not -1 < correlation < 1 or payoff not in ("indicator", "excess"):
        raise ValueError("Need |correlation| < 1 and payoff 'indicator' or 'excess'")
    z, mass, tail = _standard_normal_bins(qubits_per_factor, truncation_std)
    edges = np.linspace(-truncation_std, truncation_std, len(z) + 1)
    # Second factor conditional on the first factor's bin midpoint.
    spread = math.sqrt(1 - correlation**2)
    conditional = np.diff(norm.cdf((edges[None, :] - correlation * z[:, None]) / spread), axis=1)
    conditional /= conditional.sum(axis=1, keepdims=True)
    joint = mass[:, None] * conditional  # [i, j]
    r1, r2 = vols[0] * z[:, None], vols[1] * z[None, :]
    pnl = weights[0] * r1 + weights[1] * r2 - option_notional * np.maximum(r1 - option_strike, 0.0)
    raw = (
        (pnl < -threshold).astype(float)
        if payoff == "indicator"
        else np.maximum(-pnl - threshold, 0.0)
    )
    scaled, scale = _scaled(raw.T.reshape(-1))
    return ExpectationProblem(
        name=f"two_factor_{payoff}",
        grid=pnl.T.reshape(-1),
        probabilities=joint.T.reshape(-1),
        scaled_payoff=scaled,
        payoff_scale=scale,
        post_factor=1.0,
        meta={
            "model": "two correlated normal factors; linear book plus a short call on asset 1",
            "discretization": "equal-width bins per factor; second factor conditional on the "
            "first factor's bin midpoint; payoff at bin midpoints",
            "truncation_std": truncation_std,
            "truncated_tail_mass": tail,
            "parameters": {
                "weights": list(weights),
                "vols": list(vols),
                "correlation": correlation,
                "option_strike": option_strike,
                "option_notional": option_notional,
                "threshold": threshold,
                "payoff": payoff,
            },
        },
    )


def two_factor_payoff(parameters: dict):
    """The continuous payoff of ``two_factor_loss`` as a function of independent normals."""
    w, vols, rho = parameters["weights"], parameters["vols"], parameters["correlation"]

    def payoff(z: np.ndarray) -> np.ndarray:
        r1 = vols[0] * z[:, 0]
        r2 = vols[1] * (rho * z[:, 0] + math.sqrt(1 - rho**2) * z[:, 1])
        pnl = (
            w[0] * r1
            + w[1] * r2
            - parameters["option_notional"] * np.maximum(r1 - parameters["option_strike"], 0.0)
        )
        if parameters["payoff"] == "indicator":
            return (pnl < -parameters["threshold"]).astype(float)
        return np.maximum(-pnl - parameters["threshold"], 0.0)

    return payoff


def asian_call(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int,
    qubits_per_step: int,
    truncation_std: float,
) -> ExpectationProblem:
    """Arithmetic-average Asian call: a path-dependent payoff with no closed form.

    The price is monitored at ``steps`` equally spaced dates. Each period's log return is an
    independent normal, discretized on its own register; the scenario is the whole path, so the
    circuit uses ``steps * qubits_per_step`` scenario qubits. Step ``s`` occupies bits
    ``s * qubits_per_step`` and up of the scenario index.
    """
    if steps < 2 or steps * qubits_per_step > 10:
        raise ValueError("Need at least two steps and at most 10 scenario qubits")
    z, mass, tail = _standard_normal_bins(qubits_per_step, truncation_std)
    dt = maturity / steps
    increment = (rate - volatility**2 / 2) * dt + volatility * math.sqrt(dt) * z
    size = len(z)
    index = np.arange(size**steps)
    log_price = np.full(len(index), math.log(spot))
    probability = np.ones(len(index))
    average = np.zeros(len(index))
    for step in range(steps):
        digit = index // size**step % size
        log_price = log_price + increment[digit]
        probability = probability * mass[digit]
        average += np.exp(log_price) / steps
    scaled, scale = _scaled(np.maximum(average - strike, 0.0))
    return ExpectationProblem(
        name="asian_call",
        grid=average,
        probabilities=probability / probability.sum(),
        scaled_payoff=scaled,
        payoff_scale=scale,
        post_factor=math.exp(-rate * maturity),
        meta={
            "model": "risk-neutral lognormal path; arithmetic average over the monitoring dates",
            "discretization": "equal-width bins per period log return; product measure",
            "truncation_std": truncation_std,
            "truncated_tail_mass": 1 - (1 - tail) ** steps,
            "parameters": {
                "spot": spot,
                "strike": strike,
                "rate": rate,
                "volatility": volatility,
                "maturity": maturity,
                "steps": steps,
            },
        },
    )


def asian_payoff(parameters: dict):
    """The continuous (undiscounted) Asian payoff as a function of independent normals."""
    steps, dt = parameters["steps"], parameters["maturity"] / parameters["steps"]
    drift = (parameters["rate"] - parameters["volatility"] ** 2 / 2) * dt
    spread = parameters["volatility"] * math.sqrt(dt)

    def payoff(z: np.ndarray) -> np.ndarray:
        prices = parameters["spot"] * np.exp(np.cumsum(drift + spread * z[:, :steps], axis=1))
        return np.maximum(prices.mean(axis=1) - parameters["strike"], 0.0)

    return payoff
