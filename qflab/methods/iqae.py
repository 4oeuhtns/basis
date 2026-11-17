"""Iterative quantum amplitude estimation (IQAE), independent of any quantum SDK.

Reference: Grinko, Gacon, Zoufal and Woerner, "Iterative quantum amplitude estimation",
npj Quantum Information 7, 52 (2021).

The algorithm only needs a *sampler*: ``sampler(k, shots)`` runs the circuit ``Q^k A|0>`` for
``shots`` shots and returns how many times the indicator qubit read 1. That probability is
``sin^2((2k+1) * theta)`` with ``a = sin^2(theta)``. A CUDA-Q sampler runs real circuits;
``IdealSampler`` draws from the same binomial law directly, for tests and query-count studies.

Counting convention (documented because libraries differ):

* ``grover_calls`` = sum over rounds of ``shots * k``; one call is one application of Q.
* ``state_preparation_calls`` = sum of ``shots * (2k + 1)``; every Q contains A and A-inverse,
  and each shot starts with one A. This is the fair "work" number to set against classical
  samples, because one classical sample costs one evaluation of the scenario and payoff.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta


def clopper_pearson(ones: int, shots: int, alpha: float) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval at level ``1 - alpha``."""
    low = 0.0 if ones == 0 else float(beta.ppf(alpha / 2, ones, shots - ones + 1))
    high = 1.0 if ones == shots else float(beta.ppf(1 - alpha / 2, ones + 1, shots - ones))
    return low, high


def find_next_k(
    k: int, upper_half_circle: bool, theta_interval: tuple[float, float], min_ratio: float
) -> tuple[int, bool]:
    """Largest Grover power whose scaled interval still sits inside one half circle.

    ``theta_interval`` is in units of a full turn: ``theta = theta_a / (2 * pi)``, within
    [0, 1/4]. Returns the previous ``k`` when no power grows the scaling by ``min_ratio``.
    """
    theta_l, theta_u = theta_interval
    old_scaling = 4 * k + 2
    max_scaling = int(1 / (2 * (theta_u - theta_l)))
    scaling = max_scaling - (max_scaling - 2) % 4
    while scaling >= min_ratio * old_scaling:
        theta_min = scaling * theta_l - int(scaling * theta_l)
        theta_max = scaling * theta_u - int(scaling * theta_u)
        if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
            return int((scaling - 2) / 4), True
        if theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
            return int((scaling - 2) / 4), False
        scaling -= 4
    return int(k), upper_half_circle


def iqae(
    sampler,
    epsilon: float,
    alpha: float,
    shots: int,
    min_ratio: float = 2.0,
    max_state_preparation_calls: int | None = None,
) -> dict:
    """Estimate an amplitude to half-width ``epsilon`` with confidence ``1 - alpha``.

    Returns the estimate, its interval, every round, and the work spent. ``status`` is ``ok``
    or ``budget_exhausted``; in the latter case the interval is still valid, only wider.
    """
    if not 0 < epsilon <= 0.5 or not 0 < alpha < 1 or shots < 1 or min_ratio <= 1:
        raise ValueError("Need 0 < epsilon <= 0.5, 0 < alpha < 1, shots >= 1, min_ratio > 1")
    max_rounds = int(math.log(min_ratio * math.pi / 8 / epsilon) / math.log(min_ratio)) + 1
    round_alpha = alpha / max_rounds  # union bound over the rounds that change k
    theta_l, theta_u = 0.0, 0.25
    k, upper = 0, True
    pooled_ones = pooled_shots = 0
    rounds: list[dict] = []
    grover_calls = preparation_calls = 0
    status = "ok"
    while theta_u - theta_l > epsilon / math.pi:
        next_k, upper = find_next_k(k, upper, (theta_l, theta_u), min_ratio)
        if next_k != k or not rounds:
            pooled_ones = pooled_shots = 0  # a new power starts a fresh binomial sample
        k = next_k
        cost = shots * (2 * k + 1)
        if (
            max_state_preparation_calls is not None
            and preparation_calls + cost > max_state_preparation_calls
        ):
            status = "budget_exhausted"
            break
        ones = int(sampler(k, shots))
        if not 0 <= ones <= shots:
            raise ValueError("Sampler returned an impossible count")
        grover_calls += shots * k
        preparation_calls += cost
        pooled_ones += ones
        pooled_shots += shots
        a_min, a_max = clopper_pearson(pooled_ones, pooled_shots, round_alpha)
        if upper:
            fraction_min = math.acos(1 - 2 * a_min) / (2 * math.pi)
            fraction_max = math.acos(1 - 2 * a_max) / (2 * math.pi)
        else:
            fraction_min = 1 - math.acos(1 - 2 * a_max) / (2 * math.pi)
            fraction_max = 1 - math.acos(1 - 2 * a_min) / (2 * math.pi)
        scaling = 4 * k + 2
        # As in the paper, the new interval replaces the old one. Intersecting them instead
        # can produce an empty interval whenever one round's confidence interval misses.
        theta_l, theta_u = (
            (int(scaling * theta_l) + fraction_min) / scaling,
            (int(scaling * theta_u) + fraction_max) / scaling,
        )
        rounds.append(
            {
                "k": k,
                "shots": shots,
                "ones": ones,
                "pooled_ones": pooled_ones,
                "pooled_shots": pooled_shots,
                "a_low": math.sin(2 * math.pi * theta_l) ** 2,
                "a_high": math.sin(2 * math.pi * theta_u) ** 2,
            }
        )
    low = math.sin(2 * math.pi * theta_l) ** 2
    high = math.sin(2 * math.pi * theta_u) ** 2
    return {
        "method": "iqae",
        "target": "discrete",
        "status": status,
        "confidence": 1 - alpha,
        "epsilon_target": epsilon,
        "estimate": (low + high) / 2,
        "ci_low": low,
        "ci_high": high,
        "interval": "Clopper-Pearson per round, union bound over rounds",
        "rounds": rounds,
        "resources": {
            "shots": sum(r["shots"] for r in rounds),
            "rounds": len(rounds),
            "grover_calls": grover_calls,
            "state_preparation_calls": preparation_calls,
            "max_grover_power": max((r["k"] for r in rounds), default=0),
            "work": preparation_calls,
            "work_unit": "applications of A or its inverse",
        },
    }


class IdealSampler:
    """The noiseless measurement law of ``Q^k A|0>``, without simulating a circuit."""

    def __init__(self, amplitude: float, seed: int) -> None:
        if not 0 <= amplitude <= 1:
            raise ValueError("Amplitude must lie in [0, 1]")
        self.theta = math.asin(math.sqrt(amplitude))
        self.rng = np.random.default_rng(seed)

    def __call__(self, k: int, shots: int) -> int:
        return int(self.rng.binomial(shots, math.sin((2 * k + 1) * self.theta) ** 2))
