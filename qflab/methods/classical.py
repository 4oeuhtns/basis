"""Classical references and estimators for expectation problems.

Every result names its *target*: ``discrete`` (the truncated grid that is loaded into the
circuit) or ``continuous`` (the original model). Their difference is model error, and must not
be attributed to an estimator.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.stats import norm, qmc
from scipy.stats import t as student_t

from qflab.problems.expectation import ExpectationProblem


def black_scholes_call(
    spot: float, strike: float, rate: float, volatility: float, maturity: float
) -> float:
    """Continuous-model analytical reference."""
    spread = volatility * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + volatility**2 / 2) * maturity) / spread
    return float(spot * norm.cdf(d1) - strike * math.exp(-rate * maturity) * norm.cdf(d1 - spread))


def normal_loss_probability(mean: float, std: float, threshold: float) -> float:
    """P(return < -threshold) for a normal return; the continuous analytical reference."""
    return float(norm.cdf((-threshold - mean) / std))


def monte_carlo_discrete(
    problem: ExpectationProblem, samples: int, seed: int, confidence: float
) -> dict:
    """Plain Monte Carlo on the discrete model: draw scenarios, average the scaled payoff."""
    if samples < 2:
        raise ValueError("Monte Carlo needs at least two samples")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    draws = problem.scaled_payoff[
        rng.choice(len(problem.grid), size=samples, p=problem.probabilities)
    ]
    estimate = float(draws.mean())
    half = float(norm.ppf(0.5 + confidence / 2) * draws.std(ddof=1) / math.sqrt(samples))
    low, high = max(0.0, estimate - half), min(1.0, estimate + half)
    return {
        "method": "monte_carlo",
        "target": "discrete",
        "status": "ok",
        "confidence": confidence,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "estimate_value": problem.to_value(estimate),
        "ci_value_low": problem.to_value(low),
        "ci_value_high": problem.to_value(high),
        "interval": "normal approximation with the sample standard deviation",
        "resources": {"samples": samples, "work": samples, "work_unit": "payoff evaluations"},
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
    }


def sobol_call(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    samples_per_replicate: int,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict:
    """Randomized (scrambled) Sobol quasi-Monte Carlo on the *continuous* lognormal model.

    The interval is a Student-t interval across independently scrambled replicates.
    """
    if samples_per_replicate < 2 or samples_per_replicate & (samples_per_replicate - 1):
        raise ValueError("samples_per_replicate must be a power of two")
    if replicates < 2:
        raise ValueError("At least two scrambled replicates are needed for an interval")
    started = time.perf_counter()
    log_mean = math.log(spot) + (rate - volatility**2 / 2) * maturity
    log_std = volatility * math.sqrt(maturity)
    means = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=1, scramble=True, seed=np.random.default_rng([seed, replicate]))
        uniform = sampler.random_base2(int(math.log2(samples_per_replicate)))[:, 0]
        uniform = np.clip(uniform, 1e-16, 1 - 1e-16)
        prices = np.exp(log_mean + log_std * norm.ppf(uniform))
        means.append(math.exp(-rate * maturity) * np.maximum(prices - strike, 0.0).mean())
    means = np.array(means)
    estimate = float(means.mean())
    half = float(
        student_t.ppf(0.5 + confidence / 2, replicates - 1)
        * means.std(ddof=1)
        / math.sqrt(replicates)
    )
    total = samples_per_replicate * replicates
    return {
        "method": "sobol_qmc",
        "target": "continuous",
        "status": "ok",
        "confidence": confidence,
        "estimate_value": estimate,
        "ci_value_low": estimate - half,
        "ci_value_high": estimate + half,
        "interval": f"Student-t across {replicates} independent scrambles",
        "resources": {"samples": total, "work": total, "work_unit": "payoff evaluations"},
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
    }
