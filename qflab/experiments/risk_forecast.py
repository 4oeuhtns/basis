"""Diagnostic risk forecasts for the portfolios a backtest actually targeted.

Question, for each saved decision: under the return model fitted to data available at the
cutoff, what is the probability that the target portfolio loses more than ``threshold`` over
the forecast horizon?

The estimate is attached to the decision record. It never changes a trade (plan section 8,
stage 1). The scalar normal model is deliberately easy: its exact answer is one normal CDF,
which makes it a verification target for the estimators, not a use case for them.
"""

from __future__ import annotations

import math
import time

import numpy as np

from qflab.methods.classical import monte_carlo_discrete, normal_loss_probability
from qflab.methods.iqae import iqae
from qflab.problems.expectation import gaussian_loss


def portfolio_moments(response: dict) -> tuple[float, float]:
    """Mean and standard deviation of the target portfolio's horizon return.

    Uses the forecast the solver itself used. Cash (the reserve) earns zero in this model.
    """
    symbols = response["canonical_model"]["symbols"]
    weights = np.array([float(response["target_weights"][s]) for s in symbols])
    mean = np.array([float(response["diagnostics"]["mean_returns"][s]) for s in symbols])
    covariance = np.array(response["diagnostics"]["covariance"], dtype=float)
    return float(weights @ mean), math.sqrt(float(weights @ covariance @ weights))


def run_risk_forecast(
    config: dict, records: list[dict], equity_rows: list[dict], sampler_factory, seed: int
) -> dict:
    """One row per feasible decision: analytical, exact-discrete, Monte Carlo and IQAE."""
    timestamps = [row["timestamp"] for row in equity_rows]
    equity = [float(row["equity"]) for row in equity_rows]
    rows = []
    for index, record in enumerate(records):
        response = record.get("response")
        if record["status"] != "feasible" or not response:
            rows.append({"data_cutoff": record["request"]["data_cutoff"], "status": "no_decision"})
            continue
        horizon = response["diagnostics"]["forecast_horizon_sessions"]
        mean, std = portfolio_moments(response)
        problem = gaussian_loss(
            mean, std, config["threshold"], config["qubits"], config["truncation_std"]
        )
        started = time.perf_counter()
        quantum = iqae(
            sampler_factory(problem, seed=seed + index),
            epsilon=config["epsilon"],
            alpha=config["alpha"],
            shots=config["shots"],
        )
        quantum_seconds = time.perf_counter() - started
        classical = monte_carlo_discrete(
            problem, config["monte_carlo_samples"], seed + index, 1 - config["alpha"]
        )
        cutoff = response["data_cutoff"]
        realized = None
        if cutoff in timestamps and timestamps.index(cutoff) + horizon < len(equity):
            start = timestamps.index(cutoff)
            realized = equity[start + horizon] / equity[start] - 1
        rows.append(
            {
                "data_cutoff": cutoff,
                "status": "ok",
                "selected": response["selected"],
                "horizon_sessions": horizon,
                "forecast_mean": mean,
                "forecast_std": std,
                "analytical_continuous": normal_loss_probability(mean, std, config["threshold"]),
                "exact_discrete": problem.exact_amplitude,
                "monte_carlo": {k: classical[k] for k in ("estimate", "ci_low", "ci_high")},
                "monte_carlo_work": classical["resources"]["work"],
                "iqae": {k: quantum[k] for k in ("estimate", "ci_low", "ci_high", "status")},
                "iqae_work": quantum["resources"]["state_preparation_calls"],
                "iqae_max_grover_power": quantum["resources"]["max_grover_power"],
                "iqae_seconds": quantum_seconds,
                "iqae_covers_discrete": quantum["ci_low"]
                <= problem.exact_amplitude
                <= quantum["ci_high"],
                "realized_return": realized,
                "realized_loss_event": None
                if realized is None
                else realized < -config["threshold"],
            }
        )
    done = [r for r in rows if r["status"] == "ok"]
    scored = [r for r in done if r["realized_return"] is not None]
    return {
        "question": f"P(target portfolio return < -{config['threshold']:.0%}) over the forecast "
        "horizon, under the model fitted at each decision cutoff",
        "settings": config,
        "model": "scalar normal horizon return from the solver's own mean and covariance; "
        "cash reserve earns zero",
        "role": "diagnostic only; these numbers do not change any trade",
        "rows": rows,
        "summary": {
            "decisions": len(done),
            "iqae_covers_discrete": sum(r["iqae_covers_discrete"] for r in done),
            "largest_discretization_error": max(
                (abs(r["exact_discrete"] - r["analytical_continuous"]) for r in done), default=None
            ),
            "scored_decisions": len(scored),
            "realized_loss_events": sum(bool(r["realized_loss_event"]) for r in scored),
            "mean_forecast_probability": float(
                np.mean([r["analytical_continuous"] for r in scored])
            )
            if scored
            else None,
            "calibration_note": "a handful of overlapping monthly windows cannot validate "
            "calibration; shown for inspection only",
        },
        "realized_convention": "account equity change from the cutoff close to the close "
        "`horizon` sessions later; the account trades whole units one session after the cutoff, "
        "so it only approximates the target portfolio",
    }
