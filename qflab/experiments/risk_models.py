"""Harder expectation problems: correlated factors, a nonlinear book, a path-dependent payoff.

Plan section 8 ("then investigate a specified nonlinear or multifactor exposure") and section
14, item 5. Each problem has no one-line closed form, so the continuous reference is a
scrambled Sobol estimate with its own interval, and the exact discrete sum remains the target
for Monte Carlo, IQAE and MLAE. The scenario distribution is still loaded densely; a
structured loader for the product-form path model is not built.
"""

from __future__ import annotations

import time

from qflab.experiments.risk_lab import (
    CLASSICAL,
    SIMULATED,
    _deepest,
    _iqae_runs,
    _mlae_runs,
    _summarize,
)
from qflab.methods.classical import monte_carlo_discrete, sobol_normal
from qflab.problems.expectation import asian_call, asian_payoff, two_factor_loss, two_factor_payoff

KEEP = ("coverage", "median_abs_error", "mean_half_width", "median_work", "median_elapsed_seconds")


def _build(spec: dict):
    kind = spec["kind"]
    settings = {k: v for k, v in spec.items() if k not in ("kind", "label")}
    if kind == "two_factor":
        settings["weights"], settings["vols"] = tuple(settings["weights"]), tuple(settings["vols"])
        problem = two_factor_loss(**settings)
        return problem, two_factor_payoff(problem.meta["parameters"]), 2
    if kind == "asian_call":
        problem = asian_call(**settings)
        return problem, asian_payoff(problem.meta["parameters"]), settings["steps"]
    raise ValueError(f"Unknown risk model kind: {kind}")


def run_risk_models(config: dict, target: str, seed: int) -> dict:
    from qflab.backends import cudaq_backend

    def sampler_factory(problem, sampler_seed):
        return cudaq_backend.AmplitudeSampler(problem, seed=sampler_seed, target=target)

    started = time.perf_counter()
    alpha = 1 - config["confidence"]
    seeds = [seed + i for i in range(config["seeds"])]
    shots = config["shots"]
    rows = []
    for spec in config["models"]:
        problem, payoff, dimension = _build(spec)
        continuous = sobol_normal(
            payoff,
            dimension,
            config["sobol"]["samples_per_replicate"],
            config["sobol"]["replicates"],
            seed,
            config["confidence"],
            problem.post_factor,
        )
        checker = sampler_factory(problem, seed)
        circuit_probability = checker.indicator_probability(0)
        quantum_runs = _iqae_runs(problem, sampler_factory, config["epsilon"], alpha, shots, seeds)
        quantum = _summarize(
            "iqae", "discrete", config["epsilon"], "", quantum_runs, problem.exact_value, SIMULATED
        )
        likelihood_runs = _mlae_runs(
            problem, sampler_factory, config["mlae_circuits"], alpha, shots, seeds
        )
        likelihood = _summarize(
            "mlae",
            "discrete",
            config["mlae_circuits"],
            "",
            likelihood_runs,
            problem.exact_value,
            SIMULATED,
        )
        matched = max(2, int(quantum["median_work"]))
        classical = _summarize(
            "monte_carlo",
            "discrete",
            matched,
            "",
            [monte_carlo_discrete(problem, matched, s, config["confidence"]) for s in seeds],
            problem.exact_value,
            CLASSICAL,
        )
        preparation = checker.resources(0)
        rows.append(
            {
                "label": spec["label"],
                "kind": spec["kind"],
                "problem": problem.describe(),
                "unit": "probability" if problem.payoff_scale == 1.0 else "value",
                "exact_discrete": problem.exact_value,
                "continuous_reference": {
                    "estimate": continuous["estimate_value"],
                    "ci_low": continuous["ci_value_low"],
                    "ci_high": continuous["ci_value_high"],
                    "samples": continuous["resources"]["samples"],
                    "method": "scrambled Sobol on the continuous model",
                },
                "model_error": problem.exact_value - continuous["estimate_value"],
                "circuit_matches_discrete_sum": abs(circuit_probability - problem.exact_amplitude)
                < 1e-9,
                "state_preparation": {
                    "qubits": preparation["qubits"],
                    "rotations": preparation["rotations"],
                    "cnots": preparation["cnots_in_state_preparation"],
                },
                "target_half_width_value": problem.to_value(config["epsilon"]),
                "matched_work": matched,
                "iqae": {k: quantum[k] for k in KEEP},
                "mlae": {k: likelihood[k] for k in KEEP},
                "monte_carlo": {k: classical[k] for k in KEEP},
                "deepest_circuit": _deepest(checker, quantum_runs + likelihood_runs),
            }
        )
    return {
        "question": "Do the estimators behave the same on correlated, nonlinear and "
        "path-dependent problems, and what does loading those scenarios cost?",
        "settings": config,
        "target": target,
        "seed": seed,
        "rows": rows,
        "loading_note": "every scenario distribution is loaded densely (about 2 x 2^n "
        "rotations); the Asian path model is a product measure and could be loaded register by "
        "register, which is not built",
        "evidence_label": SIMULATED,
        "elapsed_seconds": time.perf_counter() - started,
    }
