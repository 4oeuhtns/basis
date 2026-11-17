"""The risk lab: one European call, priced by every method against clearly separated targets.

Two references are kept apart throughout (plan section 5):

* ``analytical_continuous``: Black-Scholes, the answer for the original continuous model.
* ``exact_discrete``: the exact weighted sum over the truncated grid that is loaded into the
  circuit. Monte Carlo and IQAE are judged against this number.

``model_error`` is their difference. It belongs to the grid, not to any estimator.
"""

from __future__ import annotations

import math
import time

import numpy as np

from qflab.methods.classical import black_scholes_call, monte_carlo_discrete, sobol_call
from qflab.methods.iqae import iqae
from qflab.methods.mlae import mlae, power_schedule
from qflab.problems.expectation import ExpectationProblem, lognormal_call

SIMULATED = "simulator performance: CUDA-Q statevector on a CPU; work is counted in ideal queries"
CLASSICAL = "classical computation on a CPU"


def _summarize(method, target, budget, budget_label, runs, truth, label) -> dict:
    """Collapse per-seed runs into one benchmark row; errors are in money (value) units."""
    points = [
        {
            "seed": run["seed"],
            "work": run["resources"]["work"],
            "estimate_value": run["estimate_value"],
            "abs_error": abs(run["estimate_value"] - truth),
            "half_width": (run["ci_value_high"] - run["ci_value_low"]) / 2,
            "covered": bool(run["ci_value_low"] <= truth <= run["ci_value_high"]),
            "elapsed_seconds": run["elapsed_seconds"],
        }
        for run in runs
    ]
    return {
        "method": method,
        "target": target,
        "budget": budget,
        "budget_label": budget_label,
        "seeds": len(points),
        "coverage": float(np.mean([p["covered"] for p in points])),
        "median_abs_error": float(np.median([p["abs_error"] for p in points])),
        "mean_half_width": float(np.mean([p["half_width"] for p in points])),
        "median_work": float(np.median([p["work"] for p in points])),
        "work_unit": runs[0]["resources"]["work_unit"],
        "median_elapsed_seconds": float(np.median([p["elapsed_seconds"] for p in points])),
        "evidence_label": label,
        "points": points,
    }


def _iqae_runs(problem: ExpectationProblem, sampler_factory, epsilon, alpha, shots, seeds) -> list:
    runs = []
    for seed in seeds:
        started = time.perf_counter()
        result = iqae(sampler_factory(problem, seed), epsilon=epsilon, alpha=alpha, shots=shots)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["seed"] = seed
        result["estimate_value"] = problem.to_value(result["estimate"])
        result["ci_value_low"] = problem.to_value(result["ci_low"])
        result["ci_value_high"] = problem.to_value(result["ci_high"])
        runs.append(result)
    return runs


def _mlae_runs(problem: ExpectationProblem, sampler_factory, circuits, alpha, shots, seeds) -> list:
    runs = []
    for seed in seeds:
        started = time.perf_counter()
        result = mlae(sampler_factory(problem, seed), power_schedule(circuits), shots, alpha)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["seed"] = seed
        result["estimate_value"] = problem.to_value(result["estimate"])
        result["ci_value_low"] = problem.to_value(result["ci_low"])
        result["ci_value_high"] = problem.to_value(result["ci_high"])
        runs.append(result)
    return runs


def _deepest(sampler, runs) -> dict:
    """Resources of the deepest circuit any seed executed."""
    k = max(run["resources"]["max_grover_power"] for run in runs)
    compiled = sampler.compiled_resources(k)
    return {
        **sampler.resources(k),
        "depth": compiled["depth"],
        "multi_qubit_depth": compiled["multi_qubit_depth"],
        "gate_counts": compiled["gate_counts"],
    }


def run_risk_lab(config: dict, target: str, seed: int) -> dict:
    from qflab.backends import cudaq_backend

    def sampler_factory(problem, sampler_seed, noise=None, sampler_target=target):
        return cudaq_backend.AmplitudeSampler(
            problem, seed=sampler_seed, target=sampler_target, noise=noise
        )

    started = time.perf_counter()
    call, cut = config["call"], config["truncation_std"]
    confidence = config["confidence"]
    alpha = 1 - confidence
    seeds = [seed + i for i in range(config["seeds"])]
    shots = config["iqae"]["shots"]
    analytical = black_scholes_call(**call)
    problem = lognormal_call(**call, num_qubits=config["reference_qubits"], truncation_std=cut)
    reference = {
        "analytical_continuous": analytical,
        "exact_discrete": problem.exact_value,
        "model_error": problem.exact_value - analytical,
        "model_error_meaning": "grid discretization plus tail truncation; not estimator error",
        "exact_amplitude": problem.exact_amplitude,
        "payoff_scale": problem.payoff_scale,
        "discount_factor": problem.post_factor,
        "scaling_round_trip_error": abs(
            problem.to_amplitude(problem.to_value(problem.exact_amplitude))
            - problem.exact_amplitude
        ),
        "probability_mass": float(problem.probabilities.sum()),
        "problem": problem.describe(),
        "grid": [float(v) for v in problem.grid],
        "probabilities": [float(v) for v in problem.probabilities],
        "scaled_payoff": [float(v) for v in problem.scaled_payoff],
    }

    # 1. The circuit really encodes the discrete expectation (privileged statevector check).
    checker = sampler_factory(problem, seed)
    theta = math.asin(math.sqrt(problem.exact_amplitude))
    verification = []
    for k in (0, 1, 2, 3):
        measured = checker.indicator_probability(k)
        predicted = math.sin((2 * k + 1) * theta) ** 2
        verification.append(
            {
                "grover_power": k,
                "circuit_probability": measured,
                "predicted_probability": predicted,
                "matches": abs(measured - predicted) < 1e-9,
            }
        )

    # 2. Estimators at several budgets and seeds.
    benchmark = []
    for samples in config["monte_carlo_samples"]:
        runs = [monte_carlo_discrete(problem, samples, s, confidence) for s in seeds]
        benchmark.append(
            _summarize(
                "monte_carlo",
                "discrete",
                samples,
                f"{samples:,} samples",
                runs,
                problem.exact_value,
                CLASSICAL,
            )
        )
    for epsilon in config["iqae"]["epsilons"]:
        runs = _iqae_runs(problem, sampler_factory, epsilon, alpha, shots, seeds)
        row = _summarize(
            "iqae",
            "discrete",
            epsilon,
            f"epsilon = {epsilon:g}",
            runs,
            problem.exact_value,
            SIMULATED,
        )
        row["target_half_width_value"] = problem.to_value(epsilon)
        row["max_grover_power"] = max(r["resources"]["max_grover_power"] for r in runs)
        row["median_shots"] = float(np.median([r["resources"]["shots"] for r in runs]))
        row["deepest_circuit"] = _deepest(checker, runs)
        row["example_rounds"] = runs[0]["rounds"]
        benchmark.append(row)
    # Maximum-likelihood amplitude estimation on the same circuits (plan section 14, item 3).
    for circuits in config.get("mlae", {}).get("circuits", []):
        runs = _mlae_runs(problem, sampler_factory, circuits, alpha, shots, seeds)
        row = _summarize(
            "mlae",
            "discrete",
            circuits,
            f"{circuits} circuits, powers up to {max(power_schedule(circuits))}",
            runs,
            problem.exact_value,
            SIMULATED,
        )
        row["max_grover_power"] = max(r["resources"]["max_grover_power"] for r in runs)
        row["median_shots"] = float(np.median([r["resources"]["shots"] for r in runs]))
        row["deepest_circuit"] = _deepest(checker, runs)
        benchmark.append(row)
    replicates = config["sobol"]["replicates"]
    for per_replicate in config["sobol"]["samples_per_replicate"]:
        runs = [
            {
                **sobol_call(
                    **call,
                    samples_per_replicate=per_replicate,
                    replicates=replicates,
                    seed=s,
                    confidence=confidence,
                ),
                "seed": s,
            }
            for s in seeds
        ]
        benchmark.append(
            _summarize(
                "sobol_qmc",
                "continuous",
                per_replicate * replicates,
                f"{per_replicate:,} x {replicates} scrambles",
                runs,
                analytical,
                CLASSICAL,
            )
        )

    # 3. Grid size: model error falls while the loading circuit grows.
    grid_epsilon = config.get(
        "sweep_epsilon", sorted(config["iqae"]["epsilons"])[len(config["iqae"]["epsilons"]) // 2]
    )
    grid_sweep = []
    for qubits in config["grid_qubits"]:
        build_started = time.perf_counter()
        sized = lognormal_call(**call, num_qubits=qubits, truncation_std=cut)
        sampler = sampler_factory(sized, seed)
        probability = sampler.indicator_probability(0)
        build_seconds = time.perf_counter() - build_started
        runs = _iqae_runs(sized, sampler_factory, grid_epsilon, alpha, shots, seeds)
        row = _summarize(
            "iqae",
            "discrete",
            grid_epsilon,
            f"epsilon = {grid_epsilon:g}",
            runs,
            sized.exact_value,
            SIMULATED,
        )
        preparation = sampler.resources(0)
        grid_sweep.append(
            {
                "scenario_qubits": qubits,
                "scenarios": 2**qubits,
                "exact_discrete": sized.exact_value,
                "model_error": sized.exact_value - analytical,
                "truncated_tail_mass": sized.meta["truncated_tail_mass"],
                "circuit_matches_discrete_sum": abs(probability - sized.exact_amplitude) < 1e-9,
                "build_and_verify_seconds": build_seconds,
                "state_preparation": {
                    "qubits": preparation["qubits"],
                    "rotations": preparation["rotations"],
                    "cnots": preparation["cnots_in_state_preparation"],
                    "depth": sampler.compiled_resources(0)["depth"],
                },
                "exact_sum_terms": 2**qubits,
                "iqae": {
                    k: row[k]
                    for k in (
                        "coverage",
                        "median_abs_error",
                        "mean_half_width",
                        "median_work",
                        "median_elapsed_seconds",
                    )
                },
                "target_half_width_value": sized.to_value(grid_epsilon),
                "deepest_circuit": _deepest(sampler, runs),
            }
        )

    # 4. Payoff parameter: the same comparison at other strikes, at matched work.
    strike_sweep = []
    for strike in config["strikes"]:
        changed = {**call, "strike": strike}
        sized = lognormal_call(**changed, num_qubits=config["reference_qubits"], truncation_std=cut)
        quantum = _summarize(
            "iqae",
            "discrete",
            grid_epsilon,
            "",
            _iqae_runs(sized, sampler_factory, grid_epsilon, alpha, shots, seeds),
            sized.exact_value,
            SIMULATED,
        )
        matched = max(2, int(quantum["median_work"]))
        classical = _summarize(
            "monte_carlo",
            "discrete",
            matched,
            "",
            [monte_carlo_discrete(sized, matched, s, confidence) for s in seeds],
            sized.exact_value,
            CLASSICAL,
        )
        strike_sweep.append(
            {
                "strike": strike,
                "analytical_continuous": black_scholes_call(**changed),
                "exact_discrete": sized.exact_value,
                "model_error": sized.exact_value - black_scholes_call(**changed),
                "exact_amplitude": sized.exact_amplitude,
                "matched_work": matched,
                "iqae": {
                    k: quantum[k] for k in ("coverage", "median_abs_error", "mean_half_width")
                },
                "monte_carlo": {
                    k: classical[k] for k in ("coverage", "median_abs_error", "mean_half_width")
                },
            }
        )

    # 5. Noise, as its own study on a small circuit.
    noise = config["noise"]
    small = lognormal_call(**call, num_qubits=noise["qubits"], truncation_std=cut)
    noise_seeds = seeds[: noise["seeds"]]
    noise_study = []
    for level in noise["two_qubit_depolarizing"]:
        settings = {"two_qubit_depolarizing": level} if level > 0 else None

        def noisy_factory(problem_, seed_, settings=settings):
            return sampler_factory(problem_, seed_, settings, "density-matrix-cpu")

        runs = _iqae_runs(small, noisy_factory, noise["epsilon"], alpha, shots, noise_seeds)
        row = _summarize(
            "iqae",
            "discrete",
            noise["epsilon"],
            "",
            runs,
            small.exact_value,
            "simulator performance: CUDA-Q density matrix with a depolarizing noise model",
        )
        noise_study.append(
            {
                "two_qubit_depolarizing": level,
                "scenario_qubits": noise["qubits"],
                "exact_discrete": small.exact_value,
                "mean_estimate": float(np.mean([p["estimate_value"] for p in row["points"]])),
                "coverage": row["coverage"],
                "median_abs_error": row["median_abs_error"],
                "mean_half_width": row["mean_half_width"],
                "max_grover_power": max(r["resources"]["max_grover_power"] for r in runs),
                "seeds": len(noise_seeds),
            }
        )

    return {
        "question": "How accurately and efficiently does each method estimate one expectation?",
        "settings": config,
        "target": target,
        "seed": seed,
        "reference": reference,
        "circuit_verification": verification,
        "benchmark": benchmark,
        "sweep_epsilon": grid_epsilon,
        "grid_sweep": grid_sweep,
        "strike_sweep": strike_sweep,
        "noise_study": noise_study,
        "noise_model_note": "depolarizing error after every CNOT and CZ; single-qubit gates, "
        "gates with two or more controls, and readout are noiseless, so this is optimistic",
        "counting_convention": "IQAE and MLAE work = shots x (2k + 1) applications of A or its "
        "inverse summed over rounds; one Monte Carlo sample = one scenario draw and payoff "
        "evaluation",
        "elapsed_seconds": time.perf_counter() - started,
    }
