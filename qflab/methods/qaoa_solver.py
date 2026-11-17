"""QAOA portfolio selection as a search loop around a bitstring sampler.

The financial problem arrives as the exact solver's canonical integer model. This module adds
the cardinality penalty, translates to Ising form, tunes the circuit parameters using
*sampled* energies, draws a final sample, and then scores every sampled bitstring with the
original financial model. It never trusts the circuit's own cost.

Parameter search, the same for every decision:

1. Depth 1: scan a ``grid`` x ``grid`` lattice of (gamma, beta), then refine the best point
   with COBYLA for at most ``maxiter`` evaluations.
2. Each further layer starts from the previous layer's parameters, linearly interpolated
   (the INTERP heuristic of Zhou et al., Phys. Rev. X 10, 021067), and is refined the same way.

Every evaluation costs ``shots`` circuit executions and is counted.

Selection rule, fixed before any result was inspected: among the bitstrings of the final
sample that select exactly K funds, take the lowest financial objective; ties break
lexicographically. If the final sample contains no such bitstring the status is
``no_feasible_sample`` and no portfolio is returned. Nothing is repaired.

The sampler is injected, so this file imports no quantum SDK.
"""

from __future__ import annotations

import itertools
import time
from functools import partial

import numpy as np
from scipy.optimize import minimize

from qflab.optimizer_worker import energy
from qflab.qaoa import compile_qubo, summarize_samples, verify_qubo

SOLVER_FIELDS = {
    "method": str,
    "layers": int,
    "shots": int,
    "final_shots": int,
    "maxiter": int,
    "grid": int,
    "seed": int,
    "penalty_scale": str,
    "target": str,
}


def validate_solver(settings: dict) -> dict:
    if set(settings) != set(SOLVER_FIELDS) or any(
        type(settings[name]) is not kind for name, kind in SOLVER_FIELDS.items()
    ):
        raise ValueError(f"Solver settings must be exactly {sorted(SOLVER_FIELDS)}")
    if settings["method"] != "qaoa_cudaq":
        raise ValueError("Unsupported solver method")
    limits = {
        "layers": (1, 5),
        "shots": (100, 100_000),
        "final_shots": (100, 1_000_000),
        "maxiter": (1, 500),
        "grid": (2, 64),
        "seed": (0, 2**31 - 1),
    }
    for name, (low, high) in limits.items():
        if not low <= settings[name] <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    return settings


def ising_from_qubo(qubo: dict) -> tuple[float, list[float], list[tuple[int, int, float]]]:
    """Substitute x = (1 - z) / 2 and divide by the QUBO's normalization.

    Returns ``(constant, field, couplings)`` for
    ``E(z) = constant + sum_i field[i] z_i + sum (a, b, J) J z_a z_b``.
    """
    symbols = qubo["symbols"]
    position = {s: i for i, s in enumerate(symbols)}
    divisor = int(qubo["normalization_integer"])
    linear = [int(qubo["linear"][s]) for s in symbols]
    field = [-value / 2 for value in linear]
    constant = int(qubo["constant"]) + sum(linear) / 2
    couplings = []
    for pair in qubo["pairs"]:
        a, b, q = position[pair["left"]], position[pair["right"]], int(pair["coefficient"])
        constant += q / 4
        field[a] -= q / 4
        field[b] -= q / 4
        if q:
            couplings.append((a, b, q / 4 / divisor))
    return constant / divisor, [value / divisor for value in field], couplings


def interpolate(parameters: list[float]) -> list[float]:
    """INTERP: stretch optimized depth-p angles into a starting point for depth p + 1."""
    depth = len(parameters) // 2

    def stretch(values):
        padded = [0.0, *values, 0.0]
        return [
            i / depth * padded[i] + (depth - i) / depth * padded[i + 1] for i in range(depth + 1)
        ]

    return stretch(parameters[:depth]) + stretch(parameters[depth:])


def solve_qaoa(
    reference: dict,
    settings: dict,
    sampler_factory,
    decision_seed: int,
    mixer: str = "x",
    initial_parameters: list[float] | None = None,
    refine: bool = True,
) -> dict:
    """Run one QAOA decision. ``reference`` is the exact solver's response for the same input.

    ``mixer="xy"`` selects the constraint-preserving variant: the cost is the financial
    objective without a penalty and ``sampler_factory`` must start from the K-hot subspace.
    ``initial_parameters`` (one gamma and one beta per layer) replaces the grid scan and the
    layer-by-layer build-up with a warm start; ``refine=False`` then skips optimization
    entirely and only draws the final sample.
    """
    settings = validate_solver(dict(settings))
    if mixer not in ("x", "xy"):
        raise ValueError("mixer must be 'x' or 'xy'")
    started = time.perf_counter()
    model = reference["canonical_model"]
    symbols, k = model["symbols"], model["select_count"]
    n = len(symbols)
    penalized = compile_qubo(model, settings["penalty_scale"])
    audit = verify_qubo(model, penalized, require_dominance=False)
    if mixer == "x":
        qubo = penalized
    else:
        # No penalty: the mixer cannot leave the feasible subspace. Normalize by the largest
        # coefficient so that gamma has a comparable scale.
        largest = max(
            [abs(int(v)) for v in model["linear"].values()]
            + [abs(int(p["coefficient"])) for p in model["pairs"]]
            + [1]
        )
        qubo = {**model, "normalization_integer": str(largest)}
        sampler_factory = partial(sampler_factory, select_count=k)
    _, field, couplings = ising_from_qubo(qubo)
    divisor = int(qubo["normalization_integer"])
    # Normalized QUBO energy of every bitstring; keys list qubit 0 (first symbol) first.
    table = {
        "".join(map(str, bits)): energy(qubo, [s for s, b in zip(symbols, bits, strict=True) if b])
        / divisor
        for bits in itertools.product((0, 1), repeat=n)
    }
    rng = np.random.default_rng([settings["seed"], decision_seed])
    seed = int(rng.integers(1, 2**31 - 1))
    trace: list[float] = []
    stages: list[dict] = []
    optimize_started = time.perf_counter()
    parameters: list[float] = []
    sampler = None
    if initial_parameters is not None and len(initial_parameters) != 2 * settings["layers"]:
        raise ValueError("initial_parameters needs one gamma and one beta per layer")
    first_depth = 1 if initial_parameters is None else settings["layers"]
    for depth in range(first_depth, settings["layers"] + 1):
        sampler = sampler_factory(
            field, couplings, layers=depth, seed=seed + depth, target=settings["target"]
        )

        def objective(point, sampler=sampler) -> float:
            counts = sampler(point, settings["shots"])
            value = sum(table[bits] * c for bits, c in counts.items()) / settings["shots"]
            trace.append(value)
            return value

        before = len(trace)
        if initial_parameters is not None:
            start = [float(v) for v in initial_parameters]
            if not refine:
                parameters = start
                stages.append({"layers": depth, "evaluations": 0, "sampled_energy": None})
                continue
        elif depth == 1:
            # The cost is close to periodic in gamma (integer penalty levels), so one period
            # of each angle is scanned at cell midpoints.
            axis = [(i + 0.5) * np.pi / settings["grid"] for i in range(settings["grid"])]
            _, start = min((objective([g, b]), [g, b]) for g in axis for b in axis)
        else:
            start = interpolate(parameters)
        found = minimize(
            objective,
            start,
            method="COBYLA",
            options={"maxiter": settings["maxiter"], "rhobeg": 0.1},
        )
        parameters = [float(v) for v in found.x]
        stages.append(
            {
                "layers": depth,
                "evaluations": len(trace) - before,
                "sampled_energy": float(found.fun),
            }
        )
    best_value = stages[-1]["sampled_energy"]
    optimize_seconds = time.perf_counter() - optimize_started

    sample_started = time.perf_counter()
    counts = sampler(parameters, settings["final_shots"])
    sample_seconds = time.perf_counter() - sample_started
    records = [
        {"solution": {"x": [int(b) for b in bits]}, "probability": count / settings["final_shots"]}
        for bits, count in sorted(counts.items())
    ]
    summary = summarize_samples(reference, records)

    # Privileged simulator access: the exact output distribution, free of shot noise.
    probabilities = sampler.probabilities(parameters)
    optimum = int(reference["objective_integer"])
    feasible_mass = optimal_mass = 0.0
    for index, probability in enumerate(probabilities):
        chosen = [symbols[i] for i in range(n) if index >> i & 1]
        if len(chosen) == k:
            feasible_mass += float(probability)
            if energy(model, chosen) == optimum:
                optimal_mass += float(probability)
    compiled = sampler.compiled_resources()
    return {
        "status": summary["status"],
        "selected": summary["best_selected"],
        "objective_integer": summary["best_financial_objective_integer"],
        "objective_gap_integer": summary["objective_gap_integer"],
        "objective_scale": model["scale"],
        "exact_reference": {
            "selected": reference["selected"],
            "objective_integer": reference["objective_integer"],
            "note": "answer key from exhaustive enumeration; never used to choose the portfolio",
        },
        "feasible_probability": summary["feasible_probability"],
        "optimal_probability": summary["optimal_probability"],
        "exact_distribution": {
            "feasible_probability": feasible_mass,
            "optimal_probability": optimal_mass,
            "note": "privileged simulator statevector at the final parameters",
        },
        "uniform_baseline": {
            "feasible_probability": audit["uniform_feasible_probability"],
            "optimal_probability": audit["uniform_optimal_probability"],
        },
        "mixer": mixer,
        "start": "cold" if initial_parameters is None else ("warm" if refine else "frozen"),
        "penalty": {
            "scale": settings["penalty_scale"] if mixer == "x" else None,
            "penalty_integer": penalized["penalty_integer"] if mixer == "x" else "0",
            "all_infeasible_states_are_worse": audit["all_infeasible_states_are_worse"]
            if mixer == "x"
            else None,
        },
        "parameters": parameters,
        "best_sampled_energy": best_value,
        "stages": stages,
        "trace": trace,
        "counts": dict(sorted(counts.items())),
        "resources": {
            "qubits": compiled["qubits"],
            "gates": compiled["gates"],
            "two_qubit_gates": compiled["multi_qubit_gates"],
            "depth": compiled["depth"],
            "layers": settings["layers"],
            "objective_evaluations": len(trace),
            "shots_per_evaluation": settings["shots"],
            "final_shots": settings["final_shots"],
            "total_shots": len(trace) * settings["shots"] + settings["final_shots"],
            "basis": "h, rz, rx, cx; all-to-all connectivity; no routing"
            if mixer == "x"
            else "Dicke preparation (cx, controlled ry), rz, cx, ring XY mixer; no routing",
        },
        "timings": {
            "optimize_seconds": optimize_seconds,
            "final_sample_seconds": sample_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "settings": settings,
    }
