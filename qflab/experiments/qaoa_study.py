"""QAOA optimization study on saved portfolio decisions.

Holds the forecast inputs, objective and constraints fixed (they come from the saved requests
of an exact-solver backtest) and changes only the search method and its budget. Realized
trading performance is deliberately absent: that is a different study (plan section 9).
"""

from __future__ import annotations

import itertools
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from qflab.methods.classical_portfolio import swap_search
from qflab.optimizer_worker import energy, solve_request


def _run_one(job: dict) -> dict:
    """One QAOA solve in a worker process; the simulator is imported there."""
    from qflab.backends import cudaq_backend
    from qflab.methods.qaoa_solver import solve_qaoa

    reference = solve_request(job["request"], job["request_sha256"])
    scale = int(reference["canonical_model"]["scale"])
    try:
        result = solve_qaoa(
            reference,
            job["settings"],
            cudaq_backend.QaoaSampler,
            int(job["request_sha256"][:8], 16),
        )
    except Exception as error:  # a failed run is a result, and stays in the report
        return {**job["key"], "status": "failed", "error": f"{type(error).__name__}: {error}"}
    gap = result["objective_gap_integer"]
    return {
        **job["key"],
        "status": result["status"],
        "selected": result["selected"],
        "gap": None if gap is None else int(gap) / scale,
        "found_optimum": gap == "0",
        "sampled_feasible_probability": result["feasible_probability"],
        "sampled_optimal_probability": result["optimal_probability"],
        "feasible_probability": result["exact_distribution"]["feasible_probability"],
        "optimal_probability": result["exact_distribution"]["optimal_probability"],
        "penalty_dominates": result["penalty"]["all_infeasible_states_are_worse"],
        "total_shots": result["resources"]["total_shots"],
        "objective_evaluations": result["resources"]["objective_evaluations"],
        "depth": result["resources"]["depth"],
        "gates": result["resources"]["gates"],
        "two_qubit_gates": result["resources"]["two_qubit_gates"],
        "qubits": result["resources"]["qubits"],
        "seconds": result["timings"]["total_seconds"],
        "parameters": result["parameters"],
    }


def run_qaoa_study(config: dict, records: list[dict], target: str, seed: int, workers: int) -> dict:
    started = time.perf_counter()
    chosen = [r for r in records if r["status"] == "feasible"]
    if config.get("decisions") not in (None, "all"):
        chosen = chosen[: int(config["decisions"])]
    decisions, jobs = [], []
    for index, record in enumerate(chosen):
        request = {k: v for k, v in record["request"].items() if k != "solver"}
        reference = solve_request(request, record["request_sha256"])
        model = reference["canonical_model"]
        scale, k = int(model["scale"]), model["select_count"]
        scores = sorted(
            energy(model, combo) for combo in itertools.combinations(model["symbols"], k)
        )
        optimum = int(reference["objective_integer"])
        searches = [
            swap_search(model, seed + s, config["swap_search"]["max_evaluations"])
            for s in range(config["swap_search"]["seeds"])
        ]
        decisions.append(
            {
                "index": index,
                "data_cutoff": reference["data_cutoff"],
                "exact_selected": reference["selected"],
                "exact_objective": optimum / scale,
                "feasible_portfolios": len(scores),
                "optimal_portfolios": scores.count(optimum),
                "second_best_gap": (scores[1] - optimum) / scale,
                "median_gap": (scores[len(scores) // 2] - optimum) / scale,
                "worst_gap": (scores[-1] - optimum) / scale,
                "bitstrings": 2 ** len(model["symbols"]),
                "swap_search": {
                    "runs": len(searches),
                    "found_optimum_rate": float(
                        np.mean([int(s["objective_integer"]) == optimum for s in searches])
                    ),
                    "mean_evaluations": float(np.mean([s["evaluations"] for s in searches])),
                    "max_evaluations": config["swap_search"]["max_evaluations"],
                    "mean_gap": float(
                        np.mean([(int(s["objective_integer"]) - optimum) / scale for s in searches])
                    ),
                },
            }
        )
        for layers in config["layers"]:
            for scale_text in config["penalty_scales"]:
                for run_seed in range(config["seeds"]):
                    jobs.append(
                        {
                            "key": {
                                "decision": index,
                                "layers": layers,
                                "penalty_scale": scale_text,
                                "seed": seed + run_seed,
                            },
                            "request": request,
                            "request_sha256": record["request_sha256"],
                            "settings": {
                                "method": "qaoa_cudaq",
                                "layers": layers,
                                "shots": config["shots"],
                                "final_shots": config["final_shots"],
                                "grid": config["grid"],
                                "maxiter": config["maxiter"],
                                "seed": seed + run_seed,
                                "penalty_scale": scale_text,
                                "target": target,
                            },
                        }
                    )
    # Separate processes, bounded: each owns its simulator state (plan N8).
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        runs = list(pool.map(_run_one, jobs))

    uniform_optimal = float(np.mean([d["optimal_portfolios"] / d["bitstrings"] for d in decisions]))
    uniform_feasible = decisions[0]["feasible_portfolios"] / decisions[0]["bitstrings"]
    summary = []
    for layers in config["layers"]:
        for scale_text in config["penalty_scales"]:
            group = [r for r in runs if r["layers"] == layers and r["penalty_scale"] == scale_text]
            done = [r for r in group if r["status"] in ("feasible", "no_feasible_sample")]
            if not done:
                summary.append(
                    {
                        "layers": layers,
                        "penalty_scale": scale_text,
                        "runs": len(group),
                        "failed": len(group),
                    }
                )
                continue
            optimal = float(np.mean([r["optimal_probability"] for r in done]))
            summary.append(
                {
                    "layers": layers,
                    "penalty_scale": scale_text,
                    "runs": len(group),
                    "failed": len(group) - len(done),
                    "no_feasible_sample": sum(r["status"] == "no_feasible_sample" for r in done),
                    "found_optimum_rate": float(np.mean([r["found_optimum"] for r in done])),
                    "mean_gap": float(np.mean([r["gap"] for r in done if r["gap"] is not None]))
                    if any(r["gap"] is not None for r in done)
                    else None,
                    "feasible_probability": float(
                        np.mean([r["feasible_probability"] for r in done])
                    ),
                    "optimal_probability": optimal,
                    "optimal_probability_min": float(min(r["optimal_probability"] for r in done)),
                    "optimal_probability_max": float(max(r["optimal_probability"] for r in done)),
                    "advantage_over_uniform": optimal / uniform_optimal,
                    "expected_final_shots_to_optimum": 1 / optimal if optimal > 0 else None,
                    "median_total_shots": float(np.median([r["total_shots"] for r in done])),
                    "median_objective_evaluations": float(
                        np.median([r["objective_evaluations"] for r in done])
                    ),
                    "mean_seconds": float(np.mean([r["seconds"] for r in done])),
                    "qubits": done[0]["qubits"],
                    "depth": done[0]["depth"],
                    "gates": done[0]["gates"],
                    "two_qubit_gates": done[0]["two_qubit_gates"],
                    "penalty_dominates": all(r["penalty_dominates"] for r in done),
                }
            )
    final_shots = config["final_shots"]
    return {
        "question": "With forecasts, objective and constraints fixed, how well does each search "
        "method find the best three-fund selection, and at what cost?",
        "settings": config,
        "target": target,
        "seed": seed,
        "workers": workers,
        "decisions": decisions,
        "runs": runs,
        "summary": summary,
        "baselines": {
            "exact_enumeration": {
                "evaluations": decisions[0]["feasible_portfolios"],
                "found_optimum_rate": 1.0,
                "note": "scores every feasible portfolio once; the answer key for every gap",
            },
            "swap_search": {
                "found_optimum_rate": float(
                    np.mean([d["swap_search"]["found_optimum_rate"] for d in decisions])
                ),
                "mean_evaluations": float(
                    np.mean([d["swap_search"]["mean_evaluations"] for d in decisions])
                ),
                "max_evaluations": config["swap_search"]["max_evaluations"],
                "runs_per_decision": config["swap_search"]["seeds"],
            },
            "uniform_sampling": {
                "feasible_probability": uniform_feasible,
                "optimal_probability": uniform_optimal,
                "expected_draws_to_optimum": 1 / uniform_optimal,
                "found_optimum_rate_at_final_shots": 1 - (1 - uniform_optimal) ** final_shots,
                "draws_for_95_percent": math.log(0.05) / math.log(1 - uniform_optimal),
                "note": "drawing bitstrings at random and keeping the best feasible one",
            },
        },
        "gap_unit": "financial objective units (forecast-horizon return, risk-adjusted)",
        "selection_rule": "lowest financial objective among feasible bitstrings of the final "
        "sample; no repair; exact enumeration is used only as the answer key",
        "probability_source": "privileged simulator statevector at the final parameters",
        "evidence_label": "simulator performance: CUDA-Q statevector on a CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
