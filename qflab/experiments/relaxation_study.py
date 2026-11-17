"""SQOA-style quantum relaxation on the saved portfolio decisions (plan section 13.1, R0).

Three numbers are kept apart for every setting:

* the *approximation error* of the sparsified problem: its exact optimum, scored on the
  original objective (no quantum method involved);
* the *relaxation gap*: how far the relaxed ground energy sits below the best bitstring;
* the *recovered portfolio's gap* on the original objective, after rounding and repair.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from qflab.methods.relaxation import (
    encode,
    evaluate,
    objective_terms,
    solve_relaxation,
    sparsify,
    verify_relaxation,
)
from qflab.optimizer_worker import energy, solve_request


def run_relaxation_study(config: dict, records: list[dict], target: str, seed: int) -> dict:
    from qflab.backends.cudaq_programs import ProgramRunner

    started = time.perf_counter()
    chosen = [r for r in records if r["status"] == "feasible"]
    if config.get("decisions") not in (None, "all"):
        chosen = chosen[: int(config["decisions"])]
    runs, checks = [], []
    for index, record in enumerate(chosen):
        request = {k: v for k, v in record["request"].items() if k != "solver"}
        reference = solve_request(request, record["request_sha256"])
        model = reference["canonical_model"]
        scale, k = int(model["scale"]), model["select_count"]
        symbols = model["symbols"]
        n = len(symbols)
        optimum = int(reference["objective_integer"])
        linear, all_pairs = objective_terms(model)
        combos = list(itertools.combinations(range(n), k))
        for density in config["densities"]:
            pairs = sparsify(all_pairs, density)
            placement = encode(n, pairs)
            audit = verify_relaxation(linear, pairs, 0.0, placement, k)
            sparse_best = min(
                combos, key=lambda c: evaluate(linear, pairs, [int(i in c) for i in range(n)])
            )
            approximation = (energy(model, [symbols[i] for i in sparse_best]) - optimum) / scale
            checks.append(
                {
                    "decision": index,
                    "density": density,
                    "qubits": 1 + max(q for q, _ in placement),
                    "encoded_states_reproduce_objective": audit[
                        "encoded_states_reproduce_objective"
                    ],
                    "is_lower_bound": audit["is_lower_bound"],
                    "relaxation_gap": audit["relaxation_gap"],
                    "approximation_gap": approximation,
                }
            )
            for run_seed in range(config["seeds"]):
                solve_started = time.perf_counter()
                found = solve_relaxation(
                    model,
                    ProgramRunner(seed + run_seed + 100 * index, target),
                    density,
                    config["shots"],
                    config["subspace_size"],
                    config["times"],
                    config["trotter_steps"],
                )
                runs.append(
                    {
                        "decision": index,
                        "density": density,
                        "seed": seed + run_seed,
                        "selected": found["selected"],
                        "gap": (int(found["objective_integer"]) - optimum) / scale,
                        "found_optimum": int(found["objective_integer"]) == optimum,
                        "needed_repair": found["needed_repair"],
                        "selected_before_repair": found["selected_before_repair"],
                        "qubits": found["qubits"],
                        "pairs_kept": found["pairs_kept"],
                        "total_shots": found["total_shots"],
                        "circuit_executions": found["circuit_executions"],
                        "largest_circuit_gates": found["largest_circuit_gates"],
                        "diagonal_encoding": found["diagonal_encoding"],
                        "seconds": time.perf_counter() - solve_started,
                    }
                )
    summary = []
    for density in config["densities"]:
        group = [r for r in runs if r["density"] == density]
        audits = [c for c in checks if c["density"] == density]
        summary.append(
            {
                "density": density,
                "pairs_kept": group[0]["pairs_kept"],
                "runs": len(group),
                "mean_qubits": float(np.mean([r["qubits"] for r in group])),
                "max_qubits": max(r["qubits"] for r in group),
                "diagonal_encoding": all(r["diagonal_encoding"] for r in group),
                "found_optimum_rate": float(np.mean([r["found_optimum"] for r in group])),
                "mean_gap": float(np.mean([r["gap"] for r in group])),
                "repair_rate": float(np.mean([r["needed_repair"] for r in group])),
                "approximation_optimum_rate": float(
                    np.mean([c["approximation_gap"] == 0 for c in audits])
                ),
                "mean_approximation_gap": float(np.mean([c["approximation_gap"] for c in audits])),
                "mean_relaxation_gap": float(np.mean([c["relaxation_gap"] for c in audits])),
                "median_total_shots": float(np.median([r["total_shots"] for r in group])),
                "largest_circuit_gates": max(r["largest_circuit_gates"] for r in group),
                "mean_seconds": float(np.mean([r["seconds"] for r in group])),
            }
        )
    return {
        "question": "After sparsifying the objective so that variables can share qubits, does a "
        "relaxed, sample-based solve recover a good three-fund selection?",
        "settings": config,
        "target": target,
        "seed": seed,
        "runs": runs,
        "checks": checks,
        "summary": summary,
        "validation": {
            "encoded_states_reproduce_objective": all(
                c["encoded_states_reproduce_objective"] for c in checks
            ),
            "relaxed_ground_energy_is_lower_bound": all(c["is_lower_bound"] for c in checks),
            "cases": len(checks),
        },
        "gap_unit": "financial objective units, original (dense) objective",
        "status_note": "a proposed adaptation; the published method was tested on MaxCut. It is "
        "not exposed as a backtest solver.",
        "repair_note": "top-K selection by magnetization is a classical repair: one sort of N "
        "numbers. 'Repair rate' is how often the unrepaired rounding did not select exactly K.",
        "evidence_label": "simulator performance: CUDA-Q statevector on a CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
