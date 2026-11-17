"""Grover adaptive search on the saved portfolio decisions (plan section 14, item 4).

Same forecasts, objective and constraint as the QAOA study; only the search method changes.
The comparison that matters at this size is oracle calls against the 56 evaluations of
exhaustive enumeration, plus what the coherent arithmetic costs in qubits and gates.
"""

from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from qflab.methods.gas import quantization_audit
from qflab.optimizer_worker import solve_request


def _run_one(job: dict) -> dict:
    from qflab.backends.cudaq_programs import ProgramRunner
    from qflab.methods.gas import grover_adaptive_search

    reference = solve_request(job["request"], job["request_sha256"])
    model = reference["canonical_model"]
    try:
        found = grover_adaptive_search(
            model,
            ProgramRunner(job["seed"], job["target"]),
            job["value_bits"],
            job["seed"],
            job["patience"],
            job["max_oracle_calls"],
        )
    except Exception as error:
        return {**job["key"], "status": "failed", "error": f"{type(error).__name__}: {error}"}
    gap = int(found["objective_integer"]) - int(reference["objective_integer"])
    deepest = found["deepest_circuit"]
    return {
        **job["key"],
        "status": found["status"],
        "selected": found["selected"],
        "gap": gap / int(model["scale"]),
        "found_optimum": gap == 0,
        "oracle_calls": found["oracle_calls"],
        "circuit_executions": found["circuit_executions"],
        "max_rotations": found["max_rotations"],
        "improvements": sum(h["improved"] for h in found["history"]) - 1,
        "qubits": deepest["qubits"],
        "deepest_gates": deepest["gates"],
        "deepest_depth": deepest["depth"],
        "per_rotation_gates": found["per_rotation_gates"],
        "seconds": found["seconds"],
    }


def run_gas_study(config: dict, records: list[dict], target: str, seed: int, workers: int) -> dict:
    started = time.perf_counter()
    chosen = [r for r in records if r["status"] == "feasible"]
    if config.get("decisions") not in (None, "all"):
        chosen = chosen[: int(config["decisions"])]
    jobs, audits = [], []
    space = None
    for index, record in enumerate(chosen):
        request = {k: v for k, v in record["request"].items() if k != "solver"}
        reference = solve_request(request, record["request_sha256"])
        model = reference["canonical_model"]
        space = math.comb(len(model["symbols"]), model["select_count"])
        scale = int(model["scale"])
        for row in quantization_audit(model, range(5, 13)):
            audits.append(
                {
                    "decision": index,
                    **{k: v for k, v in row.items() if k != "worst_gap_among_minimizers_integer"},
                    "worst_gap": int(row["worst_gap_among_minimizers_integer"]) / scale,
                }
            )
        for run_seed in range(config["seeds"]):
            jobs.append(
                {
                    "key": {"decision": index, "seed": seed + run_seed},
                    "request": request,
                    "request_sha256": record["request_sha256"],
                    "seed": seed + run_seed + 1000 * index,
                    "target": target,
                    "value_bits": config["value_bits"],
                    "patience": config["patience"],
                    "max_oracle_calls": config["max_oracle_calls"],
                }
            )
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        runs = list(pool.map(_run_one, jobs))
    done = [r for r in runs if r["status"] != "failed"]
    widths = sorted({row["value_bits"] for row in audits})
    return {
        "question": "How many coherent objective evaluations does Grover adaptive search need "
        "to find the best three-fund selection, and what does the arithmetic cost?",
        "settings": config,
        "target": target,
        "seed": seed,
        "search_space": space,
        "runs": runs,
        "summary": {
            "runs": len(runs),
            "failed": len(runs) - len(done),
            "found_optimum_rate": float(np.mean([r["found_optimum"] for r in done])),
            "mean_gap": float(np.mean([r["gap"] for r in done])),
            "mean_oracle_calls": float(np.mean([r["oracle_calls"] for r in done])),
            "median_oracle_calls": float(np.median([r["oracle_calls"] for r in done])),
            "mean_circuit_executions": float(np.mean([r["circuit_executions"] for r in done])),
            "mean_seconds": float(np.mean([r["seconds"] for r in done])),
            "qubits": done[0]["qubits"],
            "per_rotation_gates": done[0]["per_rotation_gates"],
            "largest_circuit_gates": max(r["deepest_gates"] for r in done),
            "largest_circuit_depth": max(r["deepest_depth"] for r in done),
        }
        if done
        else {"runs": len(runs), "failed": len(runs)},
        "baselines": {
            "exact_enumeration_evaluations": space,
            "sqrt_search_space": math.sqrt(space) if space else None,
            "note": "enumeration scores all feasible portfolios once and also proves "
            "optimality; Grover adaptive search stops on a patience rule and proves nothing",
        },
        "quantization": [
            {
                "value_bits": bits,
                "decisions": sum(a["value_bits"] == bits for a in audits),
                "true_optimum_kept": sum(
                    a["true_optimum_is_a_minimizer"] for a in audits if a["value_bits"] == bits
                ),
                "unique_minimizer": sum(
                    a["tied_at_minimum"] == 1 and a["true_optimum_is_a_minimizer"]
                    for a in audits
                    if a["value_bits"] == bits
                ),
                "worst_gap": max(a["worst_gap"] for a in audits if a["value_bits"] == bits),
                "mean_distinct_levels": float(
                    np.mean([a["distinct_levels"] for a in audits if a["value_bits"] == bits])
                ),
            }
            for bits in widths
        ],
        "quantization_note": "classical audit, no circuits: what rounding the 1e10-scale "
        "integer coefficients into a small register does to the answer",
        "counting_convention": "one oracle call = compute the objective, flip the sign-bit "
        "phase, uncompute, inside one Grover rotation; each circuit execution adds one "
        "classical evaluation of the measured portfolio",
        "evidence_label": "simulator performance: CUDA-Q statevector on a CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
