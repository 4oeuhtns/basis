"""Quantum linear algebra for the minimum-variance portfolio (plan section 12, LA0 to LA4).

Two solvers, HHL and a QSVT-based solver, are run on the same small systems. Results are kept
in two modes, as the plan requires:

* **state mode** compares the heralded solution state with the classical reference using the
  simulator's statevector. That is privileged access and not an executable portfolio.
* **allocation mode** recovers weights from measurement counts alone, with every preparation
  counted, and only then forms portfolio weights.

Nothing here is sent to the backtester: the weights can be negative and the simulated account
is long-only, so this track stops at verified weights.
"""

from __future__ import annotations

import math
import time

import numpy as np

from qflab.methods.linear_solvers import (
    estimate_inverse_quadratic,
    hhl_program,
    qsvt_program,
    recover_vector,
    solution_state,
)
from qflab.problems.linear_systems import (
    describe,
    geometric_spectrum,
    minimum_variance,
    regularized_covariance,
    spd_with_spectrum,
)


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


def _state_row(program, runner, reference: dict) -> dict:
    started = time.perf_counter()
    state, probability = solution_state(program, runner)
    resources = runner.resources(program.circuit)
    size = reference["size"]
    ideal = (program.norm_factor * reference["solution_norm"] / math.sqrt(size)) ** 2
    return {
        "fidelity": _fidelity(state, reference["solution_state"]),
        "infidelity": 1 - _fidelity(state, reference["solution_state"]),
        "success_probability": probability,
        "ideal_success_probability": ideal,
        "qubits": resources["qubits"],
        "gates": resources["gates"],
        "depth": resources["depth"],
        "dense_unitary_calls": resources["dense_unitary_calls"],
        "seconds": time.perf_counter() - started,
    }


def _clock_for(kappa: float, extra: int) -> int:
    return max(2, math.ceil(math.log2(kappa))) + extra


def run_linear_algebra(config: dict, records: list[dict], target: str, seed: int) -> dict:
    from qflab.backends.cudaq_programs import ProgramRunner

    started = time.perf_counter()
    runner = ProgramRunner(seed, target)

    # ---- LA0: systems and classical references -------------------------------------
    systems = []
    for spec in config["synthetic"]:
        matrix = spd_with_spectrum(
            geometric_spectrum(spec["size"], spec["kappa"]), seed + spec["size"]
        )
        systems.append(
            (
                f"synthetic {spec['size']}x{spec['size']}, kappa {spec['kappa']:g}",
                "synthetic",
                matrix,
                None,
            )
        )
    response = next(r["response"] for r in records if r["status"] == "feasible")
    symbols = [s.split(".")[0] for s in response["canonical_model"]["symbols"]]
    covariance = np.array(response["diagnostics"]["covariance"], dtype=float)
    # None when the unregularized covariance is singular (fewer returns than funds).
    spectrum = np.linalg.eigvalsh(covariance)
    raw_kappa = float(spectrum[-1] / spectrum[0]) if spectrum[0] > 1e-14 * spectrum[-1] else None
    for fraction in config["covariance_delta_fractions"]:
        matrix, delta = regularized_covariance(covariance, fraction)
        matrix = matrix / np.linalg.eigvalsh(matrix)[-1]  # unit largest eigenvalue; scale cancels
        systems.append(
            (
                f"ETF covariance 8x8, delta = {fraction:g} x mean variance",
                "covariance",
                matrix,
                fraction,
            )
        )

    la0, la1, la2 = [], [], []
    keep = {}
    for label, kind, matrix, fraction in systems:
        rhs = np.ones(len(matrix))
        reference = describe(matrix, rhs)
        weights = minimum_variance(matrix, reference["solution"])
        bounds = (reference["eigenvalue_min"], reference["eigenvalue_max"])
        la0.append(
            {
                "label": label,
                "kind": kind,
                "size": reference["size"],
                "system_qubits": reference["qubits"],
                "condition_number": reference["condition_number"],
                "eigenvalue_min": bounds[0],
                "eigenvalue_max": bounds[1],
                "delta_fraction": fraction,
                "classical_residual": reference["residual"],
                "weights": [float(w) for w in weights["weights"]],
                "variance": weights["variance"],
                "negative_weights": weights["negative_weights"],
            }
        )
        # ---- LA1: HHL, phase precision swept on its own ------------------------------
        base = _clock_for(reference["condition_number"], 0)
        for extra in config["hhl_extra_clock_qubits"]:
            clock = base + extra
            if reference["qubits"] + clock + 1 > config["max_qubits"]:
                continue
            program = hhl_program(matrix, rhs, clock, bounds)
            la1.append(
                {
                    "label": label,
                    "clock_qubits": clock,
                    "evolution": "exact",
                    **_state_row(program, runner, reference),
                }
            )
        # ---- LA2: QSVT on the same system --------------------------------------------
        for epsilon in config["qsvt_epsilons"]:
            program = qsvt_program(matrix, rhs, bounds, epsilon)
            la2.append(
                {
                    "label": label,
                    "epsilon": epsilon,
                    "polynomial_degree": program.info["polynomial_degree"],
                    "phase_residual": program.info["phase_residual"],
                    "phase_solve_seconds": program.info["phase_solve_seconds"],
                    **_state_row(program, runner, reference),
                }
            )
        keep[label] = (matrix, rhs, reference, bounds, weights)

    # HHL error sources separated: exactly representable eigenvalues remove the phase error;
    # Trotterized evolution adds a Hamiltonian-simulation error on top.
    clock = 4
    exact_eigenvalues = [k / 2**clock for k in (3, 6, 10, 15)]
    matrix = spd_with_spectrum(exact_eigenvalues, seed)
    reference = describe(matrix, np.ones(4))
    program = hhl_program(
        matrix,
        np.ones(4),
        clock,
        (min(exact_eigenvalues), max(exact_eigenvalues)),
        evolution_time=2 * math.pi,
    )
    controls = [
        {
            "case": "eigenvalues exactly representable on the clock",
            "clock_qubits": clock,
            "evolution": "exact",
            **_state_row(program, runner, reference),
        }
    ]
    trotter_label = config["trotter"]["system"]
    matrix, rhs, reference, bounds, _ = keep[trotter_label]
    clock = _clock_for(reference["condition_number"], config["trotter"]["extra_clock_qubits"])
    for steps in config["trotter"]["steps"]:
        program = hhl_program(matrix, rhs, clock, bounds, evolution="trotter", trotter_steps=steps)
        controls.append(
            {
                "case": f"{trotter_label}: Trotterized evolution, {steps} step(s) per unit power",
                "clock_qubits": clock,
                "evolution": "trotter",
                "trotter_steps": steps,
                "pauli_terms": program.info["pauli_terms"],
                **_state_row(program, runner, reference),
            }
        )

    # ---- LA3: readout and financial use ---------------------------------------------
    readout = []
    matrix, rhs, reference, bounds, weights = keep[config["readout"]["system"]]
    true_weights = weights["weights"]
    clock = _clock_for(reference["condition_number"], config["readout"]["hhl_extra_clock_qubits"])
    programs = {
        "hhl": hhl_program(matrix, rhs, clock, bounds),
        "qsvt": qsvt_program(matrix, rhs, bounds, config["readout"]["qsvt_epsilon"]),
    }
    quadratic = float(rhs @ reference["solution"])
    for method, program in programs.items():
        state, probability = solution_state(program, runner)
        for shots in config["readout"]["shots"]:
            rows = []
            for trial in range(config["readout"]["seeds"]):
                sampler = ProgramRunner(seed + 97 * trial + shots, target)
                found = recover_vector(program, sampler, shots)
                scalar = estimate_inverse_quadratic(program, sampler, shots)
                if found["status"] != "ok":
                    rows.append(None)
                    continue
                estimate = found["vector"]
                try:
                    recovered = minimum_variance(matrix, estimate)
                except ValueError:  # too few heralded shots: the estimate sums to zero
                    rows.append(None)
                    continue
                # Best overall scale for the residual; the weights do not depend on it.
                image = matrix @ estimate
                scale = float(image @ rhs / (image @ image))
                rows.append(
                    {
                        "weight_error": float(np.max(np.abs(recovered["weights"] - true_weights))),
                        "variance_excess": recovered["variance"] / weights["variance"] - 1,
                        "residual": float(
                            np.linalg.norm(scale * image - rhs) / np.linalg.norm(rhs)
                        ),
                        "sum": recovered["sum"],
                        "sign_errors": int(
                            np.sum(np.sign(recovered["weights"]) != np.sign(true_weights))
                        ),
                        # Size of the largest true weight whose sign came out wrong.
                        "largest_flipped_weight": float(
                            np.max(
                                np.abs(true_weights)
                                * (np.sign(recovered["weights"]) != np.sign(true_weights)),
                                initial=0.0,
                            )
                        ),
                        "success_rate": found["success_rate"],
                        "preparations": found["shots"],
                        "scalar_error": abs(scalar["value"] / quadratic - 1)
                        if scalar["status"] == "ok"
                        else None,
                        "weights": [float(w) for w in recovered["weights"]],
                    }
                )
            done = [r for r in rows if r]
            scalar_errors = [r["scalar_error"] for r in done if r["scalar_error"] is not None]
            readout.append(
                {
                    "method": method,
                    "shots_per_setting": shots,
                    "settings": len(program.system) + 1,
                    "preparations": shots * (len(program.system) + 1),
                    "trials": len(rows),
                    "failed_trials": len(rows) - len(done),
                    "state_fidelity": _fidelity(state, reference["solution_state"]),
                    "success_probability": probability,
                    "median_weight_error": float(np.median([r["weight_error"] for r in done]))
                    if done
                    else None,
                    "median_variance_excess": float(np.median([r["variance_excess"] for r in done]))
                    if done
                    else None,
                    "median_residual": float(np.median([r["residual"] for r in done]))
                    if done
                    else None,
                    "median_sign_errors": float(np.median([r["sign_errors"] for r in done]))
                    if done
                    else None,
                    "median_largest_flipped_weight": float(
                        np.median([r["largest_flipped_weight"] for r in done])
                    )
                    if done
                    else None,
                    "weights_sum_to_one": bool(all(abs(r["sum"] - 1) < 1e-9 for r in done)),
                    "scalar_preparations": shots,
                    "median_scalar_error": float(np.median(scalar_errors))
                    if scalar_errors
                    else None,
                    "example_weights": done[0]["weights"] if done else None,
                }
            )

    # ---- LA4: resources against condition number, size and regularization ------------
    sweep = []
    for size in config["sweep"]["sizes"]:
        for kappa in config["sweep"]["kappas"]:
            matrix = spd_with_spectrum(geometric_spectrum(size, kappa), seed + size)
            rhs = np.ones(size)
            reference = describe(matrix, rhs)
            bounds = (reference["eigenvalue_min"], reference["eigenvalue_max"])
            row = {"size": size, "kappa": kappa}
            clock = _clock_for(kappa, config["sweep"]["hhl_extra_clock_qubits"])
            if reference["qubits"] + clock + 1 <= config["max_qubits"]:
                row["hhl"] = {
                    "clock_qubits": clock,
                    **_state_row(hhl_program(matrix, rhs, clock, bounds), runner, reference),
                }
            program = qsvt_program(matrix, rhs, bounds, config["sweep"]["qsvt_epsilon"])
            row["qsvt"] = {
                "polynomial_degree": program.info["polynomial_degree"],
                **_state_row(program, runner, reference),
            }
            sweep.append(row)
    regularization = []
    for fraction in config["sweep"]["delta_fractions"]:
        matrix, _ = regularized_covariance(covariance, fraction)
        matrix = matrix / np.linalg.eigvalsh(matrix)[-1]
        reference = describe(matrix, np.ones(len(matrix)))
        weights = minimum_variance(matrix, reference["solution"])
        regularization.append(
            {
                "delta_fraction": fraction,
                "condition_number": reference["condition_number"],
                "negative_weights": weights["negative_weights"],
                "largest_weight": float(np.max(np.abs(weights["weights"]))),
                "hhl_clock_qubits": _clock_for(
                    reference["condition_number"], config["sweep"]["hhl_extra_clock_qubits"]
                ),
            }
        )

    return {
        "question": "Can HHL and a QSVT solver reproduce the minimum-variance weights of a small "
        "covariance matrix, and what does it cost to get numbers out of the solution state?",
        "settings": config,
        "target": target,
        "seed": seed,
        "symbols": symbols,
        "raw_covariance_condition_number": raw_kappa,
        "systems": la0,
        "hhl": la1,
        "hhl_controls": controls,
        "qsvt": la2,
        "readout": readout,
        "readout_system": config["readout"]["system"],
        "readout_reference_weights": [float(w) for w in true_weights],
        "sweep": sweep,
        "regularization": regularization,
        "disclosures": [
            "HHL (exact): exp(iAt) comes from a dense classical eigendecomposition of A.",
            "QSVT: the block encoding is built densely from an eigendecomposition of A, and "
            "the phase factors from a classical least-squares solve.",
            "Both solvers are told the exact smallest and largest eigenvalue of A.",
            "State fidelity and success probability are read from the simulator statevector "
            "(privileged access). Readout rows use measurement counts only.",
            "No fault-tolerant resource projection is made; gate counts are logical, with "
            "dense unitaries counted as single calls.",
        ],
        "trading_use": "none: the weights can be negative and the simulated account is "
        "long-only, so no decision is sent to the backtester",
        "evidence_label": "simulator performance: CUDA-Q statevector on a CPU; small algorithm "
        "demonstration, no speedup claim",
        "elapsed_seconds": time.perf_counter() - started,
    }
