"""Covariance spectral analysis with SQD and SKQD (plan section 13.3, R2).

Output fixed in advance: the largest eigenvalue and the leading risk-factor direction. Each
method chooses ``m`` asset coordinates and diagonalizes that sub-block; they differ only in
how the coordinates are chosen. The classical competitor for that choice is simply the ``m``
largest variances, and the dense eigensolve, Lanczos and a randomized range finder are the
references for the task itself.
"""

from __future__ import annotations

import time

import numpy as np

from qflab.methods.subspace import (
    clustered_covariance,
    diagonal_selection,
    downstream_error,
    krylov_circuits,
    lanczos_reference,
    localized_covariance,
    market_covariance,
    most_frequent,
    power_selection,
    project,
    randomized_reference,
    sample_indices,
    spectrum_reference,
)


def _family(spec: dict, seed: int) -> np.ndarray:
    kind, size = spec["kind"], spec["size"]
    if kind == "localized":
        return localized_covariance(size, spec["block"], spec["gap"], seed)
    if kind == "market":
        return market_covariance(size, spec["correlation"], seed)
    if kind == "clustered":
        return clustered_covariance(size, spec["clusters"], spec["gap"], seed)
    raise ValueError(f"Unknown covariance family: {kind}")


def run_spectral_study(config: dict, records: list[dict], target: str, seed: int) -> dict:
    from qflab.backends.cudaq_programs import ProgramRunner

    started = time.perf_counter()
    times = [config["time_step"] * k for k in range(config["krylov_states"])]
    seeds = [seed + i for i in range(config["seeds"])]

    runner = ProgramRunner(seed, target)
    drawn: dict[tuple, tuple] = {}

    def samples(covariance, method, initial, shots, evolution, steps):
        """Counts per seed for one sampling method; circuits are built and simulated once."""
        key = (id(covariance), method, initial, shots, evolution, steps)
        if key not in drawn:
            circuits = krylov_circuits(
                covariance, initial, times if method == "skqd" else [0.0], evolution, steps
            )
            # Same total shot budget for SQD (one circuit) and SKQD (several).
            per_circuit = shots if method == "skqd" else shots * len(times)
            counts = []
            for run_seed in seeds:
                runner.reseed(run_seed)
                counts.append(sample_indices(circuits, runner, per_circuit))
            drawn[key] = (counts, max(len(c.ops) for c in circuits))
        return drawn[key]

    def quantum(covariance, reference, method, initial, size, shots, evolution="ideal", steps=2):
        """Median scores over seeds for one sampling method at one subspace size."""
        counts, gates = samples(covariance, method, initial, shots, evolution, steps)
        scores = []
        for found in (project(covariance, most_frequent(c, size), reference) for c in counts):
            found["downstream"] = downstream_error(covariance, found["eigenvalue"], found["vector"])
            scores.append(found)
        return {
            "eigenvalue_error": float(np.median([s["eigenvalue_error"] for s in scores])),
            "vector_error": float(np.median([s["vector_error"] for s in scores])),
            "downstream_error": float(np.median([s["downstream"] for s in scores])),
            "subspace_size": int(np.median([s["subspace_size"] for s in scores])),
            "total_shots": shots * len(times),
            "largest_circuit_gates": gates,
        }

    def classical(covariance, reference, size, rule="variance"):
        """Classical coordinate choice: largest variances, or power iteration from the same
        uniform start and with as many matrix applications as SKQD has evolution steps."""
        chosen = (
            diagonal_selection(covariance, size)
            if rule == "variance"
            else power_selection(covariance, size, len(times) - 1)
        )
        found = project(covariance, chosen, reference)
        return {
            "eigenvalue_error": found["eigenvalue_error"],
            "vector_error": found["vector_error"],
            "downstream_error": downstream_error(covariance, found["eigenvalue"], found["vector"]),
            "subspace_size": found["subspace_size"],
            "total_shots": 0,
            "largest_circuit_gates": 0,
        }

    families = [(spec["label"], _family(spec, seed)) for spec in config["families"]]
    response = next(r["response"] for r in records if r["status"] == "feasible")
    families.append(
        (
            "ETF trailing covariance (8 funds)",
            np.array(response["diagnostics"]["covariance"], dtype=float),
        )
    )
    references, rows = [], []
    for label, covariance in families:
        size = len(covariance)
        dense_started = time.perf_counter()
        reference = spectrum_reference(covariance)
        dense_seconds = time.perf_counter() - dense_started
        references.append(
            {
                "family": label,
                "assets": size,
                "qubits": size.bit_length() - 1,
                "eigenvalue": reference["eigenvalue"],
                "gap_ratio": reference["gap_ratio"],
                "explained_variance": reference["explained_variance"],
                "coordinates_for_90_percent": reference["coordinates_for_90_percent"],
                "coordinates_for_99_percent": reference["coordinates_for_99_percent"],
                "dense_eigensolve_seconds": dense_seconds,
                "lanczos_matrix_vector_products": lanczos_reference(covariance)[
                    "matrix_vector_products"
                ],
                "randomized_eigenvalue_error": abs(
                    1
                    - randomized_reference(covariance, seed)["eigenvalue"] / reference["eigenvalue"]
                ),
            }
        )
        for subspace in [m for m in config["subspace_sizes"] if m < size]:
            best = 1 - reference["best_possible_overlap"][subspace - 1]
            base = {"family": label, "requested_size": subspace, "best_possible_vector_error": best}
            shots = config["shots"]
            rows.append(
                {
                    **base,
                    "method": "largest variances (classical)",
                    **classical(covariance, reference, subspace),
                }
            )
            rows.append(
                {
                    **base,
                    "method": "power iteration (classical)",
                    **classical(covariance, reference, subspace, "power"),
                }
            )
            rows.append(
                {
                    **base,
                    "method": "SQD, uniform start",
                    **quantum(covariance, reference, "sqd", "uniform", subspace, shots),
                }
            )
            rows.append(
                {
                    **base,
                    "method": "SKQD, uniform start",
                    **quantum(covariance, reference, "skqd", "uniform", subspace, shots),
                }
            )
            rows.append(
                {
                    **base,
                    "method": "SKQD, volatility start",
                    **quantum(covariance, reference, "skqd", "variance", subspace, shots),
                }
            )

    # One family, one subspace size: vary the gap, the shots and the evolution.
    probe = config["probe"]
    spec = next(s for s in config["families"] if s["label"] == probe["family"])
    gap_sweep = []
    for gap in probe["gaps"]:
        covariance = _family({**spec, "gap": gap}, seed)
        reference = spectrum_reference(covariance)
        gap_sweep.append(
            {
                "gap_setting": gap,
                "gap_ratio": reference["gap_ratio"],
                "skqd": quantum(
                    covariance,
                    reference,
                    "skqd",
                    "uniform",
                    probe["subspace_size"],
                    config["shots"],
                ),
                "classical": classical(covariance, reference, probe["subspace_size"]),
                "power": classical(covariance, reference, probe["subspace_size"], "power"),
            }
        )
    covariance = _family(spec, seed)
    reference = spectrum_reference(covariance)
    shot_sweep = [
        {
            "shots_per_circuit": shots,
            **quantum(covariance, reference, "skqd", "uniform", probe["subspace_size"], shots),
        }
        for shots in probe["shots"]
    ]
    evolution = [
        {
            "evolution": "ideal (dense unitary)",
            **quantum(
                covariance, reference, "skqd", "uniform", probe["subspace_size"], config["shots"]
            ),
        }
    ] + [
        {
            "evolution": f"Trotter, {steps} step(s) per unit time",
            **quantum(
                covariance,
                reference,
                "skqd",
                "uniform",
                probe["subspace_size"],
                config["shots"],
                "trotter",
                steps,
            ),
        }
        for steps in probe["trotter_steps"]
    ]
    return {
        "question": "When does choosing asset coordinates by quantum sampling give a better "
        "leading risk factor than choosing them classically, at the same subspace size?",
        "settings": config,
        "target": target,
        "seed": seed,
        "references": references,
        "rows": rows,
        "probe_family": probe["family"],
        "gap_sweep": gap_sweep,
        "shot_sweep": shot_sweep,
        "evolution": evolution,
        "krylov_times": times,
        "classical_methods": ["largest variances (classical)", "power iteration (classical)"],
        "power_iteration_note": f"power iteration uses {len(times) - 1} applications of the "
        "covariance to the uniform vector, the classical counterpart of the SKQD evolution "
        "times; it needs N^2 operations per step instead of quantum samples",
        "bound_note": "an m-coordinate subspace captures at most the m largest squared entries "
        "of the true eigenvector; 'best possible' is that bound for each family",
        "output_note": "the estimate is an ordinary m-sparse classical vector, so there is no "
        "tomography; the classical step reads m^2 covariance entries",
        "disclosures": [
            "Ideal evolution uses a dense unitary computed classically from the covariance.",
            "Trotter evolution uses a Pauli decomposition of the covariance (4^n traces).",
            "All matrices are tiny; nothing here measures a large-N bottleneck.",
        ],
        "evidence_label": "simulator performance: CUDA-Q statevector on a CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
