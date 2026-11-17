"""Regime prediction with a quantum kernel against tuned classical kernels (plan 14, item 6).

Task: from past-only features of the equal-weight ETF basket, predict whether its realized
volatility over the next ``horizon`` sessions will be above the training median.

Chronology: examples are ordered by date and split into training, validation and test blocks
with ``horizon`` sessions purged at each boundary, because the labels look forward. Feature
scaling and the label threshold use training data only. Every kernel gets a hyperparameter
grid of similar size, chosen on the validation block; the test block is used once.

The history is short (one snapshot, a few hundred sessions), so the test block is small and
its accuracy interval is wide. This study measures cost and checks the plumbing; it cannot
rank the kernels.
"""

from __future__ import annotations

import math
import time

import numpy as np

from qflab.methods.kernels import (
    exact_kernel,
    fit_ridge,
    polynomial_kernel,
    predict,
    rbf_kernel,
    sampled_kernel,
    wilson_interval,
)

FEATURES = [
    "5-session return",
    "10-session volatility",
    "20-session volatility",
    "20-session return",
]


def build_examples(
    returns: np.ndarray, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(session index, features, forward volatility); features use returns up to that session."""
    rows, features, forward = [], [], []
    for t in range(19, len(returns) - horizon, stride):
        features.append(
            [
                returns[t - 4 : t + 1].sum(),
                returns[t - 9 : t + 1].std(),
                returns[t - 19 : t + 1].std(),
                returns[t - 19 : t + 1].sum(),
            ]
        )
        forward.append(returns[t + 1 : t + 1 + horizon].std())
        rows.append(t)
    return np.array(rows), np.array(features), np.array(forward)


def _accuracy(predicted: np.ndarray, labels: np.ndarray) -> dict:
    hits = int((predicted == labels).sum())
    low, high = wilson_interval(hits, len(labels))
    return {"accuracy": hits / len(labels), "ci_low": low, "ci_high": high, "examples": len(labels)}


def run_kernel_study(
    config: dict, returns: np.ndarray, dates: list[str], target: str, seed: int
) -> dict:
    from qflab.backends.cudaq_programs import ProgramRunner

    started = time.perf_counter()
    horizon, stride = config["horizon"], config["stride"]
    sessions, raw, forward = build_examples(returns, horizon, stride)
    count = len(sessions)
    purge = math.ceil(horizon / stride)
    train_end = int(config["train_fraction"] * count)
    validation_end = int((config["train_fraction"] + config["validation_fraction"]) * count)
    train = np.arange(0, train_end)
    validation = np.arange(train_end + purge, validation_end)
    test = np.arange(validation_end + purge, count)
    threshold = float(np.median(forward[train]))
    labels = np.where(forward > threshold, 1, -1)
    mean, spread = raw[train].mean(axis=0), raw[train].std(axis=0)
    z = np.clip((raw - mean) / spread, -2.5, 2.5)
    runner = ProgramRunner(seed, target)

    def angles(scale: float) -> np.ndarray:
        return math.pi / 2 + scale * z * (math.pi / 2) / 2.5

    def select(candidates):
        """Best validation accuracy; ties go to the earlier (simpler) candidate."""
        best = None
        for name, settings, kernel_of in candidates:
            for ridge in config["ridges"]:
                coefficients = fit_ridge(kernel_of(train, train), labels[train], ridge)
                score = float(
                    (
                        predict(kernel_of(validation, train), coefficients) == labels[validation]
                    ).mean()
                )
                if best is None or score > best["validation_accuracy"]:
                    best = {
                        "name": name,
                        "settings": {**settings, "ridge": ridge},
                        "validation_accuracy": score,
                        "kernel_of": kernel_of,
                        "coefficients": coefficients,
                    }
        return best

    quantum_cache: dict[tuple, np.ndarray] = {}

    def quantum_kernel(scale, repetitions):
        def kernel_of(rows, columns):
            key = (scale, repetitions)
            if key not in quantum_cache:
                data = angles(scale)
                quantum_cache[key] = exact_kernel(runner, data, data, repetitions)
            return quantum_cache[key][np.ix_(rows, columns)]

        return kernel_of

    quantum_started = time.perf_counter()
    quantum = select(
        [
            (
                "quantum fidelity kernel",
                {"scale": scale, "repetitions": reps},
                quantum_kernel(scale, reps),
            )
            for scale in config["quantum_scales"]
            for reps in config["quantum_repetitions"]
        ]
    )
    quantum_seconds = time.perf_counter() - quantum_started
    gaussian = select(
        [
            ("RBF kernel", {"gamma": gamma}, lambda r, c, g=gamma: rbf_kernel(z[r], z[c], g))
            for gamma in config["rbf_gammas"]
        ]
    )
    polynomial = select(
        [
            (
                "polynomial kernel",
                {"degree": degree},
                lambda r, c, d=degree: polynomial_kernel(z[r], z[c], d),
            )
            for degree in config["polynomial_degrees"]
        ]
    )

    rows = []
    for chosen, candidates in (
        (quantum, len(config["quantum_scales"]) * len(config["quantum_repetitions"])),
        (gaussian, len(config["rbf_gammas"])),
        (polynomial, len(config["polynomial_degrees"])),
    ):
        predicted = predict(chosen["kernel_of"](test, train), chosen["coefficients"])
        gram = chosen["kernel_of"](train, train)
        rows.append(
            {
                "method": chosen["name"],
                "settings": chosen["settings"],
                "candidates_tried": candidates * len(config["ridges"]),
                "validation_accuracy": chosen["validation_accuracy"],
                **_accuracy(predicted, labels[test]),
                "mean_off_diagonal_kernel": float(
                    (gram.sum() - np.trace(gram)) / (len(gram) * (len(gram) - 1))
                ),
                "kernel_source": "simulator statevectors (privileged)"
                if chosen is quantum
                else "closed form",
            }
        )

    # The chosen quantum kernel again, estimated the way a device would: one circuit per pair.
    data = angles(quantum["settings"]["scale"])
    repetitions = quantum["settings"]["repetitions"]
    sampled_started = time.perf_counter()
    fitted = data[train]
    gram, train_circuits = sampled_kernel(runner, fitted, fitted, repetitions, config["shots"])
    cross, test_circuits = sampled_kernel(runner, data[test], fitted, repetitions, config["shots"])
    sampled_seconds = time.perf_counter() - sampled_started
    coefficients = fit_ridge(gram, labels[train], quantum["settings"]["ridge"])
    exact_gram = quantum["kernel_of"](train, train)
    from qflab.methods.kernels import feature_circuit

    pair = feature_circuit(data[0], repetitions)
    pair.extend(feature_circuit(data[1], repetitions).inverse())
    rows.append(
        {
            "method": "quantum fidelity kernel, sampled",
            "settings": {**quantum["settings"], "shots": config["shots"]},
            "candidates_tried": 0,
            "validation_accuracy": None,
            **_accuracy(predict(cross, coefficients), labels[test]),
            "mean_off_diagonal_kernel": float(
                (gram.sum() - np.trace(gram)) / (len(gram) * (len(gram) - 1))
            ),
            "kernel_source": f"{config['shots']} shots per pair circuit",
        }
    )
    majority = 1 if (labels[train] == 1).mean() >= 0.5 else -1
    persistence = np.where(raw[test, 1] > np.median(raw[train, 1]), 1, -1)
    baselines = [
        {
            "method": "always the training majority class",
            **_accuracy(np.full(len(test), majority), labels[test]),
        },
        {
            "method": "persistence: current 10-session volatility above its training median",
            **_accuracy(persistence, labels[test]),
        },
    ]
    examples = len(train)
    return {
        "question": "Does a quantum fidelity kernel predict the next volatility regime better "
        "than tuned classical kernels, and what does building it cost?",
        "settings": config,
        "target": target,
        "seed": seed,
        "data": {
            "series": "equal-weight basket of the 8 ETFs, daily log returns from the frozen "
            "snapshot",
            "first_session": dates[0],
            "last_session": dates[-1],
            "features": FEATURES,
            "label": f"realized volatility over the next {horizon} sessions above the training "
            "median",
            "examples": {"train": len(train), "validation": len(validation), "test": len(test)},
            "purged_per_boundary": purge,
            "train_positive_share": float((labels[train] == 1).mean()),
            "test_positive_share": float((labels[test] == 1).mean()),
            "train_last_session": dates[int(sessions[train[-1]]) + 1],
            "test_first_session": dates[int(sessions[test[0]]) + 1],
        },
        "rows": rows,
        "baselines": baselines,
        "cost": {
            "qubits": raw.shape[1],
            "pair_circuit_gates": len(pair.ops),
            "pair_circuit_depth": runner.resources(pair)["depth"],
            "train_pair_circuits": train_circuits,
            "test_pair_circuits": test_circuits,
            "shots_per_circuit": config["shots"],
            "total_shots": (train_circuits + test_circuits) * config["shots"],
            "sampled_kernel_seconds": sampled_seconds,
            "exact_kernel_selection_seconds": quantum_seconds,
            "sampled_kernel_max_error": float(np.max(np.abs(gram - exact_gram))),
            "scaling": "training needs M(M-1)/2 pair circuits: "
            f"{examples * (examples - 1) // 2} at M = {examples}, about "
            f"{10_000 * 9_999 // 2:,} at M = 10,000; each prediction needs M more",
        },
        "limits": "the test block is a few dozen overlapping windows from one year; the "
        "intervals overlap and no ranking of kernels is supported",
        "evidence_label": "simulator performance: CUDA-Q statevector on a CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
