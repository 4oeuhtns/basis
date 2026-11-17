"""A quantum generative model for daily returns, against classical density models (plan 14.8).

Criteria fixed before any model is fitted:

* distribution quality on held-out data: cross-entropy (negative log-likelihood per
  observation) and total variation against the later, unseen histogram;
* downstream risk: the probability of a daily loss beyond a fixed threshold, and the 95%
  value-at-risk bin, against what the held-out period actually produced.

The split is chronological: models are fitted to the earlier part of the history and judged on
the later part. The held-out period is short, so differences of a few percent are noise.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import gaussian_kde, norm
from scipy.stats import t as student_t

from qflab.methods.generative import born_circuit, parameter_count, train_born_machine


def _binned(cdf, edges: np.ndarray) -> np.ndarray:
    """Bin masses from a CDF; both tails are folded into the end bins."""
    values = np.asarray(cdf(edges), dtype=float)
    mass = np.diff(values)
    mass[0] += values[0]
    mass[-1] += 1 - values[-1]
    return mass / mass.sum()


def _histogram(values: np.ndarray, edges: np.ndarray, smoothing: float) -> np.ndarray:
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(index, minlength=len(edges) - 1) + smoothing
    return counts / counts.sum()


def run_generative_study(
    config: dict, returns: np.ndarray, dates: list[str], target: str, seed: int
) -> dict:
    from qflab.backends.cudaq_programs import ProgramRunner

    started = time.perf_counter()
    qubits, layers = config["qubits"], config["layers"]
    bins = 2**qubits
    cut = int(config["train_fraction"] * len(returns))
    train, test = returns[:cut], returns[cut:]
    mean, spread = float(train.mean()), float(train.std(ddof=1))
    edges = np.linspace(
        mean - config["range_std"] * spread, mean + config["range_std"] * spread, bins + 1
    )
    centres = (edges[:-1] + edges[1:]) / 2
    train_histogram = _histogram(train, edges, config["smoothing"])
    test_histogram = _histogram(test, edges, 0.0)
    threshold_bin = int(np.searchsorted(edges, mean - config["tail_std"] * spread))  # bins below
    realized_tail = float(test_histogram[:threshold_bin].sum())

    models = {
        "training histogram (smoothed)": (train_histogram, bins - 1),
        "normal": (_binned(lambda x: norm.cdf(x, mean, spread), edges), 2),
    }
    freedom, location, scale = student_t.fit(train)
    models["Student t"] = (_binned(lambda x: student_t.cdf(x, freedom, location, scale), edges), 3)
    kde = gaussian_kde(train)
    models["Gaussian kernel density"] = (
        _binned(
            lambda x: np.array([kde.integrate_box_1d(-np.inf, v) for v in np.atleast_1d(x)]), edges
        ),
        len(train),
    )

    runner = ProgramRunner(seed, target)
    fits = [
        train_born_machine(
            runner, train_histogram, qubits, layers, seed + restart, config["iterations"]
        )
        for restart in range(config["restarts"])
    ]
    best = min(fits, key=lambda fit: fit["train_cross_entropy"])
    circuit = born_circuit(best["parameters"], qubits, layers)
    models["quantum circuit Born machine"] = (
        best["probabilities"],
        parameter_count(qubits, layers),
    )

    # Draw from the trained circuit as a device would, and use the draws for the risk number.
    counts = runner.sample(circuit, config["samples"])
    drawn = np.zeros(bins)
    for bits, count in counts.items():
        drawn[sum(int(b) << i for i, b in enumerate(bits))] += count
    sampled_tail = float(drawn[:threshold_bin].sum() / config["samples"])

    def quantile_bin(probabilities: np.ndarray) -> float:
        return float(centres[int(np.searchsorted(np.cumsum(probabilities), 0.05))])

    entropy_floor = float(
        -(test_histogram[test_histogram > 0] * np.log(test_histogram[test_histogram > 0])).sum()
    )
    rows = []
    for name, (probabilities, parameters) in models.items():
        safe = np.clip(probabilities, 1e-12, None)
        rows.append(
            {
                "model": name,
                "parameters": parameters,
                "test_cross_entropy": float(-(test_histogram * np.log(safe)).sum()),
                "train_cross_entropy": float(-(train_histogram * np.log(safe)).sum()),
                "test_total_variation": float(0.5 * np.abs(probabilities - test_histogram).sum()),
                "tail_probability": float(probabilities[:threshold_bin].sum()),
                "tail_error": float(probabilities[:threshold_bin].sum() - realized_tail),
                "var_95": quantile_bin(probabilities),
                "probabilities": [float(p) for p in probabilities],
            }
        )
    resources = runner.resources(circuit)
    return {
        "question": "Does a small quantum generative model describe held-out daily returns, and "
        "the tail risk that follows from them, better than standard density models?",
        "settings": config,
        "target": target,
        "seed": seed,
        "data": {
            "series": "equal-weight basket of the 8 ETFs, daily log returns from the frozen "
            "snapshot",
            "train_sessions": len(train),
            "test_sessions": len(test),
            "train_last_session": dates[cut],
            "test_first_session": dates[cut + 1],
            "bins": bins,
            "bin_edges": [float(e) for e in edges],
            "bin_centres": [float(c) for c in centres],
            "tail_threshold": float(edges[threshold_bin]),
        },
        "train_histogram": [float(p) for p in train_histogram],
        "test_histogram": [float(p) for p in test_histogram],
        "realized_tail_probability": realized_tail,
        "realized_var_95": quantile_bin(test_histogram),
        "test_entropy_floor": entropy_floor,
        "rows": rows,
        "born_machine": {
            "qubits": qubits,
            "layers": layers,
            "parameters": parameter_count(qubits, layers),
            "gates": resources["gates"],
            "depth": resources["depth"],
            "restarts": len(fits),
            "restart_train_cross_entropy": [fit["train_cross_entropy"] for fit in fits],
            "circuit_evaluations": sum(fit["circuit_evaluations"] for fit in fits),
            "loss_history": best["loss_history"],
            "training_note": "the loss uses exact output probabilities from the simulator; on a "
            "device each evaluation would be a batch of shots",
            "device_shots_estimate": sum(fit["circuit_evaluations"] for fit in fits)
            * config["shots_per_evaluation"],
            "shots_per_evaluation_assumed": config["shots_per_evaluation"],
            "sampled_tail_probability": sampled_tail,
            "samples_drawn": config["samples"],
        },
        "limits": "one short history; the held-out histogram itself is noisy, so the entropy "
        "floor and the realized tail frequency carry sampling error of their own",
        "evidence_label": "simulator performance: CUDA-Q statevector on a CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
