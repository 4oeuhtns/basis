"""Quantum-assisted scenario sampling on a small correlated-default model (plan 13.2, R1).

One specified target distribution is frozen and every sampler is judged against its exact
enumeration. Two views:

* exact: the spectral gap of each sampler's Markov chain, across temperatures. The quantum
  proposal matrix is computed from simulator statevectors (privileged access);
* empirical: chains from two different starting states, with convergence, autocorrelation,
  effective sample size and tail-risk error.

Fewer chain steps is not the same as faster: the report carries wall time and gates per
quantum proposal next to effective samples.
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from qflab.methods.mcmc import (
    DefaultModel,
    QuantumProposal,
    autocorrelation_time,
    gibbs_chain,
    gibbs_matrix,
    local_move,
    local_proposal,
    metropolis_chain,
    metropolis_matrix,
    parallel_tempering,
    sector_model,
    spectral_gap,
    total_variation,
    uniform_move,
    uniform_proposal,
)

SAMPLERS = ["local", "uniform", "gibbs", "parallel_tempering", "quantum"]


def _model(config: dict, seed: int) -> DefaultModel:
    return sector_model(
        config["sectors"], config["bias"], config["within"], config["across"], config["beta"], seed
    )


def _chain(job: dict) -> dict:
    """One chain in a worker process."""
    model = _model(job["config"], job["model_seed"])
    rng = np.random.default_rng(job["seed"])
    steps, start, name = job["config"]["steps"], job["start"], job["sampler"]
    started = time.perf_counter()
    work, extra = steps, {}
    if name == "local":
        chain = metropolis_chain(model, local_move(model.n, rng), steps, start, rng)
    elif name == "uniform":
        chain = metropolis_chain(model, uniform_move(model.n, rng), steps, start, rng)
    elif name == "gibbs":
        chain = gibbs_chain(model, steps, start, rng)
    elif name == "parallel_tempering":
        chain, work = parallel_tempering(model, job["config"]["tempering_betas"], steps, start, rng)
    else:
        from qflab.backends.cudaq_programs import ProgramRunner

        proposal = QuantumProposal(model, ProgramRunner(job["seed"], job["target"]), job["seed"])
        chain = metropolis_chain(model, proposal, steps, start, rng)
        extra = {
            "circuit_executions": proposal.executions,
            "mean_gates_per_proposal": proposal.gates / proposal.executions,
        }
    return {
        "sampler": name,
        "start": start,
        "chain": chain.tolist(),
        "work": work,
        "seconds": time.perf_counter() - started,
        **extra,
    }


def run_mcmc_study(config: dict, target: str, seed: int, workers: int) -> dict:
    from qflab.backends.cudaq_programs import ProgramRunner

    started = time.perf_counter()
    model = _model(config, seed)
    n, size = model.n, 2**model.n
    pi = model.distribution()
    defaults, losses = model.defaults(), model.losses()
    tail = defaults >= config["tail_defaults"]
    exact = {
        "names": n,
        "states": size,
        "beta": model.beta,
        "tail_probability": float(pi[tail].sum()),
        "expected_loss": float(pi @ losses),
        "expected_defaults": float(pi @ defaults),
        "default_count_distribution": [float(pi[defaults == k].sum()) for k in range(n + 1)],
        "largest_state_probability": float(pi.max()),
    }

    # ---- exact spectral gaps across temperature --------------------------------------
    proposal = QuantumProposal(model, ProgramRunner(seed, target), seed)
    settings = [proposal.draw_settings() for _ in range(config["proposal_settings"])]
    matrix_started = time.perf_counter()
    quantum = proposal.matrix(settings)
    matrix_seconds = time.perf_counter() - matrix_started
    symmetry = float(np.abs(quantum - quantum.T).max())
    gaps = []
    for beta in config["betas"]:
        scaled = model.with_beta(beta)
        weights = scaled.distribution()
        gaps.append(
            {
                "beta": beta,
                "tail_probability": float(weights[tail].sum()),
                "local": spectral_gap(metropolis_matrix(scaled, local_proposal(n)), weights),
                "uniform": spectral_gap(metropolis_matrix(scaled, uniform_proposal(n)), weights),
                "gibbs": spectral_gap(gibbs_matrix(scaled), weights),
                "quantum": spectral_gap(metropolis_matrix(scaled, quantum), weights),
            }
        )

    # ---- empirical chains -------------------------------------------------------------
    starts = [0, size - 1]  # nobody defaults / everybody defaults
    jobs = [
        {
            "sampler": name,
            "config": config,
            "model_seed": seed,
            "seed": seed + 1000 * index + 17 * SAMPLERS.index(name),
            "start": starts[index % 2],
            "target": target,
        }
        for name in SAMPLERS
        for index in range(config["chains"])
    ]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        chains = list(pool.map(_chain, jobs))

    steps, burn = config["steps"], config["burn_in"]
    checkpoints = sorted({int(v) for v in np.geomspace(20, steps, 12)})
    rng = np.random.default_rng(seed)
    independent = [
        float(np.mean([total_variation(rng.choice(size, size=c, p=pi), pi) for _ in range(40)]))
        for c in checkpoints
    ]
    rows = []
    for name in SAMPLERS:
        group = [c for c in chains if c["sampler"] == name]
        arrays = [np.array(c["chain"]) for c in group]
        kept = [a[burn:] for a in arrays]
        taus = [autocorrelation_time(defaults[a]) for a in kept]
        finite = [t for t in taus if np.isfinite(t)]
        effective = [len(a) / t if np.isfinite(t) else 0.0 for a, t in zip(kept, taus, strict=True)]
        tails = [float(tail[a].mean()) for a in kept]
        pooled = np.concatenate(kept)
        total_effective = float(sum(effective))
        deviation = None
        if total_effective > 0:
            spread = (
                exact["tail_probability"] * (1 - exact["tail_probability"]) / total_effective
            ) ** 0.5
            deviation = float((tail[pooled].mean() - exact["tail_probability"]) / spread)
        seconds = float(np.mean([c["seconds"] for c in group]))
        rows.append(
            {
                "sampler": name,
                "chains": len(group),
                "steps": steps,
                "burn_in": burn,
                "acceptance_rate": float(np.mean([np.mean(a[1:] != a[:-1]) for a in arrays])),
                "autocorrelation_time": float(np.median(finite)) if finite else None,
                "stuck_chains": len(taus) - len(finite),
                "effective_samples_per_chain": float(np.mean(effective)),
                "effective_samples_per_second": float(np.mean(effective)) / seconds,
                "effective_samples_per_1000_updates": 1000
                * float(np.mean(effective))
                / group[0]["work"],
                "updates_per_chain": group[0]["work"],
                "seconds_per_chain": seconds,
                "tail_estimate_mean": float(np.mean(tails)),
                "tail_mean_abs_error": float(
                    np.mean(np.abs(np.array(tails) - exact["tail_probability"]))
                ),
                "expected_loss_mean_abs_error": float(
                    np.mean([abs(losses[a].mean() - exact["expected_loss"]) for a in kept])
                ),
                "pooled_total_variation": total_variation(pooled, pi),
                "pooled_tail_z_score": deviation,
                "consistent_with_target": deviation is not None and abs(deviation) < 3,
                "final_tv_by_start": {
                    "none_default": float(
                        np.mean(
                            [
                                total_variation(a, pi)
                                for a, c in zip(arrays, group, strict=True)
                                if c["start"] == 0
                            ]
                        )
                    ),
                    "all_default": float(
                        np.mean(
                            [
                                total_variation(a, pi)
                                for a, c in zip(arrays, group, strict=True)
                                if c["start"] != 0
                            ]
                        )
                    ),
                },
                "tv_curve": [
                    float(np.mean([total_variation(a[:c], pi) for a in arrays]))
                    for c in checkpoints
                ],
                "circuit_executions_per_chain": group[0].get("circuit_executions"),
                "mean_gates_per_proposal": float(
                    np.mean([c["mean_gates_per_proposal"] for c in group])
                )
                if name == "quantum"
                else None,
            }
        )
    return {
        "question": "For one frozen correlated-default model, does a quantum proposal give more "
        "effective samples per unit of total compute while keeping the target distribution?",
        "settings": config,
        "target": target,
        "seed": seed,
        "model": {
            "kind": "specified sector model of default indicators (not fitted to data)",
            "fields": [float(v) for v in model.fields],
            "couplings": [[float(v) for v in row] for row in model.couplings],
            "exposures": [float(v) for v in model.exposures],
            "tail_event": f"at least {config['tail_defaults']} of {n} names default",
        },
        "exact": exact,
        "gaps": gaps,
        "quantum_proposal": {
            "settings_averaged": len(settings),
            "largest_asymmetry": symmetry,
            "symmetric": symmetry < 1e-9,
            "matrix_seconds": matrix_seconds,
            "note": "gamma uniform in [0.25, 0.6], time uniform in [2, 20], palindromic Trotter "
            "steps of 0.8; exact matrix from simulator statevectors (privileged)",
        },
        "checkpoints": checkpoints,
        "independent_sample_tv": independent,
        "samplers": rows,
        "cost_note": "seconds are simulator wall time on one CPU core per chain; a device would "
        "add queue, compilation and readout time per proposal, which is not modeled here",
        "evidence_label": "simulator performance: CUDA-Q statevector on a CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
