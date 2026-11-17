"""Research-track studies end to end on tiny configurations, run on CUDA-Q.

Each study must produce a plain-JSON record with the fields the tearsheet reads, and keep the
properties its design promises (feasibility, chronology, symmetric proposals, and so on).
"""

import hashlib
import json
from datetime import date, timedelta

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("qflab.backends.cudaq_programs", exc_type=ImportError)

from test_optimizer import request as optimizer_request  # noqa: E402

from qflab.optimizer_worker import solve_request  # noqa: E402


def long_request(seed, returns=40):
    """The optimizer fixture with a seeded history long enough for a full-rank covariance."""
    data = optimizer_request()
    rng = np.random.default_rng(seed)
    symbols = data["symbols"]
    prices = np.full(len(symbols), 100.0)
    day = date(2025, 1, 2)
    history = []
    for t in range(returns + 1):
        if t:
            prices = prices * (
                1 + 0.0004 * np.arange(len(symbols)) + rng.normal(0, 0.01, len(symbols))
            )
        history.append(
            {
                "timestamp": f"{day}T21:00:00+00:00",
                "closes": {s: f"{p:.6f}" for s, p in zip(symbols, prices, strict=True)},
            }
        )
        day += timedelta(days=1)
    data["history"] = history
    data["data_cutoff"] = history[-1]["timestamp"]
    data["settings"] = {**data["settings"], "lookback_returns": returns}
    return data


def records(count=2):
    out = []
    for seed in range(count):
        data = long_request(seed)
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        out.append(
            {
                "status": "feasible",
                "request": data,
                "request_sha256": digest,
                "response": solve_request(data, digest),
            }
        )
    return out


def plain(report):
    json.dumps(report, allow_nan=False)
    return report


def test_harder_risk_models_keep_targets_apart_and_verify_their_circuits():
    from qflab.experiments.risk_models import run_risk_models

    config = {
        "confidence": 0.95,
        "seeds": 3,
        "shots": 100,
        "epsilon": 0.02,
        "mlae_circuits": 4,
        "sobol": {"samples_per_replicate": 1024, "replicates": 4},
        "models": [
            {
                "kind": "two_factor",
                "label": "loss",
                "weights": [0.6, 0.4],
                "vols": [0.08, 0.05],
                "correlation": 0.5,
                "option_strike": 0.03,
                "option_notional": 0.6,
                "threshold": 0.04,
                "qubits_per_factor": 2,
                "truncation_std": 3.0,
                "payoff": "indicator",
            },
            {
                "kind": "asian_call",
                "label": "asian",
                "spot": 100.0,
                "strike": 100.0,
                "rate": 0.03,
                "volatility": 0.2,
                "maturity": 1.0,
                "steps": 2,
                "qubits_per_step": 2,
                "truncation_std": 3.0,
            },
        ],
    }
    report = plain(run_risk_models(config, "qpp-cpu", 3))
    assert [row["label"] for row in report["rows"]] == ["loss", "asian"]
    for row in report["rows"]:
        assert row["circuit_matches_discrete_sum"]
        assert row["model_error"] == pytest.approx(
            row["exact_discrete"] - row["continuous_reference"]["estimate"]
        )
        for method in ("iqae", "mlae", "monte_carlo"):
            assert 0 <= row[method]["coverage"] <= 1 and row[method]["median_work"] > 0
        assert row["monte_carlo"]["median_work"] == row["matched_work"]
    assert report["rows"][0]["unit"] == "probability" and report["rows"][1]["unit"] == "value"


def test_qaoa_study_variants_stay_feasible_and_keep_chronology():
    from qflab.experiments.qaoa_study import run_qaoa_study

    config = {
        "decisions": "all",
        "layers": [1],
        "penalty_scales": ["1"],
        "seeds": 1,
        "shots": 256,
        "final_shots": 512,
        "grid": 4,
        "maxiter": 8,
        "swap_search": {"seeds": 5, "max_evaluations": 56},
        "xy_mixer": {"layers": [1], "seeds": 1},
        "warm_start": {"layers": 1, "penalty_scale": "1", "seeds": 1},
    }
    report = plain(run_qaoa_study(config, records(2), "qpp-cpu", 1, workers=2))
    mixer = report["variants"]["xy_mixer"]
    assert mixer["summary"][0]["feasible_probability"] == pytest.approx(1.0, abs=1e-9)
    assert all(run["mixer"] == "xy" and run["status"] == "feasible" for run in mixer["runs"])
    warm = report["variants"]["warm_start"]
    assert [row["start"] for row in warm["summary"]] == ["cold", "warm", "frozen"]
    frozen = [r for r in warm["runs"] if r["mode"] == "frozen"]
    assert [r["decision"] for r in frozen] == [0, 1]  # dates in order
    assert frozen[0]["start"] == "cold" and frozen[1]["start"] == "frozen"
    assert frozen[1]["objective_evaluations"] == 0
    assert frozen[1]["total_shots"] == config["final_shots"]
    assert warm["summary"][2]["median_total_shots"] == config["final_shots"]


def test_grover_adaptive_search_study_reports_oracle_calls_and_rounding():
    from qflab.experiments.gas_study import run_gas_study

    config = {"decisions": 1, "seeds": 1, "value_bits": 6, "patience": 3, "max_oracle_calls": 10}
    report = plain(run_gas_study(config, records(1), "qpp-cpu", 2, workers=1))
    assert report["search_space"] == 56 and report["summary"]["runs"] == 1
    run = report["runs"][0]
    assert run["status"] in ("patience_exhausted", "budget_exhausted") and len(run["selected"]) == 3
    assert run["oracle_calls"] <= 10 and run["qubits"] == 14
    assert {row["value_bits"] for row in report["quantization"]} >= {6, 8}


def test_linear_algebra_study_separates_state_and_allocation_modes():
    from qflab.experiments.linear_algebra import run_linear_algebra

    config = {
        "max_qubits": 9,
        "synthetic": [{"size": 2, "kappa": 4}],
        "covariance_delta_fractions": [0.5],
        "hhl_extra_clock_qubits": [1],
        "qsvt_epsilons": [1e-2],
        "trotter": {"system": "synthetic 2x2, kappa 4", "extra_clock_qubits": 1, "steps": [1]},
        "readout": {
            "system": "ETF covariance 8x8, delta = 0.5 x mean variance",
            "hhl_extra_clock_qubits": 1,
            "qsvt_epsilon": 1e-2,
            "shots": [2000],
            "seeds": 2,
        },
        "sweep": {
            "sizes": [2],
            "kappas": [2, 4],
            "hhl_extra_clock_qubits": 1,
            "qsvt_epsilon": 1e-2,
            "delta_fractions": [0.1, 1.0],
        },
    }
    report = plain(run_linear_algebra(config, records(1), "qpp-cpu", 4))
    assert [s["kind"] for s in report["systems"]] == ["synthetic", "covariance"]
    assert report["hhl_controls"][0]["infidelity"] < 1e-10  # exactly representable spectrum
    assert all(row["fidelity"] > 0.99 for row in report["hhl"] + report["qsvt"])
    for row in report["readout"]:
        assert row["preparations"] == row["shots_per_setting"] * row["settings"]
        assert row["weights_sum_to_one"]
    assert report["trading_use"].startswith("none")
    kappas = [r["condition_number"] for r in report["regularization"]]
    assert kappas[0] > kappas[1]  # more regularization, better conditioned


def test_mcmc_study_proposal_is_symmetric_and_chains_are_scored_against_the_target():
    from qflab.experiments.mcmc_study import run_mcmc_study

    config = {
        "sectors": [0, 0, 1, 1],
        "bias": 0.25,
        "within": 0.5,
        "across": 0.15,
        "beta": 2.0,
        "betas": [1.0, 2.0],
        "proposal_settings": 2,
        "chains": 2,
        "steps": 400,
        "burn_in": 50,
        "tail_defaults": 2,
        "tempering_betas": [0.5, 1.0, 2.0],
    }
    report = plain(run_mcmc_study(config, "qpp-cpu", 5, workers=2))
    assert report["quantum_proposal"]["symmetric"]
    assert {row["sampler"] for row in report["samplers"]} == {
        "local",
        "uniform",
        "gibbs",
        "parallel_tempering",
        "quantum",
    }
    assert sum(report["exact"]["default_count_distribution"]) == pytest.approx(1.0)
    quantum = next(row for row in report["samplers"] if row["sampler"] == "quantum")
    assert quantum["circuit_executions_per_chain"] == 400
    assert len(quantum["tv_curve"]) == len(report["checkpoints"])


def test_spectral_study_compares_quantum_and_classical_coordinate_choices():
    from qflab.experiments.spectral_study import run_spectral_study

    config = {
        "seeds": 2,
        "shots": 50,
        "time_step": 1.0,
        "krylov_states": 3,
        "subspace_sizes": [4],
        "families": [
            {"label": "clusters", "kind": "clustered", "size": 16, "clusters": 4, "gap": 2.0}
        ],
        "probe": {
            "family": "clusters",
            "subspace_size": 4,
            "gaps": [1.5],
            "shots": [20],
            "trotter_steps": [1],
        },
    }
    report = plain(run_spectral_study(config, records(1), "qpp-cpu", 6))
    methods = {row["method"] for row in report["rows"] if row["family"] == "clusters"}
    assert "largest variances (classical)" in methods and "SKQD, uniform start" in methods
    assert "power iteration (classical)" in methods
    assert "power" in report["gap_sweep"][0]
    for row in report["rows"]:
        assert row["vector_error"] >= row["best_possible_vector_error"] - 1e-9
    assert [r["family"] for r in report["references"]][-1].startswith("ETF")


def test_relaxation_study_validates_the_relaxation_on_every_case():
    from qflab.experiments.relaxation_study import run_relaxation_study

    config = {
        "decisions": "all",
        "densities": [0.25, 1.0],
        "seeds": 1,
        "shots": 100,
        "subspace_size": 8,
        "times": [0, 1],
        "trotter_steps": 1,
    }
    report = plain(run_relaxation_study(config, records(2), "qpp-cpu", 7))
    assert report["validation"]["encoded_states_reproduce_objective"]
    assert report["validation"]["relaxed_ground_energy_is_lower_bound"]
    assert report["validation"]["cases"] == 4
    dense = next(row for row in report["summary"] if row["density"] == 1.0)
    assert dense["diagonal_encoding"] and dense["max_qubits"] == 8
    assert all(run["gap"] >= 0 for run in report["runs"])


def synthetic_returns(sessions=260, seed=8):
    rng = np.random.default_rng(seed)
    regime = (np.arange(sessions) // 40) % 2
    returns = rng.normal(0.0003, np.where(regime, 0.02, 0.007))
    dates = [str(date(2024, 1, 1) + timedelta(days=i)) for i in range(sessions + 1)]
    return returns, dates


def test_kernel_study_splits_chronologically_with_purged_boundaries():
    from qflab.experiments.kernel_study import run_kernel_study

    returns, dates = synthetic_returns()
    config = {
        "horizon": 10,
        "stride": 4,
        "train_fraction": 0.55,
        "validation_fraction": 0.2,
        "ridges": [0.1],
        "quantum_scales": [0.5],
        "quantum_repetitions": [1],
        "rbf_gammas": [0.3],
        "polynomial_degrees": [1],
        "shots": 100,
    }
    report = plain(run_kernel_study(config, returns, dates, "qpp-cpu", 9))
    data = report["data"]
    assert data["train_last_session"] < data["test_first_session"]
    assert data["purged_per_boundary"] == 3
    methods = [row["method"] for row in report["rows"]]
    assert methods[0] == "quantum fidelity kernel" and methods[-1].endswith("sampled")
    train = data["examples"]["train"]
    assert report["cost"]["train_pair_circuits"] == train * (train - 1) // 2  # symmetric Gram


def test_generative_study_scores_models_on_later_data_only():
    from qflab.experiments.generative_study import run_generative_study

    returns, dates = synthetic_returns()
    config = {
        "qubits": 3,
        "layers": 2,
        "train_fraction": 0.6,
        "range_std": 3.0,
        "smoothing": 0.5,
        "tail_std": 1.5,
        "iterations": 40,
        "restarts": 1,
        "samples": 2000,
        "shots_per_evaluation": 1000,
    }
    report = plain(run_generative_study(config, returns, dates, "qpp-cpu", 10))
    assert report["data"]["train_sessions"] + report["data"]["test_sessions"] == len(returns)
    names = [row["model"] for row in report["rows"]]
    assert names[-1] == "quantum circuit Born machine" and "normal" in names
    for row in report["rows"]:
        assert sum(row["probabilities"]) == pytest.approx(1.0)
        assert row["test_cross_entropy"] >= report["test_entropy_floor"] - 1e-9
