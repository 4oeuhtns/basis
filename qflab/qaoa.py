"""One saved portfolio decision: verify its QUBO, then build or run Classiq QAOA."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from importlib.metadata import distributions, version
from pathlib import Path
from types import SimpleNamespace

from qflab.optimizer_worker import energy, solve_request


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compile_qubo(financial: dict, penalty_scale: str = "1") -> dict:
    """Expand F(x) + A(sum(x)-K)^2 using exact integer coefficients.

    ``penalty_scale`` multiplies the provably sufficient penalty. Values below one are a
    declared experiment: infeasible bitstrings may then score better than feasible ones, so
    every sampled outcome must still be checked for feasibility independently.
    """
    symbols, k = financial["symbols"], financial["select_count"]
    if not isinstance(penalty_scale, str):
        raise ValueError("penalty_scale must be a decimal string")
    scale = Decimal(penalty_scale)
    if not scale.is_finite() or not 0 < scale <= 100:
        raise ValueError("penalty_scale must be in (0, 100]")
    # The financial score's entire range is at most this sum of absolute coefficients.
    # One unit more makes every infeasible state worse than every feasible state.
    sufficient = (
        1
        + sum(abs(int(v)) for v in financial["linear"].values())
        + sum(abs(int(p["coefficient"])) for p in financial["pairs"])
    )
    penalty = max(1, int((scale * sufficient).to_integral_value(rounding=ROUND_CEILING)))
    pairs = {(p["left"], p["right"]): int(p["coefficient"]) for p in financial["pairs"]}
    return {
        "schema_version": 1,
        "symbols": symbols,
        "select_count": k,
        "scale": financial["scale"],
        "penalty_integer": str(penalty),
        "normalization_integer": str(penalty),
        "constant": str(int(financial["constant"]) + penalty * k**2),
        "linear": {s: str(int(financial["linear"][s]) + penalty * (1 - 2 * k)) for s in symbols},
        "pairs": [
            {"left": a, "right": b, "coefficient": str(pairs.get((a, b), 0) + 2 * penalty)}
            for a, b in itertools.combinations(symbols, 2)
        ],
    }


def verify_qubo(financial: dict, qubo: dict, require_dominance: bool = True) -> dict:
    """Exhaustively check the penalty and the x=(1-z)/2 Ising translation.

    With ``require_dominance`` a penalty that lets any infeasible state beat a feasible one is
    an error; without it the audit records that fact and continues.
    """
    symbols, k = financial["symbols"], financial["select_count"]
    constant = int(qubo["constant"])
    linear = {s: int(v) for s, v in qubo["linear"].items()}
    pairs = qubo["pairs"]
    # Multiply the Ising coefficients by four to keep this check entirely integral.
    ising_constant4 = (
        4 * constant + 2 * sum(linear.values()) + sum(int(p["coefficient"]) for p in pairs)
    )
    ising_linear4 = {
        s: -2 * linear[s]
        - sum(int(p["coefficient"]) for p in pairs if s in (p["left"], p["right"]))
        for s in symbols
    }
    feasible, infeasible = [], []
    for bits in itertools.product((0, 1), repeat=len(symbols)):
        chosen = [s for s, bit in zip(symbols, bits, strict=True) if bit]
        score = energy(qubo, chosen)
        expected = energy(financial, chosen) + int(qubo["penalty_integer"]) * (len(chosen) - k) ** 2
        z = {s: 1 - 2 * bit for s, bit in zip(symbols, bits, strict=True)}
        ising4 = (
            ising_constant4
            + sum(ising_linear4[s] * z[s] for s in symbols)
            + sum(int(p["coefficient"]) * z[p["left"]] * z[p["right"]] for p in pairs)
        )
        if score != expected or ising4 != 4 * score:
            raise ValueError("QUBO/Ising translation does not match the financial model")
        (feasible if len(chosen) == k else infeasible).append(score)
    dominates = min(infeasible) > max(feasible)
    if require_dominance and not dominates:
        raise ValueError("Cardinality penalty is insufficient")
    best_feasible = min(feasible)
    return {
        "binary_states_checked": 2 ** len(symbols),
        "feasible_states": len(feasible),
        "qubo_and_ising_match": True,
        "all_infeasible_states_are_worse": dominates,
        "best_feasible_integer": str(best_feasible),
        "worst_feasible_integer": str(max(feasible)),
        "best_infeasible_integer": str(min(infeasible)),
        "uniform_feasible_probability": len(feasible) / 2 ** len(symbols),
        "uniform_optimal_probability": sum(score == best_feasible for score in feasible)
        / 2 ** len(symbols),
    }


def prepare(replay: Path, index: int, output: Path) -> None:
    records = json.loads(replay.read_bytes())
    if not 0 <= index < len(records):
        raise ValueError("Decision index outside the saved replay")
    record = records[index]
    # Rust's serde_json map keys are sorted; preserve exactly the bytes sent to Python.
    raw = json.dumps(record["request"], sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["request_sha256"]:
        raise ValueError("Saved request hash mismatch")
    reference = solve_request(record["request"], digest)
    qubo = compile_qubo(reference["canonical_model"])
    audit = verify_qubo(reference["canonical_model"], qubo)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_bytes(raw)
    for name, data in [("reference.json", reference), ("qubo.json", qubo), ("audit.json", audit)]:
        write_json(output / name, data)
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "request_sha256": digest,
            "files": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ["request.json", "reference.json", "qubo.json", "audit.json"]
            },
        },
    )
    print(f"Prepared {reference['data_cutoff']}: {audit['binary_states_checked']} states checked")
    print(f"Exact reference: {', '.join(reference['selected'])}")
    print(f"Experiment: {output}")


def load_experiment(path: Path) -> tuple[dict, dict]:
    manifest = json.loads((path / "manifest.json").read_bytes())
    names = {"request.json", "reference.json", "qubo.json", "audit.json"}
    if manifest["schema_version"] != 1 or set(manifest["files"]) != names:
        raise ValueError("Unsupported experiment manifest")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Experiment file changed: {name}")
    raw = (path / "request.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    reference = solve_request(json.loads(raw), digest)
    qubo = compile_qubo(reference["canonical_model"])
    if (
        digest != manifest["request_sha256"]
        or reference != json.loads((path / "reference.json").read_bytes())
        or qubo != json.loads((path / "qubo.json").read_bytes())
    ):
        raise ValueError("Experiment does not reproduce from its frozen request")
    verify_qubo(reference["canonical_model"], qubo)
    return reference, qubo


def classiq_problem(qubo: dict, layers: int):
    """Build locally. No synthesis, authentication, or execution happens here."""
    import pyomo.environ as pyo
    from classiq.applications.combinatorial_optimization import CombinatorialProblem

    model = pyo.ConcreteModel("three_fund_selection")
    symbols = qubo["symbols"]
    indices = {s: i for i, s in enumerate(symbols)}
    model.x = pyo.Var(range(len(symbols)), domain=pyo.Binary)
    divisor = int(qubo["normalization_integer"])
    model.cost = pyo.Objective(
        expr=int(qubo["constant"]) / divisor
        + sum(int(qubo["linear"][s]) / divisor * model.x[i] for s, i in indices.items())
        + sum(
            int(p["coefficient"])
            / divisor
            * model.x[indices[p["left"]]]
            * model.x[indices[p["right"]]]
            for p in qubo["pairs"]
        ),
        sense=pyo.minimize,
    )
    problem = CombinatorialProblem(pyo_model=model, num_layers=layers)
    # Check the actual SDK cost function, including its variable order and floating conversion.
    for bits in itertools.product((0, 1), repeat=len(symbols)):
        chosen = [s for s, bit in zip(symbols, bits, strict=True) if bit]
        actual = float(problem.cost_func(SimpleNamespace(x=list(bits))))
        if not math.isclose(actual, energy(qubo, chosen) / divisor, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Classiq cost function differs from verified QUBO")
    return problem


def summarize_samples(reference: dict, records: list[dict]) -> dict:
    """Rescore each decoded outcome ourselves; reject infeasible outcomes explicitly."""
    model = reference["canonical_model"]
    symbols, k = model["symbols"], model["select_count"]
    samples, feasible = [], []
    for record in records:
        bits = record["solution"]["x"]
        probability = float(record["probability"])
        if (
            len(bits) != len(symbols)
            or any(type(bit) is not int or bit not in (0, 1) for bit in bits)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("Invalid decoded quantum sample")
        selected = [s for s, bit in zip(symbols, bits, strict=True) if bit]
        score = energy(model, selected)
        row = {
            "bits": bits,
            "selected": selected,
            "probability": probability,
            "feasible": len(selected) == k,
            "financial_objective_integer": str(score),
        }
        samples.append(row)
        if row["feasible"] and probability > 0:
            feasible.append((score, tuple(selected)))
    if not math.isclose(sum(s["probability"] for s in samples), 1.0, abs_tol=1e-8):
        raise ValueError("Sample probabilities do not sum to one")
    best = min(feasible) if feasible else None
    optimum = int(reference["objective_integer"])
    return {
        "status": "feasible" if best else "no_feasible_sample",
        "exact_selected": reference["selected"],
        "best_selected": list(best[1]) if best else None,
        "best_financial_objective_integer": str(best[0]) if best else None,
        "objective_gap_integer": str(best[0] - optimum) if best else None,
        "objective_scale": model["scale"],
        "feasible_probability": sum(s["probability"] for s in samples if s["feasible"]),
        "optimal_probability": sum(
            s["probability"]
            for s in samples
            if s["feasible"] and int(s["financial_objective_integer"]) == optimum
        ),
        "samples": samples,
    }


def run(path: Path, layers: int, shots: int, maxiter: int, build_only: bool) -> Path:
    from classiq import get_circuit_metrics, write_qmod
    from classiq.execution import ClassiqBackendPreferences, ExecutionPreferences

    if not 1 <= layers <= 3 or not 100 <= shots <= 4096 or not 1 <= maxiter <= 100:
        raise ValueError("Use 1–3 layers, 100–4096 shots, and 1–100 optimizer iterations")
    reference, qubo = load_experiment(path)
    output = path / datetime.now(UTC).strftime("attempt-%Y%m%dT%H%M%S.%fZ")
    output.mkdir()
    settings = {
        "schema_version": 1,
        "status": "building",
        "backend": "simulator",
        "classiq_version": version("classiq"),
        "python_version": sys.version,
        "layers": layers,
        "shots_per_evaluation": shots,
        "maxiter": maxiter,
        "quantile": 1.0,
        "request_sha256": reference["request_sha256"],
    }
    sources = output / "sources"
    sources.mkdir()
    settings["source_sha256"] = {}
    for name in ["qaoa.py", "optimizer_worker.py"]:
        raw = Path(__file__).with_name(name).read_bytes()
        (sources / name).write_bytes(raw)
        settings["source_sha256"][name] = hashlib.sha256(raw).hexdigest()
    write_json(output / "run.json", settings)
    write_json(
        output / "environment.json",
        {package.metadata["Name"]: package.version for package in distributions()},
    )
    problem = None
    try:
        problem = classiq_problem(qubo, layers)
        model = problem.get_model()
        (output / "model.json").write_text(str(model))
        write_qmod(model, "portfolio", directory=output, decimal_precision=16)
        settings["status"] = "built_offline"
        if not build_only:
            started = datetime.now(UTC)
            program = problem.get_qprog()
            (output / "program.json").write_text(program.model_dump_json())
            metrics = get_circuit_metrics(program)
            settings["circuit_width"] = metrics.width
            settings["circuit_depth"] = metrics.depth
            preferences = ExecutionPreferences(
                num_shots=shots,
                backend_preferences=ClassiqBackendPreferences(backend_name="simulator"),
            )
            params = problem.optimize(
                preferences, maxiter=maxiter, quantile=1.0, initial_params=[1.0] * (2 * layers)
            )
            samples = problem.sample(params).to_dict(orient="records")
            report = summarize_samples(reference, samples)
            report["uniform_baseline"] = verify_qubo(reference["canonical_model"], qubo)
            report["parameters"] = params
            report["optimizer_cost_trace"] = problem.cost_trace
            report["wall_seconds_including_synthesis"] = (
                datetime.now(UTC) - started
            ).total_seconds()
            write_json(output / "result.json", report)
            settings["status"] = "completed"
        return output
    except Exception as error:
        settings["status"] = "failed"
        settings["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "run.json", settings)
        if problem is not None:
            problem.teardown()
        print(f"QAOA evidence: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Freeze and verify one saved monthly request")
    preparation.add_argument("--replay", type=Path, required=True)
    preparation.add_argument("--index", type=int, default=0)
    preparation.add_argument("--output", type=Path, required=True)
    execution = commands.add_parser(
        "run", help="Build QAOA locally or run on the Classiq simulator"
    )
    execution.add_argument("experiment", type=Path)
    execution.add_argument("--layers", type=int, default=1)
    execution.add_argument("--shots", type=int, default=1024)
    execution.add_argument("--maxiter", type=int, default=20)
    execution.add_argument(
        "--build-only", action="store_true", help="Export Qmod without cloud calls"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.replay, args.index, args.output)
    else:
        run(args.experiment, args.layers, args.shots, args.maxiter, args.build_only)


if __name__ == "__main__":
    main()
