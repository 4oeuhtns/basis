"""Paired historical backtests: run the Rust engine, then read its saved ledgers.

The exact solver and QAOA are run from configurations that are checked to differ only in the
search method. An equal-weight monthly portfolio of all funds is the reference benchmark.
Financial numbers come from each run's own ``equity.csv`` and ``fills.json``.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path

from qflab.experiments.metrics import financial_metrics, reconcile_cash

# Fields allowed to differ between the paired configurations.
METHOD_ONLY = {
    "optimizer.solver",
    "optimizer.python_path",
    "optimizer.timeout_seconds",
    "optimizer.replay_path",
    "output_root",
}


def _read_json(path: Path):
    return json.loads(path.read_bytes())


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def derive_config(source: Path, destination: Path, output_root: Path, changes: dict) -> Path:
    """Copy a backtest config with absolute input paths, a new output root, and ``changes``."""
    config = _read_json(source)
    base = source.resolve().parent
    for asset in config["assets"]:
        asset["data_path"] = str((base / asset["data_path"]).resolve())
    if "snapshot_manifest" in config:
        config["snapshot_manifest"] = str((base / config["snapshot_manifest"]).resolve())
    if "optimizer" in config:
        # Keep the venv's symlink path: resolving it would bypass the environment's packages.
        config["optimizer"]["python_path"] = str(
            (base / config["optimizer"]["python_path"]).absolute()
        )
    config["output_root"] = str(output_root.resolve())
    config.update(changes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2) + "\n")
    return destination


def run_engine(repo: Path, arguments: list[str], pattern: str, timeout: int = 3600) -> Path:
    """Run the Rust backtester and return the directory it reports."""
    launcher = repo / "scripts" / "cargo-local"
    command = [
        str(launcher) if launcher.exists() else "cargo",
        "run",
        "--locked",
        "--offline",
        "--quiet",
        "--",
        *arguments,
    ]
    done = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=timeout)
    found = re.search(pattern + r"\s+(.+)", done.stdout)
    if done.returncode != 0 or not found:
        raise RuntimeError(f"Backtester failed ({done.returncode}): {done.stderr[-2000:]}")
    return Path(found.group(1).strip())


def _flatten(value, prefix="") -> dict:
    if isinstance(value, dict):
        flat = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: value}


def config_differences(first: dict, second: dict) -> list[str]:
    """Top-level-or-optimizer fields whose values differ, e.g. ``optimizer.solver``."""

    def view(config):
        flat = {k: v for k, v in config.items() if k != "optimizer"}
        flat.update({f"optimizer.{k}": v for k, v in config.get("optimizer", {}).items()})
        return flat

    a, b = view(first), view(second)
    return sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))


def load_run(path: Path, label: str, repo: Path) -> dict:
    """Everything the report needs from one saved run directory."""
    equity = _read_csv(path / "equity.csv")
    fills = _read_json(path / "fills.json")
    manifest = _read_json(path / "manifest.json")
    summary = _read_json(path / "summary.json")
    config = _read_json(path / "resolved_config.json")
    start = float(config["starting_cash"])
    symbols = sorted({row["instrument_id"] for row in _read_csv(path / "holdings.csv")})
    by_day: dict[str, dict] = {}
    for row in _read_csv(path / "holdings.csv"):
        by_day.setdefault(row["timestamp"], {})[row["instrument_id"]] = float(row["market_value"])
    weights = {
        s: [by_day[r["timestamp"]].get(s, 0.0) / float(r["equity"]) for r in equity]
        for s in symbols
    }
    decisions = []
    log = path / "optimizer_log.json"
    for record in _read_json(log) if log.exists() else []:
        response = record.get("response") or {}
        diagnostics = response.get("diagnostics") or {}
        scale = int((response.get("canonical_model") or {}).get("scale", 1))
        gap = diagnostics.get("objective_gap_integer")
        decisions.append(
            {
                "data_cutoff": record["request"]["data_cutoff"],
                "available_at": record["available_at"],
                "status": record["status"],
                "fallback_reason": record["fallback_reason"],
                "delivery_status": record["delivery_status"],
                "method": response.get("method"),
                "selected": [s.split(".")[0] for s in response.get("selected", [])],
                "elapsed_wall_seconds": record["elapsed_wall_seconds"],
                "exact_selected": [
                    s.split(".")[0]
                    for s in (diagnostics.get("exact_reference") or {}).get("selected", [])
                ]
                or None,
                "objective_gap": None if gap is None else int(gap) / scale,
                "feasible_probability": diagnostics.get("feasible_probability"),
                "optimal_probability": diagnostics.get("optimal_probability"),
                "total_shots": (diagnostics.get("resources") or {}).get("total_shots"),
                "evaluated_portfolios": diagnostics.get("evaluated_portfolios"),
            }
        )
    try:
        shown = str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        shown = str(path)
    return {
        "label": label,
        "run_directory": shown,
        "method": manifest.get("optimizer_method", config["strategy"]),
        "status": manifest["status"],
        "wall_seconds": manifest.get("wall_seconds"),
        "fee_bps": config["fee_bps"],
        "snapshot_sha256": manifest.get("historical_manifest_sha256"),
        "dates": [row["timestamp"][:10] for row in equity],
        "equity": [float(row["equity"]) for row in equity],
        "cash": [float(row["cash"]) for row in equity],
        "symbols": [s.split(".")[0] for s in symbols],
        "weights": {s.split(".")[0]: [round(w, 5) for w in weights[s]] for s in symbols},
        "metrics": financial_metrics(equity, fills, start),
        "cash_reconciliation": reconcile_cash(
            [{**r, "cash": float(r["cash"])} for r in equity], fills, start
        ),
        "fallbacks": summary.get("optimizer_fallbacks", 0),
        "errors": summary.get("errors", []),
        "decisions": decisions,
        "ending_holdings": [
            {
                "symbol": h["instrument_id"].split(".")[0],
                "units": h["units"],
                "market_value": float(h["market_value"]),
            }
            for h in summary["ending_holdings"]
            if float(h["units"]) > 0
        ],
        "config": config,
    }


def run_portfolio_study(config: dict, repo: Path, lab_dir: Path, config_dir: Path) -> dict:
    started = time.perf_counter()
    backtests = lab_dir / "backtests"
    exact_config = derive_config(
        config_dir / config["exact_config"], lab_dir / "configs" / "exact.json", backtests, {}
    )
    quantum_config = derive_config(
        config_dir / config["qaoa_config"], lab_dir / "configs" / "qaoa.json", backtests, {}
    )
    differences = config_differences(_read_json(exact_config), _read_json(quantum_config))
    comparison = run_engine(repo, ["compare", str(exact_config)], "Comparison saved:")
    report = _read_json(comparison / "comparison.json")
    exact = load_run(comparison / report["optimized"]["run_directory"], "Exact enumeration", repo)
    benchmark = load_run(
        comparison / report["monthly"]["run_directory"], "Equal-weight monthly (8 funds)", repo
    )
    quantum_started = time.perf_counter()
    quantum = load_run(run_engine(repo, [str(quantum_config)], "Run saved:"), "QAOA (CUDA-Q)", repo)
    quantum_seconds = time.perf_counter() - quantum_started

    sensitivity = []
    for fee in config["fee_bps_sensitivity"]:
        if fee == exact["fee_bps"]:
            run = exact
        else:
            derived = derive_config(
                config_dir / config["exact_config"],
                lab_dir / "configs" / f"exact_fee_{fee}bps.json",
                backtests,
                {"fee_bps": fee},
            )
            run = load_run(
                run_engine(repo, [str(derived)], "Run saved:"), f"Exact, {fee} bps", repo
            )
        sensitivity.append(
            {
                "fee_bps": fee,
                "ending_equity": run["metrics"]["ending_equity"],
                "total_return": run["metrics"]["total_return"],
                "fees": run["metrics"]["fees"],
                "turnover": run["metrics"]["turnover"],
                "selections": [d["selected"] for d in run["decisions"]],
                "run_directory": run["run_directory"],
            }
        )

    pairs = list(zip(exact["decisions"], quantum["decisions"], strict=False))
    same = [a["selected"] == b["selected"] and b["status"] == "feasible" for a, b in pairs]
    first_split = next(
        (a["data_cutoff"][:10] for (a, _), ok in zip(pairs, same, strict=True) if not ok), None
    )
    return {
        "question": "Does the search method change subsequent historical results after costs, "
        "when data, timing, forecasts, objective and costs are identical?",
        "runs": {"exact": exact, "qaoa": quantum, "benchmark": benchmark},
        "pairing": {
            "config_differences": differences,
            "differs_only_in_method": set(differences) <= METHOD_ONLY,
            "same_snapshot": exact["snapshot_sha256"] == quantum["snapshot_sha256"],
            "decisions": len(pairs),
            "identical_selections": sum(same),
            "first_divergence": first_split,
            "qaoa_matched_own_inputs": sum(
                d["selected"] == d["exact_selected"]
                for d in quantum["decisions"]
                if d["status"] == "feasible"
            ),
            "ending_equity_difference": quantum["metrics"]["ending_equity"]
            - exact["metrics"]["ending_equity"],
            "note": "after the first differing selection the two accounts hold different "
            "positions, so later decisions are solved from different states by design",
        },
        "fee_sensitivity": sensitivity,
        "timing": {
            "qaoa_backtest_seconds": quantum_seconds,
            "exact_solver_seconds_per_decision": [
                d["elapsed_wall_seconds"] for d in exact["decisions"]
            ],
            "qaoa_solver_seconds_per_decision": [
                d["elapsed_wall_seconds"] for d in quantum["decisions"]
            ],
            "note": "solver wall time is measured separately from the simulated clock; each "
            "answer becomes available after the configured simulated delay",
        },
        "interpretation": "one retrospective year on adjusted research units; descriptive only, "
        "no claim about future performance or about either method being better",
        "elapsed_seconds": time.perf_counter() - started,
    }
