"""Experiment runner: one configuration in, one immutable run directory out.

    python -m qflab.lab run configs/lab.json
    python -m qflab.lab run configs/lab.json --only risk_lab --reuse runs/lab/EARLIER
    python -m qflab.lab tearsheet runs/lab/SOME_RUN

A run directory holds the resolved configuration, a manifest (code commit, versions, seeds,
timings, section statuses), ``results.json``, the backtests it launched, and
``tearsheet.html``, a self-contained report of the same results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SECTIONS = ["risk_lab", "portfolio", "qaoa_study", "risk_forecast"]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def _git(*arguments: str) -> str | None:
    try:
        done = subprocess.run(["git", *arguments], cwd=REPO, capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None
    except OSError:
        return None


def _shown(path: Path) -> str:
    """Repository-relative when possible, so records carry no machine-specific prefix."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def environment() -> dict:
    packages = {}
    for name in ("cudaq", "numpy", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def _exact_records(results: dict) -> tuple[list[dict], dict]:
    """Saved requests and equity of the exact backtest; later sections study those decisions."""
    run = results["portfolio"]["runs"]["exact"]
    directory = REPO / run["run_directory"]
    records = json.loads((directory / "solver_replay.json").read_bytes())
    with (directory / "equity.csv").open(newline="") as handle:
        equity = [
            {"timestamp": r["timestamp"], "equity": r["equity"]} for r in csv.DictReader(handle)
        ]
    return records, equity


def run(config_path: Path, only: list[str] | None, reuse: Path | None, workers: int) -> Path:
    config = json.loads(config_path.read_bytes())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported lab config schema_version")
    wanted = only or SECTIONS
    unknown = set(wanted) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown sections: {sorted(unknown)}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lab_dir = REPO / "runs" / "lab" / f"{stamp}-{config['name']}"
    lab_dir.mkdir(parents=True, exist_ok=False)
    write_json(lab_dir / "config.json", config)
    target, seed = config["quantum"]["target"], config["seed"]
    previous = json.loads((reuse / "results.json").read_bytes()) if reuse else {}
    manifest = {
        "schema_version": 1,
        "run_id": lab_dir.name,
        "started": datetime.now(UTC).isoformat(),
        "status": "running",
        "config_file": _shown(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seed": seed,
        "quantum_target": target,
        "workers": workers,
        "environment": environment(),
        "reused_from": _shown(reuse) if reuse else None,
        "sections": {},
        "reproduce": f"python -m qflab.lab run {_shown(config_path)}",
    }
    write_json(lab_dir / "manifest.json", manifest)
    results: dict = {}

    def section(name: str, body) -> None:
        if name not in wanted:
            if name in previous:
                results[name] = previous[name]
                manifest["sections"][name] = {"status": "reused", "from": _shown(reuse)}
            else:
                manifest["sections"][name] = {"status": "skipped"}
            return
        started = time.perf_counter()
        print(f"[{name}] running ...", flush=True)
        try:
            results[name] = body()
            status = {"status": "completed"}
        except Exception as error:  # a failed section is recorded and the run continues
            results[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            status = {
                "status": "failed",
                "error": results[name]["error"],
                "traceback": traceback.format_exc()[-4000:],
            }
        status["seconds"] = time.perf_counter() - started
        manifest["sections"][name] = status
        print(f"[{name}] {status['status']} in {status['seconds']:.1f}s", flush=True)
        write_json(lab_dir / "results.json", results)
        write_json(lab_dir / "manifest.json", manifest)

    def risk_lab():
        from qflab.experiments.risk_lab import run_risk_lab

        return run_risk_lab(config["risk_lab"], target, seed)

    def portfolio():
        from qflab.experiments.portfolio import run_portfolio_study

        return run_portfolio_study(config["portfolio"], REPO, lab_dir, config_path.parent)

    def qaoa_study():
        from qflab.experiments.qaoa_study import run_qaoa_study

        records, _ = _exact_records(results)
        return run_qaoa_study(config["qaoa_study"], records, target, seed, workers)

    def risk_forecast():
        from qflab.backends import cudaq_backend
        from qflab.experiments.risk_forecast import run_risk_forecast

        records, equity = _exact_records(results)

        def factory(problem, seed):
            return cudaq_backend.AmplitudeSampler(problem, seed=seed, target=target)

        return run_risk_forecast(config["risk_forecast"], records, equity, factory, seed)

    section("risk_lab", risk_lab)
    section("portfolio", portfolio)
    section("qaoa_study", qaoa_study)
    section("risk_forecast", risk_forecast)

    failed = [n for n, s in manifest["sections"].items() if s["status"] == "failed"]
    manifest["status"] = "completed_with_failures" if failed else "completed"
    manifest["finished"] = datetime.now(UTC).isoformat()
    write_json(lab_dir / "results.json", results)
    write_json(lab_dir / "manifest.json", manifest)
    tearsheet(lab_dir)
    print(f"Lab run:   {lab_dir}\nTearsheet: {lab_dir / 'tearsheet.html'}")
    return lab_dir


def tearsheet(lab_dir: Path) -> Path:
    from qflab.ui.tearsheet import write_tearsheet

    return write_tearsheet(
        lab_dir / "tearsheet.html",
        json.loads((lab_dir / "results.json").read_bytes()),
        json.loads((lab_dir / "manifest.json").read_bytes()),
        json.loads((lab_dir / "config.json").read_bytes()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run", help="Run experiments and write a new run directory")
    execute.add_argument("config", type=Path)
    execute.add_argument("--only", help=f"Comma-separated subset of: {', '.join(SECTIONS)}")
    execute.add_argument("--reuse", type=Path, help="Copy the other sections from this run")
    execute.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    render = commands.add_parser("tearsheet", help="Rebuild tearsheet.html from saved results")
    render.add_argument("lab_dir", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(
            args.config.resolve(),
            args.only.split(",") if args.only else None,
            args.reuse.resolve() if args.reuse else None,
            args.workers,
        )
    else:
        print(f"Tearsheet: {tearsheet(args.lab_dir.resolve())}")


if __name__ == "__main__":
    main()
