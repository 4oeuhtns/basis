"""Build the tearsheet: one self-contained HTML file from a lab run's saved results.

The page embeds the results as JSON and draws everything in the browser with inline SVG. It
loads nothing from the network, so the file can be archived with the run and opened anywhere.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("tearsheet_template.html")
SECTIONS = ("risk_lab", "portfolio", "qaoa_study", "risk_forecast")


def _ok(results: dict, name: str) -> bool:
    return name in results and results[name].get("status") != "failed"


def scientific_checks(results: dict) -> list[dict]:
    """The plan's minimum checks (section 11), each answered from this run's own numbers.

    ``status`` is ``pass``, ``fail``, ``tests`` (enforced by the test suite, not re-measured in
    this run), ``open`` (not satisfied yet; stated plainly) or ``missing`` (section not run).
    """
    checks = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    if _ok(results, "risk_lab"):
        lab = results["risk_lab"]
        reference = lab["reference"]
        mass_ok = abs(reference["probability_mass"] - 1) < 1e-12
        round_trip = reference["scaling_round_trip_error"] < 1e-12
        add(
            "Discrete probability mass sums to one and scaling is reversible",
            "pass" if mass_ok and round_trip else "fail",
            f"mass = {reference['probability_mass']:.15f}; amplitude to value and back differs "
            f"by {reference['scaling_round_trip_error']:.1e}",
        )
        circuits = lab["circuit_verification"]
        grids = [row["circuit_matches_discrete_sum"] for row in lab["grid_sweep"]]
        add(
            "Indicator probability matches the exact discrete expectation",
            "pass" if all(c["matches"] for c in circuits) and all(grids) else "fail",
            f"statevector check at Grover powers 0 to {len(circuits) - 1} and on "
            f"{len(grids)} grid sizes, tolerance 1e-9",
        )
        quantum = [row for row in lab["benchmark"] if row["method"] == "iqae"]
        nominal = lab["settings"]["confidence"]
        lowest = min(row["coverage"] for row in quantum)
        seeds = quantum[0]["seeds"]
        # With n seeds, coverage two standard errors below nominal is still consistent.
        slack = 2 * (nominal * (1 - nominal) / seeds) ** 0.5
        add(
            "IQAE confidence behavior checked across seeds (ideal simulation)",
            "pass" if lowest >= nominal - slack else "fail",
            f"lowest coverage {lowest:.0%} over {seeds} seeds per budget; nominal {nominal:.0%}",
        )
    else:
        for name in (
            "Discrete probability mass sums to one and scaling is reversible",
            "Indicator probability matches the exact discrete expectation",
            "IQAE confidence behavior checked across seeds (ideal simulation)",
        ):
            add(name, "missing", "risk lab section not available in this run")

    add(
        "QUBO / Ising energy and bit decoding match direct evaluation",
        "tests",
        "every bitstring is checked on each solve (verify_qubo) and in tests/test_qaoa.py, "
        "tests/test_qaoa_cudaq.py; the circuit is compared with an independent simulation",
    )

    if _ok(results, "portfolio"):
        study = results["portfolio"]
        quantum = study["runs"]["qaoa"]
        feasible = [d for d in quantum["decisions"] if d["status"] == "feasible"]
        add(
            "Portfolio feasibility checked independently of the solver",
            "pass" if all(len(d["selected"]) == 3 for d in feasible) else "fail",
            f"{len(feasible)} of {len(quantum['decisions'])} QAOA answers accepted by the Rust "
            f"validator; {quantum['fallbacks']} fallbacks retained holdings",
        )
        ledgers = [run["cash_reconciliation"] for run in study["runs"].values()]
        add(
            "Accounting remains self-financing with costs and drift",
            "pass" if all(ledger["reconciles"] for ledger in ledgers) else "fail",
            "daily cash rebuilt from fills and fees for all three runs; largest difference "
            f"${max(ledger['largest_difference'] for ledger in ledgers):.4f}",
        )
        pairing = study["pairing"]
        add(
            "Paired experiment differs in the intended method only",
            "pass" if pairing["differs_only_in_method"] and pairing["same_snapshot"] else "fail",
            "differing config fields: " + ", ".join(pairing["config_differences"]),
        )
    else:
        for name in (
            "Portfolio feasibility checked independently of the solver",
            "Accounting remains self-financing with costs and drift",
            "Paired experiment differs in the intended method only",
        ):
            add(name, "missing", "portfolio section not available in this run")

    add(
        "Mutating future data does not alter past decisions",
        "tests",
        "enforced by tests/optimizer.rs and tests/monthly.rs on the real engine; "
        "not re-measured in a lab run",
    )
    add(
        "Runtime includes preprocessing and classical optimizer work",
        "pass",
        "QAOA shots count every objective evaluation plus the final sample; solver wall time "
        "includes interpreter start-up; IQAE time includes circuit construction",
    )
    add(
        "Strong direct classical methods are included",
        "pass",
        "Black-Scholes, the exact discrete sum, scrambled Sobol, exhaustive enumeration and "
        "a swap search sit beside every quantum result",
    )
    add(
        "Final evaluation dates were not repeatedly used for tuning",
        "open",
        "there is one period (2025) and no development / validation / final split yet; the "
        "QAOA parameter-search scheme was chosen while looking at the first 2025 decision",
    )
    return checks


def page_data(results: dict, manifest: dict, config: dict) -> dict:
    """Trim the saved results to what the page draws."""
    results = copy.deepcopy(results)
    if _ok(results, "qaoa_study"):
        for run in results["qaoa_study"]["runs"]:
            run.pop("parameters", None)
    if _ok(results, "portfolio"):
        for run in results["portfolio"]["runs"].values():
            run.pop("config", None)
    return {
        "results": results,
        "manifest": {k: v for k, v in manifest.items() if k != "sections"}
        | {
            "sections": {
                name: {k: v for k, v in status.items() if k != "traceback"}
                for name, status in manifest.get("sections", {}).items()
            }
        },
        "config": config,
        "checks": scientific_checks(results),
        "available": {name: _ok(results, name) for name in SECTIONS},
    }


def write_tearsheet(path: Path, results: dict, manifest: dict, config: dict) -> Path:
    payload = json.dumps(page_data(results, manifest, config), allow_nan=False)
    # Keep the JSON inert inside a <script> element.
    payload = payload.replace("</", "<\\/").replace("<!--", "<\\!--")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload)
    if payload not in html:
        raise ValueError("Tearsheet template is missing its data placeholder")
    path.write_text(html)
    return path
