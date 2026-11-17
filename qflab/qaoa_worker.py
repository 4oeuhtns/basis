"""QAOA portfolio worker for the Rust backtester: one JSON request on stdin, one answer on stdout.

The request is the exact worker's request plus a ``solver`` block. The forecast, objective and
constraints are built by the same code as the exact solver, so a paired run differs only in
how the selection is searched. This process opens no data files.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

STARTED = time.perf_counter()
# Rust starts this file with ``python -I``, which leaves the repository off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflab.methods.qaoa_solver import solve_qaoa, validate_solver  # noqa: E402
from qflab.optimizer_worker import solve_request, target_weights  # noqa: E402


def answer(raw: bytes) -> dict:
    request = json.loads(raw)
    request_hash = hashlib.sha256(raw).hexdigest()
    if not isinstance(request, dict) or "solver" not in request:
        raise ValueError("QAOA worker needs a solver block in the request")
    settings = validate_solver(request.pop("solver"))
    reference = solve_request(request, request_hash)  # validates inputs; builds the model

    from qflab.backends import cudaq_backend  # imported late so input errors fail fast

    imported = time.perf_counter()
    # Each decision gets its own reproducible random stream, tied to its exact input bytes.
    decision_seed = int(request_hash[:8], 16)
    result = solve_qaoa(reference, settings, cudaq_backend.QaoaSampler, decision_seed)
    symbols = reference["canonical_model"]["symbols"]
    feasible = result["status"] == "feasible"
    with localcontext() as context:
        context.prec = 50
        reserve = Decimal(request["settings"]["cash_reserve"])
        weights = (
            target_weights(symbols, result["selected"], reserve)
            if feasible
            else dict.fromkeys(symbols, "0")
        )
    result["timings"]["startup_seconds"] = imported - STARTED
    result["timings"]["worker_seconds"] = time.perf_counter() - STARTED
    diagnostics = {**reference["diagnostics"]}
    diagnostics.pop("evaluated_portfolios")
    diagnostics.update(
        {key: value for key, value in result.items() if key not in ("status", "selected")}
    )
    diagnostics["backend"] = cudaq_backend.environment()
    diagnostics["evidence_label"] = "simulator performance"
    return {
        "schema_version": 1,
        "method": "qaoa_cudaq",
        "status": result["status"],
        "data_cutoff": request["data_cutoff"],
        "request_sha256": request_hash,
        "selected": result["selected"] if feasible else [],
        "target_weights": weights,
        "objective_integer": result["objective_integer"],
        "canonical_model": reference["canonical_model"],
        "diagnostics": diagnostics,
    }


def main() -> None:
    raw = sys.stdin.buffer.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError("Optimizer request exceeds 1 MiB")
    sys.stdout.write(json.dumps(answer(raw), separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
