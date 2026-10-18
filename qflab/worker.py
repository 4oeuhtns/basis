"""A real Python-to-Rust boundary, with a classical teaching baseline today.

Later, a quantum solver can produce the same instruction card. It must use
only observations at or before data_cutoff. No solver runs in a Rust callback.
"""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone")
    return result


def build_decision(data: bytes, cutoff: str, symbol: str, target_weight: str) -> dict:
    """Use a fixed allocation to prove the handoff before introducing quantum SDKs."""
    weight = Decimal(target_weight)
    if not weight.is_finite() or not 0 <= weight <= 1:
        raise ValueError("target_weight must be a finite decimal between 0 and 1")
    if (
        not symbol
        or len(symbol) > 20
        or not all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for c in symbol)
    ):
        raise ValueError("symbol must be 1–20 uppercase ASCII letters or digits")
    cutoff_time = parse_time(cutoff)
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    # The snapshot hash identifies the recording; model inputs use only this prefix.
    past = [row for row in rows if parse_time(row["timestamp"]) <= cutoff_time]
    if not past or parse_time(past[-1]["timestamp"]) != cutoff_time:
        raise ValueError("cutoff must match a recorded session close")
    if any(not Decimal(row["close"]).is_finite() or Decimal(row["close"]) <= 0 for row in past):
        raise ValueError("Past close prices must be finite and positive")
    return {
        "schema_version": 1,
        "data_cutoff": cutoff_time.isoformat(),
        "available_at": cutoff_time.isoformat(),
        "target_weights": {f"{symbol}.SIM": str(weight)},
        "input_snapshot_hash": hashlib.sha256(data).hexdigest(),
        "method": "classical_fixed_weight_baseline",
        "status": "feasible",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--symbol", default="DEMO")
    parser.add_argument("--target-weight", default="0.50")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = build_decision(args.data.read_bytes(), args.cutoff, args.symbol, args.target_weight)
    content = json.dumps([decision], indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != content:
            raise FileExistsError("Use a new output path to preserve the earlier decision")
    else:
        with args.output.open("x") as stream:
            stream.write(content)
    print(f"Saved classical baseline decision: {args.output}")
    print("Rust will size the order when this decision becomes available, then wait one session.")


if __name__ == "__main__":
    main()
