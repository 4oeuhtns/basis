from pathlib import Path

import pytest

from qflab.worker import build_decision

DATA = Path(__file__).resolve().parents[1] / "data" / "demo_prices.csv"
CUTOFF = "2024-01-02T21:00:00Z"


def test_future_prices_cannot_change_past_target():
    original = DATA.read_bytes()
    changed = original.replace(b"125.00", b"925.00")
    first = build_decision(original, CUTOFF, "DEMO", "0.5")
    second = build_decision(changed, CUTOFF, "DEMO", "0.5")
    assert first["input_snapshot_hash"] != second["input_snapshot_hash"]
    first.pop("input_snapshot_hash")
    second.pop("input_snapshot_hash")
    assert first == second


@pytest.mark.parametrize("weight", ["NaN", "Infinity", "-0.1", "1.01"])
def test_invalid_weights_fail(weight):
    with pytest.raises(ValueError):
        build_decision(DATA.read_bytes(), CUTOFF, "DEMO", weight)


def test_cutoff_requires_a_timezone_and_an_existing_observation():
    for cutoff in ["2024-01-02T21:00:00", "2024-01-02T20:00:00Z"]:
        with pytest.raises(ValueError):
            build_decision(DATA.read_bytes(), cutoff, "DEMO", "0.5")
