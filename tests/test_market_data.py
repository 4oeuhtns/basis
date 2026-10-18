import json
from datetime import UTC, datetime

import pytest

pytest.importorskip("exchange_calendars")
pytest.importorskip("pyarrow")

from historical_fixture import END, START, make_fixture, source_frame

from qflab.market_data import (
    download_snapshot,
    fingerprint,
    normalize,
    schedule,
    verify_snapshot,
)


def test_calendar_handles_holidays_dst_and_early_closes():
    year = schedule("2025-01-01", "2026-01-01")
    assert len(year) == 250
    closes = {s["session_date"]: datetime.fromtimestamp(s["close_ns"] / 1e9, UTC) for s in year}
    assert "2025-01-01" not in closes
    assert "2025-01-09" not in closes  # Special NYSE closure for President Carter.
    assert "2025-07-04" not in closes
    assert closes["2025-03-07"].hour == 21
    assert closes["2025-03-10"].hour == 20
    for day in ["2025-07-03", "2025-11-28", "2025-12-24"]:
        assert closes[day].hour == 18 if day != "2025-07-03" else closes[day].hour == 17
    assert closes["2025-12-31"].hour == 21


def test_snapshot_cache_is_explicit_immutable_and_offline(tmp_path, monkeypatch):
    root = tmp_path / "snapshot"
    manifest = make_fixture(root)
    assert verify_snapshot(root) == manifest
    assert len(manifest["files"]) == 7
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", lambda *_: pytest.fail("Unexpected download"))
    assert download_snapshot(root, ["BETA", "ALPHA"], START, END) == manifest
    with pytest.raises(ValueError, match="different symbols/dates"):
        download_snapshot(root, ["ALPHA"], START, END)
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        make_fixture(root)
    assert (root / "manifest.json").read_bytes() == before
    (root / "ALPHA.parquet").write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        download_snapshot(root, ["ALPHA", "BETA"], START, END)


@pytest.mark.parametrize("broken", ["missing", "duplicate", "reverse", "nan", "volume", "range"])
def test_bad_source_data_fails_instead_of_filling_or_dropping_rows(broken):
    raw = source_frame([100] * 6)
    if broken == "missing":
        raw = raw.iloc[1:]
    elif broken == "duplicate":
        raw.index = [raw.index[0]] * len(raw)
    elif broken == "reverse":
        raw = raw.iloc[::-1]
    elif broken == "nan":
        raw.iloc[0, raw.columns.get_loc("Close")] = float("nan")
    elif broken == "volume":
        raw.iloc[0, raw.columns.get_loc("Volume")] = 0
    else:
        raw.iloc[0, raw.columns.get_loc("High")] = 1
    with pytest.raises(ValueError):
        normalize(raw, schedule(START, END))


def test_derived_marks_and_csv_must_match_preserved_source(tmp_path):
    root = tmp_path / "snapshot"
    manifest = make_fixture(root)
    path = root / "ALPHA.csv"
    path.write_text(path.read_text().replace("98.765432", "99.765432", 1))
    manifest["files"][path.name] = fingerprint(path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="CSV and Parquet"):
        verify_snapshot(root)
