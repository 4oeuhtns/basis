"""Offline snapshots for Rust; adapted from quant/src/backtest/data.py's download/cache flow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import tempfile
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from importlib.metadata import version
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KIND = "historical_adjusted_research_units"
POLICY = "adj_close_ratio_ohlc_research_units"
COLUMNS = ["session_date", "available_at_ns", "open", "high", "low", "close", "volume"]
SCHEMA = pa.schema(
    [
        pa.field(
            name,
            pa.int64() if name in {"available_at_ns", "volume"} else pa.string(),
            nullable=False,
        )
        for name in COLUMNS
    ]
)
QUANTUM = Decimal("0.000001")
MAX_VOLUME = 18_446_744_073  # Native Nautilus standard-precision Quantity limit.


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(start: str, end: str) -> list[dict]:
    """XNYS session labels and actual closes, including DST and shortened sessions."""
    first, exclusive = date.fromisoformat(start), date.fromisoformat(end)
    if first >= exclusive:
        raise ValueError("start must precede the exclusive end date")
    cal = xcals.get_calendar("XNYS", start=first, end=exclusive)
    # Inclusive/exclusive bounds may themselves be holidays, outside cal's first/last session.
    sessions = [s for s in cal.sessions if first <= s.date() < exclusive]
    rows = [
        {
            "session_date": s.date().isoformat(),
            "open_ns": int(cal.session_open(s).as_unit("ns").value),
            "close_ns": int(cal.session_close(s).as_unit("ns").value),
        }
        for s in sessions
    ]
    if len(rows) < 2:
        raise ValueError("At least two complete trading sessions are required")
    return rows


def number(value, name: str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"Non-finite {name}")
    return result


def normalize(raw: pd.DataFrame, calendar: list[dict]) -> list[dict]:
    """Preserve source values separately; derive explicitly labeled adjusted research marks."""
    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}
    if raw is None or raw.empty or not required.issubset(raw.columns):
        raise ValueError(f"Missing source data/columns: {sorted(required)}")
    labels = [pd.Timestamp(t).date().isoformat() for t in raw.index]
    expected = [s["session_date"] for s in calendar]
    if labels != expected:
        raise ValueError(
            "Source sessions differ from requested calendar: missing, duplicate, "
            "out-of-order, or out-of-range rows; no forward filling"
        )
    rows = []
    for (_, source), session in zip(raw.iterrows(), calendar, strict=True):
        close, adjusted = number(source["Close"], "Close"), number(source["Adj Close"], "Adj Close")
        if close <= 0 or adjusted <= 0:
            raise ValueError("Close and Adj Close must be positive")
        ratio = adjusted / close
        prices = {
            name.lower(): number(source[name], name) for name in ["Open", "High", "Low", "Close"]
        }
        if (
            min(prices.values()) <= 0
            or prices["high"] < max(prices.values())
            or prices["low"] > min(prices.values())
        ):
            raise ValueError("Invalid source OHLC range")
        prices = {
            name: (value * ratio).quantize(QUANTUM, rounding=ROUND_HALF_EVEN)
            for name, value in prices.items()
        }
        volume = number(source["Volume"], "Volume")
        for name in ["Dividends", "Stock Splits"]:
            if number(source[name], name) < 0:
                raise ValueError(f"Negative {name}")
        if volume != volume.to_integral_value() or not 4 <= volume <= MAX_VOLUME:
            raise ValueError("Volume must be whole units within the native Quantity range")
        if min(prices.values()) <= 0 or max(prices.values()) > 1_000_000:
            raise ValueError("Adjusted marks outside supported range")
        rows.append(
            {
                "session_date": session["session_date"],
                "available_at_ns": session["close_ns"],
                **{name: f"{value:.6f}" for name, value in prices.items()},
                "volume": int(volume),
            }
        )
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_snapshot(root: Path) -> dict:
    """Invalid cache entries are errors, never a reason to silently redownload."""
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["data_kind"] != KIND
        or manifest["adjustment_policy"] != POLICY
        or manifest["price_precision"] != 6
        or manifest["calendar"] != "XNYS"
    ):
        raise ValueError("Unsupported snapshot schema/policy")
    symbols = manifest["assets"]
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Missing/duplicate symbols")
    expected_files = {"calendar.csv"}
    for symbol in symbols:
        if not re.fullmatch(r"[A-Z0-9]{1,20}", symbol):
            raise ValueError("Unsupported symbol")
        expected_files.update(
            f"{symbol}{suffix}" for suffix in [".csv", ".parquet", ".source.parquet"]
        )
    if set(manifest["files"]) != expected_files:
        raise ValueError("Snapshot file inventory mismatch")
    for name, digest in manifest["files"].items():
        if fingerprint(root / name) != digest:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    with (root / "calendar.csv").open(newline="") as stream:
        stored = list(csv.DictReader(stream))
    expected = schedule(manifest["start"], manifest["end_exclusive"])
    if stored != [{k: str(v) for k, v in r.items()} for r in expected]:
        raise ValueError("Frozen calendar differs from exchange calendar")
    for symbol in symbols:
        table = pq.read_table(root / f"{symbol}.parquet")
        if not table.schema.equals(SCHEMA, check_metadata=False):
            raise ValueError("Unsupported normalized Parquet schema")
        rows = table.to_pylist()
        with (root / f"{symbol}.csv").open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        source = pd.read_parquet(root / f"{symbol}.source.parquet")
        if rows != normalize(source, expected):
            raise ValueError("Normalized values differ from preserved source")
        if csv_rows != [{k: str(v) for k, v in r.items()} for r in rows]:
            raise ValueError("CSV and Parquet values differ")
    return manifest


def freeze_snapshot(
    output: Path, raw_by_symbol: dict[str, pd.DataFrame], start: str, end: str, *, source: dict
) -> dict:
    """Commit a new immutable snapshot only after every instrument passes validation."""
    if output.exists():
        raise FileExistsError("Use a new output directory; existing snapshots are preserved")
    if not raw_by_symbol or any(not re.fullmatch(r"[A-Z0-9]{1,20}", s) for s in raw_by_symbol):
        raise ValueError("Symbols must be 1–20 uppercase letters/digits")
    calendar = schedule(start, end)
    if calendar[-1]["close_ns"] > pd.Timestamp.now(tz="UTC").as_unit("ns").value:
        raise ValueError("Snapshot includes an unfinished/future session")
    normalized = {symbol: normalize(raw, calendar) for symbol, raw in raw_by_symbol.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=output.parent) as temporary:
        root = Path(temporary) / "complete"
        root.mkdir()
        write_csv(root / "calendar.csv", ["session_date", "open_ns", "close_ns"], calendar)
        for symbol, rows in normalized.items():
            raw_by_symbol[symbol].to_parquet(
                root / f"{symbol}.source.parquet", compression="snappy"
            )
            pq.write_table(
                pa.Table.from_pylist(rows, schema=SCHEMA),
                root / f"{symbol}.parquet",
                compression="snappy",
            )
            write_csv(root / f"{symbol}.csv", COLUMNS, rows)
        manifest = {
            "schema_version": 1,
            "data_kind": KIND,
            "calendar": "XNYS",
            "start": start,
            "end_exclusive": end,
            "price_precision": 6,
            "adjustment_policy": POLICY,
            "assets": sorted(raw_by_symbol),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": source,
            "environment": {
                name: version(name)
                for name in ["yfinance", "exchange-calendars", "pandas", "pyarrow"]
            },
            "files": {p.name: fingerprint(p) for p in sorted(root.iterdir())},
        }
        manifest["environment"]["python"] = platform.python_version()
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_snapshot(root)
        root.rename(output)
    return manifest


def download_snapshot(output: Path, symbols: list[str], start: str, end: str) -> dict:
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Supply distinct symbols")
    if any(not re.fullmatch(r"[A-Z0-9]{1,20}", s) for s in symbols):
        raise ValueError("Only simple USD US-market ticker symbols are supported")
    if output.exists():
        manifest = verify_snapshot(output)
        if (
            manifest["assets"] != sorted(symbols)
            or manifest["start"] != start
            or manifest["end_exclusive"] != end
        ):
            raise ValueError("Cached snapshot uses different symbols/dates; use a new output path")
        print(f"Verified existing snapshot; no download: {output}")
        return manifest
    calendar = schedule(start, end)
    if calendar[-1]["close_ns"] > pd.Timestamp.now(tz="UTC").as_unit("ns").value:
        raise ValueError("Requested period includes an unfinished session")
    import yfinance as yf

    output.parent.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(output.parent / ".yfinance-cache"))
    raw = {}
    for symbol in symbols:
        # Check exchange currency/timezone instead of assuming any Yahoo ticker is a USD ETF.
        ticker = yf.Ticker(symbol)
        frame = ticker.history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            back_adjust=False,
            actions=True,
            repair=False,
            keepna=True,
            rounding=False,
        )
        metadata = ticker.get_history_metadata()
        if (
            metadata.get("currency") != "USD"
            or metadata.get("exchangeTimezoneName") != "America/New_York"
        ):
            raise ValueError(f"{symbol}: only USD US regular-session instruments are supported")
        raw[symbol] = frame
    return freeze_snapshot(
        output,
        raw,
        start,
        end,
        source={
            "provider": "Yahoo Finance via yfinance",
            "interval": "1d",
            "auto_adjust": False,
            "back_adjust": False,
            "actions": True,
            "repair": False,
            "keepna": True,
            "rounding": False,
            "currency": "USD",
            "exchange_timezone": "America/New_York",
            "note": (
                "retrospective adjusted research units; not point-in-time executable share prices; "
                "source volume is a liquidity proxy; "
                "no separate dividends or split events in backtest"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--symbols", nargs="+", required=True)
    download.add_argument("--start", required=True, help="Inclusive YYYY-MM-DD")
    download.add_argument("--end", required=True, help="Exclusive YYYY-MM-DD")
    download.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "download":
            manifest = download_snapshot(args.output, args.symbols, args.start, args.end)
        else:
            manifest = verify_snapshot(args.path)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Data error: {error}\n")
    print(
        f"Verified {', '.join(manifest['assets'])}: {manifest['start']} through "
        f"{manifest['end_exclusive']} (end exclusive)"
    )


if __name__ == "__main__":
    main()
