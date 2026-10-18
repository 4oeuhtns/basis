# Quantum Finance Lab

A small **Rust backtest using NautilusTrader**, with a separate **Python research worker**.

Python downloads and prepares data; Rust runs the trading simulation. Nautilus maintains the actual orders, fills, cash, and holdings. Usage and explanations live in this README; internal notes and lessons remain ignored.

## Run your first backtest

From this repository:

```sh
./scripts/cargo-local run --locked --offline -- configs/demo.json
```

The local Rust toolchain and dependencies have already been installed for this workspace. `cargo-local` keeps them in `.tools/`. On a fresh checkout, install Rust with the project-local instructions below, then run `./scripts/cargo-local fetch` once before using `--offline`.

Expected result:

```text
Starting cash: $1000.00
Ending cash:   $450.00
Units held:    5
Ending equity: $1075.00
Fees paid:     $0.00
Fills:         1
```

This uses six fabricated daily price bars. It proves accounting and order timing; it does not measure a strategy's usefulness on real markets.

## Next: a portfolio with two funds

Run two funds sharing one cash account:

```sh
./scripts/cargo-local run --locked --offline -- configs/portfolio.json
```

This aims for 45% in each synthetic fund and reserves 10% as cash. It waits for both prices at each close, saves quantities for the next session, and settles sales before purchases. Expected ending value: **$1,700 = $300 cash + 7 ALPHA × $100 + 7 BETA × $100**, with six fills and zero fees. `holdings.csv` adds per-fund daily quantities and values. Buys are capped by actual cash, configured fees, and the previous decision's dollar reserve; caps and skips are logged.

The original single-fund lesson and its Python handoff remain available. The multi-fund example currently uses a Rust equal-weight baseline.

## Historical data: SPY and QQQ, all of 2025

The historical pipeline adapts the earlier project's yfinance download/cache approach. Python is the delivery truck, Parquet is the storage box, and Rust is the replay machine. Downloading happens separately from backtesting, so replays never need Yahoo or a running Python process.

Install the data tools into the project's virtual environment (already installed in this workspace):

```sh
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m qflab.market_data download \
  --symbols SPY QQQ --start 2025-01-01 --end 2026-01-01 \
  --output runs/snapshots/spy-qqq-2025
.venv/bin/python -m qflab.market_data verify runs/snapshots/spy-qqq-2025
./scripts/cargo-local run --locked --offline -- configs/historical.json
```

The end date is **exclusive**, so `2026-01-01` includes December 31, 2025. This period has 250 XNYS sessions. The frozen schedule includes the January 9 special closure, ordinary holidays, daylight saving changes, and shortened sessions. Each daily bar becomes available at that session's actual close. A daily bar's date label does not mean its closing price was known at midnight.

The config starts with $10,000 and aims for 45% in each fund plus a 10% cash reserve. It makes a decision every **21 recorded sessions**, then executes the saved plan at the following session's close. That interval is not a calendar-month schedule. All prices must arrive before a decision; sales update cash and positions before purchases begin.

The snapshot contains:

| File | Purpose |
| --- | --- |
| `SPY.source.parquet`, `QQQ.source.parquet` | Preserved yfinance tables with source OHLC, adjusted close, dividends, splits, and volume |
| `SPY.parquet`, `QQQ.parquet` | Portable normalized data read directly by Rust |
| `SPY.csv`, `QQQ.csv` | The same normalized observations for inspection and format-parity checks |
| `calendar.csv` | Frozen XNYS session dates, opens, and closes from `exchange-calendars` |
| `manifest.json` | Symbols, inclusive/exclusive bounds, provider options, adjustment policy, versions, and file hashes |

Normalized prices are explicit decimal strings rounded to six places; timestamps and volume are integers. This schema avoids coupling the snapshot to an older Nautilus catalog's 128-bit binary price representation. These files are our portable dataset format, rather than a native Nautilus `ParquetDataCatalog`.

For this first historical baseline, each day's OHLC is multiplied by `Adj Close / Close`. These are **retrospective adjusted research units**, with whole-unit positions and USD cash accounting. They are not literal historical executable share prices or point-in-time data. Do not add dividends or split events again in the backtest. Source volume is only a liquidity proxy for these units. Research marks retain six decimals; Nautilus rounds USD trade notionals and fees to cents. Source tables remain available if we later implement explicit corporate-action accounting.

The downloader explicitly requests `auto_adjust=False`, `back_adjust=False`, `actions=True`, `repair=False`, `keepna=True`, and `rounding=False`. It checks USD currency and the New York exchange timezone. Missing, duplicate, out-of-order, incomplete, or out-of-range sessions fail validation; no prices are filled forward. Rust checks every file hash, the requested asset set, price/volume ranges, and each bar's availability against the frozen schedule. Python verification also reproduces the adjusted marks from the preserved source and compares the schedule with the installed calendar.

Repeating the same download command verifies and reuses the existing snapshot **without downloading**. A changed date range or corrupt cache is an error. To obtain a new source revision, choose a new output directory and point a config at it; existing snapshots are preserved. Downloads, caches, and run evidence stay under the already ignored `runs/` directory. No `.gitignore` change is needed.

Historical run folders include the entire snapshot and a replayable `config.json`. To check Parquet/CSV parity yourself, copy the historical config and change both asset paths from `.parquet` to `.csv`, keeping the snapshot manifest. Cash, holdings, fills, and daily equity should agree exactly. The integration tests perform this comparison automatically on an offline fixture with an early close and holiday gap.

Execution still assumes zero spread, impact, and latency and a fill at the next observed close. These assumptions make this a data/accounting baseline. A live trading model needs executable quotes, order arrival times, and worker delays. Yahoo data is downloaded for personal research; source usage rights are separate from the software license. See the [yfinance documentation](https://ranaroussi.github.io/yfinance/) and [exchange-calendars project](https://github.com/gerrymanoim/exchange_calendars).

To trace the code, start with `qflab/market_data.py` (prepare the recording), `src/historical.rs` (verify and read it), then `src/portfolio/strategy.rs` (react to completed bars). The synthetic examples keep their original strict weekday calendar; historical data uses a separate adapter.

## Run the Python → Rust handoff

```sh
python3 -m qflab.worker \
  --data data/demo_prices.csv \
  --cutoff 2024-01-02T21:00:00Z \
  --output runs/worker/decision.json
./scripts/cargo-local run --locked --offline -- configs/replay.json
```

The worker currently writes a **classical fixed-weight baseline**. No quantum SDK or circuit is implemented yet. A later quantum worker will write the same versioned decision format. Python runs before the backtest, outside the simulated event loop.

## What gets saved

Each run prints a new folder under `runs/` containing the input snapshot, resolved configuration, dependency lock, manifest, decision log, fills, native orders and positions, native account events, daily equity, and summary. Prior run folders are preserved. Its `config.json` points to the frozen inputs inside that folder; pass it to the executable to replay those inputs. `resolved_config.json` preserves the original resolved settings.

Read `summary.json` for ending portfolio value and `equity.csv` for each day's cash and units. Portfolio value means **cash + units × closing price**. Cash alone leaves out the units you own. Automatic Nautilus performance analysis is disabled in this first slice; `engine_result.json` retains the engine's run metadata, while our summary values come from its account and position state.

## Check the work

```sh
./scripts/cargo-local test --locked --offline -- --test-threads=1
./scripts/cargo-local clippy --locked --offline --all-targets -- -D warnings
./scripts/cargo-local fmt --all -- --check
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check qflab tests
```

Install `requirements-dev.txt` and `requirements-data.txt` into `.venv` for the full test suite. The Rust historical integration test uses `.venv/bin/python` to generate an offline fixture; set `QFLAB_PYTHON` to use another environment containing the data packages. The research worker itself uses only Python's standard library.

The Rust tests use the real Nautilus engine to check daily accounting, no trades, flat prices, fees, next-close execution, drift/rebalancing, future-data isolation, unaffordable orders, and Python replay. Portfolio checks add synchronized closes, shared cash, sales settling before purchases, reserve/fee caps, missing sessions, partial-fill failure, and frozen-input replay. Historical checks compare CSV/Parquet economic results, six-decimal marks with USD rounding, an early close and holiday gap, volume above the synthetic limit, corrupted files, missing sessions, and incorrect availability timestamps. Python tests cover the worker boundary, calendar behavior, immutable/offline cache reuse, and source normalization.

## Versions and a fresh checkout

Rust is pinned to **1.99.0**, and every direct Nautilus Rust crate to **0.66.0**. `Cargo.lock` freezes transitive dependencies. Rust crate versions and the Python package's versions are separate numbering systems. The executable uses the Rust crates and does not import the Nautilus Python package.

If Rust is already installed normally, plain `cargo run --locked -- configs/demo.json` also works and reads `rust-toolchain.toml`. To match this workspace's project-local setup, run these commands from the repository root:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/qflab-rustup-init.sh
CARGO_HOME="$PWD/.tools/cargo" RUSTUP_HOME="$PWD/.tools/rustup" \
  sh /tmp/qflab-rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain 1.99.0
./scripts/cargo-local fetch --locked
```

Nautilus supports native Rust strategies and backtests, and its Rust API is evolving. See the [official Rust guide](https://nautilustrader.io/docs/latest/concepts/rust/) and the versioned library sources downloaded by Cargo under `.tools/cargo/registry/src/`. Our setup follows the upstream `BacktestEngine` pattern: venue → instrument → strategy → data → run.
