# Quantum Finance Lab

A small **Rust backtest using NautilusTrader**, with a separate **Python research worker**.

Start with [the learning guide](docs/LEARNING_GUIDE.md). It explains the code and the arithmetic without assuming you already know Rust or trading. The [implementation plan](docs/IMPLEMENTATION_PLAN.md) connects this first slice to the larger build plan.

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
python3 -m pytest -q
```

For Python test tools, use the existing `.venv/bin/python -m pytest -q` in this workspace, or install `requirements-dev.txt` in your own virtual environment. The worker itself uses only Python's standard library.

The Rust tests use the real Nautilus engine to check daily accounting, no trades, flat prices, fees, next-close execution, drift/rebalancing, future-data isolation, unaffordable orders, and Python replay. Python tests check the decision boundary. The fixture calendar is deliberately Mon–Fri at 21:00 UTC; exchange holidays and daylight saving time need a real-data adapter.

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
