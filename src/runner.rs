//! Wire our small research adapter into the real Nautilus engine.
use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
    process::Command,
    rc::Rc,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use chrono::Utc;
use nautilus_model::{data::BarType, identifiers::InstrumentId};
use rust_decimal::Decimal;
use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    config::Config,
    data::{Snapshot, sha256},
    decision::load_decisions,
    engine,
    strategy::{BaselineStrategy, RunState},
};

pub const NAUTILUS_VERSION: &str = "0.66.0";

// Identify the code compiled into this binary even before the first Git commit exists.
pub(crate) fn compiled_source_hash() -> String {
    sha256(
        concat!(
            include_str!("main.rs"),
            include_str!("lib.rs"),
            include_str!("config.rs"),
            include_str!("data.rs"),
            include_str!("decision.rs"),
            include_str!("strategy.rs"),
            include_str!("runner.rs"),
            include_str!("engine.rs"),
            include_str!("historical.rs"),
            include_str!("portfolio/mod.rs"),
            include_str!("portfolio/config.rs"),
            include_str!("portfolio/strategy.rs"),
            include_str!("portfolio/runner.rs"),
            include_str!("../Cargo.toml"),
            include_str!("../qflab/worker.py"),
            include_str!("../qflab/market_data.py"),
        )
        .as_bytes(),
    )
}

#[derive(Debug)]
pub struct Simulation {
    pub state: RunState,
    pub engine_result: Value,
    pub orders: Vec<Value>,
    pub positions: Vec<Value>,
    pub account_events: Value,
}

pub fn simulate(config: &Config, snapshot: &Snapshot) -> Result<Simulation> {
    config.validate()?;
    let saved = load_decisions(config, snapshot)?;
    let instrument_id = InstrumentId::from(format!("{}.SIM", config.symbol).as_str());
    let bar_type: BarType = format!("{instrument_id}-1-DAY-LAST-EXTERNAL").parse()?;
    let instrument = engine::equity(&config.symbol)?;
    let mut engine = engine::new_engine(config.cash()?, config.fee_bps()?)?;
    engine.add_instrument(&instrument)?;
    let state = Rc::new(RefCell::new(RunState::default()));
    engine.add_strategy(BaselineStrategy::new(
        config.clone(),
        instrument_id,
        bar_type,
        saved,
        Rc::clone(&state),
    ))?;
    engine.add_data(snapshot.bars(bar_type)?, None, true, true)?;
    let run_status = engine.run(None, None, None, false);
    if let Err(error) = run_status {
        state
            .borrow_mut()
            .errors
            .push(format!("Engine failure: {error}"));
    }
    let reports = engine::reports(&engine)?;
    engine.dispose();
    let state = state.borrow().clone();
    ensure!(
        state.equity.len() == snapshot.sessions.len(),
        "Missing account snapshots"
    );
    Ok(Simulation {
        state,
        engine_result: reports.engine_result,
        orders: reports.orders,
        positions: reports.positions,
        account_events: reports.account_events,
    })
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

pub(crate) fn write_csv<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

pub(crate) fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub fn run(config_path: &Path) -> Result<PathBuf> {
    let started = Instant::now();
    let config = Config::load(config_path)?;
    let snapshot = Snapshot::load(&config.data_path)?;
    // A new directory for every run: never overwrite previous evidence.
    fs::create_dir_all(&config.output_root)?;
    let run_id = format!(
        "{}-{}",
        Utc::now().format("%Y%m%dT%H%M%S%.9fZ"),
        std::process::id()
    );
    let output = config.output_root.canonicalize()?.join(&run_id);
    fs::create_dir(&output)?;
    fs::write(output.join("prices.csv"), &snapshot.bytes)?;
    write_json(&output.join("resolved_config.json"), &config)?;
    // The saved recipe points to this run's frozen inputs, so it can be replayed directly.
    let mut frozen_config = config.clone();
    frozen_config.data_path = PathBuf::from("prices.csv");
    frozen_config.output_root = PathBuf::from("..");
    frozen_config.decisions_path = config
        .decisions_path
        .as_ref()
        .map(|_| PathBuf::from("input_decisions.json"));
    write_json(&output.join("config.json"), &frozen_config)?;
    if let Some(path) = &config.decisions_path {
        fs::copy(path, output.join("input_decisions.json"))?;
    }
    let manifest_path = output.join("manifest.json");
    let config_bytes = serde_json::to_vec(&frozen_config)?;
    fs::write(output.join("Cargo.lock"), include_bytes!("../Cargo.lock"))?;
    let mut manifest = json!({
        "schema_version": 1, "run_id": run_id, "status": "running",
        "created_at_utc": Utc::now().to_rfc3339(),
        "nautilus_rust_version": NAUTILUS_VERSION,
        "rustc": command_output("rustc", &["--version"]),
        "git_commit": command_output("git", &["rev-parse", "HEAD"]),
        "git_status": command_output("git", &["status", "--porcelain"]),
        "config_sha256": sha256(&config_bytes), "snapshot_sha256": snapshot.sha256,
        "resolved_config_sha256": sha256(&serde_json::to_vec(&config)?),
        "compiled_source_sha256": compiled_source_hash(),
        "cargo_lock_sha256": sha256(include_bytes!("../Cargo.lock")),
        "os": std::env::consts::OS, "architecture": std::env::consts::ARCH,
        "session_count": snapshot.sessions.len(),
        "first_session": snapshot.sessions[0].timestamp,
        "last_session": snapshot.sessions.last().unwrap().timestamp,
        "assumptions": {
            "prices": "fabricated synthetic total-return units; USD; two decimal places",
            "quantity": "whole units; no shorts, leverage, or cash borrowing",
            "calendar": "consecutive Mon–Fri closes at 21:00 UTC; no exchange holiday/DST model",
            "timestamps": "ts_event = ts_init = completed session close",
            "execution": "quantity fixed at decision close; market order submitted at next recorded close",
            "fees": "maker/taker rate = fee_bps / 10000; explicit Nautilus fee model",
            "slippage": "zero; no spread, impact, or latency; ample synthetic volume required",
            "failure": "partial/rejected/unsettled orders mark the run failed; no automatic retry",
            "terminal_positions": "held and valued at final close; no forced liquidation",
            "reporting": "summary equity = Nautilus USD cash + Nautilus units * close; automatic engine analysis disabled",
            "corporate_actions": "none; do not add dividends to these total-return units",
            "evaluation": "teaching fixture only; no performance inference or tuned date splits"
        }
    });
    write_json(&manifest_path, &manifest)?;
    let simulation = match simulate(&config, &snapshot) {
        Ok(simulation) => simulation,
        Err(error) => {
            manifest["status"] = json!("failed");
            manifest["error"] = json!(format!("{error:#}"));
            write_json(&manifest_path, &manifest)?;
            return Err(error.context(format!("Run evidence: {}", output.display())));
        }
    };
    write_csv(&output.join("equity.csv"), &simulation.state.equity)?;
    write_json(&output.join("fills.json"), &simulation.state.fills)?;
    write_json(
        &output.join("decision_log.json"),
        &simulation.state.decisions,
    )?;
    write_json(
        &output.join("order_events.json"),
        &simulation.state.order_events,
    )?;
    write_json(&output.join("orders.json"), &simulation.orders)?;
    write_json(&output.join("positions.json"), &simulation.positions)?;
    write_json(
        &output.join("account_events.json"),
        &simulation.account_events,
    )?;
    write_json(
        &output.join("engine_result.json"),
        &simulation.engine_result,
    )?;
    let last = simulation.state.equity.last().context("No final equity")?;
    let fees: Decimal = simulation.state.fills.iter().map(|f| f.commission).sum();
    let summary = json!({ "starting_cash": config.cash()?, "ending_cash": last.cash,
        "ending_units": last.units, "ending_equity": last.equity,
        "net_pnl": last.equity - config.cash()?, "total_fees": fees,
        "fill_count": simulation.state.fills.len(), "errors": simulation.state.errors });
    write_json(&output.join("summary.json"), &summary)?;
    manifest["status"] = json!(if simulation.state.errors.is_empty() {
        "completed"
    } else {
        "failed"
    });
    manifest["wall_seconds"] = json!(started.elapsed().as_secs_f64());
    write_json(&manifest_path, &manifest)?;
    println!(
        "Starting cash: ${:.2}\nEnding cash:   ${:.2}\nUnits held:    {}\nEnding equity: ${:.2}\nFees paid:     ${:.2}\nFills:         {}\nRun saved:     {}",
        config.cash()?,
        last.cash,
        last.units,
        last.equity,
        fees,
        simulation.state.fills.len(),
        output.display()
    );
    ensure!(
        simulation.state.errors.is_empty(),
        "Backtest failed; inspect {}",
        output.display()
    );
    Ok(output)
}
