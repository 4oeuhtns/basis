//! Original single-fund example and saved-instruction replay.
mod config;
mod decision;
mod strategy;

use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use nautilus_model::{data::BarType, identifiers::InstrumentId};
use rust_decimal::Decimal;
use serde_json::json;

use crate::{
    backtest::engine::{self, Reports},
    data::Snapshot,
    output::{self, create_run_directory, write_csv, write_json, write_reports},
};
pub use config::Config;
use decision::load_decisions;
use strategy::BaselineStrategy;
pub use strategy::{DecisionLog, EquityRow, FillRow, RunState};

#[derive(Debug)]
pub struct Simulation {
    pub state: RunState,
    pub reports: Reports,
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
    Ok(Simulation { state, reports })
}

pub fn run(config_path: &Path) -> Result<PathBuf> {
    let started = Instant::now();
    let config = Config::load(config_path)?;
    let snapshot = Snapshot::load(&config.data_path)?;
    let output = create_run_directory(&config.output_root, "")?;
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
    let mut manifest = output::manifest(&output, &frozen_config, &config)?;
    manifest["snapshot_sha256"] = json!(snapshot.sha256);
    manifest["session_count"] = json!(snapshot.sessions.len());
    manifest["first_session"] = json!(snapshot.sessions[0].timestamp);
    manifest["last_session"] = json!(snapshot.sessions.last().unwrap().timestamp);
    manifest["assumptions"] = json!({
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
    write_reports(&output, &simulation.reports)?;
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
