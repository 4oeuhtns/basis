//! Run and preserve the two-fund experiment, using the same Nautilus plumbing as lesson one.
use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use chrono::Utc;
use rust_decimal::Decimal;
use serde_json::json;

use super::{
    PortfolioSnapshot,
    config::PortfolioConfig,
    strategy::{PortfolioStrategy, RunState},
};
use crate::{
    data::sha256,
    engine::{self, Reports},
    runner::{NAUTILUS_VERSION, command_output, compiled_source_hash, write_csv, write_json},
};

#[derive(Debug)]
pub struct Simulation {
    pub state: RunState,
    pub reports: Reports,
}

pub fn simulate(config: &PortfolioConfig, snapshot: &PortfolioSnapshot) -> Result<Simulation> {
    snapshot.validate(config)?;
    let mut engine = engine::new_engine(config.cash()?, config.fee_bps()?)?;
    for asset in &config.assets {
        engine.add_instrument(&engine::equity_with_precision(
            &asset.symbol,
            config.price_precision(),
        )?)?;
    }
    let state = Rc::new(RefCell::new(RunState::default()));
    engine.add_strategy(PortfolioStrategy::new(config.clone(), Rc::clone(&state))?)?;
    engine.add_data(snapshot.bars(config)?, None, true, true)?;
    if let Err(error) = engine.run(None, None, None, false) {
        state
            .borrow_mut()
            .errors
            .push(format!("Engine failure: {error}"));
    }
    let reports = engine::reports(&engine)?;
    engine.dispose();
    let mut state = state.borrow().clone();
    if state.equity.len() != snapshot.session_count()
        || state.holdings.len() != snapshot.session_count() * config.assets.len()
    {
        state
            .errors
            .push("Missing portfolio account/holding snapshots".into());
    }
    Ok(Simulation { state, reports })
}

pub fn run(config_path: &Path) -> Result<PathBuf> {
    let started = Instant::now();
    let config = PortfolioConfig::load(config_path)?;
    let snapshot = PortfolioSnapshot::load(&config)?;
    fs::create_dir_all(&config.output_root)?;
    let run_id = format!(
        "{}-{}",
        Utc::now().format("%Y%m%dT%H%M%S%.9fZ"),
        std::process::id()
    );
    let output = config.output_root.canonicalize()?.join(&run_id);
    fs::create_dir(&output)?;
    let mut frozen = config.clone();
    frozen.output_root = PathBuf::from("..");
    let mut hashes = std::collections::BTreeMap::new();
    if let Some(path) = &config.snapshot_manifest {
        crate::historical::copy_snapshot(path, &output.join("snapshot"))?;
        frozen.snapshot_manifest = Some(PathBuf::from("snapshot/manifest.json"));
    }
    for asset in &mut frozen.assets {
        let data = &snapshot.assets[&asset.symbol];
        if config.is_historical() {
            asset.data_path = PathBuf::from("snapshot").join(asset.data_path.file_name().unwrap());
        } else {
            let name = format!("prices_{}.csv", asset.symbol);
            fs::write(output.join(&name), &data.bytes)?;
            asset.data_path = PathBuf::from(name);
        }
        hashes.insert(asset.symbol.clone(), data.sha256.clone());
    }
    write_json(&output.join("config.json"), &frozen)?;
    write_json(&output.join("resolved_config.json"), &config)?;
    fs::write(
        output.join("Cargo.lock"),
        include_bytes!("../../Cargo.lock"),
    )?;
    let mut manifest = json!({
        "schema_version": 1, "mode": "portfolio", "run_id": run_id, "status": "running",
        "created_at_utc": Utc::now().to_rfc3339(), "nautilus_rust_version": NAUTILUS_VERSION,
        "rustc": command_output("rustc", &["--version"]),
        "git_commit": command_output("git", &["rev-parse", "HEAD"]),
        "git_status": command_output("git", &["status", "--porcelain"]),
        "config_sha256": sha256(&serde_json::to_vec(&frozen)?), "snapshot_sha256": hashes,
        "resolved_config_sha256": sha256(&serde_json::to_vec(&config)?),
        "compiled_source_sha256": compiled_source_hash(),
        "cargo_lock_sha256": sha256(include_bytes!("../../Cargo.lock")),
        "os": std::env::consts::OS, "architecture": std::env::consts::ARCH,
        "session_count": snapshot.session_count(),
        "assumptions": {
            "prices": "fabricated synthetic total-return units; USD cents; no extra dividends",
            "calendar": "consecutive Mon–Fri at 21:00 UTC; no exchange holiday/DST model",
            "timestamps": "ts_event = ts_init = completed close; every asset must share every session",
            "decision": "equal target weights after cash reserve; target whole units fixed at decision close",
            "execution": "wait for all next-session closes, then submit native market orders sequentially",
            "ordering": "all sells settle before buys; alphabetical instrument order within each group",
            "settlement": "immediate fill bookkeeping; no separate settlement-date or unsettled-cash model",
            "funding": "cap buys at actual cash minus prior decision's USD reserve, including configured fees; log caps/skips",
            "reserve": "decision equity * cash_reserve rounded up to cents; not a continuous weight guarantee",
            "quantity": "whole units; no shorts, leverage, or cash borrowing",
            "fees": "native maker/taker fee_bps / 10000; USD money rounding",
            "slippage": "zero spread, impact and latency; ample synthetic liquidity in teaching fixtures",
            "failure": "partial/rejected/unsettled orders fail the run and abort remaining plan; no retry",
            "terminal_positions": "held and marked at final close; no forced liquidation",
            "reporting": "equity = native USD cash + sum(native units * synchronized close)",
            "evaluation": "teaching fixture only; no performance inference; no Python/quantum signal yet"
        }
    });
    if let Some(path) = &config.snapshot_manifest {
        manifest["historical_snapshot"] = serde_json::from_slice(&fs::read(path)?)?;
        manifest["historical_manifest_sha256"] = json!(sha256(&fs::read(path)?));
        manifest["assumptions"]["prices"] = json!(
            "retrospective adjusted research units; USD; six decimal marks; no separate dividends/splits"
        );
        manifest["assumptions"]["calendar"] =
            json!("frozen exchange_calendars XNYS schedule with holidays, DST and early closes");
        manifest["assumptions"]["slippage"] = json!(
            "zero spread, impact and latency; historical source volume used as a liquidity proxy"
        );
        manifest["assumptions"]["evaluation"] = json!(
            "historical plumbing baseline; retrospective adjustments; no point-in-time data or strategy-performance claim"
        );
    }
    let manifest_path = output.join("manifest.json");
    write_json(&manifest_path, &manifest)?;
    let simulation = match simulate(&config, &snapshot) {
        Ok(result) => result,
        Err(error) => {
            manifest["status"] = json!("failed");
            manifest["error"] = json!(format!("{error:#}"));
            write_json(&manifest_path, &manifest)?;
            return Err(error.context(format!("Run evidence: {}", output.display())));
        }
    };
    let state = &simulation.state;
    write_csv(&output.join("equity.csv"), &state.equity)?;
    write_csv(&output.join("holdings.csv"), &state.holdings)?;
    write_json(&output.join("fills.json"), &state.fills)?;
    write_json(&output.join("decision_log.json"), &state.decisions)?;
    write_json(&output.join("order_events.json"), &state.order_events)?;
    write_json(&output.join("orders.json"), &simulation.reports.orders)?;
    write_json(
        &output.join("positions.json"),
        &simulation.reports.positions,
    )?;
    write_json(
        &output.join("account_events.json"),
        &simulation.reports.account_events,
    )?;
    write_json(
        &output.join("engine_result.json"),
        &simulation.reports.engine_result,
    )?;
    let last = state.equity.last().context("No final portfolio snapshot")?;
    let holdings: Vec<_> = state
        .holdings
        .iter()
        .filter(|h| h.timestamp == last.timestamp)
        .collect();
    let fees: Decimal = state.fills.iter().map(|f| f.commission).sum();
    write_json(
        &output.join("summary.json"),
        &json!({
            "starting_cash": config.cash()?, "ending_cash": last.cash,
            "ending_equity": last.equity, "ending_holdings": holdings,
            "net_pnl": last.equity - config.cash()?, "total_fees": fees,
            "fill_count": state.fills.len(), "errors": state.errors,
        }),
    )?;
    manifest["status"] = json!(if state.errors.is_empty() {
        "completed"
    } else {
        "failed"
    });
    manifest["wall_seconds"] = json!(started.elapsed().as_secs_f64());
    write_json(&manifest_path, &manifest)?;
    println!(
        "Starting cash: ${:.2}\nEnding cash:   ${:.2}\nEnding equity: ${:.2}\nFees paid:     ${:.2}\nFills:         {}",
        config.cash()?,
        last.cash,
        last.equity,
        fees,
        state.fills.len()
    );
    for holding in holdings {
        println!(
            "{}: {} units @ ${:.2}",
            holding.instrument_id, holding.units, holding.close
        );
    }
    println!("Run saved:     {}", output.display());
    ensure!(
        state.errors.is_empty(),
        "Portfolio backtest failed; inspect {}",
        output.display()
    );
    Ok(output)
}
