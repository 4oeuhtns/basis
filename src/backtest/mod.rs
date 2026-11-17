//! Portfolio backtests: configure Nautilus, advance sessions, and preserve run evidence.
pub mod comparison;
pub(crate) mod engine;
mod state;
mod strategy;

pub use state::{DecisionLog, EquityRow, FillRow, HoldingRow, RunState};
use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use rust_decimal::Decimal;
use serde_json::json;

use crate::{
    config::PortfolioConfig,
    data::{PortfolioSnapshot, sha256},
    output::{self, create_run_directory, write_csv, write_json, write_reports},
};
pub use engine::Reports;
use strategy::PortfolioStrategy;

#[derive(Debug)]
pub struct Simulation {
    pub state: RunState,
    pub reports: Reports,
}

pub fn simulate(config: &PortfolioConfig, snapshot: &PortfolioSnapshot) -> Result<Simulation> {
    let bars = snapshot.bars(config)?;
    let indices = snapshot.trading_indices(config);
    ensure!(
        indices.len() >= 2,
        "At least two trading sessions are required"
    );
    let replay = if let Some(settings) = &config.optimizer {
        ensure!(
            indices[0] >= settings.lookback_returns,
            "Insufficient warmup before trading_start"
        );
        crate::research::replay(settings.replay_path.as_deref())?
    } else {
        None
    };
    let mut engine = engine::new_engine(config.cash()?, config.fee_bps()?)?;
    for asset in &config.assets {
        engine.add_instrument(&engine::equity_with_precision(
            &asset.symbol,
            config.price_precision(),
        )?)?;
    }
    let state = Rc::new(RefCell::new(RunState::default()));
    engine.add_strategy(PortfolioStrategy::new(config.clone(), Rc::clone(&state))?)?;
    if config.optimizer.is_some() {
        // Every batch contains the full same-timestamp cross-section. Pause before calling Python.
        for frame in bars.chunks(config.assets.len()) {
            engine.clear_data();
            engine.add_data(frame.to_vec(), None, true, true)?;
            if let Err(error) = engine.run(None, None, None, true) {
                state
                    .borrow_mut()
                    .errors
                    .push(format!("Engine failure: {error}"));
                break;
            }
            if !state.borrow().errors.is_empty() {
                break;
            }
            let anchor = { state.borrow_mut().optimizer_requested.take() };
            if let Some(anchor) = anchor {
                let workspace = config.output_root.join(".optimizer-work");
                fs::create_dir_all(&workspace)?;
                let result = crate::research::coordinate(
                    config,
                    snapshot,
                    anchor,
                    &mut state.borrow_mut(),
                    replay.as_deref(),
                    &workspace,
                );
                if let Err(error) = result {
                    state
                        .borrow_mut()
                        .errors
                        .push(format!("Optimizer protocol failure: {error:#}"));
                    break;
                }
            }
        }
        if let Err(error) = engine.end() {
            state
                .borrow_mut()
                .errors
                .push(format!("Engine finalization failure: {error}"));
        }
        if let Some(replay) = replay
            && state.borrow().optimizer_records.len() != replay.len()
        {
            state
                .borrow_mut()
                .errors
                .push("Replay contains unmatched optimizer records".into());
        }
    } else {
        engine.add_data(bars, None, true, true)?;
        if let Err(error) = engine.run(None, None, None, false) {
            state
                .borrow_mut()
                .errors
                .push(format!("Engine failure: {error}"));
        }
    }
    let reports = engine::reports(&engine)?;
    engine.dispose();
    let mut state = state.borrow().clone();
    if state.equity.len() != indices.len()
        || state.holdings.len() != indices.len() * config.assets.len()
    {
        state
            .errors
            .push("Missing portfolio account/holding snapshots".into());
    }
    Ok(Simulation { state, reports })
}

pub fn run(config_path: &Path) -> Result<PathBuf> {
    run_config(PortfolioConfig::load(config_path)?)
}

pub fn run_config(config: PortfolioConfig) -> Result<PathBuf> {
    let started = Instant::now();
    config.validate()?;
    let snapshot = PortfolioSnapshot::load(&config)?;
    let output = create_run_directory(&config.output_root, "")?;
    let mut frozen = config.clone();
    frozen.output_root = PathBuf::from("..");
    if let Some(settings) = &mut frozen.optimizer {
        settings.replay_path = Some(PathBuf::from("solver_replay.json"));
    }
    let mut hashes = std::collections::BTreeMap::new();
    if let Some(path) = &config.snapshot_manifest {
        crate::data::historical::copy_snapshot(path, &output.join("snapshot"))?;
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
    let trading_sessions = snapshot.trading_indices(&config).len();
    let mut manifest = output::manifest(&output, &frozen, &config)?;
    manifest["mode"] = json!("portfolio");
    manifest["snapshot_sha256"] = json!(hashes);
    manifest["session_count"] = json!(trading_sessions);
    manifest["warmup_session_count"] = json!(snapshot.session_count() - trading_sessions);
    manifest["trading_start"] = json!(config.trading_start);
    manifest["assumptions"] = json!({
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
    });
    if let Some(path) = &config.snapshot_manifest {
        let bytes = fs::read(path)?;
        manifest["historical_snapshot"] = serde_json::from_slice(&bytes)?;
        manifest["historical_manifest_sha256"] = json!(sha256(&bytes));
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
    if ["monthly", "optimized"].contains(&config.strategy.as_str()) {
        let mut dates =
            crate::data::historical::month_end_dates(config.snapshot_manifest.as_ref().unwrap())?;
        dates.retain(|date| config.trading_start.is_none_or(|start| *date >= start));
        manifest["month_end_dates"] = json!(dates);
        manifest["assumptions"]["schedule"] = json!(
            "initial allocation at first close; rebalance at calendar month-end closes; next-session-close execution; partial final calendar month excluded"
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
    if config.optimizer.is_some() {
        write_json(&output.join("optimizer_log.json"), &state.optimizer_records)?;
        write_json(&output.join("solver_replay.json"), &state.optimizer_records)?;
        write_json(
            &output.join("optimizer_events.json"),
            &state.optimizer_events,
        )?;
        manifest["assumptions"]["optimizer"] = json!(
            "past-only sequential exact enumeration outside engine callbacks; explicit simulated delay; invalid/failed/timed-out worker retains holdings; frozen responses replay without Python"
        );
    }
    write_csv(&output.join("equity.csv"), &state.equity)?;
    write_csv(&output.join("holdings.csv"), &state.holdings)?;
    write_json(&output.join("fills.json"), &state.fills)?;
    write_json(&output.join("decision_log.json"), &state.decisions)?;
    write_json(&output.join("order_events.json"), &state.order_events)?;
    write_reports(&output, &simulation.reports)?;
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
            "optimizer_decisions": state.optimizer_records.len(),
            "optimizer_fallbacks": state.optimizer_records.iter().filter(|r|r.status=="fallback").count(),
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
        last.equity.round_dp(2),
        fees,
        state.fills.len()
    );
    for holding in holdings {
        println!(
            "{}: {} units @ ${:.2}",
            holding.instrument_id,
            holding.units,
            holding.close.round_dp(2)
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
