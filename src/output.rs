//! Shared run directories, provenance, and report serialization.
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result};
use chrono::Utc;
use serde::Serialize;
use serde_json::{Value, json};

use crate::{backtest::engine::Reports, data::sha256};

const NAUTILUS_VERSION: &str = "0.66.0";
const CARGO_LOCK: &[u8] = include_bytes!("../Cargo.lock");

// Hash compiled sources, including uncommitted code, rather than runtime working-tree files.
fn compiled_source_hash() -> String {
    sha256(
        concat!(
            include_str!("main.rs"),
            include_str!("lib.rs"),
            include_str!("config.rs"),
            include_str!("output.rs"),
            include_str!("research.rs"),
            include_str!("backtest/mod.rs"),
            include_str!("backtest/engine.rs"),
            include_str!("backtest/state.rs"),
            include_str!("backtest/strategy.rs"),
            include_str!("backtest/comparison.rs"),
            include_str!("data/mod.rs"),
            include_str!("data/historical.rs"),
            include_str!("demo/mod.rs"),
            include_str!("demo/config.rs"),
            include_str!("demo/decision.rs"),
            include_str!("demo/strategy.rs"),
            include_str!("../Cargo.toml"),
            include_str!("../qflab/worker.py"),
            include_str!("../qflab/market_data.py"),
            include_str!("../qflab/optimizer_worker.py"),
        )
        .as_bytes(),
    )
}

pub(crate) fn create_run_directory(root: &Path, suffix: &str) -> Result<PathBuf> {
    fs::create_dir_all(root)?;
    let path = root.canonicalize()?.join(format!(
        "{}-{}{suffix}",
        Utc::now().format("%Y%m%dT%H%M%S%.9fZ"),
        std::process::id()
    ));
    // Each run owns a new directory. Existing evidence is never overwritten.
    fs::create_dir(&path)?;
    Ok(path)
}

pub(crate) fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub(crate) fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
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

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub(crate) fn manifest(
    output: &Path,
    frozen_config: &impl Serialize,
    resolved_config: &impl Serialize,
) -> Result<Value> {
    fs::write(output.join("Cargo.lock"), CARGO_LOCK)?;
    Ok(json!({
        "schema_version": 1,
        "run_id": output.file_name().context("Missing run directory name")?,
        "status": "running",
        "created_at_utc": Utc::now().to_rfc3339(),
        "nautilus_rust_version": NAUTILUS_VERSION,
        "rustc": command_output("rustc", &["--version"]),
        "git_commit": command_output("git", &["rev-parse", "HEAD"]),
        "git_status": command_output("git", &["status", "--porcelain"]),
        "config_sha256": sha256(&serde_json::to_vec(frozen_config)?),
        "resolved_config_sha256": sha256(&serde_json::to_vec(resolved_config)?),
        "compiled_source_sha256": compiled_source_hash(),
        "cargo_lock_sha256": sha256(CARGO_LOCK),
        "os": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
    }))
}

pub(crate) fn write_reports(output: &Path, reports: &Reports) -> Result<()> {
    write_json(&output.join("orders.json"), &reports.orders)?;
    write_json(&output.join("positions.json"), &reports.positions)?;
    write_json(&output.join("account_events.json"), &reports.account_events)?;
    write_json(&output.join("engine_result.json"), &reports.engine_result)
}
