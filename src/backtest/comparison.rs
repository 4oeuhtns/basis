//! Compare a strategy against its baseline using the same frozen inputs and execution rules.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rust_decimal::Decimal;
use serde::Serialize;
use serde_json::json;

use crate::{
    backtest::{self, EquityRow},
    config::{PortfolioConfig, decimal},
    output::{create_run_directory, read_json, write_csv, write_json},
};

fn equity(path: &Path) -> Result<Vec<EquityRow>> {
    Ok(csv::Reader::from_path(path)?
        .deserialize()
        .collect::<std::result::Result<_, _>>()?)
}

#[derive(Serialize)]
struct Row {
    timestamp: String,
    strategy_equity: Decimal,
    benchmark_equity: Decimal,
    difference: Decimal,
}

pub fn run(config_path: &Path) -> Result<PathBuf> {
    let mut config = PortfolioConfig::load(config_path)?;
    ensure!(
        ["monthly", "optimized"].contains(&config.strategy.as_str()),
        "Comparison requires a monthly or optimized historical config"
    );
    let primary = config.strategy.clone();
    let benchmark = if primary == "optimized" {
        "monthly"
    } else {
        "buy_and_hold"
    };
    let output = create_run_directory(&config.output_root, "-comparison")?;
    let report_path = output.join("comparison.json");
    let mut report = json!({"schema_version": 1, "status": "running",
        "description": format!("{primary} versus {benchmark} on identical frozen inputs"),
        "strategy_method": primary, "benchmark_method": benchmark,
        "interpretation": "descriptive baseline comparison on this frozen dataset; no strategy-selection or superiority claim"});
    write_json(&report_path, &report)?;
    let results = (|| -> Result<()> {
        config.output_root = output.join(&primary);
        let strategy_run = backtest::run_config(config)?;
        // Build the control from the first run's frozen inputs, not a second source download/read.
        let mut control = PortfolioConfig::load(&strategy_run.join("config.json"))?;
        control.strategy = benchmark.into();
        control.optimizer = None;
        control.output_root = output.join(benchmark);
        let benchmark_run = backtest::run_config(control)?;
        let strategy_manifest = read_json(&strategy_run.join("manifest.json"))?;
        let benchmark_manifest = read_json(&benchmark_run.join("manifest.json"))?;
        ensure!(
            strategy_manifest["historical_manifest_sha256"]
                == benchmark_manifest["historical_manifest_sha256"],
            "Comparison input manifests differ"
        );
        let a = equity(&strategy_run.join("equity.csv"))?;
        let b = equity(&benchmark_run.join("equity.csv"))?;
        ensure!(a.len() == b.len(), "Comparison session count mismatch");
        let rows: Vec<Row> = a
            .into_iter()
            .zip(b)
            .map(|(a, b)| {
                ensure!(a.timestamp == b.timestamp, "Comparison timestamps differ");
                Ok(Row {
                    timestamp: a.timestamp,
                    strategy_equity: a.equity,
                    benchmark_equity: b.equity,
                    difference: a.equity - b.equity,
                })
            })
            .collect::<Result<_>>()?;
        write_csv(&output.join("comparison.csv"), &rows)?;
        report["historical_manifest_sha256"] =
            strategy_manifest["historical_manifest_sha256"].clone();
        report["session_count"] = json!(rows.len());
        for (label, path) in [
            (primary.as_str(), &strategy_run),
            (benchmark, &benchmark_run),
        ] {
            report[label] = json!({"run_directory": path.strip_prefix(&output)?,
                "summary": read_json(&path.join("summary.json"))?});
        }
        report["ending_equity_difference"] =
            json!(rows.last().context("Empty comparison")?.difference);
        Ok(())
    })();
    match results {
        Ok(_) => report["status"] = json!("completed"),
        Err(error) => {
            report["status"] = json!("failed");
            report["error"] = json!(format!("{error:#}"));
            write_json(&report_path, &report)?;
            return Err(error.context(format!("Comparison evidence: {}", output.display())));
        }
    }
    write_json(&report_path, &report)?;
    for label in [primary.as_str(), benchmark] {
        let summary = &report[label]["summary"];
        let ending = decimal(
            summary["ending_equity"]
                .as_str()
                .context("Missing ending equity")?,
            "ending equity",
        )?;
        println!(
            "{label}: ending equity ${:.2}, {} fills, fees ${}",
            ending.round_dp(2),
            summary["fill_count"],
            summary["total_fees"].as_str().unwrap_or("?")
        );
    }
    println!("Comparison saved: {}", output.display());
    Ok(output)
}
