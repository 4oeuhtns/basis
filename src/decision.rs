//! The written instruction card shared with Python. No Python SDK types cross this boundary.
use std::{collections::BTreeMap, fs};

use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    config::{Config, decimal},
    data::Snapshot,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PortfolioDecision {
    pub schema_version: u32,
    pub data_cutoff: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub target_weights: BTreeMap<String, String>,
    pub input_snapshot_hash: String,
    pub method: String,
    pub status: String,
}

pub fn load_decisions(config: &Config, snapshot: &Snapshot) -> Result<Vec<PortfolioDecision>> {
    let Some(path) = &config.decisions_path else {
        return Ok(Vec::new());
    };
    let decisions: Vec<PortfolioDecision> = serde_json::from_slice(&fs::read(path)?)?;
    // One instruction is enough to prove the language boundary before adding a solver schedule.
    ensure!(
        decisions.len() == 1,
        "The first replay adapter expects exactly one decision"
    );
    for d in &decisions {
        ensure!(
            d.schema_version == 1 && d.status == "feasible",
            "Unsupported decision version/status"
        );
        ensure!(
            d.input_snapshot_hash == snapshot.sha256,
            "Decision snapshot hash does not match"
        );
        ensure!(
            d.available_at >= d.data_cutoff,
            "Decision cannot arrive before its data cutoff"
        );
        ensure!(
            snapshot
                .sessions
                .iter()
                .any(|s| s.timestamp == d.data_cutoff),
            "Decision cutoff must be a recorded session close"
        );
        ensure!(
            d.available_at < snapshot.sessions.last().unwrap().timestamp,
            "Decision arrives too late for a subsequent execution session"
        );
        let id = format!("{}.SIM", config.symbol);
        ensure!(
            d.target_weights.len() == 1 && d.target_weights.contains_key(&id),
            "Only this run's single instrument may have a target weight"
        );
        let weight = decimal(&d.target_weights[&id], "decision target_weight")?;
        ensure!(
            (rust_decimal::Decimal::ZERO..=rust_decimal::Decimal::ONE).contains(&weight),
            "Decision weight must be between 0 and 1"
        );
        ensure!(!d.method.is_empty(), "Decision method is required");
    }
    Ok(decisions)
}
