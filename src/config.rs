//! A config is the recipe for one experiment. Reject typos instead of guessing.
use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result, ensure};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub data_path: PathBuf,
    pub data_kind: String,
    pub symbol: String,
    pub strategy: String,
    pub starting_cash: String,
    pub target_weight: String,
    pub fee_bps: String,
    pub rebalance_every: usize,
    pub decisions_path: Option<PathBuf>,
    pub execution: String,
    pub output_root: PathBuf,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let mut config: Self = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("Invalid config: {}", path.display()))?;
        let parent = path.parent().unwrap_or(Path::new("."));
        config.data_path = parent.join(&config.data_path);
        config.output_root = parent.join(&config.output_root);
        config.decisions_path = config.decisions_path.map(|p| parent.join(p));
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "Unsupported config schema_version"
        );
        ensure!(
            self.data_kind == "synthetic_total_return_units",
            "This first adapter accepts synthetic_total_return_units only"
        );
        ensure!(
            !self.symbol.is_empty()
                && self.symbol.len() <= 20
                && self
                    .symbol
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()),
            "symbol must be 1–20 uppercase letters or digits"
        );
        ensure!(
            ["no_trade", "buy_and_hold", "scheduled", "replay"].contains(&self.strategy.as_str()),
            "strategy must be no_trade, buy_and_hold, scheduled, or replay"
        );
        ensure!(
            self.execution == "next_session_close",
            "Only next_session_close is implemented"
        );
        let cash = self.cash()?;
        ensure!(
            cash > Decimal::ZERO
                && cash <= Decimal::from(1_000_000_000u64)
                && cash == cash.round_dp(2),
            "starting_cash must be positive USD cents, at most 1 billion"
        );
        ensure!(
            (Decimal::ZERO..=Decimal::ONE).contains(&self.weight()?),
            "target_weight must be between 0 and 1"
        );
        ensure!(
            (Decimal::ZERO..=Decimal::from(100)).contains(&self.fee_bps()?),
            "fee_bps must be between 0 and 100 (100 bps = 1%)"
        );
        ensure!(
            self.rebalance_every >= 2,
            "rebalance_every must be at least two sessions"
        );
        ensure!(
            (self.strategy == "replay") == self.decisions_path.is_some(),
            "Only replay requires decisions_path; other strategies must leave it null"
        );
        Ok(())
    }

    pub fn cash(&self) -> Result<Decimal> {
        decimal(&self.starting_cash, "starting_cash")
    }
    pub fn weight(&self) -> Result<Decimal> {
        decimal(&self.target_weight, "target_weight")
    }
    pub fn fee_bps(&self) -> Result<Decimal> {
        decimal(&self.fee_bps, "fee_bps")
    }
}

pub fn decimal(value: &str, name: &str) -> Result<Decimal> {
    Decimal::from_str(value).with_context(|| format!("{name} must be a finite decimal string"))
}
