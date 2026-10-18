//! The second lesson's recipe: several funds, one cash account, equal target weights.
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::config::{Config, decimal};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub symbol: String,
    pub data_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PortfolioConfig {
    pub schema_version: u32,
    pub mode: String,
    pub data_kind: String,
    pub assets: Vec<Asset>,
    pub strategy: String,
    pub starting_cash: String,
    pub cash_reserve: String,
    pub fee_bps: String,
    pub rebalance_every: usize,
    pub execution: String,
    pub output_root: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_manifest: Option<PathBuf>,
}

impl PortfolioConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let mut config: Self = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("Invalid portfolio config: {}", path.display()))?;
        let parent = path.parent().unwrap();
        for asset in &mut config.assets {
            asset.data_path = parent.join(&asset.data_path);
        }
        config.output_root = parent.join(&config.output_root);
        config.snapshot_manifest = config.snapshot_manifest.map(|p| parent.join(p));
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.mode == "portfolio", "mode must be portfolio");
        ensure!(!self.assets.is_empty(), "Portfolio needs at least one fund");
        ensure!(
            ["no_trade", "buy_and_hold", "scheduled"].contains(&self.strategy.as_str()),
            "Portfolio strategy must be no_trade, buy_and_hold, or scheduled"
        );
        ensure!(
            (Decimal::ZERO..=Decimal::ONE).contains(&self.reserve()?),
            "cash_reserve must be between 0 and 1"
        );
        let mut symbols = BTreeSet::new();
        ensure!(
            self.data_kind == "synthetic_total_return_units" || self.is_historical(),
            "Unsupported portfolio data_kind"
        );
        ensure!(
            self.is_historical() == self.snapshot_manifest.is_some(),
            "Only historical data requires snapshot_manifest"
        );
        for asset in &self.assets {
            ensure!(
                symbols.insert(&asset.symbol),
                "Duplicate symbol: {}",
                asset.symbol
            );
            // Reuse lesson one's currency, precision, fee, symbol, and timing checks.
            Config {
                schema_version: self.schema_version,
                data_path: asset.data_path.clone(),
                data_kind: "synthetic_total_return_units".into(),
                symbol: asset.symbol.clone(),
                strategy: self.strategy.clone(),
                starting_cash: self.starting_cash.clone(),
                target_weight: self.weight()?.to_string(),
                fee_bps: self.fee_bps.clone(),
                rebalance_every: self.rebalance_every,
                decisions_path: None,
                execution: self.execution.clone(),
                output_root: self.output_root.clone(),
            }
            .validate()?;
        }
        Ok(())
    }

    pub fn cash(&self) -> Result<Decimal> {
        decimal(&self.starting_cash, "starting_cash")
    }
    pub fn reserve(&self) -> Result<Decimal> {
        decimal(&self.cash_reserve, "cash_reserve")
    }
    pub fn fee_bps(&self) -> Result<Decimal> {
        decimal(&self.fee_bps, "fee_bps")
    }
    pub fn weight(&self) -> Result<Decimal> {
        ensure!(!self.assets.is_empty(), "No assets");
        Ok((Decimal::ONE - self.reserve()?) / Decimal::from(self.assets.len() as u64))
    }

    pub fn is_historical(&self) -> bool {
        self.data_kind == crate::historical::KIND
    }
    pub fn price_precision(&self) -> u8 {
        if self.is_historical() { 6 } else { 2 }
    }
}
