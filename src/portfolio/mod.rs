//! Lesson two: synchronized prices, shared cash, and sell-before-buy rebalancing.
pub mod config;
pub mod runner;
pub mod strategy;

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use nautilus_model::data::{BarType, Data};

use crate::data::Snapshot;
use config::PortfolioConfig;

#[derive(Debug)]
pub struct PortfolioSnapshot {
    pub assets: BTreeMap<String, Snapshot>,
}

impl PortfolioSnapshot {
    pub fn load(config: &PortfolioConfig) -> Result<Self> {
        config.validate()?;
        if config.is_historical() {
            return crate::historical::load(config);
        }
        let assets = config
            .assets
            .iter()
            .map(|asset| {
                Ok((
                    asset.symbol.clone(),
                    Snapshot::load(&asset.data_path)
                        .with_context(|| format!("Invalid prices for {}", asset.symbol))?,
                ))
            })
            .collect::<Result<_>>()?;
        let snapshot = Self { assets };
        snapshot.validate(config)?;
        Ok(snapshot)
    }

    pub fn validate(&self, config: &PortfolioConfig) -> Result<()> {
        config.validate()?;
        if config.is_historical() {
            let expected = crate::historical::load(config)?;
            ensure!(
                self.assets.len() == expected.assets.len(),
                "Historical asset count mismatch"
            );
            for (symbol, asset) in &expected.assets {
                let actual = self
                    .assets
                    .get(symbol)
                    .context("Missing historical asset")?;
                ensure!(
                    actual.sha256 == asset.sha256
                        && actual.bytes == asset.bytes
                        && actual.sessions == asset.sessions,
                    "Historical snapshot differs from verified files"
                );
            }
            return Ok(());
        }
        ensure!(
            self.assets.len() == config.assets.len(),
            "Snapshot asset count mismatch"
        );
        let mut expected = None;
        for asset in &config.assets {
            let snapshot = self
                .assets
                .get(&asset.symbol)
                .context("Missing asset snapshot")?;
            // Validate bytes again so callers cannot supply an inconsistent parsed snapshot.
            let parsed = Snapshot::parse(snapshot.bytes.clone())?;
            ensure!(
                parsed.sha256 == snapshot.sha256
                    && serde_json::to_value(&parsed.sessions)?
                        == serde_json::to_value(&snapshot.sessions)?,
                "Snapshot bytes and parsed sessions differ"
            );
            let dates: Vec<_> = snapshot.sessions.iter().map(|s| s.timestamp).collect();
            if let Some(expected) = &expected {
                ensure!(
                    &dates == expected,
                    "All funds must have exactly the same sessions; missing bar for {}",
                    asset.symbol
                );
            } else {
                expected = Some(dates);
            }
        }
        Ok(())
    }

    pub fn session_count(&self) -> usize {
        self.assets.values().next().map_or(0, |s| s.sessions.len())
    }

    pub fn bars(&self, config: &PortfolioConfig) -> Result<Vec<Data>> {
        self.validate(config)?;
        let by_asset: Vec<Vec<Data>> = config
            .assets
            .iter()
            .map(|asset| {
                let bar_type: BarType =
                    format!("{}.SIM-1-DAY-LAST-EXTERNAL", asset.symbol).parse()?;
                self.assets[&asset.symbol].bars_with_precision(bar_type, config.price_precision())
            })
            .collect::<Result<_>>()?;
        let mut data = Vec::new();
        for index in 0..self.session_count() {
            for bars in &by_asset {
                data.push(bars[index].clone());
            }
        }
        Ok(data)
    }
}
