//! Portfolio experiment settings and validation shared with the single-fund demo.
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result, ensure};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rebalance_every: Option<usize>,
    pub execution: String,
    pub output_root: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_manifest: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trading_start: Option<NaiveDate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<OptimizerConfig>,
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
        if let Some(settings) = &mut config.optimizer {
            settings.python_path = parent.join(&settings.python_path);
            settings.replay_path = settings.replay_path.as_ref().map(|p| parent.join(p));
        }
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.mode == "portfolio", "mode must be portfolio");
        validate_execution(
            self.schema_version,
            &self.starting_cash,
            &self.fee_bps,
            &self.execution,
            self.rebalance_every,
        )?;
        ensure!(!self.assets.is_empty(), "Portfolio needs at least one fund");
        ensure!(
            [
                "no_trade",
                "buy_and_hold",
                "scheduled",
                "monthly",
                "optimized"
            ]
            .contains(&self.strategy.as_str()),
            "Unsupported portfolio strategy"
        );
        ensure!(
            self.strategy != "scheduled" || self.rebalance_every.is_some(),
            "scheduled requires rebalance_every"
        );
        ensure!(
            !["monthly", "optimized"].contains(&self.strategy.as_str())
                || (self.is_historical() && self.rebalance_every.is_none()),
            "Calendar strategies require historical data and must omit rebalance_every"
        );
        ensure!(
            (self.strategy == "optimized") == self.optimizer.is_some(),
            "Only optimized strategy requires optimizer settings"
        );
        ensure!(
            self.trading_start.is_none() || self.is_historical(),
            "trading_start requires historical data"
        );
        if let Some(settings) = &self.optimizer {
            settings.validate(self.assets.len())?;
            ensure!(
                self.trading_start.is_some() && self.reserve()? < Decimal::ONE,
                "Optimizer requires trading_start and a reserve below 1"
            );
        }
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
            validate_symbol(&asset.symbol)?;
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
        self.data_kind == crate::data::historical::KIND
    }
    pub fn price_precision(&self) -> u8 {
        if self.is_historical() { 6 } else { 2 }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerConfig {
    pub python_path: PathBuf,
    pub select_count: usize,
    pub lookback_returns: usize,
    pub forecast_horizon_sessions: usize,
    pub risk_aversion: String,
    pub turnover_penalty: String,
    pub simulated_delay_seconds: u64,
    pub timeout_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_path: Option<PathBuf>,
}

impl OptimizerConfig {
    pub fn validate(&self, count: usize) -> Result<()> {
        ensure!(
            (2..=16).contains(&count) && (1..=count).contains(&self.select_count),
            "Optimizer supports 2–16 assets and a feasible select_count"
        );
        ensure!(
            (2..=2520).contains(&self.lookback_returns)
                && (1..=252).contains(&self.forecast_horizon_sessions),
            "Unsupported lookback/horizon"
        );
        ensure!(
            (Decimal::ZERO..=Decimal::from(1000))
                .contains(&decimal(&self.risk_aversion, "risk_aversion")?)
                && (Decimal::ZERO..=Decimal::ONE)
                    .contains(&decimal(&self.turnover_penalty, "turnover_penalty")?),
            "Invalid optimizer risk/turnover parameter"
        );
        ensure!(
            (1..=60).contains(&self.timeout_seconds) && self.simulated_delay_seconds <= 604_800,
            "Worker timeout must be 1–60 seconds; simulated delay at most seven days"
        );
        Ok(())
    }
}

/// Parse numeric input without introducing floating-point money arithmetic.
pub fn decimal(value: &str, name: &str) -> Result<Decimal> {
    Decimal::from_str(value).with_context(|| format!("{name} must be a finite decimal string"))
}

pub(crate) fn validate_symbol(symbol: &str) -> Result<()> {
    ensure!(
        !symbol.is_empty()
            && symbol.len() <= 20
            && symbol
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()),
        "symbol must be 1–20 uppercase letters or digits"
    );
    Ok(())
}

pub(crate) fn validate_execution(
    schema_version: u32,
    starting_cash: &str,
    fee_bps: &str,
    execution: &str,
    rebalance_every: Option<usize>,
) -> Result<()> {
    ensure!(schema_version == 1, "Unsupported config schema_version");
    ensure!(
        execution == "next_session_close",
        "Only next_session_close is implemented"
    );
    let cash = decimal(starting_cash, "starting_cash")?;
    ensure!(
        cash > Decimal::ZERO && cash <= Decimal::from(1_000_000_000u64) && cash == cash.round_dp(2),
        "starting_cash must be positive USD cents, at most 1 billion"
    );
    ensure!(
        (Decimal::ZERO..=Decimal::from(100)).contains(&decimal(fee_bps, "fee_bps")?),
        "fee_bps must be between 0 and 100 (100 bps = 1%)"
    );
    ensure!(
        rebalance_every.is_none_or(|interval| interval >= 2),
        "rebalance_every must be at least two sessions"
    );
    Ok(())
}
