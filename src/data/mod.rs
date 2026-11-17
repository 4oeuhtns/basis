//! Validated market snapshots and deterministic conversion to native Nautilus bars.
pub mod historical;
use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Datelike, Duration, Timelike, Utc, Weekday};
use nautilus_model::{
    data::{Bar, BarType, Data},
    types::{Price, Quantity},
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{PortfolioConfig, decimal};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CsvRow {
    timestamp: DateTime<Utc>,
    open: String,
    high: String,
    low: String,
    close: String,
    volume: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Session {
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
}

impl Session {
    fn bar(&self, bar_type: BarType, precision: u8) -> Result<Data> {
        let ts = u64::try_from(
            self.timestamp
                .timestamp_nanos_opt()
                .context("Timestamp outside nanosecond range")?,
        )?;
        Ok(Data::Bar(Bar::new_checked(
            bar_type,
            Price::from_decimal_dp(self.open, precision)?,
            Price::from_decimal_dp(self.high, precision)?,
            Price::from_decimal_dp(self.low, precision)?,
            Price::from_decimal_dp(self.close, precision)?,
            Quantity::from_decimal_dp(Decimal::from(self.volume), 0)?,
            ts.into(),
            ts.into(),
        )?))
    }
}

#[derive(Debug)]
pub struct Snapshot {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub sessions: Vec<Session>,
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl Snapshot {
    pub fn load(path: &Path) -> Result<Self> {
        Self::parse(fs::read(path)?)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let mut reader = csv::Reader::from_reader(bytes.as_slice());
        let expected = ["timestamp", "open", "high", "low", "close", "volume"];
        ensure!(
            reader.headers()?.iter().eq(expected),
            "CSV header must be {}",
            expected.join(",")
        );
        let mut sessions: Vec<Session> = Vec::new();
        for row in reader.deserialize::<CsvRow>() {
            let row = row?;
            let time = row.timestamp;
            ensure!(
                time.hour() == 21
                    && time.minute() == 0
                    && time.second() == 0
                    && time.nanosecond() == 0
                    && time.timestamp() > 0,
                "Synthetic sessions must close at exactly 21:00:00 UTC after 1970"
            );
            ensure!(
                !matches!(time.weekday(), Weekday::Sat | Weekday::Sun),
                "Synthetic calendar contains weekdays only"
            );
            if let Some(previous) = sessions.last() {
                let mut next = previous.timestamp + Duration::days(1);
                while matches!(next.weekday(), Weekday::Sat | Weekday::Sun) {
                    next += Duration::days(1);
                }
                ensure!(
                    time == next,
                    "Sessions must be consecutive synthetic weekdays: expected {next}, got {time}"
                );
            }
            let [open, high, low, close] = [row.open, row.high, row.low, row.close]
                .map(|p| decimal(&p, "OHLC price"))
                .into_iter()
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .unwrap();
            for price in [open, high, low, close] {
                ensure!(
                    price > Decimal::ZERO
                        && price <= Decimal::from(1_000_000)
                        && price == price.round_dp(2),
                    "Prices must be positive cents, at most 1 million"
                );
            }
            ensure!(
                high >= open && high >= close && high >= low && low <= open && low <= close,
                "Invalid OHLC range at {time}"
            );
            ensure!(
                row.volume >= 4 && row.volume <= 1_000_000_000,
                "Synthetic volume must be 4–1 billion whole units"
            );
            sessions.push(Session {
                timestamp: time,
                open,
                high,
                low,
                close,
                volume: row.volume,
            });
        }
        ensure!(
            sessions.len() >= 2,
            "At least two sessions are needed to test a delayed order"
        );
        Ok(Self {
            sha256: sha256(&bytes),
            bytes,
            sessions,
        })
    }

    pub fn bars(&self, bar_type: BarType) -> Result<Vec<Data>> {
        self.bars_with_precision(bar_type, 2)
    }

    pub fn bars_with_precision(&self, bar_type: BarType, precision: u8) -> Result<Vec<Data>> {
        self.sessions
            .iter()
            .map(|session| session.bar(bar_type, precision))
            .collect()
    }
}

#[derive(Debug)]
pub struct PortfolioSnapshot {
    pub assets: BTreeMap<String, Snapshot>,
}

impl PortfolioSnapshot {
    pub fn load(config: &PortfolioConfig) -> Result<Self> {
        config.validate()?;
        if config.is_historical() {
            return historical::load(config);
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
            let expected = historical::load(config)?;
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
                parsed.sha256 == snapshot.sha256 && parsed.sessions == snapshot.sessions,
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

    pub fn trading_indices(&self, config: &PortfolioConfig) -> Vec<usize> {
        self.assets.values().next().map_or_else(Vec::new, |s| {
            s.sessions
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    config
                        .trading_start
                        .is_none_or(|start| s.timestamp.date_naive() >= start)
                })
                .map(|(i, _)| i)
                .collect()
        })
    }

    pub fn bars(&self, config: &PortfolioConfig) -> Result<Vec<Data>> {
        self.validate(config)?;
        let by_asset: Vec<(&Snapshot, BarType)> = config
            .assets
            .iter()
            .map(|asset| {
                let bar_type: BarType =
                    format!("{}.SIM-1-DAY-LAST-EXTERNAL", asset.symbol).parse()?;
                Ok((&self.assets[&asset.symbol], bar_type))
            })
            .collect::<Result<_>>()?;
        let indices = self.trading_indices(config);
        let mut data = Vec::with_capacity(indices.len() * by_asset.len());
        // Build each trading bar once, directly in synchronized session order.
        // Warmup observations remain available to research without becoming tradable bars.
        for index in indices {
            for (snapshot, bar_type) in &by_asset {
                data.push(snapshot.sessions[index].bar(*bar_type, config.price_precision())?);
            }
        }
        Ok(data)
    }
}

pub(crate) fn timestamp(ns: u64) -> String {
    DateTime::<Utc>::from_timestamp_nanos(ns as i64).to_rfc3339()
}
