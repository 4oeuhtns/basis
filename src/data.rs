//! Audit the recording before replaying it. No sorting or filling missing days silently.
use std::{fs, path::Path};

use anyhow::{Result, ensure};
use chrono::{DateTime, Datelike, Duration, Timelike, Utc, Weekday};
use nautilus_model::{
    data::{Bar, BarType, Data},
    types::{Price, Quantity},
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::decimal;

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
            .map(|s| {
                let ts = u64::try_from(
                    s.timestamp
                        .timestamp_nanos_opt()
                        .ok_or_else(|| anyhow::anyhow!("Timestamp outside nanosecond range"))?,
                )?;
                Ok(Data::Bar(Bar::new_checked(
                    bar_type,
                    Price::from_decimal_dp(s.open, precision)?,
                    Price::from_decimal_dp(s.high, precision)?,
                    Price::from_decimal_dp(s.low, precision)?,
                    Price::from_decimal_dp(s.close, precision)?,
                    Quantity::from_decimal_dp(Decimal::from(s.volume), 0)?,
                    ts.into(),
                    ts.into(),
                )?))
            })
            .collect()
    }
}
