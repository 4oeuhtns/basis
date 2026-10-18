//! Read our portable snapshot schema. No network, Python process, or legacy fixed-point decoding.
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path},
};

use anyhow::{Context, Result, ensure};
use bytes::Bytes;
use chrono::{DateTime, Datelike, NaiveDate, Utc, Weekday};
use parquet::{
    file::{reader::FileReader, serialized_reader::SerializedFileReader},
    record::RowAccessor,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    config::decimal,
    data::{Session, Snapshot, sha256},
    portfolio::{PortfolioSnapshot, config::PortfolioConfig},
};

pub const KIND: &str = "historical_adjusted_research_units";
const COLUMNS: [&str; 7] = [
    "session_date",
    "available_at_ns",
    "open",
    "high",
    "low",
    "close",
    "volume",
];

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub data_kind: String,
    pub calendar: String,
    pub start: NaiveDate,
    pub end_exclusive: NaiveDate,
    pub price_precision: u8,
    pub adjustment_policy: String,
    pub assets: Vec<String>,
    pub created_at_utc: DateTime<Utc>,
    pub source: Value,
    pub environment: Value,
    pub files: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalendarRow {
    session_date: NaiveDate,
    open_ns: i64,
    close_ns: i64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    session_date: NaiveDate,
    available_at_ns: i64,
    open: String,
    high: String,
    low: String,
    close: String,
    volume: u64,
}

fn safe_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_')
}

pub fn read_manifest(path: &Path) -> Result<Manifest> {
    let manifest: Manifest = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        manifest.schema_version == 1
            && manifest.data_kind == KIND
            && manifest.price_precision == 6
            && manifest.calendar == "XNYS"
            && manifest.adjustment_policy == "adj_close_ratio_ohlc_research_units",
        "Unsupported historical snapshot schema/policy"
    );
    ensure!(
        manifest.start < manifest.end_exclusive,
        "Invalid snapshot date range"
    );
    let symbols: BTreeSet<_> = manifest.assets.iter().collect();
    ensure!(
        !symbols.is_empty() && symbols.len() == manifest.assets.len(),
        "Duplicate/missing snapshot assets"
    );
    let mut expected = BTreeSet::from(["calendar.csv".to_string()]);
    for symbol in &manifest.assets {
        ensure!(
            !symbol.is_empty()
                && symbol.len() <= 20
                && symbol
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()),
            "Invalid snapshot symbol"
        );
        for suffix in [".csv", ".parquet", ".source.parquet"] {
            expected.insert(format!("{symbol}{suffix}"));
        }
    }
    ensure!(
        manifest.files.keys().cloned().collect::<BTreeSet<_>>() == expected,
        "Snapshot file inventory mismatch"
    );
    let root = path.parent().context("Manifest directory missing")?;
    for (name, digest) in &manifest.files {
        ensure!(safe_name(name), "Invalid snapshot filename");
        ensure!(
            sha256(&fs::read(root.join(name))?) == *digest,
            "Snapshot hash mismatch: {name}"
        );
    }
    Ok(manifest)
}

fn calendar(path: &Path, manifest: &Manifest) -> Result<Vec<CalendarRow>> {
    let mut reader = csv::Reader::from_path(path)?;
    ensure!(
        reader
            .headers()?
            .iter()
            .eq(["session_date", "open_ns", "close_ns"]),
        "Invalid calendar header"
    );
    let rows: Vec<CalendarRow> = reader
        .deserialize()
        .collect::<std::result::Result<_, _>>()?;
    ensure!(rows.len() >= 2, "Historical snapshot needs two sessions");
    let downloaded = manifest
        .created_at_utc
        .timestamp_nanos_opt()
        .context("Invalid snapshot creation time")?;
    for (i, row) in rows.iter().enumerate() {
        ensure!(
            row.session_date >= manifest.start && row.session_date < manifest.end_exclusive,
            "Calendar session outside requested dates"
        );
        ensure!(
            row.open_ns > 0 && row.close_ns > row.open_ns && row.close_ns <= downloaded,
            "Invalid/unfinished session in calendar"
        );
        ensure!(
            !matches!(row.session_date.weekday(), Weekday::Sat | Weekday::Sun),
            "Weekend in XNYS calendar"
        );
        ensure!(
            DateTime::<Utc>::from_timestamp_nanos(row.close_ns).date_naive() == row.session_date,
            "Calendar close and session date differ"
        );
        if i > 0 {
            ensure!(
                row.session_date > rows[i - 1].session_date && row.open_ns > rows[i - 1].close_ns,
                "Unordered/duplicate calendar sessions"
            );
        }
    }
    Ok(rows)
}

fn rows(bytes: &[u8], extension: &str) -> Result<Vec<Row>> {
    if extension == "csv" {
        let mut reader = csv::Reader::from_reader(bytes);
        ensure!(
            reader.headers()?.iter().eq(COLUMNS),
            "Historical CSV header mismatch"
        );
        return Ok(reader
            .deserialize()
            .collect::<std::result::Result<_, _>>()?);
    }
    ensure!(
        extension == "parquet",
        "Historical data must be .parquet or .csv"
    );
    let reader = SerializedFileReader::new(Bytes::copy_from_slice(bytes))?;
    ensure!(
        reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|c| c.name())
            .eq(COLUMNS),
        "Historical Parquet schema mismatch; legacy catalogs require explicit conversion"
    );
    reader
        .get_row_iter(None)?
        .map(|row| {
            let row = row?;
            Ok(Row {
                session_date: row.get_string(0)?.parse()?,
                available_at_ns: row.get_long(1)?,
                open: row.get_string(2)?.clone(),
                high: row.get_string(3)?.clone(),
                low: row.get_string(4)?.clone(),
                close: row.get_string(5)?.clone(),
                volume: u64::try_from(row.get_long(6)?)?,
            })
        })
        .collect()
}

pub fn load(config: &PortfolioConfig) -> Result<PortfolioSnapshot> {
    config.validate()?;
    let path = config
        .snapshot_manifest
        .as_ref()
        .context("Historical snapshot_manifest required")?;
    let manifest = read_manifest(path)?;
    ensure!(
        config
            .assets
            .iter()
            .map(|a| &a.symbol)
            .collect::<BTreeSet<_>>()
            == manifest.assets.iter().collect::<BTreeSet<_>>(),
        "Config/snapshot asset sets differ"
    );
    let root = path.parent().unwrap();
    let calendar = calendar(&root.join("calendar.csv"), &manifest)?;
    let mut assets = BTreeMap::new();
    for asset in &config.assets {
        let name = asset
            .data_path
            .file_name()
            .context("Missing asset filename")?
            .to_str()
            .context("Non-UTF8 filename")?;
        ensure!(
            [
                format!("{}.parquet", asset.symbol),
                format!("{}.csv", asset.symbol)
            ]
            .contains(&name.to_string()),
            "Asset filename does not match symbol"
        );
        ensure!(
            asset.data_path.canonicalize()? == root.join(name).canonicalize()?,
            "Asset must belong to snapshot directory"
        );
        let bytes = fs::read(&asset.data_path)?;
        ensure!(
            manifest.files.get(name) == Some(&sha256(&bytes)),
            "Asset hash mismatch"
        );
        let rows = rows(
            &bytes,
            asset.data_path.extension().unwrap().to_str().unwrap_or(""),
        )?;
        ensure!(
            rows.len() == calendar.len(),
            "Historical session coverage mismatch for {}",
            asset.symbol
        );
        let sessions = rows
            .into_iter()
            .zip(&calendar)
            .map(|(row, session)| {
                ensure!(
                    row.session_date == session.session_date
                        && row.available_at_ns == session.close_ns,
                    "Historical bar must match calendar session close; no missing/stale bars"
                );
                let prices = [row.open, row.high, row.low, row.close]
                    .map(|p| decimal(&p, "historical price"))
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
                let [open, high, low, close]: [Decimal; 4] = prices.try_into().unwrap();
                ensure!(
                    [open, high, low, close].iter().all(|p| *p > Decimal::ZERO
                        && *p <= Decimal::from(1_000_000)
                        && *p == p.round_dp(6)),
                    "Invalid historical price/precision"
                );
                ensure!(
                    high >= open && high >= close && high >= low && low <= open && low <= close,
                    "Invalid historical OHLC range"
                );
                ensure!(
                    (4..=18_446_744_073).contains(&row.volume),
                    "Historical volume outside native Quantity range"
                );
                Ok(Session {
                    timestamp: DateTime::<Utc>::from_timestamp_nanos(row.available_at_ns),
                    open,
                    high,
                    low,
                    close,
                    volume: row.volume,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        assets.insert(
            asset.symbol.clone(),
            Snapshot {
                sha256: sha256(&bytes),
                bytes,
                sessions,
            },
        );
    }
    Ok(PortfolioSnapshot { assets })
}

pub fn copy_snapshot(manifest_path: &Path, destination: &Path) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    fs::create_dir(destination)?;
    for (name, digest) in &manifest.files {
        let bytes = fs::read(manifest_path.parent().unwrap().join(name))?;
        ensure!(
            sha256(&bytes) == *digest,
            "Snapshot changed while freezing run"
        );
        fs::write(destination.join(name), bytes)?;
    }
    fs::copy(manifest_path, destination.join("manifest.json"))?;
    Ok(())
}
