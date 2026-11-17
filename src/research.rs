//! Sequential research coordinator. Python runs only while the native engine is paused.
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    backtest::{EquityRow, HoldingRow, RunState},
    config::{OptimizerConfig, PortfolioConfig, decimal},
    data::{PortfolioSnapshot, sha256},
};

#[derive(Debug, Clone, Serialize)]
pub struct Anchor {
    pub equity: EquityRow,
    pub holdings: Vec<HoldingRow>,
    pub trigger: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverRecord {
    pub request: Value,
    pub request_sha256: String,
    pub available_at: DateTime<Utc>,
    pub response: Option<Value>,
    pub status: String,
    pub fallback_reason: Option<String>,
    pub elapsed_wall_seconds: f64,
    pub delivery_status: String,
}

#[derive(Debug, Clone)]
pub struct ReadyDecision {
    pub anchor: Anchor,
    pub weights: BTreeMap<String, Decimal>,
    pub available_at: DateTime<Utc>,
    pub record_index: usize,
}

pub fn build_request(
    config: &PortfolioConfig,
    snapshot: &PortfolioSnapshot,
    anchor: &Anchor,
) -> Result<Value> {
    let settings = config
        .optimizer
        .as_ref()
        .context("Missing optimizer settings")?;
    let cutoff: DateTime<Utc> = anchor.equity.timestamp.parse()?;
    let first = snapshot.assets.values().next().context("No assets")?;
    let index = first
        .sessions
        .iter()
        .position(|s| s.timestamp == cutoff)
        .context("Optimizer cutoff missing")?;
    ensure!(
        index >= settings.lookback_returns,
        "Insufficient warmup: need {} prior returns",
        settings.lookback_returns
    );
    let mut history = Vec::new();
    for i in index - settings.lookback_returns..=index {
        let when = first.sessions[i].timestamp;
        ensure!(when <= cutoff, "Future price in optimizer request");
        let closes: BTreeMap<_, _> = snapshot
            .assets
            .iter()
            .map(|(s, data)| {
                ensure!(
                    data.sessions[i].timestamp == when,
                    "Optimizer prices not synchronized"
                );
                Ok((format!("{s}.SIM"), data.sessions[i].close.to_string()))
            })
            .collect::<Result<_>>()?;
        history.push(json!({"timestamp": when.to_rfc3339(), "closes": closes}));
    }
    let holdings: BTreeMap<_, _> = anchor
        .holdings
        .iter()
        .map(|h| (h.instrument_id.clone(), h.units.to_string()))
        .collect();
    let symbols: Vec<_> = holdings.keys().cloned().collect();
    Ok(
        json!({"schema_version":1, "data_cutoff":anchor.equity.timestamp, "symbols":symbols,
        "cash":anchor.equity.cash, "equity":anchor.equity.equity, "holdings":holdings, "history":history,
        "settings":{"select_count":settings.select_count,"lookback_returns":settings.lookback_returns,
            "forecast_horizon_sessions":settings.forecast_horizon_sessions,"risk_aversion":settings.risk_aversion,
            "turnover_penalty":settings.turnover_penalty,"cash_reserve":config.cash_reserve}}),
    )
}

pub fn validate_response(
    config: &PortfolioConfig,
    request: &Value,
    hash: &str,
    response: &Value,
) -> Result<BTreeMap<String, Decimal>> {
    ensure!(
        response["schema_version"] == 1
            && response["method"] == "exact_enumeration"
            && response["status"] == "feasible",
        "Unsupported optimizer response"
    );
    ensure!(
        response["request_sha256"] == hash && response["data_cutoff"] == request["data_cutoff"],
        "Optimizer response hash/cutoff mismatch"
    );
    let weights: BTreeMap<String, String> =
        serde_json::from_value(response["target_weights"].clone())?;
    let expected: Vec<String> = serde_json::from_value(request["symbols"].clone())?;
    ensure!(
        weights.keys().eq(expected.iter()),
        "Response universe differs from request"
    );
    let weights: BTreeMap<_, _> = weights
        .into_iter()
        .map(|(id, w)| Ok((id, decimal(&w, "optimizer weight")?)))
        .collect::<Result<_>>()?;
    let selected: Vec<String> = serde_json::from_value(response["selected"].clone())?;
    let positive: Vec<_> = weights
        .iter()
        .filter(|(_, w)| **w > Decimal::ZERO)
        .map(|(s, _)| s.clone())
        .collect();
    let k = config.optimizer.as_ref().unwrap().select_count;
    let invested = Decimal::ONE - config.reserve()?;
    let target = invested / Decimal::from(k as u64);
    ensure!(
        selected == positive && selected.len() == k,
        "Response violates selection cardinality"
    );
    ensure!(
        weights.values().all(|w| *w >= Decimal::ZERO
            && (*w == Decimal::ZERO || (*w - target).abs() <= Decimal::new(1, 10)))
            && weights.values().copied().sum::<Decimal>() == invested,
        "Invalid equal weights/cash reserve"
    );
    Ok(weights)
}

fn worker(settings: &OptimizerConfig, request: &[u8], workspace: &Path) -> Result<Value> {
    let start = Instant::now();
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("qflab/optimizer_worker.py");
    let mut child = Command::new(&settings.python_path)
        .arg("-I")
        .arg(script)
        .current_dir(workspace)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Start Python optimizer")?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let out = thread::spawn(move || {
        let mut v = Vec::new();
        stdout.take(1_048_577).read_to_end(&mut v).map(|_| v)
    });
    let err = thread::spawn(move || {
        let mut v = Vec::new();
        stderr.take(65_537).read_to_end(&mut v).map(|_| v)
    });
    let mut stdin = child.stdin.take().unwrap();
    let input = request.to_vec();
    let input = thread::spawn(move || stdin.write_all(&input));
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() >= Duration::from_secs(settings.timeout_seconds) {
            timed_out = true;
            let _ = child.kill();
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(5));
    };
    let stdout = out
        .join()
        .map_err(|_| anyhow::anyhow!("Worker stdout reader failed"))??;
    let stderr = err
        .join()
        .map_err(|_| anyhow::anyhow!("Worker stderr reader failed"))??;
    let written = input
        .join()
        .map_err(|_| anyhow::anyhow!("Worker stdin writer failed"))?;
    ensure!(!timed_out, "Optimizer timed out; retained current holdings");
    ensure!(
        stdout.len() <= 1_048_576 && stderr.len() <= 65_536,
        "Optimizer output exceeded size limit"
    );
    ensure!(
        status.success(),
        "Optimizer failed: {}",
        String::from_utf8_lossy(&stderr)
    );
    written?;
    Ok(serde_json::from_slice(&stdout)?)
}

pub fn coordinate(
    config: &PortfolioConfig,
    snapshot: &PortfolioSnapshot,
    anchor: Anchor,
    state: &mut RunState,
    replay: Option<&[SolverRecord]>,
    workspace: &Path,
) -> Result<()> {
    let settings = config.optimizer.as_ref().unwrap();
    let request = build_request(config, snapshot, &anchor)?;
    let bytes = serde_json::to_vec(&request)?;
    let hash = sha256(&bytes);
    let cutoff: DateTime<Utc> = anchor.equity.timestamp.parse()?;
    let available_at = cutoff + chrono::Duration::seconds(settings.simulated_delay_seconds as i64);
    let mut record;
    if let Some(saved) = replay {
        record = saved
            .get(state.optimizer_records.len())
            .context("Missing replayed optimizer decision")?
            .clone();
        ensure!(
            record.request == request
                && record.request_sha256 == hash
                && record.available_at == available_at,
            "Replay optimizer input/state/timing mismatch"
        );
        ensure!(
            record.status == "feasible"
                || (record.status == "fallback" && record.fallback_reason.is_some()),
            "Unsupported replay status"
        );
        record.delivery_status = "waiting_for_available_at".into();
    } else {
        let start = Instant::now();
        let result = worker(settings, &bytes, workspace);
        record = SolverRecord {
            request: request.clone(),
            request_sha256: hash.clone(),
            available_at,
            response: None,
            status: "fallback".into(),
            fallback_reason: None,
            elapsed_wall_seconds: start.elapsed().as_secs_f64(),
            delivery_status: "retain_current_positions".into(),
        };
        match result {
            Ok(response) => {
                match validate_response(config, &request, &hash, &response) {
                    Ok(_) => {
                        record.status = "feasible".into();
                        record.delivery_status = "waiting_for_available_at".into();
                    }
                    Err(error) => record.fallback_reason = Some(error.to_string()),
                }
                record.response = Some(response);
            }
            Err(error) => record.fallback_reason = Some(format!("{error:#}")),
        }
    }
    if record.status == "feasible" {
        let weights = validate_response(
            config,
            &request,
            &hash,
            record
                .response
                .as_ref()
                .context("Missing replay response")?,
        )?;
        state.optimizer_ready = Some(ReadyDecision {
            anchor,
            weights,
            available_at,
            record_index: state.optimizer_records.len(),
        });
    } else {
        record.delivery_status = "retain_current_positions".into();
    }
    state.optimizer_records.push(record);
    Ok(())
}

pub fn replay(path: Option<&Path>) -> Result<Option<Vec<SolverRecord>>> {
    path.map(|p| Ok(serde_json::from_slice(&fs::read(p)?)?))
        .transpose()
}
