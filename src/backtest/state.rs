//! Recorded account state, decisions, and fills shared by the runner and strategy.
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::research::{Anchor, ReadyDecision, SolverRecord};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EquityRow {
    pub timestamp: String,
    pub cash: Decimal,
    pub invested_value: Decimal,
    pub equity: Decimal,
}

#[derive(Debug, Clone, Serialize)]
pub struct HoldingRow {
    pub timestamp: String,
    pub instrument_id: String,
    pub units: Decimal,
    pub close: Decimal,
    pub market_value: Decimal,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionLog {
    pub data_cutoff: String,
    pub trigger: String,
    pub instrument_id: String,
    pub target_weight: Decimal,
    pub signal_close: Decimal,
    pub decision_equity: Decimal,
    pub reserve_cash: Decimal,
    pub target_units: i64,
    pub requested_signed_units: i64,
    pub submitted_signed_units: i64,
    pub cash_before_order: Option<Decimal>,
    pub status: String,
    pub execution_note: Option<String>,
    pub executed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FillRow {
    pub timestamp: String,
    pub instrument_id: String,
    pub client_order_id: String,
    pub side: String,
    pub units: Decimal,
    pub price: Decimal,
    pub commission: Decimal,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct RunState {
    pub equity: Vec<EquityRow>,
    pub holdings: Vec<HoldingRow>,
    pub decisions: Vec<DecisionLog>,
    pub fills: Vec<FillRow>,
    pub order_events: Vec<serde_json::Value>,
    pub errors: Vec<String>,
    pub optimizer_records: Vec<SolverRecord>,
    pub optimizer_events: Vec<serde_json::Value>,
    #[serde(skip)]
    pub optimizer_requested: Option<Anchor>,
    #[serde(skip)]
    pub optimizer_ready: Option<ReadyDecision>,
}
