//! A deliberately small strategy: decide today, submit the saved order on the next bar.
use std::{cell::RefCell, rc::Rc};

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Utc};
use nautilus_common::actor::DataActor;
use nautilus_model::{
    accounts::Account,
    data::{Bar, BarType},
    enums::OrderSide,
    events::{OrderEventAny, OrderFilled, PositionEvent},
    identifiers::{InstrumentId, Venue},
    types::{Currency, Quantity},
};
use nautilus_trading::{
    nautilus_strategy,
    strategy::{Strategy, StrategyCore},
};
use rust_decimal::{Decimal, prelude::ToPrimitive};
use serde::Serialize;

use crate::{
    config::{Config, decimal},
    decision::PortfolioDecision,
};

#[derive(Debug, Clone, Serialize)]
pub struct EquityRow {
    pub timestamp: String,
    pub cash: Decimal,
    pub units: Decimal,
    pub close: Decimal,
    pub equity: Decimal,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionLog {
    pub data_cutoff: String,
    pub observed_at: String,
    pub target_weight: Decimal,
    pub signal_close: Decimal,
    pub signed_units: i64,
    pub status: String,
    pub executed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FillRow {
    pub timestamp: String,
    pub client_order_id: String,
    pub side: String,
    pub units: Decimal,
    pub price: Decimal,
    pub commission: Decimal,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct RunState {
    pub equity: Vec<EquityRow>,
    pub decisions: Vec<DecisionLog>,
    pub fills: Vec<FillRow>,
    pub order_events: Vec<serde_json::Value>,
    pub errors: Vec<String>,
}

#[derive(Debug)]
struct Intent {
    signed_units: i64,
    decision_index: usize,
}

#[derive(Debug)]
pub struct BaselineStrategy {
    core: StrategyCore,
    config: Config,
    instrument_id: InstrumentId,
    bar_type: BarType,
    pub state: Rc<RefCell<RunState>>,
    pending: Option<Intent>,
    active: Option<Intent>,
    last_bar: Option<Bar>,
    session_index: usize,
    saved_decisions: Vec<PortfolioDecision>,
    saved_consumed: bool,
}

pub fn timestamp(ns: u64) -> String {
    DateTime::<Utc>::from_timestamp_nanos(ns as i64).to_rfc3339()
}

impl BaselineStrategy {
    pub fn new(
        config: Config,
        instrument_id: InstrumentId,
        bar_type: BarType,
        saved_decisions: Vec<PortfolioDecision>,
        state: Rc<RefCell<RunState>>,
    ) -> Self {
        Self {
            core: StrategyCore::new(Default::default()),
            config,
            instrument_id,
            bar_type,
            state,
            pending: None,
            active: None,
            last_bar: None,
            session_index: 0,
            saved_decisions,
            saved_consumed: false,
        }
    }

    fn snapshot(&self) -> Result<EquityRow> {
        let bar = self.last_bar.as_ref().context("No current bar")?;
        let account = self
            .cache()
            .account_for_venue(&Venue::from("SIM"))
            .context("Nautilus SIM account missing")?;
        let cash = account
            .balance_total(Some(Currency::USD()))
            .context("Nautilus USD balance missing")?
            .as_decimal();
        let units = self.portfolio().net_position(&self.instrument_id);
        let close = bar.close.as_decimal();
        ensure!(
            cash >= Decimal::ZERO && units >= Decimal::ZERO,
            "Long-only cash/position invariant violated"
        );
        Ok(EquityRow {
            timestamp: timestamp(bar.ts_init.as_u64()),
            cash,
            units,
            close,
            equity: cash + units * close,
        })
    }

    fn record_snapshot(&self) -> Result<()> {
        let row = self.snapshot()?;
        let mut state = self.state.borrow_mut();
        if state
            .equity
            .last()
            .is_some_and(|r| r.timestamp == row.timestamp)
        {
            *state.equity.last_mut().unwrap() = row;
        } else {
            state.equity.push(row);
        }
        Ok(())
    }

    fn plan(&mut self, weight: Decimal, cutoff: String) -> Result<()> {
        let row = self.snapshot()?;
        // Whole units only. Reserve expected fees using today's price.
        // Tomorrow's gap may still make this order unaffordable: that is a recorded failure.
        let target = (row.equity * weight / row.close).floor();
        let affordable = (row.cash
            / (row.close * (Decimal::ONE + self.config.fee_bps()? / Decimal::from(10_000))))
        .floor();
        let difference = target - row.units;
        let quantity = if difference > Decimal::ZERO {
            difference.min(affordable)
        } else {
            difference.max(-row.units)
        };
        let signed_units = quantity
            .to_i64()
            .context("Order quantity outside supported range")?;
        let mut state = self.state.borrow_mut();
        let index = state.decisions.len();
        state.decisions.push(DecisionLog {
            data_cutoff: cutoff,
            observed_at: row.timestamp,
            target_weight: weight,
            signal_close: row.close,
            signed_units,
            status: if signed_units == 0 {
                "no_order"
            } else {
                "waiting_for_next_session"
            }
            .into(),
            executed_at: None,
        });
        if signed_units != 0 {
            self.pending = Some(Intent {
                signed_units,
                decision_index: index,
            });
        }
        Ok(())
    }

    fn execute(&mut self, intent: Intent) -> Result<()> {
        let side = if intent.signed_units > 0 {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        let order = self.order().market(
            self.instrument_id,
            side,
            Quantity::from(intent.signed_units.unsigned_abs()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        self.state.borrow_mut().decisions[intent.decision_index].status = "submitted".into();
        self.active = Some(intent);
        self.submit_order(order, None, None, None)
    }
}

nautilus_strategy!(BaselineStrategy, {
    fn on_order_filled(&mut self, event: &OrderFilled) {
        let mut state = self.state.borrow_mut();
        if event.commission.is_none() {
            state.errors.push("Fill omitted commission".into());
        }
        state.fills.push(FillRow {
            timestamp: timestamp(event.ts_event.as_u64()),
            client_order_id: event.client_order_id.to_string(),
            side: event.order_side.to_string(),
            units: event.last_qty.as_decimal(),
            price: event.last_px.as_decimal(),
            commission: event.commission.map(|m| m.as_decimal()).unwrap_or_default(),
        });
        if let Some(intent) = self.active.take() {
            let log = &mut state.decisions[intent.decision_index];
            log.executed_at = Some(timestamp(event.ts_event.as_u64()));
            if event.last_qty.as_decimal() == Decimal::from(intent.signed_units.unsigned_abs()) {
                log.status = "filled".into();
            } else {
                log.status = "partial_fill_failure".into();
                state
                    .errors
                    .push("Partial fill: this first adapter requires full fills".into());
            }
        }
        // Position accounting completes after this fill callback; snapshot on_position_event.
    }

    fn on_order_event(&mut self, event: OrderEventAny) {
        let failure = matches!(
            &event,
            OrderEventAny::Denied(_)
                | OrderEventAny::Rejected(_)
                | OrderEventAny::Canceled(_)
                | OrderEventAny::Expired(_)
        );
        let mut state = self.state.borrow_mut();
        if failure {
            state.errors.push(format!("Order failed: {event:?}"));
            if let Some(intent) = self.active.take() {
                state.decisions[intent.decision_index].status = "order_failure".into();
            }
        }
        match serde_json::to_value(event) {
            Ok(value) => state.order_events.push(value),
            Err(error) => state.errors.push(error.to_string()),
        }
    }

    fn on_position_event(&mut self, _event: PositionEvent) {
        if let Err(error) = self.record_snapshot() {
            self.state.borrow_mut().errors.push(error.to_string());
        }
    }
});

impl DataActor for BaselineStrategy {
    fn on_start(&mut self) -> Result<()> {
        self.subscribe_bars(self.bar_type, None, None);
        Ok(())
    }

    fn on_bar(&mut self, bar: &Bar) -> Result<()> {
        self.last_bar = Some(*bar);
        let index = self.session_index;
        self.session_index += 1;
        if !self.state.borrow().errors.is_empty() {
            return self.record_snapshot();
        }
        if self.active.is_some() {
            self.state
                .borrow_mut()
                .errors
                .push("Order remained pending into another session".into());
            return self.record_snapshot();
        }
        if let Some(intent) = self.pending.take() {
            // This quantity was fixed on the PREVIOUS bar. Today's OHLC cannot change it.
            self.execute(intent)?;
            return self.record_snapshot();
        }
        self.record_snapshot()?;
        let now = DateTime::<Utc>::from_timestamp_nanos(bar.ts_init.as_u64() as i64);
        match self.config.strategy.as_str() {
            "buy_and_hold" if index == 0 => self.plan(self.config.weight()?, now.to_rfc3339())?,
            "scheduled" if index.is_multiple_of(self.config.rebalance_every) => {
                self.plan(self.config.weight()?, now.to_rfc3339())?
            }
            "replay" if !self.saved_consumed => {
                let d = &self.saved_decisions[0];
                if d.available_at <= now && d.data_cutoff <= now {
                    let weight =
                        decimal(&d.target_weights[&self.instrument_id.to_string()], "weight")?;
                    let cutoff = d.data_cutoff.to_rfc3339();
                    self.saved_consumed = true;
                    self.plan(weight, cutoff)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn on_stop(&mut self) -> Result<()> {
        // Positions remain held and are valued at the final close; no surprise liquidation.
        if let Some(intent) = self.pending.take() {
            self.state.borrow_mut().decisions[intent.decision_index].status =
                "expired_end_of_data".into();
        }
        if self.active.is_some() {
            self.state
                .borrow_mut()
                .errors
                .push("Unsettled final order".into());
        }
        self.record_snapshot()
    }
}
