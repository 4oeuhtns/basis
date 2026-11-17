//! Wait for a complete set of closes; save a plan; execute it one session later.
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    rc::Rc,
};

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, NaiveDate, Utc};
use nautilus_common::actor::DataActor;
use nautilus_model::{
    accounts::Account,
    data::{Bar, BarType},
    enums::OrderSide,
    events::{OrderEventAny, OrderFilled, PositionEvent},
    identifiers::{ClientOrderId, InstrumentId, Venue},
    orders::Order,
    types::{Currency, Money, Quantity},
};
use nautilus_trading::{
    nautilus_strategy,
    strategy::{Strategy, StrategyCore},
};
use rust_decimal::{Decimal, prelude::ToPrimitive};

use super::state::{DecisionLog, EquityRow, FillRow, HoldingRow, RunState};
use crate::{config::PortfolioConfig, data::timestamp, research::Anchor};

#[derive(Debug)]
struct Intent {
    instrument_id: InstrumentId,
    signed_units: i64,
    reserve_cash: Decimal,
    log_index: usize,
}

#[derive(Debug)]
struct Active {
    intent: Intent,
    submitted_units: u64,
    client_order_id: ClientOrderId,
}

#[derive(Debug)]
pub struct PortfolioStrategy {
    core: StrategyCore,
    config: PortfolioConfig,
    bar_types: BTreeMap<InstrumentId, BarType>,
    marks: BTreeMap<InstrumentId, Bar>,
    seen: BTreeSet<InstrumentId>,
    frame_ts: Option<u64>,
    session_index: usize,
    pending: Option<VecDeque<Intent>>,
    queue: VecDeque<Intent>,
    active: Option<Active>,
    awaiting_position: bool,
    month_ends: BTreeSet<NaiveDate>,
    deferred_trigger: Option<&'static str>,
    state: Rc<RefCell<RunState>>,
}

impl PortfolioStrategy {
    pub fn new(config: PortfolioConfig, state: Rc<RefCell<RunState>>) -> Result<Self> {
        config.validate()?;
        let month_ends = if ["monthly", "optimized"].contains(&config.strategy.as_str()) {
            crate::data::historical::month_end_dates(config.snapshot_manifest.as_ref().unwrap())?
        } else {
            BTreeSet::new()
        };
        let bar_types = config
            .assets
            .iter()
            .map(|a| {
                let id = InstrumentId::from(format!("{}.SIM", a.symbol).as_str());
                Ok((id, format!("{id}-1-DAY-LAST-EXTERNAL").parse()?))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            core: StrategyCore::new(Default::default()),
            config,
            bar_types,
            marks: BTreeMap::new(),
            seen: BTreeSet::new(),
            frame_ts: None,
            session_index: 0,
            pending: None,
            queue: VecDeque::new(),
            active: None,
            awaiting_position: false,
            month_ends,
            deferred_trigger: None,
            state,
        })
    }

    fn cash(&self) -> Result<Decimal> {
        let account = self
            .cache()
            .account_for_venue(&Venue::from("SIM"))
            .context("Nautilus SIM account missing")?;
        Ok(account
            .balance_total(Some(Currency::USD()))
            .context("Nautilus USD balance missing")?
            .as_decimal())
    }

    fn snapshot(&self) -> Result<(EquityRow, Vec<HoldingRow>)> {
        let ts = self.frame_ts.context("No session yet")?;
        ensure!(
            self.seen.len() == self.bar_types.len(),
            "Incomplete portfolio close"
        );
        let cash = self.cash()?;
        ensure!(cash >= Decimal::ZERO, "Cash borrowing is forbidden");
        let mut holdings = Vec::new();
        for id in self.bar_types.keys() {
            let bar = self.marks.get(id).context("Missing valuation price")?;
            ensure!(bar.ts_init.as_u64() == ts, "Stale valuation price for {id}");
            let units = self.portfolio().net_position(id);
            ensure!(units >= Decimal::ZERO, "Short position is forbidden");
            holdings.push(HoldingRow {
                timestamp: timestamp(ts),
                instrument_id: id.to_string(),
                units,
                close: bar.close.as_decimal(),
                market_value: units * bar.close.as_decimal(),
            });
        }
        let invested_value = holdings.iter().map(|h| h.market_value).sum::<Decimal>();
        Ok((
            EquityRow {
                timestamp: timestamp(ts),
                cash,
                invested_value,
                equity: cash + invested_value,
            },
            holdings,
        ))
    }

    fn record_snapshot(&self) -> Result<()> {
        let (row, holdings) = self.snapshot()?;
        let mut state = self.state.borrow_mut();
        // A fill updates the same close's row after native position accounting finishes.
        if state
            .equity
            .last()
            .is_some_and(|r| r.timestamp == row.timestamp)
        {
            *state.equity.last_mut().unwrap() = row;
            let keep = state.holdings.len() - self.bar_types.len();
            state.holdings.truncate(keep);
        } else {
            state.equity.push(row);
        }
        state.holdings.extend(holdings);
        Ok(())
    }

    fn fail(&mut self, error: impl ToString) {
        self.deferred_trigger = None;
        let mut state = self.state.borrow_mut();
        state.errors.push(error.to_string());
        for intent in self.queue.drain(..) {
            state.decisions[intent.log_index].status = "aborted_after_failure".into();
        }
        if let Some(pending) = self.pending.take() {
            for intent in pending {
                state.decisions[intent.log_index].status = "aborted_after_failure".into();
            }
        }
    }

    fn plan(&mut self, trigger: &str) -> Result<()> {
        let (row, holdings) = self.snapshot()?;
        if self.config.strategy == "optimized" {
            let mut state = self.state.borrow_mut();
            ensure!(
                state.optimizer_requested.is_none() && state.optimizer_ready.is_none(),
                "Optimizer already has a pending request/answer"
            );
            state.optimizer_requested = Some(Anchor {
                equity: row,
                holdings,
                trigger: trigger.into(),
            });
            return Ok(());
        }
        let weight = self.config.weight()?;
        let weights = holdings
            .iter()
            .map(|h| (h.instrument_id.clone(), weight))
            .collect();
        self.plan_weights(row, holdings, &weights, trigger)
    }

    fn plan_weights(
        &mut self,
        row: EquityRow,
        holdings: Vec<HoldingRow>,
        weights: &BTreeMap<String, Decimal>,
        trigger: &str,
    ) -> Result<()> {
        // Round the reserve UP to cents. Whole-unit targets round DOWN.
        let reserve_cash =
            (row.equity * self.config.reserve()? * Decimal::from(100)).ceil() / Decimal::from(100);
        let mut intents = Vec::new();
        let mut state = self.state.borrow_mut();
        for holding in holdings {
            let weight = *weights
                .get(&holding.instrument_id)
                .context("Missing target weight")?;
            let target = (row.equity * weight / holding.close)
                .floor()
                .to_i64()
                .context("Target quantity outside supported range")?;
            let current = holding
                .units
                .to_i64()
                .context("Non-whole current position")?;
            let signed_units = target - current;
            let index = state.decisions.len();
            state.decisions.push(DecisionLog {
                data_cutoff: row.timestamp.clone(),
                trigger: trigger.into(),
                instrument_id: holding.instrument_id.clone(),
                target_weight: weight,
                signal_close: holding.close,
                decision_equity: row.equity,
                reserve_cash,
                target_units: target,
                requested_signed_units: signed_units,
                submitted_signed_units: 0,
                cash_before_order: None,
                status: if signed_units == 0 {
                    "no_order"
                } else {
                    "waiting_for_next_session"
                }
                .into(),
                execution_note: None,
                executed_at: None,
            });
            if signed_units != 0 {
                intents.push(Intent {
                    instrument_id: InstrumentId::from(holding.instrument_id.as_str()),
                    signed_units,
                    reserve_cash,
                    log_index: index,
                });
            }
        }
        // False sorts before true: all sells first, then buys; symbols break ties.
        intents.sort_by_key(|i| (i.signed_units > 0, i.instrument_id));
        if !intents.is_empty() {
            self.pending = Some(intents.into());
        }
        Ok(())
    }

    fn decision_trigger(&self, index: usize, ts: u64) -> Option<&'static str> {
        if index == 0 && self.config.strategy != "no_trade" {
            return Some("initial_allocation");
        }
        match self.config.strategy.as_str() {
            "scheduled" if index.is_multiple_of(self.config.rebalance_every.unwrap()) => {
                Some("session_interval")
            }
            "monthly" | "optimized"
                if self
                    .month_ends
                    .contains(&DateTime::<Utc>::from_timestamp_nanos(ts as i64).date_naive()) =>
            {
                Some("month_end")
            }
            _ => None,
        }
    }

    fn finish_execution(&mut self) -> Result<()> {
        if self.active.is_none()
            && !self.awaiting_position
            && self.queue.is_empty()
            && self.state.borrow().errors.is_empty()
            && let Some(trigger) = self.deferred_trigger.take()
        {
            // A saved plan executed on month-end. Use the fully updated native ledger.
            self.plan(trigger)?;
        }
        Ok(())
    }

    fn optimizer_plan_ready(&mut self, ts: u64) -> Result<bool> {
        let ready = { self.state.borrow_mut().optimizer_ready.take() };
        let Some(ready) = ready else {
            return Ok(false);
        };
        let now = DateTime::<Utc>::from_timestamp_nanos(ts as i64);
        let cutoff: DateTime<Utc> = ready.anchor.equity.timestamp.parse()?;
        if now < ready.available_at || now <= cutoff {
            self.state.borrow_mut().optimizer_ready = Some(ready);
            return Ok(true);
        }
        ensure!(
            self.cash()? == ready.anchor.equity.cash,
            "Stale optimizer answer: cash changed"
        );
        for holding in &ready.anchor.holdings {
            let id = InstrumentId::from(holding.instrument_id.as_str());
            ensure!(
                self.portfolio().net_position(&id) == holding.units,
                "Stale optimizer answer: holdings changed"
            );
        }
        self.state.borrow_mut().optimizer_records[ready.record_index].delivery_status =
            format!("accepted_at_{}", timestamp(ts));
        self.plan_weights(
            ready.anchor.equity,
            ready.anchor.holdings,
            &ready.weights,
            &ready.anchor.trigger,
        )?;
        Ok(false)
    }

    fn submit_next(&mut self) -> Result<()> {
        ensure!(
            self.active.is_none() && !self.awaiting_position,
            "Previous order has not settled"
        );
        while let Some(intent) = self.queue.pop_front() {
            let cash = self.cash()?;
            let price = self.marks[&intent.instrument_id].close.as_decimal();
            let rate = self.config.fee_bps()? / Decimal::from(10_000);
            let mut quantity = intent.signed_units.unsigned_abs();
            let is_buy = intent.signed_units > 0;
            if is_buy {
                // This is an execution funding check, not a recalculation of the signal.
                let spendable = (cash - intent.reserve_cash).max(Decimal::ZERO);
                let affordable = (spendable / (price * (Decimal::ONE + rate)))
                    .floor()
                    .to_u64()
                    .context("Affordable quantity outside supported range")?;
                quantity = quantity.min(affordable);
                while quantity > 0 {
                    let notional =
                        Money::from_decimal(Decimal::from(quantity) * price, Currency::USD())?
                            .as_decimal();
                    let fee = Money::from_decimal(notional * rate, Currency::USD())?.as_decimal();
                    if notional + fee <= spendable {
                        break;
                    }
                    quantity -= 1;
                }
            } else {
                ensure!(
                    Decimal::from(quantity) <= self.portfolio().net_position(&intent.instrument_id),
                    "Sell exceeds actual holding"
                );
            }
            {
                let mut state = self.state.borrow_mut();
                let log = &mut state.decisions[intent.log_index];
                log.cash_before_order = Some(cash);
                if quantity < intent.signed_units.unsigned_abs() {
                    log.execution_note = Some("cash_and_reserve_cap".into());
                }
                if quantity == 0 {
                    log.status = "skipped_insufficient_spendable_cash".into();
                    continue;
                }
                log.submitted_signed_units = if is_buy {
                    quantity as i64
                } else {
                    -(quantity as i64)
                };
                log.status = "submitted".into();
            }
            let order = self.order().market(
                intent.instrument_id,
                if is_buy {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                },
                Quantity::from(quantity),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            );
            self.active = Some(Active {
                intent,
                submitted_units: quantity,
                client_order_id: order.client_order_id(),
            });
            // Submit ONE order. The position callback starts the next, after proceeds settle.
            return self.submit_order(order, None, None, None);
        }
        Ok(())
    }

    fn accept_bar(&mut self, bar: &Bar) -> Result<()> {
        let id = bar.bar_type.instrument_id();
        ensure!(
            self.bar_types.get(&id) == Some(&bar.bar_type),
            "Unexpected bar type"
        );
        let ts = bar.ts_init.as_u64();
        if self.frame_ts != Some(ts) {
            if let Some(previous) = self.frame_ts {
                ensure!(ts > previous, "Out-of-order portfolio session");
                ensure!(
                    self.seen.len() == self.bar_types.len(),
                    "Missing fund bar at previous close"
                );
                ensure!(
                    self.active.is_none() && self.queue.is_empty() && !self.awaiting_position,
                    "Order remained unsettled into another session"
                );
            }
            self.frame_ts = Some(ts);
            self.seen.clear();
        }
        ensure!(self.seen.insert(id), "Duplicate fund bar");
        self.marks.insert(id, *bar);
        if self.seen.len() < self.bar_types.len() {
            return Ok(());
        }
        let index = self.session_index;
        self.session_index += 1;
        self.record_snapshot()?;
        if !self.state.borrow().errors.is_empty() {
            return Ok(());
        }
        let trigger = self.decision_trigger(index, ts);
        if self.config.strategy == "optimized" && self.optimizer_plan_ready(ts)? {
            if let Some(trigger) = trigger {
                self.state.borrow_mut().optimizer_events.push(serde_json::json!({"timestamp":timestamp(ts),"trigger":trigger,"status":"skipped_while_waiting_for_available_at"}));
            }
            return Ok(());
        }
        if let Some(plan) = self.pending.take() {
            self.deferred_trigger = trigger;
            self.queue = plan;
            self.submit_next()?;
            self.finish_execution()?;
        } else if let Some(trigger) = trigger {
            self.plan(trigger)?;
        }
        Ok(())
    }
}

nautilus_strategy!(PortfolioStrategy, {
    fn on_order_filled(&mut self, event: &OrderFilled) {
        self.state.borrow_mut().fills.push(FillRow {
            timestamp: timestamp(event.ts_event.as_u64()),
            instrument_id: event.instrument_id.to_string(),
            client_order_id: event.client_order_id.to_string(),
            side: event.order_side.to_string(),
            units: event.last_qty.as_decimal(),
            price: event.last_px.as_decimal(),
            commission: event.commission.map(|m| m.as_decimal()).unwrap_or_default(),
        });
        let Some(active) = self.active.take() else {
            self.fail("Unexpected fill without an active order");
            return;
        };
        let valid = event.client_order_id == active.client_order_id
            && event.instrument_id == active.intent.instrument_id
            && event.last_qty.as_decimal() == Decimal::from(active.submitted_units)
            && event.commission.is_some();
        {
            let mut state = self.state.borrow_mut();
            let log = &mut state.decisions[active.intent.log_index];
            log.status = if valid { "filled" } else { "fill_failure" }.into();
            log.executed_at = Some(timestamp(event.ts_event.as_u64()));
        }
        if valid {
            self.awaiting_position = true;
        } else {
            self.fail("Partial/mismatched fill or missing commission; remaining plan aborted");
            if let Err(error) = self.cancel_order(active.client_order_id, None, None) {
                self.fail(error);
            }
        }
    }

    fn on_position_event(&mut self, _event: PositionEvent) {
        self.awaiting_position = false;
        if let Err(error) = self.record_snapshot() {
            self.fail(error);
        }
        if self.state.borrow().errors.is_empty() {
            if let Err(error) = self.submit_next() {
                self.fail(error);
            }
            if let Err(error) = self.finish_execution() {
                self.fail(error);
            }
        }
    }

    fn on_order_event(&mut self, event: OrderEventAny) {
        if matches!(
            &event,
            OrderEventAny::Denied(_)
                | OrderEventAny::Rejected(_)
                | OrderEventAny::Canceled(_)
                | OrderEventAny::Expired(_)
        ) {
            if let Some(active) = self.active.take() {
                self.state.borrow_mut().decisions[active.intent.log_index].status =
                    "order_failure".into();
            }
            self.fail(format!("Order failed: {event:?}"));
        }
        match serde_json::to_value(event) {
            Ok(value) => self.state.borrow_mut().order_events.push(value),
            Err(error) => self.fail(error),
        }
    }
});

impl DataActor for PortfolioStrategy {
    fn on_start(&mut self) -> Result<()> {
        for bar_type in self.bar_types.values().copied().collect::<Vec<_>>() {
            self.subscribe_bars(bar_type, None, None);
        }
        Ok(())
    }

    fn on_bar(&mut self, bar: &Bar) -> Result<()> {
        if let Err(error) = self.accept_bar(bar) {
            self.fail(error);
        }
        Ok(())
    }

    fn on_stop(&mut self) -> Result<()> {
        let ready = { self.state.borrow_mut().optimizer_ready.take() };
        if let Some(ready) = ready {
            self.state.borrow_mut().optimizer_records[ready.record_index].delivery_status =
                "expired_end_of_data".into();
        }
        if let Some(plan) = self.pending.take() {
            for intent in plan {
                self.state.borrow_mut().decisions[intent.log_index].status =
                    "expired_end_of_data".into();
            }
        }
        if self.active.is_some()
            || !self.queue.is_empty()
            || self.awaiting_position
            || self.deferred_trigger.is_some()
        {
            self.fail("Unsettled final portfolio order");
        }
        if let Err(error) = self.record_snapshot() {
            self.fail(error);
        }
        Ok(())
    }
}
