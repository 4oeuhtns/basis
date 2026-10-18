//! Shared Nautilus plumbing. Both lessons use the same native cash account.
use anyhow::{Context, Result};
use nautilus_backtest::{
    config::{BacktestEngineConfig, SimulatedVenueConfig},
    engine::BacktestEngine,
};
use nautilus_common::logging::logger::LoggerConfig;
use nautilus_execution::models::fee::{FeeModelAny, MakerTakerFeeModel};
use nautilus_model::{
    enums::{AccountType, BookType, OmsType},
    identifiers::{InstrumentId, Symbol, Venue},
    instruments::{Equity, InstrumentAny},
    types::{Currency, Money, Price, Quantity},
};
use rust_decimal::Decimal;
use serde_json::Value;

pub fn new_engine(cash: Decimal, fee_bps: Decimal) -> Result<BacktestEngine> {
    let mut engine = BacktestEngine::new(
        BacktestEngineConfig::builder()
            .logging(LoggerConfig {
                bypass_logging: true,
                ..Default::default()
            })
            .run_analysis(false)
            .build(),
    )?;
    let rate = fee_bps / Decimal::from(10_000);
    engine.add_venue(
        SimulatedVenueConfig::builder()
            .venue(Venue::from("SIM"))
            .oms_type(OmsType::Netting)
            .account_type(AccountType::Cash)
            .book_type(BookType::L1_MBP)
            .base_currency(Currency::USD())
            .starting_balances(vec![Money::from_decimal(cash, Currency::USD())?])
            .fee_model(FeeModelAny::MakerTaker(MakerTakerFeeModel::new(rate, rate)).into())
            .allow_cash_borrowing(false)
            .bar_execution(true)
            .trade_execution(false)
            .bar_adaptive_high_low_ordering(false)
            .use_message_queue(true)
            .build()?,
    )?;
    Ok(engine)
}

pub fn equity(symbol: &str) -> Result<InstrumentAny> {
    equity_with_precision(symbol, 2)
}

pub fn equity_with_precision(symbol: &str, precision: u8) -> Result<InstrumentAny> {
    Ok(InstrumentAny::Equity(
        Equity::builder()
            .instrument_id(InstrumentId::from(format!("{symbol}.SIM").as_str()))
            .raw_symbol(Symbol::from(symbol))
            .currency(Currency::USD())
            .price_precision(precision)
            .price_increment(Price::from_decimal_dp(
                Decimal::new(1, u32::from(precision)),
                precision,
            )?)
            .lot_size(Quantity::from(1u64))
            .ts_event(0.into())
            .ts_init(0.into())
            .build()?,
    ))
}

#[derive(Debug)]
pub struct Reports {
    pub engine_result: Value,
    pub orders: Vec<Value>,
    pub positions: Vec<Value>,
    pub account_events: Value,
}

pub fn reports(engine: &BacktestEngine) -> Result<Reports> {
    let cache_rc = engine.kernel().cache();
    let cache = cache_rc.borrow();
    let mut orders: Vec<Value> = cache
        .orders_refs(None, None, None, None, None)
        .iter()
        .map(|o| serde_json::to_value(&**o))
        .collect::<std::result::Result<_, _>>()?;
    orders.sort_by_key(|o| o.to_string());
    let mut positions: Vec<Value> = cache
        .positions_refs(None, None, None, None, None)
        .iter()
        .map(|p| serde_json::to_value(&**p))
        .collect::<std::result::Result<_, _>>()?;
    positions.sort_by_key(|p| p.to_string());
    let account = cache
        .account_for_venue_owned(&Venue::from("SIM"))
        .context("Final account missing")?;
    Ok(Reports {
        engine_result: serde_json::to_value(engine.get_result())?,
        orders,
        positions,
        account_events: serde_json::to_value(account.events())?,
    })
}
