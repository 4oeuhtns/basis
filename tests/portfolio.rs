//! Independent portfolio arithmetic and event-order checks against the real Nautilus engine.
use std::{collections::BTreeMap, fs, path::Path};

use qflab_backtest::{
    data::Snapshot,
    portfolio::{
        PortfolioSnapshot,
        config::PortfolioConfig,
        runner::{Simulation, simulate},
    },
};
use rust_decimal::Decimal;
use serde_json::{Value, json};

fn config() -> PortfolioConfig {
    PortfolioConfig::load(Path::new("configs/portfolio.json")).unwrap()
}

fn fixture(alpha: &[u32], beta: &[u32], volume: u64) -> PortfolioSnapshot {
    let dates = ["02", "03", "04", "05", "08", "09"];
    let mut assets = BTreeMap::new();
    for (symbol, prices) in [("ALPHA", alpha), ("BETA", beta)] {
        let mut text = "timestamp,open,high,low,close,volume\n".to_string();
        for (day, price) in dates.iter().zip(prices) {
            text += &format!(
                "2024-01-{day}T21:00:00Z,{price}.00,{price}.00,{price}.00,{price}.00,{volume}\n"
            );
        }
        assets.insert(symbol.into(), Snapshot::parse(text.into_bytes()).unwrap());
    }
    PortfolioSnapshot { assets }
}

fn economics(simulation: &Simulation) -> Value {
    json!({
        "equity": simulation.state.equity,
        "holdings": simulation.state.holdings,
        "decisions": simulation.state.decisions,
        "fills": simulation.state.fills.iter().map(|f| json!({
            "timestamp": f.timestamp, "instrument": f.instrument_id, "side": f.side,
            "units": f.units, "price": f.price, "commission": f.commission,
        })).collect::<Vec<_>>()
    })
}

fn check_ledger(result: &Simulation) {
    for row in &result.state.equity {
        let mut cash = Decimal::from(1000);
        let mut units = BTreeMap::<String, Decimal>::new();
        for fill in result
            .state
            .fills
            .iter()
            .filter(|f| f.timestamp <= row.timestamp)
        {
            let signed = if fill.side == "BUY" {
                fill.units
            } else {
                -fill.units
            };
            cash -= signed * fill.price + fill.commission;
            *units.entry(fill.instrument_id.clone()).or_default() += signed;
        }
        let mut market_value = Decimal::ZERO;
        for holding in result
            .state
            .holdings
            .iter()
            .filter(|h| h.timestamp == row.timestamp)
        {
            let expected_units = *units.get(&holding.instrument_id).unwrap_or(&Decimal::ZERO);
            assert_eq!(holding.units, expected_units);
            assert_eq!(holding.market_value, expected_units * holding.close);
            market_value += expected_units * holding.close;
        }
        assert_eq!(
            (row.cash, row.invested_value, row.equity),
            (cash, market_value, cash + market_value)
        );
        assert!(cash >= Decimal::ZERO);
    }
}

#[test]
fn shared_cash_rebalances_at_synchronized_next_closes() {
    let config = config();
    let snapshot = PortfolioSnapshot::load(&config).unwrap();
    let result = simulate(&config, &snapshot).unwrap();
    assert!(result.state.errors.is_empty(), "{:?}", result.state.errors);
    assert_eq!(
        result
            .state
            .equity
            .iter()
            .map(|r| r.equity)
            .collect::<Vec<_>>(),
        [1000, 1000, 1400, 1400, 1700, 1700].map(Decimal::from)
    );
    assert_eq!(
        result
            .state
            .equity
            .iter()
            .map(|r| r.cash)
            .collect::<Vec<_>>(),
        [1000, 150, 150, 200, 200, 300].map(Decimal::from)
    );
    let fills: Vec<_> = result
        .state
        .fills
        .iter()
        .map(|f| {
            (
                f.instrument_id.as_str(),
                f.side.as_str(),
                f.units,
                f.price,
                f.timestamp.as_str(),
            )
        })
        .collect();
    let d = Decimal::from;
    assert_eq!(
        fills,
        vec![
            (
                "ALPHA.SIM",
                "BUY",
                d(4),
                d(100),
                "2024-01-03T21:00:00+00:00"
            ),
            ("BETA.SIM", "BUY", d(9), d(50), "2024-01-03T21:00:00+00:00"),
            (
                "ALPHA.SIM",
                "SELL",
                d(1),
                d(200),
                "2024-01-05T21:00:00+00:00"
            ),
            ("BETA.SIM", "BUY", d(3), d(50), "2024-01-05T21:00:00+00:00"),
            (
                "BETA.SIM",
                "SELL",
                d(5),
                d(100),
                "2024-01-09T21:00:00+00:00"
            ),
            (
                "ALPHA.SIM",
                "BUY",
                d(4),
                d(100),
                "2024-01-09T21:00:00+00:00"
            ),
        ]
    );
    // A buy that costs $400 sees the $700 AFTER BETA's sale, rather than the $200 before it.
    let final_buy = &result.state.decisions[4];
    assert_eq!(final_buy.cash_before_order, Some(d(700)));
    assert_eq!(
        (final_buy.target_units, final_buy.requested_signed_units),
        (7, 4)
    );
    assert!(
        result
            .state
            .decisions
            .iter()
            .all(|d| d.executed_at.as_ref().unwrap() > &d.data_cutoff)
    );
    check_ledger(&result);

    // Reverse same-timestamp arrival order. A stale-price decision would change the outcome.
    let mut reversed = config.clone();
    reversed.assets.reverse();
    let reverse_result = simulate(&reversed, &snapshot).unwrap();
    assert!(reverse_result.state.errors.is_empty());
    assert_eq!(economics(&result), economics(&reverse_result));

    // Future observations cannot affect the earlier portfolio plans or their execution.
    let changed = fixture(
        &[100, 100, 200, 200, 900, 900],
        &[50, 50, 50, 50, 20, 20],
        1_000_000,
    );
    let changed_result = simulate(&config, &changed).unwrap();
    assert!(changed_result.state.errors.is_empty());
    assert_eq!(
        serde_json::to_value(&result.state.decisions[..4]).unwrap(),
        serde_json::to_value(&changed_result.state.decisions[..4]).unwrap()
    );
    let a = economics(&result);
    let b = economics(&changed_result);
    assert_eq!(
        a["fills"].as_array().unwrap()[..4],
        b["fills"].as_array().unwrap()[..4]
    );
}

#[test]
fn fees_rounding_and_gap_funding_use_one_actual_cash_account() {
    let mut config = config();
    let snapshot = PortfolioSnapshot::load(&config).unwrap();
    config.fee_bps = "100".into();
    let paid = simulate(&config, &snapshot).unwrap();
    assert!(paid.state.errors.is_empty(), "{:?}", paid.state.errors);
    assert_eq!(
        paid.state
            .fills
            .iter()
            .map(|f| f.commission)
            .sum::<Decimal>(),
        Decimal::from(21)
    );
    let last = paid.state.equity.last().unwrap();
    assert_eq!(
        (last.cash, last.equity),
        (Decimal::from(279), Decimal::from(1679))
    );
    check_ledger(&paid);

    config.fee_bps = "0".into();
    let gap = fixture(&[100, 300], &[50, 100], 1_000_000);
    let capped = simulate(&config, &gap).unwrap();
    assert!(capped.state.errors.is_empty(), "{:?}", capped.state.errors);
    assert_eq!(capped.state.fills.len(), 1);
    assert_eq!(capped.state.fills[0].units, Decimal::from(3));
    assert_eq!(capped.state.equity[1].cash, Decimal::from(100));
    assert_eq!(capped.state.decisions[0].requested_signed_units, 4);
    assert_eq!(capped.state.decisions[0].submitted_signed_units, 3);
    assert_eq!(
        capped.state.decisions[0].execution_note.as_deref(),
        Some("cash_and_reserve_cap")
    );
    assert_eq!(
        capped.state.decisions[1].status,
        "skipped_insufficient_spendable_cash"
    );
    check_ledger(&capped);

    config.fee_bps = "100".into();
    let capped_fees = simulate(&config, &gap).unwrap();
    assert!(capped_fees.state.errors.is_empty());
    assert!(capped_fees.state.equity[1].cash >= Decimal::from(100));
    check_ledger(&capped_fees);

    // Cheaper execution prices do not increase the quantities saved at the signal close.
    config.fee_bps = "0".into();
    let cheaper = simulate(&config, &fixture(&[100, 50], &[50, 25], 1_000_000)).unwrap();
    assert!(cheaper.state.errors.is_empty());
    assert_eq!(
        cheaper
            .state
            .fills
            .iter()
            .map(|f| f.units)
            .collect::<Vec<_>>(),
        [4, 9].map(Decimal::from)
    );
    assert_eq!(cheaper.state.equity[1].cash, Decimal::from(575));
    check_ledger(&cheaper);
}

#[test]
fn idle_flat_reserve_and_end_of_data_behave_explicitly() {
    let mut config = config();
    let flat = fixture(&[100; 6], &[50; 6], 1_000_000);
    let result = simulate(&config, &flat).unwrap();
    assert!(result.state.errors.is_empty());
    assert_eq!(result.state.fills.len(), 2);
    assert!(
        result
            .state
            .equity
            .iter()
            .all(|r| r.equity == Decimal::from(1000))
    );
    assert_eq!(result.state.equity[1].cash, Decimal::from(150)); // reserve + whole-unit rounding

    config.strategy = "no_trade".into();
    let idle = simulate(&config, &PortfolioSnapshot::load(&config).unwrap()).unwrap();
    assert!(idle.state.errors.is_empty());
    assert!(idle.state.fills.is_empty());
    assert!(
        idle.state
            .equity
            .iter()
            .all(|r| r.cash == Decimal::from(1000))
    );

    config.strategy = "buy_and_hold".into();
    let held = simulate(&config, &PortfolioSnapshot::load(&config).unwrap()).unwrap();
    assert!(held.state.errors.is_empty());
    assert_eq!(held.state.fills.len(), 2);
    assert_eq!(
        held.state.equity.last().unwrap().equity,
        Decimal::from(1450)
    );

    config.cash_reserve = "1".into();
    let cash_only = simulate(&config, &flat).unwrap();
    assert!(cash_only.state.errors.is_empty());
    assert!(cash_only.state.fills.is_empty());

    config.cash_reserve = "0.10".into();
    config.strategy = "scheduled".into();
    let short = fixture(
        &[100, 100, 200, 200, 100],
        &[50, 50, 50, 50, 100],
        1_000_000,
    );
    let ended = simulate(&config, &short).unwrap();
    assert!(ended.state.errors.is_empty());
    assert_eq!(ended.state.fills.len(), 4);
    assert!(
        ended.state.decisions[4..]
            .iter()
            .all(|d| d.status == "expired_end_of_data")
    );
    assert_eq!(
        ended.state.equity.last().unwrap().equity,
        Decimal::from(1700)
    );
    check_ledger(&ended);
}

#[test]
fn missing_asset_sessions_and_invalid_recipes_fail_before_simulation() {
    let mut config = config();
    let mismatch = fixture(&[100; 6], &[50; 5], 1_000_000);
    assert!(
        simulate(&config, &mismatch)
            .unwrap_err()
            .to_string()
            .contains("same sessions")
    );
    config.assets[1].symbol = config.assets[0].symbol.clone();
    assert!(config.validate().is_err());
    config = self::config();
    config.cash_reserve = "-0.1".into();
    assert!(config.validate().is_err());
    config.cash_reserve = "1.1".into();
    assert!(config.validate().is_err());
}

#[test]
fn insufficient_liquidity_aborts_the_remaining_plan() {
    let config = config();
    let tiny_book = fixture(&[100, 100], &[50, 50], 4);
    let result = simulate(&config, &tiny_book).unwrap();
    assert!(
        !result.state.errors.is_empty(),
        "A partially filled order must fail visibly"
    );
    assert!(!result.state.fills.is_empty());
    assert!(result.state.fills[0].units < Decimal::from(4));
    assert_eq!(result.state.decisions[0].status, "fill_failure");
    assert!(
        result
            .state
            .fills
            .iter()
            .all(|f| f.instrument_id == "ALPHA.SIM")
    );
    assert_eq!(result.state.decisions[1].status, "aborted_after_failure");
    check_ledger(&result);
}

#[test]
fn saved_inputs_replay_and_failed_runs_keep_their_evidence() {
    let root =
        std::env::temp_dir().join(format!("qflab-portfolio-evidence-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let mut config = config();
    config.output_root = root.join("runs");
    let recipe = root.join("config.json");
    fs::write(&recipe, serde_json::to_vec(&config).unwrap()).unwrap();
    let output = qflab_backtest::portfolio::runner::run(&recipe).unwrap();
    let replay = qflab_backtest::portfolio::runner::run(&output.join("config.json")).unwrap();
    for file in [
        "equity.csv",
        "holdings.csv",
        "summary.json",
        "decision_log.json",
    ] {
        assert_eq!(
            fs::read(output.join(file)).unwrap(),
            fs::read(replay.join(file)).unwrap()
        );
    }
    let manifest: Value =
        serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["status"], "completed");
    assert_eq!(manifest["snapshot_sha256"].as_object().unwrap().len(), 2);

    let tiny = fixture(&[100, 100], &[50, 50], 4);
    for asset in &mut config.assets {
        asset.data_path = root.join(format!("tiny_{}.csv", asset.symbol));
        fs::write(&asset.data_path, &tiny.assets[&asset.symbol].bytes).unwrap();
    }
    config.output_root = root.join("failed_runs");
    fs::write(&recipe, serde_json::to_vec(&config).unwrap()).unwrap();
    assert!(qflab_backtest::portfolio::runner::run(&recipe).is_err());
    let failed = fs::read_dir(&config.output_root)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let manifest: Value =
        serde_json::from_slice(&fs::read(failed.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["status"], "failed");
    let summary: Value =
        serde_json::from_slice(&fs::read(failed.join("summary.json")).unwrap()).unwrap();
    assert!(!summary["errors"].as_array().unwrap().is_empty());
    assert!(failed.join("orders.json").exists());
    assert!(failed.join("account_events.json").exists());
    fs::remove_dir_all(root).unwrap();
}
