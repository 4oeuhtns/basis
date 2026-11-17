//! Month-end chronology and a hand-calculated native account ledger, including an overlap.
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use qflab_backtest::{
    backtest::{self, Simulation},
    config::{Asset, PortfolioConfig},
    data::{PortfolioSnapshot, historical::month_end_dates},
};
use rust_decimal::Decimal;
use serde_json::{Value, json};

fn fixture(root: &Path, mode: Option<&str>) {
    let python = std::env::var_os("QFLAB_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".venv/bin/python"));
    let mut command = Command::new(python);
    command
        .arg("tests/historical_fixture.py")
        .arg(root)
        .env("PYTHONPATH", ".");
    if let Some(mode) = mode {
        command.arg(mode);
    }
    assert!(command.status().unwrap().success());
}

fn config(snapshot: &Path) -> PortfolioConfig {
    let mut config = PortfolioConfig::load(Path::new("configs/historical.json")).unwrap();
    config.starting_cash = "1000.00".into();
    config.snapshot_manifest = Some(snapshot.join("manifest.json"));
    config.assets = ["ALPHA", "BETA"]
        .into_iter()
        .map(|symbol| Asset {
            symbol: symbol.into(),
            data_path: snapshot.join(format!("{symbol}.parquet")),
        })
        .collect();
    config.output_root = snapshot.parent().unwrap().join("runs");
    config
}

fn run(config: &PortfolioConfig) -> Simulation {
    let snapshot = PortfolioSnapshot::load(config).unwrap();
    let result = backtest::simulate(config, &snapshot).unwrap();
    assert!(result.state.errors.is_empty(), "{:?}", result.state.errors);
    result
}

fn fills(result: &Simulation) -> Value {
    json!(
        result
            .state
            .fills
            .iter()
            .map(|f| json!({"timestamp": f.timestamp,
        "instrument": f.instrument_id, "side": f.side, "units": f.units,
        "price": f.price, "fee": f.commission}))
            .collect::<Vec<_>>()
    )
}

#[test]
fn monthly_uses_calendar_closes_and_accounts_for_orders_settling_at_month_end() {
    let root = std::env::temp_dir().join(format!("qflab-monthly-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let snapshot = root.join("base");
    fixture(&snapshot, Some("monthly"));
    let config = config(&snapshot);
    let dates = month_end_dates(config.snapshot_manifest.as_ref().unwrap()).unwrap();
    assert_eq!(
        dates.iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["2025-03-31"]
    );
    let result = run(&config);
    let d = Decimal::from;
    assert_eq!(
        result
            .state
            .equity
            .iter()
            .map(|r| r.cash)
            .collect::<Vec<_>>(),
        [1000, 100, 150, 150, 150].map(d)
    );
    assert!(result.state.equity.iter().all(|r| r.equity == d(1000)));
    assert_eq!(
        result
            .state
            .fills
            .iter()
            .map(|f| (
                f.instrument_id.as_str(),
                f.side.as_str(),
                f.units,
                f.price,
                f.timestamp.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                "ALPHA.SIM",
                "BUY",
                d(4),
                d(200),
                "2025-03-31T20:00:00+00:00"
            ),
            ("BETA.SIM", "BUY", d(2), d(50), "2025-03-31T20:00:00+00:00"),
            (
                "ALPHA.SIM",
                "SELL",
                d(2),
                d(200),
                "2025-04-01T20:00:00+00:00"
            ),
            ("BETA.SIM", "BUY", d(7), d(50), "2025-04-01T20:00:00+00:00"),
        ]
    );
    assert_eq!(result.state.decisions.len(), 4);
    assert!(
        result.state.decisions[..2]
            .iter()
            .all(|r| r.trigger == "initial_allocation")
    );
    assert!(
        result.state.decisions[2..]
            .iter()
            .all(|r| r.trigger == "month_end"
                && r.data_cutoff == "2025-03-31T20:00:00+00:00"
                && r.executed_at.as_deref() == Some("2025-04-01T20:00:00+00:00"))
    );
    assert_eq!(result.state.decisions[2].requested_signed_units, -2);
    assert_eq!(result.state.decisions[3].requested_signed_units, 7); // Actual 2 BETA held, not the unfilled 9 requested.

    // Later prices cannot change the month-end schedule, plan, or earlier execution.
    let changed = root.join("future");
    fixture(&changed, Some("monthly_future"));
    let future = run(&self::config(&changed));
    assert_eq!(
        serde_json::to_value(&result.state.decisions).unwrap(),
        serde_json::to_value(&future.state.decisions).unwrap()
    );
    assert_eq!(fills(&result), fills(&future));

    // Reverse the same-timestamp fund delivery order on the overlapping execution/decision date.
    let mut reversed = config.clone();
    reversed.assets.reverse();
    let reversed = run(&reversed);
    assert_eq!(fills(&result), fills(&reversed));
    assert_eq!(
        serde_json::to_value(&result.state.equity).unwrap(),
        serde_json::to_value(&reversed.state.equity).unwrap()
    );

    // Partial final months do not turn their last available bar into a rebalance signal.
    let partial = root.join("partial");
    fixture(&partial, None);
    let partial = run(&self::config(&partial));
    assert_eq!(partial.state.decisions.len(), 2);
    assert!(
        partial
            .state
            .decisions
            .iter()
            .all(|d| d.trigger == "initial_allocation")
    );

    // March ends but no April execution session is supplied: keep the actual holdings and expire intentions.
    let terminal = root.join("terminal");
    fixture(&terminal, Some("monthly_terminal"));
    let terminal = run(&self::config(&terminal));
    assert_eq!(terminal.state.fills.len(), 2);
    assert_eq!(terminal.state.decisions.len(), 4);
    assert!(
        terminal.state.decisions[2..]
            .iter()
            .all(|d| d.status == "expired_end_of_data")
    );
    assert_eq!(terminal.state.equity.last().unwrap().cash, d(100));

    // Full-year month ends come from the exchange schedule, including Friday/early-close ends.
    let year = root.join("year");
    fixture(&year, Some("year"));
    let year_config = self::config(&year);
    let dates = month_end_dates(year_config.snapshot_manifest.as_ref().unwrap()).unwrap();
    assert_eq!(
        dates.iter().map(ToString::to_string).collect::<Vec<_>>(),
        [
            "2025-01-31",
            "2025-02-28",
            "2025-03-31",
            "2025-04-30",
            "2025-05-30",
            "2025-06-30",
            "2025-07-31",
            "2025-08-29",
            "2025-09-30",
            "2025-10-31",
            "2025-11-28",
            "2025-12-31",
        ]
    );
    let year = run(&year_config);
    assert_eq!(year.state.equity.len(), 250);
    assert_eq!(year.state.decisions.len(), 26); // Initial allocation plus twelve month ends, two funds each.
    assert!(
        year.state
            .decisions
            .iter()
            .any(|d| d.data_cutoff == "2025-11-28T18:00:00+00:00")
    );
    assert_eq!(year.state.fills[2].timestamp, "2025-02-03T21:00:00+00:00"); // Friday close -> Monday close.

    let mut invalid = config.clone();
    invalid.rebalance_every = Some(21);
    assert!(invalid.validate().is_err());
    invalid.strategy = "scheduled".into();
    invalid.rebalance_every = None;
    assert!(invalid.validate().is_err());

    // Both controlled runs reuse the same frozen source and produce an inspectable daily comparison.
    let recipe = root.join("config.json");
    fs::write(&recipe, serde_json::to_vec(&config).unwrap()).unwrap();
    let comparison = qflab_backtest::backtest::comparison::run(&recipe).unwrap();
    let report: Value =
        serde_json::from_slice(&fs::read(comparison.join("comparison.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "completed");
    assert_eq!(report["session_count"], 5);
    assert_eq!(report["monthly"]["summary"]["fill_count"], 4);
    assert_eq!(report["buy_and_hold"]["summary"]["fill_count"], 2);
    assert_eq!(
        report["ending_equity_difference"]
            .as_str()
            .unwrap()
            .parse::<Decimal>()
            .unwrap(),
        Decimal::ZERO
    );
    assert_eq!(
        csv::Reader::from_path(comparison.join("comparison.csv"))
            .unwrap()
            .records()
            .count(),
        5
    );
    let monthly = comparison.join(report["monthly"]["run_directory"].as_str().unwrap());
    let replay = backtest::run(&monthly.join("config.json")).unwrap();
    for name in ["equity.csv", "decision_log.json", "summary.json"] {
        assert_eq!(
            fs::read(monthly.join(name)).unwrap(),
            fs::read(replay.join(name)).unwrap()
        );
    }
    fs::remove_dir_all(root).unwrap();
}
