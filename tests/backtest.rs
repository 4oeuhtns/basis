//! These are financial/temporal checks against independent arithmetic, not mocked engines.
use std::{fs, path::Path, process::Command};

use qflab_backtest::{config::Config, data::Snapshot, runner::simulate};
use rust_decimal::Decimal;

fn config() -> Config {
    Config::load(Path::new("configs/demo.json")).unwrap()
}
fn demo() -> Snapshot {
    Snapshot::load(Path::new("data/demo_prices.csv")).unwrap()
}

fn fixture(closes: &[u32]) -> Snapshot {
    let dates = ["02", "03", "04", "05", "08", "09"];
    let mut text = "timestamp,open,high,low,close,volume\n".to_string();
    for (day, close) in dates.iter().zip(closes) {
        text += &format!(
            "2024-01-{day}T21:00:00Z,{close}.00,{close}.00,{close}.00,{close}.00,1000000\n"
        );
    }
    Snapshot::parse(text.into_bytes()).unwrap()
}

#[test]
fn accounting_and_timing_gates_on_the_real_nautilus_engine() {
    let original = demo();
    let base = config();
    let result = simulate(&base, &original).unwrap();
    assert!(result.state.errors.is_empty(), "{:?}", result.state.errors);
    assert_eq!(result.state.fills.len(), 1);
    let fill = &result.state.fills[0];
    assert_eq!(fill.price, Decimal::from(110)); // next CLOSE, not next open (103) or signal (100)
    assert_eq!(fill.units, Decimal::from(5));
    assert_eq!(fill.timestamp, "2024-01-03T21:00:00+00:00");
    assert_eq!(result.state.equity[0].units, Decimal::ZERO);
    assert_eq!(result.state.equity[1].cash, Decimal::from(450));
    assert_eq!(
        result.state.equity.last().unwrap().equity,
        Decimal::from(1075)
    );
    // Independent ledger: apply actual fills to starting money, then mark held units.
    for row in &result.state.equity {
        let mut cash = Decimal::from(1000);
        let mut units = Decimal::ZERO;
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
            units += signed;
        }
        assert_eq!(
            (row.cash, row.units, row.equity),
            (cash, units, cash + units * row.close)
        );
    }

    let mut no_trade = base.clone();
    no_trade.strategy = "no_trade".into();
    let idle = simulate(&no_trade, &original).unwrap();
    assert!(idle.state.fills.is_empty());
    assert!(
        idle.state
            .equity
            .iter()
            .all(|r| r.equity == Decimal::from(1000))
    );

    let flat = simulate(&base, &fixture(&[100; 6])).unwrap();
    assert!(flat.state.errors.is_empty());
    assert!(
        flat.state
            .equity
            .iter()
            .all(|r| r.equity == Decimal::from(1000))
    );

    let mut fee_config = base.clone();
    fee_config.fee_bps = "100".into(); // 1% of 5 units * $110 = $5.50
    let paid = simulate(&fee_config, &original).unwrap();
    assert!(paid.state.errors.is_empty());
    assert_eq!(paid.state.fills[0].commission, Decimal::new(550, 2));
    assert_eq!(
        paid.state.equity.last().unwrap().equity,
        Decimal::new(106950, 2)
    );

    let mut scheduled = base.clone();
    scheduled.strategy = "scheduled".into();
    let drift = simulate(&scheduled, &fixture(&[100, 100, 200, 200, 100, 100])).unwrap();
    assert!(drift.state.errors.is_empty(), "{:?}", drift.state.errors);
    let quantities: Vec<_> = drift
        .state
        .fills
        .iter()
        .map(|f| (f.side.as_str(), f.units))
        .collect();
    assert_eq!(
        quantities,
        vec![
            ("BUY", Decimal::from(5)),
            ("SELL", Decimal::from(2)),
            ("BUY", Decimal::from(3))
        ]
    );
    let last = drift.state.equity.last().unwrap();
    assert_eq!(
        (last.cash, last.units, last.equity),
        (Decimal::from(600), Decimal::from(6), Decimal::from(1200))
    );

    // Independent precomputed next-session strategy ledger.
    assert_eq!(
        drift
            .state
            .equity
            .iter()
            .map(|r| r.equity)
            .collect::<Vec<_>>(),
        [1000, 1000, 1500, 1500, 1200, 1200].map(Decimal::from)
    );

    let changed = fixture(&[100, 100, 200, 200, 900, 900]);
    let future_changed = simulate(&scheduled, &changed).unwrap();
    assert_eq!(
        serde_json::to_value(&drift.state.decisions[..2]).unwrap(),
        serde_json::to_value(&future_changed.state.decisions[..2]).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&drift.state.fills[..2]).unwrap(),
        serde_json::to_value(&future_changed.state.fills[..2]).unwrap()
    );

    let mut full = base.clone();
    full.target_weight = "1".into();
    let gap = simulate(&full, &original).unwrap();
    assert!(
        !gap.state.errors.is_empty(),
        "A gap above available cash must fail visibly"
    );
    assert!(gap.state.equity.iter().all(|r| r.cash >= Decimal::ZERO));
}

#[test]
fn malformed_snapshots_are_rejected_before_simulation() {
    let bytes = fs::read_to_string("data/demo_prices.csv").unwrap();
    for broken in [
        bytes.replace("2024-01-03", "2024-01-02"),    // duplicate
        bytes.replace("2024-01-03", "2024-01-10"),    // out of order/gap
        bytes.replace("T21:00:00Z", "T20:00:00Z"),    // ambiguous convention
        bytes.replace("T21:00:00Z", "T21:00:00"),     // missing timezone
        bytes.replace("102.00,99.00", "98.00,99.00"), // bad high
        bytes.replace("100.00", "100.001"),           // not representable cents
        bytes.replace("100.00", "NaN"),
        bytes.replace("1000000", "0"),
        bytes
            .lines()
            .filter(|line| !line.starts_with("2024-01-04"))
            .collect::<Vec<_>>()
            .join("\n"),
    ] {
        assert!(Snapshot::parse(broken.into_bytes()).is_err());
    }
}

#[test]
fn python_instruction_replays_with_identical_economics() {
    let path =
        std::env::temp_dir().join(format!("qflab-decision-test-{}.json", std::process::id()));
    let status = Command::new("python3")
        .args([
            "-m",
            "qflab.worker",
            "--data",
            "data/demo_prices.csv",
            "--cutoff",
            "2024-01-02T21:00:00Z",
            "--output",
        ])
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());
    let mut replay = config();
    replay.strategy = "replay".into();
    replay.decisions_path = Some(path.clone());
    let direct = simulate(&config(), &demo()).unwrap();
    let from_python = simulate(&replay, &demo()).unwrap();
    assert!(from_python.state.errors.is_empty());
    assert_eq!(
        serde_json::to_value(&direct.state.equity).unwrap(),
        serde_json::to_value(&from_python.state.equity).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&direct.state.fills).unwrap(),
        serde_json::to_value(&from_python.state.fills).unwrap()
    );
    let mut decision: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    decision[0]["input_snapshot_hash"] = serde_json::json!("wrong hash");
    fs::write(&path, serde_json::to_vec(&decision).unwrap()).unwrap();
    assert!(simulate(&replay, &demo()).is_err());
    fs::remove_file(path).unwrap();
}
