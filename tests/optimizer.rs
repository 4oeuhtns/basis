//! The optimizer must change real orders while preserving causal inputs, native accounting, and replay.
use chrono::DateTime;
use nautilus_model::types::{Currency, Money};
use qflab_backtest::{
    backtest::{self, Simulation},
    config::{Asset, PortfolioConfig},
    data::PortfolioSnapshot,
    research::validate_response,
};
use rust_decimal::Decimal;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn fixture(path: &Path, mode: &str) {
    let python = std::env::var_os("QFLAB_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".venv/bin/python"));
    assert!(
        Command::new(python)
            .arg("tests/historical_fixture.py")
            .arg(path)
            .arg(mode)
            .env("PYTHONPATH", ".")
            .status()
            .unwrap()
            .success()
    );
}

fn config(root: &Path) -> PortfolioConfig {
    let mut c = PortfolioConfig::load(Path::new("configs/optimized.json")).unwrap();
    c.assets = (0..8)
        .map(|i| Asset {
            symbol: format!("A{i}"),
            data_path: root.join(format!("A{i}.parquet")),
        })
        .collect();
    c.snapshot_manifest = Some(root.join("manifest.json"));
    c.optimizer.as_mut().unwrap().lookback_returns = 2;
    c.output_root = root.parent().unwrap().join("runs");
    c
}

fn run(c: &PortfolioConfig) -> Simulation {
    let snapshot = PortfolioSnapshot::load(c).unwrap();
    let result = backtest::simulate(c, &snapshot).unwrap();
    assert!(result.state.errors.is_empty(), "{:?}", result.state.errors);
    result
}

fn economic_fills(result: &Simulation, before: &str) -> Value {
    json!(result.state.fills.iter().filter(|f|f.timestamp.as_str()<before).map(|f|json!({"timestamp":f.timestamp,"instrument":f.instrument_id,"side":f.side,"units":f.units,"price":f.price,"fee":f.commission})).collect::<Vec<_>>())
}

#[test]
fn exact_python_selection_drives_native_orders_with_causal_state_and_safe_fallbacks() {
    let root = std::env::temp_dir().join(format!("qflab-optimizer-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let source = root.join("snapshot");
    fixture(&source, "optimizer");
    let c = config(&source);
    let result = run(&c);
    assert_eq!(result.state.optimizer_records.len(), 3);
    assert!(
        result
            .state
            .optimizer_records
            .iter()
            .all(|r| r.status == "feasible")
    );
    assert!(!result.state.fills.is_empty());
    let selections: std::collections::BTreeSet<_> = result
        .state
        .optimizer_records
        .iter()
        .map(|r| r.response.as_ref().unwrap()["selected"].to_string())
        .collect();
    assert!(
        selections.len() > 1,
        "The research algorithm should change its selected funds when forecasts change"
    );
    for record in &result.state.optimizer_records {
        let cutoff = record.request["data_cutoff"].as_str().unwrap();
        let time: DateTime<chrono::Utc> = cutoff.parse().unwrap();
        assert!(record.available_at > time);
        assert_eq!(record.request["history"].as_array().unwrap().len(), 3);
        for row in record.request["history"].as_array().unwrap() {
            assert!(row["timestamp"].as_str().unwrap() <= cutoff);
        }
        let row = result
            .state
            .equity
            .iter()
            .find(|r| r.timestamp == cutoff)
            .unwrap();
        assert_eq!(record.request["cash"], json!(row.cash));
        for holding in result
            .state
            .holdings
            .iter()
            .filter(|r| r.timestamp == cutoff)
        {
            assert_eq!(
                record.request["holdings"][&holding.instrument_id],
                json!(holding.units.to_string())
            );
        }
        assert_eq!(
            record.response.as_ref().unwrap()["diagnostics"]["evaluated_portfolios"],
            56
        );
        assert_eq!(
            record.response.as_ref().unwrap()["selected"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }
    // Actual fills independently reconcile every native USD account snapshot, including fees.
    for row in &result.state.equity {
        let mut cash = Decimal::from(10000);
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
            cash -= Money::from_decimal(signed * fill.price, Currency::USD())
                .unwrap()
                .as_decimal()
                + fill.commission;
        }
        let invested: Decimal = result
            .state
            .holdings
            .iter()
            .filter(|h| h.timestamp == row.timestamp)
            .map(|h| h.units * h.close)
            .sum();
        assert_eq!((row.cash, row.equity), (cash, cash + invested));
        assert!(cash >= Decimal::ZERO);
    }
    let changed = root.join("future");
    fixture(&changed, "optimizer_future");
    let future = run(&config(&changed));
    for i in 0..2 {
        assert_eq!(
            result.state.optimizer_records[i].request_sha256,
            future.state.optimizer_records[i].request_sha256
        );
        assert_eq!(
            result.state.optimizer_records[i].response,
            future.state.optimizer_records[i].response
        );
    }
    assert_eq!(
        economic_fills(&result, "2025-02-10"),
        economic_fills(&future, "2025-02-10")
    );
    let first = &result.state.optimizer_records[0];
    let mut invalid = first.response.clone().unwrap();
    invalid["request_sha256"] = json!("wrong");
    assert!(validate_response(&c, &first.request, &first.request_sha256, &invalid).is_err());
    invalid = first.response.clone().unwrap();
    invalid["target_weights"]["A0.SIM"] = json!("1");
    assert!(validate_response(&c, &first.request, &first.request_sha256, &invalid).is_err());

    let saved = backtest::run_config(c.clone()).unwrap();
    let mut replay = PortfolioConfig::load(&saved.join("config.json")).unwrap();
    replay.optimizer.as_mut().unwrap().python_path = PathBuf::from("/no/python/needed/for/replay");
    let replay_output = backtest::run_config(replay.clone()).unwrap();
    for name in [
        "equity.csv",
        "holdings.csv",
        "decision_log.json",
        "summary.json",
        "solver_replay.json",
    ] {
        assert_eq!(
            fs::read(saved.join(name)).unwrap(),
            fs::read(replay_output.join(name)).unwrap(),
            "{name}"
        );
    }
    replay.starting_cash = "10001".into();
    let mismatch = backtest::simulate(&replay, &PortfolioSnapshot::load(&replay).unwrap()).unwrap();
    assert!(
        mismatch
            .state
            .errors
            .iter()
            .any(|e| e.contains("Replay optimizer input/state/timing mismatch"))
    );

    let mut delayed = c.clone();
    delayed.optimizer.as_mut().unwrap().simulated_delay_seconds = 172800;
    let delayed = run(&delayed);
    assert_eq!(
        delayed.state.fills[0].timestamp,
        "2025-01-06T21:00:00+00:00"
    );
    let mut unavailable = c.clone();
    unavailable.optimizer.as_mut().unwrap().python_path = PathBuf::from("/no/such/python");
    let failed = run(&unavailable);
    assert!(failed.state.fills.is_empty());
    assert!(
        failed
            .state
            .equity
            .iter()
            .all(|r| r.cash == Decimal::from(10000))
    );
    assert!(
        failed
            .state
            .optimizer_records
            .iter()
            .all(|r| r.status == "fallback" && r.fallback_reason.is_some())
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let slow = root.join("slow-worker");
        let python = c
            .optimizer
            .as_ref()
            .unwrap()
            .python_path
            .canonicalize()
            .unwrap();
        fs::write(
            &slow,
            format!("#!{}\nimport time\ntime.sleep(10)\n", python.display()),
        )
        .unwrap();
        fs::set_permissions(&slow, fs::Permissions::from_mode(0o700)).unwrap();
        let mut timed = c.clone();
        timed.trading_start = Some("2025-03-03".parse().unwrap());
        timed.optimizer.as_mut().unwrap().python_path = slow;
        timed.optimizer.as_mut().unwrap().timeout_seconds = 1;
        let timed = run(&timed);
        assert!(timed.state.fills.is_empty());
        assert_eq!(timed.state.optimizer_records.len(), 1);
        assert!(
            timed.state.optimizer_records[0]
                .fallback_reason
                .as_ref()
                .unwrap()
                .contains("timed out")
        );
    }
    fs::remove_dir_all(root).unwrap();
}
