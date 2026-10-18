//! Cross-language Parquet/CSV parity through the native Nautilus engine, without network access.
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use nautilus_model::types::{Currency, Money};
use qflab_backtest::{
    data::sha256,
    portfolio::{
        PortfolioSnapshot,
        config::{Asset, PortfolioConfig},
        runner::{self, Simulation},
    },
};
use rust_decimal::Decimal;
use serde_json::{Value, json};

fn config(root: &Path, extension: &str) -> PortfolioConfig {
    let mut config = PortfolioConfig::load(Path::new("configs/portfolio.json")).unwrap();
    config.data_kind = qflab_backtest::historical::KIND.into();
    config.snapshot_manifest = Some(root.join("manifest.json"));
    config.assets = ["ALPHA", "BETA"]
        .into_iter()
        .map(|symbol| Asset {
            symbol: symbol.into(),
            data_path: root.join(format!("{symbol}.{extension}")),
        })
        .collect();
    config.output_root = root.parent().unwrap().join("runs");
    config
}

fn economics(result: &Simulation) -> Value {
    json!({"equity": result.state.equity, "holdings": result.state.holdings,
        "decisions": result.state.decisions,
        "fills": result.state.fills.iter().map(|f| json!({"timestamp": f.timestamp,
            "instrument": f.instrument_id, "side": f.side, "units": f.units,
            "price": f.price, "fee": f.commission})).collect::<Vec<_>>()})
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
            // Native USD accounts round trade notionals to cents even with six-decimal marks.
            let notional = Money::from_decimal(signed * fill.price, Currency::USD())
                .unwrap()
                .as_decimal();
            cash -= notional + fill.commission;
            *units.entry(fill.instrument_id.clone()).or_default() += signed;
        }
        let value: Decimal = result
            .state
            .holdings
            .iter()
            .filter(|h| h.timestamp == row.timestamp)
            .map(|h| {
                assert_eq!(
                    h.units,
                    *units.get(&h.instrument_id).unwrap_or(&Decimal::ZERO)
                );
                h.units * h.close
            })
            .sum();
        assert_eq!(
            (row.cash, row.invested_value, row.equity),
            (cash, value, cash + value)
        );
        assert!(cash >= Decimal::ZERO);
    }
}

fn rehash(root: &Path, name: &str) {
    let path = root.join("manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["files"][name] = json!(sha256(&fs::read(root.join(name)).unwrap()));
    fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
}

#[test]
fn historical_formats_have_identical_execution_and_frozen_replay() {
    let root = std::env::temp_dir().join(format!("qflab-history-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let snapshot = root.join("snapshot");
    let python = std::env::var_os("QFLAB_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".venv/bin/python"));
    assert!(
        Command::new(python)
            .args(["tests/historical_fixture.py"])
            .arg(&snapshot)
            .env("PYTHONPATH", ".")
            .status()
            .unwrap()
            .success()
    );
    let parquet_config = config(&snapshot, "parquet");
    let csv_config = config(&snapshot, "csv");
    let parquet = PortfolioSnapshot::load(&parquet_config).unwrap();
    let csv = PortfolioSnapshot::load(&csv_config).unwrap();
    assert_eq!(parquet.session_count(), 6);
    assert_eq!(
        parquet.assets["ALPHA"].sessions[2].timestamp.to_rfc3339(),
        "2025-07-03T17:00:00+00:00"
    );
    assert_eq!(parquet.assets["ALPHA"].sessions[0].volume, 1_600_000_000);
    let from_parquet = runner::simulate(&parquet_config, &parquet).unwrap();
    let from_csv = runner::simulate(&csv_config, &csv).unwrap();
    assert!(
        from_parquet.state.errors.is_empty(),
        "{:?}",
        from_parquet.state.errors
    );
    assert!(
        from_csv.state.errors.is_empty(),
        "{:?}",
        from_csv.state.errors
    );
    assert_eq!(economics(&from_parquet), economics(&from_csv));
    check_ledger(&from_parquet);
    assert_eq!(
        from_parquet.state.fills[2].timestamp,
        "2025-07-07T20:00:00+00:00"
    );

    let mut fees = parquet_config.clone();
    fees.fee_bps = "25".into();
    let paid = runner::simulate(&fees, &parquet).unwrap();
    assert!(paid.state.errors.is_empty(), "{:?}", paid.state.errors);
    check_ledger(&paid);

    let recipe = root.join("config.json");
    fs::write(&recipe, serde_json::to_vec(&parquet_config).unwrap()).unwrap();
    let output = runner::run(&recipe).unwrap();
    let replay = runner::run(&output.join("config.json")).unwrap();
    for file in [
        "summary.json",
        "equity.csv",
        "holdings.csv",
        "decision_log.json",
    ] {
        assert_eq!(
            fs::read(output.join(file)).unwrap(),
            fs::read(replay.join(file)).unwrap()
        );
    }
    assert!(output.join("snapshot/ALPHA.source.parquet").exists());

    // Invalid cached files cannot be accepted or silently replaced.
    let original = fs::read(snapshot.join("ALPHA.csv")).unwrap();
    fs::write(snapshot.join("ALPHA.csv"), b"broken").unwrap();
    assert!(
        PortfolioSnapshot::load(&csv_config)
            .unwrap_err()
            .to_string()
            .contains("hash mismatch")
    );
    fs::write(snapshot.join("ALPHA.csv"), &original).unwrap();
    let text = String::from_utf8(original).unwrap();
    let missing = text
        .lines()
        .filter(|line| !line.starts_with("2025-07-03,"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(snapshot.join("ALPHA.csv"), missing).unwrap();
    rehash(&snapshot, "ALPHA.csv");
    assert!(
        PortfolioSnapshot::load(&csv_config)
            .unwrap_err()
            .to_string()
            .contains("coverage mismatch")
    );
    // Even with a matching file hash, midnight cannot masquerade as a session close.
    let rows: Vec<_> = text.lines().collect();
    let close = rows[1].split(',').nth(1).unwrap();
    fs::write(snapshot.join("ALPHA.csv"), text.replace(close, "1")).unwrap();
    rehash(&snapshot, "ALPHA.csv");
    assert!(
        PortfolioSnapshot::load(&csv_config)
            .unwrap_err()
            .to_string()
            .contains("calendar session close")
    );
    fs::remove_dir_all(root).unwrap();
}
