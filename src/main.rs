use std::path::Path;

fn run(path: &Path) -> anyhow::Result<()> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    if value.get("mode").is_some() {
        qflab_backtest::portfolio::runner::run(path)?;
    } else {
        qflab_backtest::runner::run(path)?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command] if command == "--help" || command == "-h" => {
            println!("Usage: qflab-backtest [CONFIG.json]\nDefault: configs/demo.json");
        }
        [] => {
            run(Path::new("configs/demo.json"))?;
        }
        [path] => {
            run(Path::new(path))?;
        }
        _ => anyhow::bail!("Expected one config path. Try --help."),
    }
    Ok(())
}
