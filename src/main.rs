use std::path::Path;

fn run(path: &Path) -> anyhow::Result<()> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    if value.get("mode").is_some() {
        qflab_backtest::backtest::run(path)?;
    } else {
        qflab_backtest::demo::run(path)?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command] if command == "--help" || command == "-h" => {
            println!(
                "Usage: qflab-backtest [CONFIG.json]\n       qflab-backtest compare CONFIG.json\nDefault: configs/demo.json"
            );
        }
        [] => {
            run(Path::new("configs/demo.json"))?;
        }
        [path] => {
            run(Path::new(path))?;
        }
        [command, path] if command == "compare" => {
            qflab_backtest::backtest::comparison::run(Path::new(path))?;
        }
        _ => anyhow::bail!("Expected one config path. Try --help."),
    }
    Ok(())
}
