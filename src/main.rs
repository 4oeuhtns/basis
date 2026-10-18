use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command] if command == "--help" || command == "-h" => {
            println!("Usage: qflab-backtest [CONFIG.json]\nDefault: configs/demo.json");
        }
        [] => {
            qflab_backtest::runner::run(Path::new("configs/demo.json"))?;
        }
        [path] => {
            qflab_backtest::runner::run(Path::new(path))?;
        }
        _ => anyhow::bail!("Expected one config path. Try --help."),
    }
    Ok(())
}
