use anyhow::Result;
use clap::{Parser, Subcommand};
use tpt_gpu_model_registry::ModelRegistry;

#[derive(Parser)]
#[command(
    name = "tpt-gpu-models",
    about = "Manage the ~/.tpt/models/ shared model registry"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List all registered models.
    List,
    /// Show the registry directory path.
    Dir,
    /// Remove a model entry from the manifest (does not delete the file).
    Remove { name: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut registry = ModelRegistry::open()?;

    match cli.cmd {
        Cmd::List => {
            let models = registry.models();
            if models.is_empty() {
                println!("No models registered. Run `tpt-gpu-models` --help for usage.");
            } else {
                println!("{:<30} {:<10} {:<8} {}", "NAME", "ARCH", "SIZE_GB", "FILE");
                for m in models {
                    println!(
                        "{:<30} {:<10} {:<8.1} {}",
                        m.name, m.arch, m.size_gb, m.file
                    );
                }
            }
        }
        Cmd::Dir => {
            println!("{}", registry.dir().display());
        }
        Cmd::Remove { name } => {
            if registry.unregister(&name)? {
                println!("Removed '{}'.", name);
            } else {
                println!("No model named '{}' found.", name);
            }
        }
    }

    Ok(())
}
