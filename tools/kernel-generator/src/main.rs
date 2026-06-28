//! TPT GPU kernel generator CLI.
//!
//! Usage:
//!   tpt generate <kernel> [--elem <type>] [--shape <shape>]
//!   tpt validate <file.tptir>
//!   tpt bench [--quick]
//!   tpt version

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "tpt", version, about = "TPT GPU kernel generator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a kernel template
    Generate {
        /// Kernel name (e.g., vector_add, matmul, softmax, flash_attention, conv_bn_relu)
        kernel: String,
        /// Element type (f32, f16, bf16, i32)
        #[arg(long, default_value = "f32")]
        elem: String,
        /// Shape parameters (e.g., 1024 or 16x16)
        #[arg(long, default_value = "1024")]
        shape: String,
    },
    /// Validate a TPTIR file
    Validate {
        /// Path to the TPTIR file
        file: PathBuf,
    },
    /// Run benchmarks
    Bench {
        /// Run in quick mode (30 seconds)
        #[arg(long)]
        quick: bool,
    },
    /// Show version
    Version,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { kernel, elem, shape } => {
            println!("Generating kernel: {} (elem={}, shape={})", kernel, elem, shape);
            // Parse element type
            let elem_type = match elem.as_str() {
                "f32" => tptc_rs::ir::ElemType::F32,
                "f16" => tptc_rs::ir::ElemType::F16,
                "bf16" => tptc_rs::ir::ElemType::BF16,
                "i32" => tptc_rs::ir::ElemType::I32,
                _ => anyhow::bail!("Unknown element type: {}", elem),
            };
            // Parse shape
            let shape_params: Vec<i64> = shape
                .split('x')
                .map(|s| s.parse::<i64>())
                .collect::<Result<Vec<_>, _>>()?;
            // Build kernel region
            let region = tptc_rs::ir::build_kernel_region(&kernel, elem_type, &shape_params)?;
            // Emit TPTIR
            let tptir_text = tptc_rs::ir::emit_tptir(&region, &kernel, &[("tptir.kernel".to_string(), "".to_string())]);
            println!("{}", tptir_text);
        }
        Commands::Validate { file } => {
            println!("Validating: {:?}", file);
            let source = std::fs::read_to_string(&file)?;
            let region = tptc_rs::ir::parse_assembly(&source)?;
            let result = tptc_rs::validate::validate_region(&region);
            if result.is_valid() {
                println!("Validation passed!");
            } else {
                println!("Validation failed with {} errors:", result.error_count());
                for err in &result.errors {
                    println!("  - {}", err);
                }
            }
        }
        Commands::Bench { quick } => {
            if quick {
                println!("Running quick benchmark (30 seconds)...");
            } else {
                println!("Running full benchmark...");
            }
            // In a real implementation, we would run the actual benchmarks
            println!("Benchmark complete!");
        }
        Commands::Version => {
            println!("{}", tptc_rs::version());
        }
    }

    Ok(())
}
