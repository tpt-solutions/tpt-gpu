//! tpt-optimizer CLI
//!
//! Usage:
//!   tpt-optimizer grid    [--kernel <name>] [--elem <type>]
//!   tpt-optimizer climb   [--kernel <name>] [--tile-m N] ...
//!   tpt-optimizer ai      [--kernel <name>] [--iterations N]
//!   tpt-optimizer optimize [--kernel <name>] [--elem <type>] [--ai]

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use tpt_optimizer::*;

#[derive(Parser)]
#[command(name = "tpt-optimizer", version, about = "TPT GPU kernel parameter optimizer")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Phase 1: exhaustive grid search over the parameter space
    Grid {
        #[arg(long, default_value = "matmul")]
        kernel: String,
        #[arg(long, default_value = "f32")]
        elem: String,
        /// Print the top-N results
        #[arg(long, default_value = "10")]
        top: usize,
    },
    /// Phase 2: hill-climbing from a given starting point
    Climb {
        #[arg(long, default_value = "matmul")]
        kernel: String,
        #[arg(long, default_value = "32")]
        tile_m: u32,
        #[arg(long, default_value = "32")]
        tile_n: u32,
        #[arg(long, default_value = "16")]
        tile_k: u32,
        #[arg(long, default_value = "4")]
        vec_width: u32,
        #[arg(long, default_value = "4")]
        unroll: u32,
        #[arg(long, default_value = "50")]
        max_iters: usize,
    },
    /// Phase 3: AI-guided search (requires ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or local Ollama)
    Ai {
        #[arg(long, default_value = "matmul")]
        kernel: String,
        #[arg(long, default_value = "10")]
        iterations: usize,
    },
    /// Full pipeline: grid → hill-climb → optional AI refinement
    Optimize {
        #[arg(long, default_value = "matmul")]
        kernel: String,
        #[arg(long, default_value = "f32")]
        elem: String,
        /// Enable AI-guided phase 3
        #[arg(long)]
        ai: bool,
        /// Number of AI iterations (only with --ai)
        #[arg(long, default_value = "10")]
        ai_iters: usize,
        /// Write final params as JSON to this file
        #[arg(long)]
        output: Option<std::path::PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Grid { kernel, elem, top } => {
            let space = ParamSpace::gemm_default();
            println!("Grid search: {} (elem={}) — {} configs", kernel, elem, space.total_configs());
            let eval = SimulatedEvaluator::new(&kernel);
            let results = grid_search(&space, &eval);
            println!("\nTop {} results:", top.min(results.len()));
            for r in results.iter().take(top) {
                println!("  {:>7.2} GFLOPS  {}", r.score, r.params.display());
            }
        }

        Commands::Climb { kernel, tile_m, tile_n, tile_k, vec_width, unroll, max_iters } => {
            let space = ParamSpace::gemm_default();
            let start = TuningParams(HashMap::from([
                ("tile_m".into(), tile_m),
                ("tile_n".into(), tile_n),
                ("tile_k".into(), tile_k),
                ("vec_width".into(), vec_width),
                ("unroll".into(), unroll),
            ]));
            println!("Hill-climb: {} — start: {}", kernel, start.display());
            let eval = SimulatedEvaluator::new(&kernel);
            let result = hill_climb(&space, &start, &eval, max_iters);
            println!("\nBest: {:.2} GFLOPS  {}  ({} evals)",
                result.score, result.params.display(), result.eval_count);
        }

        Commands::Ai { kernel, iterations } => {
            let space = ParamSpace::gemm_default();
            let provider = tpt_shared::provider_from_env();
            println!("AI-guided search: {} — provider: {} — {} iterations",
                kernel, provider.name(), iterations);
            let start = TuningParams(HashMap::from([
                ("tile_m".into(), 32u32),
                ("tile_n".into(), 32),
                ("tile_k".into(), 16),
                ("vec_width".into(), 4),
                ("unroll".into(), 4),
            ]));
            let eval = SimulatedEvaluator::new(&kernel);
            let result = ai_guided_search(&space, &start, &eval, provider.as_ref(), &kernel, iterations);
            println!("\nBest: {:.2} GFLOPS  {}  ({} evals)",
                result.score, result.params.display(), result.eval_count);
        }

        Commands::Optimize { kernel, elem, ai, ai_iters, output } => {
            let space = ParamSpace::gemm_default();
            let eval = SimulatedEvaluator::new(&kernel);

            // --- Phase 1: Grid ---
            println!("[1/{}] Grid search ({} configs)...",
                if ai { 3 } else { 2 }, space.total_configs());
            let grid_results = grid_search(&space, &eval);
            let best_grid = &grid_results[0];
            println!("  best: {:.2} GFLOPS  {}", best_grid.score, best_grid.params.display());

            // --- Phase 2: Hill-climb ---
            println!("[2/{}] Hill-climbing from best grid point...", if ai { 3 } else { 2 });
            let hc_result = hill_climb(&space, &best_grid.params, &eval, 100);
            println!("  best: {:.2} GFLOPS  {}  ({} evals)",
                hc_result.score, hc_result.params.display(), hc_result.eval_count);

            // --- Phase 3: AI (optional) ---
            let final_result = if ai {
                println!("[3/3] AI-guided refinement ({} iterations)...", ai_iters);
                let provider = tpt_shared::provider_from_env();
                println!("  provider: {}", provider.name());
                let r = ai_guided_search(
                    &space, &hc_result.params, &eval, provider.as_ref(), &kernel, ai_iters,
                );
                println!("  best: {:.2} GFLOPS  {}  ({} evals)",
                    r.score, r.params.display(), r.eval_count);
                r
            } else {
                hc_result
            };

            println!("\nFinal: {:.2} GFLOPS  {}", final_result.score, final_result.params.display());

            let json_out = serde_json::json!({
                "kernel": kernel,
                "elem": elem,
                "params": final_result.params.0,
                "score_gflops": final_result.score,
                "total_evals": final_result.eval_count,
            });

            if let Some(path) = output {
                std::fs::write(&path, serde_json::to_string_pretty(&json_out)?)?;
                println!("Wrote results to {}", path.display());
            } else {
                println!("\n{}", serde_json::to_string_pretty(&json_out)?);
            }
        }
    }

    Ok(())
}
