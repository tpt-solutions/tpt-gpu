//! Comprehensive benchmark runner with vendor baselines and statistical analysis
//!
//! Runs TPT vs vendor library comparisons across all supported primitives

use clap::Parser;
use std::path::PathBuf;
use std::fs;
use std::collections::HashMap;

use tptp_benches::harness::{BenchConfig, BenchHarness};
use tptp_benches::kernels::{GemmBench, AttentionBench, Conv2DBench};
use tptp_benches::report::{BenchReport, BaselineComparison};
use tptp_benches::problem_configs::{get_gemm_config, get_attention_config, get_conv2d_config, get_all_baselines};
use tptp_benches::stats::{compute_statistics, StatisticalSummary};

#[derive(Parser)]
#[command(name = "tpt-comprehensive", version, about = "Comprehensive TPT Benchmark Suite")]
struct Cli {
    /// Run in quick mode (30-second sanity check)
    #[arg(long)]
    quick: bool,
    
    /// Run in CI mode (minimal iterations)
    #[arg(long)]
    ci: bool,
    
    /// Run comprehensive mode (30-minute thorough benchmark)
    #[arg(long)]
    comprehensive: bool,
    
    /// Output file path (.json, .md, or .csv)
    #[arg(long, short)]
    output: Option<PathBuf>,
    
    /// Skip vendor baselines
    #[arg(long)]
    no_baselines: bool,
    
    /// Specific vendor to compare against
    #[arg(long)]
    vendor: Option<String>,
    
    /// JSON file with custom problem sizes
    #[arg(long)]
    config: Option<PathBuf>,
    
    /// Compute detailed statistics
    #[arg(long, default_value = "true")]
    stats: bool,
    
    /// Export raw measurements to CSV for further analysis
    #[arg(long)]
    csv_export: bool,
}

/// Run kernel with optional vendor baseline
fn run_with_baseline(
    harness: &mut BenchHarness,
    kernel_name: &str,
    problem_desc: &str,
    baseline_time_ms: Option<f64>,
    baseline_vendor: Option<&str>,
) {
    // Print baseline info if available
    if let (Some(time), Some(vendor)) = (baseline_time_ms, baseline_vendor) {
        println!("  Baseline: {} = {:.3} ms", vendor, time);
    }
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    // Determine benchmark configuration
    let config = if cli.ci {
        println!("Running in CI mode (minimal iterations)");
        BenchConfig::ci()
    } else if cli.quick {
        println!("Running in quick mode (30-second sanity check)");
        BenchConfig::quick()
    } else if cli.comprehensive {
        println!("Running comprehensive benchmark (up to 30 minutes)");
        BenchConfig::comprehensive()
    } else {
        println!("Running standard benchmark");
        BenchConfig::standard()
    };

    let mut config = config;
    if cli.no_baselines {
        config.run_baselines = false;
    }
    if let Some(ref vendor) = cli.vendor {
        config.baseline_vendor = Some(vendor.clone());
    }

    println!("Configuration:");
    println!("  Warmup iterations: {}", config.warmup_iterations);
    println!("  Measurement iterations: {}", config.measurement_iterations);
    println!("  Run baselines: {}", config.run_baselines);
    if let Some(ref vendor) = config.baseline_vendor {
        println!("  Baseline vendor: {}", vendor);
    }
    println!();

    let mut harness = BenchHarness::new(config.clone());
    let quick = cli.quick || cli.ci;

    // Get problem configurations
    let gemm_problems = get_gemm_config(quick);
    let attention_problems = get_attention_config(quick);
    let conv2d_problems = get_conv2d_config(quick);

    // Build baseline lookup
    let baselines = get_all_baselines();
    let mut baseline_map: HashMap<(String, String), (f64, String)> = HashMap::new();
    for (kernel, problem, vendor, time) in &baselines {
        baseline_map.insert(
            (kernel.clone(), problem.clone()),
            (*time, vendor.clone()),
        );
    }

    // Run GEMM benchmarks
    println!("═══ GEMM Benchmarks ═══");
    println!("Comparing TPT vs cuBLAS / rocBLAS / OpenBLAS");
    let mut gemm_results = Vec::new();
    for prob in &gemm_problems {
        let problem_desc = format!("{}x{}x{}", prob.m, prob.k, prob.n);
        let shape = vec![prob.m, prob.k, prob.n];
        
        println!("\n  Problem: {}", problem_desc);
        run_with_baseline(
            &mut harness,
            "gemm",
            &problem_desc,
            Some(prob.baseline_ms),
            Some(&prob.baseline_vendor),
        );
        
        // Note: In real implementation, we'd execute the kernel here
        // For now, this shows the structure
        println!("    -> TPT execution would go here [{}, {}, {}]", prob.m, prob.k, prob.n);
    }

    // Run Attention benchmarks
    println!("\n═══ Attention Benchmarks ═══");
    println!("Comparing TPT vs FlashAttention v2 / cuDNN");
    for prob in &attention_problems {
        let problem_desc = format!("S={} D={}", prob.seq_len, prob.d_k);
        println!("\n  Problem: {}", problem_desc);
        run_with_baseline(
            &mut harness,
            "attention",
            &problem_desc,
            Some(prob.baseline_ms),
            Some(&prob.baseline_vendor),
        );
        println!("    -> TPT execution would go here [seq={}, d={}]", prob.seq_len, prob.d_k);
    }

    // Run Conv2D benchmarks
    println!("\n═══ Conv2D Benchmarks ═══");
    println!("Comparing TPT vs cuDNN");
    for prob in &conv2d_problems {
        let problem_desc = format!("{}x{} C={} K={} k={}", prob.h, prob.w, prob.c_in, prob.c_out, prob.kernel_size);
        println!("\n  Problem: {}", problem_desc);
        run_with_baseline(
            &mut harness,
            "conv2d",
            &problem_desc,
            Some(prob.baseline_ms),
            Some(&prob.baseline_vendor),
        );
        println!("    -> TPT execution would go here");
    }

    println!("\n═══ Benchmark Complete ═══");
    println!("To see actual results, run with proper tptp-core backend enabled.");
}