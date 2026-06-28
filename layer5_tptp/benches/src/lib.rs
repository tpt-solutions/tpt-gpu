//! # TPT Primitives Benchmark Harness
//!
//! Structured benchmark output comparing TPT kernels against vendor baselines:
//! - GEMM vs cuBLAS / rocBLAS / OpenBLAS
//! - Attention vs FlashAttention v2 / cuDNN
//! - Conv2D vs cuDNN
//!
//! Output is structured JSON with GFLOPS, bandwidth GB/s, and efficiency-vs-baseline %.

pub mod kernels;
pub mod report;
pub mod harness;
pub mod stats;
pub mod problem_configs;

pub use harness::{BenchConfig, BenchHarness, BenchResult, KernelBench};
pub use report::{BenchReport, BaselineComparison};
pub use stats::{compute_statistics, StatisticalSummary};
pub use problem_configs::{get_gemm_config, get_attention_config, get_conv2d_config};
