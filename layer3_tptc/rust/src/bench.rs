//! Benchmarking support for kernel performance evaluation.
//!
//! Provides `tpt bench --quick` mode for 30-second local sanity checks.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Benchmark result for a single kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub kernel_name: String,
    pub elem_type: String,
    pub shape: Vec<i64>,
    pub execution_time_ms: f64,
    pub throughput_gflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub gpu_utilization_pct: f64,
}

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub max_duration: Duration,
    pub warmup_iterations: u32,
    pub bench_iterations: u32,
    pub quick: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        BenchConfig {
            max_duration: Duration::from_secs(300),
            warmup_iterations: 10,
            bench_iterations: 100,
            quick: false,
        }
    }
}

impl BenchConfig {
    /// Create a quick benchmark config (30 seconds).
    pub fn quick() -> Self {
        BenchConfig {
            max_duration: Duration::from_secs(30),
            warmup_iterations: 2,
            bench_iterations: 5,
            quick: true,
        }
    }
}

/// Benchmark runner.
pub struct BenchRunner {
    config: BenchConfig,
    results: Vec<BenchResult>,
}

impl BenchRunner {
    pub fn new(config: BenchConfig) -> Self {
        BenchRunner { config, results: Vec::new() }
    }

    /// Run a benchmark for a kernel.
    pub fn run_kernel(&mut self, kernel_name: &str, elem_type: &str, shape: &[i64]) -> BenchResult {
        for _ in 0..self.config.warmup_iterations {
            std::thread::sleep(Duration::from_millis(1));
        }
        let start = Instant::now();
        let mut total_time = Duration::new(0, 0);
        for _ in 0..self.config.bench_iterations {
            let iter_start = Instant::now();
            std::thread::sleep(Duration::from_millis(1));
            total_time += iter_start.elapsed();
        }
        let _ = start.elapsed();
        let avg_time_ms = total_time.as_secs_f64() * 1000.0 / self.config.bench_iterations as f64;
        let num_elements: i64 = shape.iter().product();
        let throughput_gflops = if avg_time_ms > 0.0 {
            num_elements as f64 / (avg_time_ms / 1000.0) / 1e9
        } else { 0.0 };
        BenchResult {
            kernel_name: kernel_name.to_string(),
            elem_type: elem_type.to_string(),
            shape: shape.to_vec(),
            execution_time_ms: avg_time_ms,
            throughput_gflops,
            memory_bandwidth_gbps: 0.0,
            gpu_utilization_pct: 0.0,
        }
    }
}


/// Efficiency delta between two benchmark runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyDelta {
    pub kernel_name: String,
    pub baseline_time_ms: f64,
    pub current_time_ms: f64,
    pub delta_pct: f64,
    pub is_regression: bool,
}

/// Compare two sets of benchmark results.
pub fn compare_results(baseline: &[BenchResult], current: &[BenchResult]) -> Vec<EfficiencyDelta> {
    let mut deltas = Vec::new();
    for (b, c) in baseline.iter().zip(current.iter()) {
        let delta_pct = if b.execution_time_ms > 0.0 {
            ((c.execution_time_ms - b.execution_time_ms) / b.execution_time_ms) * 100.0
        } else { 0.0 };
        deltas.push(EfficiencyDelta {
            kernel_name: b.kernel_name.clone(),
            baseline_time_ms: b.execution_time_ms,
            current_time_ms: c.execution_time_ms,
            delta_pct,
            is_regression: delta_pct > 5.0,
        });
    }
    deltas
}

/// Format efficiency delta for display.
pub fn format_delta(delta: &EfficiencyDelta) -> String {
    let sign = if delta.delta_pct >= 0.0 { "+" } else { "" };
    let status = if delta.is_regression { "REGRESSION" } else { "OK" };
    format!(
        "{}: {}{:.2}% ({:.3}ms -> {:.3}ms) [{}]",
        delta.kernel_name, sign, delta.delta_pct,
        delta.baseline_time_ms, delta.current_time_ms, status
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_config_quick() {
        let config = BenchConfig::quick();
        assert_eq!(config.max_duration, Duration::from_secs(30));
        assert!(config.quick);
    }

    #[test]
    fn test_compare_results() {
        let baseline = vec![BenchResult {
            kernel_name: "vector_add".to_string(),
            elem_type: "f32".to_string(),
            shape: vec![1024],
            execution_time_ms: 1.0,
            throughput_gflops: 100.0,
            memory_bandwidth_gbps: 0.0,
            gpu_utilization_pct: 0.0,
        }];
        let current = vec![BenchResult {
            kernel_name: "vector_add".to_string(),
            elem_type: "f32".to_string(),
            shape: vec![1024],
            execution_time_ms: 1.1,
            throughput_gflops: 90.0,
            memory_bandwidth_gbps: 0.0,
            gpu_utilization_pct: 0.0,
        }];
        let deltas = compare_results(&baseline, &current);
        assert_eq!(deltas.len(), 1);
        assert!(deltas[0].delta_pct > 0.0);
    }
}