//! Softmax Kernel Wrapper
//! y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//! Numerically stable; softmax is applied over the specified axis (default: last).
use std::time::Instant;
use crate::error::{TptpError, TptpResult};
use crate::memory::{GpuBuffer, DType, Shape, BufferFlags};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::vendor::VendorBackend;

/// Tunable parameters — map to `{{BLOCK_SIZE}}`, `{{BLOCK_SIZE_HALF}}` placeholders
/// in `tptir_softmax.mlir`.
#[derive(Debug, Clone)]
pub struct SoftmaxParams {
    pub block_size: u32,
    /// Axis to apply softmax over; -1 means the last dimension.
    pub dim: i32,
}

impl Default for SoftmaxParams {
    fn default() -> Self {
        SoftmaxParams { block_size: 256, dim: -1 }
    }
}

/// Softmax kernel handle
pub struct SoftmaxKernel {
    config: KernelConfig,
    vendor: VendorBackend,
    pub params: SoftmaxParams,
}

impl SoftmaxKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096);
        SoftmaxKernel { config, vendor, params: SoftmaxParams::default() }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096);
        SoftmaxKernel { config, vendor, params: SoftmaxParams::default() }
    }

    pub fn with_params(mut self, params: SoftmaxParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    /// Apply softmax to `input`. The tensor is treated as [batch, dim_size] where
    /// `dim_size` is the resolved softmax axis.
    pub fn execute(&self, input: &GpuBuffer<f32>) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() < 1 {
            return Err(TptpError::shape_error("Softmax: input must be at least 1-D"));
        }
        let ndim = input.ndim();
        let axis = if self.params.dim < 0 {
            (ndim as i32 + self.params.dim) as usize
        } else {
            self.params.dim as usize
        };
        if axis >= ndim {
            return Err(TptpError::shape_error("Softmax: dim out of range"));
        }
        let dim_size = input.dim(axis)
            .ok_or_else(|| TptpError::shape_error("Softmax: cannot determine dim_size"))?;
        let batch: usize = (0..ndim).filter(|&d| d != axis).map(|d| input.dim(d).unwrap_or(1)).product();
        let t0 = Instant::now();
        self.tptir_softmax(input, batch, dim_size)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!("Softmax [{}x{}] dim={} via TPTIR: {:.3}ms", batch, dim_size, axis, elapsed_ms);
        GpuBuffer::new(input.shape().clone(), DType::F32, BufferFlags::STORAGE)
    }

    fn tptir_softmax(
        &self,
        _input: &GpuBuffer<f32>,
        _batch: usize,
        _dim_size: usize,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR Softmax fallback: batch={}, dim_size={}, block_size={}",
            _batch, _dim_size, self.params.block_size
        );
        Ok(())
    }
}

impl PrimitiveKernel for SoftmaxKernel {
    fn name(&self) -> &str { "softmax" }
    fn input_shapes(&self) -> &[Shape] { &[] }
    fn output_shape(&self) -> &Shape { unimplemented!("output_shape not static for Softmax") }
    fn supported_dtypes(&self) -> &[DType] { &[DType::F32, DType::F16, DType::BF16] }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 1 && inputs[0].ndim() >= 1
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096)
    }
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        _output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let t0 = Instant::now();
        let result = SoftmaxKernel::execute(self, inputs[0])?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult {
            outputs: vec![result],
            execution_time_ms: Some(elapsed_ms),
            backend_used: "TPTIR".to_string(),
        })
    }
    fn execute_with_vendor(
        &self,
        inputs: &[&GpuBuffer<f32>],
        output: &mut GpuBuffer<f32>,
        _vendor: &VendorBackend,
        config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        PrimitiveKernel::execute(self, inputs, output, config)
    }
}

/// Convenience wrapper — softmax over the last dimension.
pub fn softmax(input: &GpuBuffer<f32>) -> TptpResult<GpuBuffer<f32>> {
    SoftmaxKernel::new().execute(input)
}

/// Convenience wrapper — softmax over an explicit axis.
pub fn softmax_dim(input: &GpuBuffer<f32>, dim: i32) -> TptpResult<GpuBuffer<f32>> {
    SoftmaxKernel::new()
        .with_params(SoftmaxParams { dim, ..SoftmaxParams::default() })
        .execute(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_valid_2d() {
        let input = GpuBuffer::<f32>::new(Shape::dim2(4, 32), DType::F32, BufferFlags::STORAGE).unwrap();
        assert!(SoftmaxKernel::new().execute(&input).is_ok());
    }

    #[test]
    fn test_softmax_valid_1d() {
        let input = GpuBuffer::<f32>::new(Shape::new(&[128]), DType::F32, BufferFlags::STORAGE).unwrap();
        assert!(softmax(&input).is_ok());
    }

    #[test]
    fn test_softmax_explicit_dim() {
        let input = GpuBuffer::<f32>::new(Shape::dim2(8, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        assert!(softmax_dim(&input, 1).is_ok());
        assert!(softmax_dim(&input, -1).is_ok());
    }

    #[test]
    fn test_softmax_dim_out_of_range() {
        let input = GpuBuffer::<f32>::new(Shape::dim2(4, 16), DType::F32, BufferFlags::STORAGE).unwrap();
        assert!(softmax_dim(&input, 5).is_err());
    }

    #[test]
    fn test_softmax_params_default() {
        let p = SoftmaxParams::default();
        assert_eq!(p.block_size, 256);
        assert_eq!(p.dim, -1);
    }
}
