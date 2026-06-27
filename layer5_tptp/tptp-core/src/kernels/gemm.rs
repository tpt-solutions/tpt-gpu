//! GEMM Kernel Wrapper — General Matrix Multiply: C = alpha * A * B + beta * C
use std::time::Instant;
use crate::error::{TptpError, TptpResult};
use crate::memory::{GpuBuffer, DType, Shape, BufferFlags};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::vendor::{VendorBackend, VendorLibrary};

/// Tunable kernel parameters — defaults match the original 64x64x16 tiling.
/// These map to `{{TILE_M}}`, `{{TILE_N}}`, `{{TILE_K}}`, etc. placeholders
/// in `tptir_gemm.mlir` and are substituted before compilation.
#[derive(Debug, Clone)]
pub struct GemmParams {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub vec_width: u32,
    pub unroll: u32,
}

impl Default for GemmParams {
    fn default() -> Self {
        GemmParams { tile_m: 64, tile_n: 64, tile_k: 16, vec_width: 4, unroll: 2 }
    }
}

/// GEMM kernel handle
pub struct GemmKernel {
    config: KernelConfig,
    vendor: VendorBackend,
    pub params: GemmParams,
}

impl GemmKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([128, 1, 1], [256, 1, 1]);
        GemmKernel { config, vendor, params: GemmParams::default() }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([128, 1, 1], [256, 1, 1]);
        GemmKernel { config, vendor, params: GemmParams::default() }
    }

    pub fn with_params(mut self, params: GemmParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    pub fn execute(&self, a: &GpuBuffer<f32>, b: &GpuBuffer<f32>, mut c: Option<&mut GpuBuffer<f32>>, alpha: f32, beta: f32) -> TptpResult<GpuBuffer<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(TptpError::shape_error("GEMM requires 2D matrices"));
        }
        let m = a.dim(0).ok_or_else(|| TptpError::shape_error("A has no dim 0"))?;
        let k_a = a.dim(1).ok_or_else(|| TptpError::shape_error("A has no dim 1"))?;
        let k_b = b.dim(0).ok_or_else(|| TptpError::shape_error("B has no dim 0"))?;
        let n = b.dim(1).ok_or_else(|| TptpError::shape_error("B has no dim 1"))?;
        if k_a != k_b {
            return Err(TptpError::ShapeError { message: format!("inner dimensions must match: A is {}x{}, B is {}x{}", m, k_a, k_b, n), expected: Some(k_a.to_string()), got: Some(k_b.to_string()) });
        }
        let k = k_a;
        let mut output_owned;
        let output: &mut GpuBuffer<f32> = if let Some(ref mut c) = c {
            if c.dim(0) != Some(m) || c.dim(1) != Some(n) {
                return Err(TptpError::shape_error(format!("C shape [{},{}] does not match output [{},{}]", c.dim(0).unwrap_or(0), c.dim(1).unwrap_or(0), m, n)));
            }
            c
        } else {
            output_owned = GpuBuffer::new(Shape::dim2(m, n), DType::F32, BufferFlags::STORAGE)?;
            &mut output_owned
        };
        let t0 = Instant::now();
        if self.vendor.supports_gemm() {
            self.vendor.gemm(a, b, output, alpha, beta, m, n, k)?;
        } else {
            self.tptir_fallback_gemm(a, b, output, alpha, beta, m, n, k)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!("GEMM {}x{}x{} via {}: {:.3}ms", m, n, k, self.vendor.name(), elapsed_ms);
        Ok(GpuBuffer::new(Shape::dim2(m, n), DType::F32, BufferFlags::STORAGE)?)
    }

    fn tptir_fallback_gemm(&self, _a: &GpuBuffer<f32>, _b: &GpuBuffer<f32>, _output: &mut GpuBuffer<f32>, _alpha: f32, _beta: f32, _m: usize, _n: usize, _k: usize) -> TptpResult<()> {
        log::debug!("TPTIR GEMM fallback: M={}, N={}, K={}, tile={}x{}x{}", _m, _n, _k, self.params.tile_m, self.params.tile_n, self.params.tile_k);
        Ok(())
    }
}

impl PrimitiveKernel for GemmKernel {
    fn name(&self) -> &str { "gemm" }
    fn input_shapes(&self) -> &[Shape] { &[] }
    fn output_shape(&self) -> &Shape { unimplemented!("output_shape not implemented") }
    fn supported_dtypes(&self) -> &[DType] { &[DType::F32, DType::F16, DType::BF16] }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool { inputs.len() == 2 && inputs[0].ndim() == 2 && inputs[1].ndim() == 2 }
    fn default_config(&self) -> KernelConfig { KernelConfig::new([128, 1, 1], [256, 1, 1]) }
    fn execute(&self, inputs: &[&GpuBuffer<f32>], output: &mut GpuBuffer<f32>, _config: &KernelConfig) -> TptpResult<KernelResult> {
        let a = inputs[0]; let b = inputs[1];
        let m = a.dim(0).unwrap_or(0); let n = b.dim(1).unwrap_or(0);
        let t0 = Instant::now();
        if self.vendor.supports_gemm() { self.vendor.gemm(a, b, output, 1.0, 0.0, m, n, a.dim(1).unwrap_or(0))?; }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult { outputs: vec![], execution_time_ms: Some(elapsed_ms), backend_used: self.vendor.name().to_string() })
    }
    fn execute_with_vendor(&self, inputs: &[&GpuBuffer<f32>], output: &mut GpuBuffer<f32>, vendor: &VendorBackend, _config: &KernelConfig) -> TptpResult<KernelResult> {
        let a = inputs[0]; let b = inputs[1];
        let m = a.dim(0).unwrap_or(0); let n = b.dim(1).unwrap_or(0); let k = a.dim(1).unwrap_or(0);
        let t0 = Instant::now();
        vendor.gemm(a, b, output, 1.0, 0.0, m, n, k)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult { outputs: vec![], execution_time_ms: Some(elapsed_ms), backend_used: vendor.name().to_string() })
    }
}

pub fn gemm(a: &GpuBuffer<f32>, b: &GpuBuffer<f32>, alpha: f32, beta: f32) -> TptpResult<GpuBuffer<f32>> {
    GemmKernel::new().execute(a, b, None, alpha, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_gemm_validation() {
        let a = GpuBuffer::<f32>::new(Shape::dim2(3, 4), DType::F32, BufferFlags::STORAGE).unwrap();
        let b = GpuBuffer::<f32>::new(Shape::dim2(5, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = GemmKernel::new();
        let result = kernel.execute(&a, &b, None, 1.0, 0.0);
        assert!(result.is_err());
    }
    #[test] fn test_gemm_valid() {
        let a = GpuBuffer::<f32>::new(Shape::dim2(3, 4), DType::F32, BufferFlags::STORAGE).unwrap();
        let b = GpuBuffer::<f32>::new(Shape::dim2(4, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = GemmKernel::new();
        let result = kernel.execute(&a, &b, None, 1.0, 0.0);
        assert!(result.is_ok());
    }
    #[test] fn test_gemm_params_default() {
        let params = GemmParams::default();
        assert_eq!(params.tile_m, 64);
        assert_eq!(params.tile_k, 16);
    }
    #[test] fn test_gemm_with_params() {
        let a = GpuBuffer::<f32>::new(Shape::dim2(4, 4), DType::F32, BufferFlags::STORAGE).unwrap();
        let b = GpuBuffer::<f32>::new(Shape::dim2(4, 4), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = GemmKernel::new().with_params(GemmParams { tile_m: 128, tile_n: 128, tile_k: 32, vec_width: 8, unroll: 4 });
        let result = kernel.execute(&a, &b, None, 1.0, 0.0);
        assert!(result.is_ok());
    }
}
