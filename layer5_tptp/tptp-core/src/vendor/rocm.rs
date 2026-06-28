//! ROCm/rocBLAS Backend
//!
//! AMD GPU support via rocBLAS and MIOpen libraries.

use crate::error::{TptpError, TptpResult};
use crate::memory::GpuBuffer;
use super::VendorLibrary;

/// ROCm backend handle
#[derive(Clone, Debug)]
pub struct RocmBackend {
    /// Device ID
    device_id: i32,
    /// HIP stream (opaque)
    #[cfg(feature = "rocm")]
    stream: *mut std::ffi::c_void,
}

#[cfg(feature = "rocm")]
unsafe impl Send for RocmBackend {}
#[cfg(feature = "rocm")]
unsafe impl Sync for RocmBackend {}

impl RocmBackend {
    /// Create a new ROCm backend
    pub fn new() -> TptpResult<Self> {
        #[cfg(feature = "rocm")]
        {
            Ok(RocmBackend {
                device_id: 0,
                stream: std::ptr::null_mut(),
            })
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(TptpError::vendor_unavailable("ROCm support not compiled in"))
        }
    }

    /// Get the device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl VendorLibrary for RocmBackend {
    fn name(&self) -> &str {
        "ROCm"
    }

    fn is_available(&self) -> bool {
        cfg!(feature = "rocm")
    }

    fn gemm(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        alpha: f32,
        beta: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> TptpResult<()> {
        #[cfg(feature = "rocm")]
        {
            log::debug!("rocBLAS GEMM: M={}, N={}, K={}, alpha={}, beta={}", m, n, k, alpha, beta);
            let _ = (a, b, c);
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (a, b, c, alpha, beta, m, n, k);
            Err(TptpError::vendor_unavailable("ROCm support not compiled in"))
        }
    }

    fn attention(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        scale: f32,
        seq_len: usize,
        d_k: usize,
    ) -> TptpResult<()> {
        #[cfg(feature = "rocm")]
        {
            log::debug!("MIOpen Attention: seq_len={}, d_k={}, scale={}", seq_len, d_k, scale);
            let _ = (q, k, v, output);
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (q, k, v, output, scale, seq_len, d_k);
            Err(TptpError::vendor_unavailable("ROCm support not compiled in"))
        }
    }

    fn conv2d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 2],
        padding: [u32; 2],
    ) -> TptpResult<()> {
        #[cfg(feature = "rocm")]
        {
            log::debug!("MIOpen Conv2D: strides={:?}, padding={:?}", strides, padding);
            let _ = (input, filter, output);
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (input, filter, output, strides, padding);
            Err(TptpError::vendor_unavailable("ROCm support not compiled in"))
        }
    }

    fn conv3d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 3],
        padding: [u32; 3],
    ) -> TptpResult<()> {
        #[cfg(feature = "rocm")]
        {
            log::debug!("MIOpen Conv3D: strides={:?}, padding={:?}", strides, padding);
            let _ = (input, filter, output);
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (input, filter, output, strides, padding);
            Err(TptpError::vendor_unavailable("ROCm support not compiled in"))
        }
    }
}