//! # TPT Primitives (tptp-core)
//!
//! GPU compute primitives for the TPT GPU platform.
//! Provides high-level Rust wrappers for GEMM, Attention, Conv2D, Conv3D,
//! LayerNorm, BatchNorm, and GroupNorm kernels with TPTIR compilation and vendor dispatch.

pub mod error;
pub mod kernel;
pub mod memory;
pub mod ffi;
pub mod kernels;
pub mod vendor;
pub mod tptir;

pub use error::{TptpError, TptpResult};
pub use kernel::{PrimitiveKernel, KernelConfig, KernelDispatch, KernelBuilder, KernelResult};
pub use memory::{GpuBuffer, BufferFlags, DType};
pub use kernels::{GemmKernel, AttentionKernel, Conv2DKernel, Conv3DKernel,
                  LayerNormKernel, BatchNormKernel, GroupNormKernel};
pub use kernels::gemm::GemmParams;
pub use kernels::attention::AttentionParams;
pub use kernels::conv2d::Conv2DParams;
pub use kernels::conv3d::Conv3DParams;
pub use kernels::layernorm::LayerNormParams;
pub use kernels::batchnorm::BatchNormParams;
pub use kernels::groupnorm::GroupNormParams;
pub use vendor::{VendorBackend, VendorLibrary};
pub use tptir::{TptirCompiler, CompilationOptions, CompilationTarget};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{TptpError, TptpResult, GpuBuffer, DType, BufferFlags};
    pub use crate::kernel::{PrimitiveKernel, KernelConfig, KernelBuilder, KernelResult};
    pub use crate::kernels::{GemmKernel, AttentionKernel, Conv2DKernel, Conv3DKernel,
                              LayerNormKernel, BatchNormKernel, GroupNormKernel};
    pub use crate::vendor::VendorBackend;
}

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name
pub const NAME: &str = env!("CARGO_PKG_NAME");