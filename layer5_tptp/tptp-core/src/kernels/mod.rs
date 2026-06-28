//! Kernel Host Wrappers
//!
//! High-level Rust wrappers for GPU compute kernels.
//! Each kernel validates inputs, dispatches to vendor library or TPTIR fallback,
//! and manages output buffer allocation.

pub mod gemm;
pub mod attention;
pub mod conv2d;
pub mod conv3d;
pub mod layernorm;
pub mod batchnorm;
pub mod groupnorm;
pub mod rmsnorm;
pub mod softmax;
pub mod elementwise;
pub mod embedding;
pub mod pooling;

pub use gemm::GemmKernel;
pub use attention::AttentionKernel;
pub use conv2d::Conv2DKernel;
pub use conv3d::Conv3DKernel;
pub use layernorm::LayerNormKernel;
pub use batchnorm::BatchNormKernel;
pub use groupnorm::GroupNormKernel;
pub use rmsnorm::RmsNormKernel;
pub use softmax::SoftmaxKernel;
pub use elementwise::{ElementwiseKernel, ActivationKind};
pub use embedding::EmbeddingKernel;
pub use pooling::{MaxPool2DKernel, AvgPool2DKernel};