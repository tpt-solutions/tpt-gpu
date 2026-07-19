//! # TPT Runtime / tptr-core
//!
//! Core library for the TPT GPU compute runtime. Provides GPU memory management,
//! command queue / scheduler, kernel launch interface, and device abstraction.

pub mod error;
pub mod memory;
pub mod command;
pub mod kernel;
pub mod device;
pub mod arch;
pub mod kv_cache;
pub mod inference;

/// Re-export the error types at the crate root.
pub use error::{TptrError, TptrResult, ErrorCode, ErrorContext};

/// Re-export the device types at the crate root.
pub use device::{Device, DeviceInfo, DeviceHandle, DeviceProperties};

/// Re-export memory types at the crate root.
pub use memory::{MemoryAllocation, MemoryRegion, MemType, MemAccess};

/// Re-export command types at the crate root.
pub use command::{CommandQueue, CommandScheduler, Command, QueuePriority, QueueHandle};

/// Re-export kernel types at the crate root.
pub use kernel::{Kernel, KernelConfig, KernelHandle, Dim3, KernelLaunchMode};

/// Re-export inference types at the crate root.
pub use inference::{LlmInference, GpuInferenceEngine, ModelInfo, parse_gguf_header};

/// Re-export arch-template types at the crate root.
pub use arch::{ArchTemplate, ForwardOp, template_for_arch};

/// Re-export KV cache at the crate root.
pub use kv_cache::KvCache;
