import pathlib
path = pathlib.Path(r'c:\Programming\tpt-gpu\layer4_tptr\tptr-core\src\lib.rs')
content = """//! # TPT Runtime / tptr-core
//!
//! Core library for the TPT GPU compute runtime. Provides GPU memory management,
//! command queue / scheduler, kernel launch interface, and device abstraction.

pub mod error;
pub mod memory;
pub mod command;
pub mod kernel;
pub mod device;

/// Re-export the error types at the crate root.
pub use error::{TptrError, TptrResult, ErrorCode, ErrorContext};

/// Re-export the device types at the crate root.
pub use device::{Device, DeviceInfo, DeviceHandle};

/// Re-export memory types at the crate root.
pub use memory::{MemoryAllocation, MemoryRegion, MemType, MemAccess};

/// Re-export command types at the crate root.
pub use command::{CommandQueue, CommandScheduler, Command, QueuePriority, QueueHandle};

/// Re-export kernel types at the crate root.
pub use kernel::{Kernel, KernelConfig, KernelHandle, Dim3, KernelLaunchMode};
"""
path.write_text(content)
print("Fixed lib.rs")

