"""
TPT Framework Backends (tptr) - Python thin wrapper over Rust runtime.

Provides a Pythonic API over the TPT GPU runtime (tptr-core) via PyO3 bindings.
Includes PyTorch and JAX integration for seamless ML framework interop.
"""

__version__ = "1.0.0"
__license__ = "Apache-2.0"

# Re-export core types from the Rust-backed tptr module
from ._ffi import (
    Device,
    MemoryAllocation,
    CommandQueue,
    Queue,
    Kernel,
    KernelConfig,
    KernelHandle,
    TptrError,
)

# Re-export high-level wrappers
from .core import (
    TptrDevice,
    TptrContext,
    TptrStream,
    TptrKernel,
    TptrMemory,
    get_device,
    get_context,
    synchronize,
)

# Re-export tensor utilities
from .tensor import (
    TptrTensor,
    TptrDType,
    dtype,
    zeros,
    ones,
    empty,
    full,
)

# Re-export dispatch utilities
from .dispatch import (
    DispatchRegistry,
    register_op,
    get_dispatch_table,
)

__all__ = [
    "Device", "MemoryAllocation", "CommandQueue", "Queue", "Kernel",
    "KernelConfig", "KernelHandle", "TptrError",
    "TptrDevice", "TptrContext", "TptrStream", "TptrKernel",
    "TptrMemory", "get_device", "get_context", "synchronize",
    "TptrTensor", "TptrDType", "dtype", "zeros", "ones", "empty", "full",
    "DispatchRegistry", "register_op", "get_dispatch_table",
]