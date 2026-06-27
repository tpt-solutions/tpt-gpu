"""
FFI bridge to the Rust tptr library.

This module provides direct access to the PyO3-backed tptr extension.
When the native extension is not available, a simulation fallback is provided.
"""

import os
import sys
import warnings
import importlib.util

# Try to import the native Rust extension
_native_ext = None

# First, try to find the native extension in the target directory
# The native extension is built by tptr-py in layer4_tptr
def _find_native_extension():
    """Find and load the native tptr extension module."""
    # Possible locations for the native extension
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "layer4_tptr", "target", "release"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "layer4_tptr", "target", "debug"),
    ]
    
    # Look for the native extension file
    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue
        
        # Look for tptr.pyd (Windows) or tptr.so (Linux/Mac)
        for filename in ["tptr.pyd", "tptr.so", "tptr.cpython-312-x86_64-linux-gnu.so"]:
            ext_path = os.path.join(base_path, filename)
            if os.path.exists(ext_path):
                try:
                    spec = importlib.util.spec_from_file_location("tptr_native", ext_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return module
                except Exception:
                    pass
    
    return None

_native_ext = _find_native_extension()

if _native_ext is not None:
    # Re-export all native types from the native extension
    Device = _native_ext.Device
    MemoryAllocation = _native_ext.MemoryAllocation
    CommandQueue = _native_ext.CommandQueue
    Kernel = _native_ext.Kernel
    KernelConfig = _native_ext.KernelConfig
    KernelHandle = _native_ext.KernelHandle
    TptrError = _native_ext.TptrError
else:
    # Simulation fallback for development/testing without native extension
    # Use a simple warning without stacklevel to avoid issues
    warnings.warn(
        "Native tptr extension not found. Using simulation fallback. "
        "Build with: cd layer4_tptr && cargo build -p tptr-py",
        RuntimeWarning,
    )
    from ._sim import (
        Device,
        MemoryAllocation,
        CommandQueue,
        Kernel,
        KernelConfig,
        KernelHandle,
        TptrError,
    )

