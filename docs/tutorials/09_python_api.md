# Tutorial 9: Python API

**Estimated Time:** 40 minutes  
**Prerequisites:** Tutorial 1, Python basics

---

## Introduction

The Python API provides access to TPT GPU through PyO3 bindings, enabling rapid prototyping and integration with the Python ML ecosystem.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Application                            │
├─────────────────────────────────────────────────────────────────┤
│  import tptr                                                     │
│  device = tptr.Device(0)                                        │
├─────────────────────────────────────────────────────────────────┤
│                    PyO3 Bindings                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Rust Core (tptr-core)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
cd layer4_tptr
cargo build -p tptr-py

cd layer6_framework
pip install -e ".[dev]"
```

Verify:
```python
import tptr
print(tptr.__version__)
```

---

## Device Management

```python
import tptr

# Create device
device = tptr.Device(0)

# Query device info
info = device.info()
print(f"Device: {info['name']}")
print(f"Memory: {info['total_memory'] / (1024**3):.1f} GB")

# Device context manager
with tptr.device_context(0) as dev:
    mem = dev.allocate(4096)
    # Use device...
# Device released automatically
```

---

## Memory Allocation

```python
import tptr

device = tptr.Device(0)

# Allocate GPU memory
mem = device.allocate(4096)
print(f"Handle: {mem.handle}")
print(f"Size: {mem.size}")
print(f"Device Ptr: 0x{mem.device_ptr:x}")

# Allocate with flags
mem = device.allocate(4096, flags=tptr.MEM_FLAG_READ_ONLY)

# Memory freed automatically when mem goes out of scope
```

---

## Kernel Operations

```python
import tptr

device = tptr.Device(0)

# Create kernel
kernel = device.create_kernel("matmul")

# Configure kernel
config = tptr.KernelConfig(
    grid=(64, 1, 1),
    block=(256, 1, 1)
)

# Allocate arguments
a = device.allocate(1024 * 1024 * 4)  # 4 MB
b = device.allocate(1024 * 1024 * 4)
c = device.allocate(1024 * 1024 * 4)

# Launch kernel
handle = kernel.launch(config, [a, b, c])

# Wait for completion
handle.wait()
```

---

## Stream Operations

```python
import tptr

device = tptr.Device(0)

# Create stream
stream = device.create_stream("normal")

# Submit commands to stream
stream.memcpy_h2d(dest, src, size)
stream.launch_kernel(kernel, config, args)
stream.memcpy_d2h(dest, src, size)

# Synchronize stream
stream.synchronize()

# Stream with context manager
with device.create_stream("high") as stream:
    stream.launch_kernel(kernel, config, args)
# Stream synchronized on exit
```

---

## Error Handling

```python
import tptr

try:
    device = tptr.Device(0)
    mem = device.allocate(1024**4)  # 1 TB - will fail
except tptr.TptrError as e:
    print(f"Error: {e.code} - {e.message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Example: Vector Addition

```python
import tptr
import numpy as np

def vector_add():
    device = tptr.Device(0)
    
    # Create data
    n = 1024 * 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    
    # Allocate GPU memory
    d_a = device.allocate(a.nbytes)
    d_b = device.allocate(b.nbytes)
    d_c = device.allocate(c.nbytes)
    
    # Copy to GPU
    d_a.copy_from(a.ctypes.data, a.nbytes)
    d_b.copy_from(b.ctypes.data, b.nbytes)
    
    # Launch kernel
    kernel = device.create_kernel("vector_add")
    config = tptr.KernelConfig(grid=(n // 256, 1, 1), block=(256, 1, 1))
    kernel.launch(config, [d_a, d_b, d_c, n]).wait()
    
    # Copy result back
    d_c.copy_to(c.ctypes.data, c.nbytes)
    
    return c

result = vector_add()
print(f"Result: {result[:5]}")
```

---

## Exercises

1. **Device Info**: Write a script that prints all device properties
2. **Memory Benchmark**: Measure allocation and copy performance
3. **Kernel Launch**: Implement a simple element-wise operation

---

## Summary

- ✅ Device creation and management
- ✅ Memory allocation with automatic cleanup
- ✅ Kernel creation and launch
- ✅ Stream-based command submission
- ✅ Error handling with Python exceptions

**Next:** [Tutorial 10: PyTorch Integration](10_pytorch_integration.md)
