# TPT Runtime / tptr — Layer 4

**Tensor Processing Technology — Runtime Layer**

## Overview

Layer 4 defines the TPT Runtime (tptr), the Rust-based runtime system that
manages GPU device resources, command execution, and memory allocation.

### Directory Structure

```
layer4_tptr/
├── Cargo.toml                  # Workspace root
├── README.md                   # This file
├── spec/
│   └── tptr_spec.md            — Runtime specification document
├── tptr-core/                  # Core runtime library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              — Crate root, re-exports
│       ├── error.rs            — Error handling framework
│       ├── memory/
│       │   ├── mod.rs          — Module exports
│       │   ├── types.rs        — Memory types (regions, permissions, handles)
│       │   └── allocator.rs    — Slab, buddy, and fallback allocators
│       ├── command/
│       │   ├── mod.rs          — Module exports
│       │   └── queue.rs        — Command queue and priority scheduler
│       ├── kernel/
│       │   ├── mod.rs          — Module exports
│       │   └── launch.rs       — Kernel config, argument buffer, launch handle
│       └── device/
│           ├── mod.rs          — Module exports
│           └── device.rs       — Device abstraction with backends
└── tptr-py/                    # Python bindings (PyO3)
    ├── Cargo.toml
    └── src/
        └── lib.rs              — PyO3 module exposing tptr API to Python
```

### Key Components

#### 1. GPU Memory Allocator
- **Slab allocator** — Fixed-size blocks for small, frequent allocations
- **Buddy allocator** — Power-of-two blocks for general-purpose allocation
- **Fallback allocator** — Raw linear device allocation for large requests
- RAII-based handle (`MemoryAllocation`) with automatic lifetime tracking

#### 2. Command Queue / Scheduler
- Priority-based queues (High, Normal, Low) with aging for starvation prevention
- Command types: Allocate, Free, Memcpy, Memset, LaunchKernel, Barrier, Events
- `CommandScheduler` manages multiple queues with round-robin dispatch
- Event-based synchronization between queues

#### 3. Kernel Launch Interface
- `KernelConfig` with grid/block dimensions and shared memory configuration
- `ArgumentBuffer` for serializing kernel arguments via bytemuck
- `KernelHandle` for tracking async kernel execution with spin-loop wait

#### 4. Device Abstraction
- `Device` struct unifying allocator, scheduler, and kernel operations
- `DeviceProperties` with capability metadata
- Backend enum: TPTNative, CUDA, ROCm, Metal, Simulated

#### 5. Error Handling Framework
- `TptrError` with structured error codes (E0001–E0099)
- Source location tracking via `#[track_caller]`
- Context metadata and error chaining
- `tptr_err!` macro for ergonomic error creation

#### 6. Python Bindings (PyO3)
- `tptr.Device` class — device management, memory allocation, queue creation
- `tptr.MemoryAllocation` — GPU memory handle with properties
- `tptr.CommandQueue` — queue handle for submission
- `tptr.Kernel` / `tptr.KernelConfig` / `tptr.KernelHandle` — kernel lifecycle
- `tptr.TptrError` — structured Python exception

### Building

#### Core library
```bash
cd layer4_tptr
cargo build -p tptr-core
cargo test -p tptr-core
```

#### Python bindings
```bash
cd layer4_tptr
cargo build -p tptr-py
```

### Usage (Python)
```python
import tptr

device = tptr.Device(0)
mem = device.allocate(4096)

kernel = device.create_kernel("my_kernel")
config = tptr.KernelConfig(grid=(16,1,1), block=(256,1,1))

info = device.info()
print(f"Device: {info['name']}, Memory: {info['total_memory']}")
```

### Thread Safety

- `Device` — `Send` + `Sync` (shared via `Arc`)
- `MemoryAllocation` — `Send` + `Sync` (reference-counted)
- `CommandQueue` — `Send` (channel-based)
- `KernelHandle` — `Send` (atomic state tracking)

### License

Apache License 2.0 (with Express Patent Grant)
