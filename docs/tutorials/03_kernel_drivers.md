# Tutorial 3: Kernel Drivers

**Estimated Time:** 40 minutes  
**Prerequisites:** Tutorial 2, C/Rust basics

---

## Introduction

Layer 2 provides the kernel driver interface between the TPT GPU hardware (Layer 1) and the software stack (Layers 3-7). It handles device initialization, memory management, and command submission.

### Driver Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Userspace Applications                        │
├─────────────────────────────────────────────────────────────────┤
│  Rust Userspace Library (tptd)  │  C API (libtptd.so)           │
├─────────────────────────────────────────────────────────────────┤
│                    Kernel Driver                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Linux DRM │  │  Windows WDM│  │  macOS DEXT │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                    Hardware (PCIe)                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Linux DRM Driver

### Building the Driver

```bash
cd layer2_driver/linux
make KDIR=/lib/modules/$(uname -r)/build
sudo insmod tpt_gpu.ko
```

### IOCTL Interface

| IOCTL | Description |
|-------|-------------|
| `TPT_IOCTL_GEM_CREATE` | Allocate GEM buffer (VRAM/GTT) |
| `TPT_IOCTL_GEM_FREE` | Release GEM handle |
| `TPT_IOCTL_GEM_INFO` | Query size and GPU address |
| `TPT_IOCTL_GEM_MMAP` | Get mmap offset for CPU mapping |
| `TPT_IOCTL_SUBMIT` | Submit command buffer |
| `TPT_IOCTL_WAIT_FENCE` | Block until fence completes |
| `TPT_IOCTL_QUERY_INFO` | Query device properties |

---

## Windows WDM Driver

### Building

```bash
msbuild tpt_gpu.vcxproj /p:Configuration=Release;Platform=x64
```

Device appears as `\\\\.\\TPT_GPU0` after installation.

---

## macOS DriverKit Extension

### Building

Open `tpt_gpu.xcodeproj` in Xcode 14+ and build for macOS 12+.

### Required Entitlements

- `com.apple.developer.driverkit`
- `com.apple.developer.driverkit.transport.pci`

---

## Rust Userspace Library (tptd)

### Building

```bash
cd layer2_driver/rust
cargo build --release
```

### Rust API

```rust
use tptd::{Device, BufferFlags, CmdBuf};
use std::time::Duration;

let dev = Device::open("/dev/dri/card0")?;
let mut buf = dev.alloc(4 * 1024 * 1024, BufferFlags::VRAM)?;
println!("GPU addr: 0x{:016x}", buf.gpu_addr());

let mut cmdbuf = CmdBuf::new(dev.fd.clone(), 4096)?;
cmdbuf.launch(buf.gpu_addr(), (64, 1, 1), (32, 1, 1))?;

let fence = dev.submit(&cmdbuf)?;
fence.wait(Duration::from_secs(5))?;
```

### C API

```c
#include <tpt_driver.h>

tpt_device_t *dev = tpt_open("/dev/dri/card0");
tpt_buffer_t *buf = tpt_buffer_alloc(dev, 4 * 1024 * 1024, TPT_BUF_FLAG_VRAM);
void *ptr = tpt_buffer_map(buf);

tpt_fence_t *f = tpt_submit(dev, cmdbuf, 0, cmd_size);
tpt_fence_wait(f, UINT64_MAX);
```

---

## FFI Boundary Design

1. **Opaque handles**: Forward-declared structs
2. **Ownership pairs**: Every alloc has a free
3. **No callbacks**: Fully synchronous
4. **Error codes**: 0 = OK, negative = error
5. **Thread safety**: Device is Send + Sync

---

## PCI Device ID

| Field | Value |
|-------|-------|
| Vendor | 0x1A2E |
| Device | 0x0001 (rev A) |
| Class | 0x030200 (3D display) |

---

## Exercises

1. **Buffer Allocation**: Allocate a 16 MB VRAM buffer and print its GPU address
2. **Command Submission**: Create a command buffer that launches a simple kernel
3. **Fence Synchronization**: Submit multiple command buffers and wait for completion

---

## Summary

- ✅ Linux DRM driver at `/dev/dri/card*`
- ✅ Windows WDM driver at `\\\\.\\TPT_GPU0`
- ✅ macOS DriverKit extension
- ✅ Rust userspace library with C API
- ✅ IOCTL interface for buffer management

**Next:** [Tutorial 4: TPTIR Overview](04_tptir_overview.md)
