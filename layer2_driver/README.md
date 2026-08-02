# Layer 2 — TPT GPU Driver (`tptd`)

Platform-specific kernel drivers and a Rust userspace library that form the
hardware abstraction layer between the TPT ISA (Layer 1) and the compiler
stack (Layer 3).

---

## Directory Layout

```
layer2_driver/
├── include/
│   └── tpt_driver.h          Shared C ABI: ioctl structs, error codes, C API
├── linux/
│   ├── Kbuild                Kernel build integration (CONFIG_DRM_TPT_GPU)
│   ├── Makefile              Out-of-tree build (make KDIR=...)
│   └── src/
│       ├── lib.rs            Module entry, PCI probe/remove, DRM driver vtable
│       ├── device.rs         Device state, submit ioctl, wait-fence, query
│       ├── gem.rs            GEM buffer objects, mmap, VRAM allocator
│       └── regs.rs           MMIO register map (BAR0 / BAR2)
├── windows/
│   ├── tpt_wdm.h             Internal WDM types and helpers
│   └── tpt_wdm.c             WDM driver: DriverEntry, PnP, ISR, DPC, ioctls
├── macos/
│   ├── TPTDriver.h           DriverKit IOService + IOUserClient declarations
│   ├── TPTDriver.cpp         DriverKit implementation
│   └── Info.plist            DEXT bundle metadata + PCI matching
└── rust/  (moved to crates/tpt-gpu-driver-daemon)
```

---

## Kernel Driver (Linux)

**Requirements:** Linux kernel ≥ 6.1, `CONFIG_RUST=y`, `CONFIG_DRM=y/m`.

```bash
# Out-of-tree build
cd layer2_driver/linux
make KDIR=/lib/modules/$(uname -r)/build

# Load
sudo insmod tpt_gpu.ko

# Unload
sudo rmmod tpt_gpu
```

The driver registers under `/dev/dri/card*` via DRM and exposes:

| ioctl | Description |
|---|---|
| `TPT_IOCTL_GEM_CREATE` | Allocate a GEM buffer (VRAM or GTT) |
| `TPT_IOCTL_GEM_FREE`   | Release a GEM handle |
| `TPT_IOCTL_GEM_INFO`   | Query size and GPU virtual address |
| `TPT_IOCTL_GEM_MMAP`   | Get mmap offset for CPU mapping |
| `TPT_IOCTL_SUBMIT`     | Submit a command buffer; returns fence seqno |
| `TPT_IOCTL_WAIT_FENCE` | Block until a seqno completes |
| `TPT_IOCTL_QUERY_INFO` | Query VRAM size, warp count, driver version |

---

## Windows WDM Driver

Build with WDK 10.0.22621.0 (Visual Studio 2022):

```
msbuild tpt_gpu.vcxproj /p:Configuration=Release;Platform=x64
```

Sign with an EV code-signing certificate and Microsoft cross-certificate.
Device appears as `\\.\TPT_GPU0` after installation.

---

## macOS DriverKit Extension

Build with Xcode 14+ targeting macOS 12+.  Requires entitlements:

- `com.apple.developer.driverkit`
- `com.apple.developer.driverkit.transport.pci`

PCI matching in `Info.plist` targets vendor `0x1A2E`, device `0x0001`.

---

## Rust Userspace Library (`tptd`)

The Rust userspace library now lives in `crates/tpt-gpu-driver-daemon`.

```bash
cd crates/tpt-gpu-driver-daemon
cargo build --release

# Run the daemon (Listens on /run/tptd.sock, JSON requests from layer4 clients)
sudo ./target/release/tpt-gpu-daemon --device 0000:03:00.0
```

### Rust API

```rust
use tptd::{Device, BufferFlags, CmdBuf};
use std::time::Duration;

let dev = Device::open("/dev/dri/card0")?;

// Allocate a 4 MiB VRAM buffer
let mut buf = dev.alloc(4 * 1024 * 1024, BufferFlags::VRAM)?;
println!("GPU addr: 0x{:016x}", buf.gpu_addr());

// Build a command buffer and launch a kernel
let mut cmdbuf = CmdBuf::new(dev.fd.clone(), 4096)?;
cmdbuf.launch(buf.gpu_addr(), (64, 1, 1), (32, 1, 1))?;

let fence = dev.submit(&cmdbuf)?;
fence.wait(Duration::from_secs(5))?;
```

### C API (via FFI)

Link against `libtptd.so` (Linux) / `tptd.dll` (Windows) and include
`include/tpt_driver.h`:

```c
tpt_device_t *dev = tpt_open("/dev/dri/card0");
tpt_buffer_t *buf = tpt_buffer_alloc(dev, 4 * 1024 * 1024, TPT_BUF_FLAG_VRAM);
void         *ptr = tpt_buffer_map(buf);

/* ... write commands ... */

tpt_fence_t  *f = tpt_submit(dev, cmdbuf, 0, cmd_size);
tpt_fence_wait(f, UINT64_MAX);
tpt_fence_free(f);
tpt_buffer_free(buf);
tpt_close(dev);
```

---

## FFI Boundary Design

The FFI boundary (`ffi.rs` ↔ `tpt_driver.h`) follows these invariants:

1. **Opaque handles** — `tpt_device_t`, `tpt_buffer_t`, `tpt_fence_t` are
   forward-declared `struct` types; callers hold pointers, never dereference.
2. **Ownership via pairs** — every `tpt_*_alloc` has a `tpt_*_free`;
   callers are responsible for calling free exactly once.
3. **No callbacks** — the C API is fully synchronous; async notification
   is via fence wait with timeout.
4. **Error codes** — all fallible C functions return `int` (0 = OK, negative
   = error), matching the `TPT_ERR_*` constants in `tpt_driver.h`.
5. **Thread safety** — `Device` is `Send + Sync`; multiple threads may
   call `tpt_submit` concurrently; the kernel serialises ring writes.

---

## PCI Device ID

| Field  | Value  |
|--------|--------|
| Vendor | 0x1A2E |
| Device | 0x0001 (rev A prototype) |
| Device | 0x0002 (rev B, planned) |
| Class  | 0x030200 (Display controller, 3D) |
