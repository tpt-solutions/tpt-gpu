# Tutorial 6: Memory Management

**Estimated Time:** 50 minutes  
**Prerequisites:** Tutorial 1, Rust basics

---

## Introduction

Layer 4 (TPT Runtime) provides GPU memory management through a three-tier allocator system.

### Memory Hierarchy

| Memory Type | Scope | Latency | Size |
|-------------|-------|---------|------|
| Global | All threads | ~400 cycles | GBs |
| Shared | Thread block | ~30 cycles | 48 KB/block |
| Local | Single thread | ~30 cycles | 512 KB |
| Constant | All threads (RO) | Cached | 64 KB |

---

## Three-Tier Allocator

### Architecture

```rust
pub enum Allocator {
    Slab(SlabAllocator),      // Fast path: < 4 KB
    Buddy(BuddyAllocator),    // Medium: 4 KB - 1 MB
    Fallback(FallbackAllocator), // Large: > 1 MB
}
```

### Slab Allocator

Fixed-size block allocator for small, frequent allocations.

```rust
pub struct SlabAllocator {
    block_size: usize,
    free_list: Vec<*mut u8>,
    slabs: Vec<Slab>,
}

impl SlabAllocator {
    pub fn allocate(&mut self) -> Result<MemoryAllocation> {
        if let Some(ptr) = self.free_list.pop() {
            return Ok(MemoryAllocation::new(ptr, self.block_size));
        }
        let slab = Slab::new(self.block_size * 64)?;
        let ptr = slab.first_block();
        self.slabs.push(slab);
        Ok(MemoryAllocation::new(ptr, self.block_size))
    }
}
```

**Characteristics:** O(1) allocation, no fragmentation, ideal for < 4 KB

### Buddy Allocator

Power-of-two block allocator for medium-sized allocations.

```rust
pub struct BuddyAllocator {
    min_block_size: usize,
    max_block_size: usize,
    free_lists: Vec<Vec<*mut u8>>,
}

impl BuddyAllocator {
    pub fn allocate(&mut self, size: usize) -> Result<MemoryAllocation> {
        let order = size.next_power_of_two().trailing_zeros() as usize;
        for i in order..self.free_lists.len() {
            if let Some(block) = self.free_lists[i].pop() {
                self.split_to_order(block, i, order);
                return Ok(MemoryAllocation::new(block, size));
            }
        }
        Err(AllocationError::OutOfMemory)
    }
}
```

**Characteristics:** O(log n), low external fragmentation, up to 2x internal fragmentation

### Fallback Allocator

Raw linear allocator for large allocations.

```rust
pub struct FallbackAllocator {
    base: *mut u8,
    size: usize,
    offset: usize,
}

impl FallbackAllocator {
    pub fn allocate(&mut self, size: usize) -> Result<MemoryAllocation> {
        if self.offset + size > self.size {
            return Err(AllocationError::OutOfMemory);
        }
        let ptr = unsafe { self.base.add(self.offset) };
        self.offset += size;
        Ok(MemoryAllocation::new(ptr, size))
    }
}
```

**Characteristics:** O(1) allocation, no deallocation, ideal for large allocations

---

## Memory Allocation API

```rust
use tptr_core::memory::{MemoryAllocation, MemoryFlags};

// Allocate GPU memory
let alloc = device.allocate(4096)?;

// Allocate with flags
let alloc = device.allocate_with_flags(
    4096,
    MemoryFlags::READ_ONLY | MemoryFlags::COHERENT,
)?;

// Get device pointer
let dev_ptr = alloc.device_ptr();

// Memory is automatically freed when alloc goes out of scope
```

### Memory Flags

```rust
bitflags! {
    pub struct MemoryFlags: u32 {
        const READ_ONLY = 0x01;
        const WRITE_ONLY = 0x02;
        const READ_WRITE = 0x03;
        const COHERENT = 0x04;
        const UNCACHED = 0x08;
        const COMBINED = 0x10;
    }
}
```

---

## Memory Operations

```rust
// Host to device
device.memcpy_h2d(device_ptr, host_ptr, size)?;

// Device to host
device.memcpy_d2h(host_ptr, device_ptr, size)?;

// Device to device
device.memcpy_d2d(dest_ptr, src_ptr, size)?;

// Fill memory
device.memset(device_ptr, 0, size)?;
```

---

## RAII Memory Management

```rust
pub struct MemoryAllocation {
    ptr: *mut u8,
    size: usize,
    device: Arc<Device>,
}

impl Drop for MemoryAllocation {
    fn drop(&mut self) {
        self.device.free(self.ptr, self.size);
    }
}

// Memory freed automatically when scope ends
{
    let alloc = device.allocate(4096)?;
    // Use alloc...
} // alloc is freed here
```

---

## Example: Matrix Allocation

```rust
fn allocate_matrices(m: usize, n: usize) -> Result<(MemoryAllocation, MemoryAllocation, MemoryAllocation)> {
    let size = m * n * std::mem::size_of::<f32>();
    let a = device.allocate(size)?;
    let b = device.allocate(size)?;
    let c = device.allocate(size)?;
    Ok((a, b, c))
}
```

---

## Performance Tips

1. **Use appropriate allocator tier**: Small = slab, large = fallback
2. **Minimize allocations**: Reuse buffers when possible
3. **Align allocations**: Align to cache lines (64 bytes)
4. **Use coherent memory**: For CPU-GPU shared data
5. **Batch allocations**: Allocate multiple buffers at once

---

## Exercises

1. **Custom Allocator**: Implement a pool allocator for fixed-size objects
2. **Memory Pool**: Create a memory pool that pre-allocates a large buffer
3. **Allocation Tracking**: Add tracking to detect memory leaks

---

## Summary

- ✅ Three-tier allocator: Slab, Buddy, Fallback
- ✅ Memory hierarchy: Global, Shared, Local, Constant
- ✅ RAII-based memory management
- ✅ Memory flags for access patterns

**Next:** [Tutorial 7: Kernel Scheduling](07_kernel_scheduling.md)
