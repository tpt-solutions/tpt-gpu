# Tutorial 16: Performance Tuning

**Estimated Time:** 60 minutes  
**Prerequisites:** Tutorial 14

---

## Introduction

This tutorial covers profiling and optimization strategies for TPT GPU kernels.

---

## Profiling Tools

### Kernel Profiling

```bash
# Profile kernel execution
tpt profile kernel.tpts --iterations=100

# Output:
# Kernel: vector_add
#   Cycles: 12345
#   Instructions: 6789
#   Memory accesses: 1024
#   Cache hit rate: 98.5%
#   Occupancy: 75%
```

### Memory Profiling

```bash
# Profile memory usage
tpt profile memory.tpts --track-allocations

# Output:
# Peak memory: 1.5 GB
# Allocations: 1234
# Deallocations: 1234
# Leaks: 0
```

---

## Optimization Strategies

### 1. Memory Coalescing

```tpts
// Bad: Strided access
@requires_gpu(true)
fn bad_access(x: Tensor[f32, N, M]) -> Tensor[f32, N, M] {
    for i in 0..N {
        for j in 0..M {
            let val = x[i, j]  // Strided by M
        }
    }
}

// Good: Coalesced access
@requires_gpu(true)
fn good_access(x: Tensor[f32, N, M]) -> Tensor[f32, N, M] {
    let tid = tpt.thread_idx_x()
    let stride = tpt.block_dim_x()
    for i in tid..N..stride {  // Consecutive threads access consecutive memory
        for j in 0..M {
            let val = x[i, j]
        }
    }
}
```

### 2. Shared Memory Tiling

```tpts
@requires_gpu(true)
@shared_mem(16384)
fn tiled_matmul(
    a: Tensor[f32, M, K],
    b: Tensor[f32, K, N],
    c: Tensor[f32, M, N],
) {
    let tile = tpt.shared_memory[f32, 32, 32]
    
    let row = tpt.block_idx_y() * 32 + tpt.thread_idx_y()
    let col = tpt.block_idx_x() * 32 + tpt.thread_idx_x()
    
    let mut acc = 0.0
    for k_tile in 0..K/32 {
        // Load tile into shared memory
        tile[tpt.thread_idx_y(), tpt.thread_idx_x()] = a[row, k_tile * 32 + tpt.thread_idx_x()]
        tpt.barrier()
        
        // Compute partial result
        for k in 0..32 {
            acc = acc + tile[tpt.thread_idx_y(), k] * b[k_tile * 32 + k, col]
        }
        tpt.barrier()
    }
    c[row, col] = acc
}
```

### 3. Kernel Fusion

```tpts
// Bad: Multiple kernel launches
@requires_gpu(true)
fn relu(x: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.relu(x)
}

@requires_gpu(true)
fn add_bias(x: Tensor[f32, *], bias: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.add(x, bias)
}

// Good: Fused kernel
@requires_gpu(true)
fn fused_relu_bias(x: Tensor[f32, *], bias: Tensor[f32, *]) -> Tensor[f32, *] {
    let h = tpt.add(x, bias)
    return tpt.relu(h)
}
```

### 4. Vectorization

```tpts
@requires_gpu(true)
@block_size(256)
fn vectorized_add(a: Tensor[f32, *], b: Tensor[f32, *]) -> Tensor[f32, *] {
    let tid = tpt.thread_idx_x()
    let stride = tpt.block_dim_x()
    
    // Process 4 elements per thread
    for i in (tid * 4)..N..(stride * 4) {
        let va = tpt.vector_load(a, i)  // Load 4 floats
        let vb = tpt.vector_load(b, i)
        let vc = tpt.vector_add(va, vb)
        tpt.vector_store(vc, c, i)
    }
}
```

---

## Performance Counters

| Counter | Description |
|---------|-------------|
| `INST_RETIRED` | Total instructions retired |
| `CORE_CYCLES` | Total core cycles |
| `L1D_MISSES` | L1 data cache misses |
| `L1I_MISSES` | L1 instruction cache misses |
| `BRANCH_MISPRED` | Branch mispredictions |
| `WARP_STALLS` | Cycles any warp is stalled |

---

## Optimization Checklist

1. **Memory**: Coalesced access, shared memory tiling
2. **Compute**: Kernel fusion, vectorization
3. **Occupancy**: Balance threads per block, registers per thread
4. **Latency**: Overlap compute and memory with streams

---

## Exercises

1. **Profile**: Profile a matrix multiplication kernel
2. **Optimize**: Apply tiling to improve cache performance
3. **Fuse**: Fuse multiple element-wise operations

---

## Summary

- ✅ Profiling tools for kernels and memory
- ✅ Memory coalescing for efficient access
- ✅ Shared memory tiling for data reuse
- ✅ Kernel fusion to reduce launch overhead
- ✅ Vectorization for SIMD execution

**Next:** [Tutorial 17: Distributed Computing](17_distributed_computing.md)
