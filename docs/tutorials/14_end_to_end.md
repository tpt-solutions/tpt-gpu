# Tutorial 14: End-to-End Workflow

**Estimated Time:** 60 minutes  
**Prerequisites:** Tutorials 1-13

---

## Introduction

This tutorial walks through the complete compilation pipeline from TPT Script source to hardware execution.

### Pipeline Overview

```
TPT Script Source (.tpts)
         │
         ▼
┌─────────────────┐
│ Lexer           │  Tokenize source
├─────────────────┤
│ Parser          │  Build AST
├─────────────────┤
│ Type Checker    │  Validate types
├─────────────────┤
│ CodeGen         │  Generate Rust + TPTIR
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  Rust      TPTIR
    │         │
    ▼         ▼
  cargo     tptc
    │         │
    ▼         ▼
  binary    TPT ISA
              │
              ▼
           tptd driver
              │
              ▼
           GPU Hardware
```

---

## Step 1: Write TPT Script

```tpts
// vector_add.tpts
import tpt

@doc("Element-wise vector addition")
@requires_gpu(true)
@complexity("O(n)")
fn vector_add(a: Tensor[f32, *], b: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.add(a, b)
}

@doc("Run vector addition")
@requires_gpu(false)
fn run_add(n: i64) -> f32 {
    let a = tpt.randn([n], dtype=f32)
    let b = tpt.randn([n], dtype=f32)
    let c = vector_add(a, b)
    tpt.sync()
    return tpt.sum(c)
}
```

---

## Step 2: Type Check

```bash
tpt check vector_add.tpts
```

Output:
```
✓ Type-checked vector_add.tpts (0 errors, 0 warnings)
```

---

## Step 3: Compile

```bash
tpt compile vector_add.tpts -o out/
```

Generated files:
- `out/vector_add.rs` — Rust host code
- `out/vector_add.tptir` — GPU kernel

### Generated Rust

```rust
// out/vector_add.rs
use tptr::*;

pub fn run_add(n: i64) -> f32 {
    let a = randn(&[n as usize], DType::Float32);
    let b = randn(&[n as usize], DType::Float32);
    let c = vector_add(&a, &b);
    sync();
    sum(&c)
}
```

### Generated TPTIR

```tptir
// out/vector_add.tptir
module {
  func.func @vector_add(
    %a: memref<*xf32>,
    %b: memref<*xf32>
  ) -> memref<*xf32> attributes {tptir.kernel, tptir.block_size = 256 : i32} {
    ^entry:
      %c0 = tptir.constant 0 : i32
      %tid = tptir.get_thread_id : i32
      %idx = tptir.addi(%tid, %c0)
      %va = tptir.load(%a, %idx)
      %vb = tptir.load(%b, %idx)
      %vc = tptir.addf(%va, %vb)
      tptir.store(%vc, %c, %idx)
      tptir.return
  }
}
```

---

## Step 4: Optimize TPTIR

```bash
tptc opt vector_add.tptir -o optimized.tptir \
    --passes=canonicalize,dce,constfold,vectorize
```

---

## Step 5: Generate ISA

```bash
tptc compile optimized.tptir -o vector_add.isa --target=tptisa
```

---

## Step 6: Build Host Code

```bash
cd out/
cargo build --release
```

---

## Step 7: Execute

```bash
./target/release/vector_add
```

---

## Inspecting Intermediate Representations

### AST

```bash
tpt ast vector_add.tpts
```

### Typed AST

```bash
tpt typed-ast vector_add.tpts
```

### TPTIR with Debug Info

```bash
tptc ir vector_add.tptir --debug
```

---

## Debugging

### Enable Debug Output

```bash
RUST_LOG=debug tpt check vector_add.tpts
```

### View Generated Code

```bash
tpt compile vector_add.tpts -o out/ --emit-ir
ls out/
# vector_add.rs  vector_add.tptir  vector_add.isa
```

---

## Performance Profiling

```bash
# Profile kernel execution
tpt profile vector_add.tpts --iterations=100

# Output:
# Kernel: vector_add
#   Cycles: 12345
#   Instructions: 6789
#   Memory accesses: 1024
#   Cache hit rate: 98.5%
```

---

## Complete Example: Transformer Block

```tpts
import tpt

type Batch = Tensor[f32, batch, seq, d_model]

@doc("Transformer block")
@requires_gpu(true)
@requires_tensor_cores(true)
@complexity("O(batch * seq^2 * d_model)")
fn transformer_block(
    x: Batch,
    attn_w: Tensor[f32, 3, d_model, d_model],
    ffn_w1: Tensor[f32, d_ff, d_model],
    ffn_w2: Tensor[f32, d_model, d_ff],
) -> Batch {
    // Self-attention
    let q = tpt.matmul(x, tpt.transpose(attn_w[0]))
    let k = tpt.matmul(x, tpt.transpose(attn_w[1]))
    let v = tpt.matmul(x, tpt.transpose(attn_w[2]))
    let attn_out = tpt.attention(q, k, v)
    let residual1 = tpt.add(x, attn_out)
    let normed1 = tpt.layer_norm(residual1)
    
    // Feed-forward
    let ff = tpt.matmul(normed1, tpt.transpose(ffn_w1))
    let ff = tpt.gelu(ff)
    let ff = tpt.matmul(ff, tpt.transpose(ffn_w2))
    let residual2 = tpt.add(normed1, ff)
    return tpt.layer_norm(residual2)
}
```

---

## Exercises

1. **Pipeline**: Run the complete pipeline for a matrix multiplication kernel
2. **Optimization**: Compare performance with and without optimization passes
3. **Debugging**: Use debug output to understand code generation

---

## Summary

- ✅ Lexer → Parser → Type Checker → CodeGen pipeline
- ✅ Dual output: Rust (host) + TPTIR (device)
- ✅ TPTIR optimization passes
- ✅ ISA code generation
- ✅ Profiling and debugging tools

**Next:** [Tutorial 15: Building a Model](15_building_a_model.md)
