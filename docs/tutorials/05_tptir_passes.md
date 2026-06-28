# Tutorial 5: TPTIR Passes

**Estimated Time:** 40 minutes  
**Prerequisites:** Tutorial 4

---

## Introduction

TPTIR passes are optimization and transformation passes that operate on TPTIR to improve code quality, enable vectorization, and prepare for backend code generation.

### Pass Pipeline

```
Input TPTIR
    │
    ▼
┌─────────────────┐
│ Canonicalize    │  Normalize IR form
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DCE             │  Dead Code Elimination
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Constant Fold   │  Evaluate constant expressions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vectorize       │  Convert scalar to vector ops
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TensorLower     │  Lower tensor ops to loops
└────────┬────────┘
         │
         ▼
Output TPTIR
```

---

## Pass Reference

### Canonicalize Pass

Normalizes IR to a standard form for downstream passes.

**Before:**
```tptir
%a = tptir.addi(%x, %c0)    // Add zero
%b = tptir.muli(%a, %c1)    // Multiply by one
```

**After:**
```tptir
// Operations with identity operands are eliminated
// %a is replaced with %x, %b is replaced with %a
```

### Dead Code Elimination (DCE)

Removes operations whose results are never used.

**Before:**
```tptir
%unused = tptir.load(%ptr, %idx)  // Result never used
%used = tptir.addi(%x, %y)
tptir.return %used
```

**After:**
```tptir
%used = tptir.addi(%x, %y)
tptir.return %used
```

### Constant Folding

Evaluates constant expressions at compile time.

**Before:**
```tptir
%c4 = tptir.constant 4 : i32
%c5 = tptir.constant 5 : i32
%sum = tptir.addi(%c4, %c5)
```

**After:**
```tptir
%c9 = tptir.constant 9 : i32
```

### Vectorize Pass

Converts scalar operations to vector operations for SIMD execution.

**Before:**
```tptir
%a0 = tptir.load(%base, %i0)
%b0 = tptir.load(%base, %i1)
%c0 = tptir.addf(%a0, %b0)
```

**After:**
```tptir
%va = tptir.vector_load(%base, %idx) : vector<32xf32>
%vb = tptir.vector_load(%base2, %idx) : vector<32xf32>
%vc = tptir.vector_add(%va, %vb) : vector<32xf32>
```

### Tensor Lowering Pass

Lowers high-level tensor operations to explicit loops and memory operations.

**Before:**
```tptir
%C = tptir.contraction(%A, %B) : tensor<16x16xf32>
```

**After:**
```tptir
// Lowered to nested loops with load/store
for %i in 0..16 {
  for %j in 0..16 {
    for %k in 0..16 {
      %a = tptir.tensor_load(%A, [%i, %k])
      %b = tptir.tensor_load(%B, [%k, %j])
      %c = tptir.tensor_load(%C, [%i, %j])
      %prod = tptir.mulf(%a, %b)
      %sum = tptir.addf(%c, %prod)
      tptir.tensor_store(%sum, %C, [%i, %j])
    }
  }
}
```

---

## Pass Manager

Passes are registered and executed by the pass manager:

```rust
// From layer3_tptc/rust/src/passes.rs
use tptir_passes::{Canonicalize, DCE, ConstantFold, Vectorize, TensorLower};

let mut pm = PassManager::new();
pm.add_pass(Canonicalize::new());
pm.add_pass(DCE::new());
pm.add_pass(ConstantFold::new());
pm.add_pass(Vectorize::new());
pm.add_pass(TensorLower::new());

pm.run(&mut ir)?;
```

---

## Writing Custom Passes

```rust
use tptir_passes::{Pass, PassResult};
use tptir::{Operation, Block};

struct MyCustomPass;

impl Pass for MyCustomPass {
    fn name(&self) -> &str {
        "my-custom-pass"
    }
    
    fn run_on_block(&self, block: &mut Block) -> PassResult {
        for op in block.operations() {
            // Transform operations
            if let Some(new_op) = try_optimize(op) {
                block.replace(op, new_op);
            }
        }
        Ok(())
    }
}
```

---

## Pass Verification

Each pass includes verification to ensure IR validity:

```rust
impl Pass for DCE {
    fn verify(&self, block: &Block) -> Result<(), String> {
        // Ensure no dangling references
        for op in block.operations() {
            for result in op.results() {
                if !result.is_used() && !op.has_side_effects() {
                    return Err(format!("Dead operation: {}", op.name()));
                }
            }
        }
        Ok(())
    }
}
```

---

## Example: Optimization Pipeline

**Input:**
```tptir
func.func @example(%a: f32, %b: f32) -> f32 {
  %c0 = tptir.constant 0.0 : f32
  %c1 = tptir.constant 1.0 : f32
  %x = tptir.addf(%a, %c0)    // Identity add
  %y = tptir.mulf(%x, %c1)    // Identity mul
  %unused = tptir.addf(%b, %c0) // Dead code
  tptir.return %y
}
```

**After Canonicalize + DCE + Constant Fold:**
```tptir
func.func @example(%a: f32, %b: f32) -> f32 {
  tptir.return %a
}
```

---

## Exercises

1. **DCE Pass**: Write a pass that removes redundant loads
2. **Vectorize Pass**: Convert a scalar loop to vector operations
3. **Custom Pass**: Implement a pass that fuses multiply-add patterns into FMA

---

## Summary

- ✅ Canonicalize: Normalize IR form
- ✅ DCE: Remove dead code
- ✅ Constant Fold: Evaluate constants at compile time
- ✅ Vectorize: Convert scalar to vector ops
- ✅ Tensor Lower: Lower tensor ops to loops
- ✅ Pass manager for orchestration
- ✅ Custom pass development

**Next:** [Tutorial 6: Memory Management](06_memory_management.md)
