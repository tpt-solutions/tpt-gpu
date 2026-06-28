# Tutorial 8: GPU Primitives

**Estimated Time:** 60 minutes  
**Prerequisites:** Tutorial 4, linear algebra

---

## Introduction

Layer 5 provides optimized GPU primitives: GEMM, Attention, Conv2D. These kernels are written in TPTIR and can dispatch to vendor libraries (cuBLAS, rocBLAS, Metal) when available.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application                                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              TptpDevice                                   │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │  │
│  │  │  cuBLAS │  │ rocBLAS │  │  Metal  │  │  TPTIR  │    │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    TPT Runtime (Layer 4)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## GEMM (General Matrix Multiply)

### Operation

```
C = alpha * A * B + beta * C
```

Where:
- A: M x K matrix
- B: K x N matrix
- C: M x N matrix

### Rust API

```rust
use tptp::Gemm;

let gemm = Gemm::new(&device)?;
let c = gemm.execute(&a, &b, M, N, K)?;
```

### TPTIR Implementation

```tptir
func.func @gemm(
    %A: tensor<?x?xf32>,
    %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>,
    %M: index, %N: index, %K: index,
    %alpha: f32, %beta: f32
) attributes {tptir.kernel} {
    ^entry:
    // Tile over output dimensions
    for %i in 0..%M {
        for %j in 0..%N {
            // Accumulate dot product
            %acc = tptir.constant 0.0 : f32
            for %k in 0..%K {
                %a = tptir.tensor_load(%A, [%i, %k])
                %b = tptir.tensor_load(%B, [%k, %j])
                %prod = tptir.mulf(%a, %b)
                %acc = tptir.addf(%acc, %prod)
            }
            %c = tptir.tensor_load(%C, [%i, %j])
            %result = tptir.addf(
                tptir.mulf(%alpha, %acc),
                tptir.mulf(%beta, %c)
            )
            tptir.tensor_store(%result, %C, [%i, %j])
        }
    }
    tptir.return
}
```

---

## Attention

### Operation

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

### Rust API

```rust
use tptp::Attention;

let attention = Attention::new(&device)?;
let output = attention.execute(&q, &k, &v, Some(&mask))?;
```

### TPTIR Implementation Strategy

1. Flash Attention-style tiling over sequence dimension
2. Online softmax (rescaling)
3. Shared memory for Q, K, V tiles
4. Register accumulation for output

```tptir
func.func @attention(
    %Q: tensor<?x?xf32>,
    %K: tensor<?x?xf32>,
    %V: tensor<?x?xf32>,
    %scale: f32
) attributes {tptir.kernel} {
    ^entry:
    // Tile Q, K, V into shared memory blocks
    // Compute Q * K^T in tiles
    // Apply online softmax
    // Multiply by V
    tptir.return
}
```

---

## Conv2D

### Operation

```
Output = conv2d(Input, Filter, strides, padding)
```

### Rust API

```rust
use tptp::Conv2D;

let conv = Conv2d::new(&device)?
    .strides(1, 1)
    .padding(1, 1);
let output = conv.execute(&input, &filter)?;
```

### TPTIR Implementation Strategy

1. im2col + GEMM for large filters
2. Direct convolution with shared memory for small filters
3. Tiling over output spatial dimensions
4. Channel-level parallelism

---

## Vendor Library Integration

### Backend Selection Priority

1. **cuBLAS** (NVIDIA) — GEMM, Attention (via cuDNN)
2. **ROCm/MIOpen** (AMD) — GEMM via rocBLAS, Attention via MIOpen
3. **Metal Performance Shaders** (Apple) — GEMM, Attention via MPS
4. **TPTIR Fallback** — All primitives via TPTIR compilation

### Dispatch

```rust
pub enum VendorBackend {
    Cuda(CublasHandle),
    Rocm(RocblasHandle),
    Metal(MetalDevice),
    Tptir(TptirCompiler),
}

impl VendorBackend {
    pub fn gemm(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        match self {
            VendorBackend::Cuda(handle) => handle.gemm(a, b),
            VendorBackend::Rocm(handle) => handle.gemm(a, b),
            VendorBackend::Metal(device) => device.gemm(a, b),
            VendorBackend::Tptir(compiler) => compiler.compile_gemm(a, b),
        }
    }
}
```

---

## Example: Matrix Multiplication

```rust
use tptp::{Gemm, DType};

fn main() -> Result<()> {
    let device = TptpDevice::new(0)?;
    
    // Create matrices
    let a = device.randn(&[1024, 512], DType::Float32)?;
    let b = device.randn(&[512, 768], DType::Float32)?;
    
    // Execute GEMM
    let gemm = Gemm::new(&device)?;
    let c = gemm.execute(&a, &b, 1024, 512, 768)?;
    
    println!("Result shape: {:?}", c.shape());
    Ok(())
}
```

---

## Exercises

1. **GEMM Optimization**: Implement tiled GEMM with shared memory
2. **Attention**: Implement Flash Attention with online softmax
3. **Conv2D**: Implement im2col-based convolution

---

## Summary

- ✅ GEMM: General matrix multiply with vendor dispatch
- ✅ Attention: Scaled dot-product attention with tiling
- ✅ Conv2D: 2D convolution with im2col
- ✅ Vendor backend selection: cuBLAS, rocBLAS, Metal, TPTIR

**Next:** [Tutorial 9: Python API](09_python_api.md)
