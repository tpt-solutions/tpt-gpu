# Tutorial 1: Introduction & Setup

**Estimated Time:** 30 minutes  
**Prerequisites:** Basic programming knowledge, Rust toolchain

---

## What is TPT GPU?

TPT GPU is an open-source, hardware-agnostic GPU compute platform designed from the ground up for AI-assisted development. Unlike CUDA, which evolved over 15 years, TPT GPU is built with modern tooling and AI-native design principles from day one.

### Key Features

1. **7-Layer Architecture**: Clean separation from hardware ISA to high-level language
2. **AI-Native Language**: TPT Script is designed for LLMs to reason about easily
3. **Hardware Agnostic**: Works across vendors (NVIDIA, AMD, Intel, custom silicon)
4. **Rust Core**: Memory-safe runtime and compiler
5. **Python Ecosystem**: Seamless PyTorch and JAX integration

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 7: TPT Script — AI-native GPU language (Rust)           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Framework Backends — PyTorch, JAX (Python + Rust)     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: TPT Primitives — GEMM, Attention, Conv2D             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: TPT Runtime — Memory, scheduling, kernel launch       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: TPTIR Compiler — MLIR-compatible IR (C++ & Rust)      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: TPT Drivers — OS interface (C, Rust, DriverKit)       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: TPT ISA — Hardware instruction set (SystemVerilog)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

1. **Rust Toolchain** (≥ 1.75)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update
```

2. **Git**
```bash
# Windows: Download from git-scm.com
# macOS: brew install git
# Linux: sudo apt install git
```

3. **C++ Build Tools** (for Layer 3)
```bash
# Windows: Visual Studio Build Tools
# macOS: xcode-select --install
# Linux: sudo apt install build-essential cmake
```

### Clone and Build

```bash
git clone https://github.com/tpt-gpu/tpt-gpu.git
cd tpt-gpu

# Build the TPT Script compiler (Layer 7)
cd layer7_tptb
cargo build --release -p tptb-cli

# Build the runtime (Layer 4)
cd ../layer4_tptr
cargo build -p tptr-core

# Build Python bindings (optional)
cargo build -p tptr-py

# Install Python framework (Layer 6)
cd ../layer6_framework
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check tpt CLI
cargo run -p tptb-cli -- --help

# Check Python bindings
python -c "import tptr; print('TPT GPU loaded successfully')"
```

---

## Your First Program

### Step 1: Create the Source File

Create `hello.tpts`:

```tpts
// hello.tpts — First TPT Script program
import tpt

@doc("Compute the ReLU activation function")
@requires_gpu(true)
fn relu(x: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.relu(x)
}
```

### Step 2: Type-Check

```bash
tpt check hello.tpts
```

Expected output:
```
✓ Type-checked hello.tpts (0 errors, 0 warnings)
```

### Step 3: Compile

```bash
tpt compile hello.tpts -o out/
```

This generates:
- `out/hello.rs` — Rust host code
- `out/hello.tptir` — GPU kernel in TPTIR

### Step 4: Inspect Generated Code

```bash
cat out/hello.tptir
```

Output:
```tptir
module {
  func.func @relu(%arg0: memref<*xf32>) -> memref<*xf32> {
    %0 = tptir.relu(%arg0) : memref<*xf32>
    return %0 : memref<*xf32>
  }
}
```

---

## Understanding the Components

### Layer 7: TPT Script
- **Types**: `Tensor[f32, 224, 224]`, `f32`, `i64`, `bool`
- **Functions**: Annotated with `@doc`, `@requires_gpu`, `@constraint`

### Layer 6: Python API
```python
import tptr
device = tptr.Device(0)
mem = device.allocate(4096)
```

### Layer 4: Runtime
- Memory allocators (slab, buddy, fallback)
- Command queues with priority scheduling

### Layer 3: TPTIR
- SSA form with typed operations
- Optimization passes (DCE, constant folding, vectorization)

### Layer 1: ISA
- 32-bit fixed-length instruction set
- 9-stage SIMT pipeline

---

## Project Structure

```
tpt-gpu/
├── layer1_isa/          # SystemVerilog ISA
├── layer2_driver/       # Kernel drivers
├── layer3_tptc/         # TPTIR compiler
├── layer4_tptr/         # GPU runtime (Rust)
├── layer5_tptp/         # GPU primitives
├── layer6_framework/    # Framework backends
├── layer7_tptb/         # TPT Script compiler
├── docs/                # Documentation
└── examples/            # Examples
```

---

## Next Steps

Now that you have TPT GPU installed:
1. **Tutorial 2**: TPT ISA Fundamentals
2. **Tutorial 6**: Memory Management
3. **Tutorial 11**: TPT Script Basics
4. **Tutorial 9**: Python API Basics

### Quick Reference

| Task | Command |
|------|---------|
| Type-check | `tpt check file.tpts` |
| Compile | `tpt compile file.tpts -o out/` |
| Inspect | `tpt inspect matmul` |
| List ops | `tpt ops` |

---

## Summary

- ✅ Learned about the 7-layer TPT GPU architecture
- ✅ Installed the toolchain (Rust, C++, Python)
- ✅ Written your first TPT Script program
- ✅ Compiled and inspected generated code

**Next:** [Tutorial 2: TPT ISA Fundamentals](02_isa_fundamentals.md)
