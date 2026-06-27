# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Architecture Overview

TPT GPU is a hardware-agnostic, full-stack GPU compute platform organized into 7 independent layers. Each layer has its own build system, spec, and workspace. They communicate through well-defined FFI/API boundaries — not shared source.

```
layer1_isa/      SystemVerilog ISA — 32-bit fixed-length, 9-stage SIMT pipeline
layer2_tptd/     Kernel drivers — Linux DRM (Rust for Linux), Windows WDM (C), macOS DriverKit (C)
layer3_tptc/     TPTIR compiler — MLIR-compatible dialect (C++) + parallel Rust port
layer4_tptr/     GPU runtime — allocator, scheduler, kernel launch (Rust), PyO3 Python bindings
layer5_tptp/     GPU primitives — GEMM, Attention, Conv2D (TPTIR kernels + Rust host wrappers)
layer6_tptf/     Framework backends — PyTorch dispatch, JAX integration (Python + Rust)
layer7_tptb/     TPT Script compiler — lexer → parser → type checker → codegen (Rust)
```

The primary development direction flows **downward**: layer7 TPT Script compiles to TPTIR (layer3), which the layer3 backend lowers to TPT ISA (layer1) via the layer4 runtime dispatch.

---

## Build & Test Commands

### Rust layers (4, 5, 7) — Cargo workspaces

```bash
# Layer 4 — Runtime
cd layer4_tptr
cargo build -p tptr-core
cargo test -p tptr-core
cargo test -p tptr-core -- test_name    # Single test

# Layer 5 — Primitives
cd layer5_tptp
cargo build --features sim              # Simulation mode (no hardware)
cargo test

# Layer 7 — TPT Script compiler
cd layer7_tptb
cargo build -p tptb-core
cargo test -p tptb-core
cargo test -p tptb-core -- test_name    # Single test
```

### Layer 3 — TPTIR Compiler

```bash
# C++ compiler stack
cd layer3_tptc
cmake -B build && cmake --build build
ctest --test-dir build

# Rust port
cd layer3_tptc/rust
cargo test
```

### Layer 2 — Drivers

```bash
# Linux (Rust for Linux kernel module)
cd layer2_tptd/linux
make KDIR=/lib/modules/$(uname -r)/build

# Rust userspace daemon
cd layer2_tptd/rust
cargo build --release
```

### Layer 1 — ISA Simulation

```bash
cd layer1_isa/sim
iverilog -g2012 -o sim.vvp ../rtl/*.sv tpt_tb.sv
vvp sim.vvp
python tpt_assemble.py programs/simple_add.asm
```

### Layer 6 — Framework Backends

```bash
cd layer6_tptf
pip install -e ".[dev]"
pytest tests/
```

---

## Layer 7: TPT Script Compiler (active development)

The compiler pipeline in `layer7_tptb/tptb-core/src/`:

```
lexer.rs → parser.rs → ast.rs → semantic/ → codegen/
```

- **`lexer.rs`** — Tokenizer producing `Token` / `Span`
- **`parser.rs`** — Recursive-descent parser → `Program` AST
- **`ast.rs`** — All AST node types (`Item`, `FunctionDecl`, `Expr`, `Type`, etc.)
- **`semantic/`** — Type checker (`mod.rs`), constraint evaluator, metadata extractor, builtin registry
- **`codegen/`** — Two backends:
  - `rust_emit.rs` — Non-GPU functions → Rust source; rewrites `tpt.xxx(args)` → `tptr::xxx(args)`
  - `tptir_emit.rs` — `@requires_gpu(true)` functions → TPTIR text for the layer3 compiler

**Key API:**
```rust
compile_str(source)              // lex + parse only → Program
type_check(&program)             // → TypeChecker { errors, type_map }
emit(&program)                   // → CodegenOutput { rust_source, tptir_source }
compile_full(source)             // full pipeline → (TypeChecker, CodegenOutput)
```

**Parser quirk:** `tpt.relu(x)` is parsed as `ExprKind::MethodCall { expr: Ident("tpt"), method: "relu" }`, NOT as `Call(FieldAccess)`. Both the Rust and TPTIR emitters handle this pattern explicitly.

**Annotations** on functions (`@requires_gpu`, `@constraint`, `@doc`, etc.) are extracted by `semantic/metadata.rs` into `FunctionMeta`. Constraints are evaluated at compile time via `semantic/constraints.rs`.

---

## Layer 3: TPTIR Integration

TPTIR is an SSA-based, MLIR-compatible IR. The Rust port lives in `layer3_tptc/rust/src/`:

- **`ir.rs`** — Core IR types: `Type`, `Operation`, `Block`, `Region`
- **`passes.rs`** — Optimization passes (DCE, constant fold, vectorize)
- **`lib.rs`** — `compile_native(source, target)` where target is `"tptisa"` or `"llvmir"`

The TPTIR text format uses `^label:` blocks. TPTIR emitted by layer7 feeds directly into `compile_native`.

---

## Layer 4: Runtime Architecture

`layer4_tptr/tptr-core/src/` modules:
- **`memory/allocator.rs`** — Three-tier: Slab (fast path) → Buddy (medium) → Fallback (system)
- **`command/queue.rs`** — Priority queue scheduler with aging to prevent starvation
- **`kernel/launch.rs`** — `KernelConfig`, `ArgumentBuffer`, `KernelHandle`
- **`error.rs`** — `TptrError` with structured error codes for Python surface

Python bindings (`tptr-py`) wrap these via PyO3: `Device`, `Memory`, `Queue`, `Kernel`.

---

## Cross-Layer Boundaries

| From | To | Mechanism |
|------|----|-----------|
| Layer 7 codegen | Layer 3 | TPTIR text → `compile_native()` |
| Layer 3 C++ | Layer 3 Rust | C API (`include/tptir/CAPI/tptir_capi.h`) + Rust FFI (`ffi.rs`) |
| Layer 4 Rust | Layer 2 driver | `libc` ioctl via `tpt_driver.h` ABI |
| Layer 4 | Python | PyO3 (`tptr-py`) |
| Layer 6 | Layer 4 | `tptr` crate or Python `tptr` package |

---

## Specification Files

Each layer has a spec that is authoritative for design decisions:

- `spec.txt` — Executive summary of the whole stack
- `layer1_isa/spec/tpt_isa_spec.md` — ISA opcodes, pipeline stages, memory model
- `layer3_tptc/spec/tptir_spec.md` — TPTIR types, operations, dialects, passes
- `layer4_tptr/spec/tptr_spec.md` — Runtime interface and error codes
- `layer5_tptp/spec/tptp_spec.md` — Kernel calling conventions, primitive interfaces
- `layer7_tptb/spec/tpts_spec.md` — Full TPT Script language specification (51KB)

When in doubt about intended behavior, consult the relevant spec before modifying code.

---

## Task Tracking

`todo.md` at the repo root tracks all work across all phases and layers. Mark items `[x]` when complete.
