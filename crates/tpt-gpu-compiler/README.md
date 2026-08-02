# tpt-gpu-compiler — Rust Port of TPTIR Compiler Stack
## Build
```bash
# From the repo root (single root Cargo workspace)
cargo build -p tpt-gpu-compiler
cargo test -p tpt-gpu-compiler
```
## Strategy
1. FFI bindings to C++ tptc
2. Native Rust IR + parser
3. Native Rust passes
4. Native Rust codegen
5. Complete Rust migration
