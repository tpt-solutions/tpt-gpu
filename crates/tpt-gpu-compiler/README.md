# tpt-gpu-compiler — Rust Port of TPTIR Compiler Stack
## Build
```bash
cd layer3_tptc/rust && cargo build && cargo test
```
## Strategy
1. FFI bindings to C++ tptc
2. Native Rust IR + parser
3. Native Rust passes
4. Native Rust codegen
5. Complete Rust migration
