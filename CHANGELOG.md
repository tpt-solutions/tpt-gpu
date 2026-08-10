# Changelog

All notable changes to TPT GPU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning note.** The Rust crates are published to
> [crates.io](https://crates.io) at `0.1.0` (the Cargo workspace version). The
> version headings below therefore follow the *published crate* versions. The
> **TPT Script language** reached its `v1.0.0` (2026-06-28) and `v1.1.0`
> (2026-06-29) milestones, but those shipped as part of the `0.1.0` crate line —
> `0.1.x` is pre-1.0, so the public API is **not yet guaranteed stable** and may
> change in a `0.2.0` release.

---

## [Unreleased]

Targets the next crate release after `0.1.0` (expected `0.2.0`).

### Added

#### Runtime integration & real execution
- **C ABI for `tpt-gpu-runtime`** — `tpt-gpu-runtime-c` exposes a stable C API (`tptr_device_create`, `tptr_device_allocate`/`free`, `tptr_memcpy_htod`/`dtoh`, `tptr_device_create_kernel`/`launch_kernel`, `tptr_device_synchronize`, error accessors) generated via cbindgen, mirroring the TPTIR C API pattern. A minimal C example (`examples/c/hello_device.c`) links and runs.
- **Python bindings** — `tpt-gpu-runtime-py` (`import tptr`) now compiles against the real runtime; `tptr.Device(0)` works, `Queue = CommandQueue`, and `__version__` is exported.
- **TPTIR module loading** — `Device::load_module(tptir_text)` compiles TPTIR to a runnable `Kernel` and is exposed through the Python bindings and C ABI; an integration test round-trips hand-written TPTIR (load + reduce + return) through `load_module` → `launch_kernel`.
- **Real CUDA backend** — `tpt-gpu-primitives` now dlopen `libcuda` and runs real `cuMemAlloc`/`cuMemcpy*`/`cuLaunchKernel`-backed GEMM/Attention/Conv2D/Conv3D via cuBLAS/cuDNN behind `feature = "cuda"`; verified with byte-for-byte memcpy round-trips on real hardware (RTX 3050). ROCm and Metal backends are present but unverified (no hardware).

#### Numerically-correct CPU fallback (no GPU required)
- The simulated runtime is now functionally real: `memcpy` copies bytes into a host-side arena, `launch_kernel` runs real arg validation + completion timing, and `synchronize()` drains the scheduler queue.
- All layer5 software-fallback kernels (GEMM, FusedGEMM, Attention, Conv2D, Conv3D, Embedding, RMSNorm) now perform real arithmetic instead of returning no-op/zeroed output. CPU inference is numerically meaningful end-to-end.
- `ScratchPool` replaces per-op fresh allocations with a shape-keyed free-list buffer pool; `GpuBuffer::reshape()` added.

#### LLM inference correctness
- **RoPE is now applied** at the correct decode position in `forward_step` (previously existed but was never called — inference was position-blind). Regression test `rope_is_applied_to_kv_cache` fails if reverted.
- **Real GGUF tokenizer** — `tpt-gpu-runtime` parses and exposes the model's GGUF vocab + BPE merges + special tokens; `tpt-serve` emits actual model text instead of `<id>` placeholders.
- `GpuInferenceEngine::load_with_vendor` implemented (format detection → header parse → arch template → weight load → kernels/KV cache/RoPE/scratch pool).
- mmap-backed `.tptf` loading (`memmap2`), and a numeric end-to-end golden-generation regression test (`inference_generation_golden`).

#### `tpt-serve` — OpenAI-compatible inference server
- New `tpt-gpu-serve` crate + `tpt-serve` binary serving `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions` (non-streaming + SSE) using only `std::net::TcpListener`. Loads a GGUF/TPTF model via `LlmInference` and uses the real tokenizer when present.

#### TPT-UIR ingestion adapter
- New `tpt-gpu-uir-adapter` crate: lossless TPTIR → TPT-UIR (`tpt-uir` GPU dialect) converter and reverse pass, plus a GGUF→TPTIR lowering for a representative Llama-3 decoder block. Wired into `tpt-gpu-kernelgen` (`--output-tptuir`) and `tpt-gpu-runtime` (`Device::load_module_tptuir`).

#### Usability & automation
- **`tpt` is now the canonical CLI binary** — `tpt-gpu-script-cli` builds both `tpt` and `tpt-gpu-script`; `cargo install tpt-gpu-script-cli` installs both, so docs that say `tpt ...` work verbatim.
- **`tpt-gpu-doctor`** — environment health check mirroring CI's fmt/clippy gates plus optional vendor SDK detection; usable as a pre-commit hook.
- Numeric-regression CI gate (`.github/workflows/numeric-regression.yml`) runs the runtime golden tests + `tpt-serve` end-to-end test + clippy on numeric-sensitive crates.
- Dependency/security hygiene: `cargo-audit`/`cargo-deny` in CI + Dependabot.

### Changed
- All Rust crates consolidated into a single root `crates/` Cargo workspace (Phase 8): 21 surviving crates moved from `layer2_tptd/rust`, `layer3_tptc/rust`, `layer4_tptr`, `layer5_tptp`, `layer6_tptf`, `layer7_tptb`, and `tools/` into `crates/<package-name>/`. Stale orphan workspace manifests and lockfiles removed.
- `layer6_tptf` deleted; its real JAX primitive/autodiff implementation was ported into `layer6_framework/tptr/jax/ops.py`.
- Docs paths reconciled to the `crates/` layout (CLAUDE.md, CI workflows, tutorials).

### Fixed
- **CI was fully broken on every runner**: `tpt-gpu-uir-adapter` depended on the sibling `tpt-uir` workspace via `path = "../../../tpt-uir/..."`, which only resolves when both repos are checked out side by side. On CI that directory never exists, so *workspace manifest loading itself* failed (`failed to read .../tpt-uir/crates/tpt-uir-core/Cargo.toml`) and every cargo job died before running. The three `tpt-uir-*` deps are now git deps pinned to an exact rev, so a standalone clone of `tpt-gpu` builds.
- Latent failures unmasked by the above (CI had never reached them): workspace-wide `cargo fmt` drift across 11 files, and `clippy -D warnings` dead-code/unused-import errors in the `argus_exporter` and `loopback_probe` runtime examples.
- `test_ollama_list_models_fallback` made a live network call to `localhost:11434` instead of testing the fallback path, so it passed or failed depending on whether the developer had Ollama running. It now points at a closed port.
- Critical correctness: GEMM/FusedGEMM now return the buffer actually written to; the HuggingFace download stream writes to disk and registers the manifest only after the file is confirmed.
- `tpt-gpu-vendor-cert` now uses real CPU reference implementations and calls the vendor library against them (gated on `detect_backend()`).
- Benchmark/certification docs reworded to state all numbers are analytical cost-model projections, not hardware measurements (community scoreboard points where real numbers will appear).

---

## [0.1.0] - 2026-08-04

First crates.io publication of the Rust workspace. This release encompasses the
**TPT Script v1.0.0** (2026-06-28) and **v1.1.0** (2026-06-29) language milestones
(described below); the published crate version is `0.1.0` and the API is pre-1.0.

### Added

#### TPT Script Language
- Complete standard library with 200+ orthogonal operations
- Tensor operations (arithmetic, shape manipulation, indexing)
- Neural network layers (linear, conv2d, attention, normalization)
- Optimization algorithms (SGD, Adam, learning rate schedulers)
- Distributed computing primitives (FSDP, pipeline parallelism)
- Data loading and preprocessing utilities

#### Compiler Infrastructure
- Production-ready lexer and parser
- Type checker with tensor shape inference
- Constraint evaluation system (`@constraint` annotations)
- Dual codegen: Rust (host) and TPTIR (GPU kernels)
- Structured error reporting with error codes and auto-fix suggestions
- Introspection API (`tpt.introspect.*`)

#### IDE Support
- Full Language Server Protocol (LSP) implementation
- Code completions, hover information, go-to-definition
- Real-time diagnostics and error reporting
- VS Code extension with syntax highlighting
- Formatter and linter

#### Runtime & Primitives
- Three-tier memory allocator (Slab → Buddy → Fallback)
- Priority-based command queue scheduler with aging
- Optimized GPU kernels: GEMM, Attention, Conv2D, Conv3D
- Normalization layers: BatchNorm, LayerNorm, GroupNorm
- AI-assisted kernel generation and optimization tools

#### Framework Integration
- PyTorch dispatch backend
- JAX integration planned (not yet implemented — `tptr.jax` exposes not-implemented stubs)
- HuggingFace model loading and inference support
- Distributed training examples (8-GPU FSDP)

#### Documentation
- Comprehensive user guide
- Formal language specification (51KB)
- Tutorials from basics to advanced
- API reference documentation
- Architecture overview and developer guide

#### Build & Release
- Cargo workspace configuration
- Automated CI/CD pipeline
- Benchmark regression testing
- crates.io publishing support
- Release automation scripts

#### TPT Script v1.1.0 (module system)
- Module system across 8 standard-library namespaces: `tpt`, `tpt.introspect`, `tpt.nn`, `tpt.optim`, `tpt.data`, `tpt.io`, `tpt.dist`, `tpt.compat`
- Project configuration via `tpt.toml` (`[package]`, `[features]`, `[profile]`, `[dependencies]`)
- Project scaffolding: `tpt new`, `tpt init`, `tpt modules`, `tpt compat` (Python stubs)
- `tpt check` validates imports; `tpt compile` auto-detects `tpt.toml`; `compile_project()` API
- `ProjectConfig` (serde) and `StdModule` registry

### Changed
- Improved error messages with structured error codes
- Enhanced type inference for tensor shapes
- Optimized compiler performance with parallel processing

### Fixed
- Error span accuracy improvements
- Parser edge cases in complex expressions
- Type checker false positives in generic functions
- Memory leaks in runtime allocator

### Security
- Added security policy (SECURITY.md)
- Implemented responsible disclosure process

---

## [0.1.0-beta] - 2026-03-15

### Added
- Initial beta release of TPT Script
- Lexer and parser implementation
- Basic type checker
- Rust and TPTIR codegen
- LSP server prototype
- Formatter and linter
- CLI tool (`tpt`)
- Basic standard library
- Documentation and tutorials

### Known Limitations
- No real hardware execution (simulation only)
- Partial standard library
- REPL not implemented
- Distributed execution not wired

---

*For releases, see the [GitHub Releases](https://github.com/tpt-solutions/tpt-gpu/releases) page.*
