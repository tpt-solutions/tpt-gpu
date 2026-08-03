# Changelog

All notable changes to TPT GPU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-06-29

### Added

#### TPT Script as Recommended API
- **Module system** - Full support for all 8 standard library namespaces: `tpt`, `tpt.introspect`, `tpt.nn`, `tpt.optim`, `tpt.data`, `tpt.io`, `tpt.dist`, `tpt.compat`
- **Project configuration** - `tpt.toml` project files with `[package]`, `[features]`, `[profile]`, `[dependencies]` sections
- **Project scaffolding** - `tpt new <name>` creates a new project with `tpt.toml`, `src/main.tpts`, `.gitignore`
- **`tpt init`** - Initialize a project in the current directory
- **`tpt modules`** - List all standard library modules with their operations
- **`tpt compat`** - Generate Python type stubs (`.pyi`) from TPT Script source for IDE integration
- **`tpt check`** now validates imports against the standard library
- **`tpt compile`** now auto-detects `tpt.toml` and applies feature flags
- **`compile_project()` API** - New recommended entry point combining compilation with module resolution
- **Import validation** - Unknown `tpt.*` modules produce clear error messages
- **Feature-gated imports** - `tpt.nn` with `gpu = true` auto-imports GPU-accelerated layers
- **`ProjectConfig`** - Serde-compatible configuration struct with `from_toml()` / `to_toml()` support
- **`StdModule` registry** - Query available modules and their operations programmatically

---

# 
All notable changes to TPT GPU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-06-28

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
- Formatter and linter (`tptb-format`)

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
- 17 hands-on tutorials from basics to advanced
- API reference documentation
- Architecture overview and developer guide

#### Build & Release
- Cargo workspace configuration
- Automated CI/CD pipeline
- Benchmark regression testing
- crates.io publishing support
- Release automation scripts

### Changed
- Updated version from 0.1.0-beta to 1.0.0
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

## Release Notes

### v1.0.0 Public Release

This is the first production-ready release of TPT GPU. It includes:

- **Complete Standard Library** — All planned operations implemented
- **Stable API** — Language surface is now stable and backwards-compatible
- **Production Tooling** — Full IDE support, formatter, linter
- **Framework Integration** — PyTorch backend ready; JAX planned
- **Comprehensive Documentation** — Complete user guide and tutorials

### Migration from v0.1.0-beta

If you were using the beta version:

1. Update your TPT Script files (no breaking changes expected)
2. Rebuild the compiler: `cargo build --release -p tpt-gpu-script-cli`
3. Update VS Code extension (if installed)
4. Review new features in the user guide

### Feedback

We welcome your feedback! Please open issues on GitHub or join the discussion.

---

*For older releases, see the [GitHub Releases](https://github.com/tpt-solutions/tpt-gpu/releases) page.*