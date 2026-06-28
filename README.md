# TPT GPU — Hardware-Agnostic Full-Stack GPU Compute Platform

[![License](https://img.shields.io/badge/license-Apache%202.0%20WITH%20LLVM--exception-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Docs](https://img.shields.io/badge/docs-latest-green.svg)](docs/user-guide.md)
[![CI](https://github.com/tpt-gpu/tpt-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-gpu/tpt-gpu/actions)

**TPT GPU** is an open-source, hardware-agnostic, full-stack GPU compute platform designed for AI/ML workloads. It features **TPT Script** — an AI-native programming language with a minimal, orthogonal API surface that LLMs can reason over without truncation.

---

## 🚀 What's New in v1.0

- **Complete Standard Library** — 200+ orthogonal operations covering tensors, neural networks, optimization, and distributed computing
- **Production-Ready Compiler** — Lexer, parser, type checker with tensor shape inference, and dual codegen (Rust + TPTIR)
- **IDE Support** — Full LSP server, VS Code extension, formatter, and linter
- **Framework Integration** — PyTorch and JAX backends with seamless dispatch
- **AI-Assisted Kernel Generation** — Automated kernel optimization and generation tools
- **Comprehensive Documentation** — 17 tutorials, complete language spec, and API reference

---

## 📖 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tpt-gpu/tpt-gpu.git
cd tpt-gpu

# Build the TPT Script compiler
cd layer7_tptb
cargo build --release -p tptb-cli

# The binary is at: target/release/tpt
```

### Your First TPT Script

Create `hello.tpts`:

```tpts
import tpt

@doc("Compute the ReLU activation function")
fn relu(x: Tensor[f32, n]) -> Tensor[f32, n] {
    return tpt.relu(x)
}
```

---

## 🛠️ Building

### Prerequisites

- Rust toolchain ≥ 1.75 (`rustup update`)
- Cargo workspace support
- Optional: VS Code for IDE features

### Build Commands

```bash
# Build all Rust layers
cargo build --release

# Build specific components
cd layer7_tptb
cargo build --release -p tptb-cli      # CLI tool
cargo build --release -p tptb-lsp      # LSP server
cargo build --release -p tptb-format   # Formatter/linter

# Run tests
cargo test --workspace

# Build with simulation mode (no hardware required)
cd layer5_tptp
cargo build --features sim
```

---

## 🎯 Key Features

### TPT Script Language
- **Statically typed** with tensor shape inference
- **Minimal API** — ~200 orthogonal operations (vs PyTorch's ~2000)
- **AI-native** — Every operation has machine-readable metadata (`@doc`, `@constraint`, `@complexity`)
- **Dual compilation** — Host functions → Rust, GPU kernels → TPTIR
- **Rich annotations** — `@requires_gpu`, `@distributed`, `@deploy`, and more

### Compiler Infrastructure
- **Fast compilation** — Parallel Rust implementation
- **Structured errors** — Error codes, locations, suggestions, and auto-fixes
- **Introspection API** — `tpt.introspect.list_operations()`, `get_schema()`, `validate_code()`
- **LSP support** — Completions, hover, diagnostics, go-to-definition

### Runtime & Primitives
- **Three-tier allocator** — Slab → Buddy → Fallback
- **Priority scheduler** — With aging to prevent starvation
- **Optimized kernels** — GEMM, Attention, Conv2D, Conv3D, normalization layers
- **AI-guided optimization** — Automated kernel tuning and generation

### Framework Integration
- **PyTorch dispatch** — Seamless backend integration
- **JAX integration** — XLA-compatible primitives
- **HuggingFace support** — Model loading and inference
- **Distributed training** — FSDP and pipeline parallelism

---

## 🔧 Tools

| Tool | Description | Command |
|------|-------------|---------|
| `tpt` | CLI compiler | `tpt check`, `tpt compile`, `tpt run` |
| `tptb-lsp` | Language Server | IDE integration |
| `tptb-format` | Formatter/Linter | `tptb-format fmt`, `tptb-format lint` |
| `kernel-generator` | AI-assisted kernel gen | Spec → TPTIR → validate → benchmark |
| `kernel-optimizer` | Auto-tuning | Grid → hill-climb → AI-guided search |

---

## 📦 Crates.io Publishing

TPT GPU components are published to crates.io for easy integration:

```toml

---

## 📋 Roadmap

### v1.0 (Current) ✅
- Complete standard library
- Production-ready compiler
- IDE support (LSP, VS Code extension)
- Framework integration (PyTorch, JAX)
- AI-assisted kernel generation

### v1.1 (Next)
- REPL for interactive development
- Enhanced error recovery
- Performance profiling tools
- Expanded hardware support

### v2.0 (Future)
- Custom silicon support
- Advanced distributed computing
- Web-based compiler playground
- TPT Script as recommended API

---

## 🔐 Security

Please see [SECURITY.md](SECURITY.md) for security policies and reporting vulnerabilities.

---

## 📄 License

TPT GPU is licensed under the **Apache License 2.0 with LLVM Exception**.

See [LICENSE](LICENSE) for the full license text.

---

## 🌟 Acknowledgments

- **Rust Community** — For the amazing ecosystem and tooling
- **MLIR Project** — For compiler infrastructure inspiration
- **Open Source Contributors** — For making this project possible

---

## 📬 Contact

- **Issues:** [GitHub Issues](https://github.com/tpt-gpu/tpt-gpu/issues)
- **Discussions:** [GitHub Discussions](https://github.com/tpt-gpu/tpt-gpu/discussions)

---

**Star ⭐ this repository if you find it useful!**

*Built with ❤️ by the TPT GPU Contributors*
[dependencies]
tptb-core = "1.0"    # TPT Script compiler
tptp-core = "1.0"    # GPU primitives
tptr-core = "1.0"    # Runtime (when available)
```

Publish commands:

```bash
cd layer7_tptb/tptb-core && cargo publish
cd layer5_tptp/tptp-core && cargo publish
cd layer4_tptr/tptr-core && cargo publish
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Start for Contributors

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`cargo test --workspace`)
5. Format code (`cargo fmt`)
6. Commit (`git commit -m 'Add amazing feature'`)
7. Push (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Areas

- **Layer 1 (ISA):** SystemVerilog simulation and verification
- **Layer 2 (Driver):** Kernel module development (Linux, Windows, macOS)
- **Layer 3 (Compiler):** TPTIR optimization passes and codegen
- **Layer 4 (Runtime):** Memory management and scheduling
- **Layer 5 (Primitives):** GPU kernel development and optimization
- **Layer 6 (Frameworks):** PyTorch/JAX integration
- **Layer 7 (TPT Script):** Language features and tooling

Compile and check:

```bash
# Type-check
tpt check hello.tpts

# Compile to Rust + TPTIR
tpt compile hello.tpts -o output/

# List all available operations
tpt ops

# Get docs for an operation
tpt docs matmul
```

---

## 🏗️ Architecture

TPT GPU is organized into 7 independent layers with well-defined FFI/API boundaries:

```
layer1_isa/      SystemVerilog ISA — 32-bit fixed-length, 9-stage SIMT pipeline
layer2_tptd/     Kernel drivers — Linux DRM, Windows WDM, macOS DriverKit
layer3_tptc/     TPTIR compiler — MLIR-compatible dialect (C++ + Rust)
layer4_tptr/     GPU runtime — allocator, scheduler, kernel launch (Rust)
layer5_tptp/     GPU primitives — GEMM, Attention, Conv2D (TPTIR + Rust)
layer6_tptf/     Framework backends — PyTorch, JAX integration (Python + Rust)
layer7_tptb/     TPT Script compiler — lexer → parser → type checker → codegen (Rust)
```

**Development flow:** TPT Script (L7) → TPTIR (L3) → TPT ISA (L1) via Runtime (L4)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/user-guide.md) | Complete TPT Script language reference |
| [Language Spec](layer7_tptb/spec/tpts_spec.md) | Formal language specification (51KB) |
| [Tutorials](docs/tutorials/) | 17 hands-on tutorials from basics to advanced |
| [Architecture](CLAUDE.md) | Developer guide and build instructions |