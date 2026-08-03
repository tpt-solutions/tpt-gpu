# TPT GPU — Hardware-Agnostic Full-Stack GPU Compute Platform

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-APACHE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Docs](https://img.shields.io/badge/docs-latest-green.svg)](docs/user-guide.md)
[![CI](https://github.com/tpt-solutions/tpt-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-gpu/actions)

**TPT GPU** is an open-source, hardware-agnostic, full-stack GPU compute platform designed for AI/ML workloads. It features **TPT Script** — an AI-native programming language with a minimal, orthogonal API surface that LLMs can reason over without truncation.

---

## What's New in v1.0

- **Complete Standard Library** — 200+ orthogonal operations covering tensors, neural networks, optimization, and distributed computing
- **Production-Ready Compiler** — Lexer, parser, type checker with tensor shape inference, and dual codegen (Rust + TPTIR)
- **LLM Inference Runtime** — `GpuInferenceEngine` with arch-template dispatch (LLaMA 3, Mistral, Qwen2, Phi-3, Gemma 2), sliding-window KV cache, and automatic vendor routing (CUDA → ROCm → Metal → TPTIR)
- **Shared Model Registry** — GGUF models stored once in `~/.tpt/models/` and shared across all TPT tools
- **IDE Support** — Full LSP server, VS Code extension, formatter, and linter
- **Browser Playground** — Try TPT Script live in your browser, no install required: [`crates/tpt-gpu-playground/`](crates/tpt-gpu-playground/)
- **Framework Integration** — PyTorch and JAX backends with seamless dispatch
- **AI-Assisted Kernel Generation** — Automated kernel optimization and generation tools
- **Comprehensive Documentation** — 17 tutorials, complete language spec, and API reference

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tpt-solutions/tpt-gpu.git
cd tpt-gpu

# Build the TPT Script compiler
cargo build --release -p tpt-gpu-script-cli

# The binary is at: target/release/tpt-gpu-script
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

Looking for a specific end-to-end scenario (training loop, LLM inference, benchmarking, vendor certification)? See [`docs/use-cases.md`](docs/use-cases.md).

---

## Building

### Prerequisites

- Rust toolchain >= 1.75 (`rustup update`)
- Cargo workspace support
- Optional: VS Code for IDE features

### Build Commands

```bash
# Build all Rust layers
cargo build --release

# Build specific components
cargo build --release -p tpt-gpu-script-cli      # CLI tool
cargo build --release -p tpt-gpu-script-lsp      # LSP server
cargo build --release -p tpt-gpu-script-format   # Formatter/linter

# Run tests
cargo test --workspace

# Build with simulation mode (no hardware required)
cargo build -p tpt-gpu-primitives --features sim
```

---

## Key Features

### TPT Script Language
- **Statically typed** with tensor shape inference
- **Minimal API** — ~200 orthogonal operations (vs PyTorch's ~2000)
- **AI-native** — Every operation has machine-readable metadata (`@doc`, `@constraint`, `@complexity`)
- **Dual compilation** — Host functions → Rust, GPU kernels → TPTIR
- **Rich annotations** — `@requires_gpu`, `@distributed`, `@deploy`, and more

### LLM Inference
- **Architecture-agnostic dispatch** — Add new model architectures by registering one `ArchTemplate`
- **Supported architectures** — LLaMA 3, Mistral, Qwen2, Phi-3, Gemma 2 (GGUF format)
- **Sliding-window KV cache** — Autoregressive decoding with overflow eviction
- **Automatic vendor routing** — CUDA → ROCm → Metal → TPTIR fallback
- **Shared model registry** — Models downloaded once to `~/.tpt/models/` via HuggingFace

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

## Architecture

TPT GPU is organized into 7 independent layers with well-defined FFI/API boundaries:

```
layer1_isa/      SystemVerilog ISA — 32-bit fixed-length, 9-stage SIMT pipeline
layer2_tptd/     Kernel drivers — Linux DRM, Windows WDM, macOS DriverKit
layer3_tptc/     TPTIR compiler — MLIR-compatible dialect (C++ + Rust)
layer4_tptr/     GPU runtime — allocator, scheduler, kernel launch, LLM inference (Rust)
layer5_tptp/     GPU primitives — GEMM, Attention, Conv2D (TPTIR + Rust)
layer6_tptf/     Framework backends — PyTorch, JAX integration (Python + Rust)
layer7_tptb/     TPT Script compiler — lexer → parser → type checker → codegen (Rust)
```

**Development flow:** TPT Script (L7) → TPTIR (L3) → TPT ISA (L1) via Runtime (L4)

---

## Tools

| Tool | Description | Command |
|------|-------------|---------|
| `tpt-gpu-script` | CLI compiler | `tpt-gpu-script check`, `tpt-gpu-script compile`, `tpt-gpu-script run` |
| `tpt-gpu-script-lsp` | Language Server | IDE integration |
| `tpt-gpu-script-format` | Formatter/Linter | `tpt-gpu-script-format fmt`, `tpt-gpu-script-format lint` |
| `tpt-gpu-models` | Shared GGUF model registry | `tpt-gpu-models add/list/fetch` |
| `tpt-gpu-kernelgen` | AI-assisted kernel gen | Spec → TPTIR → validate → benchmark |
| `tpt-gpu-kernel-optimizer` | Auto-tuning | Grid → hill-climb → AI-guided search |

---

## Crates.io Publishing

TPT GPU components are published to crates.io for easy integration:

```toml
[dependencies]
tpt-gpu-script-core = "1.0"    # TPT Script compiler
tpt-gpu-primitives = "1.0"     # GPU primitives
tpt-gpu-runtime = "1.0"        # Runtime
```

Publish commands:

```bash
cd crates/tpt-gpu-script-core && cargo publish
cd crates/tpt-gpu-primitives && cargo publish
cd crates/tpt-gpu-runtime && cargo publish
```

---

## Roadmap

### v1.0 (Current)
- Complete standard library
- Production-ready compiler
- LLM inference runtime with KV cache
- Shared GGUF model registry
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

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on reporting bugs, submitting pull requests, code style requirements, and community tuning profile contributions.

---

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/user-guide.md) | Complete TPT Script language reference |
| [Language Spec](layer7_tptb/spec/tpts_spec.md) | Formal language specification (51KB) |
| [Tutorials](docs/tutorials/) | 17 hands-on tutorials from basics to advanced |
| [Architecture](CLAUDE.md) | Developer guide and build instructions |
| [Model Registry](MODELS_REGISTRY.md) | Shared GGUF model registry format |

---

## Security

Please see [SECURITY.md](SECURITY.md) for security policies and reporting vulnerabilities.

---

## License

TPT GPU is dual-licensed under your choice of the **MIT License** or the **Apache License 2.0 with LLVM Exception**.

See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for the full license text.

---

## Acknowledgments

- **Rust Community** — For the amazing ecosystem and tooling
- **MLIR Project** — For compiler infrastructure inspiration
- **Open Source Contributors** — For making this project possible
