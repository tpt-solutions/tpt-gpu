# Contributing to TPT GPU

Thank you for your interest in contributing to TPT GPU!

---

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/tpt-solutions/tpt-gpu/issues). Please include:

- A minimal reproducer (ideally a `.tpts` file or a short Rust snippet)
- The output of `cargo run -p tpt-gpu-script-cli -- --version`
- Your OS and Rust toolchain version (`rustc --version`)

---

## Submitting Pull Requests

1. **Fork** the repository and create a branch from `master`:
   ```bash
   git checkout -b my-fix
   ```
2. Make your changes and add tests where applicable.
3. Run the formatting and lint checks (see Code Style below).
4. Open a **pull request against `master`** with a clear description of what the PR changes and why.

---

## Build Setup

Prerequisites: Rust toolchain ≥ 1.75 (`rustup update`).

```bash
# Clone
git clone https://github.com/tpt-solutions/tpt-gpu.git
cd tpt-gpu

# Build the entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace
```

See `CLAUDE.md` for per-layer build instructions (C++ compiler, SystemVerilog simulation, Python framework).

---

## Code Style

Before submitting, run:

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

The CI pipeline enforces both. PRs that fail either check will not be merged.

---

## Community GPU Tuning Profiles

The `tuning/` directory contains per-GPU tuning profiles (`tuning/<gpu_model>.json`) that feed the kernel optimizer and auto-dispatch table. To contribute a profile for your GPU:

1. Run `tpt-gpu-bench --contribute` — it benchmarks your hardware and writes a candidate `tuning/<gpu_model>.json`.
2. Open a PR adding that file. The CI job in `.github/workflows/validate-profiles.yml` validates it against `tuning/schema.json` automatically.

---

## Hardware CI Note

Some CI jobs (CUDA/ROCm kernel correctness, real-hardware GEMM round-trips) are gated behind self-hosted runners with physical GPUs. These jobs will show as skipped on standard GitHub-hosted runners — this is expected. If your change touches vendor backends (`crates/tpt-gpu-primitives/src/vendor/`), a maintainer will trigger the self-hosted run before merging.
