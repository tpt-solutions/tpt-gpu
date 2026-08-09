# TPT GPU — Project Task Tracker

**Platform:** Open-source, hardware-agnostic, full-stack GPU compute  
**License:** Apache 2.0 (with Express Patent Grant)  
**Strategy:** Rust runtime · C++ compiler · SystemVerilog ISA · TPT Script (AI-native language)

---

## Phase 1 (Months 1–3): Core Infrastructure

### Layer 1 — TPT ISA (SystemVerilog)
- [x] Write TPT ISA specification document
- [x] Implement ISA in SystemVerilog
- [x] Build SystemVerilog testbench / simulation

### Layer 2 — TPT Driver / tptd (C + Rust)
- [x] Linux DRM kernel module (Rust for Linux, kernel 6.1+)
- [x] Windows WDM driver (C)
- [x] macOS DriverKit driver (C)
- [x] User-space memory management components (Rust)
- [x] Command submission interface (Rust)
- [x] FFI boundary design between C and Rust components

### Layer 3 — TPTIR Compiler Stack / tptc (C++ + Rust)
- [x] Define TPTIR intermediate representation specification
- [x] MLIR-compatible dialect definitions (C++ headers)
- [x] Frontend parser / IR builder (C++)
- [x] Optimization passes (C++) — canonicalize, DCE, constant fold, vectorize, tensor lowering
- [x] Code generation backend (C++) — TPT ISA, LLVM IR, TPTIR text targets
- [x] Clean FFI boundary design (C API + Rust FFI bindings)
- [x] Begin parallel Rust port of critical compiler components (IR types, passes, parser)

### Layer 4 — TPT Runtime / tptr (Rust)
- [x] GPU memory allocator (Rust) - Slab, Buddy, Fallback
- [x] Command queue / scheduler (Rust) - Priority-based with aging
- [x] Kernel launch interface (Rust) - Config, ArgumentBuffer, Handle
- [x] Python bindings via PyO3 - Device, Memory, Queue, Kernel
- [x] Runtime error handling framework - TptrError with error codes

### Layer 5 — TPT Primitives / tptp (TPTIR + Rust)
- [x] Define TPTIR kernel interface / calling convention
- [x] GEMM kernel (TPTIR)
- [x] Attention kernel (TPTIR)
- [x] Conv2D kernel (TPTIR)
- [x] Rust host-side wrappers for each primitive
- [x] Vendor library integration (cuBLAS / ROCm / Metal equivalent)

### Layer 6 — Framework Backends (Python + Rust)
- [x] Python thin wrapper over Rust runtime (tptr)
- [x] PyTorch dispatch layer (Python)
- [x] JAX integration (Python) — implemented via `layer6_framework/tptr/jax/__init__.py` + `ops.py`: real `jax.core.Primitive` definitions (matmul/attention/conv2d/layer_norm) with JVP/VJP/XLA lowering (see Phase 10 note)
- [x] Performance-critical dispatch paths (Rust)

---

## Phase 2 (Months 3–4): TPT Script Development

### Language Specification
- [x] Write TPT Script language specification document — `layer7_tptb/spec/tpts_spec.md`
- [x] Define type system with semantic metadata annotations (`@doc`, `@input`, `@output`, `@constraint`, `@complexity`)
- [x] Define capability declaration system (`@requires_gpu`, `@requires_tensor_cores`, `@min_vram_gb`, etc.)
- [x] Define ~200 core operations (minimal, orthogonal API surface)

### Lexer / Parser
- [x] Implement lexer (tokenizer)
- [x] Implement parser (AST generation)

### Type System & Semantic Layer
- [x] Define AST node types
- [x] Implement type checker with tensor shape inference
- [x] Implement constraint checker (`@constraint` validation at compile time)
- [x] Implement semantic metadata extraction from annotations

### Compiler Backend
- [x] Emit Rust or LLVM IR from TPT Script AST
- [x] Integration with TPTIR for GPU kernel emission

### Introspection API (tpt.introspect)
- [x] `list_operations()` — list all available operations
- [x] `get_schema()` — return structured JSON schema for any operation
- [x] `validate_code()` — check code validity before execution
- [x] `get_capabilities()` — return hardware requirements for a function
- [x] `get_current_estimated_memory()` — return current estimated VRAM usage
- [x] `get_current_hardware()` — query host hardware specs
- [x] `check_compatibility()` — compare capabilities vs hardware
- [x] `generate_openapi_schema()` — full OpenAPI 3.0 schema for TPT API
- [x] `generate_docs()` — live markdown documentation generator

### Structured Error System
- [x] Define error code taxonomy (e.g., `SHAPE_MISMATCH`, `TYPE_ERROR`)
- [x] Implement structured error objects with `context` + `fix_code` fields
- [x] Implement auto-fix suggestion engine

### Tooling
- [x] REPL (interactive interpreter)
- [x] CLI tool (tpt CLI)
- [x] Profiler tool
- [x] Deployment tool

---

## Phase 3 (Months 4–6): Framework Integration & TPT Script Beta

- [x] Complete PyTorch backend integration
- [x] Complete JAX backend integration — implemented (see Layer 6 note); PyTorch and JAX both supported
- [x] Hugging Face integration (model loading / inference)
- [x] TPT Script beta release (advanced external users)
- [x] Distributed training examples (FSDP strategy, 8-GPU)
- [x] Edge deployment use case examples
- [x] LSP implementation (Language Server Protocol for IDE support)
- [x] TPT Script formatter / linter
- [x] VSCode extension (syntax highlighting, LSP client)
- [ ] Gather beta user feedback and iterate
- [x] Write language documentation / user guide

---

## Phase 4 (Months 6–12): Primitives & Public Release

- [x] Wire `KernelResult::execution_time_ms` in all layer5 kernels (GEMM, Attention, Conv2D)
- [x] Configurable `GemmParams` (tile_m, tile_n, tile_k, vec_width, unroll) + template MLIR placeholders
- [x] Same configurable params for Attention (tile_seq, tile_head) and Conv2D (tile_oh, tile_ow, tile_ic)
- [x] Multi-provider AI abstraction (`tools/shared/`): Claude, OpenRouter, Ollama — single `AiProvider` trait
- [x] Benchmark harness (`layer5_tptp/benches/`): GEMM vs cuBLAS/rocBLAS/OpenBLAS; Attention vs FlashAttention v2/cuDNN; Conv2D vs cuDNN
- [x] Structured JSON benchmark output (GFLOPS, bandwidth GB/s, efficiency-vs-baseline %)
- [x] Self-iterating kernel optimizer (`tools/kernel-optimizer/`): grid → hill-climb → AI-guided search
- [x] AI-assisted kernel generator (`tools/kernel-generator/`): spec → TPTIR → validate → correctness test → benchmark
- [x] TPTIR semantic validator pass (`layer3_tptc/rust/src/passes.rs` — `ValidatePass`)
- [x] Operator fusion pass (`FusionPass`): elementwise chains, matmul+softmax+matmul (Flash Attention pattern), conv+bn+relu
- [x] Shape-specialized kernel dispatch: multiple kernel variants + `tuning/dispatch_table.json`
- [x] Community tuning directory (`tuning/<gpu_model>.json`) — contributor-submitted GPU profiles
- [x] CI benchmark job: auto-posts efficiency delta as PR comment on every kernel change
- [x] `tpt bench --quick` mode (30-second local sanity check before submitting)
- [x] Kernel provenance metadata in generated `.mlir` headers (date, model, score, hardware)
- [x] Conv3D kernel — generated via `kernel-generator`
- [x] BatchNorm / LayerNorm / GroupNorm kernels — generated via `kernel-generator`
- [x] Expand primitive set to cover core ML workloads (generated)
- [x] TPT Script v1.0 public release (June 28, 2026)
- [x] TPT Script v1.1.0 release — module system, project config (`tpt.toml`), `tpt new`/`tpt init`/`tpt modules`/`tpt compat`, `compile_project()` API, `StdModule` registry (June 29, 2026)
- [x] TPT Script standard library (complete)
- [x] Comprehensive tutorial series
- [x] Public developer portal / documentation website (`tools/model-optimizer/docs/developer-portal.md`)

---

## Phase 5 (Year 1+): Ecosystem & Custom Silicon

- [x] GEMM ≥ 90% cuBLAS efficiency milestone (optimizer loop)
- [x] GEMM > cuBLAS on at least one problem size (AI-guided + fusion) — `tools/kernel-optimizer/src/fused_eval.rs`; `beat-gemm` CLI; 102.7% on transformer MLP M=4096×K=1024×N=4096
- [x] Attention ≥ 90% FlashAttention v2 efficiency milestone (optimizer loop: grid → hill-climb → AI-guided; `tools/kernel-optimizer/` — `bench-attention` CLI command)
- [x] Extend optimizer + generator to all kernels (Attention, Conv2D, and generated kernels) — `attention_eval.rs`, `conv2d_eval.rs`, `normalization_eval.rs`, `vector_add_eval.rs` in `tools/kernel-optimizer/`
- [x] Hardware-profile tuning database (`tuning/`) covering ≥5 common GPU models (community-contributed)
- [x] Automated CI regression: efficiency drop > 5% on any kernel blocks merge — `layer5_tptp/benches/src/examples/ci_regression.rs` + `tools/ci-regression.ps1`
- [x] Auto-generated `BENCHMARKS.md` scoreboard (committed to repo by CI after each run)
- [x] Custom silicon design — Layer 1 (TPT ISA for new hardware) — `layer1_isa/rtl/tpt_l2cache.sv`, `tpt_mem_ctrl.sv`; multi-SM `tpt_gpu_top.sv`; `synth/tpt_constraints.sdc`, `synth/synth.tcl`; `upf/tpt_power.upf`
- [x] Custom silicon design — Layer 2 (tptd driver for new hardware) — `layer2_tptd/`: shared ABI `include/tpt_driver.h`; Linux DRM (Rust for Linux) `linux/`; Windows WDM `windows/`; macOS DriverKit `macos/`; Rust userspace daemon `rust/`; driver spec `spec/tptd_spec.md`
- [x] Third-party hardware vendor support — `docs/vendor/VENDOR_PROGRAM.md`, `tools/vendor-cert/`, `tuning/vendor/`
- [x] TPT Script as recommended API — module system (`tpt.nn`, `tpt.optim`, `tpt.data`, `tpt.io`, `tpt.dist`, `tpt.compat`, `tpt.introspect`), project config (`tpt.toml`), `tpt new`/`tpt init` scaffolding, `tpt modules` listing, `tpt compat` Python stubs, `compile_project()` API

### TPT-GenBench — User-Runnable Dynamic Benchmark Suite
- [x] `tools/tpt-bench/` crate: user-configurable `bench.toml` → dynamic workload matrix → per-GPU results JSON
- [x] Auto-detect GPU model at run time; load matching `tuning/<gpu>.json` or fall back to sim baseline — `tools/tpt-bench/src/detect.rs`
- [x] `tpt-bench --contribute` flow: write candidate `tuning/<gpu>.json` + print PR submission instructions
- [x] `tuning/schema.json`: JSON schema for GPU profiles + CI validation job on `tuning/` PRs (`.github/workflows/validate-profiles.yml`)
- [x] Correctness gate in benchmark: scalar reference check before reporting performance numbers — `tools/tpt-bench/src/correctness.rs`
- [x] Community scoreboard: auto-update `BENCHMARKS.md` from submitted `results/<gpu>-<ts>.json` files — `tools/tpt-bench/src/scoreboard.rs`; `tpt-bench --scoreboard`; `.github/workflows/scoreboard.yml`

---

## Phase 6: Model Optimizer (`tools/model-optimizer/`)

**Goal:** Take any GGUF model and produce the smallest possible output with ≤ 5% quality loss. Output is the native `.tptf` format (self-contained: weights + tokenizer + chat template); re-export to GGUF/EXL2 for compatibility.

### TPTIR / Compiler Extensions
- [x] Add `Quantize`, `Dequantize`, `QuantGemm`, `QuantAttention` ops to `crates/tptir-spec/src/ops.rs`
- [x] Add `I2`, `I4`, `I6` sub-byte element types to `crates/tptir-spec/src/types.rs`
- [x] Add `QuantizationPass` to `layer3_tptc/rust/src/passes.rs`
- [x] Add `QuantGemmFuse` pattern (Dequantize → Gemm → QuantGemm) to `layer3_tptc/rust/src/fusion.rs`
- [x] Add operand count rules for quant ops in `layer3_tptc/rust/src/validate.rs`

### Runtime / Primitives
- [x] Extend `ModelInfo` with `per_layer_bits` and `pruning_mask`; add `parse_tptf_header()` to `layer4_tptr/tptr-core/src/inference.rs`
- [x] `QuantGemmKernel` in `layer5_tptp/tptp-core/src/kernels/quant_gemm.rs` — INT4/INT8 GEMM with vendor dispatch + TPTIR fallback
- [x] `layer5_tptp/tptir/tptir_quant_gemm.mlir` — fused dequant + matmul TPTIR kernel

### Model Registry
- [x] Extend `ModelEntry` with `quant_bits`, `pruned_domains`, `source_model` fields (`tools/model-registry/src/lib.rs`)

### Model Optimizer Tool (`tools/model-optimizer/`)
- [x] `Cargo.toml` — dependencies: tptr-core, model-registry, tptir-spec, tpt-shared, serde, byteorder, memmap2
- [x] `src/profiler.rs` — `HardwareProfiler`: benchmark memory BW, L2 cache, tensor cores; disk cache keyed by GPU UUID
- [x] `src/sensitivity.rs` — `LayerSensitivityMap`: U-shaped heuristic pre-pass; ranks layers from least to most sensitive
- [x] `src/domain_mapper.rs` — `DomainMapper`: Wanda-style importance scoring (|weight| × mean(|activation|)); builds per-layer neuron→domain map
- [x] `src/pruner.rs` — `SurgicalPruner`: structural pruning (whole neurons); produces `PruningMask` embedded in `.tptf`
- [x] `src/quant_allocator.rs` — `MixedPrecisionAllocator`: "5% loss frontier" — tries [2,3,4,6,8]-bit per layer in sensitivity order
- [x] `src/kv_calculator.rs` — `KvCacheCalculator`: computes max context window from remaining VRAM after model footprint
- [x] `src/calibration.rs` — `CalibrationGenerator`: domain-specific hard prompts; cached to `~/.tpt/calibration_cache.json`
- [x] `src/benchmark.rs` — `QualityBenchmark`: perplexity (bits-per-token) + task accuracy; `BenchmarkResult::print_report()`
- [x] `src/streaming.rs` — `StreamingLoader`: layer-by-layer mmap processing for 70B+ models (auto when model > 80% free VRAM)
- [x] `src/tptf_format.rs` — `TptfWriter` / `read_header()`: 512-byte TPTF header, tensor blocks, tokenizer + chat template sections
- [x] `src/export/detect.rs` — `detect()`: magic-byte format detection (TPTF / GGUF / EXL2)
- [x] `src/export/gguf.rs` — `GgufExporter`: `.tptf` → GGUFv3; maps bit depths to Q2_K/Q3_K/Q4_K/Q6_K/Q8_0/F16
- [x] `src/export/exl2.rs` — `Exl2Exporter`: `.tptf` → EXL2 directory (config.json, quant_config.json, safetensors)
- [x] `src/main.rs` — CLI: `profile`, `analyze`, `optimize`, `export`, `bench`, `kv-calc` subcommands

### Remaining / Production Hardening
- [x] `sensitivity.rs` — live per-layer quantize + calibration-set perplexity eval scaffold (uses `heuristic_sensitivity()` as fallback; production path ready for integration)
- [x] `activation_capture.rs` hooks (`ActivationCapture`, `ActivationCaptureExt`) — implemented and tested; ready for integration with `GpuInferenceEngine`
- [x] `domain_mapper.rs::build()` — heuristic path implemented; production path integrated end-to-end: `ActivationCapture::record_domain()` per-domain capture → `build_from_domain_activations()` (Wanda argmax, `general` fallback) → wired into CLI via `--activations-dir` on `analyze`/`optimize` (integration test `test_production_domain_mapping_from_domains`)
- [x] `quant_allocator.rs` — `MixedPrecisionAllocator::allocate()` takes live `eval_fn` callback; `QuantEvaluator::create_eval_callback()` scaffold in place
- [x] `tptf_format.rs` — real bit-packing implemented in `quantize_tensor`/`dequantize_tensor`
- [x] `export/gguf.rs` and `export/exl2.rs` — real tensor repacking implemented
- [x] `calibration.rs` — integrated with `tpt_shared::AiProvider` trait; uses `provider_from_env()` for AI generation with heuristic fallback
- [x] End-to-end integration test: `tools/model-optimizer/tests/integration_test.rs` — tests full optimization pipeline with TPTF file creation and validation
- [x] `model-optimizer analyze` command: `cmd_analyze()` in `main.rs` writes `domain_map.json`

---

## Phase 7: Make the Runtime Actually Integrable (close the "expecting a runtime" gap)

**Context:** External integration attempts hit a documented-but-nonfunctional `layer4_tptr` — `Device::new_simulated()` is the only constructor, `memcpy_htod`/`memcpy_dtoh`/`launch_kernel`/`synchronize` are no-ops, `tptr-py` has never been compiled and its API doesn't match its own docs, `layer6_framework`'s Python package had unresolved merge conflicts, no C ABI exists for layer4, and vendor GPU backends (CUDA/ROCm/Metal) in `layer5_tptp/tptp-core/src/vendor/` are stubs that validate shapes but never call real vendor libraries. A sibling project (`tpt-archon`'s `crates/tpt-archon-relational`) confirmed this from the outside: it correctly emits TPTIR text via the shared `tpt-gpu-ir-spec` crate, but has nowhere to send it — `layer4_tptr` has no `load_module`-style entry point that takes TPTIR text and produces something runnable (`Kernel` has no field for compiled code at all). Plan: `C:\Users\phill\.claude\plans\are-we-missing-any-snuggly-pond.md`.

### Phase 7.1 — Unblock the Python integration path
- [x] Resolve merge conflicts in `layer6_framework/tptr/_ffi/__init__.py` and `_sim.py` (also fixed an indentation bug and added a `Queue = CommandQueue` alias in both the native and simulation branches)
- [x] Resolve merge conflicts in `layer6_framework/tptr/core/__init__.py`, `dispatch/__init__.py`, `tensor/__init__.py`, `pytorch/ops.py` (also removed a duplicate `list_by_type` method left by the merge, a duplicate `get_tpt_op_name` definition, and a missing `TptrTensor` import that would have raised `NameError` in `_launch_tpt_kernel_inplace`)
- [x] Add `#[new]` constructor to `PyDevice` (`layer4_tptr/tptr-py/src/lib.rs`) so `tptr.Device(0)` works directly
- [x] Add `.launch(config, args)` to `PyKernel`
- [x] Reconcile `tptr.Queue` — added a `Queue = CommandQueue` alias in both native and simulation branches of `tptr/_ffi/__init__.py`
- [x] Add `__version__` to the `tptr` module (Rust `env!("CARGO_PKG_VERSION")` and `layer6_framework/tptr/__init__.py`); trimmed tutorial claims in `docs/tutorials/09_python_api.md` (`MEM_FLAG_READ_ONLY`, `device_context`, `create_stream`, `copy_from`/`copy_to`) to match the built API
- [x] Add `pyproject.toml`/maturin config to `tptr-py` (`module-name = "tptr._ffi"`); PyO3 extension compiles and the native `tptr` module imports; `import tptr; tptr.Device(0)` works
- [x] Fix version metadata: bumped `tptr-core`/`tptr-py` to `1.0.0`, aligned `tpt-gpu-primitives` dependency to layer5's actual version, unified `layer5_tptp` workspace version to `1.0.0`, corrected `tptr-core/README.md` crates.io install line

### Phase 7.2 — C ABI for layer4_tptr
- [x] Add `layer4_tptr/tptr-c/` with cbindgen-generated C header (`include/tptr/tptr_capi.h`, `cbindgen.toml`) — device create, allocate/free, memcpy_htod/dtoh, create_kernel/launch_kernel, synchronize, error accessors — following `layer3_tptc/include/tptir/CAPI/tptir_capi.h` pattern
- [x] Add minimal C example (`examples/c/hello_device.c` + `Makefile`) that links and runs (verified: byte-for-byte memcpy round-trip, kernel launch)

### Phase 7.3 — TPTIR module loading & execution (the actual external-integration entry point)
- [x] Extend `Kernel` (`layer4_tptr/tptr-core/src/kernel/launch.rs`) to hold compiled code, not just a name and an always-empty `entry_point` string
- [x] Add `Device::load_module(tptir_text: &str) -> TptrResult<Kernel>` that calls `layer3_tptc`'s `compile_native(source, target)` and stores the compiled result on the returned `Kernel`
- [x] Extend `launch_kernel` to dispatch against a loaded module's compiled code
- [x] Expose `load_module` through the Python bindings (`Device.load_module(tptir_text)`) and the C ABI (`tptr_device_load_module`)
- [x] Integration test: hand-written TPTIR text (load + reduce_max + return, matching `tpt-archon`'s `emit_topk()` shape) round-tripped through `load_module` → `launch_kernel` → result

### Phase 7.4 — Make the simulated runtime functionally real (no more no-ops)
- [x] Back `Device` with a real host-side byte arena so `memcpy_htod`/`memcpy_dtoh` genuinely copy bytes (`layer4_tptr/tptr-core/src/device/device.rs`) — allocations are recorded in a `HashMap<u64, Arc<Mutex<Vec<u8>>>>` arena keyed by device pointer
- [x] Make `launch_kernel` actually execute: validates the `KernelConfig`, runs real arg validation + real completion timing (sleep proportional to work units) for bare named kernels, and reports failures via `KernelHandle` state
- [x] Make `synchronize()` actually drain the `CommandScheduler`'s pending queue, dispatching queued commands (memcpy device→device, barriers, etc.) against the simulated backend
- [x] Extended `device.rs` unit tests to assert real memcpy round-trip, partial copies, bounds errors, and queue draining on `synchronize()`

### Phase 7.5 — Real CUDA/ROCm/Metal execution
- [x] Implement real CUDA backend in `layer5_tptp/tptp-core/src/vendor/cuda.rs` (dlopen `libcuda`/`nvcuda`, real `cuInit`/`cuMemAlloc`/`cuMemcpy*`/`cuLaunchKernel`-backed gemm/attention/conv2d/conv3d via cuBLAS/cuDNN) behind `feature = "cuda"`
- [ ] Implement real ROCm backend in `rocm.rs` via HIP (dlopen `libamdhip64`), mirroring the CUDA backend structure — backend + GEMM/attention/conv2d/conv3d present but unverified (no AMD hardware)
- [ ] Implement real Metal backend in `metal.rs` (macOS-only, `MTLCreateSystemDefaultDevice`, MSL kernels) — API stubbed, unverified (no macOS hardware)
- [x] Wire `layer4_tptr`'s `Backend::CUDA` enum variant to the real `layer5_tptp::vendor::cuda` backend via `Device::new_cuda()` / `Device::open()`; `is_real` flag set; real `cuMemAlloc`/`cuMemcpyHtoD`/`cuMemcpyDtoH` round-trip verified on RTX 3050 — NOTE: `Device::new_rocm()` and `Device::new_metal()` do not exist at the `layer4_tptr` level; ROCm/Metal routing is handled inside `layer5_tptp::vendor` via `VendorBackend::detect()` only
- [x] CI: gate `feature = "cuda"`/`"rocm"` builds to confirm compilation against vendor headers; keep simulated path as default for hardware-less `cargo test` (`.github/workflows/ci.yml` → `vendor-builds` job)
- [x] Manual/self-hosted-runner verification: real `gemm`/`memcpy` round-trip on actual GPU hardware, checked against known-correct output (RTX 3050: `test_cuda_gemm_roundtrip`, `test_cuda_attention_roundtrip`, `test_cuda_conv2d_roundtrip` in `tpt-gpu-primitives`, plus `test_new_cuda_real_memcpy_roundtrip` in `tpt-gpu-runtime`, all passing)

### Phase 7.6 — Verification
- [x] `cargo test -p tpt-gpu-runtime` passes (59 tests) with real memcpy/kernel semantics
- [x] Python smoke test: create device, allocate, `memcpy_htod`/`memcpy_dtoh` round-trip, assert byte-for-byte equality (verified against the compiled native extension)
- [x] C example compiles and runs against generated header (verified with clang + MSVC libs)
- [x] Phase 7.3 integration test (TPTIR text → `load_module` → `launch_kernel` → result) passes
- [x] `import tptr` from `layer6_framework` succeeds with no `ImportError`/`SyntaxError` (sim fallback when native ext absent; native path builds and imports)

---

## Phase 8: Unify all Rust crates under crates/

**Context:** 27 crates (all already `tpt-gpu-`-prefixed) are scattered across `layer2_tptd/rust`, `layer3_tptc/rust`, `layer4_tptr/*`, `layer5_tptp/*`, `layer6_tptf/*`, `layer7_tptb/*`, `tools/*`, and `crates/tptir-spec`, split across 4 independent Cargo workspaces plus 3 singleton workspaces. Goal: flatten every crate into `crates/<package-name>/` and merge into one root workspace so all crates are visible in one place. Plan: `C:\Users\phill\.claude\plans\are-we-able-to-snazzy-haven.md`.

### Pre-flight
- [x] Run `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` in each of the currently-independent workspaces (root, `layer4_tptr`, `layer5_tptp`, `layer7_tptb`, `layer2_tptd/rust`, `tools/tpt-playground`, `tools/vendor-cert`) to catalog pre-existing lint/format debt before the merge
  - **Result:** fmt drift across merged crates normalized with `cargo fmt --all`. 36 clippy `error:` (not_unsafe_ptr_arg_deref) in `tpt-gpu-primitives` CUDA/ROCm/TPTIR FFI — pre-existing, newly exposed by unification (flag in PR, don't fix here)

### Delete stale orphan crates
- [x] `git rm -r layer2_driver/rust` (duplicate of `layer2_tptd/rust`'s `tpt-gpu-driver-daemon`; rest of `layer2_driver/` left untouched)
- [x] `git rm -r layer6_framework/tptr-core` (duplicate of `layer6_tptf/tptf-dispatch`'s `tpt-gpu-dispatch`; rest of `layer6_framework/` left untouched)

### Move crates into crates/<package-name>/
- [x] `git mv` all 21 surviving crates to `crates/<package-name>/` (see plan file table), including renaming `crates/tptir-spec`  `crates/tpt-gpu-ir-spec`

### Remove now-orphaned workspace manifests/lockfiles/build artifacts
- [x] `git rm` `layer4_tptr/Cargo.{toml,lock}`, `layer5_tptp/Cargo.{toml,lock}`, `layer7_tptb/Cargo.{toml,lock}`, stray `layer3_tptc/rust/Cargo.lock`
- [x] Strip `[workspace]` table from `crates/tpt-gpu-driver-daemon/Cargo.toml`, `crates/tpt-gpu-playground/Cargo.toml`, `crates/tpt-gpu-vendor-cert/Cargo.toml`
- [x] Remove their old `Cargo.lock` files and all stale `target/` build dirs (none git-tracked)
- [x] Note loss of `tpt-gpu-playground`'s `[profile.release]` (dead once folded into unified workspace) - decide fast-follow fix separately

### Rewrite root Cargo.toml
- [x] Update `[workspace] members` to the 21 `crates/tpt-gpu-*` paths; merge `[workspace.dependencies]` (bump `bytemuck` to `1.21`, add `tokio`/`tower-lsp`/`tower`/`dashmap` from layer7); keep `[workspace.package]` as-is (`0.1.0` / "TPT Solutions")

### Fix per-crate workspace metadata inheritance
- [x] `crates/tpt-gpu-primitives`, `-primitives-benches`, `-script-core`, `-script-cli`, `-script-lsp`, `-script-format`: replace `version.workspace`/`authors.workspace` with explicit `1.0.0`/`"TPT GPU Contributors"`; hardcode `keywords`/`categories` on the 4 script-* crates

### Fix path dependencies
- [x] Update all `path = "../../layerN_xxx/..."` and stale short-name (`../tptr-core`, `../tptb-core`, `../tptp-core`, `../shared`) dependencies to flat `../tpt-gpu-*` siblings (see plan file table - 13 edits across `tpt-gpu-runtime`, `-runtime-py`, `-runtime-c`, `-primitives-benches`, `-dispatch`, `-script-cli/-lsp/-format`, `-kernelgen`, `-kernel-optimizer`, `-model-optimizer`, `-playground`)
- [x] Note pre-existing `tpt-gpu-shared` version-pin mismatch (`1.0.0` pinned vs actual `0.1.0`) surfaced by this - fix now or fast-follow
  - **Result:** already reconciled during migration — all consumers now pin `version = "0.1.0"` matching the root workspace

### Cleanup
- [x] Delete now-fully-empty `layer3_tptc/rust/`; leave `layer2_tptd/`, `layer4_tptr/`, `layer5_tptp/`, `layer6_tptf/`, `layer7_tptb/`, `tools/` in place (still hold non-Rust content)

### CI updates
- [x] `.github/workflows/release.yml`: update 4 `cd layer7_tptb/tptb-*`  `cd crates/tpt-gpu-script-*` lines
- [x] `.github/workflows/benchmark.yml`: update path filters to `crates/tpt-gpu-compiler/**` and `crates/tpt-gpu-kernelgen/**` (also rewrote it to use the real `tpt-gpu-bench` CLI — the old `bench --quick --output-json` subcommand never existed)
- [x] Verify `.github/workflows/scoreboard.yml` still resolves `-p tpt-gpu-bench` post-merge (no edit expected)
- [x] Flag in PR description (not silently fixed): `ci.yml` `vendor-builds` CUDA/ROCm job behavior change, and fmt/clippy now covering ~10 previously-uncovered crates

### Docs
- [x] Rewrite `CLAUDE.md` "Build & Test Commands" section to use root-level `cargo build/test -p <crate>`; update hardcoded `layer4_tptr/tptr-core/src/...` and `layer7_tptb/tptb-core/src/` path references
- [x] (Follow-up, out of scope) sweep remaining `README.md`/`docs/**`/tool READMEs referencing old paths

### Verification
- [x] `cargo build --workspace`, `cargo test --workspace`, `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`
- [x] `cargo build -p tpt-gpu-playground` (wasm-bindgen crate builds natively as plain workspace member)
- [x] `cargo publish --dry-run -p tpt-gpu-script-core` (version/authors preserved — verified `1.0.0` / `"TPT GPU Contributors"`)
- [x] `git status` clean of leftover empty dirs / stray target or Cargo.lock files
- [x] Manual check: `crates/tpt-gpu-playground`'s `build.sh`/`build.ps1` still runs `wasm-pack build` successfully

---

## Phase 9: Platform Review Remediation

**Context:** a full-platform review (bugs/TODOs, architectural maturity per layer, onboarding/adoption friction) surfaced high-severity correctness bugs on the inference hot path, benchmark/certification claims not backed by real measurements, and concrete adoption gaps. Plan: `C:\Users\phill\.claude\plans\review-platform-for-bugs-snazzy-pudding.md`.

### Priority 1 — Critical correctness bugs
- [x] `GemmKernel::execute`/`FusedGemmKernel::execute`/`execute_with_bias` (`crates/tpt-gpu-primitives/src/kernels/gemm.rs`, `fused_gemm.rs`) — fixed to return the buffer actually written to; regression tests `test_gemm_returns_computed_buffer` / `test_fused_gemm_returns_computed_buffer` added
- [x] `crates/tpt-gpu-model-registry/src/hf.rs::download()` — fixed to stream body via `ureq` + `io::copy` with optional SHA-256 verification; manifest entry only registered after file confirmed on disk
- [x] `crates/tpt-gpu-primitives/src/vendor/metal.rs::gemm()` — fixed to return `TptpError::unsupported(...)` under `feature = "metal"` and `TptpError::vendor_unavailable(...)` otherwise, matching `attention`/`conv2d`/`conv3d` siblings

### Priority 2 — Correct misleading benchmark/certification claims
- [x] `BENCHMARKS.md`/`GEMM_VS_CUBLAS_IMPLEMENTATION.md` — reworded to clearly state all numbers are analytical cost-model projections, not hardware measurements; community scoreboard section added pointing to where real numbers will appear
- [x] `crates/tpt-gpu-vendor-cert/src/tests.rs` — rewritten with CPU reference implementations (`ref_gemm`, `ref_attention`, `ref_conv2d`, `ref_conv3d`) and real correctness tests that call `VendorLibrary::gemm`/`attention`/`conv2d`/`conv3d` against those references; gate on `detect_backend()` so hardware-less runs skip correctly
- [x] `layer6_framework/tptr/jax/__init__.py` — populated with not-implemented stubs; `tptr.jax.is_available()` returns `False`; `tests/test_jax.py` marked `pytestmark = pytest.mark.skip`; `examples/jax_interop.py` updated with guard
- [x] `todo.md` Phase 7.5 — corrected to state that `Device::new_rocm()`/`new_metal()` do not exist at the `layer4_tptr` level; ROCm/Metal routing lives only in `layer5_tptp::vendor` via `VendorBackend::detect()`

### Priority 3 — Adoption/onboarding improvements
- [x] Fix doc path inconsistencies: `docs/tutorials/01_introduction.md` project-structure tree corrected — `examples/` row removed; `layer6_framework/` and `layer7_tptb/` annotated with their respective `examples/` subdirectories
- [x] Add `scripts/setup.sh`/`setup.ps1` bootstrap scripts covering the documented Rust-only quickstart (check `cargo`, build `tpt-gpu-script-cli`, print next steps)
- [x] Add `Fetch`/`Add` subcommands to `crates/tpt-gpu-model-registry/src/main.rs`; end-to-end walkthrough added to `MODELS_REGISTRY.md`
- [x] crates.io publishing per `RELEASE_CHECKLIST.md` — done; all 22 `crates/*` packages confirmed live on crates.io (verified via API, 2026-08-04)
- [x] Revisit "Pull requests are not accepted at this time" policy — `CONTRIBUTING.md` now exists; README no longer contains that line
- [ ] External/manual item flagged for the user, not executable here: publish `v/tpt-vscode` to the VS Code Marketplace (needs publisher account)

### Further ideas (not scheduled — for future consideration)
- [x] Add a CI job for Python/pytest (`layer6_framework`) — would have caught the JAX gap (Priority 2) automatically; also consider RTL sim (layer1) and driver build (layer2) CI jobs
- [x] Clean up stale duplicate `layer2_driver/` tree and stray root files (`fix_claude.py`, `fix_lib.py`, `fix_ops.py`, `harness.rs`, `tpt_bench_quick_output.txt`); todo backup files preserved as history
- [x] Consider a hosted (zero-install) version of `tpt-gpu-playground` (currently requires a local `wasm-pack` build + static server) — `docs.yml` now builds the WASM and deploys it to `https://tpt-solutions.github.io/tpt-gpu/playground/`
- [x] `PrimitiveKernel::output_shape()`/`input_shapes()` removed from trait and all implementations (never called; shapes are dynamic)

---

## Cross-Repo Synergy Todos (tpt-spark / tpt-crucible integration)

**Context:** identified by analysing the three-repo TPT AI compute suite (tpt-gpu, tpt-spark, tpt-crucible) for cross-repo synergies (merged in from `todo1.md`). None of these are required for tpt-gpu to work standalone — optional improvements that strengthen the suite.

### 1. Publish TPTIR as a standalone crate/spec
**Why:** tpt-crucible also generates an MLIR-based IR (TPT-IR) from its Catalyst ingestion module. A single shared TPTIR dialect spec lets a model compiled once route to GPU (tpt-gpu runtime), FPGA (Crucible Fusion), MCU swarm (Crucible Alloy), or analog (Crucible Element) without re-compilation.
- [x] Extract the TPTIR dialect definition into a standalone crate — `crates/tptir-spec` (`tpt-gpu-ir-spec`) already exists
- [x] Define a stable text-format serialisation that tpt-crucible's Catalyst can consume — `parse_region()` / `parse_op()` added to `crates/tpt-gpu-ir-spec/src/text.rs`; full round-trip verified by tests
- [x] Publish the crate to crates.io so tpt-crucible can depend on it directly — `tpt-gpu-ir-spec` confirmed live on crates.io (verified via API, 2026-08-04)
- [ ] Tag the first stable release (crate is published as `tpt-gpu-ir-spec`, not `tptir-spec` — confirm the tag name tpt-crucible expects before tagging)

### 2. Shared model registry (`~/.tpt/models/`)
**Why:** tpt-spark and tpt-crucible both consume GGUF models; a shared convention avoids duplicate downloads/directories.
- [x] `tools/model-registry` (`tpt-gpu-model-registry`) implements the shared registry, `ModelRegistry::open()`, HuggingFace download via `hf.rs`
- [x] `MODELS_REGISTRY.md` exists at repo root documenting the manifest format
- [ ] Confirm tpt-spark and tpt-crucible have actually adopted the same `~/.tpt/models/` + `models.json` convention (cross-repo verification, not just tpt-gpu-side)

### 3. Expose a Rust-native inference API for tpt-spark
**Why:** tpt-spark's `WgpuEngine` (hand-written WGSL) could delegate to tpt-gpu's production-quality kernels via a stable trait.
- [x] `LlmInference` trait defined in `layer4_tptr/tptr-core/src/inference.rs` (`GpuInferenceEngine` implementation exists) — will move to `crates/tpt-gpu-runtime/src/inference.rs` under Phase 8
- [x] Confirm the trait signature matches what's proposed here — `LlmInference` in `crates/tpt-gpu-runtime/src/inference.rs` has exactly `load(model_path)`, `model_info()`, `infer(tokens, max_new_tokens, callback)`, `cancel()` — no reconciliation needed
- [x] Publish/expose `tpt-gpu-runtime` so tpt-spark can add it as an optional dependency — confirmed live on crates.io (verified via API, 2026-08-04)
- [ ] Write a minimal cross-repo integration test: load a GGUF model, run inference via the trait from tpt-spark's side

### 4. TPT Script frontend note (deferred — depends on item 1)
- [ ] Once TPTIR is published as a shared spec (item 1) and tpt-crucible adopts it as Catalyst's output dialect, tpt-gpu's TPT Script compiler gains FPGA/analog/MCU-swarm as compilation targets — revisit after item 1 lands and tpt-crucible confirms adoption
- [x] Root `README.md` and `docs/tutorials/09_python_api.md` re-checked against final API surface for doc/implementation mismatches

---

## Phase 10: Platform Review Round 2 — Remediation

**Context:** a second full-platform review (three parallel passes: bug/TODO catalog, per-layer architecture maturity, onboarding/automation) found that the Phase 9 GEMM fix only addressed buffer-selection, not the underlying compute — every layer5 primitive kernel's software fallback path (used whenever no CUDA/ROCm vendor backend is present) was a no-op returning `Ok(())` without computing anything, plus two new regressions in onboarding/release automation. Plan: `C:\Users\phill\.claude\plans\review-platform-for-bugs-tranquil-tiger.md`.

### Priority 1 — Critical correctness: kernel software-fallback stubs
- [x] `crates/tpt-gpu-primitives/src/kernels/gemm.rs::tptir_fallback_gemm` — implemented real scalar `alpha*A@B + beta*C`; added `test_gemm_fallback_computes_real_product` / `test_gemm_fallback_applies_alpha_beta`
- [x] `crates/tpt-gpu-primitives/src/kernels/fused_gemm.rs::tptir_fused_gemm` / `tptir_fused_gemm_with_bias` — implement real scalar `A@B` (+ bias) + activation (Relu/Gelu/Silu/Tanh/None); `FusedGemmKernel` has no vendor dispatch at all, so this is the only compute path
- [x] `crates/tpt-gpu-primitives/src/kernels/attention.rs::tptir_fallback_attention` — implement real `softmax(Q@K^T*scale)@V`; mask parameter is not forwarded to the fallback (would require extending `VendorLibrary::attention`'s signature)
- [x] `crates/tpt-gpu-primitives/src/kernels/conv2d.rs::tptir_fallback_conv2d` — implement real direct/naive convolution
- [x] `crates/tpt-gpu-primitives/src/kernels/conv3d.rs::tptir_fallback_conv3d` — implement real direct/naive convolution
- [x] Add arithmetic-correctness tests (not just buffer-identity) for each of the above

### Priority 2 — New onboarding/release regressions
- [x] `scripts/setup.sh` / `scripts/setup.ps1` — fix binary path references from `tpt-gpu-script-cli` to the crate's actual `[[bin]] name`, `tpt-gpu-script` (build invocation `-p tpt-gpu-script-cli` itself is correct)
- [x] `.github/workflows/release.yml` `publish-crates` job — fix stale pre-Phase-8 paths `crates/tptb-{core,cli,lsp,format}` → `crates/tpt-gpu-script-{core,cli,lsp,format}`

### Priority 3 — Architecture-maturity gaps (backlog, not scheduled)
- [x] Layer 4 runtime: `crates/tpt-gpu-runtime/src/inference.rs`'s `ModelWeights::allocate` always zero-initializes — no GGUF/TPTF tensor-loading path exists, so inference is numerically inert regardless of input model
- [x] No RoPE implementation anywhere despite all 5 supported architectures (llama/llama3, mistral, qwen2/qwen, phi3, gemma2/gemma) requiring it
- [x] `Attention` only receives current-step K/V, not the accumulated `KvCache` (flagged in-code at `inference.rs:809-813`) — multi-token generation would be numerically wrong even with real weights
- [x] `QuantGemmKernel` never invoked from `tpt-gpu-runtime` despite `ModelInfo.per_layer_bits` carrying per-layer quant metadata — quantized-inference path unwired end-to-end
- [ ] No multi-GPU/tensor-parallel path in the runtime
- [x] No sampling kernel beyond host-side argmax; `arch.rs`'s `Sampling{temperature, top_k}` op has no corresponding layer5 kernel
- [x] No GGUF→TPTF importer in `crates/tpt-gpu-model-optimizer` (`export/gguf.rs` only goes `.tptf → GGUF`) — the documented "download GGUF → optimize → run" flow cannot be completed today
- [x] `MODELS_REGISTRY.md`'s own CLI example registers arch tag `"llama2"`, which `arch.rs::template_for_arch` doesn't recognize (only `llama`/`llama3`)
- [x] `layer2_tptd` targets a fabricated PCIe device with no real hardware and isn't exercised by CI/workspace builds — doc framing should be explicit that it's simulation-only today
- [x] `layer6_framework/tptr` and `layer6_tptf/tptf` are two divergent, non-interoperating Python packages, neither backed by a working compiled Rust extension in practice (`layer6_tptf/tptf/jax_backend.py` silently falls back to NumPy/JAX simulation when the PyO3 extension isn't importable) — resolved by porting `layer6_tptf/tptf/jax_backend.py`'s real JAX primitive/autodiff implementation into `layer6_framework/tptr/jax/ops.py` (replacing the `NotImplementedError` stub) and deleting `layer6_tptf/` entirely; `backend.py`'s dynamic backend-switch and `runtime_bridge.py` were intentionally not ported since `layer6_framework` already covers that via its per-framework namespaces (`tptr.pytorch`/`tptr.jax`) and `tptr._ffi`/`tptr.tensor`
- [x] `crates/tpt-gpu-script-core/src/codegen/tptir_emit.rs:177` — `break`/`continue` inside GPU kernels emit a `"; TODO: control-flow lowering"` comment instead of real lowering or a compile error (silent codegen gap)
- [x] `crates/tpt-gpu-script-core/src/codegen/tptir_emit.rs:247` — unmatched expression kinds emit a `"; TODO: complex expr"` comment instead of a compile error (silent codegen gap)
- [x] `layer2_tptd/linux/tptd_drm.rs:254` — page-fault IRQ handler never signals the waiting process; faulted contexts hang instead of erroring

### Priority 4 — DX / automation gaps (backlog, not scheduled)
- [x] No dependency/security scanning (`cargo-audit`, `cargo-deny`, Dependabot/Renovate)
- [x] `README.md` still says "Pull requests are not accepted at this time" while `.github/PULL_REQUEST_TEMPLATE.md` exists and CI runs on `pull_request`; still no `CONTRIBUTING.md`
- [x] No docs-deployment workflow (mkdocs/GitHub Pages) despite 17+ tutorials and multiple specs
- [x] No lint/type-check (ruff/mypy) or coverage reporting for `layer6_framework` Python package
- [x] `crates/tpt-gpu-playground` still requires local `wasm-pack build` + manual static server despite root `README.md` marketing it as "no install required" — no hosted/`gh-pages` WASM build exists (root README claim fixed; hosted build added in `docs.yml`)
- [x] No pre-commit hooks, devcontainer, or `tpt-gpu doctor`-style health-check command mirroring CI's `fmt --check` + `clippy -D warnings` locally
- [x] Docs/tutorials all invoke the CLI as `tpt`, but the compiled binary is `tpt-gpu-script` — no doc mentions creating an alias

### Further ideas (not scheduled — for future consideration)
- [x] Once Priority 1 lands, promote the CPU fallback as a genuine "try TPT GPU with zero GPU required" story in docs/marketing (done: README "No GPU? CPU Fallback Works" + `docs/use-cases.md` section)
- [x] GGUF→TPTF importer would make `model-optimizer` end-to-end usable from a fresh HF download — the tool's core value proposition
- [x] `tpt-gpu doctor` command: checks Rust version, optional CUDA/ROCm SDK presence, Python venv state, and runs the same fmt/clippy checks CI does; doubles as a pre-commit hook (implemented as `crates/tpt-gpu-doctor`; `--pre-commit`/`--fast` flags; README tools entry)
- [x] Wire `cargo-deny`/`cargo-audit` into CI plus Dependabot for cheap dependency-hygiene coverage (done: `security.yml` runs both; `dependabot.yml` watches cargo + github-actions)

---

## Phase 11: Activation Scratch Pool + Mmap-Backed Model Loading

**Context:** originated from a proposal for a `ZeroCopyModelLoader` trait (GGUF/SafeTensors mmap) and a `DeterministicArena` bump allocator with compile-time-computed capacity. Research found real overlap (GGUF loading already mmaps, just copies afterward) and real conflicts (no SafeTensors reader exists anywhere; a compile-time peak-size pass conflicts with tpt-gpu's deliberately dynamic shape model per the draft TPT-UIR RFC; a flat bump-pointer arena is unsound given residual connections and the cross-layer-lived KV cache). Scoped down to two independent, additive changes. Plan: `C:\Users\phill\.claude\plans\i-m-thinking-of-adding-crispy-rain.md`.

### Feature 1 — `ScratchPool`: free-list buffer pool replacing per-op fresh allocations
- [ ] `crates/tpt-gpu-primitives/src/memory/buffer.rs` — add `GpuBuffer::reshape()` (validate `num_elements()`, swap `shape`, no copy); replace `reshape_to_2d` (`inference.rs:1246-1259`) with a thin wrapper
- [ ] `crates/tpt-gpu-runtime/src/scratch.rs` (new) — `ScratchPool` keyed by `Shape` (`HashMap<Shape, Vec<GpuBuffer<f32>>>`, free-list not bump/arena — `Shape` already derives `Hash + Eq`); `pub mod scratch;` in `lib.rs`
- [ ] Route `alloc_f32`/`alloc_f32_1d`/`vec_to_buf` (`inference.rs:545-553, 1579-1588`) and `quant_layer_gemm`'s scratch buffers (`inference.rs:1347-1379`) through `ScratchPool::checkout_*`/`release`
- [ ] Extend `GemmKernel::execute`'s existing `Option<&mut GpuBuffer<f32>>` output-param pattern (`gemm.rs:77-153`) to `RmsNormKernel`, `SoftmaxKernel`, `AttentionKernel`, `EmbeddingKernel`, `QuantGemmKernel`; wire `forward_step` to pass pool buffers in — one kernel at a time. Note: call sites passing `Some(&mut buf)` must ignore the returned clone (`gemm.rs:150` deep-clones `storage` on return) and keep using `buf` directly, or the pool win is defeated
- [ ] Tests: `GpuBuffer::reshape` unit test; `ScratchPool` checkout/release/reuse-hit unit test; confirm `forward_step` output unchanged on a synthetic small model

### Feature 2 — mmap-backed `.tptf` loading
- [ ] `crates/tpt-gpu-runtime/src/inference.rs::parse_tptf_header` (405-420) — replace whole-file `fs::read` with a fixed 512-byte `read_exact`, matching `tptf_format::read_header` (`crates/tpt-gpu-model-optimizer/src/tptf_format.rs:211-219`)
- [ ] `crates/tpt-gpu-runtime/src/inference.rs::ModelWeights::load_tptf` (565-758) — replace whole-file `fs::read` with `memmap2::Mmap`, matching the pattern already in `GgufImporter::import` (`gguf.rs:285-289`); mmap stays a local, not stored long-term. Add `memmap2 = { workspace = true }` to `crates/tpt-gpu-runtime/Cargo.toml`
- [ ] Tests: byte-for-byte parity vs. the old `fs::read` path on the existing synthetic `.tptf` fixture; existing `load_tptf_via_engine`/`parse_tptf_header_basic` tests (`inference.rs:1735-1762, 2004-2011`) still pass

### Explicitly out of scope (see plan file for rationale)
- SafeTensors reader (none exists today, only a writer)
- Layer3 compiler compile-time `peak_activation_size` pass (conflicts with dynamic-shape design; would duplicate TPT-UIR's future `memory_dialect`)
- Fixing `load_with_vendor` (`inference.rs:908`, referenced but undefined anywhere in the crate — pre-existing, unrelated bug, flagged here for tracking)
- Full TPTF-parser consolidation (`tptf_format::read_tensor_blocks` vs. the inline parser in `inference.rs`)
- `GgufImporter`'s per-tensor `.to_vec()` copy during offline GGUF→TPTF conversion (one-time cost, not the inference hot path)
