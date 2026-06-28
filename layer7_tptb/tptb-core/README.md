# tptb-core

TPT Script compiler frontend — lexer, AST, and parser.

This crate provides the core compiler infrastructure for TPT Script, including:

- **Lexer** — Tokenization of TPT Script source code
- **Parser** — Recursive-descent parser producing AST
- **AST** — Abstract syntax tree node types
- **Semantic Analysis** — Type checker, constraint evaluator, metadata extraction
- **Introspection API** — Runtime API for querying operation metadata

## Usage

```rust
use tptb_core::{compile_str, type_check, compile_full};

// Parse source code
let program = compile_str(source)?;

// Type-check
let checker = type_check(&program);
if !checker.errors.is_empty() {
    // Handle errors
}

// Full compilation (parse + type-check + codegen)
let (checker, output) = compile_full(source)?;
println!("Rust: {}", output.rust_source);
println!("TPTIR: {}", output.tptir_source);
```

## Features

- **Dual Compilation** — Generates both Rust (host) and TPTIR (GPU kernels)
- **Tensor Shape Inference** — Automatic shape propagation with symbolic dimensions
- **Constraint Checking** — Compile-time validation via `@constraint` annotations
- **Rich Metadata** — Every operation has `@doc`, `@input`, `@output`, `@complexity`
- **Structured Errors** — Error codes, locations, suggestions, and auto-fixes

## License

Licensed under Apache 2.0 WITH LLVM-exception. See [LICENSE](../../LICENSE) for details.