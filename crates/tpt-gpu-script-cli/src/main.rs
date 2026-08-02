// tpt � TPT Script compiler CLI
//
// Usage:
//   tpt new     <name>                  Create a new TPT Script project
//   tpt init                            Initialize project in current directory
//   tpt check   <file>                  Type-check without emitting output
//   tpt compile <file> [-o <out>]       Compile to Rust + TPTIR sources
//   tpt inspect <op>                    Print JSON schema for a built-in op
//   tpt run     <file>                  Compile and show generated output
//   tpt ops                             List all built-in operation names
//   tpt modules                         List all standard library modules
//   tpt docs    <op>                    Print Markdown docs for a built-in op
//   tpt compat  <file>                  Generate Python compatibility stubs
//   tpt --version                       Print the compiler version

use std::{env, fs, process};

use tpt_gpu_script_core::{compile_full, compile_str, introspect, modules, DocFormat};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    let exit = run(&args);
    process::exit(exit);
}

fn run(args: &[String]) -> i32 {
    match args.get(1).map(String::as_str) {
        Some("--version") | Some("-V") => {
            println!("tpt {}", env!("CARGO_PKG_VERSION"));
            0
        }
        Some("--help") | Some("-h") | None => {
            print_help();
            0
        }
        Some("new") => cmd_new(args),
        Some("init") => cmd_init(),
        Some("check") => cmd_check(args),
        Some("compile") => cmd_compile(args),
        Some("inspect") => cmd_inspect(args),
        Some("run") => cmd_run(args),
        Some("ops") => cmd_ops(),
        Some("modules") => cmd_modules(),
        Some("docs") => cmd_docs(args),
        Some("compat") => cmd_compat(args),
        Some(cmd) => {
            eprintln!("tpt: unknown subcommand `{cmd}`");
            eprintln!("Run `tpt --help` for usage.");
            1
        }
    }
}

// ---------------------------------------------------------------------------
// new � Create a new TPT Script project
// ---------------------------------------------------------------------------

fn cmd_new(args: &[String]) -> i32 {
    let name = match args.get(2) {
        Some(n) => n.clone(),
        None => {
            eprintln!("tpt new: missing <name>");
            eprintln!("Usage: tpt new <name>");
            return 2;
        }
    };

    let path = env::current_dir().unwrap_or_default().join(&name);

    if path.exists() {
        eprintln!("tpt new: directory `{}` already exists", path.display());
        return 1;
    }

    if let Err(e) = fs::create_dir_all(path.join("src")) {
        eprintln!("tpt new: cannot create directory: {e}");
        return 1;
    }

    let toml = modules::generate_project_toml(&name);
    if let Err(e) = fs::write(path.join("tpt.toml"), toml) {
        eprintln!("tpt new: cannot write tpt.toml: {e}");
        return 1;
    }

    let main_tpts = modules::generate_main_tpts();
    if let Err(e) = fs::write(path.join("src/main.tpts"), main_tpts) {
        eprintln!("tpt new: cannot write src/main.tpts: {e}");
        return 1;
    }

    let gitignore = modules::generate_gitignore();
    if let Err(e) = fs::write(path.join(".gitignore"), gitignore) {
        eprintln!("tpt new: cannot write .gitignore: {e}");
        return 1;
    }

    println!(
        "Created TPT Script project `{}` at {}",
        name,
        path.display()
    );
    println!("");
    println!("To get started:");
    println!("  cd {}", name);
    println!("  tpt check src/main.tpts");
    println!("  tpt compile src/main.tpts -o output.rs");
    println!("  tpt run src/main.tpts");

    0
}

// ---------------------------------------------------------------------------
// init � Initialize project in current directory
// ---------------------------------------------------------------------------

fn cmd_init() -> i32 {
    let cwd = env::current_dir().unwrap_or_default();

    if cwd.join("tpt.toml").exists() {
        eprintln!("tpt init: tpt.toml already exists in current directory");
        return 1;
    }

    let name = cwd
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("tpt-project");

    let toml = modules::generate_project_toml(name);
    if let Err(e) = fs::write(cwd.join("tpt.toml"), toml) {
        eprintln!("tpt init: cannot write tpt.toml: {e}");
        return 1;
    }

    if !cwd.join("src").exists() {
        if let Err(e) = fs::create_dir_all(cwd.join("src")) {
            eprintln!("tpt init: cannot create src directory: {e}");
            return 1;
        }
    }

    let main_tpts = modules::generate_main_tpts();
    if let Err(e) = fs::write(cwd.join("src/main.tpts"), main_tpts) {
        eprintln!("tpt init: cannot write src/main.tpts: {e}");
        return 1;
    }

    println!(
        "Initialized TPT Script project `{}` in current directory",
        name
    );
    println!("");
    println!("To get started:");
    println!("  tpt check src/main.tpts");
    println!("  tpt compile src/main.tpts -o output.rs");

    0
}

// ---------------------------------------------------------------------------
// modules � List all standard library modules
// ---------------------------------------------------------------------------

fn cmd_modules() -> i32 {
    println!("TPT Script Standard Library Modules");
    println!("====================================");
    println!();

    for module in modules::std_library_modules() {
        println!("  {:20} {}", module.path, module.description);
        if !module.operations.is_empty() {
            let ops: Vec<&str> = module.operations.iter().take(8).copied().collect();
            println!(
                "    Operations: {} ({} total)",
                ops.join(", "),
                module.operations.len()
            );
        }
        println!();
    }

    println!("Import in TPT Script:");
    println!("  import tpt");
    println!("  import tpt.nn");
    println!("  import tpt.optim");
    println!("  import tpt.data");
    println!("  import tpt.io");
    println!("  import tpt.dist");
    println!("  import tpt.compat");
    println!("  import tpt.introspect");

    0
}

// ---------------------------------------------------------------------------
// compat � Generate Python compatibility stubs
// ---------------------------------------------------------------------------

fn cmd_compat(args: &[String]) -> i32 {
    let path = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("tpt compat: missing <file>");
            eprintln!("Usage: tpt compat <file> [-o <out.pyi>]");
            return 2;
        }
    };

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("tpt compat: cannot read `{path}`: {e}");
            return 2;
        }
    };

    let program = match compile_str(&source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    let mut stub = String::from("# Generated Python type stubs for TPT Script\n");
    stub.push_str("# This file is auto-generated by `tpt compat`\n\n");
    stub.push_str("from typing import Any, Optional, List, Tuple\n");
    stub.push_str("import numpy as np\n\n");

    for item in &program.items {
        if let tpt_gpu_script_core::ast::Item::Function(f) = &item {
            stub.push_str(&format!("def {}(", f.name));
            let params: Vec<String> = f
                .params
                .iter()
                .map(|p| format!("{}: Any", p.name))
                .collect();
            stub.push_str(&params.join(", "));
            stub.push_str(") -> Any: ...\n\n");
        }
    }

    let output_arg = find_flag(args, "-o");
    match output_arg {
        Some(out_path) => {
            if let Err(e) = fs::write(&out_path, &stub) {
                eprintln!("tpt compat: cannot write `{out_path}`: {e}");
                return 1;
            }
            println!("Generated Python stubs: {out_path}");
        }
        None => {
            print!("{stub}");
        }
    }

    0
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------

fn cmd_check(args: &[String]) -> i32 {
    let path = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("tpt check: missing <file>");
            return 2;
        }
    };

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("tpt check: cannot read `{path}`: {e}");
            return 2;
        }
    };

    let program = match compile_str(&source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    let unresolved = modules::validate_imports(&program);
    for issue in &unresolved {
        eprintln!("warning: {issue}");
    }

    let checker = tpt_gpu_script_core::type_check(&program);
    if checker.errors.is_empty() && unresolved.is_empty() {
        println!("{path}: ok � 0 errors");
        return 0;
    }

    let n = checker.errors.len();
    for err in &checker.errors {
        print_error(err);
    }
    if !unresolved.is_empty() {
        eprintln!("\n{} import issue(s) found.", unresolved.len());
    }
    if n > 0 {
        eprintln!("\n{n} error(s) found.");
    }
    1
}

// ---------------------------------------------------------------------------
// compile
// ---------------------------------------------------------------------------

fn cmd_compile(args: &[String]) -> i32 {
    let path = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("tpt compile: missing <file>");
            return 2;
        }
    };

    let output_arg = find_flag(args, "-o");

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("tpt compile: cannot read `{path}`: {e}");
            return 2;
        }
    };

    let config = find_project_config(path);

    let (_checker, output) = match &config {
        Some(cfg) => match tpt_gpu_script_core::compile_project(&source, cfg) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("error: {e}");
                return 1;
            }
        },
        None => match compile_full(&source) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("error: {e}");
                return 1;
            }
        },
    };

    match output_arg {
        Some(out_path) => {
            let combined = format!(
                "{}\n\n// === TPTIR Output ===\n\n{}",
                output.rust_source, output.tptir_source
            );
            if let Err(e) = fs::write(&out_path, &combined) {
                eprintln!("tpt compile: cannot write `{out_path}`: {e}");
                return 1;
            }
            println!("Compiled: {out_path}");
        }
        None => {
            print!("{}", output.rust_source);
            if !output.tptir_source.is_empty() {
                println!("\n// === TPTIR Output ===\n");
                print!("{}", output.tptir_source);
            }
        }
    }

    0
}

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------

fn cmd_inspect(args: &[String]) -> i32 {
    let op = match args.get(2) {
        Some(o) => o,
        None => {
            eprintln!("tpt inspect: missing <op>");
            return 2;
        }
    };

    match introspect::get_schema(op) {
        Some(schema) => {
            println!("{}", schema.to_json());
            0
        }
        None => {
            eprintln!("tpt inspect: unknown operation `{op}`");
            eprintln!("Run `tpt ops` to list available operations.");
            1
        }
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: &[String]) -> i32 {
    let path = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("tpt run: missing <file>");
            return 2;
        }
    };

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("tpt run: cannot read `{path}`: {e}");
            return 2;
        }
    };

    let (checker, output) = match compile_full(&source) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    if !checker.errors.is_empty() {
        for err in &checker.errors {
            print_error(err);
        }
        eprintln!(
            "\n{} error(s) found. Output may be incomplete.",
            checker.errors.len()
        );
    }

    println!("=== Rust Output ===");
    println!("{}", output.rust_source);

    if !output.tptir_source.is_empty() {
        println!("\n=== TPTIR Output ===");
        println!("{}", output.tptir_source);
    }

    0
}

// ---------------------------------------------------------------------------
// ops
// ---------------------------------------------------------------------------

fn cmd_ops() -> i32 {
    let ops = introspect::list_operations();
    for op in &ops {
        println!("{op}");
    }
    println!("\nTotal: {} operations", ops.len());
    0
}

// ---------------------------------------------------------------------------
// docs
// ---------------------------------------------------------------------------

fn cmd_docs(args: &[String]) -> i32 {
    let op = match args.get(2) {
        Some(o) => o,
        None => {
            eprintln!("tpt docs: missing <op>");
            return 2;
        }
    };

    let format_arg = find_flag(args, "-f");
    let format = match format_arg.as_deref() {
        Some("pyi") => DocFormat::Pyi,
        _ => DocFormat::Markdown,
    };

    let docs = introspect::generate_docs(op, format);
    if docs.is_empty() {
        eprintln!("tpt docs: unknown operation `{op}`");
        return 1;
    }
    print!("{docs}");
    0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn print_help() {
    println!("tpt � TPT Script compiler {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("USAGE:");
    println!("    tpt <SUBCOMMAND>");
    println!();
    println!("SUBCOMMANDS:");
    println!("    new <name>        Create a new TPT Script project");
    println!("    init              Initialize project in current directory");
    println!("    check <file>      Type-check without emitting output");
    println!("    compile <file>    Compile to Rust + TPTIR sources");
    println!("    inspect <op>      Print JSON schema for a built-in op");
    println!("    run <file>        Compile and show generated output");
    println!("    ops               List all built-in operation names");
    println!("    modules           List all standard library modules");
    println!("    docs <op>         Print Markdown docs for a built-in op");
    println!("    compat <file>     Generate Python compatibility stubs");
    println!();
    println!("OPTIONS:");
    println!("    -o <path>         Output file path");
    println!("    -f <format>       Output format (markdown, pyi)");
    println!("    --version, -V     Print version");
    println!("    --help, -h        Print help");
    println!();
    println!("EXAMPLES:");
    println!("    tpt new my-project");
    println!("    tpt check src/main.tpts");
    println!("    tpt compile src/main.tpts -o output.rs");
    println!("    tpt inspect matmul");
    println!("    tpt modules");
    println!("    tpt docs attention");
}

fn print_error(err: &tpt_gpu_script_core::errors::TptError) {
    eprintln!("error [{}]: {}", err.code, err.message);
    if let Some(fix) = &err.fix_code {
        eprintln!("  fix: {fix}");
    }
    if let Some(suggestion) = &err.suggestion {
        eprintln!("  suggestion: {suggestion}");
    }
}

fn find_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn find_project_config(source_path: &str) -> Option<modules::ProjectConfig> {
    let mut path = std::path::Path::new(source_path).parent()?;
    loop {
        let toml_path = path.join("tpt.toml");
        if toml_path.exists() {
            if let Ok(content) = fs::read_to_string(&toml_path) {
                if let Ok(config) = modules::ProjectConfig::from_toml(&content) {
                    return Some(config);
                }
            }
        }
        path = path.parent()?;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_version_flag() {
        assert_eq!(run(&args(&["tpt", "--version"])), 0);
    }

    #[test]
    fn test_help_flag() {
        assert_eq!(run(&args(&["tpt", "--help"])), 0);
    }

    #[test]
    fn test_no_args_shows_help() {
        assert_eq!(run(&args(&["tpt"])), 0);
    }

    #[test]
    fn test_unknown_subcommand() {
        assert_eq!(run(&args(&["tpt", "frobnicate"])), 1);
    }

    #[test]
    fn test_ops_lists_operations() {
        assert_eq!(run(&args(&["tpt", "ops"])), 0);
    }

    #[test]
    fn test_modules_lists_modules() {
        assert_eq!(run(&args(&["tpt", "modules"])), 0);
    }

    #[test]
    fn test_inspect_known_op() {
        assert_eq!(run(&args(&["tpt", "inspect", "matmul"])), 0);
    }

    #[test]
    fn test_inspect_unknown_op() {
        assert_eq!(run(&args(&["tpt", "inspect", "nonexistent_op_xyz"])), 1);
    }

    #[test]
    fn test_docs_known_op() {
        assert_eq!(run(&args(&["tpt", "docs", "matmul"])), 0);
    }

    #[test]
    fn test_check_missing_file_arg() {
        assert_eq!(run(&args(&["tpt", "check"])), 2);
    }

    #[test]
    fn test_compile_missing_file_arg() {
        assert_eq!(run(&args(&["tpt", "compile"])), 2);
    }

    #[test]
    fn test_run_missing_file_arg() {
        assert_eq!(run(&args(&["tpt", "run"])), 2);
    }

    #[test]
    fn test_inspect_missing_op_arg() {
        assert_eq!(run(&args(&["tpt", "inspect"])), 2);
    }

    #[test]
    fn test_docs_missing_op_arg() {
        assert_eq!(run(&args(&["tpt", "docs"])), 2);
    }

    #[test]
    fn test_new_missing_name_arg() {
        assert_eq!(run(&args(&["tpt", "new"])), 2);
    }

    #[test]
    fn test_compat_missing_file_arg() {
        assert_eq!(run(&args(&["tpt", "compat"])), 2);
    }

    #[test]
    fn test_find_flag_present() {
        let a = args(&["tpt", "compile", "foo.tpts", "-o", "out"]);
        assert_eq!(find_flag(&a, "-o"), Some("out".to_string()));
    }

    #[test]
    fn test_find_flag_absent() {
        let a = args(&["tpt", "compile", "foo.tpts"]);
        assert_eq!(find_flag(&a, "-o"), None);
    }

    #[test]
    fn test_check_valid_source_via_api() {
        let source = r#"
fn add(a: f32, b: f32) -> f32 {
    return a + b
}
"#;
        let program = compile_str(source).expect("parse failed");
        let checker = tpt_gpu_script_core::type_check(&program);
        assert!(checker.errors.is_empty(), "{:?}", checker.errors);
    }

    #[test]
    fn test_check_bad_source_via_api() {
        let source = r#"
fn f() -> f32 {
    return true
}
"#;
        let program = compile_str(source).expect("parse failed");
        let checker = tpt_gpu_script_core::type_check(&program);
        assert!(!checker.errors.is_empty());
    }

    #[test]
    fn test_compile_simple_fn_via_api() {
        let source = r#"
import tpt
fn relu_add(x: Tensor[f32, m, n], y: Tensor[f32, m, n]) -> Tensor[f32, m, n] {
    let r = tpt.relu(x)
    let s = r + y
    return s
}
"#;
        let (checker, output) = compile_full(source).expect("compile_full failed");
        assert!(checker.errors.is_empty(), "{:?}", checker.errors);
        assert!(!output.rust_source.is_empty());
    }

    #[test]
    fn test_compile_project_with_config() {
        let source = r#"
import tpt
import tpt.nn
fn main() { }
"#;
        let config = modules::ProjectConfig::new("test");
        let (checker, output) =
            tpt_gpu_script_core::compile_project(source, &config).expect("compile_project failed");
        assert!(checker.errors.is_empty(), "{:?}", checker.errors);
        assert!(output.rust_source.contains("use tpt::nn"));
    }

    #[test]
    fn test_resolve_imports_with_gpu() {
        let source = r#"
import tpt.nn
fn main() { }
"#;
        let mut config = modules::ProjectConfig::new("test");
        config.features.insert("gpu".into());
        let (_, output) =
            tpt_gpu_script_core::compile_project(source, &config).expect("compile_project failed");
        assert!(output.rust_source.contains("use tpt::nn::layers::*"));
    }
}
