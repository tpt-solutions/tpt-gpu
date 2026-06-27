pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod semantic;

pub use ast::Program;
pub use codegen::{emit, CodegenOutput};
pub use lexer::{tokenize, LexError, Span, Token, TokenKind};
pub use parser::{parse, ParseError};
pub use semantic::{type_check, TypeChecker};

/// Convenience: lex and parse a TPT Script source string in one call.
pub fn compile_str(source: &str) -> Result<Program, CompileError> {
    let tokens = tokenize(source)?;
    let program = parse(tokens)?;
    Ok(program)
}

/// Full pipeline: lex → parse → type-check → codegen.
///
/// Returns `Err` if lexing or parsing fails.  Type errors are non-fatal
/// and are accessible via [`TypeChecker::errors`] on the returned checker.
pub fn compile_full(source: &str) -> Result<(TypeChecker, CodegenOutput), CompileError> {
    let program = compile_str(source)?;
    let checker = type_check(&program);
    let output  = emit(&program);
    Ok((checker, output))
}

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("lex error: {0}")]
    Lex(#[from] LexError),
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),
}
