use tower_lsp::lsp_types::*;
use tptb_core;
use crate::document::DocumentStore;

fn lex_error_to_position(e: &tptb_core::LexError) -> (u32, u32) {
    match e {
        tptb_core::LexError::UnexpectedChar { line, col, .. } => (*line, *col),
        tptb_core::LexError::UnterminatedString { line, col, .. } => (*line, *col),
        tptb_core::LexError::UnterminatedBlockComment { line, col, .. } => (*line, *col),
        tptb_core::LexError::InvalidEscape { line, col, .. } => (*line, *col),
        tptb_core::LexError::InvalidNumber { line, col, .. } => (*line, *col),
    }
}

pub fn compute_diagnostics(doc: &DocumentStore) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    match tptb_core::tokenize(&doc.source) {
        Err(e) => {
            let (line, col) = lex_error_to_position(&e);
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position { line: line - 1, character: col - 1 },
                    end: Position { line: line - 1, character: col },
                },
                severity: Some(DiagnosticSeverity::ERROR),
                code: Some(NumberOrString::String("LEX_ERROR".to_string())),
                source: Some("tptb-lsp".to_string()),
                message: e.to_string(),
                ..Default::default()
            });
            return diagnostics;
        }
        Ok(_) => {}
    }
    let program = match tptb_core::compile_str(&doc.source) {
        Err(e) => {
            let span = match &e {
                tptb_core::CompileError::Parse(pe) => &pe.span,
                _ => {
                    diagnostics.push(Diagnostic {
                        severity: Some(DiagnosticSeverity::ERROR),
                        code: Some(NumberOrString::String("COMPILE_ERROR".to_string())),
                        source: Some("tptb-lsp".to_string()),
                        message: e.to_string(),
                        ..Default::default()
                    });
                    return diagnostics;
                }
            };
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position { line: span.line - 1, character: span.col - 1 },
                    end: Position { line: span.line - 1, character: span.col },
                },
                severity: Some(DiagnosticSeverity::ERROR),
                code: Some(NumberOrString::String("PARSE_ERROR".to_string())),
                source: Some("tptb-lsp".to_string()),
                message: e.to_string(),
                ..Default::default()
            });
            return diagnostics;
        }
        Ok(p) => p,
    };
    let checker = tptb_core::type_check(&program);
    for err in &checker.errors {
        diagnostics.push(Diagnostic {
            range: Range {
                start: Position { line: err.span.line - 1, character: err.span.col - 1 },
                end: Position { line: err.span.line - 1, character: err.span.col },
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String(err.code.to_string())),
            source: Some("tptb-lsp".to_string()),
            message: err.message.clone(),
            ..Default::default()
        });
    }
    diagnostics
}