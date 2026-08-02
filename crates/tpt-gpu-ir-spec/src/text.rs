//! Stable TPTIR text-format emitter and parser.
//!
//! The text format is stable from v0.1.0: tools that consume TPTIR (e.g.
//! tpt-crucible Catalyst) can rely on this module for round-trippable output.

use crate::dialect::TPTIR_DIALECT_VERSION;
use crate::ops::Op;
use crate::types::{AddressSpace, ElemType, Type};

/// Configuration for the text emitter.
#[derive(Debug, Clone)]
pub struct EmitOptions {
    /// Emit a version header comment at the top of the output.
    pub emit_version_header: bool,
    /// Indent string for nested regions (default: two spaces).
    pub indent: String,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            emit_version_header: true,
            indent: "  ".to_string(),
        }
    }
}

/// A basic IR node used for text-format emission.
#[derive(Debug, Clone)]
pub struct Instruction {
    pub result: Option<String>,
    pub op: Op,
    pub operands: Vec<String>,
    pub attrs: Vec<(String, String)>,
    pub result_type: Option<Type>,
}

/// A basic block containing a label and a list of instructions.
#[derive(Debug, Clone)]
pub struct Block {
    pub label: String,
    pub instructions: Vec<Instruction>,
}

/// A function/kernel region for emission.
#[derive(Debug, Clone)]
pub struct Region {
    pub name: String,
    pub args: Vec<(String, Type)>,
    pub return_types: Vec<Type>,
    pub blocks: Vec<Block>,
}

/// Emit a region to TPTIR text format.
pub fn emit(region: &Region, opts: &EmitOptions) -> String {
    let mut out = String::new();

    if opts.emit_version_header {
        out.push_str(&format!(
            "// tptir-spec v{} — stable text format\n",
            TPTIR_DIALECT_VERSION
        ));
    }

    // Function signature
    let args: Vec<String> = region
        .args
        .iter()
        .map(|(name, ty)| format!("{}: {}", name, ty))
        .collect();
    let rets: Vec<String> = region.return_types.iter().map(|t| t.to_string()).collect();

    let ret_str = if rets.is_empty() {
        String::new()
    } else {
        format!(
            " -> {}",
            if rets.len() == 1 {
                rets[0].clone()
            } else {
                format!("({})", rets.join(", "))
            }
        )
    };

    out.push_str(&format!(
        "func @{}({}){} {{\n",
        region.name,
        args.join(", "),
        ret_str
    ));

    for block in &region.blocks {
        out.push_str(&format!("^{}:\n", block.label));
        for instr in &block.instructions {
            out.push_str(&opts.indent);
            if let Some(res) = &instr.result {
                out.push_str(&format!("%{} = ", res));
            }
            out.push_str(&instr.op.to_string());
            if !instr.operands.is_empty() {
                let ops: Vec<String> = instr.operands.iter().map(|o| format!("%{}", o)).collect();
                out.push_str(&format!(" {}", ops.join(", ")));
            }
            if !instr.attrs.is_empty() {
                let pairs: Vec<String> = instr
                    .attrs
                    .iter()
                    .map(|(k, v)| format!("{} = {}", k, v))
                    .collect();
                out.push_str(&format!(" {{{}}}", pairs.join(", ")));
            }
            if let Some(ty) = &instr.result_type {
                out.push_str(&format!(" : {}", ty));
            }
            out.push('\n');
        }
    }

    out.push_str("}\n");
    out
}

/// Parse a TPTIR type string into a `Type`.
///
/// Supports the minimal subset needed for cross-tool consumption:
/// scalars, vectors, tensors, and memrefs.
pub fn parse_type(s: &str) -> Result<Type, String> {
    let s = s.trim();
    match s {
        "i1" => return Ok(Type::scalar(ElemType::I1)),
        "i8" => return Ok(Type::scalar(ElemType::I8)),
        "i16" => return Ok(Type::scalar(ElemType::I16)),
        "i32" => return Ok(Type::scalar(ElemType::I32)),
        "i64" => return Ok(Type::scalar(ElemType::I64)),
        "f16" => return Ok(Type::scalar(ElemType::F16)),
        "bf16" => return Ok(Type::scalar(ElemType::BF16)),
        "f32" => return Ok(Type::scalar(ElemType::F32)),
        "f64" => return Ok(Type::scalar(ElemType::F64)),
        "index" => return Ok(Type::scalar(ElemType::Index)),
        "none" => return Ok(Type::none()),
        _ => {}
    }

    if s.starts_with("vector<") && s.ends_with('>') {
        let inner = &s[7..s.len() - 1];
        if let Some(x_pos) = inner.find('x') {
            let lanes: u32 = inner[..x_pos]
                .parse()
                .map_err(|_| format!("bad vector lanes in '{}'", s))?;
            let elem = parse_type(&inner[x_pos + 1..])?;
            return Ok(Type::vector(lanes, elem));
        }
    }

    if s.starts_with("tensor<") && s.ends_with('>') {
        return parse_shaped("tensor", &s[7..s.len() - 1]);
    }

    if s.starts_with("memref<") && s.ends_with('>') {
        return parse_shaped("memref", &s[7..s.len() - 1]);
    }

    Err(format!("unrecognized type '{}'", s))
}

fn parse_shaped(kind: &str, inner: &str) -> Result<Type, String> {
    // Split on the last 'x' that is followed by a letter (the elem type start)
    // Simple approach: find the comma for address space if present
    let (shape_and_elem, addr) = if let Some(comma) = inner.rfind(',') {
        let addr_str = inner[comma + 1..].trim();
        let addr = match addr_str {
            "shared" => AddressSpace::Shared,
            "local" => AddressSpace::Local,
            "constant" => AddressSpace::Constant,
            "generic" => AddressSpace::Generic,
            _ => AddressSpace::Global,
        };
        (&inner[..comma], addr)
    } else {
        (inner, AddressSpace::Global)
    };

    // Parse dims and elem type from "NxMxelem"
    let parts: Vec<&str> = shape_and_elem.split('x').collect();
    if parts.is_empty() {
        return Err(format!("empty {} shape", kind));
    }

    let mut dims = Vec::new();
    let mut elem_idx = parts.len() - 1;
    for (i, part) in parts.iter().enumerate() {
        if part.parse::<i64>().is_ok() || *part == "?" {
            dims.push(if *part == "?" {
                -1
            } else {
                part.parse::<i64>().unwrap()
            });
        } else {
            elem_idx = i;
            break;
        }
    }
    let elem_str = parts[elem_idx..].join("x");
    let elem = parse_type(elem_str.trim())?;

    match kind {
        "tensor" => Ok(Type::tensor(dims, elem, addr)),
        "memref" => Ok(Type::memref(dims, elem, addr)),
        _ => unreachable!(),
    }
}

/// Parse TPTIR text and return the block labels found (minimal structural parse).
pub fn parse(source: &str) -> Result<Vec<String>, String> {
    let mut labels = Vec::new();
    for line in source.lines() {
        let line = line.trim();
        if line.starts_with('^') {
            let label = line.trim_end_matches(':');
            labels.push(label[1..].to_string());
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ElemType;

    #[test]
    fn scalar_type_parse() {
        assert_eq!(parse_type("f32").unwrap(), Type::scalar(ElemType::F32));
        assert_eq!(parse_type("i64").unwrap(), Type::scalar(ElemType::I64));
    }

    #[test]
    fn vector_type_parse() {
        let t = parse_type("vector<8xf32>").unwrap();
        assert_eq!(t, Type::vector(8, Type::scalar(ElemType::F32)));
    }

    #[test]
    fn tensor_type_parse() {
        let t = parse_type("tensor<4x4xf16>").unwrap();
        assert_eq!(
            t,
            Type::tensor(
                vec![4, 4],
                Type::scalar(ElemType::F16),
                AddressSpace::Global
            )
        );
    }

    #[test]
    fn emit_round_trip() {
        let region = Region {
            name: "vector_add".to_string(),
            args: vec![
                (
                    "a".to_string(),
                    Type::tensor(
                        vec![1024],
                        Type::scalar(ElemType::F32),
                        AddressSpace::Global,
                    ),
                ),
                (
                    "b".to_string(),
                    Type::tensor(
                        vec![1024],
                        Type::scalar(ElemType::F32),
                        AddressSpace::Global,
                    ),
                ),
            ],
            return_types: vec![Type::tensor(
                vec![1024],
                Type::scalar(ElemType::F32),
                AddressSpace::Global,
            )],
            blocks: vec![Block {
                label: "entry".to_string(),
                instructions: vec![
                    Instruction {
                        result: Some("c".to_string()),
                        op: Op::Addf,
                        operands: vec!["a".to_string(), "b".to_string()],
                        attrs: vec![],
                        result_type: Some(Type::tensor(
                            vec![1024],
                            Type::scalar(ElemType::F32),
                            AddressSpace::Global,
                        )),
                    },
                    Instruction {
                        result: None,
                        op: Op::Return,
                        operands: vec!["c".to_string()],
                        attrs: vec![],
                        result_type: None,
                    },
                ],
            }],
        };

        let opts = EmitOptions::default();
        let text = emit(&region, &opts);
        assert!(text.contains("func @vector_add"));
        assert!(text.contains("^entry:"));
        assert!(text.contains("addf"));
        assert!(text.contains("return"));
    }
}
