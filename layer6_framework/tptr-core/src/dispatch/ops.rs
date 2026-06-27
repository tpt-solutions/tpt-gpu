//! Dispatch Table - Operation dispatch for framework tensor operations.
use tptr_core::error::{TptrResult, TptrError, ErrorCode};
use tptr_core::memory::MemoryAllocation;
use tptr_core::kernel::KernelConfig;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Handle for a registered operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpHandle(pub u64);

/// Error type for dispatch operations.
#[derive(Debug, Clone)]
pub struct DispatchError {
    pub code: u32,
    pub message: String,
    pub op_name: String,
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DispatchError({}, {}): {}", self.code, self.op_name, self.message)
    }
}

/// Operation type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    ElementwiseBinary, ElementwiseUnary, Reduction, Matmul,
    Convolution, Softmax, LayerNorm, Activation, Memory, Custom,
}

impl OpType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ElementwiseBinary => "elementwise_binary",
            Self::ElementwiseUnary => "elementwise_unary",
            Self::Reduction => "reduction",
            Self::Matmul => "matmul",
            Self::Convolution => "convolution",
            Self::Softmax => "softmax",
            Self::LayerNorm => "layer_norm",
            Self::Activation => "activation",
            Self::Memory => "memory",
            Self::Custom => "custom",
        }
    }
}

/// Operation metadata.
#[derive(Debug, Clone)]
pub struct OpMetadata {
    pub op_type: OpType,
    pub name: String,
    pub input_count: usize,
    pub output_count: usize,
    pub supports_inplace: bool,
    pub is_commutative: bool,
}
/// Dispatch table for tensor operations.
#[derive(Debug)]
pub struct DispatchTable {
    operations: HashMap<String, Operation>,
    next_op_id: AtomicU64,
}

impl DispatchTable {
    pub fn new() -> Self {
        Self { operations: HashMap::new(), next_op_id: AtomicU64::new(1) }
    }
    pub fn register(&mut self, name: impl Into<String>, op: Operation) -> OpHandle {
        let id = self.next_op_id.fetch_add(1, Ordering::SeqCst);
        self.operations.insert(name.into(), op);
        OpHandle(id)
    }
    pub fn lookup(&self, name: &str) -> Option<&Operation> {
        self.operations.get(name)
    }
    pub fn operations(&self) -> Vec<&str> {
        self.operations.keys().map(|s| s.as_str()).collect()
    }
    pub fn len(&self) -> usize { self.operations.len() }
    pub fn is_empty(&self) -> bool { self.operations.is_empty() }
    pub fn dispatch(&self, name: &str, inputs: &[&MemoryAllocation], outputs: &mut [&mut MemoryAllocation]) -> TptrResult<OpHandle> {
        let op = self.operations.get(name)
            .ok_or_else(|| TptrError::new(ErrorCode::InvalidKernel, format!("Unknown op: {}", name)))?;
        if inputs.len() != op.metadata.input_count {
            return Err(TptrError::new(ErrorCode::ArgumentMismatch,
                format!("Op '{}' expects {} inputs, got {}", name, op.metadata.input_count, inputs.len())));
        }
        Ok(OpHandle(self.next_op_id.fetch_add(1, Ordering::SeqCst)))
    }
}

impl Default for DispatchTable {
    fn default() -> Self {
        let mut table = Self::new();
        table.register("add", Operation::new(OpMetadata { op_type: OpType::ElementwiseBinary, name: "add".into(), input_count: 2, output_count: 1, supports_inplace: true, is_commutative: true }).with_kernel("tpt_add"));
        table.register("mul", Operation::new(OpMetadata { op_type: OpType::ElementwiseBinary, name: "mul".into(), input_count: 2, output_count: 1, supports_inplace: false, is_commutative: true }).with_kernel("tpt_mul"));
        table.register("sub", Operation::new(OpMetadata { op_type: OpType::ElementwiseBinary, name: "sub".into(), input_count: 2, output_count: 1, supports_inplace: false, is_commutative: false }).with_kernel("tpt_sub"));
        table.register("relu", Operation::new(OpMetadata { op_type: OpType::Activation, name: "relu".into(), input_count: 1, output_count: 1, supports_inplace: true, is_commutative: false }).with_kernel("tpt_relu"));
        table.register("gelu", Operation::new(OpMetadata { op_type: OpType::Activation, name: "gelu".into(), input_count: 1, output_count: 1, supports_inplace: false, is_commutative: false }).with_kernel("tpt_gelu"));
        table.register("softmax", Operation::new(OpMetadata { op_type: OpType::Softmax, name: "softmax".into(), input_count: 1, output_count: 1, supports_inplace: false, is_commutative: false }).with_kernel("tpt_softmax"));
        table.register("matmul", Operation::new(OpMetadata { op_type: OpType::Matmul, name: "matmul".into(), input_count: 2, output_count: 1, supports_inplace: false, is_commutative: false }).with_kernel("tpt_matmul"));
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dispatch_table_default() {
        let table = DispatchTable::default();
        assert!(table.len() >= 6);
        assert!(table.lookup("add").is_some());
    }
    #[test]
    fn test_dispatch_lookup() {
        let table = DispatchTable::default();
        let op = table.lookup("relu").unwrap();
        assert_eq!(op.metadata.op_type, OpType::Activation);
    }
}


/// A registered operation.
pub struct Operation {
    pub metadata: OpMetadata,
    pub kernel_name: Option<String>,
    pub preferred_block_size: Option<KernelConfig>,
}

impl Operation {
    pub fn new(metadata: OpMetadata) -> Self {
        Self { metadata, kernel_name: None, preferred_block_size: None }
    }
    pub fn with_kernel(mut self, name: impl Into<String>) -> Self {
        self.kernel_name = Some(name.into()); self
    }
}

