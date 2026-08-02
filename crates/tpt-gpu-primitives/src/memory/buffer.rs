//! GPU Buffer Types
use crate::error::{TptpError, TptpResult};
use bytemuck::Pod;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    Bool = 0,
    I8 = 1,
    I16 = 2,
    I32 = 3,
    I64 = 4,
    F16 = 5,
    BF16 = 6,
    F32 = 7,
    F64 = 8,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DType::Bool | DType::I8 | DType::I16 | DType::I32 | DType::I64
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DType::Bool => "bool",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferFlags(u32);

impl BufferFlags {
    pub const fn empty() -> Self {
        BufferFlags(0)
    }
    pub const HOST_VISIBLE: Self = BufferFlags(1 << 0);
    pub const HOST_COHERENT: Self = BufferFlags(1 << 1);
    pub const STORAGE: Self = BufferFlags(1 << 3);
    pub const fn with(self, other: Self) -> Self {
        BufferFlags(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl std::ops::BitOr for BufferFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.with(rhs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }
    pub fn dim2(a: usize, b: usize) -> Self {
        Shape { dims: vec![a, b] }
    }
    pub fn dim4(a: usize, b: usize, c: usize, d: usize) -> Self {
        Shape {
            dims: vec![a, b, c, d],
        }
    }
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    pub fn dim(&self, i: usize) -> Option<usize> {
        self.dims.get(i).copied()
    }
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn is_valid(&self) -> bool {
        self.dims.iter().all(|&d| d > 0)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

/// GPU buffer handle.
///
/// Always maintains a host-side `storage` backing store so that
/// `copy_from_host` / `copy_to_host` work in all build configurations —
/// including sim mode and CI without hardware. On real hardware the backing
/// store serves as the staging buffer for transfers; the GPU-side handle
/// would be carried separately once Layer 2 driver integration is complete.
pub struct GpuBuffer<T: Pod> {
    shape: Shape,
    dtype: DType,
    byte_size: usize,
    flags: BufferFlags,
    storage: Vec<u8>,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Pod> Send for GpuBuffer<T> {}
unsafe impl<T: Pod> Sync for GpuBuffer<T> {}

impl<T: Pod> GpuBuffer<T> {
    pub fn new(shape: Shape, dtype: DType, flags: BufferFlags) -> TptpResult<Self> {
        if !shape.is_valid() {
            return Err(TptpError::ShapeError {
                message: format!("invalid shape: {}", shape),
                expected: None,
                got: None,
            });
        }
        let num_elements = shape.num_elements();
        let byte_size = num_elements
            .checked_mul(dtype.size_bytes())
            .ok_or_else(|| TptpError::ShapeError {
                message: "shape too large".to_string(),
                expected: None,
                got: None,
            })?;
        Ok(GpuBuffer {
            shape,
            dtype,
            byte_size,
            flags,
            storage: vec![0u8; byte_size],
            _phantom: std::marker::PhantomData,
        })
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    pub fn dim(&self, i: usize) -> Option<usize> {
        self.shape.dim(i)
    }

    pub fn copy_from_host(&mut self, data: &[T]) -> TptpResult<()> {
        if data.len() != self.num_elements() {
            return Err(TptpError::ShapeError {
                message: format!(
                    "data length {} != buffer size {}",
                    data.len(),
                    self.num_elements()
                ),
                expected: Some(self.num_elements().to_string()),
                got: Some(data.len().to_string()),
            });
        }
        let bytes = bytemuck::cast_slice(data);
        self.storage[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [T]) -> TptpResult<()> {
        if data.len() != self.num_elements() {
            return Err(TptpError::ShapeError {
                message: format!(
                    "output length {} != buffer size {}",
                    data.len(),
                    self.num_elements()
                ),
                expected: Some(self.num_elements().to_string()),
                got: Some(data.len().to_string()),
            });
        }
        let bytes = bytemuck::cast_slice_mut(data);
        bytes.copy_from_slice(&self.storage[..bytes.len()]);
        Ok(())
    }
}

impl<T: Pod> fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("byte_size", &self.byte_size)
            .finish()
    }
}
