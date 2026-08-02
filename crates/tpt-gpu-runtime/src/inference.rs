//! LLM inference API for the TPT runtime.
//!
//! # Architecture
//! - `LlmInference` — stable public trait; tpt-spark and tpt-crucible depend on this.
//! - `GpuInferenceEngine` — implementation that routes through layer5 kernel handles
//!   (`EmbeddingKernel`, `RmsNormKernel`, `AttentionKernel`, `GemmKernel`, `SoftmaxKernel`)
//!   with `VendorBackend::detect()` providing automatic CUDA → ROCm → Metal → TPTIR routing.
//! - `ModelInfo` — fields extracted from the GGUF binary header.
//!
//! Weights are zero-initialised at `load()` time; real weight loading from the
//! GGUF tensor section is the next increment. All kernel calls are real — the
//! output is numerically trivial (uniform logits → argmax → token 0) but the
//! full forward-pass path is exercised end-to-end.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tpt_gpu_primitives::memory::{BufferFlags, DType, Shape};
use tpt_gpu_primitives::{
    AttentionKernel, EmbeddingKernel, GemmKernel, GpuBuffer, RmsNormKernel, SoftmaxKernel,
    VendorBackend,
};

use crate::arch::{template_for_arch, ArchTemplate, ForwardOp};
use crate::error::{ErrorCode, TptrError, TptrResult};
use crate::kv_cache::KvCache;

// ---------------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------------

/// Metadata extracted from the GGUF or TPTF file header.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub path: PathBuf,
    /// Architecture tag (e.g. `"llama3"`, `"qwen2"`).
    pub arch: String,
    /// Maximum sequence length supported by this model.
    pub context_len: u32,
    /// Vocabulary size (number of tokens).
    pub vocab_size: u32,
    /// Width of hidden state vectors.
    pub hidden_dim: u32,
    /// Number of query attention heads.
    pub num_heads: u32,
    /// Number of key/value attention heads (GQA; equals `num_heads` for MHA).
    pub num_kv_heads: u32,
    /// Intermediate dimension of the FFN block.
    pub ffn_dim: u32,
    /// Number of transformer layers (blocks).
    pub num_layers: u32,
    /// Per-layer quantization bit depths from model-optimizer (2/4/6/8/16).
    /// Empty means all layers use f32. Indexed `[layer_idx]`.
    pub per_layer_bits: Vec<u8>,
    /// Neuron indices zeroed by the surgical pruner, one `Vec<u32>` per layer.
    /// `None` means no pruning was applied.
    pub pruning_mask: Option<Vec<Vec<u32>>>,
}

/// The on-disk format of a model file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    Gguf,
    Tptf,
    Unknown,
}

/// Detect the on-disk format by reading the 4-byte magic.
pub fn detect_format(path: &Path) -> ModelFormat {
    use std::fs::File;
    use std::io::Read;
    let Ok(mut f) = File::open(path) else {
        return ModelFormat::Unknown;
    };
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return ModelFormat::Unknown;
    }
    match &magic {
        b"GGUF" => ModelFormat::Gguf,
        b"TPTF" => ModelFormat::Tptf,
        _ => ModelFormat::Unknown,
    }
}

// ---------------------------------------------------------------------------
// GGUF header parser
// ---------------------------------------------------------------------------

/// GGUF KV-metadata value types (from the GGUF spec v2/v3).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }

    /// Byte width for scalar types (None for String and Array).
    fn scalar_bytes(self) -> Option<usize> {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// Minimal reader over a byte buffer with LE integer helpers.
struct BufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "GGUF truncated",
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }
    fn read_u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    fn read_i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    fn read_u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }
    fn read_f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    fn read_bool(&mut self) -> io::Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_gguf_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Skip one GGUF value of the given type without parsing it into Rust.
    fn skip_value(&mut self, vtype: GgufType) -> io::Result<()> {
        match vtype {
            GgufType::String => {
                self.read_gguf_string()?;
            }
            GgufType::Array => {
                let elem_type = GgufType::from_u32(self.read_u32()?).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "unknown array elem type")
                })?;
                let count = self.read_u64()? as usize;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
            }
            scalar => {
                let n = scalar.scalar_bytes().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "scalar_bytes unset")
                })?;
                self.read_bytes(n)?;
            }
        }
        Ok(())
    }
}

/// Parse the GGUF v2/v3 binary header and extract fields needed for `ModelInfo`.
///
/// Only reads the KV-metadata section; the tensor index and weight data are not
/// loaded here (that is the next increment — weight loading from GGUF).
pub fn parse_gguf_header(path: &Path) -> TptrResult<ModelInfo> {
    use std::fs;

    if !path.exists() {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("model file not found: {}", path.display()),
        ));
    }

    let data = fs::read(path).map_err(|e| {
        TptrError::new(
            ErrorCode::InternalError,
            format!("cannot read {}: {}", path.display(), e),
        )
    })?;

    let mut r = BufReader::new(&data);

    // Magic
    let magic = r.read_bytes(4).map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("cannot read GGUF magic: {}", e),
        )
    })?;
    if magic != b"GGUF" {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("not a GGUF file (magic = {:?})", magic),
        ));
    }

    // Version
    let version = r.read_u32().map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("cannot read GGUF version: {}", e),
        )
    })?;
    if version == 0 || version > 3 {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("unsupported GGUF version {} (supported: 1, 2, 3)", version),
        ));
    }

    // tensor_count and kv_count (u64 in v2/v3, u32 in v1)
    let (_tensor_count, kv_count) = if version >= 2 {
        (r.read_u64().unwrap_or(0), r.read_u64().unwrap_or(0))
    } else {
        (
            r.read_u32().unwrap_or(0) as u64,
            r.read_u32().unwrap_or(0) as u64,
        )
    };

    // Collect the KV keys we care about
    let mut arch = String::new();
    let mut context_len: u32 = 2048;
    let mut hidden_dim: u32 = 0;
    let mut num_heads: u32 = 0;
    let mut num_kv_heads: u32 = 0;
    let mut ffn_dim: u32 = 0;
    let mut num_layers: u32 = 0;
    let mut vocab_size: u32 = 0;

    for _ in 0..kv_count {
        let key = match r.read_gguf_string() {
            Ok(k) => k,
            Err(_) => break,
        };
        let vtype_raw = match r.read_u32() {
            Ok(v) => v,
            Err(_) => break,
        };
        let vtype = match GgufType::from_u32(vtype_raw) {
            Some(t) => t,
            None => break,
        };

        match key.as_str() {
            "general.architecture" if vtype == GgufType::String => {
                arch = r.read_gguf_string().unwrap_or_default();
            }
            "llm.context_length" if vtype == GgufType::Uint32 => {
                context_len = r.read_u32().unwrap_or(2048);
            }
            "llm.embedding_length" if vtype == GgufType::Uint32 => {
                hidden_dim = r.read_u32().unwrap_or(0);
            }
            "llm.attention.head_count" if vtype == GgufType::Uint32 => {
                num_heads = r.read_u32().unwrap_or(0);
            }
            "llm.attention.head_count_kv" if vtype == GgufType::Uint32 => {
                num_kv_heads = r.read_u32().unwrap_or(0);
            }
            "llm.feed_forward_length" if vtype == GgufType::Uint32 => {
                ffn_dim = r.read_u32().unwrap_or(0);
            }
            "llm.block_count" if vtype == GgufType::Uint32 => {
                num_layers = r.read_u32().unwrap_or(0);
            }
            "tokenizer.ggml.tokens" if vtype == GgufType::Array => {
                // Array: u32 elem_type + u64 count + elements
                let _elem_type = r.read_u32().unwrap_or(0);
                let count = r.read_u64().unwrap_or(0);
                vocab_size = count as u32;
                // Skip the array contents
                let elem_t = GgufType::String; // tokens are strings
                for _ in 0..count {
                    let _ = r.skip_value(elem_t);
                }
            }
            _ => {
                if let Err(_) = r.skip_value(vtype) {
                    break;
                }
            }
        }
    }

    // Apply safe defaults for missing fields
    if hidden_dim == 0 {
        hidden_dim = 4096;
    }
    if num_heads == 0 {
        num_heads = 32;
    }
    if num_layers == 0 {
        num_layers = 32;
    }
    if vocab_size == 0 {
        vocab_size = 32000;
    }
    if ffn_dim == 0 {
        ffn_dim = hidden_dim * 8 / 3;
    } // typical Llama ratio

    Ok(ModelInfo {
        path: path.to_owned(),
        arch,
        context_len,
        vocab_size,
        hidden_dim,
        num_heads,
        num_kv_heads,
        ffn_dim,
        num_layers,
        per_layer_bits: Vec::new(),
        pruning_mask: None,
    })
}

// ---------------------------------------------------------------------------
// TPTF header parser
// ---------------------------------------------------------------------------

/// Parse the TPTF v1 binary header and extract `ModelInfo`.
///
/// TPTF layout (first 512 bytes):
/// - [0..4]   magic "TPTF"
/// - [4..8]   version u32
/// - [8..12]  flags u32 (bit 0 = has_pruning_mask, bit 1 = has_chat_template)
/// - [12..76] arch string: u8 length prefix + UTF-8 bytes (max 63 chars)
/// - [76..80] context_len u32
/// - [80..84] vocab_size u32
/// - [84..88] hidden_dim u32
/// - [88..92] num_heads u32
/// - [92..96] num_kv_heads u32
/// - [96..100] ffn_dim u32
/// - [100..104] num_layers u32
/// - [104..232] per_layer_bits[u8; 128]
pub fn parse_tptf_header(path: &Path) -> TptrResult<ModelInfo> {
    use std::fs;

    if !path.exists() {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("model file not found: {}", path.display()),
        ));
    }

    let data = fs::read(path).map_err(|e| {
        TptrError::new(
            ErrorCode::InternalError,
            format!("cannot read {}: {}", path.display(), e),
        )
    })?;

    if data.len() < 512 {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("TPTF file too short ({} bytes, need ≥ 512)", data.len()),
        ));
    }

    let mut r = BufReader::new(&data);

    let magic = r.read_bytes(4).map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("cannot read TPTF magic: {}", e),
        )
    })?;
    if magic != b"TPTF" {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("not a TPTF file (magic = {:?})", magic),
        ));
    }

    let version = r.read_u32().map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("cannot read TPTF version: {}", e),
        )
    })?;
    if version != 1 {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("unsupported TPTF version {} (supported: 1)", version),
        ));
    }

    let flags = r.read_u32().unwrap_or(0);
    let has_pruning_mask = (flags & 1) != 0;

    // Arch string: 1-byte length prefix + bytes (packed into [12..76])
    let arch_len = r.read_u8().unwrap_or(0) as usize;
    let arch = if arch_len > 0 && arch_len <= 63 {
        let bytes = r.read_bytes(arch_len).unwrap_or_default();
        String::from_utf8_lossy(bytes).to_string()
    } else {
        r.read_bytes(63.min(r.remaining())).ok(); // skip padding
        String::new()
    };
    // Seek to byte 76 (skip any remaining arch padding)
    let consumed = 4 + 4 + 4 + 1 + arch_len;
    if consumed < 76 {
        let _ = r.read_bytes(76 - consumed);
    }

    let context_len = r.read_u32().unwrap_or(2048);
    let vocab_size = r.read_u32().unwrap_or(32000);
    let hidden_dim = r.read_u32().unwrap_or(4096);
    let num_heads = r.read_u32().unwrap_or(32);
    let num_kv_heads = r.read_u32().unwrap_or(8);
    let ffn_dim = r.read_u32().unwrap_or(0);
    let num_layers = r.read_u32().unwrap_or(32);

    // Per-layer bits array [104..232]
    let bits_bytes = r.read_bytes(128).unwrap_or_default();
    let per_layer_bits: Vec<u8> = bits_bytes[..num_layers.min(128) as usize]
        .iter()
        .copied()
        .filter(|&b| b > 0)
        .collect();

    let pruning_mask = if has_pruning_mask {
        Some(vec![Vec::new(); num_layers as usize])
    } else {
        None
    };

    let ffn_dim = if ffn_dim == 0 {
        hidden_dim * 8 / 3
    } else {
        ffn_dim
    };

    Ok(ModelInfo {
        path: path.to_owned(),
        arch,
        context_len,
        vocab_size,
        hidden_dim,
        num_heads,
        num_kv_heads,
        ffn_dim,
        num_layers,
        per_layer_bits,
        pruning_mask,
    })
}

// ---------------------------------------------------------------------------
// Model weights (zero-initialised)
// ---------------------------------------------------------------------------

/// Per-layer weight buffers for a standard transformer block.
struct LayerWeights {
    q_proj: GpuBuffer<f32>,    // [hidden_dim, num_heads * head_dim]
    k_proj: GpuBuffer<f32>,    // [hidden_dim, num_kv_heads * head_dim]
    v_proj: GpuBuffer<f32>,    // [hidden_dim, num_kv_heads * head_dim]
    o_proj: GpuBuffer<f32>,    // [num_heads * head_dim, hidden_dim]
    gate_proj: GpuBuffer<f32>, // [hidden_dim, ffn_dim]
    up_proj: GpuBuffer<f32>,   // [hidden_dim, ffn_dim]
    down_proj: GpuBuffer<f32>, // [ffn_dim, hidden_dim]
    norm1: GpuBuffer<f32>,     // [hidden_dim]
    norm2: GpuBuffer<f32>,     // [hidden_dim]
}

/// Full model weight set (zero-initialised; populated from GGUF tensors in the
/// next increment when the tensor section is parsed).
struct ModelWeights {
    embed_table: GpuBuffer<f32>, // [vocab_size, hidden_dim]
    layers: Vec<LayerWeights>,
    final_norm: GpuBuffer<f32>, // [hidden_dim]
    lm_head: GpuBuffer<f32>,    // [hidden_dim, vocab_size] — or tied to embed_table
    lm_head_tied: bool,
}

fn alloc_f32(rows: usize, cols: usize) -> TptrResult<GpuBuffer<f32>> {
    GpuBuffer::new(Shape::dim2(rows, cols), DType::F32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))
}

fn alloc_f32_1d(n: usize) -> TptrResult<GpuBuffer<f32>> {
    GpuBuffer::new(Shape::new(&[n]), DType::F32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))
}

impl ModelWeights {
    fn allocate(info: &ModelInfo, tied: bool) -> TptrResult<Self> {
        let h = info.hidden_dim as usize;
        let v = info.vocab_size as usize;
        let nh = info.num_heads.max(1) as usize;
        let nk = info.num_kv_heads.max(1) as usize;
        let hd = h / nh; // head_dim
        let f = info.ffn_dim as usize;

        let embed_table = alloc_f32(v, h)?;
        let final_norm = alloc_f32_1d(h)?;
        let lm_head = if tied {
            alloc_f32(1, 1)?
        } else {
            alloc_f32(h, v)?
        }; // placeholder when tied

        let mut layers = Vec::with_capacity(info.num_layers as usize);
        for _ in 0..info.num_layers {
            layers.push(LayerWeights {
                q_proj: alloc_f32(h, nh * hd)?,
                k_proj: alloc_f32(h, nk * hd)?,
                v_proj: alloc_f32(h, nk * hd)?,
                o_proj: alloc_f32(nh * hd, h)?,
                gate_proj: alloc_f32(h, f)?,
                up_proj: alloc_f32(h, f)?,
                down_proj: alloc_f32(f, h)?,
                norm1: alloc_f32_1d(h)?,
                norm2: alloc_f32_1d(h)?,
            });
        }

        Ok(Self {
            embed_table,
            layers,
            final_norm,
            lm_head,
            lm_head_tied: tied,
        })
    }
}

// ---------------------------------------------------------------------------
// LlmInference trait
// ---------------------------------------------------------------------------

/// Core trait for LLM token-level inference via the TPT runtime.
///
/// # Contract
/// - `load` is called once; subsequent `infer` calls reuse the loaded weights.
/// - `callback` is invoked for each generated token in order.
/// - `cancel` signals the engine to stop at the next safe preemption point.
pub trait LlmInference: Send + Sync {
    fn load(model_path: &Path) -> TptrResult<Self>
    where
        Self: Sized;
    fn model_info(&self) -> &ModelInfo;
    fn infer(
        &mut self,
        tokens: &[u32],
        max_new_tokens: u32,
        callback: impl FnMut(u32) + Send,
    ) -> TptrResult<()>;
    fn cancel(&mut self);
}

// ---------------------------------------------------------------------------
// GpuInferenceEngine
// ---------------------------------------------------------------------------

/// GPU-backed inference engine.
///
/// Routes through layer5 kernel handles with automatic hardware detection:
/// `VendorBackend::detect()` selects CUDA → ROCm → Metal → TPTIR fallback.
pub struct GpuInferenceEngine {
    info: ModelInfo,
    template: ArchTemplate,
    weights: ModelWeights,
    embed_k: EmbeddingKernel,
    rmsnorm_k: RmsNormKernel,
    attn_k: AttentionKernel,
    gemm_k: GemmKernel,
    softmax_k: SoftmaxKernel,
    kv_cache: KvCache,
    cancelled: Arc<Mutex<bool>>,
}

impl std::fmt::Debug for GpuInferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuInferenceEngine")
            .field("arch", &self.info.arch)
            .field("num_layers", &self.info.num_layers)
            .finish()
    }
}

impl LlmInference for GpuInferenceEngine {
    fn load(model_path: &Path) -> TptrResult<Self> {
        let info = match detect_format(model_path) {
            ModelFormat::Gguf => parse_gguf_header(model_path)?,
            ModelFormat::Tptf => parse_tptf_header(model_path)?,
            ModelFormat::Unknown => {
                // Fall back to GGUF parser which will give a descriptive error.
                parse_gguf_header(model_path)?
            }
        };
        let template = template_for_arch(&info.arch, &info)?;

        // Determine if this arch uses tied embeddings.
        let tied = template
            .post_ops
            .iter()
            .any(|op| matches!(op, ForwardOp::LinearOut { tied: true, .. }));

        let weights = ModelWeights::allocate(&info, tied)?;

        // All kernel handles share a single VendorBackend::detect() result.
        let vendor = VendorBackend::detect();
        let embed_k = EmbeddingKernel::with_vendor(vendor.clone());
        let rmsnorm_k = RmsNormKernel::with_vendor(vendor.clone());
        let attn_k = AttentionKernel::with_vendor(vendor.clone());
        let gemm_k = GemmKernel::with_vendor(vendor.clone());
        let softmax_k = SoftmaxKernel::with_vendor(vendor);

        let head_dim = info.hidden_dim / info.num_heads.max(1);
        let kv_cache = KvCache::new(
            info.num_layers,
            info.context_len,
            info.num_kv_heads.max(1),
            head_dim,
        );

        Ok(Self {
            info,
            template,
            weights,
            embed_k,
            rmsnorm_k,
            attn_k,
            gemm_k,
            softmax_k,
            kv_cache,
            cancelled: Arc::new(Mutex::new(false)),
        })
    }

    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn infer(
        &mut self,
        tokens: &[u32],
        max_new_tokens: u32,
        mut callback: impl FnMut(u32) + Send,
    ) -> TptrResult<()> {
        if tokens.is_empty() {
            return Err(TptrError::new(
                ErrorCode::ArgumentMismatch,
                "token list is empty",
            ));
        }

        self.kv_cache.reset();

        // Decode mode: process one token at a time starting from the last prompt token.
        // Prefill (processing all prompt tokens at once) is a future optimisation.
        let mut current_token = *tokens.last().unwrap();

        for _ in 0..max_new_tokens {
            if *self.cancelled.lock().unwrap() {
                break;
            }

            let next = self.forward_step(current_token)?;
            callback(next);
            current_token = next;
        }

        Ok(())
    }

    fn cancel(&mut self) {
        *self.cancelled.lock().unwrap() = true;
    }
}

impl GpuInferenceEngine {
    /// Run one autoregressive decode step for a single token.
    ///
    /// Returns the sampled next token ID.
    fn forward_step(&mut self, token: u32) -> TptrResult<u32> {
        let hidden_dim = self.info.hidden_dim as usize;
        let vocab_size = self.info.vocab_size as usize;
        let num_heads = self.info.num_heads.max(1) as usize;
        let num_kv = self.info.num_kv_heads.max(1) as usize;
        let head_dim = hidden_dim / num_heads;
        let ffn_dim = self.info.ffn_dim as usize;

        // --- Pre-ops: Embedding ---
        let idx_buf = token_to_index_buffer(token)?;
        let mut hidden = self
            .embed_k
            .execute(&self.weights.embed_table, &idx_buf)
            .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, e.to_string()))?;
        // Reshape to [1, hidden_dim] for GEMM compatibility
        hidden = reshape_to_2d(hidden, 1, hidden_dim)?;

        // --- Per-layer ops ---
        for layer_idx in 0..self.info.num_layers as usize {
            if *self.cancelled.lock().unwrap() {
                return Ok(0);
            }

            let layer = &self.weights.layers[layer_idx];

            // Pre-attention RmsNorm
            let normed = self.rmsnorm_k.execute(&hidden, &layer.norm1).map_err(|e| {
                TptrError::new(
                    ErrorCode::LaunchFailure,
                    format!("rmsnorm1 layer {}: {}", layer_idx, e),
                )
            })?;
            let normed = reshape_to_2d(normed, 1, hidden_dim)?;

            // Q, K, V projections
            let q = self
                .gemm_k
                .execute(&normed, &layer.q_proj, None, 1.0, 0.0)
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("q_proj layer {}: {}", layer_idx, e),
                    )
                })?;
            let k = self
                .gemm_k
                .execute(&normed, &layer.k_proj, None, 1.0, 0.0)
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("k_proj layer {}: {}", layer_idx, e),
                    )
                })?;
            let v = self
                .gemm_k
                .execute(&normed, &layer.v_proj, None, 1.0, 0.0)
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("v_proj layer {}: {}", layer_idx, e),
                    )
                })?;

            // KV cache: append current step's K and V
            let k_data = buf_to_vec(&k);
            let v_data = buf_to_vec(&v);
            self.kv_cache.append(layer_idx, &k_data, &v_data);

            // The AttentionKernel requires Q, K, V to share the same seq_len
            // (self-attention semantics). For autoregressive decode we pass only
            // the current step's K/V vectors (shape [1, qk_dim]) — the KV cache
            // is still maintained above for the full causal context; wiring it
            // to the kernel requires a cross-attention API (future layer5 work).
            let qk_dim = num_heads * head_dim;
            let k_cur = pad_to_len(&k_data, qk_dim);
            let v_cur = pad_to_len(&v_data, qk_dim);
            let k_cache = vec_to_buf(k_cur, 1, qk_dim)?;
            let v_cache = vec_to_buf(v_cur, 1, qk_dim)?;

            let scale = Some(1.0 / (head_dim as f32).sqrt());
            let q2d = reshape_to_2d(q, 1, qk_dim)?;
            let attn_out = self
                .attn_k
                .execute(&q2d, &k_cache, &v_cache, scale, None)
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("attention layer {}: {}", layer_idx, e),
                    )
                })?;
            let attn_dim = attn_out_dim(&attn_out);
            let attn_out = reshape_to_2d(attn_out, 1, attn_dim)?;

            // Output projection + residual
            let o = self
                .gemm_k
                .execute(&attn_out, &layer.o_proj, None, 1.0, 0.0)
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("o_proj layer {}: {}", layer_idx, e),
                    )
                })?;
            let o = reshape_to_2d(o, 1, hidden_dim)?;
            hidden = elementwise_add_2d(&hidden, &o, hidden_dim)?;

            // Pre-FFN RmsNorm
            let normed2 = self.rmsnorm_k.execute(&hidden, &layer.norm2).map_err(|e| {
                TptrError::new(
                    ErrorCode::LaunchFailure,
                    format!("rmsnorm2 layer {}: {}", layer_idx, e),
                )
            })?;
            let normed2 = reshape_to_2d(normed2, 1, hidden_dim)?;

            // SwiGLU FFN: gate, up, silu(gate)*up, down
            let gate = self
                .gemm_k
                .execute(&normed2, &layer.gate_proj, None, 1.0, 0.0)
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("gate_proj layer {}: {}", layer_idx, e),
                    )
                })?;
            let up = self
                .gemm_k
                .execute(&normed2, &layer.up_proj, None, 1.0, 0.0)
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("up_proj layer {}: {}", layer_idx, e),
                    )
                })?;

            let ffn_mid = swiglu_host(&gate, &up, ffn_dim)?;
            let ffn_mid = reshape_to_2d(ffn_mid, 1, ffn_dim)?;

            let ffn_out = self
                .gemm_k
                .execute(&ffn_mid, &layer.down_proj, None, 1.0, 0.0)
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("down_proj layer {}: {}", layer_idx, e),
                    )
                })?;
            let ffn_out = reshape_to_2d(ffn_out, 1, hidden_dim)?;

            hidden = elementwise_add_2d(&hidden, &ffn_out, hidden_dim)?;
        }

        // --- Post-ops: final norm, lm_head, softmax, sample ---
        let normed_final = self
            .rmsnorm_k
            .execute(&hidden, &self.weights.final_norm)
            .map_err(|e| {
                TptrError::new(ErrorCode::LaunchFailure, format!("final rmsnorm: {}", e))
            })?;
        let normed_final = reshape_to_2d(normed_final, 1, hidden_dim)?;

        let lm_weight = if self.weights.lm_head_tied {
            &self.weights.embed_table
        } else {
            &self.weights.lm_head
        };
        // lm_head: [1, hidden_dim] × [hidden_dim, vocab_size] → [1, vocab_size]
        // For tied embeddings the embed_table is [vocab_size, hidden_dim]; we transpose by
        // swapping arguments: GEMM(hidden, W^T) where W^T is [hidden_dim, vocab_size].
        // We approximate by treating the embed_table as already transposed here.
        let logits_raw = if self.weights.lm_head_tied {
            // Tied: embed_table is [vocab_size, hidden_dim]; compute host-side dot products
            tied_lm_head_host(
                &normed_final,
                &self.weights.embed_table,
                hidden_dim,
                vocab_size,
            )?
        } else {
            self.gemm_k
                .execute(&normed_final, lm_weight, None, 1.0, 0.0)
                .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, format!("lm_head: {}", e)))?
        };

        let logits = reshape_to_2d(logits_raw, 1, vocab_size)?;
        let probs = self
            .softmax_k
            .execute(&logits)
            .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, format!("softmax: {}", e)))?;

        Ok(sample_argmax(&probs, vocab_size))
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Wrap a single token ID as a `GpuBuffer<i32>` of shape `[1]`.
fn token_to_index_buffer(token: u32) -> TptrResult<GpuBuffer<i32>> {
    let mut buf = GpuBuffer::<i32>::new(Shape::new(&[1]), DType::I32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    buf.copy_from_host(&[token as i32])
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    Ok(buf)
}

/// Re-interpret a `GpuBuffer` as `[rows, cols]` — creates a new buffer with
/// the same data and the target shape.
fn reshape_to_2d(buf: GpuBuffer<f32>, rows: usize, cols: usize) -> TptrResult<GpuBuffer<f32>> {
    let n = rows * cols;
    let mut src = vec![0.0f32; buf.num_elements().max(1)];
    let copy_len = src.len().min(buf.num_elements());
    let _ = buf.copy_to_host(&mut src[..copy_len]);
    let mut out = GpuBuffer::<f32>::new(Shape::dim2(rows, cols), DType::F32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    let mut dst = vec![0.0f32; n];
    let fill_len = dst.len().min(src.len());
    dst[..fill_len].copy_from_slice(&src[..fill_len]);
    out.copy_from_host(&dst)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    Ok(out)
}

/// Infer the output dimension of an attention result buffer.
fn attn_out_dim(buf: &GpuBuffer<f32>) -> usize {
    buf.dim(1).unwrap_or(buf.num_elements())
}

/// Return a new `Vec<f32>` of exactly `target_len` elements from `src`,
/// padding with zeros if `src` is shorter.
fn pad_to_len(src: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; target_len];
    let copy = src.len().min(target_len);
    out[..copy].copy_from_slice(&src[..copy]);
    out
}

/// Copy GpuBuffer contents to a `Vec<f32>`.
fn buf_to_vec(buf: &GpuBuffer<f32>) -> Vec<f32> {
    let n = buf.num_elements();
    let mut v = vec![0.0f32; n];
    let _ = buf.copy_to_host(&mut v);
    v
}

/// Create a `GpuBuffer<f32>` of shape `[rows, cols]` from a `Vec<f32>`.
fn vec_to_buf(data: Vec<f32>, rows: usize, cols: usize) -> TptrResult<GpuBuffer<f32>> {
    let mut buf = GpuBuffer::<f32>::new(Shape::dim2(rows, cols), DType::F32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    let n = (rows * cols).min(data.len());
    let mut padded = vec![0.0f32; rows * cols];
    padded[..n].copy_from_slice(&data[..n]);
    buf.copy_from_host(&padded)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    Ok(buf)
}

/// Elementwise addition of two `[1, dim]` buffers, host-side.
fn elementwise_add_2d(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    dim: usize,
) -> TptrResult<GpuBuffer<f32>> {
    let mut va = vec![0.0f32; dim];
    let mut vb = vec![0.0f32; dim];
    let _ = a.copy_to_host(&mut va);
    let bn = vb.len().min(b.num_elements());
    let _ = b.copy_to_host(&mut vb[..bn]);
    for (x, y) in va.iter_mut().zip(vb.iter()) {
        *x += y;
    }
    vec_to_buf(va, 1, dim)
}

/// SwiGLU activation: result[i] = silu(gate[i]) * up[i].
/// Both `gate` and `up` are `[1, ffn_dim]` buffers.
fn swiglu_host(
    gate: &GpuBuffer<f32>,
    up: &GpuBuffer<f32>,
    ffn_dim: usize,
) -> TptrResult<GpuBuffer<f32>> {
    let mut g = vec![0.0f32; ffn_dim];
    let mut u = vec![0.0f32; ffn_dim];
    let gn = g.len().min(gate.num_elements());
    let _ = gate.copy_to_host(&mut g[..gn]);
    let un = u.len().min(up.num_elements());
    let _ = up.copy_to_host(&mut u[..un]);
    for (gi, ui) in g.iter_mut().zip(u.iter()) {
        let silu = *gi / (1.0 + (-*gi).exp()); // silu(x) = x * sigmoid(x)
        *gi = silu * ui;
    }
    vec_to_buf(g, 1, ffn_dim)
}

/// Tied lm_head: compute logits as hidden @ embed_table^T (host-side).
/// `hidden` is [1, hidden_dim]; `embed_table` is [vocab_size, hidden_dim].
/// Result is [1, vocab_size].
fn tied_lm_head_host(
    hidden: &GpuBuffer<f32>,
    embed_table: &GpuBuffer<f32>,
    hidden_dim: usize,
    vocab_size: usize,
) -> TptrResult<GpuBuffer<f32>> {
    let mut h = vec![0.0f32; hidden_dim];
    let mut e = vec![0.0f32; vocab_size * hidden_dim];
    let hn = h.len().min(hidden.num_elements());
    let _ = hidden.copy_to_host(&mut h[..hn]);
    let en = e.len().min(embed_table.num_elements());
    let _ = embed_table.copy_to_host(&mut e[..en]);

    let mut logits = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let row = &e[v * hidden_dim..(v + 1) * hidden_dim];
        logits[v] = h.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
    }
    vec_to_buf(logits, 1, vocab_size)
}

/// Greedy sampling: returns the index of the maximum probability.
fn sample_argmax(probs: &GpuBuffer<f32>, vocab_size: usize) -> u32 {
    let mut p = vec![0.0f32; vocab_size];
    let pn = p.len().min(probs.num_elements());
    let _ = probs.copy_to_host(&mut p[..pn]);
    p.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    /// Write a minimal valid GGUF v2 file with the given KV metadata.
    fn write_gguf(path: &Path, kv: &[(&str, GgufKvVal)]) {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&(kv.len() as u64).to_le_bytes()); // kv_count
        for (key, val) in kv {
            let kb = key.as_bytes();
            buf.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            buf.extend_from_slice(kb);
            match val {
                GgufKvVal::Str(s) => {
                    buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
                    let sb = s.as_bytes();
                    buf.extend_from_slice(&(sb.len() as u64).to_le_bytes());
                    buf.extend_from_slice(sb);
                }
                GgufKvVal::U32(v) => {
                    buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        fs::write(path, &buf).unwrap();
    }

    enum GgufKvVal {
        Str(String),
        U32(u32),
    }

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("tptr_infer_{}_{}", std::process::id(), name));
        p
    }

    // ----- GGUF parser tests -----

    // ----- TPTF parser tests -----

    fn write_tptf(path: &Path, arch: &str, num_layers: u32, per_layer_bits: &[u8]) {
        let mut buf = vec![0u8; 512];
        buf[0..4].copy_from_slice(b"TPTF");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
        buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // flags
        let ab = arch.as_bytes();
        buf[12] = ab.len().min(63) as u8;
        buf[13..13 + ab.len().min(63)].copy_from_slice(&ab[..ab.len().min(63)]);
        buf[76..80].copy_from_slice(&64u32.to_le_bytes()); // context_len
        buf[80..84].copy_from_slice(&32000u32.to_le_bytes()); // vocab_size
        buf[84..88].copy_from_slice(&64u32.to_le_bytes()); // hidden_dim
        buf[88..92].copy_from_slice(&4u32.to_le_bytes()); // num_heads
        buf[92..96].copy_from_slice(&2u32.to_le_bytes()); // num_kv_heads
        buf[96..100].copy_from_slice(&128u32.to_le_bytes()); // ffn_dim
        buf[100..104].copy_from_slice(&num_layers.to_le_bytes());
        for (i, &b) in per_layer_bits.iter().enumerate().take(128) {
            buf[104 + i] = b;
        }
        fs::write(path, &buf).unwrap();
    }

    #[test]
    fn parse_tptf_header_basic() {
        let p = tmp("basic.tptf");
        write_tptf(&p, "llama3", 4, &[4, 4, 6, 8]);
        let info = parse_tptf_header(&p).unwrap();
        assert_eq!(info.arch, "llama3");
        assert_eq!(info.num_layers, 4);
        assert_eq!(info.per_layer_bits, vec![4, 4, 6, 8]);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn detect_format_tptf() {
        let p = tmp("detect.tptf");
        write_tptf(&p, "llama3", 2, &[]);
        assert_eq!(detect_format(&p), ModelFormat::Tptf);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn load_tptf_via_engine() {
        let p = tmp("engine.tptf");
        write_tptf(&p, "llama3", 2, &[4, 4]);
        let engine = GpuInferenceEngine::load(&p).unwrap();
        assert_eq!(engine.model_info().arch, "llama3");
        assert_eq!(engine.model_info().per_layer_bits, vec![4, 4]);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn parse_bad_magic_errors() {
        let p = tmp("bad_magic.gguf");
        fs::write(
            &p,
            b"NOTG\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        .unwrap();
        let err = parse_gguf_header(&p).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidKernel);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn parse_missing_file_errors() {
        let err = parse_gguf_header(Path::new("/nonexistent/model.gguf")).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidKernel);
    }

    #[test]
    fn parse_extracts_arch_and_dims() {
        let p = tmp("parse_arch.gguf");
        write_gguf(
            &p,
            &[
                ("general.architecture", GgufKvVal::Str("llama3".into())),
                ("llm.context_length", GgufKvVal::U32(8192)),
                ("llm.embedding_length", GgufKvVal::U32(4096)),
                ("llm.attention.head_count", GgufKvVal::U32(32)),
                ("llm.attention.head_count_kv", GgufKvVal::U32(8)),
                ("llm.feed_forward_length", GgufKvVal::U32(11008)),
                ("llm.block_count", GgufKvVal::U32(32)),
            ],
        );
        let info = parse_gguf_header(&p).unwrap();
        assert_eq!(info.arch, "llama3");
        assert_eq!(info.context_len, 8192);
        assert_eq!(info.hidden_dim, 4096);
        assert_eq!(info.num_heads, 32);
        assert_eq!(info.num_kv_heads, 8);
        assert_eq!(info.num_layers, 32);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn parse_qwen2_arch() {
        let p = tmp("qwen2.gguf");
        write_gguf(
            &p,
            &[
                ("general.architecture", GgufKvVal::Str("qwen2".into())),
                ("llm.context_length", GgufKvVal::U32(32768)),
                ("llm.embedding_length", GgufKvVal::U32(3584)),
                ("llm.attention.head_count", GgufKvVal::U32(28)),
                ("llm.attention.head_count_kv", GgufKvVal::U32(4)),
                ("llm.feed_forward_length", GgufKvVal::U32(18944)),
                ("llm.block_count", GgufKvVal::U32(28)),
            ],
        );
        let info = parse_gguf_header(&p).unwrap();
        assert_eq!(info.arch, "qwen2");
        assert_eq!(info.hidden_dim, 3584);
        assert_eq!(info.num_kv_heads, 4);
        let _ = fs::remove_file(&p);
    }

    // ----- Engine load + infer tests -----

    fn write_minimal_gguf(path: &Path, arch: &str) {
        write_gguf(
            path,
            &[
                ("general.architecture", GgufKvVal::Str(arch.into())),
                ("llm.context_length", GgufKvVal::U32(64)),
                ("llm.embedding_length", GgufKvVal::U32(64)),
                ("llm.attention.head_count", GgufKvVal::U32(4)),
                ("llm.attention.head_count_kv", GgufKvVal::U32(2)),
                ("llm.feed_forward_length", GgufKvVal::U32(128)),
                ("llm.block_count", GgufKvVal::U32(2)),
            ],
        );
    }

    #[test]
    fn load_llama3_succeeds() {
        let p = tmp("llama3_load.gguf");
        write_minimal_gguf(&p, "llama3");
        let engine = GpuInferenceEngine::load(&p).unwrap();
        assert_eq!(engine.model_info().arch, "llama3");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn load_qwen2_succeeds() {
        let p = tmp("qwen2_load.gguf");
        write_minimal_gguf(&p, "qwen2");
        let engine = GpuInferenceEngine::load(&p).unwrap();
        assert_eq!(engine.model_info().arch, "qwen2");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn load_unknown_arch_errors() {
        let p = tmp("gpt2_load.gguf");
        write_minimal_gguf(&p, "gpt2");
        let err = GpuInferenceEngine::load(&p).unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedFeature);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn infer_produces_max_new_tokens() {
        let p = tmp("infer_count.gguf");
        write_minimal_gguf(&p, "llama3");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        let mut generated = Vec::new();
        engine
            .infer(&[1, 2, 3], 4, |tok| generated.push(tok))
            .unwrap();
        assert_eq!(generated.len(), 4);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn infer_empty_tokens_errors() {
        let p = tmp("infer_empty.gguf");
        write_minimal_gguf(&p, "llama3");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        let err = engine.infer(&[], 4, |_| {}).unwrap_err();
        assert_eq!(err.code, ErrorCode::ArgumentMismatch);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn cancel_stops_generation() {
        let p = tmp("cancel_gen.gguf");
        write_minimal_gguf(&p, "llama3");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        engine.cancel();
        let mut generated = Vec::new();
        engine
            .infer(&[1, 2, 3], 100, |tok| generated.push(tok))
            .unwrap();
        assert!(
            generated.is_empty(),
            "cancel before infer should produce no tokens"
        );
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn infer_gemma2_succeeds() {
        let p = tmp("gemma2_infer.gguf");
        write_minimal_gguf(&p, "gemma2");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        let mut generated = Vec::new();
        engine.infer(&[1], 2, |tok| generated.push(tok)).unwrap();
        assert_eq!(generated.len(), 2);
        let _ = fs::remove_file(&p);
    }
}
