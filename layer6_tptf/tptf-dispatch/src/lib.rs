// tptf-dispatch — PyO3 extension module
//
// Exposes performance-critical dispatch paths to Python as `tptf._dispatch`.
// Each operation is a thin Rust wrapper that:
//   1. Validates array layout / dtype.
//   2. Routes to the tptr runtime (hardware feature) or a pure-Rust fallback.
//   3. Returns a NumPy array to Python.

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyArray4, PyReadonlyArray2, PyReadonlyArray4};

mod gemm;
mod attention;
mod conv2d;
mod router;

pub use router::{DispatchTable, OpRouter};

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

#[pymodule]
fn _dispatch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_gemm, m)?)?;
    m.add_function(wrap_pyfunction!(py_attention, m)?)?;
    m.add_function(wrap_pyfunction!(py_conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_dispatch, m)?)?;
    m.add_class::<PyDispatchTable>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

/// alpha * A @ B + beta * C
///
/// A: (M, K) f32, B: (K, N) f32 → (M, N) f32
#[pyfunction]
#[pyo3(signature = (a, b, alpha=1.0, beta=0.0, c=None))]
fn py_gemm<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
    alpha: f32,
    beta: f32,
    c: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a_s = a.as_slice()?;
    let b_s = b.as_slice()?;
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];

    if b.shape()[0] != k {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "GEMM inner-dim mismatch: A is ({m}×{k}), B is ({}×{n})",
            b.shape()[0]
        )));
    }

    let c_buf: Vec<f32> = match &c {
        Some(c_arr) => {
            let cs = c_arr.as_slice()?;
            cs.to_vec()
        }
        None => vec![0.0f32; m * n],
    };

    let result = gemm::dispatch(a_s, b_s, &c_buf, m, k, n, alpha, beta);
    Ok(result.into_pyarray_bound(py))
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// Q, K, V: (batch, heads, seq, d) f32 → (batch, heads, seq, d_v) f32
#[pyfunction]
#[pyo3(signature = (q, k, v, scale=None))]
fn py_attention<'py>(
    py: Python<'py>,
    q: PyReadonlyArray4<'py, f32>,
    k: PyReadonlyArray4<'py, f32>,
    v: PyReadonlyArray4<'py, f32>,
    scale: Option<f32>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let q_shape = q.shape().to_vec();
    let k_shape = k.shape().to_vec();
    let v_shape = v.shape().to_vec();

    if q_shape[3] != k_shape[3] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Attention: Q d_k={} != K d_k={}",
            q_shape[3], k_shape[3]
        )));
    }
    if k_shape[2] != v_shape[2] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Attention: K seq={} != V seq={}",
            k_shape[2], v_shape[2]
        )));
    }

    let s = scale.unwrap_or_else(|| (q_shape[3] as f32).powf(-0.5));
    let q_s = q.as_slice()?;
    let k_s = k.as_slice()?;
    let v_s = v.as_slice()?;

    let result = attention::dispatch(
        q_s, k_s, v_s,
        q_shape[0], q_shape[1], q_shape[2], q_shape[3], v_shape[3],
        s,
    );
    Ok(result.into_pyarray_bound(py))
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------

/// 2-D convolution: x (N,C_in,H,W) × weight (C_out,C_in,kH,kW) → (N,C_out,H_out,W_out)
#[pyfunction]
#[pyo3(signature = (x, weight, stride=1, padding=0, dilation=1, groups=1))]
fn py_conv2d<'py>(
    py: Python<'py>,
    x: PyReadonlyArray4<'py, f32>,
    weight: PyReadonlyArray4<'py, f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let xs = x.shape().to_vec();
    let ws = weight.shape().to_vec();

    let (n, c_in, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let (c_out, c_in_per_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);

    if c_in != c_in_per_g * groups {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Conv2d: x C_in={c_in} != weight C_in_per_group={c_in_per_g} × groups={groups}"
        )));
    }

    let x_s = x.as_slice()?;
    let w_s = weight.as_slice()?;

    let result = conv2d::dispatch(
        x_s, w_s, n, c_in, h, w, c_out, kh, kw,
        stride, padding, dilation, groups,
    );
    Ok(result.into_pyarray_bound(py))
}

// ---------------------------------------------------------------------------
// Generic dispatch by operation name
// ---------------------------------------------------------------------------

/// Route an operation by name with a JSON-encoded argument blob.
///
/// This is the hot path used by the JAX/PyTorch dispatch layers when they
/// want to express operations symbolically rather than calling typed functions.
#[pyfunction]
fn py_dispatch(_py: Python<'_>, op: &str, args_json: &str) -> PyResult<String> {
    let router = OpRouter::default();
    router.route(op, args_json).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("dispatch error for `{op}`: {e}"))
    })
}

// ---------------------------------------------------------------------------
// DispatchTable — Python-accessible op registry
// ---------------------------------------------------------------------------

#[pyclass]
struct PyDispatchTable {
    inner: DispatchTable,
}

#[pymethods]
impl PyDispatchTable {
    #[new]
    fn new() -> Self {
        Self { inner: DispatchTable::default() }
    }

    fn list_ops(&self) -> Vec<String> {
        self.inner.ops()
    }

    fn has_op(&self, name: &str) -> bool {
        self.inner.contains(name)
    }
}
