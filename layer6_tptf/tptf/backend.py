"""Backend registry — selects between PyTorch and JAX dispatch."""

from __future__ import annotations

import threading
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


class TptBackend(Enum):
    PYTORCH = "pytorch"
    JAX = "jax"


_lock = threading.Lock()
_current: TptBackend = TptBackend.PYTORCH


def get_backend() -> TptBackend:
    return _current


def set_backend(backend: TptBackend | str) -> None:
    global _current
    if isinstance(backend, str):
        backend = TptBackend(backend)
    with _lock:
        _current = backend


class _BackendDispatch:
    """Route ops to the active backend's implementation."""

    def matmul(self, a, b):
        if _current is TptBackend.JAX:
            from .jax_backend import tpt_matmul_jax
            return tpt_matmul_jax(a, b)
        from .torch_backend import tpt_matmul_torch
        return tpt_matmul_torch(a, b)

    def gemm(self, a, b, alpha: float = 1.0, beta: float = 0.0, c=None):
        if _current is TptBackend.JAX:
            from .jax_backend import tpt_gemm_jax
            return tpt_gemm_jax(a, b, alpha, beta, c)
        from .torch_backend import tpt_gemm_torch
        return tpt_gemm_torch(a, b, alpha, beta, c)

    def attention(self, q, k, v, mask=None, scale: float | None = None):
        if _current is TptBackend.JAX:
            from .jax_backend import tpt_attention_jax
            return tpt_attention_jax(q, k, v, mask, scale)
        from .torch_backend import tpt_attention_torch
        return tpt_attention_torch(q, k, v, mask, scale)

    def conv2d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if _current is TptBackend.JAX:
            from .jax_backend import tpt_conv2d_jax
            return tpt_conv2d_jax(x, weight, bias, stride, padding, dilation, groups)
        from .torch_backend import tpt_conv2d_torch
        return tpt_conv2d_torch(x, weight, bias, stride, padding, dilation, groups)

    def relu(self, x):
        if _current is TptBackend.JAX:
            import jax.nn as jnn
            return jnn.relu(x)
        import torch.nn.functional as F
        return F.relu(x)

    def gelu(self, x):
        if _current is TptBackend.JAX:
            import jax.nn as jnn
            return jnn.gelu(x)
        import torch.nn.functional as F
        return F.gelu(x)

    def softmax(self, x, axis: int = -1):
        if _current is TptBackend.JAX:
            import jax.nn as jnn
            return jnn.softmax(x, axis=axis)
        import torch.nn.functional as F
        return F.softmax(x, dim=axis)

    def layer_norm(self, x, normalized_shape, weight=None, bias=None, eps: float = 1e-5):
        if _current is TptBackend.JAX:
            from .jax_backend import tpt_layer_norm_jax
            return tpt_layer_norm_jax(x, normalized_shape, weight, bias, eps)
        import torch.nn.functional as F
        return F.layer_norm(x, normalized_shape, weight, bias, eps)


dispatch = _BackendDispatch()
