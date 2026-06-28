"""PyTorch dispatch backend for TPT GPU primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .runtime_bridge import TptRuntime


def tpt_matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiply via tptr runtime (falls back to torch.matmul in sim)."""
    if not a.is_cuda:
        # CPU path: use runtime sim for consistency, torch as accelerator
        try:
            import numpy as np
            rt = TptRuntime.default()
            result = rt.launch_gemm(a.numpy(), b.numpy())
            return torch.from_numpy(result)
        except Exception:
            pass
    return torch.matmul(a, b)


def tpt_gemm_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    c: torch.Tensor | None = None,
) -> torch.Tensor:
    """alpha * A @ B + beta * C."""
    result = alpha * tpt_matmul_torch(a, b)
    if c is not None:
        result = result + beta * c
    return result


def tpt_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention."""
    # Use PyTorch's built-in SDPA when available (torch 2.0+)
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    d_k = q.shape[-1]
    s = scale if scale is not None else d_k ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * s
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def tpt_conv2d_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> torch.Tensor:
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)
