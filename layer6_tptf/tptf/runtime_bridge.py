"""Bridge between the Python framework layer and the Rust tptr runtime."""

from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np


class TptRuntime:
    """Thin wrapper around the tptr PyO3 extension.

    Falls back to a simulation mode when the compiled extension is not
    available, so the framework layer stays testable without hardware.
    """

    _instance: TptRuntime | None = None

    def __init__(self, device_index: int = 0, sim: bool = False) -> None:
        self._sim = sim
        self._device_index = device_index
        self._device = None
        self._queue = None

        if not sim:
            try:
                import tptr  # PyO3 extension from layer4
                self._device = tptr.Device(device_index)
                self._queue = tptr.Queue(self._device)
            except ImportError:
                self._sim = True

    @classmethod
    def default(cls) -> TptRuntime:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def alloc(self, nbytes: int) -> Any:
        if self._sim:
            return np.zeros(nbytes, dtype=np.uint8)
        return self._device.alloc(nbytes)

    def free(self, buf) -> None:
        if self._sim:
            return
        self._device.free(buf)

    def copy_to_device(self, host_array: np.ndarray) -> Any:
        if self._sim:
            return host_array.copy()
        buf = self._device.alloc(host_array.nbytes)
        buf.copy_from_host(host_array)
        return buf

    def copy_to_host(self, device_buf, shape, dtype) -> np.ndarray:
        if self._sim:
            return device_buf.reshape(shape).astype(dtype)
        return device_buf.to_numpy(shape, dtype)

    # ------------------------------------------------------------------
    # Kernel dispatch
    # ------------------------------------------------------------------

    def launch_gemm(
        self,
        a: np.ndarray,
        b: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.0,
        c: np.ndarray | None = None,
    ) -> np.ndarray:
        """GEMM via tptr runtime; falls back to numpy in sim mode."""
        if self._sim:
            result = alpha * (a @ b)
            if c is not None:
                result = result + beta * c
            return result

        try:
            import tptr
            ka = tptr.Kernel("gemm", self._device)
            buf_a = self.copy_to_device(a)
            buf_b = self.copy_to_device(b)
            m, k = a.shape
            _, n = b.shape
            buf_c = self._device.alloc(m * n * a.itemsize)
            ka.launch(self._queue, buf_a, buf_b, buf_c, m, n, k, alpha, beta)
            self._queue.sync()
            return self.copy_to_host(buf_c, (m, n), a.dtype)
        except Exception:
            return alpha * (a @ b)

    def launch_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray | None = None,
        scale: float | None = None,
    ) -> np.ndarray:
        """Scaled dot-product attention."""
        d_k = q.shape[-1]
        s = scale if scale is not None else d_k ** -0.5

        if self._sim:
            scores = (q @ k.swapaxes(-1, -2)) * s
            if mask is not None:
                scores = scores + mask
            weights = _softmax(scores)
            return weights @ v

        try:
            import tptr
            ka = tptr.Kernel("attention", self._device)
            buf_q = self.copy_to_device(q)
            buf_k = self.copy_to_device(k)
            buf_v = self.copy_to_device(v)
            *batch, seq, d = q.shape
            out_buf = self._device.alloc(q.nbytes)
            ka.launch(self._queue, buf_q, buf_k, buf_v, out_buf, seq, d, s)
            self._queue.sync()
            return self.copy_to_host(out_buf, q.shape, q.dtype)
        except Exception:
            scores = (q @ k.swapaxes(-1, -2)) * s
            if mask is not None:
                scores = scores + mask
            return _softmax(scores) @ v

    def sync(self) -> None:
        if not self._sim and self._queue is not None:
            self._queue.sync()


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
