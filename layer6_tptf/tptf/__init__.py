"""tptf — TPT GPU framework backends (PyTorch + JAX)."""

from .backend import TptBackend, get_backend, set_backend
from .jax_backend import register_jax_primitives
from .runtime_bridge import TptRuntime

__all__ = [
    "TptBackend",
    "get_backend",
    "set_backend",
    "register_jax_primitives",
    "TptRuntime",
]

__version__ = "0.1.0"
