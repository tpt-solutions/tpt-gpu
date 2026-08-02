"""TPT-GPU JAX backend integration — not yet implemented.

JAX integration is planned but not yet available.  Importing this module will
succeed; calling any symbol raises NotImplementedError so callers can detect
the gap without an unhandled ImportError.

See layer6_framework/tptr/pytorch/ for the fully-implemented PyTorch backend.
"""

__all__: list = []

_NOT_IMPLEMENTED_MSG = (
    "JAX integration is not yet implemented. "
    "Use the PyTorch backend (tptr.pytorch) or the raw Python API (tptr) instead."
)


def is_available() -> bool:
    return False


def register_backend() -> bool:
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def get_backend_name() -> str:
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def is_registered() -> bool:
    return False


def get_supported_ops() -> list:
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def is_op_supported(_op: str) -> bool:
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def get_tpt_op_name(_op: str) -> str:
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def jax_to_tptr(_array):
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


class TptrJaxArray:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
