"""
TPT Tensor - GPU tensor abstraction backed by tptr runtime.
"""
from __future__ import annotations
from typing import Optional, Tuple, Union
from enum import IntEnum
import math
from tptr._ffi import Device as _NativeDevice, TptrError


class TptrDType(IntEnum):
    """Supported tensor data types."""
    FLOAT16 = 1
    FLOAT32 = 2
    FLOAT64 = 3
    INT8 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    UINT8 = 8
    UINT16 = 9
    UINT32 = 10
    BOOL = 11

    @property
    def itemsize(self) -> int:
        sizes = {1: 2, 2: 4, 3: 8, 4: 1, 5: 2, 6: 4, 7: 8, 8: 1, 9: 2, 10: 4, 11: 1}
        return sizes.get(self.value, 4)

    @property
    def name(self) -> str:
        names = {1: "float16", 2: "float32", 3: "float64", 4: "int8", 5: "int16",
                 6: "int32", 7: "int64", 8: "uint8", 9: "uint16", 10: "uint32", 11: "bool"}
        return names.get(self.value, "float32")


# Convenience aliases
dtype = TptrDType
float16 = TptrDType.FLOAT16
float32 = TptrDType.FLOAT32
float64 = TptrDType.FLOAT64
int8 = TptrDType.INT8
int16 = TptrDType.INT16
int32 = TptrDType.INT32
class TptrTensor:
    """GPU tensor backed by TPT runtime memory."""
    def __init__(self, shape: Tuple[int, ...], dtype: TptrDType = float32,
                 data: Optional[bytes] = None, device_index: int = 0):
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device_index = device_index
        self._nelem = math.prod(self._shape) if shape else 0
        self._nbytes = self._nelem * dtype.itemsize
        self._native_device = _NativeDevice(device_index)
        self._native_alloc = self._native_device.allocate(self._nbytes)
        if data is not None:
            self._native_device.memcpy_htod(self._alloc, data, len(data))

    @property
    def shape(self) -> Tuple[int, ...]: return self._shape
    @property
    def ndim(self) -> int: return len(self._shape)
    @property
    def dtype(self) -> TptrDType: return self._dtype
    @property
    def size(self) -> int: return self._nelem
    @property
    def nbytes(self) -> int: return self._nbytes
    @property
    def _alloc(self): return self._native_alloc
    @property
    def is_valid(self) -> bool: return not self._native_alloc.is_freed()

    def copy_to_host(self) -> bytes:
        return self._native_device.memcpy_dtoh(self._alloc, self._nbytes)

    def copy_from_host(self, data: bytes) -> None:
        self._native_device.memcpy_htod(self._alloc, data, len(data))

    def __add__(self, other: Union["TptrTensor", float, int]) -> "TptrTensor":
        if isinstance(other, TptrTensor):
            return _binary_op("add", self, other)
        return _binary_op("add", self, other)

    def __mul__(self, other: Union["TptrTensor", float, int]) -> "TptrTensor":
        if isinstance(other, TptrTensor):
            return _binary_op("mul", self, other)
        return _binary_op("mul", self, other)

    def __sub__(self, other: Union["TptrTensor", float, int]) -> "TptrTensor":
        if isinstance(other, TptrTensor):
            return _binary_op("sub", self, other)
        return _binary_op("sub", self, other)

    def __repr__(self) -> str:
        return f"TptrTensor(shape={self._shape}, dtype={self._dtype.name})"

    def __del__(self) -> None:
        try:
            if hasattr(self, '_native_device') and hasattr(self, '_native_alloc'):
                if not self._native_alloc.is_freed():
                    self._native_device.free(self._native_alloc)
        except Exception:
            pass

def _binary_op(op: str, a: TptrTensor, b: Union[TptrTensor, float, int]) -> TptrTensor:
    """Execute a binary operation, returning a new tensor."""
    if isinstance(b, TptrTensor):
        out_shape = _broadcast_shapes(a.shape, b.shape)
    else:
        out_shape = a.shape
    return TptrTensor(out_shape, a.dtype, device_index=a._device_index)


def _broadcast_shapes(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute broadcast shape for two tensor shapes."""
    result = []
    for i in range(max(len(a), len(b))):
        da = a[len(a) - 1 - i] if i < len(a) else 1
        db = b[len(b) - 1 - i] if i < len(b) else 1
        if da == db: result.append(da)
        elif da == 1: result.append(db)
        elif db == 1: result.append(da)
        else: raise TptrError("E0022", f"Shapes {a} and {b} are not broadcastable")
    return tuple(reversed(result))


def zeros(shape: Union[int, Tuple[int, ...]], dtype: TptrDType = float32, device_index: int = 0) -> TptrTensor:
    if isinstance(shape, int): shape = (shape,)
    return TptrTensor(shape, dtype, device_index=device_index)


def ones(shape: Union[int, Tuple[int, ...]], dtype: TptrDType = float32, device_index: int = 0) -> TptrTensor:
    if isinstance(shape, int): shape = (shape,)
    return TptrTensor(shape, dtype, device_index=device_index)


def empty(shape: Union[int, Tuple[int, ...]], dtype: TptrDType = float32, device_index: int = 0) -> TptrTensor:
    if isinstance(shape, int): shape = (shape,)
    return TptrTensor(shape, dtype, device_index=device_index)


def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[float, int],
         dtype: TptrDType = float32, device_index: int = 0) -> TptrTensor:
    if isinstance(shape, int): shape = (shape,)
    return TptrTensor(shape, dtype, device_index=device_index)

    def __repr__(self) -> str:
        return f"TptrTensor(shape={self._shape}, dtype={self._dtype.name})"

    def __del__(self) -> None:
        try:
            if hasattr(self, '_native_device') and hasattr(self, '_native_alloc'):
                if not self._native_alloc.is_freed():
                    self._native_device.free(self._native_alloc)
        except Exception:
            pass

int64 = TptrDType.INT64
uint8 = TptrDType.UINT8
uint16 = TptrDType.UINT16
uint32 = TptrDType.UINT32
bool_ = TptrDType.BOOL

