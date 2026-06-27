# TPT PyTorch Tensor Wrapper
from __future__ import annotations
from typing import Optional, Tuple, Union
import tptr._ffi as _ffi


class TptrTorchTensor:
    def __init__(self, shape, dtype="float32", device_index=0, data=None, native_alloc=None):
        self._shape = tuple(shape)
        self._device_index = device_index
        self._nelem = 1
        for s in shape:
            self._nelem *= s
        self._dtype = _resolve_dtype(dtype)
        self._itemsize = self._dtype.itemsize if hasattr(self._dtype, "itemsize") else 4
        self._nbytes = self._nelem * self._itemsize
        if native_alloc is not None:
            self._native_alloc = native_alloc
        else:
            device = _ffi.Device(device_index)
            self._native_alloc = device.allocate(self._nbytes)
        if data is not None:
            device = _ffi.Device(device_index)
            device.memcpy_htod(self._native_alloc, data, len(data))

