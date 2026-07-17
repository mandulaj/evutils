"""ctypes bindings for the native Prophesee DAT parser.

:class:`DatInput` views the 2x-uint32 records zero-copy; :class:`DatParser`
owns the opaque C state (tracks the 32-bit timestamp overflow). Shared SoA
buffers and parse helpers live in :mod:`evutils.io._native_core`.
"""
from __future__ import annotations
import ctypes
from ctypes import POINTER, c_uint32, cast as c_cast, c_char, c_void_p, byref
from typing import cast
import numpy as np
from ._native_core import register_bindings, NativeError, EventSoABuffers, TriggerSoABuffers, ParserResult, lib

class DatInputBuffer(ctypes.Structure):
    _fields_ = [("begin", POINTER(c_uint32)), ("end", POINTER(c_uint32))]

class DatInput:
    __slots__ = ("arr", "c")
    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint32 or not words.flags["C_CONTIGUOUS"]: raise NativeError("DatInput needs a C-contiguous uint32 array")
        self.arr = words
        base = words.ctypes.data
        self.c = DatInputBuffer()
        self.c.begin = c_cast(base, POINTER(c_uint32))
        self.c.end = c_cast(base + words.nbytes, POINTER(c_uint32))
    def consumed(self, result: ParserResult) -> int:
        cur = c_cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 4

def _bind_dat(handle: ctypes.CDLL) -> None:
    from ._native_core import EventBufferSOA, TriggerBufferSOA, ParserResult
    if hasattr(handle, "DAT_state_size"):
        handle.DAT_state_size.argtypes = []
        handle.DAT_state_size.restype = ctypes.c_size_t
    if hasattr(handle, "DAT_parse_chunk_soa"):
        handle.DAT_parse_chunk_soa.argtypes = [c_void_p, POINTER(DatInputBuffer), POINTER(EventBufferSOA), POINTER(TriggerBufferSOA)]
        handle.DAT_parse_chunk_soa.restype = ParserResult

register_bindings(_bind_dat)

class DatParser:
    __slots__ = ("_state", "_buf")
    def __init__(self) -> None:
        self._buf = (c_char * int(lib().DAT_state_size()))()
        self._state = c_cast(self._buf, c_void_p)
    def reset(self, wrap_offset: int = 0) -> None:
        ctypes.memset(self._buf, 0, len(self._buf))
        if wrap_offset > 0:
            import struct
            struct.pack_into("=Q", self._buf, 0, wrap_offset)
    def parse_chunk_soa(self, inp: DatInput, events: EventSoABuffers, triggers: TriggerSoABuffers) -> ParserResult:
        return cast(ParserResult, lib().DAT_parse_chunk_soa(self._state, byref(inp.c), byref(events.c), byref(triggers.c)))
    def __enter__(self) -> "DatParser": return self
