"""ctypes bindings for the native AER parser.

AER records carry no timestamps; :class:`AerParser` configures how ``t`` is
generated (``AER_TS_ZERO`` or ``AER_TS_SEQUENTIAL``, carried across chunks).
Shared SoA buffers and parse helpers live in :mod:`evutils.io._native_core`.
"""
from __future__ import annotations
import ctypes
from ctypes import POINTER, c_uint32, cast as c_cast, c_char, c_void_p, c_uint64, byref
from typing import cast
import numpy as np
from ._native_core import register_bindings, NativeError, EventSoABuffers, TriggerSoABuffers, ParserResult, lib

class AerInputBuffer(ctypes.Structure):
    _fields_ = [("begin", POINTER(c_uint32)), ("end", POINTER(c_uint32))]

class AerInput:
    __slots__ = ("arr", "c")
    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint32 or not words.flags["C_CONTIGUOUS"]: raise NativeError("AerInput needs a C-contiguous uint32 array")
        self.arr = words
        base = words.ctypes.data
        self.c = AerInputBuffer()
        self.c.begin = c_cast(base, POINTER(c_uint32))
        self.c.end = c_cast(base + words.nbytes, POINTER(c_uint32))
    def consumed(self, result: ParserResult) -> int:
        cur = c_cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 4

AER_TS_ZERO = 0
AER_TS_SEQUENTIAL = 1

def _bind_aer(handle: ctypes.CDLL) -> None:
    from ._native_core import EventBufferSOA, TriggerBufferSOA, ParserResult
    if hasattr(handle, "AER_state_size"):
        handle.AER_state_size.argtypes = []
        handle.AER_state_size.restype = ctypes.c_size_t
    if hasattr(handle, "AER_state_configure"):
        handle.AER_state_configure.argtypes = [c_void_p, ctypes.c_int32, c_uint64, c_uint64]
        handle.AER_state_configure.restype = None
    if hasattr(handle, "AER_parse_chunk_soa"):
        handle.AER_parse_chunk_soa.argtypes = [c_void_p, POINTER(AerInputBuffer), POINTER(EventBufferSOA), POINTER(TriggerBufferSOA)]
        handle.AER_parse_chunk_soa.restype = ParserResult

register_bindings(_bind_aer)

class AerParser:
    __slots__ = ("_state", "_buf", "_mode", "_t_start", "_t_step")
    def __init__(self, mode: int = AER_TS_ZERO, t_start: int = 0, t_step: int = 1):
        self._buf = (c_char * int(lib().AER_state_size()))()
        self._state = c_cast(self._buf, c_void_p)
        self._mode = mode
        self._t_start = t_start
        self._t_step = t_step
        lib().AER_state_configure(self._state, mode, t_start, t_step)
    def reset(self) -> None:
        lib().AER_state_configure(self._state, self._mode, self._t_start, self._t_step)
    def parse_chunk_soa(self, inp: AerInput, events: EventSoABuffers, triggers: TriggerSoABuffers) -> ParserResult:
        return cast(ParserResult, lib().AER_parse_chunk_soa(self._state, byref(inp.c), byref(events.c), byref(triggers.c)))
    def __enter__(self) -> "AerParser": return self
