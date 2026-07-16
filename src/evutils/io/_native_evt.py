"""ctypes bindings for the native EVT2 / EVT2.1 / EVT3 / EVT4 parsers.

Input wrappers (:class:`Evt2Input`, :class:`Evt21Input`, :class:`Evt3Input`,
:class:`Evt4Input`) view the payload words zero-copy; the ``Evt*Parser`` classes
own the opaque C parser state. Shared SoA output buffers and the parse loop
helpers live in :mod:`evutils.io._native_core`.
"""
from __future__ import annotations
import ctypes
from ctypes import POINTER, c_uint16, c_uint32, c_uint64, cast as c_cast, c_char, c_void_p, byref
from typing import cast
import numpy as np
from ._native_core import register_bindings, NativeError, EventSoABuffers, TriggerSoABuffers, ParserResult, lib

class Evt3InputBuffer(ctypes.Structure):
    _fields_ = [("begin", POINTER(c_uint16)), ("end", POINTER(c_uint16))]
class Evt2InputBuffer(ctypes.Structure):
    _fields_ = [("begin", POINTER(c_uint32)), ("end", POINTER(c_uint32))]
class Evt21InputBuffer(ctypes.Structure):
    _fields_ = [("begin", POINTER(c_uint64)), ("end", POINTER(c_uint64))]
class Evt4InputBuffer(ctypes.Structure):
    _fields_ = [("begin", POINTER(c_uint32)), ("end", POINTER(c_uint32))]

class Evt3Input:
    __slots__ = ("arr", "c")
    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint16 or not words.flags["C_CONTIGUOUS"]: raise NativeError("Evt3Input needs a C-contiguous uint16 array")
        self.arr = words
        base = words.ctypes.data
        self.c = Evt3InputBuffer()
        self.c.begin = c_cast(base, POINTER(c_uint16))
        self.c.end = c_cast(base + words.nbytes, POINTER(c_uint16))
    def consumed(self, result: ParserResult) -> int:
        cur = c_cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 2

class Evt2Input:
    __slots__ = ("arr", "c")
    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint32 or not words.flags["C_CONTIGUOUS"]: raise NativeError("Evt2Input needs a C-contiguous uint32 array")
        self.arr = words
        base = words.ctypes.data
        self.c = Evt2InputBuffer()
        self.c.begin = c_cast(base, POINTER(c_uint32))
        self.c.end = c_cast(base + words.nbytes, POINTER(c_uint32))
    def consumed(self, result: ParserResult) -> int:
        cur = c_cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 4

class Evt21Input:
    __slots__ = ("arr", "c")
    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint64 or not words.flags["C_CONTIGUOUS"]: raise NativeError("Evt21Input needs a C-contiguous uint64 array")
        self.arr = words
        base = words.ctypes.data
        self.c = Evt21InputBuffer()
        self.c.begin = c_cast(base, POINTER(c_uint64))
        self.c.end = c_cast(base + words.nbytes, POINTER(c_uint64))
    def consumed(self, result: ParserResult) -> int:
        cur = c_cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 8

class Evt4Input:
    __slots__ = ("arr", "c")
    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint32 or not words.flags["C_CONTIGUOUS"]: raise NativeError("Evt4Input needs a C-contiguous uint32 array")
        self.arr = words
        base = words.ctypes.data
        self.c = Evt4InputBuffer()
        self.c.begin = c_cast(base, POINTER(c_uint32))
        self.c.end = c_cast(base + words.nbytes, POINTER(c_uint32))
    def consumed(self, result: ParserResult) -> int:
        cur = c_cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 4

def _bind_evt(handle: ctypes.CDLL) -> None:
    from ._native_core import EventBufferSOA, TriggerBufferSOA, ParserResult
    if hasattr(handle, "EVT3_state_size"):
        handle.EVT3_state_size.argtypes = []
        handle.EVT3_state_size.restype = ctypes.c_size_t
    if hasattr(handle, "EVT3_parse_chunk_soa"):
        handle.EVT3_parse_chunk_soa.argtypes = [c_void_p, POINTER(Evt3InputBuffer), POINTER(EventBufferSOA), POINTER(TriggerBufferSOA)]
        handle.EVT3_parse_chunk_soa.restype = ParserResult
    if hasattr(handle, "EVT3_parse_delta_t_soa"):
        handle.EVT3_parse_delta_t_soa.argtypes = [c_void_p, POINTER(Evt3InputBuffer), POINTER(EventBufferSOA), POINTER(TriggerBufferSOA), c_uint64]
        handle.EVT3_parse_delta_t_soa.restype = ParserResult
    if hasattr(handle, "EVT2_state_size"):
        handle.EVT2_state_size.argtypes = []
        handle.EVT2_state_size.restype = ctypes.c_size_t
    if hasattr(handle, "EVT2_parse_chunk_soa"):
        handle.EVT2_parse_chunk_soa.argtypes = [c_void_p, POINTER(Evt2InputBuffer), POINTER(EventBufferSOA), POINTER(TriggerBufferSOA)]
        handle.EVT2_parse_chunk_soa.restype = ParserResult
    if hasattr(handle, "EVT21_state_size"):
        handle.EVT21_state_size.argtypes = []
        handle.EVT21_state_size.restype = ctypes.c_size_t
    if hasattr(handle, "EVT21_parse_chunk_soa"):
        handle.EVT21_parse_chunk_soa.argtypes = [c_void_p, POINTER(Evt21InputBuffer), POINTER(EventBufferSOA), POINTER(TriggerBufferSOA)]
        handle.EVT21_parse_chunk_soa.restype = ParserResult
    if hasattr(handle, "EVT4_state_size"):
        handle.EVT4_state_size.argtypes = []
        handle.EVT4_state_size.restype = ctypes.c_size_t
    if hasattr(handle, "EVT4_parse_chunk_soa"):
        handle.EVT4_parse_chunk_soa.argtypes = [c_void_p, POINTER(Evt4InputBuffer), POINTER(EventBufferSOA), POINTER(TriggerBufferSOA)]
        handle.EVT4_parse_chunk_soa.restype = ParserResult

register_bindings(_bind_evt)

class Evt3Parser:
    __slots__ = ("_state", "_buf")
    def __init__(self) -> None:
        self._buf = (c_char * int(lib().EVT3_state_size()))()
        self._state = c_cast(self._buf, c_void_p)
    def reset(self) -> None: ctypes.memset(self._buf, 0, len(self._buf))
    def parse_chunk_soa(self, inp: Evt3Input, events: EventSoABuffers, triggers: TriggerSoABuffers) -> ParserResult:
        return cast(ParserResult, lib().EVT3_parse_chunk_soa(self._state, byref(inp.c), byref(events.c), byref(triggers.c)))
    def parse_delta_t_soa(self, inp: Evt3Input, events: EventSoABuffers, triggers: TriggerSoABuffers, end_ts: int) -> ParserResult:
        return cast(ParserResult, lib().EVT3_parse_delta_t_soa(self._state, byref(inp.c), byref(events.c), byref(triggers.c), c_uint64(end_ts)))
    def __enter__(self) -> "Evt3Parser": return self

class Evt2Parser:
    __slots__ = ("_state", "_buf")
    def __init__(self) -> None:
        self._buf = (c_char * int(lib().EVT2_state_size()))()
        self._state = c_cast(self._buf, c_void_p)
    def reset(self) -> None: ctypes.memset(self._buf, 0, len(self._buf))
    def parse_chunk_soa(self, inp: Evt2Input, events: EventSoABuffers, triggers: TriggerSoABuffers) -> ParserResult:
        return cast(ParserResult, lib().EVT2_parse_chunk_soa(self._state, byref(inp.c), byref(events.c), byref(triggers.c)))
    def __enter__(self) -> "Evt2Parser": return self

class Evt21Parser:
    __slots__ = ("_state", "_buf")
    def __init__(self) -> None:
        self._buf = (c_char * int(lib().EVT21_state_size()))()
        self._state = c_cast(self._buf, c_void_p)
    def reset(self) -> None: ctypes.memset(self._buf, 0, len(self._buf))
    def parse_chunk_soa(self, inp: Evt21Input, events: EventSoABuffers, triggers: TriggerSoABuffers) -> ParserResult:
        return cast(ParserResult, lib().EVT21_parse_chunk_soa(self._state, byref(inp.c), byref(events.c), byref(triggers.c)))
    def __enter__(self) -> "Evt21Parser": return self

class Evt4Parser:
    __slots__ = ("_state", "_buf")
    def __init__(self) -> None:
        self._buf = (c_char * int(lib().EVT4_state_size()))()
        self._state = c_cast(self._buf, c_void_p)
    def reset(self) -> None: ctypes.memset(self._buf, 0, len(self._buf))
    def parse_chunk_soa(self, inp: Evt4Input, events: EventSoABuffers, triggers: TriggerSoABuffers) -> ParserResult:
        return cast(ParserResult, lib().EVT4_parse_chunk_soa(self._state, byref(inp.c), byref(events.c), byref(triggers.c)))
    def __enter__(self) -> "Evt4Parser": return self
