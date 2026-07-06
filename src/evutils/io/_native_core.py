"""Low-level ctypes binding and common structs for the native library.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER, Structure, byref, cast as c_cast, c_char_p, c_char,
    c_int, c_size_t, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p,
)
from pathlib import Path
from typing import cast, Any, Callable

import numpy as np
from ..types import EventArray, TriggerArray

__all__ = [
    "NativeError", "lib", "register_bindings",
    "Event32", "Trigger32",
    "EventBufferSOA", "TriggerBufferSOA", "EventBuffer", "TriggerBuffer",
    "ParserResult", "EventSoABuffers", "TriggerSoABuffers",
    "EVENT_DTYPE", "TRIGGER_DTYPE",
    "events_view", "triggers_view",
    "parse_step", "decode_all_soa",
    "EVUTILS_PARSE_OK", "EVUTILS_PARSE_INPUT_EMPTY", "EVUTILS_PARSE_OUTPUT_FULL", 
    "EVUTILS_PARSE_ERROR", "EVUTILS_PARSE_INCOMPLETE",
    "_T_DTYPE", "_X_DTYPE", "_Y_DTYPE", "_P_DTYPE", "_ID_DTYPE"
]

class NativeError(RuntimeError):
    pass

EVENT_DTYPE = np.dtype({"names": ["t", "x", "y", "p"], "formats": ["<u4", "<u2", "<u2", "u1"], "offsets": [0, 4, 6, 8], "itemsize": 12})
TRIGGER_DTYPE = np.dtype({"names": ["t", "id", "p"], "formats": ["<u4", "u1", "u1"], "offsets": [0, 4, 5], "itemsize": 8})

_T_DTYPE = np.uint64
_X_DTYPE = np.uint16
_Y_DTYPE = np.uint16
_P_DTYPE = np.uint8
_ID_DTYPE = np.uint8

class Event32(Structure):
    _fields_ = [("t", c_uint32), ("x", c_uint16), ("y", c_uint16), ("p", c_uint8)]

class Trigger32(Structure):
    _fields_ = [("t", c_uint32), ("id", c_uint8), ("p", c_uint8)]

class EventBuffer(Structure):
    _fields_ = [("events", POINTER(Event32)), ("capacity", c_size_t), ("size", c_size_t)]

class TriggerBuffer(Structure):
    _fields_ = [("triggers", POINTER(Trigger32)), ("capacity", c_size_t), ("size", c_size_t)]

class EventBufferSOA(Structure):
    _fields_ = [
        ("t", POINTER(c_uint64)), ("x", POINTER(c_uint16)),
        ("y", POINTER(c_uint16)), ("p", POINTER(c_uint8)),
        ("capacity", c_size_t), ("size", c_size_t),
    ]

class TriggerBufferSOA(Structure):
    _fields_ = [
        ("t", POINTER(c_uint64)), ("id", POINTER(c_uint8)), ("p", POINTER(c_uint8)),
        ("capacity", c_size_t), ("size", c_size_t),
    ]

EVUTILS_PARSE_OK = 0
EVUTILS_PARSE_INPUT_EMPTY = 1
EVUTILS_PARSE_OUTPUT_FULL = 2
EVUTILS_PARSE_ERROR = 3
EVUTILS_PARSE_INCOMPLETE = 4

class ParserResult(Structure):
    _fields_ = [("current", POINTER(c_uint16)), ("status", c_int)]

def _candidate_filenames() -> list[str]:
    base = "evutils_native"
    if sys.platform.startswith("win"): return [f"{base}.dll", f"lib{base}.dll"]
    if sys.platform == "darwin": return [f"lib{base}.dylib", f"{base}.dylib"]
    return [f"lib{base}.so", f"{base}.so"]

def _search_roots() -> list[Path]:
    here = Path(__file__).resolve().parent
    roots = [here]
    for parent in (here, *here.parents[:5]):
        if (parent / "pyproject.toml").exists():
            roots.append(parent / "build")
            break
    roots.append(Path.cwd() / "build")
    return roots

def _find_library() -> str:
    override = os.environ.get("EVUTILS_NATIVE_LIB")
    if override: return override
    names = set(_candidate_filenames())
    for root in _search_roots():
        if not root.exists(): continue
        for name in names:
            p = root / name
            if p.is_file(): return str(p)
        for p in root.glob("**/*evutils_native*"):
            if p.is_file() and p.name in names: return str(p)
    raise NativeError("Could not find the evutils native library.")

_BINDINGS: list[Callable[[ctypes.CDLL], None]] = []

def register_bindings(binder: Callable[[ctypes.CDLL], None]) -> None:
    _BINDINGS.append(binder)
    # If already loaded, bind immediately
    if _LIB is not None:
        binder(_LIB)

def _bind(handle: ctypes.CDLL) -> ctypes.CDLL:
    handle.evutils_version.argtypes = []
    handle.evutils_version.restype = c_char_p
    for binder in _BINDINGS:
        binder(handle)
    return handle

_LIB: ctypes.CDLL | None = None

def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        try:
            handle = ctypes.CDLL(_find_library())
        except OSError as exc:
            raise NativeError(f"Failed to load evutils native library: {exc}") from exc
        _LIB = _bind(handle)
    return _LIB

class EventSoABuffers:
    __slots__ = ("capacity", "t", "x", "y", "p", "c")
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.t = np.empty(self.capacity, dtype=_T_DTYPE)
        self.x = np.empty(self.capacity, dtype=_X_DTYPE)
        self.y = np.empty(self.capacity, dtype=_Y_DTYPE)
        self.p = np.empty(self.capacity, dtype=_P_DTYPE)
        self.c = EventBufferSOA()
        self.c.t = self.t.ctypes.data_as(POINTER(c_uint64))
        self.c.x = self.x.ctypes.data_as(POINTER(c_uint16))
        self.c.y = self.y.ctypes.data_as(POINTER(c_uint16))
        self.c.p = self.p.ctypes.data_as(POINTER(c_uint8))
        self.c.capacity = self.capacity
        self.c.size = 0
    @property
    def size(self) -> int: return int(self.c.size)
    def reset(self) -> None: self.c.size = 0
    def grow(self, new_capacity: int) -> None:
        new_capacity = int(new_capacity)
        if new_capacity <= self.capacity: return
        n = self.size
        for name, dtype, field, ptr in (
            ("t", _T_DTYPE, "t", c_uint64), ("x", _X_DTYPE, "x", c_uint16),
            ("y", _Y_DTYPE, "y", c_uint16), ("p", _P_DTYPE, "p", c_uint8),
        ):
            grown = np.empty(new_capacity, dtype=dtype)
            grown[:n] = getattr(self, name)[:n]
            setattr(self, name, grown)
            setattr(self.c, field, grown.ctypes.data_as(POINTER(ptr)))
        self.capacity = new_capacity
        self.c.capacity = new_capacity
    def view(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.size
        return self.t[:n], self.x[:n], self.y[:n], self.p[:n]

class TriggerSoABuffers:
    __slots__ = ("capacity", "t", "id", "p", "c")
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.t = np.empty(self.capacity, dtype=_T_DTYPE)
        self.id = np.empty(self.capacity, dtype=_ID_DTYPE)
        self.p = np.empty(self.capacity, dtype=_P_DTYPE)
        self.c = TriggerBufferSOA()
        self.c.t = self.t.ctypes.data_as(POINTER(c_uint64))
        self.c.id = self.id.ctypes.data_as(POINTER(c_uint8))
        self.c.p = self.p.ctypes.data_as(POINTER(c_uint8))
        self.c.capacity = self.capacity
        self.c.size = 0
    @property
    def size(self) -> int: return int(self.c.size)
    def reset(self) -> None: self.c.size = 0
    def grow(self, new_capacity: int) -> None:
        new_capacity = int(new_capacity)
        if new_capacity <= self.capacity: return
        n = self.size
        for name, dtype, ptr in (
            ("t", _T_DTYPE, c_uint64), ("id", _ID_DTYPE, c_uint8), ("p", _P_DTYPE, c_uint8),
        ):
            grown = np.empty(new_capacity, dtype=dtype)
            grown[:n] = getattr(self, name)[:n]
            setattr(self, name, grown)
            setattr(self.c, name, grown.ctypes.data_as(POINTER(ptr)))
        self.capacity = new_capacity
        self.c.capacity = new_capacity
    def view(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.size
        return self.t[:n], self.id[:n], self.p[:n]

def events_view(ev: EventSoABuffers) -> EventArray:
    n = ev.size
    return EventArray(ev.t[:n].view(np.int64), ev.x[:n], ev.y[:n], ev.p[:n])

def triggers_view(tr: TriggerSoABuffers) -> TriggerArray:
    n = tr.size
    return TriggerArray(tr.t[:n].view(np.int64), tr.p[:n], tr.id[:n])

def parse_step(words: Any, offset: int, make_input: Any, parser: Any, events: Any, triggers: Any, *, tail_pad: int = 0, word_dtype: Any = None) -> tuple[int, int]:
    n_words = len(words)
    if offset >= n_words: return 0, n_words
    before = events.size
    inp = make_input(words[offset:])
    res = parser.parse_chunk_soa(inp, events, triggers)
    if res.status == EVUTILS_PARSE_ERROR: raise RuntimeError(f"parse error near word {offset}")
    consumed = inp.consumed(res)
    if consumed == 0:
        if tail_pad and word_dtype is not None:
            tail = words[offset:]
            if len(tail):
                scratch = np.zeros(len(tail) + tail_pad, dtype=word_dtype)
                scratch[: len(tail)] = tail
                parser.parse_chunk_soa(make_input(scratch), events, triggers)
        return events.size - before, n_words
    return events.size - before, offset + consumed

def decode_all_soa(words: Any, start_offset: int, make_input: Any, parser: Any, *, est_events_per_word: float = 1.0, tail_pad: int = 0, word_dtype: Any = None, trigger_cap: int = 1 << 12) -> tuple[EventArray, int]:
    n_words = len(words) if words is not None else 0
    if n_words == 0 or start_offset >= n_words: return EventArray.empty(), n_words
    remaining = n_words - start_offset
    cap = int(remaining * est_events_per_word) + 1024
    ev = EventSoABuffers(cap)
    tr = TriggerSoABuffers(max(trigger_cap, 1))
    offset = start_offset
    while offset < n_words:
        if ev.capacity - ev.size < 128: ev.grow(int(ev.capacity * 1.5) + (1 << 16))
        if tr.capacity - tr.size < 64: tr.grow(tr.capacity * 2 + 64)
        inp = make_input(words[offset:])
        res = parser.parse_chunk_soa(inp, ev, tr)
        if res.status == EVUTILS_PARSE_ERROR: raise RuntimeError(f"parse error near word {offset}")
        consumed = inp.consumed(res)
        if consumed == 0:
            if tail_pad and word_dtype is not None:
                tail = words[offset:]
                if len(tail):
                    if ev.capacity - ev.size < len(tail) + 128: ev.grow(ev.size + len(tail) + (1 << 16))
                    scratch = np.zeros(len(tail) + tail_pad, dtype=word_dtype)
                    scratch[: len(tail)] = tail
                    parser.parse_chunk_soa(make_input(scratch), ev, tr)
            offset = n_words
            break
        offset += consumed
    n = ev.size
    if n == 0: return EventArray.empty(), offset
    out = EventArray(ev.t[:n].view(np.int64), ev.x[:n], ev.y[:n], ev.p[:n])
    return out, offset
