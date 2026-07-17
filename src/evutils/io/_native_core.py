"""Low-level ctypes binding and common structs for the native library.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER, Structure, byref, cast as c_cast, c_char_p, c_char,
    c_int, c_size_t, c_uint8, c_uint16, c_uint64, c_void_p,
)
from pathlib import Path
from typing import cast, Callable

import numpy as np
from ..types import EventArray, TriggerArray

__all__ = [
    "NativeError", "lib", "register_bindings",
    "EventBufferSOA", "TriggerBufferSOA",
    "ParserResult", "EventSoABuffers", "TriggerSoABuffers",
    "EVENT_DTYPE", "TRIGGER_DTYPE",
    "events_view", "triggers_view",
    "parse_step", "decode_all_soa",
    "EVUTILS_PARSE_OK", "EVUTILS_PARSE_INPUT_EMPTY", "EVUTILS_PARSE_OUTPUT_FULL",
    "EVUTILS_PARSE_ERROR", "EVUTILS_PARSE_INCOMPLETE", "EVUTILS_PARSE_WINDOW_DONE", "EVUTILS_PARSE_WARNING",
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
EVUTILS_PARSE_WINDOW_DONE = 5
EVUTILS_PARSE_WARNING = 6

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

def parse_step(words: "np.ndarray", offset: int, make_input: "Callable", parser: "Callable", events: "EventArray", triggers: "TriggerArray", *, tail_pad: int = 0, word_dtype: "np.dtype | None" = None, strict: bool = False) -> tuple[int, int]:
    n_words = len(words)
    if offset >= n_words: return 0, n_words
    before = events.size
    inp = make_input(words[offset:])
    while True:
        res = parser.parse_chunk_soa(inp, events, triggers)
        consumed = inp.consumed(res)
        if res.status == EVUTILS_PARSE_WARNING:
            if strict:
                raise RuntimeError(f"malformed packet near word {offset + consumed} (strict mode)")
            import warnings
            warnings.warn(f"Malformed packets ignored near word {offset + consumed}")
            if consumed > 0:
                offset += consumed
                inp = make_input(words[offset:])
            continue
        elif res.status == EVUTILS_PARSE_ERROR:
            raise RuntimeError(f"parse error near word {offset + consumed}")
        break
    if consumed == 0:
        # Distinguish WHY nothing was consumed. A full output buffer means the
        # input is intact and the caller must drain/grow and retry -- flushing
        # the tail here would silently skip unread input.
        if res.status == EVUTILS_PARSE_OUTPUT_FULL:
            return events.size - before, offset
        # Input truly drained mid-group: only the sub-padding tail remains.
        if tail_pad and word_dtype is not None:
            tail = words[offset:]
            if len(tail):
                scratch = np.zeros(len(tail) + tail_pad, dtype=word_dtype)
                scratch[: len(tail)] = tail
                parser.parse_chunk_soa(make_input(scratch), events, triggers)
        return events.size - before, n_words
    return events.size - before, offset + consumed

def decode_all_soa(words: "np.ndarray", start_offset: int, make_input: "Callable", parser: "Callable", *, est_events_per_word: float = 1.0, tail_pad: int = 0, word_dtype: "np.dtype | None" = None, trigger_cap: int = 1 << 12, strict: bool = False) -> tuple[EventArray, int]:
    n_words = len(words) if words is not None else 0
    if n_words == 0 or start_offset >= n_words: return EventArray.empty(), n_words
    remaining = n_words - start_offset
    cap = int(remaining * est_events_per_word) + 1024
    ev = EventSoABuffers(cap)
    tr = TriggerSoABuffers(max(trigger_cap, 1))
    offset = start_offset
    while offset < n_words:
        if ev.capacity - ev.size < 128:
            # Extrapolate the true event count from the fraction of input
            # consumed so far and grow straight to it (+15% slack). A too-low
            # est_events_per_word therefore self-corrects in a *single* realloc
            # instead of repeatedly growing by a constant factor -- dense EVT2.1
            # vector streams emit up to 32 events per 64-bit word, which the old
            # 1.5x-per-grow schedule reached only after ~7 full-buffer copies.
            consumed_frac = (offset - start_offset) / remaining
            if consumed_frac > 0.02:
                projected = int(ev.size / consumed_frac * 1.15) + (1 << 16)
            else:
                projected = ev.capacity * 2 + (1 << 16)
            ev.grow(max(projected, ev.capacity + (1 << 16)))
        if tr.capacity - tr.size < 64: tr.grow(tr.capacity * 2 + 64)
        inp = make_input(words[offset:])
        while True:
            res = parser.parse_chunk_soa(inp, ev, tr)
            consumed = inp.consumed(res)
            if res.status == EVUTILS_PARSE_WARNING:
                if strict:
                    raise RuntimeError(f"malformed packet near word {offset + consumed} (strict mode)")
                import warnings
                warnings.warn(f"Malformed packets ignored near word {offset + consumed}")
                if consumed > 0:
                    offset += consumed
                    inp = make_input(words[offset:])
                continue
            elif res.status == EVUTILS_PARSE_ERROR:
                raise RuntimeError(f"parse error near word {offset + consumed}")
            break
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
