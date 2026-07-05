"""Low-level ctypes binding to the evutils native library.

This module mirrors the C ABI declared in ``csrc/include/evutils/*.h`` and is
the *only* place that talks to the shared library. Everything above it
(EventReader, augmentations, vis, ...) works in terms of numpy arrays.

Design choices:
  * ctypes, not cffi/pybind11 — keeps the build dependency-free (just a C
    compiler) and matches the library's "as few dependencies as possible" goal.
  * The native handle is loaded lazily: importing ``evutils`` never fails just
    because the C library hasn't been built yet. The error surfaces only when
    you actually call into C, with a message that says how to build it.
  * Struct-of-arrays is the primary path. Each column is its own numpy array of
    a single dtype, so there is no struct padding to reconcile (the AoS
    ``event_t`` is 12 bytes, not 9 — see ``EVENT_DTYPE`` below).
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    cast,
    c_char_p,
    c_char,
    c_int,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from pathlib import Path

import numpy as np

from ..types import EventArray

__all__ = [
    "NativeError", "lib",
    "Event32", "Trigger32",
    "EventBufferSOA", "TriggerBufferSOA", "EventBuffer", "TriggerBuffer",
    "Evt3InputBuffer", "ParserResult",
    "SoABuffers", "TriggerSoABuffers", "Evt3Input", "Evt3Parser",
    "EVENT_DTYPE", "TRIGGER_DTYPE",
    "EVT3_STATUS_OK", "EVT3_STATUS_INPUT_EXHAUSTED",
    "EVT3_STATUS_OUTPUT_FULL", "EVT3_STATUS_ERROR",
]


class NativeError(RuntimeError):
    """Raised when the native library cannot be located or loaded."""


# --------------------------------------------------------------------------- #
# numpy dtypes that match the C structs byte-for-byte
# --------------------------------------------------------------------------- #
# AoS event_t: t@0(u4) x@4(u2) y@6(u2) p@8(u1), padded to itemsize 12.
EVENT_DTYPE = np.dtype(
    {
        "names": ["t", "x", "y", "p"],
        "formats": ["<u4", "<u2", "<u2", "u1"],
        "offsets": [0, 4, 6, 8],
        "itemsize": 12,
    }
)
# trigger_t: t@0(u4) id@4(u1) p@5(u1), padded to itemsize 8.
TRIGGER_DTYPE = np.dtype(
    {
        "names": ["t", "id", "p"],
        "formats": ["<u4", "u1", "u1"],
        "offsets": [0, 4, 5],
        "itemsize": 8,
    }
)


# SoA column dtypes — these must match timestamp64_t / uint16 / uint8 in types.h.
_T_DTYPE = np.uint64   # timestamp64_t
_X_DTYPE = np.uint16
_Y_DTYPE = np.uint16
_P_DTYPE = np.uint8
_ID_DTYPE = np.uint8
 


# --------------------------------------------------------------------------- #
# ctypes structs mirroring csrc/include/evutils/*.h
# --------------------------------------------------------------------------- #
class Event32(Structure):
    _fields_ = [("t", c_uint32), ("x", c_uint16), ("y", c_uint16), ("p", c_uint8)]


class Trigger32(Structure):
    _fields_ = [("t", c_uint32), ("id", c_uint8), ("p", c_uint8)]



class EventBuffer(Structure):
    _fields_ = [("events", POINTER(Event32)), ("capacity", c_size_t), ("size", c_size_t)]


class TriggerBuffer(Structure):
    _fields_ = [
        ("triggers", POINTER(Trigger32)),
        ("capacity", c_size_t),
        ("size", c_size_t),
    ]


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



class Evt3InputBuffer(Structure):
    _fields_ = [("begin", POINTER(c_uint16)), ("end", POINTER(c_uint16))]


class Evt2InputBuffer(Structure):
    _fields_ = [("begin", POINTER(c_uint32)), ("end", POINTER(c_uint32))]


class Evt21InputBuffer(Structure):
    _fields_ = [("begin", POINTER(c_uint64)), ("end", POINTER(c_uint64))]


class DatInputBuffer(Structure):
    _fields_ = [("begin", POINTER(c_uint32)), ("end", POINTER(c_uint32))]


class AerInputBuffer(Structure):
    _fields_ = [("begin", POINTER(c_uint32)), ("end", POINTER(c_uint32))]



EVUTILS_PARSE_OK = 0
EVUTILS_PARSE_INPUT_EMPTY = 1
EVUTILS_PARSE_OUTPUT_FULL = 2
EVUTILS_PARSE_ERROR = 3
EVUTILS_PARSE_INCOMPLETE = 4


class ParserResult(Structure):
    """Mirror of parser_result_t."""
    _fields_ = [("current", POINTER(c_uint16)), ("status", c_int)]
 
 

# --------------------------------------------------------------------------- #
# Library discovery + loading (lazy)
# --------------------------------------------------------------------------- #
def _candidate_filenames() -> list[str]:
    base = "evutils_native"
    if sys.platform.startswith("win"):
        return [f"{base}.dll", f"lib{base}.dll"]
    if sys.platform == "darwin":
        return [f"lib{base}.dylib", f"{base}.dylib"]
    return [f"lib{base}.so", f"{base}.so"]


def _search_roots() -> list[Path]:
    """Directories to look in, most-specific first."""
    here = Path(__file__).resolve().parent
    roots = [here]  # installed wheel: .so sits next to this file
    # Walk up to a repo root (has pyproject.toml) and look in its build/ dir,
    # which is where `cmake --build build` and scikit-build-core leave things.
    for parent in (here, *here.parents[:5]):
        if (parent / "pyproject.toml").exists():
            roots.append(parent / "build")
            break
    roots.append(Path.cwd() / "build")
    return roots


def _find_library() -> str:
    override = os.environ.get("EVUTILS_NATIVE_LIB")
    if override:
        return override

    names = set(_candidate_filenames())
    for root in _search_roots():
        if not root.exists():
            continue
        # direct hit
        for name in names:
            p = root / name
            if p.is_file():
                return str(p)
        # shallow search of build subdirs (e.g. build/cp312-.../)
        for p in root.glob("**/*evutils_native*"):
            if p.is_file() and p.name in names:
                return str(p)
    raise NativeError(
        "Could not find the evutils native library.\n"
        "Build it with one of:\n"
        "    uv pip install -e .         # editable dev install (auto-rebuild)\n"
        "    ./scripts/build.sh          # standalone build into ./build\n"
        "Or point EVUTILS_NATIVE_LIB at an existing "
        f"{' / '.join(sorted(names))}."
    )


def _bind(handle: ctypes.CDLL) -> ctypes.CDLL:
    handle.evutils_version.argtypes = []
    handle.evutils_version.restype = c_char_p
    
    if hasattr(handle, "EVT3_state_size"):
        handle.EVT3_state_size.argtypes = []
        handle.EVT3_state_size.restype = c_size_t
    if hasattr(handle, "EVT3_parse_chunk_soa"):
        handle.EVT3_parse_chunk_soa.argtypes = [
            c_void_p,                    # opaque evt3_state_t*
            POINTER(Evt3InputBuffer),
            POINTER(EventBufferSOA),
            POINTER(TriggerBufferSOA),
        ]
        handle.EVT3_parse_chunk_soa.restype = ParserResult

    if hasattr(handle, "EVT2_state_size"):
        handle.EVT2_state_size.argtypes = []
        handle.EVT2_state_size.restype = c_size_t
    if hasattr(handle, "EVT2_parse_chunk_soa"):
        handle.EVT2_parse_chunk_soa.argtypes = [
            c_void_p,                    # opaque evt2_state_t*
            POINTER(Evt2InputBuffer),
            POINTER(EventBufferSOA),
            POINTER(TriggerBufferSOA),
        ]
        handle.EVT2_parse_chunk_soa.restype = ParserResult

    if hasattr(handle, "EVT21_state_size"):
        handle.EVT21_state_size.argtypes = []
        handle.EVT21_state_size.restype = c_size_t
    if hasattr(handle, "EVT21_parse_chunk_soa"):
        handle.EVT21_parse_chunk_soa.argtypes = [
            c_void_p,                    # opaque evt21_state_t*
            POINTER(Evt21InputBuffer),
            POINTER(EventBufferSOA),
            POINTER(TriggerBufferSOA),
        ]
        handle.EVT21_parse_chunk_soa.restype = ParserResult

    if hasattr(handle, "DAT_state_size"):
        handle.DAT_state_size.argtypes = []
        handle.DAT_state_size.restype = c_size_t
    if hasattr(handle, "DAT_parse_chunk_soa"):
        handle.DAT_parse_chunk_soa.argtypes = [
            c_void_p,                    # opaque dat_state_t*
            POINTER(DatInputBuffer),
            POINTER(EventBufferSOA),
            POINTER(TriggerBufferSOA),
        ]
        handle.DAT_parse_chunk_soa.restype = ParserResult

    if hasattr(handle, "AER_parse_chunk_soa"):
        # AER is stateless: (input, events, triggers) -- no state pointer.
        handle.AER_parse_chunk_soa.argtypes = [
            POINTER(AerInputBuffer),
            POINTER(EventBufferSOA),
            POINTER(TriggerBufferSOA),
        ]
        handle.AER_parse_chunk_soa.restype = ParserResult
    return handle
 
 

_LIB: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    """Return the loaded native library, loading + binding it on first call."""
    global _LIB
    if _LIB is None:
        try:
            handle = ctypes.CDLL(_find_library())
        except OSError as exc:  # pragma: no cover - platform specific
            raise NativeError(f"Failed to load evutils native library: {exc}") from exc
        _LIB = _bind(handle)
    return _LIB



# --------------------------------------------------------------------------- #
# numpy <-> C SoA bridges
# --------------------------------------------------------------------------- #
class EventSoABuffers:
    """Owns the four event columns and a ctypes event_buffer_soa_t aimed at them.
    Keep this object alive as long as any view() slice is in use."""
 
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
    def size(self) -> int:
        return int(self.c.size)

    def reset(self) -> None:
        self.c.size = 0

    def grow(self, new_capacity: int) -> None:
        """Enlarge the columns to ``new_capacity``, preserving the first ``size``
        elements, and re-aim the ctypes struct at the new storage. Used by the
        single-buffer decode path when the parser fills the output."""
        new_capacity = int(new_capacity)
        if new_capacity <= self.capacity:
            return
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

    def view(self):
        n = self.size
        return self.t[:n], self.x[:n], self.y[:n], self.p[:n]


class TriggerSoABuffers:
    """Owns the three trigger columns and a ctypes trigger_buffer_soa_t."""
 
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
    def size(self) -> int:
        return int(self.c.size)

    def reset(self) -> None:
        self.c.size = 0

    def grow(self, new_capacity: int) -> None:
        """Enlarge the columns to ``new_capacity``, preserving the first ``size``
        elements, and re-aim the ctypes struct at the new storage."""
        new_capacity = int(new_capacity)
        if new_capacity <= self.capacity:
            return
        n = self.size
        for name, dtype, ptr in (
            ("t", _T_DTYPE, c_uint64), ("id", _ID_DTYPE, c_uint8),
            ("p", _P_DTYPE, c_uint8),
        ):
            grown = np.empty(new_capacity, dtype=dtype)
            grown[:n] = getattr(self, name)[:n]
            setattr(self, name, grown)
            setattr(self.c, name, grown.ctypes.data_as(POINTER(ptr)))
        self.capacity = new_capacity
        self.c.capacity = new_capacity

    def view(self):
        n = self.size
        return self.t[:n], self.id[:n], self.p[:n]
 
 
class Evt3Input:
    """Wraps a contiguous uint16 array (e.g. an mmap viewed with np.frombuffer)
    as an evt3_input_buffer_t. No copy is made; keep the array alive."""
 
    __slots__ = ("arr", "c")
 
    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint16 or not words.flags["C_CONTIGUOUS"]:
            raise NativeError("Evt3Input needs a C-contiguous uint16 array")
        self.arr = words
        base = words.ctypes.data
        self.c = Evt3InputBuffer()
        self.c.begin = cast(base, POINTER(c_uint16))
        self.c.end = cast(base + words.nbytes, POINTER(c_uint16))  # one-past-last
 
    def consumed(self, result: "ParserResult") -> int:
        """Number of uint16 words consumed, from a returned result.current."""
        cur = cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 2
 
 
class Evt3Parser:
    """Owns an opaque evt3_state_t handle from EVT3_state_create. Python never
    looks inside it, so changing the C struct layout needs only a recompile.
 
    Because the handle must be freed, use it as a context manager or call
    close() explicitly; a finalizer is a best-effort backstop.
 
        with Evt3Parser() as p:
            res = p.parse_chunk_soa(inp, events, triggers)   # state carried across calls
            p.reset()                                        # start a fresh stream
    """
 
    __slots__ = ("_state", "_buf")
 
    def __init__(self):
        handle = lib()

        self._buf = (c_char * int(handle.EVT3_state_size()))()  # zero-initialised
        self._state = cast(self._buf, c_void_p)


    def reset(self) -> None:
        ctypes.memset(self._buf, 0, len(self._buf))
        
 
    def parse_chunk_soa(self, inp: Evt3Input, events: EventSoABuffers,
                        triggers: TriggerSoABuffers) -> ParserResult:
        return lib().EVT3_parse_chunk_soa(
            self._state, byref(inp.c), byref(events.c), byref(triggers.c)
        )

    def __enter__(self):
        return self


# --------------------------------------------------------------------------- #
# EVT2 (32-bit words) and EVT2.1 (64-bit words) bindings.
#
# The output SoA buffers (EventSoABuffers / TriggerSoABuffers) are shared across
# all EVT formats; only the input word width and the parse/state entry points
# differ.
# --------------------------------------------------------------------------- #
class Evt2Input:
    """Wraps a contiguous uint32 array as an evt2_input_buffer_t (no copy)."""

    __slots__ = ("arr", "c")

    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint32 or not words.flags["C_CONTIGUOUS"]:
            raise NativeError("Evt2Input needs a C-contiguous uint32 array")
        self.arr = words
        base = words.ctypes.data
        self.c = Evt2InputBuffer()
        self.c.begin = cast(base, POINTER(c_uint32))
        self.c.end = cast(base + words.nbytes, POINTER(c_uint32))

    def consumed(self, result: "ParserResult") -> int:
        """Number of uint32 words consumed, from a returned result.current."""
        cur = cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 4


class Evt21Input:
    """Wraps a contiguous uint64 array as an evt21_input_buffer_t (no copy)."""

    __slots__ = ("arr", "c")

    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint64 or not words.flags["C_CONTIGUOUS"]:
            raise NativeError("Evt21Input needs a C-contiguous uint64 array")
        self.arr = words
        base = words.ctypes.data
        self.c = Evt21InputBuffer()
        self.c.begin = cast(base, POINTER(c_uint64))
        self.c.end = cast(base + words.nbytes, POINTER(c_uint64))

    def consumed(self, result: "ParserResult") -> int:
        """Number of uint64 words consumed, from a returned result.current."""
        cur = cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 8


class Evt2Parser:
    """Owns an opaque evt2_state_t buffer, mirroring Evt3Parser."""

    __slots__ = ("_state", "_buf")

    def __init__(self):
        handle = lib()
        self._buf = (c_char * int(handle.EVT2_state_size()))()  # zero-initialised
        self._state = cast(self._buf, c_void_p)

    def reset(self) -> None:
        ctypes.memset(self._buf, 0, len(self._buf))

    def parse_chunk_soa(self, inp: Evt2Input, events: EventSoABuffers,
                        triggers: TriggerSoABuffers) -> ParserResult:
        return lib().EVT2_parse_chunk_soa(
            self._state, byref(inp.c), byref(events.c), byref(triggers.c)
        )

    def __enter__(self):
        return self


class Evt21Parser:
    """Owns an opaque evt21_state_t buffer, mirroring Evt3Parser."""

    __slots__ = ("_state", "_buf")

    def __init__(self):
        handle = lib()
        self._buf = (c_char * int(handle.EVT21_state_size()))()  # zero-initialised
        self._state = cast(self._buf, c_void_p)

    def reset(self) -> None:
        ctypes.memset(self._buf, 0, len(self._buf))

    def parse_chunk_soa(self, inp: Evt21Input, events: EventSoABuffers,
                        triggers: TriggerSoABuffers) -> ParserResult:
        return lib().EVT21_parse_chunk_soa(
            self._state, byref(inp.c), byref(events.c), byref(triggers.c)
        )

    def __enter__(self):
        return self


# --------------------------------------------------------------------------- #
# DAT (fixed 2x uint32 records) and AER (fixed uint32 records) bindings.
# --------------------------------------------------------------------------- #
class DatInput:
    """Wraps a contiguous uint32 array (2 words per event) as a dat_input_buffer_t."""

    __slots__ = ("arr", "c")

    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint32 or not words.flags["C_CONTIGUOUS"]:
            raise NativeError("DatInput needs a C-contiguous uint32 array")
        self.arr = words
        base = words.ctypes.data
        self.c = DatInputBuffer()
        self.c.begin = cast(base, POINTER(c_uint32))
        self.c.end = cast(base + words.nbytes, POINTER(c_uint32))

    def consumed(self, result: "ParserResult") -> int:
        cur = cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 4


class AerInput:
    """Wraps a contiguous uint32 array (1 word per event) as an aer_input_buffer_t."""

    __slots__ = ("arr", "c")

    def __init__(self, words: np.ndarray):
        if words.dtype != np.uint32 or not words.flags["C_CONTIGUOUS"]:
            raise NativeError("AerInput needs a C-contiguous uint32 array")
        self.arr = words
        base = words.ctypes.data
        self.c = AerInputBuffer()
        self.c.begin = cast(base, POINTER(c_uint32))
        self.c.end = cast(base + words.nbytes, POINTER(c_uint32))

    def consumed(self, result: "ParserResult") -> int:
        cur = cast(result.current, c_void_p).value or self.arr.ctypes.data
        return (cur - self.arr.ctypes.data) // 4


class DatParser:
    """Owns an opaque dat_state_t buffer (tracks 32-bit timestamp overflow)."""

    __slots__ = ("_state", "_buf")

    def __init__(self):
        handle = lib()
        self._buf = (c_char * int(handle.DAT_state_size()))()  # zero-initialised
        self._state = cast(self._buf, c_void_p)

    def reset(self) -> None:
        ctypes.memset(self._buf, 0, len(self._buf))

    def parse_chunk_soa(self, inp: DatInput, events: EventSoABuffers,
                        triggers: TriggerSoABuffers) -> ParserResult:
        return lib().DAT_parse_chunk_soa(
            self._state, byref(inp.c), byref(events.c), byref(triggers.c)
        )

    def __enter__(self):
        return self


class AerParser:
    """Stateless AER parser (no timestamps, no cross-chunk carry)."""

    __slots__ = ()

    def reset(self) -> None:
        pass

    def parse_chunk_soa(self, inp: AerInput, events: EventSoABuffers,
                        triggers: TriggerSoABuffers) -> ParserResult:
        return lib().AER_parse_chunk_soa(
            byref(inp.c), byref(events.c), byref(triggers.c)
        )

    def __enter__(self):
        return self


# --------------------------------------------------------------------------- #
# SoA buffer -> EventArray helpers
# --------------------------------------------------------------------------- #
def events_view(ev: EventSoABuffers) -> EventArray:
    """Zero-copy :class:`EventArray` over the first ``ev.size`` decoded events.

    Timestamps are *reinterpreted* ``uint64 -> int64`` (positive, in range) rather
    than converted, so no data is copied. The result aliases ``ev``'s reusable
    columns and is only valid until the next parse into ``ev`` -- callers that need
    to retain it must ``.copy()`` (the EventReader copies when it appends to its
    ring buffer)."""
    n = ev.size
    return EventArray(ev.t[:n].view(np.int64), ev.x[:n], ev.y[:n], ev.p[:n])


# --------------------------------------------------------------------------- #
# Single parser step, appending into a caller-provided buffer
# --------------------------------------------------------------------------- #
def parse_step(words, offset, make_input, parser, events, triggers, *,
               tail_pad: int = 0, word_dtype=None):
    """Run the parser once, *appending* into ``events`` (from ``events.size``)
    up to ``events.c.capacity``, and advance through ``words``.

    This is the decode-in-place primitive behind ``EventReader``'s windowed
    read: the caller (its :class:`EventAccumulator`) hands the parser its own
    reused storage, so decoding writes straight into the accumulator with no
    intermediate copy. Returns ``(appended, new_offset)``. ``appended == 0`` with
    ``new_offset == len(words)`` means the input is exhausted; ``appended == 0``
    with progress means the step consumed only state/timing words (caller should
    step again).
    """
    n_words = len(words)
    if offset >= n_words:
        return 0, n_words
    before = events.size
    inp = make_input(words[offset:])
    res = parser.parse_chunk_soa(inp, events, triggers)
    if res.status == EVUTILS_PARSE_ERROR:
        raise RuntimeError(f"parse error near word {offset}")
    consumed = inp.consumed(res)
    if consumed == 0:
        # Trailing < tail_pad words need end-of-input look-ahead padding (EVT3).
        if tail_pad and word_dtype is not None:
            tail = words[offset:]
            if len(tail):
                scratch = np.zeros(len(tail) + tail_pad, dtype=word_dtype)
                scratch[: len(tail)] = tail
                parser.parse_chunk_soa(make_input(scratch), events, triggers)
        return events.size - before, n_words
    return events.size - before, offset + consumed


# --------------------------------------------------------------------------- #
# Single-buffer full decode (the read_all fast path)
# --------------------------------------------------------------------------- #
def decode_all_soa(words, start_offset, make_input, parser, *,
                   est_events_per_word: float = 1.0, tail_pad: int = 0,
                   word_dtype=None, trigger_cap: int = 1 << 12):
    """Decode ``words[start_offset:]`` fully into a single, growing SoA buffer.

    This is the zero-extra-copy path behind ``EventReader.read_all`` /
    ``EventDecoder.read_all``. The native parsers append at ``event_buffer->size``
    and stop just short of capacity, so we hand them one big output buffer and
    keep calling until the input is drained, growing the buffer geometrically if
    an estimate turns out too small. The returned :class:`EventArray` *views* the
    decoded columns directly -- timestamps are reinterpreted ``uint64 -> int64``
    (values are positive and fit), so there is no per-chunk copy and no final
    ``concatenate``.

    Parameters
    ----------
    words
        Contiguous numpy array of the format's native word width (the whole
        payload viewed zero-copy over the mmap/buffer).
    start_offset
        Word offset to start decoding from.
    make_input
        Callable ``view -> <Format>Input`` wrapping a word slice for the parser.
    parser
        The native parser instance (state carried across calls).
    est_events_per_word
        Initial output-capacity estimate as events per input word. Chosen per
        format to avoid growth in the common case (growth is correct but copies).
    tail_pad
        Number of zero words the parser needs as end-of-input look-ahead
        padding (EVT3 only). When the parser can no longer make progress on the
        trailing ``< tail_pad`` words, they are flushed through a zero-padded copy.
    word_dtype
        dtype of ``words`` (needed to build the zero-padded tail scratch).
    trigger_cap
        Initial trigger-buffer capacity (grown if exceeded).

    Returns
    -------
    (EventArray, int)
        The decoded events and the word offset reached (== len(words)).
    """
    n_words = len(words) if words is not None else 0
    if n_words == 0 or start_offset >= n_words:
        return EventArray.empty(), n_words

    remaining = n_words - start_offset
    cap = int(remaining * est_events_per_word) + 1024
    ev = EventSoABuffers(cap)
    tr = TriggerSoABuffers(max(trigger_cap, 1))
    offset = start_offset

    while offset < n_words:
        # Keep headroom: the parsers append at `size` and reserve up to 64 slots.
        if ev.capacity - ev.size < 128:
            ev.grow(int(ev.capacity * 1.5) + (1 << 16))
        if tr.capacity - tr.size < 64:
            tr.grow(tr.capacity * 2 + 64)

        inp = make_input(words[offset:])
        res = parser.parse_chunk_soa(inp, ev, tr)
        if res.status == EVUTILS_PARSE_ERROR:
            raise RuntimeError(f"parse error near word {offset}")

        consumed = inp.consumed(res)
        if consumed == 0:
            # No progress: the trailing < tail_pad words can't be parsed in place
            # because of end-of-input look-ahead. Flush them through a zero-padded
            # scratch (appending into the same buffer).
            if tail_pad and word_dtype is not None:
                tail = words[offset:]
                if len(tail):
                    if ev.capacity - ev.size < len(tail) + 128:
                        ev.grow(ev.size + len(tail) + (1 << 16))
                    scratch = np.zeros(len(tail) + tail_pad, dtype=word_dtype)
                    scratch[: len(tail)] = tail
                    parser.parse_chunk_soa(make_input(scratch), ev, tr)
            offset = n_words
            break
        offset += consumed

    n = ev.size
    if n == 0:
        return EventArray.empty(), offset
    # Zero-copy hand-off: the EventArray keeps the (possibly slightly oversized)
    # column arrays alive; t is a bit-reinterpretation, not a conversion.
    out = EventArray(ev.t[:n].view(np.int64), ev.x[:n], ev.y[:n], ev.p[:n])
    return out, offset
