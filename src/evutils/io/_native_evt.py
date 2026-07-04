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
