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
    c_char_p,
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
    "NativeError",
    "lib",
    "Event32",
    "Trigger32",
    "EventBufferSOA",
    "EventBuffer",
    "TriggerBuffer",
    "Evt3InputBuffer",
    "Evt3ParserResult",
    "SoABuffers",
    "EVENT_DTYPE",
    "TRIGGER_DTYPE",
    "EVT3_STATUS_OK",
    "EVT3_STATUS_INPUT_EXHAUSTED",
    "EVT3_STATUS_OUTPUT_FULL",
    "EVT3_STATUS_ERROR",
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


# --------------------------------------------------------------------------- #
# ctypes structs mirroring csrc/include/evutils/*.h
# --------------------------------------------------------------------------- #
class Event32(Structure):
    _fields_ = [("t", c_uint32), ("x", c_uint16), ("y", c_uint16), ("p", c_uint8)]


class Trigger32(Structure):
    _fields_ = [("t", c_uint32), ("id", c_uint8), ("p", c_uint8)]


class EventBufferSOA(Structure):
    _fields_ = [
        ("t", POINTER(c_uint64)),
        ("x", POINTER(c_uint16)),
        ("y", POINTER(c_uint16)),
        ("p", POINTER(c_uint8)),
        ("capacity", c_size_t),
        ("size", c_size_t),
    ]


class EventBuffer(Structure):
    _fields_ = [("events", POINTER(Event32)), ("capacity", c_size_t), ("size", c_size_t)]


class TriggerBuffer(Structure):
    _fields_ = [
        ("triggers", POINTER(Trigger32)),
        ("capacity", c_size_t),
        ("size", c_size_t),
    ]


class Evt3InputBuffer(Structure):
    _fields_ = [("begin", POINTER(c_uint16)), ("end", POINTER(c_uint16))]


# Keep these in sync with evt3_parse_status_t in evt3.h.
EVT3_STATUS_OK = 0
EVT3_STATUS_INPUT_EXHAUSTED = 1
EVT3_STATUS_OUTPUT_FULL = 2
EVT3_STATUS_ERROR = 3


class Evt3ParserResult(Structure):
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
    """Attach argtypes/restypes. Tolerates format modules not yet compiled."""
    handle.evutils_version.argtypes = []
    handle.evutils_version.restype = c_char_p

    handle.evutils_debug_fill_soa.argtypes = [POINTER(EventBufferSOA), c_uint64]
    handle.evutils_debug_fill_soa.restype = c_size_t

    # EVT3 is bound only once evt3.c (with the lifecycle helpers) is linked in.
    if hasattr(handle, "EVT3_state_create"):
        handle.EVT3_state_create.argtypes = []
        handle.EVT3_state_create.restype = c_void_p
        handle.EVT3_state_reset.argtypes = [c_void_p]
        handle.EVT3_state_reset.restype = None
        handle.EVT3_state_destroy.argtypes = [c_void_p]
        handle.EVT3_state_destroy.restype = None

        handle.EVT3_parse_chunk_soa.argtypes = [
            c_void_p,
            POINTER(Evt3InputBuffer),
            POINTER(EventBufferSOA),
            POINTER(TriggerBuffer),
        ]
        handle.EVT3_parse_chunk_soa.restype = Evt3ParserResult

        handle.EVT3_parse_chunk.argtypes = [
            c_void_p,
            POINTER(Evt3InputBuffer),
            POINTER(EventBuffer),
            POINTER(TriggerBuffer),
        ]
        handle.EVT3_parse_chunk.restype = Evt3ParserResult
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
# numpy <-> C SoA bridge (the piece EventReader will build on)
# --------------------------------------------------------------------------- #
class SoABuffers:
    """Owns four numpy column arrays and a ctypes ``EventBufferSOA`` aimed at
    them. C parsers fill the arrays in place; ``view()`` returns the populated
    prefix as zero-copy slices.

    The numpy arrays are kept alive for as long as this object lives, which
    keeps the C-side pointers valid. Do not let a ``view()`` slice outlive the
    ``SoABuffers`` it came from.
    """

    __slots__ = ("capacity", "t", "x", "y", "p", "c")

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.t = np.empty(self.capacity, dtype=np.uint64)
        self.x = np.empty(self.capacity, dtype=np.uint16)
        self.y = np.empty(self.capacity, dtype=np.uint16)
        self.p = np.empty(self.capacity, dtype=np.uint8)
        self.c = EventBufferSOA()
        self._rebind()

    def _rebind(self) -> None:
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
        """Return (t, x, y, p) slices covering the events written so far."""
        n = self.size
        return self.t[:n], self.x[:n], self.y[:n], self.p[:n]


def make_trigger_buffer(capacity: int):
    """Allocate a numpy-backed trigger buffer; returns (TriggerBuffer, ndarray)."""
    arr = np.empty(capacity, dtype=TRIGGER_DTYPE)
    buf = TriggerBuffer()
    buf.triggers = arr.ctypes.data_as(POINTER(Trigger32))
    buf.capacity = capacity
    buf.size = 0
    return buf, arr
