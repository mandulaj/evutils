"""NPZ file decoder and encoder.

Events are stored as four flat arrays under the keys ``t``, ``x``, ``y`` and
``p`` (the SoA layout of :class:`~evutils.types.EventArray`), plus optional
scalar ``width`` / ``height`` entries. A single structured array under the key
``events`` (:data:`~evutils.types.Event_dtype`-like) is also accepted when
reading. The layout is fully compatible with plain
``np.savez(f, t=..., x=..., y=..., p=...)`` / ``np.load``.

Both directions stream and never materialise the whole recording:

* The decoder reads the ``.npy`` members through zip streams chunk by chunk
  (works for stored and deflated members alike).
* The encoder cannot write four zip members simultaneously (the zip format is
  strictly sequential), so :meth:`~EventEncoder_Npz.write` spools each column
  to an unlinked temporary file as raw bytes; :meth:`~EventEncoder_Npz.close`
  then streams every spool into the archive as a proper ``.npy`` member.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from datetime import datetime
from typing import IO

import numpy as np
from numpy.lib import format as npy_format

from ..types import EventArray, TriggerArray
from .common import EventDecoder, EventEncoder
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()

#: Column name -> on-disk dtype (matches EventArray's column dtypes).
_COLUMNS = (("t", np.dtype(np.int64)), ("x", np.dtype(np.uint16)),
            ("y", np.dtype(np.uint16)), ("p", np.dtype(np.uint8)))

def _read_npy_header(fp: IO[bytes]) -> tuple[tuple[int, ...], np.dtype]:
    """Read the ``.npy`` magic + header from a stream, returning (shape, dtype).

    Leaves ``fp`` positioned at the first data byte. Fortran-ordered arrays are
    rejected (event columns are 1-D, written C-ordered).
    """
    version = npy_format.read_magic(fp)
    read_header = {
        (1, 0): npy_format.read_array_header_1_0,
        (2, 0): npy_format.read_array_header_2_0,
    }.get(version)
    if read_header is None:
        raise ValueError(f"unsupported .npy format version {version}")
    shape, fortran, dtype = read_header(fp)
    if fortran:
        raise ValueError("Fortran-ordered .npy members are not supported")
    return shape, dtype

def _read_exact(fp: IO[bytes], nbytes: int) -> bytearray:
    """Read exactly ``nbytes`` from a (possibly decompressing) stream.

    Returns a writable buffer so the numpy views over it are mutable.
    """
    out = bytearray(nbytes)
    view = memoryview(out)
    got = 0
    while got < nbytes:
        n = fp.readinto(view[got:])  # type: ignore[attr-defined]
        if not n:
            raise EOFError(f"truncated .npy member: expected {nbytes} bytes, got {got}")
        got += n
    return out

class EventDecoder_Npz(EventDecoder):
    """Decode events from an ``.npz`` archive, streaming chunk by chunk.

    The archive members are read through zip streams: only ``chunk_size``
    events are held in memory at a time, so arbitrarily large recordings can
    be iterated. Accepts either the four column members ``t/x/y/p`` or a
    single structured ``events`` member.

    Parameters
    ----------
    source
        Byte source to read from (must be seekable, as required by the zip
        format).
    chunk_size
        Number of events returned per :meth:`read_chunk` call.

    """

    #: read_chunk returns fresh, independent arrays bounded by n_events_hint, so
    #: EventReader can hand them out directly (skipping the staging accumulator).
    _independent_windows = True

    #: NPZ columns are index-addressable, so seeking is a stream reposition (by
    #: event index) or a searchsorted over the timestamp column (by time).
    SUPPORTS_SEEK = True

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000):
        super().__init__(source, chunk_size)
        self._zf: zipfile.ZipFile | None = None
        self._streams: dict[str, IO[bytes]] = {}
        self._aos_dtype: np.dtype | None = None  # set when reading an 'events' member
        self._n = 0
        self._pos = 0

    def _open_streams(self) -> None:
        """(Re)open the member streams and position them past the npy headers."""
        assert self._zf is not None
        for fp in self._streams.values():
            fp.close()
        self._streams = {}

        names = set(self._zf.namelist())
        if {"t.npy", "x.npy", "y.npy", "p.npy"} <= names:
            n = None
            for name, _ in _COLUMNS:
                fp = self._zf.open(f"{name}.npy")
                shape, dtype = _read_npy_header(fp)
                if len(shape) != 1:
                    raise ValueError(f"member {name}.npy is not 1-D: shape {shape}")
                if n is None:
                    n = shape[0]
                elif shape[0] != n:
                    raise ValueError("event columns have mismatched lengths")
                self._streams[name] = fp
            self._n = n or 0
            self._aos_dtype = None
        elif "events.npy" in names:
            fp = self._zf.open("events.npy")
            shape, dtype = _read_npy_header(fp)
            if dtype.names is None or not {"t", "x", "y", "p"} <= set(dtype.names):
                raise ValueError("'events' member must be a structured array with t/x/y/p fields")
            self._streams["events"] = fp
            self._aos_dtype = dtype
            self._n = shape[0]
        else:
            raise ValueError(
                f"NPZ archive does not contain event data: expected members "
                f"'t/x/y/p' or 'events', found {sorted(names)}"
            )

    def init(self) -> None:
        """Open the archive and locate the event members."""
        if self._is_initialized:
            return

        f: "io.BytesIO | io.BufferedIOBase" = self._source if self._source.seekable() else io.BytesIO(self._source.read(-1))
        self._zf = zipfile.ZipFile(f)

        names = set(self._zf.namelist())
        for attr, member in (("_width", "width.npy"), ("_height", "height.npy")):
            if member in names:
                with self._zf.open(member) as fp:
                    setattr(self, attr, int(npy_format.read_array(fp).item()))

        self._open_streams()
        self._pos = 0
        self._is_initialized = True

    def _read_n(self, n: int) -> EventArray:
        """Stream the next ``n`` events out of the member streams."""
        if self._aos_dtype is not None:
            fp = self._streams["events"]
            buf = _read_exact(fp, n * self._aos_dtype.itemsize)
            return EventArray.from_aos(np.frombuffer(buf, dtype=self._aos_dtype))
        cols = {}
        for name, dtype in _COLUMNS:
            # The member's own dtype was validated against 1-D at open; event
            # columns are cast to the canonical dtypes by EventArray.
            buf = _read_exact(self._streams[name], n * dtype.itemsize)
            cols[name] = np.frombuffer(buf, dtype=dtype)
        return EventArray(cols["t"], cols["x"], cols["y"], cols["p"])

    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> EventArray:
        if not self._is_initialized:
            self.init()

        if self._pos >= self._n:
            self._eof = True
            return _EMPTY_EVENTS

        n = min(n_events_hint or self._chunk_size, self._n - self._pos)
        chunk = self._read_n(n)
        self._pos += n
        if self._pos >= self._n:
            self._eof = True
        return chunk

    def _itemsize(self, name: str) -> int:
        if name == "events":
            assert self._aos_dtype is not None
            return self._aos_dtype.itemsize
        return dict(_COLUMNS)[name].itemsize

    def _load_all_t(self) -> np.ndarray:
        """Read the full timestamp column once (for a time->index search)."""
        assert self._zf is not None
        if self._aos_dtype is not None:
            with self._zf.open("events.npy") as fp:
                return npy_format.read_array(fp)["t"]
        with self._zf.open("t.npy") as fp:
            return npy_format.read_array(fp)

    def _ts_at(self, idx: int) -> int | None:
        """Timestamp of event ``idx`` (``None`` if at/after the end)."""
        assert self._zf is not None
        if idx >= self._n:
            return None
        if self._aos_dtype is not None:
            with self._zf.open("events.npy") as fp:
                _read_npy_header(fp)
                base = fp.tell()
                fp.seek(base + idx * self._aos_dtype.itemsize)
                rec = np.frombuffer(_read_exact(fp, self._aos_dtype.itemsize),
                                    dtype=self._aos_dtype)
            return int(rec["t"][0])
        with self._zf.open("t.npy") as fp:
            _read_npy_header(fp)
            base = fp.tell()
            fp.seek(base + idx * 8)
            return int(np.frombuffer(_read_exact(fp, 8), dtype=np.int64)[0])

    def _seek_to_index(self, idx: int) -> None:
        """Reposition every member stream so the next read starts at ``idx``."""
        self._open_streams()  # streams sit at their first data byte (headers consumed)
        for name, fp in self._streams.items():
            base = fp.tell()
            fp.seek(base + idx * self._itemsize(name))
        self._pos = idx
        self._eof = idx >= self._n

    def seek(self, t: int | None = None, n: int | None = None) -> int:
        if not self._is_initialized:
            self.init()
        axis, val = self._seek_axis(t, n)
        if axis == "t":
            ts = self._load_all_t()
            idx = int(np.searchsorted(ts, val, side="left"))
        else:
            idx = val
        idx = max(0, min(idx, self._n))
        self._seek_to_index(idx)
        landed = self._ts_at(idx)
        return landed if landed is not None else val

    def reset(self) -> None:
        """Reset the reader to the beginning of the archive."""
        if self._is_initialized:
            self._open_streams()
        self._pos = 0
        self._eof = False

    def tell(self) -> int:
        """Current position, in events (npz has no meaningful byte offset)."""
        return self._pos

    def close(self) -> None:
        """Close the member streams and the archive."""
        for fp in self._streams.values():
            fp.close()
        self._streams = {}
        if self._zf is not None:
            self._zf.close()
            self._zf = None

class EventEncoder_Npz(EventEncoder):
    """Encode events into an ``.npz`` archive with bounded memory.

    Zip members can only be written one after another, while :meth:`write`
    receives all four columns interleaved -- so each column is spooled to an
    unlinked temporary file (raw bytes, no size limit from RAM) and the
    archive is assembled on :meth:`close` by streaming every spool into its
    ``.npy`` member.

    Parameters
    ----------
    writable
        Destination stream to write to.
    width, height : int
        Frame geometry stored in the archive.
    dt : datetime, optional
        Unused; npz stores no recording timestamp.
    compressed : bool
        Deflate the archive members (like ``np.savez_compressed``).

    """

    def __init__(self, writable: io.BufferedIOBase, width: int = 1280, height: int = 720,
                 dt: datetime | None = None, compressed: bool = False):
        super().__init__(writable, width, height, dt)
        self._compressed = compressed
        self._spools: dict[str, IO[bytes]] = {}
        self._closed = False

    def init(self) -> None:
        """Open one spool file per column (unlinked, cleaned up automatically)."""
        if self._is_initialized:
            return
        self._spools = {name: tempfile.TemporaryFile() for name, _ in _COLUMNS}
        self._is_initialized = True

    def write(self, events: 'np.ndarray | EventArray', triggers: 'np.ndarray | TriggerArray | None' = None) -> int:
        """Append a chunk of events to the column spools.

        Parameters
        ----------
        events : np.ndarray or EventArray
            Array of events to write.

        Returns
        -------
        int
            Number of events written.

        """
        if not self._is_initialized:
            self.init()

        n = len(events)
        for name, dtype in _COLUMNS:
            col = np.ascontiguousarray(events[name], dtype=dtype)
            self._spools[name].write(col.data)
        self._n_written_events += n
        return n

    def flush(self) -> None:
        """No-op: the archive can only be assembled once, on :meth:`close`."""

    def close(self) -> None:
        """Assemble the archive: stream each column spool into a ``.npy`` member."""
        if self._closed:
            return
        self._closed = True
        if not self._is_initialized:
            self.init()

        compression = zipfile.ZIP_DEFLATED if self._compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(self._fd, "w", compression=compression, allowZip64=True) as zf:
            for name, dtype in _COLUMNS:
                spool = self._spools[name]
                spool.flush()
                spool.seek(0)
                header = {
                    "descr": npy_format.dtype_to_descr(dtype),
                    "fortran_order": False,
                    "shape": (self._n_written_events,),
                }
                with zf.open(f"{name}.npy", "w", force_zip64=True) as dest:
                    npy_format.write_array_header_2_0(dest, header)
                    while True:
                        block = spool.read(1 << 22)
                        if not block:
                            break
                        dest.write(block)
                spool.close()
            self._spools = {}

            for name, value in (("width", np.uint16(self._width)),
                                ("height", np.uint16(self._height))):
                with zf.open(f"{name}.npy", "w") as dest:
                    npy_format.write_array(dest, np.asarray(value))
        self._fd.flush()
