"""HDF5 file decoder and encoder.

Layout (DSEC-compatible): the four event columns are stored under
``events/{t,x,y,p}``, with ``width`` / ``height`` file attributes and a
top-level ``ms_to_idx`` index: ``ms_to_idx[ms]`` is the index of the first
event with ``t >= ms * 1000`` (µs), which makes millisecond-range reads O(1)
lookups. A ``t_offset`` attribute (DSEC) is honoured on read when present.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import h5py
import hdf5plugin
import numpy as np

from .._jit import lazy_njit
from ..types import EventArray
from .common import EventDecoder, EventEncoder
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()


@lazy_njit
def _fill_ms_to_idx(t: np.ndarray, ms_to_idx: np.ndarray, start_ms: int,
                    end_ms: int, base_idx: int) -> None:
    """Fill ``ms_to_idx[start_ms:end_ms+1]`` from the chunk timestamps ``t``.

    ``ms_to_idx[ms]`` is the global index of the first event with
    ``t >= ms * 1000``; ``base_idx`` is the global index of ``t[0]``.

    Parameters
    ----------
    t : np.ndarray
        Timestamps (µs) of the chunk, monotonically non-decreasing.
    ms_to_idx : np.ndarray
        The full index array to fill.
    start_ms : int
        First millisecond entry to fill.
    end_ms : int
        Last millisecond entry to fill (inclusive).
    base_idx : int
        Global event index of the first element of ``t``.

    """
    idx = 0
    for ms in range(start_ms, end_ms + 1):
        while idx < len(t) and t[idx] < ms * 1000:
            idx += 1
        ms_to_idx[ms] = base_idx + idx


class EventDecoder_HDF5(EventDecoder):
    """Decode events from an HDF5 file (``events/{t,x,y,p}`` layout, DSEC-style).

    Supports both the streaming :meth:`read_chunk` interface used by
    :class:`~evutils.io.EventReader` and random-access millisecond-range reads
    via :meth:`read` (backed by the ``ms_to_idx`` index when present).

    Parameters
    ----------
    source
        Byte source to read from (must be seekable).
    chunk_size
        Number of events returned per :meth:`read_chunk` call.

    """

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000):
        super().__init__(source, chunk_size)
        self._h5: h5py.File | None = None
        self._ms_to_idx: np.ndarray | None = None
        self._t_offset: int = 0
        self._n: int = 0
        self._pos = 0

    def init(self) -> None:
        """Open the HDF5 file and locate the event datasets."""
        if self._is_initialized:
            return

        self._h5 = h5py.File(self._source, "r")
        if "events" not in self._h5:
            raise ValueError("HDF5 file does not contain an 'events' group")
        ev = self._h5["events"]
        self._n = ev["t"].shape[0]

        if "ms_to_idx" in self._h5:
            self._ms_to_idx = np.asarray(self._h5["ms_to_idx"], dtype=np.int64)
        if "t_offset" in self._h5:
            self._t_offset = int(np.asarray(self._h5["t_offset"]).item())
        if "width" in self._h5.attrs:
            self._width = int(self._h5.attrs["width"])
        if "height" in self._h5.attrs:
            self._height = int(self._h5.attrs["height"])

        self._pos = 0
        self._is_initialized = True

    def _slice(self, start: int, end: int) -> EventArray:
        """Materialise ``events[start:end]`` as an :class:`EventArray`."""
        assert self._h5 is not None
        ev = self._h5["events"]
        t = ev["t"][start:end].astype(np.int64)
        if self._t_offset:
            t += self._t_offset
        return EventArray(t, ev["x"][start:end], ev["y"][start:end], ev["p"][start:end])

    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> EventArray:
        if not self._is_initialized:
            self.init()

        if self._pos >= self._n:
            self._eof = True
            return _EMPTY_EVENTS

        chunk = self._slice(self._pos, min(self._pos + self._chunk_size, self._n))
        self._pos += len(chunk)
        if self._pos >= self._n:
            self._eof = True
        return chunk

    def read_all(self) -> EventArray:
        """Return every remaining event at once."""
        if not self._is_initialized:
            self.init()
        out = self._slice(self._pos, self._n)
        self._pos = self._n
        self._eof = True
        return out

    def read(self, start_ms: int = 0, end_ms: int = -1) -> EventArray:
        """Random-access read of a millisecond time range via ``ms_to_idx``.

        Parameters
        ----------
        start_ms : int, optional
            Start time in milliseconds, by default 0.
        end_ms : int, optional
            End time in milliseconds (exclusive), by default -1 (until the end).

        Returns
        -------
        EventArray
            Events with ``start_ms * 1000 <= t < end_ms * 1000``.

        """
        if not self._is_initialized:
            self.init()
        if self._ms_to_idx is None:
            raise ValueError("HDF5 file has no 'ms_to_idx' index; use read_chunk/read_all")
        if start_ms < 0:
            raise ValueError("start_ms must be greater or equal to 0")

        last_ms = len(self._ms_to_idx) - 1
        if start_ms > last_ms:
            return _EMPTY_EVENTS
        if end_ms < 0 or end_ms > last_ms:
            end_ms = last_ms
        if start_ms > end_ms:
            raise ValueError("start_ms must be smaller than end_ms")

        return self._slice(int(self._ms_to_idx[start_ms]), int(self._ms_to_idx[end_ms]))

    def reset(self) -> None:
        """Reset the reader to the beginning of the file."""
        self._pos = 0
        self._eof = False

    def tell(self) -> int:
        """Current position, in events (HDF5 has no meaningful byte offset)."""
        return self._pos

    def close(self) -> None:
        """Close the HDF5 handle (the byte source is closed by the reader)."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


class EventEncoder_HDF5(EventEncoder):
    """Encode events into an HDF5 file (``events/{t,x,y,p}`` + ``ms_to_idx``).

    Events must be written in timestamp order (chunks are appended and the
    millisecond index is built incrementally). The index and final flush
    happen on :meth:`close`.

    Parameters
    ----------
    writable : io.BufferedIOBase
        The file-like object to write to (must be readable and seekable,
        as required by HDF5).
    width : int, optional
        The width of the frame.
    height : int, optional
        The height of the frame.
    dt : datetime, optional
        Unused; HDF5 stores no recording timestamp.
    chunksize : int, optional
        HDF5 dataset chunk size, default 10000.

    """

    def __init__(self, writable: io.BufferedIOBase, width: int = 1280, height: int = 720,
                 dt: datetime | None = None, chunksize: int = 10000):
        super().__init__(writable, width=width, height=height, dt=dt)

        self._chunksize = chunksize
        self._h5: h5py.File | None = None
        self._ms_to_idx = np.zeros(0, dtype=np.int64)
        self._next_ms = 0  # first ms entry not yet filled
        self._closed = False

    def init(self) -> None:
        """Create the HDF5 structure (groups, datasets, attributes)."""
        if self._is_initialized:
            return

        self._h5 = h5py.File(self._fd, "w")
        self._compressor = hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

        self._h5.attrs["width"] = self._width
        self._h5.attrs["height"] = self._height

        group = self._h5.create_group("events")
        for name, dtype in (("t", "int64"), ("x", "uint16"), ("y", "uint16"), ("p", "uint8")):
            group.create_dataset(name, shape=(0,), chunks=(self._chunksize,),
                                 maxshape=(None,), dtype=dtype, **self._compressor)

        self._is_initialized = True

    def write(self, events: 'np.ndarray | EventArray') -> int:
        """Append a chunk of events and extend the millisecond index.

        Parameters
        ----------
        events : np.ndarray or EventArray
            Array of events to write (timestamps must not go backwards
            between chunks).

        Returns
        -------
        int
            Number of events written.

        """
        if not self._is_initialized:
            self.init()

        n = len(events)
        if n == 0:
            return 0
        assert self._h5 is not None

        t = np.ascontiguousarray(events["t"], dtype=np.int64)

        # Extend ms_to_idx up to the last full millisecond of this chunk.
        max_ms = int(t[-1] // 1000)
        if max_ms + 1 > len(self._ms_to_idx):
            self._ms_to_idx = np.resize(self._ms_to_idx, max_ms + 1)
        _fill_ms_to_idx(t, self._ms_to_idx, self._next_ms, max_ms, self._n_written_events)
        self._next_ms = max_ms + 1

        group = self._h5["events"]
        total = self._n_written_events + n
        for name, col in (("t", t), ("x", events["x"]), ("y", events["y"]), ("p", events["p"])):
            ds = group[name]
            ds.resize((total,))
            ds[-n:] = col

        self._n_written_events += n
        return n

    def flush(self) -> None:
        """Flush the HDF5 buffers to the underlying stream."""
        if self._h5 is not None:
            self._h5.flush()

    def close(self) -> None:
        """Write the ``ms_to_idx`` index and close the HDF5 handle."""
        if self._closed or not self._is_initialized:
            self._closed = True
            return
        self._closed = True
        assert self._h5 is not None

        # Terminate the index: one entry past the last ms points at the end.
        idx = np.append(self._ms_to_idx, self._n_written_events)
        self._h5.create_dataset("ms_to_idx", data=idx.astype(np.uint64), **self._compressor)
        self._h5.close()
        self._h5 = None
