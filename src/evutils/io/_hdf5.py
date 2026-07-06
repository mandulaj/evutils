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
                    end_ms: int, base_idx: int, base_ms: int) -> None:
    """Fill ``ms_to_idx[start_ms:end_ms+1]`` from the chunk timestamps ``t``.

    ``ms_to_idx[ms]`` is the global index of the first event with
    ``t >= (ms + base_ms) * 1000``; ``base_idx`` is the global index of
    ``t[0]``. ``base_ms`` anchors the index at the recording's first
    millisecond so absolute (e.g. epoch) timestamps do not blow it up.

    Parameters
    ----------
    t : np.ndarray
        Timestamps (µs) of the chunk, monotonically non-decreasing.
    ms_to_idx : np.ndarray
        The full index array to fill.
    start_ms : int
        First (relative) millisecond entry to fill.
    end_ms : int
        Last (relative) millisecond entry to fill (inclusive).
    base_idx : int
        Global event index of the first element of ``t``.
    base_ms : int
        Millisecond of the first event in the recording (index anchor).

    """
    idx = 0
    for ms in range(start_ms, end_ms + 1):
        while idx < len(t) and t[idx] < (ms + base_ms) * 1000:
            idx += 1
        ms_to_idx[ms] = base_idx + idx


class EventDecoder_HDF5(EventDecoder):
    """Decode events from an HDF5 file.

    Two on-disk layouts are detected automatically:

    * **DSEC / RVT layout** (what :class:`EventEncoder_HDF5` writes): the four
      columns under ``events/{t,x,y,p}``, optional ``ms_to_idx`` index and
      DSEC ``t_offset``.
    * **Prophesee layout** (Metavision ``.hdf5``): a compound ``CD/events``
      dataset with ``x``/``y``/``p``/``t`` fields. Prophesee files are usually
      compressed with the ECF codec, a separate HDF5 plugin
      (https://github.com/prophesee-ai/hdf5_ecf) -- a clear error points there
      when it is missing. A compound ``events`` dataset at the root is read
      the same way.

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
        self._aos: h5py.Dataset | None = None  # compound dataset (Prophesee layout)
        self._ms_to_idx: np.ndarray | None = None
        self._ms_idx_offset: int = 0  # ms of index entry 0 (absolute-t recordings)
        self._t_offset: int = 0
        self._n: int = 0
        self._pos = 0

    def init(self) -> None:
        """Open the HDF5 file and locate the event datasets."""
        if self._is_initialized:
            return

        self._h5 = h5py.File(self._source, "r")
        node = None
        if "events" in self._h5:
            node = self._h5["events"]
        elif "CD" in self._h5 and "events" in self._h5["CD"]:
            node = self._h5["CD"]["events"]  # Prophesee Metavision layout
        if node is None:
            raise ValueError(
                "HDF5 file contains neither an 'events' group/dataset nor "
                "'CD/events' (Prophesee layout)"
            )

        if isinstance(node, h5py.Dataset):
            if node.dtype.names is None or not {"t", "x", "y", "p"} <= set(node.dtype.names):
                raise ValueError(
                    "HDF5 events dataset must be a compound type with t/x/y/p fields"
                )
            self._aos = node
            self._n = node.shape[0]
        else:
            self._n = node["t"].shape[0]

        if "ms_to_idx" in self._h5:
            self._ms_to_idx = np.asarray(self._h5["ms_to_idx"], dtype=np.int64)
        if "ms_to_idx_offset" in self._h5:
            self._ms_idx_offset = int(np.asarray(self._h5["ms_to_idx_offset"]).item())
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
        try:
            if self._aos is not None:
                rec = self._aos[start:end]
                t = rec["t"].astype(np.int64)
                if self._t_offset:
                    t += self._t_offset
                return EventArray(t, rec["x"], rec["y"],
                                  np.clip(rec["p"], 0, 1).astype(np.uint8))
            ev = self._h5["events"]
            t = ev["t"][start:end].astype(np.int64)
            if self._t_offset:
                t += self._t_offset
            return EventArray(t, ev["x"][start:end], ev["y"][start:end], ev["p"][start:end])
        except OSError as exc:
            # h5py raises OSError when a dataset's filter/codec is not loaded.
            raise OSError(
                f"Failed to read the HDF5 events dataset ({exc}). If this is a "
                "Prophesee Metavision file, its events are compressed with the "
                "ECF codec: install the plugin from "
                "https://github.com/prophesee-ai/hdf5_ecf (or put it on "
                "HDF5_PLUGIN_PATH) and retry."
            ) from exc

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
        if 0 <= end_ms < start_ms:
            raise ValueError("start_ms must be smaller than end_ms")

        # The index may be anchored at the recording's first millisecond
        # (absolute-timestamp files); shift the requested range accordingly.
        last_ms = len(self._ms_to_idx) - 1
        rel_start = max(start_ms - self._ms_idx_offset, 0)
        if rel_start > last_ms:
            return _EMPTY_EVENTS
        rel_end = end_ms - self._ms_idx_offset if end_ms >= 0 else last_ms
        if rel_end < 0:
            return _EMPTY_EVENTS
        rel_end = min(rel_end, last_ms)

        return self._slice(int(self._ms_to_idx[rel_start]), int(self._ms_to_idx[rel_end]))

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
        self._next_ms = 0        # first (relative) ms entry not yet filled
        self._ms_base = -1       # ms of the first written event (index anchor)
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

        # Extend ms_to_idx up to the last full millisecond of this chunk. The
        # index is anchored at the first event's millisecond so absolute
        # (epoch-style) timestamps don't inflate it.
        if self._ms_base < 0:
            self._ms_base = int(t[0] // 1000)
        max_ms = int(t[-1] // 1000) - self._ms_base
        if max_ms + 1 > len(self._ms_to_idx):
            self._ms_to_idx = np.resize(self._ms_to_idx, max_ms + 1)
        _fill_ms_to_idx(t, self._ms_to_idx, self._next_ms, max_ms,
                        self._n_written_events, self._ms_base)
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
        if self._closed:
            return
        self._closed = True
        if not self._is_initialized:
            self.init()  # produce a valid (empty) file even with no writes
        assert self._h5 is not None

        # Terminate the index: one entry past the last ms points at the end.
        idx = np.append(self._ms_to_idx, self._n_written_events)
        self._h5.create_dataset("ms_to_idx", data=idx.astype(np.uint64), **self._compressor)
        if self._ms_base > 0:
            # Anchor for absolute-timestamp recordings; the decoder shifts
            # requested millisecond ranges by this.
            self._h5.create_dataset("ms_to_idx_offset", data=np.int64(self._ms_base))
        self._h5.close()
        self._h5 = None
