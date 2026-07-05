"""NPZ file decoder and encoder.

Events are stored as four flat arrays under the keys ``t``, ``x``, ``y`` and
``p`` (the SoA layout of :class:`~evutils.types.EventArray`), plus optional
scalar ``width`` / ``height`` entries. A single structured array under the key
``events`` (:data:`~evutils.types.Event_dtype`-like) is also accepted when
reading.

``.npz`` is a zip container, so it cannot be appended to incrementally: the
encoder buffers written chunks in memory and writes the archive once, when the
writer is closed.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import numpy as np

from ..types import EventArray
from .common import EventDecoder, EventEncoder
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()


class EventDecoder_Npz(EventDecoder):
    """Decode events from an ``.npz`` archive.

    The whole archive is loaded on :meth:`init` (npz is a random-access
    container, not a stream format); :meth:`read_chunk` then serves
    ``chunk_size``-sized slices of the loaded columns.

    Parameters
    ----------
    source
        Byte source to read from (must be seekable, as required by the zip
        format).
    chunk_size
        Number of events returned per :meth:`read_chunk` call.

    """

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000):
        super().__init__(source, chunk_size)
        self._data: EventArray | None = None
        self._pos = 0

    def init(self) -> None:
        """Load the archive and locate the event columns."""
        if self._is_initialized:
            return

        f: Any = self._source if self._source.seekable() else io.BytesIO(self._source.read(-1))
        with np.load(f) as npz:
            keys = set(npz.files)
            if {"t", "x", "y", "p"} <= keys:
                self._data = EventArray(npz["t"], npz["x"], npz["y"], npz["p"])
            elif "events" in keys:
                self._data = EventArray.from_aos(npz["events"])
            else:
                raise ValueError(
                    f"NPZ archive does not contain event data: expected keys "
                    f"'t'/'x'/'y'/'p' or 'events', found {sorted(keys)}"
                )
            if "width" in keys:
                self._width = int(npz["width"])
            if "height" in keys:
                self._height = int(npz["height"])

        self._pos = 0
        self._is_initialized = True

    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> EventArray:
        if not self._is_initialized:
            self.init()
        assert self._data is not None

        if self._pos >= len(self._data):
            self._eof = True
            return _EMPTY_EVENTS

        chunk = self._data[self._pos:self._pos + self._chunk_size]
        self._pos += len(chunk)
        if self._pos >= len(self._data):
            self._eof = True
        return chunk

    def read_all(self) -> EventArray:
        """Return every remaining event at once (zero-copy slice of the archive)."""
        if not self._is_initialized:
            self.init()
        assert self._data is not None
        out = self._data[self._pos:] if self._pos else self._data
        self._pos = len(self._data)
        self._eof = True
        return out

    def reset(self) -> None:
        """Reset the reader to the beginning of the archive."""
        self._pos = 0
        self._eof = False

    def tell(self) -> int:
        """Current position, in events (npz has no meaningful byte offset)."""
        return self._pos

    def close(self) -> None:
        """Drop the loaded columns."""
        self._data = None


class EventEncoder_Npz(EventEncoder):
    """Encode events into an ``.npz`` archive.

    Chunks passed to :meth:`write` are buffered in memory; the archive itself
    is written once on :meth:`close` (zip containers cannot be appended to).

    Parameters
    ----------
    writable
        Destination stream to write to.
    width, height : int
        Frame geometry stored in the archive.
    dt : datetime, optional
        Unused; npz stores no recording timestamp.
    compressed : bool
        Use ``np.savez_compressed`` instead of ``np.savez``.

    """

    def __init__(self, writable: io.BufferedWriter, width: int = 1280, height: int = 720,
                 dt: datetime | None = None, compressed: bool = False):
        super().__init__(writable, width, height, dt)
        self._compressed = compressed
        self._chunks: list[EventArray] = []
        self._closed = False

    def init(self) -> None:
        """Nothing to write up front; the archive is produced on close."""
        self._is_initialized = True

    def write(self, events: 'np.ndarray | EventArray') -> int:
        """Buffer a chunk of events for the archive.

        Parameters
        ----------
        events : np.ndarray or EventArray
            Array of events to write.

        Returns
        -------
        int
            Number of buffered events.

        """
        if not self._is_initialized:
            self.init()
        arr = events if isinstance(events, EventArray) else EventArray.from_aos(events)
        self._chunks.append(arr.copy())
        self._n_written_events += len(arr)
        return len(arr)

    def flush(self) -> None:
        """No-op: the archive can only be written once, on :meth:`close`."""

    def close(self) -> None:
        """Concatenate the buffered chunks and write the archive."""
        if self._closed:
            return
        self._closed = True
        if len(self._chunks) == 1:
            all_events = self._chunks[0]
        elif self._chunks:
            all_events = EventArray(
                np.concatenate([c.t for c in self._chunks]),
                np.concatenate([c.x for c in self._chunks]),
                np.concatenate([c.y for c in self._chunks]),
                np.concatenate([c.p for c in self._chunks]),
            )
        else:
            all_events = _EMPTY_EVENTS
        save = np.savez_compressed if self._compressed else np.savez
        save(
            self._fd,
            t=all_events.t, x=all_events.x, y=all_events.y, p=all_events.p,
            width=np.uint16(self._width), height=np.uint16(self._height),
        )
        self._chunks = []
        self._fd.flush()
