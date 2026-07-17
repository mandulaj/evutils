"""Prophesee DAT (.dat) CD-event decoder/encoder.

DAT is a fixed-record format: an ASCII ``% ...`` header, two bytes (event type
``0x0C`` = CD, event size ``0x08``), then 8-byte little-endian events
(``uint32`` timestamp + ``uint32`` data with ``x[0:13]``, ``y[14:27]``,
``p[28]``). Decoding is done by the native ``DAT_parse_chunk_soa`` (which also
tracks the 32-bit timestamp overflow); encoding is vectorised numpy.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np

from ..types import EventArray, TriggerArray
from .common import EventDecoder, EventEncoder
from ._native_core import (
    EVUTILS_PARSE_ERROR,
    EventSoABuffers,
    TriggerSoABuffers,
    decode_all_soa,
    events_view,
    triggers_view,
    parse_step,
)
from ._native_dat import (
    DatInput,
    DatParser,
)
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()

DAT_EVENT_TYPE_CD = 0x0C
DAT_EVENT_SIZE = 0x08

class EventDecoder_Dat(EventDecoder):
    """Decode Prophesee DAT files into ``EventArray`` chunks.
    
    Parameters
    ----------
    source
        Byte source to read from.
    chunk_size
        Maximum number of events produced per :meth:`read_chunk` call (the
        native output-buffer capacity). Does not bound the file size.

    References
    ----------
    [1] Prophesee DAT file format
        https://docs.prophesee.ai/stable/data/file_formats/dat.html

    """

    #: DAT is a fixed 2-word-per-event format, so the parser fills an output
    #: buffer to exactly its capacity -> eligible for EventReader's zero-copy
    #: n_events fast path.
    _exact_window = True

    #: Fixed 8-byte records make event-index <-> byte-offset exact, and the raw
    #: 32-bit timestamps are the even-strided words -> seek is direct index math
    #: / a searchsorted, no index file needed. Assumes timestamps do not wrap
    #: the 32-bit field (~71 min); a wrapped recording would seek approximately.
    SUPPORTS_SEEK = True

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000):
        super().__init__(source, chunk_size)
        self._header: dict[str, str | int | float] = {}
        self._buf: bytes | bytearray | None = None
        self._payload_off: int = 0
        self._words: "np.ndarray | None" = None       # uint32 view of the payload (2 words / event)
        self._offset: int = 0         # current uint32-word offset
        self._parser: "Callable | None" = None
        self._events: "EventArray | None" = None
        self._triggers: "TriggerArray | None" = None

    # ------------------------------------------------------------------ #
    def _parse_header(self, buf: bytes) -> int:
        """Scan the leading ``%`` ASCII header; return the byte offset of the
        first non-header byte (the event-type byte).
        """
        mv = memoryview(buf)
        n = len(mv)
        off = 0
        # Header lines start with "% "; the binary section (event-type byte,
        # then records) begins at the first line without that prefix.
        while off + 1 < n and mv[off] == 0x25 and mv[off + 1] == 0x20:
            window = bytes(mv[off:off + 8192])
            rel = window.find(b"\n")
            if rel < 0:
                break
            self._consume_header_line(window[:rel])
            off += rel + 1
        return off

    def _consume_header_line(self, line: bytes) -> None:
        try:
            parts = line.decode("ascii", "ignore").strip().split()
        except ValueError:
            return
        # e.g. ["%", "Width", "1280"]
        if len(parts) >= 3:
            key = parts[1].lower()
            try:
                if key == "width":
                    self._width = int(parts[2])
                elif key == "height":
                    self._height = int(parts[2])
            except ValueError:
                pass

    def init(self) -> None:
        """Initialize the DAT reader.

        Returns
        -------
        None

        """
        if self._is_initialized:
            return

        if self._source.mappable():
            self._buf = self._source.buffer()
        else:
            self._buf = memoryview(self._source.read(-1))

        off = self._parse_header(self._buf)
        # Two-byte binary header: event type, event size.
        if off + 2 <= len(self._buf):
            event_size = self._buf[off + 1]
            off += 2
            if event_size not in (0, DAT_EVENT_SIZE):
                raise NotImplementedError(
                    f"DAT event size {event_size} not supported (only 8-byte CD)"
                )
        self._payload_off = off

        n_events = (len(self._buf) - off) // 8
        if n_events > 0:
            self._words = np.frombuffer(
                self._buf, dtype=np.uint32, count=n_events * 2, offset=off
            )
        else:
            self._words = np.empty(0, dtype=np.uint32)

        self._offset = 0
        self._parser = DatParser()
        self._input_cls = DatInput
        self._word_dtype = np.uint32
        cap = int(self._chunk_size)
        self._events = EventSoABuffers(cap)
        self._triggers = TriggerSoABuffers(1)  # DAT CD files have no triggers
        self._is_initialized = True

    def parse_step(self, events: EventSoABuffers, triggers: TriggerSoABuffers) -> int:
        """Run the parser once, appending into ``events``; advance the offset.
        See :meth:`EventDecoder_EVT.parse_step`.
        """
        if not self._is_initialized:
            self.init()
        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            return 0
        appended, self._offset = parse_step(
            self._words, self._offset, DatInput, self._parser, events, triggers,
        )
        if self._offset >= len(self._words):
            self._eof = True
        return appended

    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> 'EventArray | tuple[EventArray, TriggerArray]':
        if not self._is_initialized:
            self.init()

        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            if self.read_external_triggers:
                from ..types import TriggerArray
                return _EMPTY_EVENTS, TriggerArray.empty()
            return _EMPTY_EVENTS

        ev, tr = self._events, self._triggers
        ev.reset()
        tr.reset()
        appended = 0
        while appended == 0 and self._offset < len(self._words):
            appended = self.parse_step(ev, tr)

        n = ev.size
        if n == 0:
            if self.read_external_triggers:
                from ..types import TriggerArray
                return _EMPTY_EVENTS, TriggerArray.empty()
            return _EMPTY_EVENTS
        # Zero-copy view (valid until the next read_chunk); see EVT decoder.
        if self.read_external_triggers:
            return events_view(ev), triggers_view(tr)
        return events_view(ev)

    def read_all(self) -> 'EventArray | tuple[EventArray, TriggerArray]':
        """Decode the whole remaining payload into one buffer (no per-chunk copy)."""
        if not self._is_initialized:
            self.init()
        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            return _EMPTY_EVENTS
        # Exactly one event per two uint32 words.
        out, self._offset = decode_all_soa(
            self._words, self._offset, DatInput, self._parser,
            est_events_per_word=0.5,
        )
        self._eof = True
        return out

    def seek(self, t: int | None = None, n: int | None = None) -> tuple["SeekResult", "EventArray", "TriggerArray | None"]:
        """Seek to an absolute timestamp (µs) or event index. See base class.

        Fixed 8-byte records: event ``k`` lives at word offset ``2k`` and its
        timestamp is the even-strided word ``_words[2k]``. Time seek is a
        ``searchsorted`` over those words; index seek is direct. Assumes the
        raw 32-bit timestamps are non-decreasing (no wrap).
        """
        from .common import SeekResult
        if not self._is_initialized:
            self.init()
        axis, val = self._seek_axis(t, n)

        n_events = 0 if self._words is None else len(self._words) // 2
        ts_view = (self._words[0::2] if n_events
                   else np.empty(0, dtype=np.uint32))
        
        if not hasattr(self, "_wrap_indices"):
            if n_events > 0:
                wraps = np.where(ts_view[1:] < ts_view[:-1])[0] + 1
                self._wrap_indices = np.concatenate(([0], wraps, [n_events]))
            else:
                self._wrap_indices = np.array([0, 0])

        if axis == "t":
            bucket = int(val >> 32)
            val_rem = int(val & 0xFFFFFFFF)
            
            if bucket >= len(self._wrap_indices) - 1:
                idx = n_events
            else:
                start_idx = self._wrap_indices[bucket]
                end_idx = self._wrap_indices[bucket + 1]
                idx = start_idx + int(np.searchsorted(ts_view[start_idx:end_idx], val_rem, side="left"))
        else:
            idx = val
        idx = max(0, min(idx, n_events))

        self._offset = idx * 2
        self._eof = idx >= n_events
        if self._parser is not None:
            wrap_count = max(0, int(np.searchsorted(self._wrap_indices, idx, side="right")) - 1)
            wrap_offset = wrap_count * (1 << 32)
            self._parser.reset(wrap_offset)
        else:
            wrap_offset = max(0, int(np.searchsorted(self._wrap_indices, idx, side="right")) - 1) * (1 << 32)
            
        landed_ts = int(ts_view[idx]) + wrap_offset if idx < n_events else val
        return SeekResult(ts=landed_ts, index=idx, eof=self._eof), _EMPTY_EVENTS, None

    def reset(self) -> None:
        """Reset the DAT reader to the beginning.

        Returns
        -------
        None

        """
        self._offset = 0
        self._eof = False
        if self._parser is not None:
            self._parser.reset()

    def tell(self) -> int:
        """Get the current byte offset.

        Returns
        -------
        int
            Current byte offset.

        """
        return self._payload_off + self._offset * 4

    def close(self) -> None:
        """Close the DAT reader.

        Returns
        -------
        None

        """
        self._words = None
        self._buf = None

class EventEncoder_Dat(EventEncoder):
    """Encode events into a Prophesee DAT (.dat) CD file.

    Parameters
    ----------
    writable
        Destination stream to write to.
    width, height : int
        Frame geometry written into the header.
    dt : datetime, optional
        Recording timestamp (defaults to now).
    version : int
        DAT version (defaults to 2).

    References
    ----------
    [1] Prophesee DAT file format
        https://docs.prophesee.ai/stable/data/file_formats/dat.html

    """

    def __init__(self, writable: io.BufferedWriter, width: int = 1280, height: int = 720,
                 dt: datetime | None = None, version: int = 2):
        super().__init__(writable, width, height, dt)
        self._version = version

    def init(self) -> None:
        """Initialize the DAT writer.

        Returns
        -------
        None

        """
        if self._is_initialized:
            return
        header = (
            "% Data file containing CD events.\n"
            f"% Version {self._version}\n"
            f"% Date {self._dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"% Height {self._height}\n"
            f"% Width {self._width}\n"
        )
        self._fd.write(header.encode("ascii"))
        self._fd.write(bytes([DAT_EVENT_TYPE_CD, DAT_EVENT_SIZE]))
        self._is_initialized = True

    def write(self, events: 'np.ndarray | EventArray', triggers: 'np.ndarray | TriggerArray | None' = None) -> int:
        """Write events to the DAT file.

        Parameters
        ----------
        events : np.ndarray or EventArray
            Array of events to write.

        Returns
        -------
        int
            Number of written events.

        """
        if not self._is_initialized:
            self.init()

        if isinstance(events, EventArray):
            t, x, y, p = events.t, events.x, events.y, events.p
        else:
            t, x, y, p = events["t"], events["x"], events["y"], events["p"]

        n = len(t)
        out = np.empty(n * 2, dtype=np.uint32)
        out[0::2] = (np.asarray(t).astype(np.uint64) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        out[1::2] = (
            (x.astype(np.uint32) & np.uint32(0x3FFF))
            | ((y.astype(np.uint32) & np.uint32(0x3FFF)) << np.uint32(14))
            | ((p.astype(np.uint32) & np.uint32(0x1)) << np.uint32(28))
        )
        self._fd.write(out.tobytes())
        self._n_written_events += n
        return n
