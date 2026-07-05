"""Prophesee DAT (.dat) CD-event decoder/encoder.

DAT is a fixed-record format: an ASCII ``% ...`` header, two bytes (event type
``0x0C`` = CD, event size ``0x08``), then 8-byte little-endian events
(``uint32`` timestamp + ``uint32`` data with ``x[0:13]``, ``y[14:27]``,
``p[28]``). Decoding is done by the native ``DAT_parse_chunk_soa`` (which also
tracks the 32-bit timestamp overflow); encoding is vectorised numpy.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from ..types import EventArray
from .common import EventDecoder, EventEncoder
from ._native_evt import (
    EVUTILS_PARSE_ERROR,
    DatInput,
    DatParser,
    EventSoABuffers,
    TriggerSoABuffers,
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

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000):
        super().__init__(source, chunk_size)
        self._header: dict = {}
        self._buf = None
        self._payload_off = 0
        self._words = None       # uint32 view of the payload (2 words / event)
        self._offset = 0         # current uint32-word offset
        self._parser = None
        self._events = None
        self._triggers = None

    # ------------------------------------------------------------------ #
    def _parse_header(self, buf) -> int:
        """Scan the leading ``%`` ASCII header; return the byte offset of the
        first non-header byte (the event-type byte)."""
        mv = memoryview(buf)
        n = len(mv)
        off = 0
        while off < n and mv[off] == 0x25:  # '%'
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
        cap = int(self._chunk_size)
        self._events = EventSoABuffers(cap)
        self._triggers = TriggerSoABuffers(1)  # DAT CD files have no triggers
        self._is_initialized = True

    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> EventArray:
        if not self._is_initialized:
            self.init()

        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            return _EMPTY_EVENTS

        ev, tr = self._events, self._triggers
        ev.reset()
        tr.reset()
        inp = DatInput(self._words[self._offset:])
        res = self._parser.parse_chunk_soa(inp, ev, tr)
        if res.status == EVUTILS_PARSE_ERROR:
            raise RuntimeError(f"DAT parse error near word {self._offset}")

        self._offset += inp.consumed(res)
        if self._offset >= len(self._words):
            self._eof = True

        n = ev.size
        if n == 0:
            return _EMPTY_EVENTS
        t, x, y, p = ev.view()
        return EventArray(t, x, y, p).copy()

    def reset(self) -> None:
        self._offset = 0
        self._eof = False
        if self._parser is not None:
            self._parser.reset()

    def tell(self) -> int:
        return self._payload_off + self._offset * 4

    def close(self) -> None:
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

    def __init__(self, writable, width: int = 1280, height: int = 720,
                 dt: datetime | None = None, version: int = 2):
        super().__init__(writable, width, height, dt)
        self._version = version

    def init(self):
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

    def write(self, events) -> int:
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
        out.tofile(self._fd)
        self._n_written_events += n
        return n
