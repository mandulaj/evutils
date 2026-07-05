"""Prophesee AER CD-event decoder/encoder.

AER is a raw 32-bit-per-event encoding with **no header and no timestamps**:
``y[0:8]`` (9 bits), ``x[9:17]`` (9 bits), ``p[18]``. The 9-bit fields cap
coordinates at 512 (e.g. GenX320). Decoded events carry ``t = 0`` since the
format has no time information. Decoding uses the native ``AER_parse_chunk_soa``;
encoding is vectorised numpy.
"""
from __future__ import annotations

import numpy as np

from ..types import EventArray
from .common import EventDecoder, EventEncoder
from ._native_evt import (
    EVUTILS_PARSE_ERROR,
    AerInput,
    AerParser,
    EventSoABuffers,
    TriggerSoABuffers,
    decode_all_soa,
    events_view,
    parse_step,
)
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()


class EventDecoder_AER(EventDecoder):
    """Decode raw AER streams into ``EventArray`` chunks. Since AER is designed
    for real-time streaming, it has no header and no timestamps. The decoder will
    return events with ``t = 0``.

    Parameters
    ----------
    source
        Byte source to read from.
    chunk_size
        Maximum number of events produced per :meth:`read_chunk` call (the
        native output-buffer capacity). Does not bound the file size.

    References
    ----------
    [1] Prophesee AER format: https://docs.prophesee.ai/stable/data/encoding_formats/aer.html
    
    
    """

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000):
        super().__init__(source, chunk_size)
        self._buf = None
        self._words = None       # uint32 view of the payload (1 word / event)
        self._offset = 0
        self._parser = None
        self._events = None
        self._triggers = None

    def init(self) -> None:
        if self._is_initialized:
            return

        if self._source.mappable():
            self._buf = self._source.buffer()
        else:
            self._buf = memoryview(self._source.read(-1))

        n_events = len(self._buf) // 4  # AER has no header, 4 bytes / event
        if n_events > 0:
            self._words = np.frombuffer(self._buf, dtype=np.uint32, count=n_events)
        else:
            self._words = np.empty(0, dtype=np.uint32)

        self._offset = 0
        self._parser = AerParser()
        self._input_cls = AerInput
        self._word_dtype = np.uint32
        cap = int(self._chunk_size)
        self._events = EventSoABuffers(cap)
        self._triggers = TriggerSoABuffers(1)  # AER has no triggers
        self._is_initialized = True

    def parse_step(self, events, triggers) -> int:
        '''Run the parser once, appending into ``events``; advance the offset.
        See :meth:`EventDecoder_EVT.parse_step`.'''
        if not self._is_initialized:
            self.init()
        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            return 0
        appended, self._offset = parse_step(
            self._words, self._offset, AerInput, self._parser, events, triggers,
        )
        if self._offset >= len(self._words):
            self._eof = True
        return appended

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
        appended = 0
        while appended == 0 and self._offset < len(self._words):
            appended = self.parse_step(ev, tr)

        n = ev.size
        if n == 0:
            return _EMPTY_EVENTS
        # Zero-copy view (valid until the next read_chunk); see EVT decoder.
        return events_view(ev)

    def read_all(self) -> EventArray:
        """Decode the whole remaining payload into one buffer (no per-chunk copy)."""
        if not self._is_initialized:
            self.init()
        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            return _EMPTY_EVENTS
        # Exactly one event per uint32 word.
        out, self._offset = decode_all_soa(
            self._words, self._offset, AerInput, self._parser,
            est_events_per_word=1.0,
        )
        self._eof = True
        return out

    def reset(self) -> None:
        self._offset = 0
        self._eof = False

    def tell(self) -> int:
        return self._offset * 4

    def close(self) -> None:
        self._words = None
        self._buf = None


class EventEncoder_AER(EventEncoder):
    """Encode events into a raw AER stream. Since AER is designed for real-time 
    streaming, it has no header and no timestamps.
    Timestamps are dropped and coordinates are masked to 9 bits (values >= 512
    are truncated), per the AER encoding.
    
    Parameters
    ----------
    writable
        Destination stream to write to.
    width, height : int
        Frame geometry written into the header.
    dt : datetime, optional
        No effect, since AER has no timestamps.

    References
    ----------
    [1] Prophesee AER format: https://docs.prophesee.ai/stable/data/encoding_formats/aer.html
    
    
    """

    

    def __init__(self, writable, width: int = 512, height: int = 512, dt=None):
        super().__init__(writable, width, height, dt)

    def init(self):
        self._is_initialized = True  # AER has no header

    def write(self, events) -> int:
        if not self._is_initialized:
            self.init()

        if isinstance(events, EventArray):
            x, y, p = events.x, events.y, events.p
        else:
            x, y, p = events["x"], events["y"], events["p"]

        out = (
            (y.astype(np.uint32) & np.uint32(0x1FF))
            | ((x.astype(np.uint32) & np.uint32(0x1FF)) << np.uint32(9))
            | ((p.astype(np.uint32) & np.uint32(0x1)) << np.uint32(18))
        )
        out.astype(np.uint32).tofile(self._fd)
        self._n_written_events += len(out)
        return len(out)
