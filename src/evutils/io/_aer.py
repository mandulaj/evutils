"""Prophesee AER CD-event decoder/encoder.

AER is a raw 32-bit-per-event encoding with **no header and no timestamps**:
``y[0:8]`` (9 bits), ``x[9:17]`` (9 bits), ``p[18]``. The 9-bit fields cap
coordinates at 512 (e.g. GenX320). Decoding uses the native
``AER_parse_chunk_soa``; encoding is vectorised numpy.

Since the format carries no time information, the decoder's ``timestamps``
parameter selects how the ``t`` column is generated:

* ``"zero"`` (default) -- every event gets ``t = 0``;
* ``"sequential"`` -- ``t = t_start + i * t_step``, generated in the native
  parser and carried across chunks;
* an array -- user-provided timestamps, assigned positionally (event ``i`` in
  the stream gets ``timestamps[i]``).
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import numpy as np

from ..types import EventArray, TriggerArray
from .common import EventDecoder, EventEncoder
from ._native_core import (
    EventSoABuffers,
    TriggerSoABuffers,
    decode_all_soa,
    events_view,
    parse_step,
)
from ._native_aer import (
    AER_TS_SEQUENTIAL,
    AER_TS_ZERO,
    AerInput,
    AerParser,
)
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()


class EventDecoder_AER(EventDecoder):
    """Decode raw AER streams into ``EventArray`` chunks. Since AER is designed
    for real-time streaming, it has no header and no timestamps; see the
    ``timestamps`` parameter for how the ``t`` column is generated.

    Parameters
    ----------
    source
        Byte source to read from.
    chunk_size
        Maximum number of events produced per :meth:`read_chunk` call (the
        native output-buffer capacity). Does not bound the file size.
    timestamps : {"zero", "sequential"} or array_like, default "zero"
        Timestamp generation mode: ``"zero"`` fills ``t = 0``,
        ``"sequential"`` fills ``t = t_start + i * t_step``, and an integer
        array assigns user-provided timestamps positionally (its length must
        cover every decoded event).
    t_start, t_step : int
        Start value and per-event increment for ``"sequential"`` mode.

    References
    ----------
    [1] Prophesee AER format: https://docs.prophesee.ai/stable/data/encoding_formats/aer.html


    """

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000,
                 timestamps: 'str | np.ndarray' = "zero",
                 t_start: int = 0, t_step: int = 1):
        super().__init__(source, chunk_size)
        self._buf: Any = None
        self._words: Any = None       # uint32 view of the payload (1 word / event)
        self._offset: int = 0
        self._parser: Any = None
        self._events: Any = None
        self._triggers: Any = None

        self._custom_ts: np.ndarray | None = None
        if isinstance(timestamps, str):
            modes = {"zero": AER_TS_ZERO, "sequential": AER_TS_SEQUENTIAL}
            if timestamps not in modes:
                raise ValueError(
                    f"timestamps must be 'zero', 'sequential' or an array, got {timestamps!r}"
                )
            self._ts_mode = modes[timestamps]
        else:
            ts = np.ascontiguousarray(timestamps, dtype=np.int64)
            if ts.ndim != 1:
                raise ValueError("custom timestamps must be a 1-D array")
            self._custom_ts = ts
            self._ts_mode = AER_TS_ZERO  # parser fills 0; overwritten below
        self._t_start = int(t_start)
        self._t_step = int(t_step)
        self._n_decoded = 0  # stream position, for indexing the custom array

    def init(self) -> None:
        """Initialize the AER reader.

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

        n_events = len(self._buf) // 4  # AER has no header, 4 bytes / event
        if n_events > 0:
            self._words = np.frombuffer(self._buf, dtype=np.uint32, count=n_events)
        else:
            self._words = np.empty(0, dtype=np.uint32)

        if self._custom_ts is not None and len(self._custom_ts) < n_events:
            raise ValueError(
                f"custom timestamps array has {len(self._custom_ts)} entries, "
                f"but the stream contains {n_events} events"
            )

        self._offset = 0
        self._parser = AerParser(self._ts_mode, self._t_start, self._t_step)
        self._input_cls = AerInput
        self._word_dtype = np.uint32
        cap = int(self._chunk_size)
        self._events = EventSoABuffers(cap)
        self._triggers = TriggerSoABuffers(1)  # AER has no triggers
        self._is_initialized = True

    def _apply_custom_ts(self, t_out: np.ndarray) -> None:
        """Overwrite ``t_out`` (the decoded slice's t column, int64/uint64 view)
        with the next ``len(t_out)`` user-provided timestamps.
        """
        assert self._custom_ts is not None
        n = len(t_out)
        t_out.view(np.int64)[:] = self._custom_ts[self._n_decoded:self._n_decoded + n]

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
            self._words, self._offset, AerInput, self._parser, events, triggers,
        )
        if appended and self._custom_ts is not None:
            self._apply_custom_ts(events.t[events.size - appended:events.size])
        self._n_decoded += appended
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
        start = self._n_decoded
        # Exactly one event per uint32 word.
        out, self._offset = decode_all_soa(
            self._words, self._offset, AerInput, self._parser,
            est_events_per_word=1.0,
        )
        self._n_decoded = start + len(out)
        if self._custom_ts is not None and len(out) > 0:
            out.t[:] = self._custom_ts[start:start + len(out)]
        self._eof = True
        return out

    def reset(self) -> None:
        """Reset the AER reader to the beginning.

        Returns
        -------
        None

        """
        self._offset = 0
        self._n_decoded = 0
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
        return self._offset * 4

    def close(self) -> None:
        """Close the AER reader.

        Returns
        -------
        None

        """
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

    def __init__(self, writable: io.BufferedWriter, width: int = 512, height: int = 512, dt: datetime | None = None):
        super().__init__(writable, width, height, dt)

    def init(self) -> None:
        """Initialize the AER writer.

        Returns
        -------
        None

        """
        self._is_initialized = True  # AER has no header

    def write(self, events: 'np.ndarray | EventArray', triggers: 'np.ndarray | TriggerArray | None' = None) -> int:
        """Write events to the AER file.

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
