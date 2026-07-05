"""EVT (Prophesee EVT2 / EVT2.1 / EVT3) decoder backed by the native C parser.

Reads the Prophesee ASCII header from a :class:`ByteSource`, then decodes the
binary EVT payload into ``Event_dtype`` chunks using the compiled parser in
``csrc`` (via :mod:`evutils.io._native_evt`).

Input strategy:

* If the source is *mappable* (mmap / in-memory), the whole payload is exposed
  as a single zero-copy ``uint16`` view and the parser walks it in windows --
  no per-chunk copy, no vector-group carry across chunk boundaries.
* Otherwise the remaining stream is slurped into memory once and treated the
  same way. (Truly incremental streaming for live devices is future work; the
  parser ABI already supports it via ``result.current``.)

Only EVT3 is wired to the native parser today; EVT2/EVT2.1 raise
``NotImplementedError``.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import List

import numba as nb
import numpy as np

from ..types import Event_dtype, EventArray, Trigger_dtype
from .common import EventDecoder, EventEncoder
from ._native_evt import (
    EVUTILS_PARSE_ERROR,
    EventSoABuffers,
    Evt3Input,
    Evt3Parser,
    TriggerSoABuffers,
)
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()


class EventDecoder_EVT(EventDecoder):
    """Decode Prophesee EVT2, EVT2.1, and EVT3 streams into ``Event_dtype`` chunks.

    Parameters
    ----------
    source
        Byte source to read from.
    chunk_size
        Maximum number of events produced per :meth:`read_chunk` call (the
        native output-buffer capacity). Does not bound the file size.

    References
    ----------
    [1] Prophesee RAW file format
        https://docs.prophesee.ai/stable/data/file_formats/raw.html
    """

    FORMATS = {"evt3": "evt 3.0", "evt21": "evt 2.1", "evt2": "evt 2"}
    EVT_FORMATS = {"3.0": "evt3", "2.1": "evt21", "2.0": "evt2"}

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000):
        super().__init__(source, chunk_size)

        self._format: str | None = None
        self._header: dict = {
             "date": datetime.now(),
             "evt": None,
             "format": None,
             "generation": None,
             "serial_number": "00000000",
             "system_id": 49,
             "camera_integrator_name": "Prophesee",
             "integrator_name": "Prophesee",
             "sensor_name": None,
             "sensor_generation": None,
             "geometry": None,
             "plugin_name": None,
             "plugin_integrator_name": None,
        }

        # Filled in init()
        self._buf = None            # keeps the underlying storage alive
        self._payload_off = 0       # byte offset where the binary payload starts
        self._words = None          # uint16 view of the whole payload
        self._offset = 0            # current word offset into _words
        self._parser = None


    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    def _parse_header(self, buf) -> int:
        """Scan the leading ``%``-prefixed ASCII header of ``buf`` (a bytes-like).

        Returns the byte offset of the first non-header byte (start of payload).
        """
        mv = memoryview(buf)
        n = len(mv)
        off = 0
        while off < n and mv[off] == 0x25:  # '%'
            window = bytes(mv[off:off + 8192])
            rel = window.find(b"\n")
            if rel < 0:
                break
            line = window[:rel]
            if line.startswith(b"% end"):
                off += rel + 1
                break
            self._consume_header_line(line)
            off += rel + 1
        return off

    def _consume_header_line(self, line: bytes) -> None:
        try:
            split = line.decode("utf-8").strip().split(" ")
        except UnicodeDecodeError:
            return
        if len(split) < 2:
            return
        key = split[1].lower()
        value = " ".join(split[2:])

        try:
            if key == "date":
                value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            elif key in ("height", "width", "system_id"):
                value = int(value)
        except ValueError:
            return
        self._header[key] = value

    def _finalize_header(self) -> None:
        """Resolve format / width / height from the parsed header fields."""
        fmt = self._header.get("format")
        if isinstance(fmt, str):
            for s in fmt.split(";"):
                if s.startswith("height"):
                    self._height = int(s.split("=")[1])
                elif s.startswith("width"):
                    self._width = int(s.split("=")[1])
                else:
                    s = s.lower().replace(".", "")
                    if s in self.FORMATS:
                        self._format = s

        geom = self._header.get("geometry")
        if isinstance(geom, str):
            parts = geom.split("x")
            if len(parts) == 2:
                self._width = int(parts[0])
                self._height = int(parts[1])

        evt = self._header.get("evt")
        if evt in self.EVT_FORMATS:
            self._format = self.EVT_FORMATS[evt]

        if self._format is None:
            self._format = "evt3"  # sensible default for Prophesee RAW

        if self._width is None or self._height is None:
            if self._header.get("sensor_name") == "IMX636":
                self._width, self._height = 1280, 720
        if self._width is None or not (0 < self._width <= 2048):
            self._width = 2048
        if self._height is None or not (0 < self._height <= 2048):
            self._height = 2048

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def init(self) -> None:
        if self._is_initialized:
            return

        if self._source.mappable():
            self._buf = self._source.buffer()          # zero-copy, whole file
        else:
            self._buf = memoryview(self._source.read(-1))  # slurp the stream

        self._payload_off = self._parse_header(self._buf)
        self._finalize_header()

        n_words = (len(self._buf) - self._payload_off) // 2
        if n_words > 0:
            self._words = np.frombuffer(
                self._buf, dtype=np.uint16, count=n_words, offset=self._payload_off
            )
        else:
            self._words = np.empty(0, dtype=np.uint16)

        self._offset = 0
        self._parser = Evt3Parser()
        cap = int(self._chunk_size)
        self._events = EventSoABuffers(cap)
        self._triggers = TriggerSoABuffers(max(cap // 16, 1))
        self._is_initialized = True

    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> np.ndarray:
        if not self._is_initialized:
            self.init()

        if self._format != "evt3":
            raise NotImplementedError(
                f"native decoder supports evt3 only, got {self._format!r}"
            )

        # Nothing left: signal EOF with an empty array (never a stale buffer).
        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            return _EMPTY_EVENTS

        ev, tr = self._events, self._triggers

        # Parse until we produce something or genuinely exhaust the input. A
        # window can consume words yet emit no events (pure timing packets), so
        # we must not treat an empty result as EOF unless the input is drained.
        while self._offset < len(self._words):
            ev.reset()
            tr.reset()
            view = self._words[self._offset:]
            inp = Evt3Input(view)
            res = self._parser.parse_chunk_soa(inp, ev, tr)
            if res.status == EVUTILS_PARSE_ERROR:
                raise RuntimeError(f"EVT3 parse error near word {self._offset}")

            consumed = inp.consumed(res)
            if consumed == 0:
                # The parser keeps a few words of look-ahead padding, so it will
                # not consume the final <PADDING words of the stream. Flush that
                # tail through a zero-padded scratch copy (zero words are
                # harmless EVT_ADDR_Y state updates) so trailing events aren't
                # stranded.
                self._flush_tail(view, ev, tr)
                self._offset = len(self._words)
                break
            self._offset += consumed
            if ev.size or tr.size:
                break

        if self._offset >= len(self._words):
            self._eof = True



        n = ev.size
        if n == 0:
            return _EMPTY_EVENTS
        t, x, y, p = ev.view()
        # The SoA view aliases reusable native buffers; .copy() makes the
        # returned EventArray independent (and converts t uint64 -> int64).
        return EventArray(t, x, y, p).copy()

    _TAIL_PAD = 8  # >= parser look-ahead padding (EVT3_INPUT_PADDING)

    def _flush_tail(self, view: np.ndarray, ev, tr) -> None:
        """Parse the trailing ``view`` words through a zero-padded copy so the
        parser's end-of-input look-ahead doesn't strand real events."""
        if len(view) == 0:
            return
        scratch = np.zeros(len(view) + self._TAIL_PAD, dtype=np.uint16)
        scratch[: len(view)] = view
        ev.reset()
        tr.reset()
        inp = Evt3Input(scratch)
        res = self._parser.parse_chunk_soa(inp, ev, tr)
        if res.status == EVUTILS_PARSE_ERROR:
            raise RuntimeError(f"EVT3 parse error in tail near word {self._offset}")

    def reset(self) -> None:
        self._offset = 0
        self._eof = False
        if self._parser is not None:
            self._parser.reset()

    def tell(self) -> int:
        return self._payload_off + self._offset * 2

    def close(self) -> None:
        # Drop numpy views into the (possibly mmap-backed) storage so the source
        # can be closed without BufferError.
        self._words = None
        self._buf = None


# --------------------------------------------------------------------------- #
# Encoder (EVT3 writer)
#
# The writer is still the numba EVT3 encoder (there is no native encoder yet).
# It lives here with the format module now that _raw.py is gone.
# --------------------------------------------------------------------------- #
EVT3_EVT_ADDR_Y = 0x0000
EVT3_EVT_ADDR_X = 0x2000
EVT3_VECT_BASE_X = 0x3000
EVT3_VECT_12 = 0x4000
EVT3_VECT_8 = 0x5000
EVT3_EVT_TIME_LOW = 0x6000
EVT3_CONTINUED_4 = 0x7000
EVT3_EVT_TIME_HIGH = 0x8000
EVT3_EXT_TRIGGER = 0xA000
EVT3_OTHERS = 0xE000
EVT3_CONTINUED_12 = 0xF000


@nb.njit
def get_raw_evt3_buffer(events: np.ndarray, last_lower12_ts: int, last_upper12_ts: int, last_y: int, master=True):
    # Pre-allocate large buffer
    buffer = np.zeros(len(events) * 8, dtype=np.uint8)

    # Prepare the master/slave bit
    if master:
        master_slave = 0x000
    else:
        master_slave = 0x800

    # Current position of the buffer
    i = 0

    for ev in events:
        upper12_ts = (int(ev['t']) & 0x0FFF000) >> 12
        lower12_ts = int(ev['t']) & 0x00000FFF

        # EVT_TIME_HIGH - Updates the higher 12-bit portion of the 24-bit time base
        if upper12_ts != last_upper12_ts:
            last_upper12_ts = upper12_ts
            value = EVT3_EVT_TIME_HIGH | (upper12_ts & 0xFFF)

            buffer[i] = value & 0xFF
            buffer[i + 1] = (value >> 8) & 0xFF
            i += 2

        # EVT_TIME_LOW - Updates the lower 12-bit portion of the 24-bit time base
        if lower12_ts != last_lower12_ts:
            last_lower12_ts = lower12_ts
            value = EVT3_EVT_TIME_LOW | (lower12_ts & 0xFFF)

            buffer[i] = value & 0xFF
            buffer[i + 1] = (value >> 8) & 0xFF
            i += 2

        # EVT_ADDR_Y - Y coordinate, and system type (master/slave camera)
        if last_y != ev['y']:
            last_y = ev['y']
            value = (EVT3_EVT_ADDR_Y | master_slave | (int(ev['y']) & 0x7FF))

            buffer[i] = value & 0xFF
            buffer[i + 1] = (value >> 8) & 0xFF
            i += 2

        # EVT_ADDR_X - Single valid event, X coordinate and polarity
        value = EVT3_EVT_ADDR_X | (int(ev['x']) & 0x7FF) | ((int(ev['p']) & 0x01) << 11)

        buffer[i] = value & 0xFF
        buffer[i + 1] = (value >> 8) & 0xFF
        i += 2

    return buffer[:i], last_lower12_ts, last_upper12_ts, last_y


class EventEncoder_EVT(EventEncoder):
    '''
    Encoder for Prophesee RAW/EVT files.

    Parameters
    ----------
    writable
        Destination stream to write to.
    width, height : int
        Frame geometry written into the header.
    dt : datetime, optional
        Recording timestamp (defaults to now).
    serial : str
        Camera serial number written into the header.
    format : {"evt3", "evt21", "evt2"}
        Output format. Only EVT3 is implemented.

    Notes
    -----
    The encoder currently supports EVT3 only; EVT2/EVT2.1 raise
    ``NotImplementedError``.

    References
    ----------
    [1] Prophesee RAW file format
        https://docs.prophesee.ai/stable/data/file_formats/raw.html
    '''
    FORMATS = {"evt3": "evt 3.0", "evt21": "evt 2.1", "evt2": "evt 2"}

    def __init__(self, writable: io.BufferedWriter, width: int = 1280, height: int = 720,
                 dt: datetime | None = None, serial: str = "00000000", format: str = "evt3"):
        super().__init__(writable, width, height, dt)

        format = format.lower().replace(".", "")
        if format not in EventEncoder_EVT.FORMATS.keys():
            raise ValueError(f"Unsupported format {format}. Supported formats are {list(EventEncoder_EVT.FORMATS.keys())}")
        self._format = format

        self._system_id = 49

        self._last_upper12_ts = -1
        self._last_lower12_ts = -1
        self._last_y = -1

        self._serial_number = serial

        self._formatted_datetime = self._dt.strftime("%Y-%m-%d %H:%M:%S")

    def init(self):
        if self._is_initialized:
            return

        self._fd.write(
f"""% camera_integrator_name Prophesee
% date {self._formatted_datetime}
% {self.FORMATS[self._format]}
% format {self._format};height={self._height};width={self._width}
% generation 4.2
% geometry {self._width}x{self._height}
% integrator_name Prophesee
% plugin_integrator_name Prophesee
% plugin_name hal_plugin_prophesee
% sensor_generation 4.2
% serial_number {self._serial_number}
% system_ID {self._system_id}
% end
""".encode('utf-8'))
        self._is_initialized = True

    def write(self, events: np.ndarray) -> int:
        assert self._fd is not None

        if not self._is_initialized:
            self.init()

        # Accept EventArray transparently (SoA -> AoS for the numba encoder).
        if isinstance(events, EventArray):
            events = events.to_aos()

        if self._format == "evt3":
            buffer, self._last_lower12_ts, self._last_upper12_ts, self._last_y = get_raw_evt3_buffer(
                events,
                self._last_lower12_ts,
                self._last_upper12_ts,
                self._last_y)
        else:
            raise NotImplementedError(
                f"format {self._format!r} not implemented (EVT3 only)"
            )

        self._n_written_events += len(events)

        buffer.tofile(self._fd)

        return len(events)
