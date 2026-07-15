"""EVT (Prophesee EVT2 / EVT2.1 / EVT3) decoder backed by the native C parser.

Reads the Prophesee ASCII header from a :class:`ByteSource`, then decodes the
binary EVT payload into ``EventArray`` chunks using the compiled parser in
``csrc`` (via :mod:`evutils.io._native_evt`).

Input strategy:

* If the source is *mappable* (mmap / in-memory), the whole payload is exposed
  as a single zero-copy ``uint16`` view and the parser walks it in windows --
  no per-chunk copy, no vector-group carry across chunk boundaries.
* Otherwise the remaining stream is slurped into memory once and treated the
  same way. (Truly incremental streaming for live devices is future work; the
  parser ABI already supports it via ``result.current``.)

The Prophesee formats are wired to native parsers, dispatched by the header
``format`` field: EVT3 (16-bit words), EVT2 / EVT4 (32-bit words) and EVT2.1
(64-bit words). EVT4 is not a standard Prophesee RAW variant -- it reuses EVT2's
CD/TIME_HIGH layout with distinct type codes plus vectorised CD; evutils defines
its own ``% evt 4.0`` header token for a self-consistent round-trip. All formats
have encoders.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..types import TriggerArray

from .._jit import lazy_njit
import numpy as np

from ..types import EventArray
from .common import EventDecoder, EventEncoder
from ._native_core import (
    EventSoABuffers,
    TriggerSoABuffers,
    decode_all_soa,
    events_view,
    triggers_view,
    parse_step,
)
from ._native_evt import (
    Evt2Input,
    Evt2Parser,
    Evt3Input,
    Evt3Parser,
    Evt21Input,
    Evt21Parser,
    Evt4Input,
    Evt4Parser,
)
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()

# Prophesee sensor generation -> (width, height). Older EVT2 RAWs omit an
# explicit `format`/`geometry` field, so the geometry has to be inferred from
# the sensor identity -- the same thing the Metavision SDK does.
_GEN_RESOLUTION = {
    "1": (304, 240), "1.0": (304, 240),       # Gen1 ATIS
    "2": (640, 480), "2.0": (640, 480),       # Gen2 VGA
    "3": (640, 480), "3.0": (640, 480),       # Gen3 VGA
    "3.1": (640, 480),                        # Gen3.1 VGA
    "4": (1280, 720), "4.0": (1280, 720),     # Gen4 HD
    "4.1": (1280, 720),                       # Gen4.1 HD
    "4.2": (1280, 720),                       # Gen4.2 (IMX636) HD
}

# Sensor model name -> (width, height).
_SENSOR_RESOLUTION = {
    "imx636": (1280, 720),                    # Gen4.2 HD
}

# system_ID -> (width, height), for headers that carry nothing else. Values
# observed on Prophesee EVKs: 21-29 are Gen3/Gen3.1 VGA, 40-49 are Gen4.x HD.
_SYSTEM_ID_RESOLUTION = {
    21: (640, 480), 22: (640, 480), 23: (640, 480),
    28: (640, 480), 29: (640, 480),
    40: (1280, 720), 41: (1280, 720), 42: (1280, 720),
    48: (1280, 720), 49: (1280, 720),
}


def _resolution_from_generation(gen: Any) -> tuple[int, int] | None:
    """Map a generation string (``"4.2"``, ``"gen31"``, ...) to a resolution."""
    if gen is None:
        return None
    g = str(gen).strip().lower()
    if g in _GEN_RESOLUTION:
        return _GEN_RESOLUTION[g]
    m = re.fullmatch(r"gen(\d)(\d?)", g)  # "gen31" -> "3.1", "gen4" -> "4"
    if m:
        key = m.group(1) + ("." + m.group(2) if m.group(2) else "")
        return _GEN_RESOLUTION.get(key)
    return None


# Per-format native backend: parser class, zero-copy input wrapper, and the
# numpy word dtype the binary payload is viewed as.
_BACKENDS = {
    "evt3": (Evt3Parser, Evt3Input, np.uint16),
    "evt2": (Evt2Parser, Evt2Input, np.uint32),
    "evt21": (Evt21Parser, Evt21Input, np.uint64),
    "evt4": (Evt4Parser, Evt4Input, np.uint32),
}

# Canonical format name -> (`% evt` token, `% format` token). Drives both header
# emission (encoder writes `% evt 3.0` + `% format EVT3`) and header parsing
# (decoder maps either token back to the canonical name). Single source of truth
# for the three Prophesee EVT variants.
_EVT_FORMATS: dict[str, tuple[str, str]] = {
    "evt3":  ("3.0", "EVT3"),
    "evt21": ("2.1", "EVT21"),
    "evt2":  ("2.0", "EVT2"),
    # EVT4 is not a standard Prophesee RAW header variant; there is no public
    # `% evt 4.0` recording. These tokens are evutils' own convention so the
    # encoder/decoder round-trip is self-consistent.
    "evt4":  ("4.0", "EVT4"),
}

# `% evt <token>` value -> canonical name, e.g. "3.0" -> "evt3".
_EVT_TOKEN_TO_NAME = {token: name for name, (token, _) in _EVT_FORMATS.items()}


class EventDecoder_EVT(EventDecoder):
    """Decode Prophesee EVT2, EVT2.1, and EVT3 streams into ``EventArray`` chunks.

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

    _TAIL_PAD = 8  # >= parser look-ahead padding (EVT3_INPUT_PADDING)
    SUPPORTS_EXT_TRIGGERS = True

    # Per-format TIME_HIGH record descriptor: (type-field right-shift, type code).
    # Used to skip leading records until the first TIME_HIGH establishes a valid
    # time base -- events before it carry an undefined timestamp (a stream sliced
    # after capture start begins mid-group). Matches OpenEB / the reference
    # decoders, which drop those events.
    _TIME_HIGH_TYPE = {
        "evt3":  (12, 0x8),   # 16-bit words, type in bits 12..15
        "evt2":  (28, 0x8),   # 32-bit words, type in bits 28..31
        "evt21": (28, 0x8),
        "evt4":  (28, 0xE),   # EVT4 TIME_HIGH code differs (0xE)
    }

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000, read_external_triggers: bool = False):
        super().__init__(source, chunk_size, read_external_triggers=read_external_triggers)

        self._format: str | None = None
        self._header: Dict[str, Any] = {
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
        self._buf: Any = None            # keeps the underlying storage alive
        self._payload_off: int = 0       # byte offset where the binary payload starts
        self._words: Any = None          # uint16 view of the whole payload
        self._offset: int = 0            # current word offset into _words
        self._start_offset: int = 0      # word offset of the first TIME_HIGH
        self._parser: Any = None


    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    def _parse_header(self, buf: Any) -> int:
        """Scan the leading ``%``-prefixed ASCII header of ``buf`` (a bytes-like).

        Returns the byte offset of the first non-header byte (start of payload).
        """
        mv = memoryview(buf)
        n = len(mv)
        off = 0
        # A header line always starts with "% " (0x25 0x20). The `% end` marker
        # is optional -- the payload begins at the first line that does *not*
        # start with "% ", so that two-byte prefix is the real terminator.
        while off + 1 < n and mv[off] == 0x25 and mv[off + 1] == 0x20:
            window = bytes(mv[off:off + 8192])
            rel = window.find(b"\n")
            if rel < 0:
                break
            line = window[:rel]
            # Exact end-of-header marker. Must not match other keys such as
            # "% endianness ...".
            if line.strip() == b"% end":
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
        raw = " ".join(split[2:])

        value: str | int | datetime = raw
        try:
            if key == "date":
                value = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
            elif key in ("height", "width", "system_id"):
                value = int(raw)
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
                    if s in _EVT_FORMATS:
                        self._format = s

        geom = self._header.get("geometry")
        if isinstance(geom, str):
            parts = geom.split("x")
            if len(parts) == 2:
                self._width = int(parts[0])
                self._height = int(parts[1])

        evt = self._header.get("evt")
        if evt in _EVT_TOKEN_TO_NAME:
            self._format = _EVT_TOKEN_TO_NAME[evt]

        if self._format is None:
            self._format = "evt3"  # sensible default for Prophesee RAW

        # Geometry not stated explicitly (older EVT2 RAWs): infer it from the
        # sensor identity, the way the Metavision SDK does.
        if self._width is None or self._height is None:
            res = self._infer_resolution()
            if res is not None:
                self._width, self._height = res

        if self._width is None or not (0 < self._width <= 2048):
            self._width = 2048
        if self._height is None or not (0 < self._height <= 2048):
            self._height = 2048

    def _infer_resolution(self) -> tuple[int, int] | None:
        """Guess (width, height) from the sensor identity fields in the header.

        Tries, in order of reliability: an explicit sensor model, the
        ``sensor_generation`` / ``generation`` fields, the generation token
        embedded in ``plugin_name`` (e.g. ``hal_plugin_gen31_fx3``), and
        finally the ``system_ID``. Returns ``None`` if nothing matches.
        """
        h = self._header

        name = h.get("sensor_name")
        if isinstance(name, str):
            res = _SENSOR_RESOLUTION.get(name.strip().lower())
            if res:
                return res

        for key in ("sensor_generation", "generation"):
            res = _resolution_from_generation(h.get(key))
            if res:
                return res

        plugin = h.get("plugin_name")
        if isinstance(plugin, str):
            low = plugin.lower()
            for model, res in _SENSOR_RESOLUTION.items():
                if model in low:
                    return res
            m = re.search(r"gen(\d)(\d?)", low)  # hal_plugin_gen31_fx3 -> "3.1"
            if m:
                key = m.group(1) + ("." + m.group(2) if m.group(2) else "")
                res = _GEN_RESOLUTION.get(key)
                if res:
                    return res

        return _SYSTEM_ID_RESOLUTION.get(h.get("system_id"))

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def init(self) -> None:
        """Initialize the EVT reader.

        Returns
        -------
        None

        """
        if self._is_initialized:
            return

        if self._source.mappable():
            self._buf = self._source.buffer()          # zero-copy, whole file
        else:
            self._buf = memoryview(self._source.read(-1))  # slurp the stream

        self._payload_off = self._parse_header(self._buf)
        self._finalize_header()

        if self._format not in _BACKENDS:
            raise NotImplementedError(
                f"native decoder does not support format {self._format!r}"
            )
        parser_cls, input_cls, word_dtype = _BACKENDS[self._format]
        self._input_cls = input_cls
        self._word_dtype = word_dtype

        # View the binary payload as words of the format's native width. The
        # payload can be unaligned relative to the word size (the header length
        # is arbitrary); numpy tolerates this and x86 handles the unaligned
        # loads in the C parser.
        itemsize = np.dtype(word_dtype).itemsize
        n_words = (len(self._buf) - self._payload_off) // itemsize
        if n_words > 0:
            self._words = np.frombuffer(
                self._buf, dtype=word_dtype, count=n_words, offset=self._payload_off
            )
        else:
            self._words = np.empty(0, dtype=word_dtype)

        self._start_offset = self._find_first_time_high()
        self._offset = self._start_offset
        self._parser = parser_cls()
        cap = int(self._chunk_size)
        self._events = EventSoABuffers(cap)
        self._triggers = TriggerSoABuffers(max(cap // 16, 1))
        self._is_initialized = True

    def _find_first_time_high(self) -> int:
        """Word offset of the first TIME_HIGH record in the payload.

        Records before the first TIME_HIGH have no established time base (their
        timestamp would decode to 0), which happens when a stream is sliced
        after capture start and begins mid-group. The reference decoders drop
        those records; starting decode at the first TIME_HIGH does the same
        while keeping the raw, absolute timestamps that follow.

        Scans in exponentially growing blocks -- the first TIME_HIGH is almost
        always within the first few thousand words, so the whole payload is
        rarely touched. Returns 0 if the format has no TIME_HIGH descriptor or
        none is found (leave the stream untouched).
        """
        desc = self._TIME_HIGH_TYPE.get(self._format or "")
        if desc is None or self._words is None:
            return 0
        shift, code = desc
        words = self._words
        n = len(words)
        start = 0
        block = 1 << 16
        while start < n:
            stop = min(start + block, n)
            seg = words[start:stop]
            hits = ((seg >> shift) & 0xF) == code
            idx = int(np.argmax(hits))
            if hits[idx]:
                return start + idx
            start = stop
            block = min(block * 4, 1 << 24)
        return 0

    @property
    def _tail_pad(self) -> int:
        return self._TAIL_PAD if self._format == "evt3" else 0

    @property
    def _exact_window(self) -> bool:
        """True when the parser emits exactly one event per record, so it fills
        an output buffer to precisely its capacity. Only EVT2 qualifies: EVT3,
        EVT2.1 *and EVT4* all expand vector groups (a single word can emit up to
        32 events and the parser stops a few short of capacity), so they cannot
        use EventReader's zero-copy n_events fast path. Note EVT4's encoder only
        writes scalar CD, but the decoder must stay correct for vectorised EVT4
        input too."""
        return self._format == "evt2"

    def parse_step(self, events: EventSoABuffers, triggers: TriggerSoABuffers) -> int:
        """Run the parser once, appending decoded events into ``events``.

        Advances the internal word offset and sets EOF when the input is drained.

        Parameters
        ----------
        events : EventSoABuffers
            Buffer to append events to.
        triggers : TriggerSoABuffers
            Buffer to append triggers to.

        Returns
        -------
        int
            Number of events appended.

        """
        if not self._is_initialized:
            self.init()
        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            return 0
        appended, self._offset = parse_step(
            self._words, self._offset, self._input_cls, self._parser,
            events, triggers, tail_pad=self._tail_pad, word_dtype=self._word_dtype,
        )
        if self._offset >= len(self._words):
            self._eof = True
        return appended

    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> 'EventArray | tuple[EventArray, TriggerArray]':
        if not self._is_initialized:
            self.init()

        # Nothing left: signal EOF with an empty array (never a stale buffer).
        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            if self.read_external_triggers:
                from ..types import TriggerArray
                return _EMPTY_EVENTS, TriggerArray.empty()
            return _EMPTY_EVENTS

        ev, tr = self._events, self._triggers
        ev.reset()
        tr.reset()

        # Parse until we produce something or genuinely exhaust the input. A
        # window can consume words yet emit no events (pure timing packets), so
        # we must not treat an empty result as EOF unless the input is drained.
        appended = 0
        while appended == 0 and self._offset < len(self._words):
            appended = self.parse_step(ev, tr)

        n = ev.size
        ev_view = events_view(ev) if n > 0 else _EMPTY_EVENTS
        if self.read_external_triggers:
            from ..types import TriggerArray
            tr_view = triggers_view(tr) if tr.size > 0 else TriggerArray.empty()
            return ev_view, tr_view
        return ev_view

    # Initial output-capacity estimate (events per input word) for the
    # single-buffer read_all() path. EVT2/EVT4 are exact upper bounds (<=1 event
    # per 32-bit word); EVT3/EVT2.1 vary with vector density. The estimate only
    # needs to be a rough ballpark: if it is too small decode_all_soa grows the
    # buffer once, extrapolating the true count from the fraction of input
    # consumed so far (no repeated reallocation even for dense EVT2.1 vector
    # streams, which reach ~14+ events per word).
    _READ_ALL_EST = {"evt3": 1.0, "evt2": 1.0, "evt21": 1.5, "evt4": 1.0}

    def read_all(self) -> 'EventArray | tuple[EventArray, TriggerArray]':
        """Decode the whole remaining payload into one buffer (no per-chunk copy).

        See :func:`evutils.io._native_core.decode_all_soa`. Note this materialises
        every event at once; for very large recordings that do not fit in memory,
        iterate with :meth:`read_chunk` (via ``EventReader``) instead.
        """
        if not self._is_initialized:
            self.init()

        # decode_all_soa is an events-only fast path; with external triggers
        # requested, fall back to the chunked base implementation (which
        # carries the trigger stream through).
        if self.read_external_triggers:
            return super().read_all()

        if self._words is None or self._offset >= len(self._words):
            self._eof = True
            return _EMPTY_EVENTS

        assert self._format is not None  # set by init()
        out, self._offset = decode_all_soa(
            self._words, self._offset, self._input_cls, self._parser,
            est_events_per_word=self._READ_ALL_EST.get(self._format, 1.0),
            tail_pad=self._tail_pad, word_dtype=self._word_dtype,
        )
        self._eof = True
        return out

    def reset(self) -> None:
        """Reset the EVT reader to the beginning.

        Returns
        -------
        None

        """
        self._offset = self._start_offset
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
        if not self._is_initialized:
            return 0
        word_size = np.dtype(self._word_dtype).itemsize
        return self._payload_off + self._offset * word_size

    def close(self) -> None:
        """Close the EVT reader.

        Returns
        -------
        None

        """
        # Drop numpy views into the (possibly mmap-backed) storage so the source
        # can be closed without BufferError.
        self._words = None
        self._buf = None


# --------------------------------------------------------------------------- #
# Encoders (EVT3 / EVT2 / EVT2.1 writers)
#
# The writers are numba (there is no native encoder yet). They live here with
# the format module now that _raw.py is gone.
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


@lazy_njit
def get_raw_evt3_buffer(events: np.ndarray, last_lower12_ts: int, last_upper12_ts: int, last_y: int, master: bool = True) -> tuple[np.ndarray, int, int, int]:
    """Encode events as EVT3.

    Parameters
    ----------
    events : np.ndarray
        Array of events to encode.
    last_lower12_ts : int
        Last lower 12-bit timestamp.
    last_upper12_ts : int
        Last upper 12-bit timestamp.
    last_y : int
        Last y coordinate.
    master : bool, optional
        Whether this is the master camera, by default True.

    Returns
    -------
    tuple
        A tuple containing the raw buffer, last lower 12-bit timestamp, last upper 12-bit timestamp, and last y coordinate.

    """
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


@lazy_njit
def get_raw_evt2_buffer(events: np.ndarray, last_ts_high: int) -> tuple[np.ndarray, int]:
    """Encode events as EVT2 (32-bit words).

    Timestamp is split into a 28-bit high part (EVT_TIME_HIGH word) and a 6-bit
    low part carried in each CD word. A TIME_HIGH word is emitted only when the
    high part changes. Layout: type[28:31], ts_low[22:27], x[11:21], y[0:10].

    Parameters
    ----------
    events : np.ndarray
        Array of events to encode.
    last_ts_high : int
        Last high timestamp part.

    Returns
    -------
    tuple
        A tuple containing the raw buffer and last high timestamp part.

    """
    n = len(events)
    buffer = np.empty(2 * n, dtype=np.uint32)  # <= 1 TIME_HIGH + 1 CD per event
    i = 0
    for k in range(n):
        ts = np.int64(events[k]['t'])
        x = np.int64(events[k]['x']) & 0x7FF
        y = np.int64(events[k]['y']) & 0x7FF
        p = np.int64(events[k]['p']) & 0x1

        ts_high = (ts >> 6) & 0x0FFFFFFF
        ts_low = ts & 0x3F

        if ts_high != last_ts_high:
            buffer[i] = np.uint32((8 << 28) | ts_high)  # EVT2_EVT_TIME_HIGH
            i += 1
            last_ts_high = int(ts_high)

        buffer[i] = np.uint32((p << 28) | (ts_low << 22) | (x << 11) | y)
        i += 1

    return buffer[:i], last_ts_high


@lazy_njit
def get_raw_evt21_buffer(events: np.ndarray, last_ts_high: int) -> tuple[np.ndarray, int]:
    """Encode events as EVT2.1 (64-bit words, legacy endianness).

    Same descriptor layout as EVT2 in the low 32 bits (type[28:31],
    ts_low[22:27], x_base[11:21], y[0:10]); the high 32 bits are a validity
    bitmask for x_base..x_base+31. This writer emits one event per word (mask
    with a single bit set at x_base = x) -- valid EVT2.1, not yet vectorised.

    Parameters
    ----------
    events : np.ndarray
        Array of events to encode.
    last_ts_high : int
        Last high timestamp part.

    Returns
    -------
    tuple
        A tuple containing the raw buffer and last high timestamp part.

    """
    n = len(events)
    buffer = np.empty(2 * n, dtype=np.uint64)  # <= 1 TIME_HIGH + 1 CD per event
    i = 0
    for k in range(n):
        ts = np.int64(events[k]['t'])
        x = np.int64(events[k]['x']) & 0x7FF
        y = np.int64(events[k]['y']) & 0x7FF
        p = np.int64(events[k]['p']) & 0x1

        ts_high = (ts >> 6) & 0x0FFFFFFF
        ts_low = ts & 0x3F

        if ts_high != last_ts_high:
            buffer[i] = np.uint64((8 << 28) | ts_high)  # EVT21_EVT_TIME_HIGH
            i += 1
            last_ts_high = int(ts_high)

        desc = (p << 28) | (ts_low << 22) | (x << 11) | y
        # High 32 bits: validity mask with bit 0 set (single event at x_base=x).
        buffer[i] = (np.uint64(1) << np.uint64(32)) | np.uint64(desc)
        i += 1

    return buffer[:i], last_ts_high


@lazy_njit
def get_raw_evt4_buffer(events: np.ndarray, last_ts_high: int) -> tuple[np.ndarray, int]:
    """Encode events as EVT4 (32-bit words).

    Same CD / TIME_HIGH bit layout as EVT2 (type[28:31], ts_low[22:27],
    x[11:21], y[0:10]) but with EVT4's type codes: CD_OFF=0xA / CD_ON=0xB and
    TIME_HIGH=0xE. One CD word per event (EVT4's vectorised CD_VEC form is a
    decode-side optimisation and is not emitted here).

    Parameters
    ----------
    events : np.ndarray
        Array of events to encode.
    last_ts_high : int
        Last high timestamp part.

    Returns
    -------
    tuple
        A tuple containing the raw buffer and last high timestamp part.

    """
    n = len(events)
    buffer = np.empty(2 * n, dtype=np.uint32)  # <= 1 TIME_HIGH + 1 CD per event
    i = 0
    for k in range(n):
        ts = np.int64(events[k]['t'])
        x = np.int64(events[k]['x']) & 0x7FF
        y = np.int64(events[k]['y']) & 0x7FF
        p = np.int64(events[k]['p']) & 0x1

        ts_high = (ts >> 6) & 0x0FFFFFFF
        ts_low = ts & 0x3F

        if ts_high != last_ts_high:
            buffer[i] = np.uint32((0xE << 28) | ts_high)  # EVT4_EVT_TIME_HIGH
            i += 1
            last_ts_high = int(ts_high)

        # type = CD_OFF (0xA) for p=0, CD_ON (0xB) for p=1.
        buffer[i] = np.uint32(((0xA | p) << 28) | (ts_low << 22) | (x << 11) | y)
        i += 1

    return buffer[:i], last_ts_high


class EventEncoder_EVT(EventEncoder):
    """Encoder for Prophesee RAW/EVT files.

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
    format : {"evt3", "evt21", "evt2", "evt4"}
        Output format. All are supported; EVT2.1 and EVT4 are written one event
        per word (valid but not vectorised).

    References
    ----------
    [1] Prophesee RAW file format
        https://docs.prophesee.ai/stable/data/file_formats/raw.html

    """

    def __init__(self, writable: io.BufferedWriter, width: int = 1280, height: int = 720,
                 dt: datetime | None = None, serial: str = "00000000", format: str = "evt3"):
        super().__init__(writable, width, height, dt)

        format = format.lower().replace(".", "")
        if format not in _EVT_FORMATS:
            raise ValueError(f"Unsupported format {format}. Supported formats are {list(_EVT_FORMATS)}")
        self._format = format

        self._system_id = 49

        self._last_upper12_ts = -1
        self._last_lower12_ts = -1
        self._last_y = -1
        self._last_ts_high = -1  # EVT2 / EVT2.1 time-high state

        self._serial_number = serial

        self._formatted_datetime = self._dt.strftime("%Y-%m-%d %H:%M:%S")

    def init(self) -> None:
        """Initialize the EVT writer.

        Returns
        -------
        None

        """
        if self._is_initialized:
            return

        # EVT2.1 packs its 64-bit words as two swapped 32-bit halves ("legacy").
        endianness = "% endianness legacy\n" if self._format == "evt21" else ""
        self._fd.write(
f"""% camera_integrator_name Prophesee
% date {self._formatted_datetime}
{endianness}% evt {_EVT_FORMATS[self._format][0]}
% format {_EVT_FORMATS[self._format][1]};height={self._height};width={self._width}
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

    def write(self, events: 'np.ndarray | EventArray', triggers: 'np.ndarray | TriggerArray | None' = None) -> int:
        """Write events to the EVT file.

        Parameters
        ----------
        events : np.ndarray or EventArray
            Array of events to write.

        Returns
        -------
        int
            Number of written events.

        """
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
        elif self._format == "evt2":
            buffer, self._last_ts_high = get_raw_evt2_buffer(events, self._last_ts_high)
        elif self._format == "evt21":
            buffer, self._last_ts_high = get_raw_evt21_buffer(events, self._last_ts_high)
        elif self._format == "evt4":
            buffer, self._last_ts_high = get_raw_evt4_buffer(events, self._last_ts_high)
        else:
            raise NotImplementedError(
                f"format {self._format!r} not implemented"
            )

        self._n_written_events += len(events)

        buffer.tofile(self._fd)

        return len(events)
