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

from datetime import datetime
from typing import List

import numpy as np

from ..types import Event_dtype, Trigger_dtype
from .common import EventDecoder
from ._native_evt import (
    EVUTILS_PARSE_ERROR,
    EventSoABuffers,
    Evt3Input,
    Evt3Parser,
    TriggerSoABuffers,
)
from ._source import ByteSource

_EMPTY_EVENTS = np.empty(0, dtype=Event_dtype)


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
                    self.height = int(s.split("=")[1])
                elif s.startswith("width"):
                    self.width = int(s.split("=")[1])
                else:
                    s = s.lower().replace(".", "")
                    if s in self.FORMATS:
                        self._format = s

        geom = self._header.get("geometry")
        if isinstance(geom, str):
            parts = geom.split("x")
            if len(parts) == 2:
                self.width = int(parts[0])
                self.height = int(parts[1])

        evt = self._header.get("evt")
        if evt in self.EVT_FORMATS:
            self._format = self.EVT_FORMATS[evt]

        if self._format is None:
            self._format = "evt3"  # sensible default for Prophesee RAW

        if self.width is None or self.height is None:
            if self._header.get("sensor_name") == "IMX636":
                self.width, self.height = 1280, 720
        if self.width is None or not (0 < self.width <= 2048):
            self.width = 2048
        if self.height is None or not (0 < self.height <= 2048):
            self.height = 2048

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
        out = np.empty(n, dtype=Event_dtype)
        out["t"], out["x"], out["y"], out["p"] = t, x, y, p
        return out

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
        self.eof = False
        if self._parser is not None:
            self._parser.reset()

    def tell(self) -> int:
        return self._payload_off + self._offset * 2

    def close(self) -> None:
        # Drop numpy views into the (possibly mmap-backed) storage so the source
        # can be closed without BufferError.
        self._words = None
        self._buf = None
