"""AEDAT (jAER / cAER / DV) file decoder, versions 1.0 -- 4.0.

All four on-disk layouts used by iniVation event cameras are supported for
reading (writing is not implemented yet):

* **AEDAT 1.0** (jAER, 2008): optional ``#``-prefixed ASCII header, then
  6-byte big-endian records: ``uint16`` address + ``uint32`` timestamp (µs).
  DVS128 address layout: ``p = bit 0``, ``x = bits 1-7``, ``y = bits 8-14``.
* **AEDAT 2.0** (jAER, 2010): ``#!AER-DAT2.0`` header, then 8-byte big-endian
  records: ``uint32`` address + ``uint32`` timestamp (µs). The address layout
  depends on the camera -- see the ``layout`` parameter (default ``"davis"``:
  ``p = bit 11``, ``x = bits 12-21``, ``y = bits 22-30``; APS/IMU words with
  bit 31 set are skipped).
* **AEDAT 3.1** (cAER): header terminated by ``#!END-HEADER``, then
  little-endian packets with a 28-byte header; polarity-event packets carry
  8-byte events (``uint32`` data + ``uint32`` timestamp): validity ``bit 0``,
  ``p = bit 1``, ``y = bits 2-16``, ``x = bits 17-31``. The 31-bit packet
  timestamp is extended by the header's TS-overflow counter to 64 bits.
* **AEDAT 4.0** (DV framework): ``#!AER-DAT4.0`` version line, a FlatBuffer
  ``IOHeader`` (compression type, stream table), then packets of size-prefixed
  ``EventPacket`` FlatBuffers (identifier ``EVTS``), optionally LZ4- or
  Zstd-compressed, holding 16-byte event structs (``int64`` t, ``int16`` x,
  ``int16`` y, ``uint8`` p). Compressed files need the optional ``lz4`` /
  ``zstandard`` package (``pip install evutils[aedat]``).

The byte order and record layouts for 1.0/2.0/3.1 follow the official
iniVation file-format documentation (jAER writes big-endian); the 4.0 layout
follows dv-processing (cross-checked against the evlib reference reader).

Decoding streams packet-by-packet / chunk-by-chunk -- the whole recording is
never materialised.

References
----------
[1] https://docs.inivation.com/software/software-advanced-usage/file-formats/
[2] https://github.com/tallamjr/evlib (aedat_reader.rs, aedat4_reader.rs)
"""
from __future__ import annotations

import re
import struct
from typing import Callable

import numpy as np

from ..types import EventArray, TriggerArray
from .common import EventDecoder, EventEncoder
from ._source import ByteSource

_EMPTY_EVENTS = EventArray.empty()

# ---------------------------------------------------------------------------#
# Record dtypes
# ---------------------------------------------------------------------------#
#: AEDAT 1.0: big-endian uint16 address + uint32 timestamp (6 bytes).
_V1_DTYPE = np.dtype([("a", ">u2"), ("t", ">u4")])
#: AEDAT 2.0: big-endian uint32 address + uint32 timestamp (8 bytes).
_V2_DTYPE = np.dtype([("a", ">u4"), ("t", ">u4")])
#: AEDAT 3.1 polarity event: little-endian uint32 data + uint32 timestamp.
_V3_EVENT_DTYPE = np.dtype([("d", "<u4"), ("t", "<u4")])
#: AEDAT 3.1 packet header (28 bytes, little-endian).
_V3_HEADER = struct.Struct("<hhiiiiii")
#: AEDAT 4.0 event struct: int64 t, int16 x, int16 y, uint8 p, 3 pad bytes.
_V4_EVENT_DTYPE = np.dtype({
    "names": ["t", "x", "y", "p"],
    "formats": ["<i8", "<i2", "<i2", "u1"],
    "offsets": [0, 8, 10, 12],
    "itemsize": 16,
})

_V3_POLARITY_EVENT = 1  # cAER event type id for polarity events

# AEDAT 2.0 address layouts: name -> (extractor, aps_mask_bit31).
_V2_LAYOUTS = {
    # DAVIS (jAER): bit 31 = readout type (1 = APS/IMU, skip), p = bit 11,
    # x = bits 12-21 (10 bits), y = bits 22-30 (9 bits).
    "davis": lambda a: (((a >> 12) & 0x3FF), ((a >> 22) & 0x1FF), ((a >> 11) & 0x1)),
    # DVS128: same address layout as AEDAT 1.0, in the low 16 bits.
    "dvs128": lambda a: (((a >> 1) & 0x7F), ((a >> 8) & 0x7F), (a & 0x1)),
}

# ---------------------------------------------------------------------------#
# Minimal FlatBuffer field access (AEDAT 4.0). The two schemas involved are
# tiny and fixed, so the offsets are walked by hand instead of depending on a
# flatbuffers runtime (cf. the evlib reference reader).
# ---------------------------------------------------------------------------#
def _u16(b: bytes, o: int) -> int:
    return int(struct.unpack_from("<H", b, o)[0])

def _i32(b: bytes, o: int) -> int:
    return int(struct.unpack_from("<i", b, o)[0])

def _u32(b: bytes, o: int) -> int:
    return int(struct.unpack_from("<I", b, o)[0])

def _i64(b: bytes, o: int) -> int:
    return int(struct.unpack_from("<q", b, o)[0])

def _fb_field_pos(buf: bytes, table: int, index: int) -> int | None:
    """Absolute position of field ``index`` of the table at ``table``, or None
    if the field is absent from the vtable."""
    vtable = table - _i32(buf, table)
    vtable_size = _u16(buf, vtable)
    slot = vtable + 4 + 2 * index
    if slot + 2 > vtable + vtable_size:
        return None
    voffset = _u16(buf, slot)
    return table + voffset if voffset else None

def _parse_io_header(buf: bytes) -> tuple[int, int, str]:
    """Parse the AEDAT4 ``IOHeader`` FlatBuffer.

    Returns ``(compression, data_table_position, info_node_xml)``.
    Fields: 0 = compression (i32), 1 = dataTablePosition (i64), 2 = infoNode.
    """
    table = _u32(buf, 0)
    compression = 0
    data_table_position = -1
    info_node = ""

    pos = _fb_field_pos(buf, table, 0)
    if pos is not None:
        compression = _i32(buf, pos)
    pos = _fb_field_pos(buf, table, 1)
    if pos is not None:
        data_table_position = _i64(buf, pos)
    pos = _fb_field_pos(buf, table, 2)
    if pos is not None:
        str_pos = pos + _u32(buf, pos)
        str_len = _u32(buf, str_pos)
        info_node = bytes(buf[str_pos + 4:str_pos + 4 + str_len]).decode("utf-8", "replace")
    return compression, data_table_position, info_node

def _parse_event_packet(body: bytes) -> np.ndarray | None:
    """Extract the event structs from a size-prefixed ``EventPacket`` FlatBuffer.

    Returns a ``_V4_EVENT_DTYPE`` view, or ``None`` when the body is not an
    ``EVTS`` packet (frame/IMU/trigger streams).
    """
    if len(body) < 12 or body[8:12] != b"EVTS":
        return None
    root = 4 + _u32(body, 4)
    pos = _fb_field_pos(body, root, 0)  # single field: `elements` vector
    if pos is None:
        return _np_empty_v4()
    vector = pos + _u32(body, pos)
    count = _u32(body, vector)
    start = vector + 4
    if start + count * _V4_EVENT_DTYPE.itemsize > len(body):
        raise ValueError("truncated AEDAT4 EventPacket")
    return np.frombuffer(body, dtype=_V4_EVENT_DTYPE, count=count, offset=start)

def _np_empty_v4() -> np.ndarray:
    return np.empty(0, dtype=_V4_EVENT_DTYPE)

def _parse_streams_xml(xml: str) -> tuple[set[int], int | None, int | None]:
    """Extract event-stream ids and the sensor geometry from the ``IOHeader``
    infoNode XML (a DV config tree).

    Returns ``(event_stream_ids, width, height)``. Parsing is best-effort:
    on any failure the id set is empty and the caller falls back to
    identifying event packets by their FlatBuffer identifier.
    """
    ids: set[int] = set()
    width: int | None = None
    height: int | None = None
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        for node in root.iter("node"):
            name = node.get("name", "")
            if not name.lstrip("-").isdigit():
                continue
            type_id = None
            size_x = size_y = None
            for attr in node.iter("attr"):
                key = attr.get("key")
                if key == "typeIdentifier":
                    type_id = (attr.text or "").strip()
                elif key == "sizeX":
                    size_x = int((attr.text or "0").strip())
                elif key == "sizeY":
                    size_y = int((attr.text or "0").strip())
            if type_id == "EVTS":
                ids.add(int(name))
                if width is None and size_x:
                    width, height = size_x, size_y
    except Exception:
        return set(), None, None
    return ids, width, height

def _decompress_lz4(body: bytes) -> bytes:
    try:
        import lz4.frame
    except ImportError as exc:
        raise ImportError(
            "This AEDAT4 file uses LZ4-compressed packets: install "
            "`evutils[aedat]` (or `pip install lz4`)."
        ) from exc
    return bytes(lz4.frame.decompress(body))

def _decompress_zstd(body: bytes) -> bytes:
    try:
        from compression import zstd  # Python >= 3.14
        return bytes(zstd.decompress(body))
    except ImportError:
        pass
    try:
        import zstandard
    except ImportError as exc:
        raise ImportError(
            "This AEDAT4 file uses Zstd-compressed packets: install "
            "`evutils[aedat]` (or `pip install zstandard`)."
        ) from exc
    return bytes(zstandard.ZstdDecompressor().decompress(body))

#: DV CompressionType enum -> decompressor. LZ4_HIGH/ZSTD_HIGH share decoders.
_DECOMPRESSORS: dict[int, Callable[[bytes], bytes]] = {
    0: lambda b: b,           # NONE
    1: _decompress_lz4,       # LZ4
    2: _decompress_lz4,       # LZ4_HIGH
    3: _decompress_zstd,      # ZSTD
    4: _decompress_zstd,      # ZSTD_HIGH
}

class EventDecoder_Aedat(EventDecoder):
    SUPPORTS_EXT_TRIGGERS = True

    #: init() slurps the whole payload into memory (or mmaps it).
    _buffers_in_memory = True
    """Decode AEDAT 1.0 / 2.0 / 3.1 / 4.0 files into ``EventArray`` chunks.

    The version is detected from the ``#!AER-DATx.y`` header line (a file
    with a bare ``#`` header, or none at all, is treated as AEDAT 1.0, per
    the jAER convention).

    Parameters
    ----------
    source
        Byte source to read from.
    chunk_size
        Maximum number of events produced per :meth:`read_chunk` call
        (AEDAT 3.1/4.0 packets are never split, so a chunk can exceed this
        by at most one packet's worth).
    layout : {"davis", "dvs128"}, default "davis"
        AEDAT 2.0 address layout (the 2.0 container does not name the
        camera). Ignored for the other versions.

    """

    def __init__(self, source: ByteSource, chunk_size: int = 1_000_000,
                 layout: str = "davis"):
        super().__init__(source, chunk_size)
        if layout not in _V2_LAYOUTS:
            raise ValueError(f"layout must be one of {sorted(_V2_LAYOUTS)}, got {layout!r}")
        self._layout = layout

        self._buf: bytes | bytearray | None = None
        self._version: int = 0          # 1, 2, 3 or 4 (x10: 31 -> 3, 40 -> 4)
        self._payload_off: int = 0      # first byte after the ASCII header
        self._cursor: int = 0           # current byte offset into _buf

        # v1/v2 32-bit timestamp unwrap state.
        self._ts_wraps: int = 0
        self._last_raw_ts: int = -1

        # v4 packet-region metadata.
        self._v4_compression: int = 0
        self._v4_region_end: int = 0
        self._v4_stream_ids: set[int] = set()

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    def _parse_header(self) -> None:
        """Detect the version and consume the ASCII header."""
        buf = self._buf
        head = bytes(buf[:16])

        if head.startswith(b"#!AER-DAT4.0"):
            self._version = 4
            self._parse_v4_header()
            return
        if head.startswith(b"#!AER-DAT3"):
            self._version = 3
        elif head.startswith(b"#!AER-DAT2.0"):
            self._version = 2
        else:
            # "#!AER-DAT1.0", a bare "#" header, or no header at all (raw
            # DVS128 dumps): all AEDAT 1.0 per the jAER convention.
            self._version = 1

        # Consume '#'-prefixed header lines; harvest sizeX/sizeY when present.
        n = len(buf)
        off = 0
        header_lines = []
        while off < n and buf[off] == 0x23:  # '#'
            window = bytes(buf[off:off + 8192])
            rel = window.find(b"\n")
            if rel < 0:
                off = n
                break
            line = window[:rel]
            header_lines.append(line)
            off += rel + 1
            if self._version == 3 and line.strip() == b"#!END-HEADER":
                break
        self._payload_off = off

        header_text = b"\n".join(header_lines).decode("ascii", "replace")
        for key, attr in (("sizeX", "_width"), ("sizeY", "_height")):
            m = re.search(rf"{key}\s*[=:]?\s*(\d+)", header_text)
            if m:
                setattr(self, attr, int(m.group(1)))

    def _parse_v4_header(self) -> None:
        """Parse the AEDAT4 version line + IOHeader FlatBuffer."""
        buf = self._buf
        # 14-byte version line, then a u32-size-prefixed IOHeader FlatBuffer.
        header_size = _u32(bytes(buf[14:18]), 0)
        io_start, io_end = 18, 18 + header_size
        if io_end > len(buf):
            raise ValueError("truncated AEDAT4 IOHeader")
        compression, data_table_pos, info_node = _parse_io_header(bytes(buf[io_start:io_end]))

        if compression not in _DECOMPRESSORS:
            raise ValueError(f"unknown AEDAT4 compression type {compression}")
        self._v4_compression = compression
        self._payload_off = io_end
        # Packets occupy the bytes between the IOHeader and the trailing
        # FileDataTable (when its position is known).
        if 0 <= data_table_pos <= len(buf):
            self._v4_region_end = int(data_table_pos)
        else:
            self._v4_region_end = len(buf)

        ids, width, height = _parse_streams_xml(info_node)
        self._v4_stream_ids = ids
        if width:
            self._width = width
        if height:
            self._height = height

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def init(self) -> None:
        """Read the header and position the cursor at the first record/packet."""
        if self._is_initialized:
            return

        if self._source.mappable():
            self._buf = self._source.buffer()
        else:
            self._buf = memoryview(self._source.read(-1))

        self._parse_header()
        self._cursor = self._payload_off
        self._is_initialized = True

    # ------------------------------------------------------------------ #
    # Per-version batch decoding
    # ------------------------------------------------------------------ #
    def _unwrap_ts(self, ts_raw: np.ndarray) -> np.ndarray:
        """Extend raw 32-bit µs timestamps to int64, accumulating wraps
        (a drop of more than 2^31 between consecutive timestamps counts as a
        wrap; smaller decreases are genuine jitter and pass through).
        """
        t = ts_raw.astype(np.int64)
        if len(t) == 0:
            return t
        prev = np.empty_like(t)
        prev[0] = self._last_raw_ts if self._last_raw_ts >= 0 else int(t[0])
        prev[1:] = t[:-1]
        wraps = self._ts_wraps + np.cumsum((prev - t) > (1 << 31))
        self._ts_wraps = int(wraps[-1])
        self._last_raw_ts = int(t[-1])
        return t + (wraps << np.int64(32))

    def _batch_v1_v2(self) -> tuple[EventArray, TriggerArray] | None:
        """Decode the next ``chunk_size`` fixed-size records (AEDAT 1.0/2.0)."""
        from ..types import TriggerArray
        dtype = _V1_DTYPE if self._version == 1 else _V2_DTYPE
        remaining = (len(self._buf) - self._cursor) // dtype.itemsize
        if remaining <= 0:
            return None
        count = min(self._chunk_size, remaining)
        rec = np.frombuffer(self._buf, dtype=dtype, count=count, offset=self._cursor)
        self._cursor += count * dtype.itemsize

        a = rec["a"]
        t = self._unwrap_ts(rec["t"])
        if self._version == 1:
            x = (a >> 1) & np.uint16(0x7F)
            y = (a >> 8) & np.uint16(0x7F)
            p = (a & np.uint16(0x1)).astype(np.uint8)
            events = EventArray(t, x.astype(np.uint16), y.astype(np.uint16), p)
            triggers = TriggerArray.empty()
        else:
            # Skip non-DVS words (APS samples / IMU, flagged by bit 31).
            dvs = (a >> np.uint32(31)) == 0
            if not dvs.all():
                a, t = a[dvs], t[dvs]
            
            # Bits 11-10: 00=OFF, 10=ON, 01/11=External Event
            subtype = (a >> np.uint32(10)) & np.uint32(0x3)
            is_trigger = (subtype == 1) | (subtype == 3)
            is_event = ~is_trigger

            a_ev, t_ev = a[is_event], t[is_event]
            x, y, p = _V2_LAYOUTS[self._layout](a_ev)
            events = EventArray(t_ev, x.astype(np.uint16), y.astype(np.uint16), p.astype(np.uint8))

            a_tr, t_tr = a[is_trigger], t[is_trigger]
            tr_p = ((a_tr >> np.uint32(11)) & np.uint32(0x1)).astype(np.uint8)
            triggers = TriggerArray(t_tr, tr_p, np.zeros_like(tr_p))

        return events, triggers

    def _batch_v3(self) -> tuple[EventArray, TriggerArray] | None:
        """Decode the next AEDAT 3.1 packet holding polarity or trigger events."""
        from ..types import TriggerArray
        buf = self._buf
        n = len(buf)
        while self._cursor + _V3_HEADER.size <= n:
            (ev_type, _source, ev_size, ts_offset, ts_overflow,
             capacity, number, _valid) = _V3_HEADER.unpack_from(buf, self._cursor)
            body_start = self._cursor + _V3_HEADER.size
            body_end = body_start + capacity * ev_size
            if ev_size <= 0 or body_end > n:
                self._cursor = n
                return None
            self._cursor = body_end

            if number == 0:
                continue

            if ev_type == _V3_POLARITY_EVENT:
                if ev_size != _V3_EVENT_DTYPE.itemsize or ts_offset != 4:
                    raise ValueError(f"unsupported AEDAT3 polarity event layout (eventSize={ev_size})")
                rec = np.frombuffer(buf, dtype=_V3_EVENT_DTYPE, count=number, offset=body_start)
                d = rec["d"]
                valid = (d & np.uint32(0x1)) != 0
                if not valid.all():
                    rec, d = rec[valid], d[valid]
                    if len(rec) == 0:
                        continue
                t = rec["t"].astype(np.int64) + (np.int64(ts_overflow) << np.int64(31))
                x = ((d >> np.uint32(17)) & np.uint32(0x7FFF)).astype(np.uint16)
                y = ((d >> np.uint32(2)) & np.uint32(0x7FFF)).astype(np.uint16)
                p = ((d >> np.uint32(1)) & np.uint32(0x1)).astype(np.uint8)
                return EventArray(t, x, y, p), TriggerArray.empty()

            elif ev_type == 0:  # SPECIAL_EVENT
                rec = np.frombuffer(buf, dtype=_V3_EVENT_DTYPE, count=number, offset=body_start)
                d = rec["d"]
                valid = (d & np.uint32(0x1)) != 0
                if not valid.all():
                    rec, d = rec[valid], d[valid]
                type_id = (d >> np.uint32(1)) & np.uint32(0x7F)
                is_ext = (type_id >= 2) & (type_id <= 13)
                if is_ext.any():
                    tr_rec, tr_d, tr_type = rec[is_ext], d[is_ext], type_id[is_ext]
                    tr_t = tr_rec["t"].astype(np.int64) + (np.int64(ts_overflow) << np.int64(31))
                    tr_p = (tr_type % 2 == 0).astype(np.uint8) # Even types are rising (1), odd are falling (0)
                    return _EMPTY_EVENTS, TriggerArray(tr_t, tr_p, np.zeros_like(tr_p))
        self._cursor = n
        return None

    def _batch_v4(self) -> tuple[EventArray, TriggerArray] | None:
        """Decode the next AEDAT4 packet carrying events or triggers."""
        from ..types import TriggerArray
        buf = self._buf
        end = self._v4_region_end
        decompress = _DECOMPRESSORS[self._v4_compression]
        while self._cursor + 8 <= end:
            stream_id = _i32(buf, self._cursor)
            size = _i32(buf, self._cursor + 4)
            body_start = self._cursor + 8
            body_end = body_start + size
            if size < 0 or body_end > end:
                self._cursor = end
                return None
            self._cursor = body_end

            # Unconditionally decompress if we're reading triggers (don't skip non-EVTS streams just yet)
            body = decompress(bytes(buf[body_start:body_end]))
            
            if len(body) >= 12 and body[8:12] == b"EVTS":
                rec = _parse_event_packet(body)
                if rec is not None and len(rec) > 0:
                    return EventArray(
                        rec["t"],
                        rec["x"].astype(np.uint16),
                        rec["y"].astype(np.uint16),
                        rec["p"].astype(np.uint8),
                    ), TriggerArray.empty()
            elif len(body) >= 12 and body[8:12] == b"TRIG":
                # Trigger packet parsing:
                # Trigger struct: int64 t, int8 type, 7 pad bytes = 16 bytes.
                # Types: 0=TIMESTAMP_RESET, 1=EXTERNAL_INPUT_RISING_EDGE, 2=EXTERNAL_INPUT_FALLING_EDGE, 3=EXTERNAL_INPUT_PULSE, etc.
                root = 4 + _u32(body, 4)
                pos = _fb_field_pos(body, root, 0)
                if pos is not None:
                    vector = pos + _u32(body, pos)
                    count = _u32(body, vector)
                    start = vector + 4
                    if start + count * 16 <= len(body):
                        tr_rec = np.frombuffer(body, dtype=np.dtype([("t", "<i8"), ("type", "i1"), ("pad", "V7")]), count=count, offset=start)
                        t = tr_rec["t"]
                        type_id = tr_rec["type"]
                        # Rising edge is typically 1 (odd), falling edge is 2 (even).
                        # Let's map it simply:
                        tr_p = (type_id % 2 != 0).astype(np.uint8)
                        return _EMPTY_EVENTS, TriggerArray(t, tr_p, np.zeros_like(tr_p))
        self._cursor = end
        return None

    # ------------------------------------------------------------------ #
    # EventDecoder interface
    # ------------------------------------------------------------------ #
    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> 'EventArray | tuple[EventArray, TriggerArray]':
        from ..types import TriggerArray
        if not self._is_initialized:
            self.init()

        if self._version in (1, 2):
            batch = self._batch_v1_v2()
        elif self._version == 3:
            batch = self._batch_v3()
        else:
            batch = self._batch_v4()

        if batch is None:
            self._eof = True
            if self.read_external_triggers:
                return _EMPTY_EVENTS, TriggerArray.empty()
            return _EMPTY_EVENTS
        
        events, triggers = batch
        if self.read_external_triggers:
            return events, triggers
        return events

    def reset(self) -> None:
        """Reset the reader to the first event."""
        self._cursor = self._payload_off
        self._ts_wraps = 0
        self._last_raw_ts = -1
        self._eof = False

    def tell(self) -> int:
        """Get the current byte offset.

        Returns
        -------
        int
            Current byte offset.

        """
        return self._cursor

    def close(self) -> None:
        """Release the buffer view so the source can be closed."""
        self._buf = None

def _fb_io_header(compression: int, data_table_pos: int, info_node: bytes) -> bytes:
    """Hand-built IOHeader FlatBuffer (not size-prefixed)."""
    import struct
    buf = bytearray()
    buf += struct.pack("<I", 16)                       # root table offset
    buf += struct.pack("<HHHHH", 10, 20, 4, 8, 16)     # vtable @4: 3 fields
    buf += b"\x00\x00"                                 # pad, table at 16
    buf += struct.pack("<i", 12)                       # soffset: table(16) - vtable(4)
    buf += struct.pack("<i", compression)              # field 0 @20 (voffset 4)
    buf += struct.pack("<q", data_table_pos)           # field 1 @24 (voffset 8)
    buf += struct.pack("<I", 36 - 32)                  # field 2 @32: string @36
    buf += struct.pack("<I", len(info_node)) + info_node + b"\x00"
    return bytes(buf)

def _fb_event_packet(events: 'EventArray') -> bytes:
    """Hand-built size-prefixed EventPacket FlatBuffer (identifier EVTS)."""
    import struct
    n = len(events)
    # AEDAT 4.0 event struct layout: int64 t, int16 x, int16 y, uint8 p, 3 pad bytes
    rec = np.zeros(n, dtype=_V4_EVENT_DTYPE)
    rec["t"] = events["t"]
    rec["x"] = events["x"]
    rec["y"] = events["y"]
    rec["p"] = events["p"]
    events_bytes = rec.tobytes()
    
    buf = bytearray()
    buf += struct.pack("<I", 0)                    # size prefix (patched below)
    buf += struct.pack("<I", 16)                   # root table offset, relative to pos 4
    buf += b"EVTS"
    buf += struct.pack("<HHH", 6, 8, 4)            # vtable: size 6, table size 8, field0 @4
    buf += b"\x00\x00"                             # pad to 4-aligned table at 20
    buf += struct.pack("<i", 8)                    # soffset: table(20) - vtable(12)
    buf += struct.pack("<I", 4)                    # field0: vector offset rel to pos 24
    buf += struct.pack("<I", n)                    # vector length
    buf += events_bytes
    struct.pack_into("<I", buf, 0, len(buf) - 4)
    return bytes(buf)

def _fb_trigger_packet(triggers: 'TriggerArray') -> bytes:
    """Hand-built size-prefixed TriggerPacket FlatBuffer (identifier TRIG)."""
    import struct
    n = len(triggers)
    _V4_TRIGGER_DTYPE = np.dtype({
        "names": ["t", "type"],
        "formats": ["<i8", "i1"],
        "offsets": [0, 8],
        "itemsize": 16,
    })
    rec = np.zeros(n, dtype=_V4_TRIGGER_DTYPE)
    rec["t"] = triggers["t"]
    # 1=EXTERNAL_INPUT_RISING_EDGE, 2=EXTERNAL_INPUT_FALLING_EDGE
    rec["type"] = np.where(triggers["p"], 1, 2)
    triggers_bytes = rec.tobytes()
    
    buf = bytearray()
    buf += struct.pack("<I", 0)                    # size prefix (patched below)
    buf += struct.pack("<I", 16)                   # root table offset, relative to pos 4
    buf += b"TRIG"
    buf += struct.pack("<HHH", 6, 8, 4)            # vtable: size 6, table size 8, field0 @4
    buf += b"\x00\x00"                             # pad to 4-aligned table at 20
    buf += struct.pack("<i", 8)                    # soffset: table(20) - vtable(12)
    buf += struct.pack("<I", 4)                    # field0: vector offset rel to pos 24
    buf += struct.pack("<I", n)                    # vector length
    buf += triggers_bytes
    struct.pack_into("<I", buf, 0, len(buf) - 4)
    return bytes(buf)

class EventEncoder_Aedat(EventEncoder):
    """Encoder for AEDAT 4.0 files."""
    
    SUPPORTS_WRITE_TRIGGERS = True

    def __init__(self, writable, width:int = 1280, height:int = 720, dt=None, compression: int = 0, **kwargs):
        super().__init__(writable, width, height, dt)
        self._compression = compression  # 0=NONE, 1=LZ4, 2=LZ4_HIGH, 3=ZSTD, 4=ZSTD_HIGH

    def init(self) -> None:
        if self._is_initialized:
            return
            
        import struct
        
        # Version line
        self._fd.write(b"#!AER-DAT4.0\r\n")
        
        # IOHeader XML Config
        info_xml = f'<?xml version="1.0" encoding="UTF-8"?><node name="info"><node name="0"><attr key="typeIdentifier" type="string">EVTS</attr><attr key="sizeX" type="int">{self._width}</attr><attr key="sizeY" type="int">{self._height}</attr></node><node name="1"><attr key="typeIdentifier" type="string">TRIG</attr></node></node>'.encode("utf-8")
        
        io_header = _fb_io_header(self._compression, -1, info_xml)
        self._fd.write(struct.pack("<I", len(io_header)) + io_header)
        
        self._is_initialized = True

    def write(self, events, triggers = None) -> int:
        if not self._is_initialized:
            self.init()
            
        if len(events) == 0 and (triggers is None or len(triggers) == 0):
            return 0
            
        import struct
        
        if len(events) > 0:
            body = _fb_event_packet(events)
            if self._compression in (1, 2):
                import lz4.frame
                body = lz4.frame.compress(body)
            elif self._compression in (3, 4):
                import zstandard
                ctx = zstandard.ZstdCompressor(level=3 if self._compression == 3 else 10)
                body = ctx.compress(body)
                
            # Write Packet header: StreamID (0), Size, Body
            self._fd.write(struct.pack("<iI", 0, len(body)) + body)
            self._n_written_events += len(events)
        
        if triggers is not None and len(triggers) > 0:
            tr_body = _fb_trigger_packet(triggers)
            if self._compression in (1, 2):
                import lz4.frame
                tr_body = lz4.frame.compress(tr_body)
            elif self._compression in (3, 4):
                import zstandard
                ctx = zstandard.ZstdCompressor(level=3 if self._compression == 3 else 10)
                tr_body = ctx.compress(tr_body)
            self._fd.write(struct.pack("<iI", 1, len(tr_body)) + tr_body)
        
        return len(events)
