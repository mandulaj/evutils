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

    def _batch_v1_v2(self) -> EventArray | None:
        """Decode the next ``chunk_size`` fixed-size records (AEDAT 1.0/2.0)."""
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
        else:
            # Skip non-DVS words (APS samples / IMU, flagged by bit 31).
            dvs = (a >> np.uint32(31)) == 0
            if not dvs.all():
                a, t = a[dvs], t[dvs]
            x, y, p = _V2_LAYOUTS[self._layout](a)
            p = p.astype(np.uint8)
        return EventArray(t, x.astype(np.uint16), y.astype(np.uint16), p)

    def _batch_v3(self) -> EventArray | None:
        """Decode the next AEDAT 3.1 packet holding polarity events."""
        buf = self._buf
        n = len(buf)
        while self._cursor + _V3_HEADER.size <= n:
            (ev_type, _source, ev_size, ts_offset, ts_overflow,
             capacity, number, _valid) = _V3_HEADER.unpack_from(buf, self._cursor)
            body_start = self._cursor + _V3_HEADER.size
            body_end = body_start + capacity * ev_size
            if ev_size <= 0 or body_end > n:
                # Corrupt/truncated trailing packet: stop cleanly.
                self._cursor = n
                return None
            self._cursor = body_end

            if ev_type != _V3_POLARITY_EVENT or number == 0:
                continue  # frame / IMU / trigger packet: skip

            if ev_size != _V3_EVENT_DTYPE.itemsize or ts_offset != 4:
                raise ValueError(
                    f"unsupported AEDAT3 polarity event layout "
                    f"(eventSize={ev_size}, tsOffset={ts_offset})"
                )
            rec = np.frombuffer(buf, dtype=_V3_EVENT_DTYPE, count=number, offset=body_start)
            d = rec["d"]
            valid = (d & np.uint32(0x1)) != 0
            if not valid.all():
                rec, d = rec[valid], d[valid]
                if len(rec) == 0:
                    continue
            # 31-bit in-packet timestamp + 64-bit overflow counter.
            t = rec["t"].astype(np.int64) + (np.int64(ts_overflow) << np.int64(31))
            x = ((d >> np.uint32(17)) & np.uint32(0x7FFF)).astype(np.uint16)
            y = ((d >> np.uint32(2)) & np.uint32(0x7FFF)).astype(np.uint16)
            p = ((d >> np.uint32(1)) & np.uint32(0x1)).astype(np.uint8)
            return EventArray(t, x, y, p)
        self._cursor = n
        return None

    def _batch_v4(self) -> EventArray | None:
        """Decode the next AEDAT4 packet carrying events."""
        buf = self._buf
        end = self._v4_region_end
        decompress = _DECOMPRESSORS[self._v4_compression]
        while self._cursor + 8 <= end:
            stream_id = _i32(buf, self._cursor)
            size = _i32(buf, self._cursor + 4)
            body_start = self._cursor + 8
            body_end = body_start + size
            if size < 0 or body_end > end:
                # Truncated trailing packet (or we ran into the data table).
                self._cursor = end
                return None
            self._cursor = body_end

            # When the header named the event streams, filter cheaply by id;
            # otherwise decompress and check the FlatBuffer identifier.
            if self._v4_stream_ids and stream_id not in self._v4_stream_ids:
                continue
            body = decompress(bytes(buf[body_start:body_end]))
            rec = _parse_event_packet(body)
            if rec is None or len(rec) == 0:
                continue
            return EventArray(
                rec["t"],
                rec["x"].astype(np.uint16),
                rec["y"].astype(np.uint16),
                rec["p"].astype(np.uint8),
            )
        self._cursor = end
        return None

    # ------------------------------------------------------------------ #
    # EventDecoder interface
    # ------------------------------------------------------------------ #
    def read_chunk(self, delta_t_hint: int | None = None,
                   n_events_hint: int | None = None) -> EventArray:
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
            return _EMPTY_EVENTS
        return batch

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

class EventEncoder_Aedat(EventEncoder):
    """An encoder for writing events to AEDAT files.

    .. note::
        Not implemented yet -- constructing this class raises
        :class:`NotImplementedError`. (Reading AEDAT 1.0--4.0 is supported
        via :class:`EventDecoder_Aedat`.)

    Parameters
    ----------
    writable : io.BufferedIOBase
        Destination for writing events.
    **kwargs
        Additional encoder arguments (``width``, ``height``, ``dt``, ...).

    Raises
    ------
    NotImplementedError
        Always, until the format is implemented.

    """

    _NOT_IMPLEMENTED = (
        "Writing AEDAT files is not implemented yet. "
        "Write a supported format instead (RAW/EVT, DAT, AER, HDF5, NPZ, CSV)."
    )

    def __init__(self, writable: "io.BufferedIOBase", **kwargs):
        raise NotImplementedError(self._NOT_IMPLEMENTED)

    def init(self) -> None:
        """Initialize the file for writing."""
        raise NotImplementedError(self._NOT_IMPLEMENTED)

    def write(self, events: 'np.ndarray | EventArray', triggers: 'np.ndarray | TriggerArray | None' = None) -> int:
        """Write a chunk of events."""
        raise NotImplementedError(self._NOT_IMPLEMENTED)
