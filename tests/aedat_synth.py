"""Spec-accurate AEDAT 1.0/2.0/3.1/4.0 file synthesizers.

Shared between the correctness tests (tests/io/test_aedat.py) and the
benchmarks. The builders follow the iniVation file-format documentation
(v1-v3, big-endian jAER layouts) and the DV framework layout (v4,
hand-built FlatBuffers), so decoders are checked against the spec rather
than against themselves.
"""
import struct

import numpy as np
from typing import Any

#: AEDAT 4.0 event struct layout (16 bytes: int64 t, int16 x, int16 y, uint8 p).
_V4_EVENT_DTYPE = np.dtype({
    "names": ["t", "x", "y", "p"],
    "formats": ["<i8", "<i2", "<i2", "u1"],
    "offsets": [0, 8, 10, 12],
    "itemsize": 16,
})

# --------------------------------------------------------------------------- #
# Synthesis helpers
# --------------------------------------------------------------------------- #
def make_aedat1(t: Any, x: Any, y: Any, p: Any, header: bool=True) -> bytes:
    """AEDAT 1.0: '#' header + big-endian (uint16 addr, uint32 ts) records.
    DVS128 layout: p = bit 0, x = bits 1-7, y = bits 8-14."""
    out = bytearray()
    if header:
        out += b"#!AER-DAT1.0\r\n# This is a raw AE data file\r\n# sizeX 128\r\n# sizeY 128\r\n"
    for ti, xi, yi, pi in zip(t, x, y, p):
        addr = (int(yi) << 8) | (int(xi) << 1) | int(pi)
        out += struct.pack(">HI", addr, int(ti))
    return bytes(out)


def make_aedat2(t: Any, x: Any, y: Any, p: Any, aps_every: Any=None) -> bytes:
    """AEDAT 2.0: '#' header + big-endian (uint32 addr, uint32 ts) records.
    DAVIS layout: p = bit 11, x = bits 12-21, y = bits 22-30; bit 31 = APS."""
    out = bytearray(b"#!AER-DAT2.0\r\n# sizeX 240\r\n# sizeY 180\r\n")
    for i, (ti, xi, yi, pi) in enumerate(zip(t, x, y, p)):
        addr = (int(yi) << 22) | (int(xi) << 12) | (int(pi) << 11)
        out += struct.pack(">II", addr, int(ti))
        if aps_every and (i + 1) % aps_every == 0:
            # Interleave an APS word (bit 31 set) that must be skipped.
            out += struct.pack(">II", 0x8000_0000 | addr, int(ti))
    return bytes(out)


def _aedat3_packet(ev_type: int, records: list[bytes]) -> bytes:
    """One AEDAT 3.1 packet: 28-byte LE header + raw records."""
    body = b"".join(records)
    ev_size = len(body) // max(len(records), 1) if records else 8
    return struct.pack(
        "<hhiiiiii", ev_type, 0, ev_size, 4,
        getattr(_aedat3_packet, "ts_overflow", 0), len(records), len(records), len(records),
    ) + body


setattr(_aedat3_packet, "ts_overflow", 0)


def make_aedat3(t: Any, x: Any, y: Any, p: Any, events_per_packet: int=4, ts_overflow: int=0) -> bytes:
    """AEDAT 3.1: header through '#!END-HEADER' + polarity-event packets.
    Event: uint32 data (valid bit 0, p bit 1, y bits 2-16, x bits 17-31)
    + uint32 timestamp, little-endian."""
    setattr(_aedat3_packet, "ts_overflow", ts_overflow)
    out = bytearray(
        b"#!AER-DAT3.1\r\n#Format: RAW\r\n#Source 1: Test sizeX 346 sizeY 260\r\n#!END-HEADER\r\n"
    )
    recs = [
        struct.pack("<II", (int(xi) << 17) | (int(yi) << 2) | (int(pi) << 1) | 1, int(ti))
        for ti, xi, yi, pi in zip(t, x, y, p)
    ]
    # An IMU packet (type 3) that must be skipped.
    out += _aedat3_packet(3, [b"\x00" * 8] * 2)
    for i in range(0, len(recs), events_per_packet):
        out += _aedat3_packet(1, recs[i:i + events_per_packet])
    return bytes(out)


def _fb_event_packet(t: Any, x: Any, y: Any, p: Any) -> bytes:
    """Hand-built size-prefixed EventPacket FlatBuffer (identifier EVTS)."""
    n = len(t)
    rec = np.zeros(n, dtype=_V4_EVENT_DTYPE)
    rec["t"] = t; rec["x"] = x; rec["y"] = y; rec["p"] = p
    events = rec.tobytes()
    # Layout (offsets relative to buffer start):
    #  0: u32 size prefix          12: vtable [6, 8, 4]     20: table
    #  4: u32 root offset (rel 4)  18: padding              28: vector
    #  8: 'EVTS'
    buf = bytearray()
    buf += struct.pack("<I", 0)                    # size prefix (patched below)
    buf += struct.pack("<I", 16)                   # root table offset, relative to pos 4
    buf += b"EVTS"
    buf += struct.pack("<HHH", 6, 8, 4)            # vtable: size 6, table size 8, field0 @4
    buf += b"\x00\x00"                             # pad to 4-aligned table at 20
    buf += struct.pack("<i", 8)                    # soffset: table(20) - vtable(12)
    buf += struct.pack("<I", 4)                    # field0: vector offset rel to pos 24
    buf += struct.pack("<I", n)                    # vector length
    buf += events
    struct.pack_into("<I", buf, 0, len(buf) - 4)
    return bytes(buf)


def _fb_io_header(compression: int, data_table_pos: int, info_node: bytes) -> bytes:
    """Hand-built IOHeader FlatBuffer (not size-prefixed)."""
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


_V4_INFO_XML = (
    b'<dv version="2.0"><node name="outInfo" path="/outInfo/">'
    b'<node name="0" path="/outInfo/0/">'
    b'<attr key="typeIdentifier" type="string">EVTS</attr>'
    b'<node name="info" path="/outInfo/0/info/">'
    b'<attr key="sizeX" type="int">640</attr>'
    b'<attr key="sizeY" type="int">480</attr>'
    b"</node></node></node></dv>"
)


def make_aedat4(t: Any, x: Any, y: Any, p: Any, events_per_packet: int=5, compression: int=0, info: bytes=_V4_INFO_XML) -> bytes:
    """AEDAT 4.0: version line + IOHeader FlatBuffer + (StreamID, Size)-framed
    EventPacket FlatBuffers."""
    io_header = _fb_io_header(compression, -1, info)
    out = bytearray(b"#!AER-DAT4.0\r\n")
    out += struct.pack("<I", len(io_header)) + io_header
    for i in range(0, len(t), events_per_packet):
        s = slice(i, i + events_per_packet)
        body = _fb_event_packet(t[s], x[s], y[s], p[s])
        if compression in (1, 2):
            import lz4.frame
            body = lz4.frame.compress(body)
        out += struct.pack("<ii", 0, len(body)) + body
    return bytes(out)


