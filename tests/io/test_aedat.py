"""AEDAT 1.0 / 2.0 / 3.1 / 4.0 decoder tests against synthesized files.

The writers below build byte-exact files following the iniVation file-format
documentation (v1-v3, big-endian jAER layouts) and the DV framework layout
(v4, FlatBuffers), so the decoder is checked against the spec rather than
against itself.
"""
import struct

import numpy as np
import pytest

from evutils.io import EventReader


# --------------------------------------------------------------------------- #
# Synthesis helpers
# --------------------------------------------------------------------------- #
def make_aedat1(t, x, y, p, header=True):
    """AEDAT 1.0: '#' header + big-endian (uint16 addr, uint32 ts) records.
    DVS128 layout: p = bit 0, x = bits 1-7, y = bits 8-14."""
    out = bytearray()
    if header:
        out += b"#!AER-DAT1.0\r\n# This is a raw AE data file\r\n# sizeX 128\r\n# sizeY 128\r\n"
    for ti, xi, yi, pi in zip(t, x, y, p):
        addr = (int(yi) << 8) | (int(xi) << 1) | int(pi)
        out += struct.pack(">HI", addr, int(ti))
    return bytes(out)


def make_aedat2(t, x, y, p, aps_every=None):
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


def _aedat3_packet(ev_type, records):
    """One AEDAT 3.1 packet: 28-byte LE header + raw records."""
    body = b"".join(records)
    ev_size = len(body) // max(len(records), 1) if records else 8
    return struct.pack(
        "<hhiiiiii", ev_type, 0, ev_size, 4,
        _aedat3_packet.ts_overflow, len(records), len(records), len(records),
    ) + body


_aedat3_packet.ts_overflow = 0


def make_aedat3(t, x, y, p, events_per_packet=4, ts_overflow=0):
    """AEDAT 3.1: header through '#!END-HEADER' + polarity-event packets.
    Event: uint32 data (valid bit 0, p bit 1, y bits 2-16, x bits 17-31)
    + uint32 timestamp, little-endian."""
    _aedat3_packet.ts_overflow = ts_overflow
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


def _fb_event_packet(t, x, y, p):
    """Hand-built size-prefixed EventPacket FlatBuffer (identifier EVTS)."""
    n = len(t)
    events = bytearray()
    for ti, xi, yi, pi in zip(t, x, y, p):
        events += struct.pack("<qhhB3x", int(ti), int(xi), int(yi), int(pi))
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


def _fb_io_header(compression, data_table_pos, info_node: bytes):
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


def make_aedat4(t, x, y, p, events_per_packet=5, compression=0, info=_V4_INFO_XML):
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


def random_events(n, width, height, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 1_000_000, n))
    x = rng.integers(0, width, n)
    y = rng.integers(0, height, n)
    p = rng.integers(0, 2, n)
    return t, x, y, p


def check(out, t, x, y, p):
    assert len(out) == len(t)
    assert np.array_equal(out.t, t)
    assert np.array_equal(out.x, x)
    assert np.array_equal(out.y, y)
    assert np.array_equal(out.p, p)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_aedat1(tmp_path):
    t, x, y, p = random_events(500, 128, 128)
    f = tmp_path / "v1.aedat"
    f.write_bytes(make_aedat1(t, x, y, p))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)
        assert r.shape() == (128, 128)


def test_aedat1_headerless(tmp_path):
    """A bare-'#' or headerless file defaults to AEDAT 1.0 (jAER convention)."""
    t, x, y, p = random_events(100, 128, 128, seed=1)
    f = tmp_path / "v1_bare.aedat"
    f.write_bytes(make_aedat1(t, x, y, p, header=False))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)


def test_aedat2_davis_skips_aps(tmp_path):
    t, x, y, p = random_events(500, 240, 180, seed=2)
    f = tmp_path / "v2.aedat"
    f.write_bytes(make_aedat2(t, x, y, p, aps_every=10))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)
        assert r.shape() == (240, 180)


def test_aedat2_timestamp_wrap(tmp_path):
    """32-bit µs timestamps wrap; the decoder must extend them to 64 bits."""
    t32 = np.array([2**32 - 20, 2**32 - 10, 5, 15], dtype=np.uint64)
    x = np.array([1, 2, 3, 4]); y = np.array([5, 6, 7, 8]); p = np.array([0, 1, 0, 1])
    f = tmp_path / "wrap.aedat"
    f.write_bytes(make_aedat2(t32 & 0xFFFFFFFF, x, y, p))
    with EventReader(f) as r:
        out = r.read_all()
    expected = np.array([2**32 - 20, 2**32 - 10, 2**32 + 5, 2**32 + 15], dtype=np.int64)
    assert np.array_equal(out.t, expected)


def test_aedat3(tmp_path):
    t, x, y, p = random_events(500, 346, 260, seed=3)
    f = tmp_path / "v3.aedat"
    f.write_bytes(make_aedat3(t, x, y, p))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)


def test_aedat3_ts_overflow(tmp_path):
    """The packet header's TS-overflow counter extends the 31-bit timestamps."""
    t, x, y, p = random_events(8, 346, 260, seed=4)
    f = tmp_path / "v3_ovf.aedat"
    f.write_bytes(make_aedat3(t, x, y, p, ts_overflow=2))
    with EventReader(f) as r:
        out = r.read_all()
    assert np.array_equal(out.t, t.astype(np.int64) + (2 << 31))


def test_aedat4_uncompressed(tmp_path):
    t, x, y, p = random_events(500, 640, 480, seed=5)
    t = t + 1_663_249_605_734_020  # DV timestamps are absolute epoch µs
    f = tmp_path / "v4.aedat4"
    f.write_bytes(make_aedat4(t, x, y, p))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)
        assert r.shape() == (640, 480)


def test_aedat4_no_stream_info(tmp_path):
    """Without a parseable infoNode the decoder falls back to identifying
    event packets by their EVTS FlatBuffer identifier."""
    t, x, y, p = random_events(100, 640, 480, seed=6)
    f = tmp_path / "v4_noinfo.aedat4"
    f.write_bytes(make_aedat4(t, x, y, p, info=b"not xml at all"))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)


def test_aedat4_lz4(tmp_path):
    pytest.importorskip("lz4")
    t, x, y, p = random_events(500, 640, 480, seed=7)
    f = tmp_path / "v4_lz4.aedat4"
    f.write_bytes(make_aedat4(t, x, y, p, compression=1))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)


def test_aedat_chunked_iteration_and_reset(tmp_path):
    t, x, y, p = random_events(1000, 640, 480, seed=8)
    f = tmp_path / "v4_chunks.aedat4"
    f.write_bytes(make_aedat4(t, x, y, p, events_per_packet=64))
    with EventReader(f, n_events=100) as r:
        chunks = [np.asarray(c) for c in r]
        assert sum(len(c) for c in chunks) == 1000
        assert all(len(c) <= 100 for c in chunks)
        got = np.concatenate(chunks)
        assert np.array_equal(got["t"], t)
        r.reset()
        check(r.read_all(), t, x, y, p)


def test_aedat_magic_sniffing(tmp_path):
    """Version line is recognised even without a known file extension."""
    t, x, y, p = random_events(50, 128, 128, seed=9)
    f = tmp_path / "recording.bin_dump"
    f.write_bytes(make_aedat1(t, x, y, p))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)
