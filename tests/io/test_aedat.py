"""AEDAT 1.0 / 2.0 / 3.1 / 4.0 decoder tests against synthesized files.

The writers below build byte-exact files following the iniVation file-format
documentation (v1-v3, big-endian jAER layouts) and the DV framework layout
(v4, FlatBuffers), so the decoder is checked against the spec rather than
against itself.
"""
import numpy as np
import pytest

from aedat_synth import make_aedat1, make_aedat2, make_aedat3, make_aedat4 # type: ignore
from evutils.io import EventReader


from typing import Any
def random_events(n: int, width: int, height: int, seed: int=0) -> tuple[Any, Any, Any, Any]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 1_000_000, n))
    x = rng.integers(0, width, n)
    y = rng.integers(0, height, n)
    p = rng.integers(0, 2, n)
    return t, x, y, p


def check(out: Any, t: Any, x: Any, y: Any, p: Any) -> None:
    assert not isinstance(out, tuple)
    assert len(out) == len(t)
    assert np.array_equal(out["t"], t)
    assert np.array_equal(out["x"], x)
    assert np.array_equal(out["y"], y)
    assert np.array_equal(out["p"], p)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_aedat1(tmp_path: Any) -> None:
    t, x, y, p = random_events(500, 128, 128)
    f = tmp_path / "v1.aedat"
    f.write_bytes(make_aedat1(t, x, y, p))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)
        assert r.shape() == (128, 128)


def test_aedat1_headerless(tmp_path: Any) -> None:
    """A bare-'#' or headerless file defaults to AEDAT 1.0 (jAER convention)."""
    t, x, y, p = random_events(100, 128, 128, seed=1)
    f = tmp_path / "v1_bare.aedat"
    f.write_bytes(make_aedat1(t, x, y, p, header=False))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)


def test_aedat2_davis_skips_aps(tmp_path: Any) -> None:
    t, x, y, p = random_events(500, 240, 180, seed=2)
    f = tmp_path / "v2.aedat"
    f.write_bytes(make_aedat2(t, x, y, p, aps_every=10))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)
        assert r.shape() == (240, 180)


def test_aedat2_timestamp_wrap(tmp_path: Any) -> None:
    """32-bit µs timestamps wrap
    the decoder must extend them to 64 bits."""
    t32 = np.array([2**32 - 20, 2**32 - 10, 5, 15], dtype=np.uint64)
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, 7, 8])
    p = np.array([0, 1, 0, 1])
    f = tmp_path / "wrap.aedat"
    f.write_bytes(make_aedat2(t32 & 0xFFFFFFFF, x, y, p))
    with EventReader(f) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    expected = np.array([2**32 - 20, 2**32 - 10, 2**32 + 5, 2**32 + 15], dtype=np.int64)
    assert np.array_equal(out["t"], expected)


def test_aedat3(tmp_path: Any) -> None:
    t, x, y, p = random_events(500, 346, 260, seed=3)
    f = tmp_path / "v3.aedat"
    f.write_bytes(make_aedat3(t, x, y, p))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)


def test_aedat3_ts_overflow(tmp_path: Any) -> None:
    """The packet header's TS-overflow counter extends the 31-bit timestamps."""
    t, x, y, p = random_events(8, 346, 260, seed=4)
    f = tmp_path / "v3_ovf.aedat"
    f.write_bytes(make_aedat3(t, x, y, p, ts_overflow=2))
    with EventReader(f) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert np.array_equal(out["t"], t.astype(np.int64) + (2 << 31))


def test_aedat4_uncompressed(tmp_path: Any) -> None:
    t, x, y, p = random_events(500, 640, 480, seed=5)
    t = t + 1_663_249_605_734_020  # DV timestamps are absolute epoch µs
    f = tmp_path / "v4.aedat4"
    f.write_bytes(make_aedat4(t, x, y, p))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)
        assert r.shape() == (640, 480)


def test_aedat4_no_stream_info(tmp_path: Any) -> None:
    """Without a parseable infoNode the decoder falls back to identifying
    event packets by their EVTS FlatBuffer identifier."""
    t, x, y, p = random_events(100, 640, 480, seed=6)
    f = tmp_path / "v4_noinfo.aedat4"
    f.write_bytes(make_aedat4(t, x, y, p, info=b"not xml at all"))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)


def test_aedat4_lz4(tmp_path: Any) -> None:
    pytest.importorskip("lz4")
    t, x, y, p = random_events(500, 640, 480, seed=7)
    f = tmp_path / "v4_lz4.aedat4"
    f.write_bytes(make_aedat4(t, x, y, p, compression=1))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)


def test_aedat_chunked_iteration_and_reset(tmp_path: Any) -> None:
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


def test_aedat_magic_sniffing(tmp_path: Any) -> None:
    """Version line is recognised even without a known file extension."""
    t, x, y, p = random_events(50, 128, 128, seed=9)
    f = tmp_path / "recording.bin_dump"
    f.write_bytes(make_aedat1(t, x, y, p))
    with EventReader(f) as r:
        check(r.read_all(), t, x, y, p)

def test_AEDAT_empty_file(tmp_path: Any) -> None:
    f = tmp_path / "empty.aedat"
    f.touch()
    with EventReader(f) as r:
        out = r.read_all()
        assert len(out) == 0

def test_AEDAT_truncated_payload(tmp_path: Any) -> None:
    f = tmp_path / "truncated.aedat"
    f.write_bytes(b"\x01\x02\x03\x04") # 4 bytes
    with EventReader(f) as r:
        out = r.read_all()
        assert len(out) == 0

def test_aedat4_encoder_roundtrip(tmp_path: Any) -> None:
    from evutils.io import EventWriter
    from evutils.types import EventArray
    events = EventArray(
        t=np.array([1000, 2000, 3000], dtype=np.int64),
        x=np.array([10, 20, 30], dtype=np.uint16),
        y=np.array([15, 25, 35], dtype=np.uint16),
        p=np.array([1, 0, 1], dtype=np.uint8)
    )

    fpath = tmp_path / "test.aedat4"
    with EventWriter(fpath, format="aedat") as w:
        w.write(events)

    with EventReader(fpath) as r:
        out = r.read_all()
        if isinstance(out, tuple):
            out = out[0]
        assert np.array_equal(out.t, events.t)
        assert np.array_equal(out.x, events.x)
        assert np.array_equal(out.y, events.y)
        assert np.array_equal(out.p, events.p)

