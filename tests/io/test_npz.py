"""NPZ decoder/encoder tests: numpy interoperability, compression, geometry
and malformed-archive errors. (Bulk/chunked/windowed round-trips are covered
by the cross-format matrix in test_roundtrip_matrix.py.)"""
import numpy as np
import pytest

from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype

from typing import Any

def make_events(n: int=10_000, seed: int=11) -> Any:
    rng = np.random.default_rng(seed)
    ev = np.zeros(n, dtype=Event_dtype)
    ev["t"] = np.sort(rng.integers(0, 1_000_000, n))
    ev["x"] = rng.integers(0, 1280, n)
    ev["y"] = rng.integers(0, 720, n)
    ev["p"] = rng.integers(0, 2, n)
    return ev


def test_np_load_reads_our_files(tmp_path: Any) -> None:
    """Files we write are plain npz archives np.load understands."""
    ev = make_events()
    p = tmp_path / "ours.npz"
    with EventWriter(p, width=640, height=480) as w:
        w.write(ev)

    d = np.load(p)
    assert sorted(d.files) == ["height", "p", "t", "width", "x", "y"]
    assert np.array_equal(d["t"], ev["t"])
    assert np.array_equal(d["x"], ev["x"])
    assert (int(d["width"]), int(d["height"])) == (640, 480)


def test_we_read_np_savez_soa(tmp_path: Any) -> None:
    ev = make_events()
    p = tmp_path / "soa.npz"
    np.savez(p, t=ev["t"], x=ev["x"], y=ev["y"], p=ev["p"])
    with EventReader(p) as r:
        assert np.array_equal(np.asarray(r.read_all()), ev)


def test_we_read_np_savez_structured(tmp_path: Any) -> None:
    """A single structured 'events' member (compressed) is accepted too."""
    ev = make_events()
    p = tmp_path / "aos.npz"
    np.savez_compressed(p, events=ev)
    with EventReader(p) as r:
        assert np.array_equal(np.asarray(r.read_all()), ev)


def test_compressed_write(tmp_path: Any) -> None:
    ev = make_events()
    plain = tmp_path / "plain.npz"
    packed = tmp_path / "packed.npz"
    with EventWriter(plain) as w:
        w.write(ev)
    with EventWriter(packed, compressed=True) as w:
        w.write(ev)
    assert packed.stat().st_size < plain.stat().st_size
    with EventReader(packed) as r:
        assert np.array_equal(np.asarray(r.read_all()), ev)


def test_geometry_roundtrip(tmp_path: Any) -> None:
    p = tmp_path / "geo.npz"
    with EventWriter(p, width=346, height=260) as w:
        w.write(make_events(10))
    with EventReader(p) as r:
        r.read()
        assert r.shape() == (346, 260)


def test_missing_event_keys_errors(tmp_path: Any) -> None:
    p = tmp_path / "bogus.npz"
    np.savez(p, foo=np.arange(5))
    with pytest.raises(ValueError, match="does not contain event data"):
        EventReader(p).read_all()


def test_mismatched_column_lengths_errors(tmp_path: Any) -> None:
    p = tmp_path / "ragged.npz"
    np.savez(p, t=np.arange(10), x=np.arange(9, dtype=np.uint16),
             y=np.zeros(10, np.uint16), p=np.zeros(10, np.uint8))
    with pytest.raises(ValueError, match="mismatched lengths"):
        EventReader(p).read_all()


def test_large_timestamps(tmp_path: Any) -> None:
    ev = make_events(100)
    ev["t"] += 2**40  # far beyond 32 bits
    p = tmp_path / "big_t.npz"
    with EventWriter(p) as w:
        w.write(ev)
    with EventReader(p) as r:
        out = r.read_all()
        assert not isinstance(out, tuple)
        assert np.array_equal(out["t"], ev["t"])

def test_NPZ_empty_arrays(tmp_path: Any) -> None:
    p = tmp_path / "empty_arrays.npz"
    np.savez(p, t=np.zeros(0, dtype=np.int64), x=np.zeros(0, dtype=np.uint16),
             y=np.zeros(0, dtype=np.uint16), p=np.zeros(0, dtype=np.uint8))
    with EventReader(p) as r:
        out = r.read_all()
        assert len(out) == 0

def test_NPZ_empty_file(tmp_path: Any) -> None:
    import zipfile
    p = tmp_path / "empty_file.npz"
    p.touch()
    with pytest.raises(zipfile.BadZipFile):
        with EventReader(p) as r:
            r.read_all()
