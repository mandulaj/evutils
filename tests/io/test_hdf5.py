"""HDF5 decoder/encoder tests: DSEC/RVT layout, Prophesee layout, the
``ms_to_idx`` millisecond index and DSEC ``t_offset`` handling."""
import numpy as np
import pytest


from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype

from typing import Any

h5py = pytest.importorskip("h5py")

def make_events(n: int=20_000, t_max: int=200_000, seed: int=3) -> Any:
    rng = np.random.default_rng(seed)
    ev = np.zeros(n, dtype=Event_dtype)
    ev["t"] = np.sort(rng.integers(0, t_max, n))
    ev["x"] = rng.integers(0, 1280, n)
    ev["y"] = rng.integers(0, 720, n)
    ev["p"] = rng.integers(0, 2, n)
    return ev


def test_geometry_attrs_roundtrip(tmp_path: Any) -> None:
    p = tmp_path / "geo.h5"
    with EventWriter(p, width=640, height=480) as w:
        w.write(make_events(100))
    with EventReader(p) as r:
        r.read()
        assert r.shape() == (640, 480)


def test_ms_to_idx_random_access(tmp_path: Any) -> None:
    """decoder.read(start_ms, end_ms) must return exactly the events with
    start_ms*1000 <= t < end_ms*1000."""
    ev = make_events()
    p = tmp_path / "idx.h5"
    with EventWriter(p) as w:
        for part in np.array_split(ev, 7):  # index must survive chunked writes
            w.write(part)

    with EventReader(p) as r:
        r.init()
        dec: Any = r._file_decoder
        for start_ms, end_ms in [(0, 50), (50, 120), (13, 14), (0, -1), (150, 10_000)]:
            got = dec.read(start_ms, end_ms)
            lo = start_ms * 1000
            hi = ev["t"].max() + 1 if end_ms == -1 or end_ms * 1000 > ev["t"].max() \
                else end_ms * 1000
            ref = ev[(ev["t"] >= lo) & (ev["t"] < hi)]
            assert len(got) == len(ref), (start_ms, end_ms)
            assert np.array_equal(got["t"], ref["t"])

        # Past-the-end start is empty; invalid ranges raise.
        assert len(dec.read(10**6, -1)) == 0
        with pytest.raises(ValueError):
            dec.read(-1, 10)
        with pytest.raises(ValueError):
            dec.read(100, 50)


def test_dsec_t_offset(tmp_path: Any) -> None:
    """A DSEC-style ``t_offset`` dataset shifts the decoded timestamps."""
    ev = make_events(1000)
    p = tmp_path / "dsec.h5"
    with EventWriter(p) as w:
        w.write(ev)
    with h5py.File(p, "r+") as f:
        f.create_dataset("t_offset", data=np.int64(1_000_000))

    with EventReader(p) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert np.array_equal(out["t"], ev["t"].astype(np.int64) + 1_000_000)


def _write_prophesee_layout(path: Any, ev: Any, root: Any="CD") -> None:
    """Synthesize the Metavision HDF5 layout: compound CD/events dataset."""
    prophesee_dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
    rec = np.zeros(len(ev), dtype=prophesee_dtype)
    for f in ("x", "y", "p", "t"):
        rec[f] = ev[f]
    chunks = (min(16384, len(rec)),)
    with h5py.File(path, "w") as f:
        if root:
            f.create_group(root).create_dataset("events", data=rec, chunks=chunks)
        else:
            f.create_dataset("events", data=rec, chunks=chunks)


def test_prophesee_layout(tmp_path: Any) -> None:
    """Uncompressed Metavision-layout files (CD/events compound dataset) read
    transparently through the same EventReader interface."""
    ev = make_events()
    p = tmp_path / "metavision.hdf5"
    _write_prophesee_layout(p, ev)

    with EventReader(p) as r:
        out = np.asarray(r.read_all())
    for f in ("t", "x", "y", "p"):
        assert np.array_equal(out[f], ev[f]), f

    # Chunked iteration streams the compound dataset too.
    with EventReader(p, n_events=1024) as r:
        total = sum(len(c) for c in r)
    assert total == len(ev)


def test_root_level_structured_dataset(tmp_path: Any) -> None:
    """A compound 'events' dataset at the root (no CD group) also reads."""
    ev = make_events(500)
    p = tmp_path / "flat.h5"
    _write_prophesee_layout(p, ev, root=None)
    with EventReader(p) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert np.array_equal(out["t"], ev["t"])


def test_missing_events_errors(tmp_path: Any) -> None:
    p = tmp_path / "bogus.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("unrelated", data=np.arange(3))
    with pytest.raises(ValueError, match="neither an 'events'"):
        EventReader(p).read_all()


def test_no_index_read_raises(tmp_path: Any) -> None:
    """Millisecond random access needs ms_to_idx; a clear error otherwise."""
    ev = make_events(100)
    p = tmp_path / "noidx.hdf5"
    _write_prophesee_layout(p, ev)
    with EventReader(p) as r:
        r.init()
        dec: Any = r._file_decoder
        with pytest.raises(ValueError, match="ms_to_idx"):
            dec.read(0, 10)


def test_large_timestamps(tmp_path: Any) -> None:
    """int64 timestamps (e.g. absolute epoch microseconds) survive, and the
    millisecond index stays small by anchoring at the first event
    (ms_to_idx_offset)."""
    epoch = 1_663_249_605_734_020
    ev = make_events()
    ev["t"] += epoch
    p = tmp_path / "epoch.h5"
    with EventWriter(p) as w:
        w.write(ev)
    with EventReader(p) as r:
        out = r.read_all()
        assert not isinstance(out, tuple)
        assert np.array_equal(out["t"], ev["t"])
        # Random access uses absolute milliseconds; the offset is transparent.
        dec: Any = r._file_decoder
        start_ms = (epoch // 1000) + 50
        got = dec.read(start_ms, start_ms + 20)
        ref = ev[(ev["t"] >= start_ms * 1000) & (ev["t"] < (start_ms + 20) * 1000)]
        assert len(got) == len(ref)
        assert np.array_equal(got["t"], ref["t"])

def test_HDF5_empty_dataset(tmp_path: Any) -> None:
    p = tmp_path / "empty.h5"
    with EventWriter(p) as w:
        pass # write nothing
    with EventReader(p) as r:
        out = r.read_all()
        assert len(out) == 0
