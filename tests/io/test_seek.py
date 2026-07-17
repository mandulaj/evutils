"""Tests for timestamp / event-index random access (``EventReader.seek``).

Synthetic, CI-safe: a known ramp (``t = i*dt``) is written in each seekable
format and seeked by time, event index, relatively and backward. The ramp
timestamps exceed the EVT3 TIME_HIGH wrap period (2**24 µs), so the post-seek
wrap correction is exercised. A separate test validates the Metavision
``.tmp_index`` reader against the real recordings in ``data/`` when present.
"""
from pathlib import Path

import numpy as np
import pytest

from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype

# (format, extension). HDF5 needs the optional h5py dependency.
FORMATS = [
    ("evt3", "raw"), ("evt4", "raw"), ("evt2", "raw"), ("evt21", "raw"),
    ("dat", "dat"), ("csv", "csv"), ("npz", "npz"), ("hdf5", "hdf5"),
]

N = 20_000
DT = 1_000  # max t = 20_000_000 > 2**24 (16_777_216): EVT wrap is exercised.


def _ramp(n: int = N, dt: int = DT):
    ev = np.zeros(n, dtype=Event_dtype)
    ev["t"] = np.arange(n, dtype=np.int64) * dt
    ev["x"] = np.arange(n) % 1280
    ev["y"] = np.arange(n) % 720
    ev["p"] = np.arange(n) % 2
    return ev


def _write(path, fmt, ev, tr=None):
    if fmt in ("evt3", "evt4", "evt2", "evt21"):
        with EventWriter(path, format=fmt) as w:
            w.write(ev)
            if tr is not None:
                w.write_triggers(tr)
    else:  # dispatched by extension; needs geometry for evt/hdf5-style writers
        with EventWriter(path, width=1280, height=720) as w:
            w.write(ev)
            if tr is not None:
                w.write_triggers(tr)


@pytest.fixture(params=FORMATS, ids=[f[0] for f in FORMATS])
def recording(request, tmp_path):
    fmt, ext = request.param
    if fmt == "hdf5":
        pytest.importorskip("h5py")
    ev = _ramp()
    p = tmp_path / f"ramp_{fmt}.{ext}"
    _write(p, fmt, ev)
    return p, ev


def test_seek_by_time(recording):
    p, ev = recording
    T = 10_000_000
    exp = int(np.searchsorted(ev["t"], T))
    with EventReader(p, n_events=3000) as r:
        landed = r.seek(t=T)
        c = r.read()
    assert landed == int(ev["t"][exp])
    assert int(c.t[0]) == int(ev["t"][exp])
    assert int(c.t[0]) >= T


def test_seek_by_event_index(recording):
    p, ev = recording
    n = 7000
    with EventReader(p, n_events=3000) as r:
        landed = r.seek(n=n)
        c = r.read()
    assert landed == int(ev["t"][n])
    assert int(c.t[0]) == int(ev["t"][n])
    assert int(c.x[0]) == int(ev["x"][n])
    assert int(c.y[0]) == int(ev["y"][n])


def test_seek_backward(recording):
    p, ev = recording
    with EventReader(p, n_events=3000) as r:
        r.seek(n=7000)
        r.seek(n=200)  # backward
        c = r.read()
    assert int(c.t[0]) == int(ev["t"][200])


def test_seek_relative(recording):
    p, ev = recording
    with EventReader(p, n_events=3000) as r:
        r.seek(t=5_000_000)
        landed = r.seek(t=5_000_000, relative=True)  # -> 10_000_000
        c = r.read()
    exp = int(np.searchsorted(ev["t"], 10_000_000))
    assert landed == int(ev["t"][exp])
    assert int(c.t[0]) == int(ev["t"][exp])


def test_seek_then_drain_is_contiguous_tail(recording):
    p, ev = recording
    T = 8_000_000
    exp = int(np.searchsorted(ev["t"], T))
    with EventReader(p, n_events=1500) as r:
        r.seek(t=T)
        got = []
        while True:
            c = r.read()
            if len(c) == 0:
                break
            got.append(np.asarray(c.t).copy())
    tail = np.concatenate(got)
    assert np.array_equal(tail, ev["t"][exp:])


def test_seek_in_delta_t_mode(recording):
    p, ev = recording
    T = 10_000_000
    exp = int(np.searchsorted(ev["t"], T))
    with EventReader(p, delta_t=1_000_000) as r:
        r.seek(t=T)
        c = r.read()
    assert len(c) > 0
    assert int(c.t[0]) == int(ev["t"][exp])
    assert int(c.t[-1]) < T + 1_000_000


def test_seek_index_disabled_matches(recording):
    """index=False (no sidecar) yields the same result as the default."""
    p, ev = recording
    T = 12_345_000
    with EventReader(p, n_events=3000, index=False) as r:
        a = r.seek(t=T)
    with EventReader(p, n_events=3000) as r:
        b = r.seek(t=T)
    assert a == b == int(ev["t"][int(np.searchsorted(ev["t"], T))])


def test_seek_out_of_range(recording):
    p, ev = recording
    with EventReader(p, n_events=3000) as r:
        r.seek(t=10**15)  # far past the end
        assert len(r.read()) == 0


def test_seek_requires_exactly_one_axis(recording):
    p, _ = recording
    with EventReader(p, n_events=3000) as r:
        with pytest.raises(ValueError, match="exactly one"):
            r.seek()
        with pytest.raises(ValueError, match="exactly one"):
            r.seek(t=1, n=1)


# --------------------------------------------------------------------------- #
# Non-seekable source: linear iterate-and-skip fallback
# --------------------------------------------------------------------------- #

class _NonSeekable:
    """A read-only, non-seekable byte stream (like a pipe) over a bytes payload."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._data) - self._pos
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def seek(self, *args):
        import io as _io
        raise _io.UnsupportedOperation("underlying stream is not seekable")

    def tell(self) -> int:
        return self._pos


def test_seek_linear_fallback_non_seekable(tmp_path):
    from evutils.io._csv import EventDecoder_Csv

    ev = _ramp()
    p = tmp_path / "ramp.csv"
    _write(p, "csv", ev)
    data = Path(p).read_bytes()

    T = 6_000_000  # forward-only (backward on a non-seekable stream can't rewind)
    exp = int(np.searchsorted(ev["t"], T))
    with EventReader(_NonSeekable(data), n_events=3000,
                     file_decoder=EventDecoder_Csv) as r:
        landed = r.seek(t=T)
        c = r.read()
    assert landed == int(ev["t"][exp])
    assert int(c.t[0]) == int(ev["t"][exp])


# --------------------------------------------------------------------------- #
# Metavision .tmp_index interop (gated on the local data/ fixtures)
# --------------------------------------------------------------------------- #

_DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.mark.skipif(not (_DATA / "short.raw.tmp_index").is_file(),
                    reason="data/short.raw(.tmp_index) fixtures not present")
def test_metavision_index_matches_built_index():
    """A sidecar-driven seek must land identically to a built-index seek."""
    from evutils.io._evt import EventDecoder_EVT
    from evutils.io._index import metavision_index_path, read_metavision_index
    from evutils.io._source import make_source

    raw = _DATA / "short.raw"
    d = EventDecoder_EVT(make_source(str(raw)))
    d.init()
    ws = np.dtype(d._word_dtype).itemsize
    idx = read_metavision_index(metavision_index_path(str(raw)), str(raw),
                                d._payload_off, ws)
    d.close()
    assert idx is not None and len(idx) > 0
    assert bool(np.all(np.diff(idx.ts) >= 0))  # monotonic timeline

    T = int(idx.ts[len(idx) // 2]) + 12_345

    with EventReader(str(raw), n_events=1000, index="metavision") as r:  # sidecar
        la = r.seek(t=T)
        ca = r.read()
    with EventReader(str(raw), n_events=1000, index=False) as r:  # exact built
        lb = r.seek(t=T)
        cb = r.read()

    assert la == lb
    assert int(ca.t[0]) == int(cb.t[0]) >= T
    assert int(ca.x[0]) == int(cb.x[0])
    assert int(ca.y[0]) == int(cb.y[0])



def test_seek_with_normalize_ts(tmp_path):
    fmt, ext = "evt3", "raw"
    ev = _ramp()
    ev["t"] += 100_000 # start at 100k
    p = tmp_path / f"ramp_norm_{fmt}.{ext}"
    _write(p, fmt, ev)

    T = 5_000_000
    with EventReader(p, n_events=3000, normalize_ts=True) as r:
        r.read() # Anchor first_ts so normalize_ts knows the start of the file
        landed = r.seek(t=T)
        c = r.read()
    
    # EventReader.seek() expects/returns absolute timestamps
    assert landed == T
    # EventReader.read() yields normalized timestamps (T - 100_000)
    assert int(c.t[0]) >= T - 100_000

def test_repeated_seek_past_wrap(tmp_path):
    fmt, ext = "evt3", "raw"
    ev = _ramp()
    p = tmp_path / f"ramp_wrap_{fmt}.{ext}"
    _write(p, fmt, ev)

    T1 = 17_000_000 # Past EVT3 wrap (16.7M)
    T2 = 18_000_000
    with EventReader(p, n_events=3000) as r:
        r.seek(t=T1)
        r.seek(t=T2)
        r.seek(t=T1)
        c = r.read()
    assert int(c.t[0]) >= T1

def test_seek_past_eof(tmp_path):
    fmt, ext = "evt3", "raw"
    ev = _ramp()
    p = tmp_path / f"ramp_eof_{fmt}.{ext}"
    _write(p, fmt, ev)

    with EventReader(p, n_events=3000) as r:
        r.seek(t=100_000_000) # Far past end
        c = r.read()
    assert len(c) == 0

