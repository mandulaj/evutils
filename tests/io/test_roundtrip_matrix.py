"""Cross-format round-trip matrix.

Every read+write format goes through the same battery: bulk round-trip,
chunked writing, windowed reading (``n_events`` and ``delta_t``), reset,
iteration, empty files and timestamp normalization. Format-specific edge
cases (timestamp wraps, container layouts, ...) live in the per-format test
modules.
"""
import numpy as np
import pytest

from typing import Any
from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype

@pytest.fixture(autouse=True)
def skip_missing_deps(request: Any) -> None:
    if "fmt" in request.fixturenames:
        fmt = request.getfixturevalue("fmt")

        if fmt == "csv":
            pass
        elif fmt == "hdf5":
            pytest.importorskip("h5py")
            pytest.importorskip("hdf5plugin")

# name -> (suffix, writer kwargs, capabilities)
FORMATS = {
    "evt3": (".raw", {"format": "evt3"}, {}),
    "evt21": (".raw", {"format": "evt21"}, {}),
    "evt2": (".raw", {"format": "evt2"}, {}),
    "dat": (".dat", {}, {}),
    "aer": (".aer", {}, {"lossy_t": True, "coord_max": 512}),
    "npz": (".npz", {}, {}),
    "hdf5": (".h5", {}, {}),
    "csv": (".csv", {}, {}),
}

PARAMS = sorted(FORMATS)


def make_events(n: int=5000, coord_max: Any=None, seed: int=42, t_max: int=100_000) -> Any:
    rng = np.random.default_rng(seed)
    ev = np.zeros(n, dtype=Event_dtype)
    ev["t"] = np.sort(rng.integers(0, t_max, n))
    ev["x"] = rng.integers(0, coord_max or 1280, n)
    ev["y"] = rng.integers(0, coord_max or 720, n)
    ev["p"] = rng.integers(0, 2, n)
    return ev


def expected(ev: Any, caps: Any) -> Any:
    """The array a lossless read should return (AER drops timestamps)."""
    if caps.get("lossy_t"):
        ev = ev.copy()
        ev["t"] = 0
    return ev


def write_file(tmp_path: Any, fmt: str, ev: Any, n_chunks: int=1) -> Any:
    suffix, kwargs, _ = FORMATS[fmt]
    p = tmp_path / f"events_{fmt}{suffix}"
    with EventWriter(p, **kwargs) as w: # type: ignore
        for part in np.array_split(ev, n_chunks):
            w.write(part)
    return p


def assert_events_equal(out: Any, ref: Any, context: str="") -> None:
    out = np.asarray(out)
    assert len(out) == len(ref), f"{context}: length {len(out)} != {len(ref)}"
    for f in ("t", "x", "y", "p"):
        assert np.array_equal(out[f], ref[f]), f"{context}: field {f!r} differs"


@pytest.mark.parametrize("fmt", PARAMS)
def test_bulk_roundtrip(tmp_path: Any, fmt: str) -> None:
    caps = FORMATS[fmt][2]
    ev = make_events(coord_max=caps.get("coord_max"))
    p = write_file(tmp_path, fmt, ev)
    with EventReader(p) as r:
        assert_events_equal(r.read_all(), expected(ev, caps), fmt)


@pytest.mark.parametrize("fmt", PARAMS)
def test_chunked_write_roundtrip(tmp_path: Any, fmt: str) -> None:
    """Writing in many small chunks must be byte-equivalent to one bulk write."""
    caps = FORMATS[fmt][2]
    ev = make_events(coord_max=caps.get("coord_max"))
    p = write_file(tmp_path, fmt, ev, n_chunks=13)
    with EventReader(p) as r:
        assert_events_equal(r.read_all(), expected(ev, caps), fmt)


@pytest.mark.parametrize("fmt", PARAMS)
def test_windowed_n_events(tmp_path: Any, fmt: str) -> None:
    caps = FORMATS[fmt][2]
    ev = make_events(coord_max=caps.get("coord_max"))
    p = write_file(tmp_path, fmt, ev)
    with EventReader(p, n_events=137) as r:
        chunks = [np.asarray(c) for c in r]
    assert all(len(c) <= 137 for c in chunks)
    assert_events_equal(np.concatenate(chunks), expected(ev, caps), fmt)


@pytest.mark.parametrize("fmt", PARAMS)
def test_windowed_delta_t(tmp_path: Any, fmt: str) -> None:
    caps = FORMATS[fmt][2]
    if caps.get("lossy_t"):
        pytest.skip("format has no timestamps")
    ev = make_events()
    p = write_file(tmp_path, fmt, ev)
    with EventReader(p, delta_t=1000) as r:
        chunks = [np.asarray(c) for c in r]
    for c in chunks:
        if len(c) > 1:
            assert int(c["t"].max()) - int(c["t"].min()) <= 1000
    assert_events_equal(np.concatenate(chunks), ev, fmt)


@pytest.mark.parametrize("fmt", PARAMS)
def test_reset(tmp_path: Any, fmt: str) -> None:
    caps = FORMATS[fmt][2]
    ev = make_events(coord_max=caps.get("coord_max"))
    p = write_file(tmp_path, fmt, ev)
    with EventReader(p) as r:
        first = np.asarray(r.read_all()).copy()
        r.reset()
        second = np.asarray(r.read_all())
    assert_events_equal(second, first, fmt)


@pytest.mark.parametrize("fmt", PARAMS)
def test_empty_write_then_read(tmp_path: Any, fmt: str) -> None:
    """A writer that never receives events must still produce a readable,
    zero-event file."""
    suffix, kwargs, _ = FORMATS[fmt]
    p = tmp_path / f"empty{suffix}"
    with EventWriter(p, **kwargs) as w: # type: ignore
        w.write(np.zeros(0, dtype=Event_dtype))
    with EventReader(p) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert len(out) == 0
    # The empty columns keep their canonical dtypes.
    assert out["t"].dtype == np.int64
    assert out["x"].dtype == np.uint16


@pytest.mark.parametrize("fmt", PARAMS)
def test_read_after_eof_is_empty(tmp_path: Any, fmt: str) -> None:
    caps = FORMATS[fmt][2]
    ev = make_events(n=100, coord_max=caps.get("coord_max"))
    p = write_file(tmp_path, fmt, ev)
    with EventReader(p) as r:
        r.read_all()
        assert r.is_eof()
        assert len(r.read_all()) == 0


@pytest.mark.parametrize("fmt", PARAMS)
def test_normalize_ts(tmp_path: Any, fmt: str) -> None:
    caps = FORMATS[fmt][2]
    if caps.get("lossy_t"):
        pytest.skip("format has no timestamps")
    ev = make_events()
    ev["t"] += 5000  # ensure a non-zero first timestamp
    p = write_file(tmp_path, fmt, ev)
    with EventReader(p, normalize_ts=True) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert int(out["t"][0]) == 0
    assert np.array_equal(np.diff(out["t"]), np.diff(ev["t"].astype(np.int64)))
