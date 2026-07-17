"""Transparent compression: reading and writing compressed event files.

Covers auto-open by path (``.gz`` / ``.zst`` / ``.xz`` / ``.bz2``), passing an
already-open compressed file object as source, CSV over a compressed stream,
and seeking over a compressed (non-seekable) native source.
"""
import gzip
import io

import numpy as np
import pytest

from typing import Any
from evutils.io import EventReader, EventWriter
from evutils.io._compression import (
    COMPRESSION_SUFFIXES,
    is_compressed_path,
    open_compressed,
    strip_compression_suffix,
)
from evutils.types import Event_dtype


def _has_zstd() -> bool:
    try:
        open_compressed("x.zst", "rb")  # raises ImportError if no backend
    except ImportError:
        return False
    except Exception:
        # FileNotFoundError etc. means a backend *is* available.
        return True
    return True


# Suffixes always available via stdlib, plus zst when a backend exists.
_SUFFIXES = [".gz", ".xz", ".bz2"]
if _has_zstd():
    _SUFFIXES.append(".zst")


def make_events(n: int = 5000, t_max: int = 100_000, seed: int = 42) -> Any:
    rng = np.random.default_rng(seed)
    ev = np.zeros(n, dtype=Event_dtype)
    ev["t"] = np.sort(rng.integers(0, t_max, n))
    ev["x"] = rng.integers(0, 1280, n)
    ev["y"] = rng.integers(0, 720, n)
    ev["p"] = rng.integers(0, 2, n)
    return ev


def assert_events_equal(out: Any, ref: Any, context: str = "") -> None:
    out = np.asarray(out)
    assert len(out) == len(ref), f"{context}: length {len(out)} != {len(ref)}"
    for f in ("t", "x", "y", "p"):
        assert np.array_equal(out[f], ref[f]), f"{context}: field {f!r} differs"


# --------------------------------------------------------------------------- #
# Helper unit tests
# --------------------------------------------------------------------------- #
def test_is_compressed_path() -> None:
    assert is_compressed_path("foo.raw.zst")
    assert is_compressed_path("foo.csv.gz")
    assert is_compressed_path("a.xz")
    assert is_compressed_path("a.bz2")
    assert not is_compressed_path("foo.raw")
    assert not is_compressed_path("foo.csv")
    # exactly the documented set
    assert COMPRESSION_SUFFIXES == {".gz", ".zst", ".xz", ".bz2"}


def test_strip_compression_suffix() -> None:
    assert strip_compression_suffix("foo.raw.zst") == "foo.raw"
    assert strip_compression_suffix("events.csv.gz") == "events.csv"
    assert strip_compression_suffix("bar.raw") == "bar.raw"          # unchanged
    assert strip_compression_suffix("nested/foo.dat.xz") == "nested/foo.dat"


def test_open_compressed_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError):
        open_compressed("foo.raw", "rb")


# --------------------------------------------------------------------------- #
# Round-trip: write compressed by path, read back by path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suffix", _SUFFIXES)
def test_evt_roundtrip_compressed(tmp_path: Any, suffix: str) -> None:
    ev = make_events()
    p = tmp_path / f"out.raw{suffix}"
    with EventWriter(p, format="evt3") as w:
        w.write(ev)
    with EventReader(p) as r:
        out = np.asarray(r.read_all())
    assert_events_equal(out, ev, suffix)
    # first + last events match, as required.
    assert out["t"][0] == ev["t"][0] and out["t"][-1] == ev["t"][-1]


@pytest.mark.parametrize("suffix", _SUFFIXES)
def test_compressed_matches_uncompressed(tmp_path: Any, suffix: str) -> None:
    """A compressed read yields exactly the same events as the plain read."""
    ev = make_events()
    plain = tmp_path / "out.raw"
    comp = tmp_path / f"out.raw{suffix}"
    with EventWriter(plain, format="evt3") as w:
        w.write(ev)
    with EventWriter(comp, format="evt3") as w:
        w.write(ev)
    with EventReader(plain) as r:
        a = np.asarray(r.read_all()).copy()
    with EventReader(comp) as r:
        b = np.asarray(r.read_all())
    assert_events_equal(b, a, suffix)


# --------------------------------------------------------------------------- #
# Passing an already-open compressed file object as source
# --------------------------------------------------------------------------- #
def test_read_open_gzip_fileobj(tmp_path: Any) -> None:
    ev = make_events()
    p = tmp_path / "out.raw.gz"
    with EventWriter(p, format="evt3") as w:
        w.write(ev)
    with gzip.open(p, "rb") as f:
        with EventReader(f) as r:
            out = np.asarray(r.read_all())
    assert_events_equal(out, ev, "gzip-fileobj")


def test_read_bytesio_of_gzip(tmp_path: Any) -> None:
    """A GzipFile wrapping an in-memory buffer (no name) sniffs by content."""
    ev = make_events()
    p = tmp_path / "out.raw.gz"
    with EventWriter(p, format="evt3") as w:
        w.write(ev)
    data = p.read_bytes()
    gz = gzip.GzipFile(fileobj=io.BytesIO(data))  # no .name -> content sniff
    with EventReader(gz) as r:
        out = np.asarray(r.read_all())
    assert_events_equal(out, ev, "gzip-bytesio")


# --------------------------------------------------------------------------- #
# CSV over a compressed (non-seekable) stream: header + no-header
# --------------------------------------------------------------------------- #
def test_csv_gzip_with_header(tmp_path: Any) -> None:
    ev = make_events()
    p = tmp_path / "events.csv.gz"
    with EventWriter(p) as w:            # CSV encoder writes a header by default
        w.write(ev)
    with EventReader(p) as r:
        out = np.asarray(r.read_all())
    assert_events_equal(out, ev, "csv.gz+header")


def test_csv_gzip_no_header(tmp_path: Any) -> None:
    ev = make_events()
    p = tmp_path / "events.csv.gz"
    with EventWriter(p, header=False) as w:
        w.write(ev)
    with EventReader(p) as r:
        out = np.asarray(r.read_all())
    assert_events_equal(out, ev, "csv.gz-noheader")


def test_csv_open_gzip_fileobj(tmp_path: Any) -> None:
    ev = make_events()
    p = tmp_path / "events.csv.gz"
    with EventWriter(p) as w:
        w.write(ev)
    with gzip.open(p, "rb") as f:
        with EventReader(f) as r:
            out = np.asarray(r.read_all())
    assert_events_equal(out, ev, "csv gzip-fileobj")


# --------------------------------------------------------------------------- #
# Seeking over a compressed (non-seekable) native source
# --------------------------------------------------------------------------- #
def test_seek_over_compressed_evt(tmp_path: Any) -> None:
    ev = make_events()
    p = tmp_path / "out.raw.gz"
    with EventWriter(p, format="evt3") as w:
        w.write(ev)
    with EventReader(p) as r:
        landed = r.seek(n=1000)
        chunk = np.asarray(r.read(n_events=5))
    assert landed.ts == ev["t"][1000]
    assert chunk["t"][0] == ev["t"][1000]


# --------------------------------------------------------------------------- #
# Chunked write over a compressed stream
# --------------------------------------------------------------------------- #
def test_chunked_write_compressed(tmp_path: Any) -> None:
    ev = make_events()
    p = tmp_path / "out.raw.bz2"
    with EventWriter(p, format="evt3") as w:
        for part in np.array_split(ev, 11):
            w.write(part)
    with EventReader(p) as r:
        out = np.asarray(r.read_all())
    assert_events_equal(out, ev, "chunked.bz2")
