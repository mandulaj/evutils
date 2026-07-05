"""Registry of third-party EVT readers for the comparison benchmarks.

Each :class:`Reader` lazily imports its library *inside* ``read`` so that a
missing or broken install surfaces as an ``ImportError`` (which the benchmark
turns into a skip) rather than a collection error. To add a library, append one
``Reader`` entry.

The reader APIs marked ``VERIFY`` are best-effort against the libraries' recent
releases -- if your installed version differs, adjust the one small ``read``
helper below; the benchmark harness stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

ALL_FORMATS = ("evt3", "evt2", "evt21")

# How many events to pull per chunk for chunked/streaming readers.
_CHUNK = 5_000_000


def _count(obj) -> int:
    """Best-effort event count across the return types the libraries use
    (numpy structured array, polars DataFrame, dict of columns, ...)."""
    try:
        return len(obj)
    except TypeError:
        pass
    if hasattr(obj, "height"):      # polars DataFrame
        return int(obj.height)
    if hasattr(obj, "shape"):       # numpy-like
        return int(obj.shape[0])
    if isinstance(obj, dict) and obj:
        return len(next(iter(obj.values())))
    raise TypeError(f"cannot count events in {type(obj)!r}")


# --------------------------------------------------------------------------- #
# Per-library read helpers: (path, fmt) -> event count
# --------------------------------------------------------------------------- #
def _read_expelliarmus(path: str, fmt: str) -> int:
    # https://github.com/open-neuromorphic/expelliarmus  (evt2, evt3, dat)
    from expelliarmus import Wizard
    return _count(Wizard(encoding=fmt).read(str(path)))


# NOTE: tonic (https://github.com/neuromorphs/tonic) has no standalone EVT/.raw
# reader -- it reads Prophesee data via `expelliarmus.Wizard` inside its dataset
# pipelines. Benchmarking it would just re-measure expelliarmus (already below,
# without the torchdata overhead), so it is intentionally not registered.


def _read_evlib(path: str, fmt: str) -> int:
    # https://github.com/tallamjr/evlib  (Rust-backed). Auto-detects the format;
    # load_events() returns a polars LazyFrame, so collect() to force decoding.
    import evlib
    df = evlib.load_events(str(path))
    if hasattr(df, "collect"):
        df = df.collect()
    return _count(df)


def _read_openeb(path: str, fmt: str) -> int:
    # OpenEB / Metavision SDK (not on PyPI -- see benchmarks/docker).
    from metavision_core.event_io import RawReader
    reader = RawReader(str(path))
    n = 0
    while not reader.is_done():
        n += len(reader.load_n_events(_CHUNK))
    return n


@dataclass(frozen=True)
class Reader:
    name: str
    formats: tuple            # subset of ALL_FORMATS this library can read
    read: Callable[[str, str], int]


#: Third-party readers benchmarked against evutils. evutils itself is measured
#: separately in test_read.py; with ``--benchmark-group-by=param:fmt`` all rows
#: for a given format line up in one table.
READERS = [
    Reader("expelliarmus", ("evt2", "evt3"), _read_expelliarmus),
    Reader("evlib", ("evt2", "evt3"), _read_evlib),
    Reader("openeb", ALL_FORMATS, _read_openeb),
]
