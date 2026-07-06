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
from typing import Any, Callable

ALL_FORMATS = ("evt3", "evt2", "evt21")

# How many events to pull per chunk for chunked/streaming readers.
_CHUNK = 5_000_000


def _count(obj: Any) -> int:
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
def _read_expelliarmus(path: str, fmt: str) -> tuple[int, int]:
    # https://github.com/open-neuromorphic/expelliarmus  (evt2, evt3, dat)
    from expelliarmus import Wizard  # type: ignore
    import numpy as np
    wiz = Wizard(encoding=fmt)
    wiz.set_file(str(path))
    arr = wiz.read()
    total = len(arr)
    n_pos = int(np.count_nonzero(arr['p'] == 1))
    return total, n_pos


def _read_expelliarmus_chunked(path: str, fmt: str) -> tuple[int, int]:
    from expelliarmus import Wizard
    import numpy as np
    wiz = Wizard(encoding=fmt)
    wiz.set_file(str(path))
    total = 0
    n_pos = 0
    for chunk in wiz.read_chunk():
        total += len(chunk)
        n_pos += int(np.count_nonzero(chunk['p'] == 1))
    return total, n_pos


# NOTE: tonic (https://github.com/neuromorphs/tonic) has no standalone EVT/.raw
# reader -- it reads Prophesee data via `expelliarmus.Wizard` inside its dataset
# pipelines. Benchmarking it would just re-measure expelliarmus (already below,
# without the torchdata overhead), so it is intentionally not registered.


def _read_evlib(path: str, fmt: str) -> tuple[int, int]:
    # https://github.com/tallamjr/evlib  (Rust-backed). Auto-detects the format.
    import evlib  # type: ignore
    import polars as pl  # type: ignore
    import numpy as np
    lf = evlib.load_events(str(path))
    df = lf.collect(engine="streaming")
    total = len(df)
    p_array = df["polarity"].to_numpy()
    n_pos = int(np.sum(p_array == 1))
    return total, n_pos


def _read_openeb(path: str, fmt: str) -> tuple[int, int]:
    # OpenEB / Metavision SDK (not on PyPI -- see benchmarks/docker).
    from metavision_core.event_io import RawReader  # type: ignore
    import numpy as np
    reader = RawReader(str(path))
    n = 0
    n_pos = 0
    while not reader.is_done():
        chunk = reader.load_n_events(_CHUNK)
        n += len(chunk)
        if len(chunk) > 0:
            n_pos += int(np.count_nonzero(chunk['p']))
    return n, n_pos


def _read_evt3(path: str, fmt: str) -> tuple[int, int]:
    import evt3  # type: ignore
    import numpy as np
    if fmt != "evt3":
        raise ValueError(f"evt3 package does not support {fmt}")
    events = evt3.decode_file(str(path))
    total = len(events)
    n_pos = int(np.count_nonzero(events.p))
    return total, n_pos


def _read_event_vision_library(path: str, fmt: str) -> tuple[int, int]:
    # event-vision-library by shiba24 (installs as evlib, shadows Rust evlib)
    import evlib.codec.fileformat as ff  # type: ignore
    import numpy as np
    if fmt != "evt3":
        raise ValueError(f"event-vision-library benchmark snippet only supports evt3")
    it = ff.IteratorEvt3Event(str(path))
    total = 0
    n_pos = 0
    for chunk in it:
        total += len(chunk.x)
        n_pos += int(np.sum(chunk.p))
    return total, n_pos


@dataclass(frozen=True)
class Reader:
    name: str
    formats: tuple[str, ...]            # subset of ALL_FORMATS this library can read
    read: Callable[[str, str], tuple[int, int]]


#: Third-party readers benchmarked against evutils. evutils itself is measured
#: separately in test_read.py; with ``--benchmark-group-by=param:fmt`` all rows
#: for a given format line up in one table.
READERS = [
    Reader("expelliarmus-read_all", ("evt2", "evt3"), _read_expelliarmus),
    Reader("expelliarmus-chunked", ("evt2", "evt3"), _read_expelliarmus_chunked),
    Reader("evlib", ("evt2", "evt3"), _read_evlib),
    Reader("openeb", ALL_FORMATS, _read_openeb),
    Reader("evt3", ("evt3",), _read_evt3),
    Reader("event-vision-library", ("evt3",), _read_event_vision_library),
]
