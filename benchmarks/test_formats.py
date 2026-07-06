"""Uniform per-format read/write benchmarks.

Every format is benchmarked on **the same payload**: the first ``N_REFERENCE``
events of the real EVT3 recording, transcoded once per session into every
format (see the ``uniform_files`` fixture). Throughput numbers are therefore
directly comparable across formats.

Groups:

* ``read-<fmt>`` / ``write-<fmt>`` for dat, aer, npz, hdf5, csv, aedat4.
* ``read-<fmt>-uniform`` for the EVT formats, so they do not mix with the
  whole-recording real-file benchmarks in test_read.py / test_compare.py
  (those decode ~7x more events).

Third-party libraries are compared on the identical file where they can read
it (expelliarmus: DAT; evlib: HDF5, AEDAT4 -- these skip with the reason when
the installed build rejects the file).

    pytest benchmarks/test_formats.py --benchmark-group-by=group
"""
import numpy as np
import pytest

from evutils.io import EventReader, EventWriter

# fmt -> benchmark group (EVT formats get a -uniform suffix; see module docs).
READ_FORMATS = {
    "evt3": "read-evt3-uniform",
    "evt21": "read-evt21-uniform",
    "evt2": "read-evt2-uniform",
    "dat": "read-dat",
    "aer": "read-aer",
    "npz": "read-npz",
    "hdf5": "read-hdf5",
    "csv": "read-csv",
    "aedat4": "read-aedat4",
}

# Writable formats (EVT writes are covered by test_write.py on the same
# payload already; AEDAT has no writer yet).
WRITE_FORMATS = ("dat", "aer", "npz", "hdf5", "csv")
_SUFFIX = {"dat": ".dat", "aer": ".aer", "npz": ".npz", "hdf5": ".h5", "csv": ".csv"}


def _read_all(path):
    with EventReader(path) as reader:
        return len(reader.read_all())


# --------------------------------------------------------------------------- #
# Reads (evutils, every format, same events)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", sorted(READ_FORMATS))
def test_read_uniform_evutils(benchmark, benchmark_rounds, uniform_files, fmt):
    benchmark.group = READ_FORMATS[fmt]
    n = benchmark.pedantic(lambda: _read_all(uniform_files[fmt]),
                           rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evutils", n_events=n, fmt=fmt)


# --------------------------------------------------------------------------- #
# Writes (evutils, same events)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", sorted(WRITE_FORMATS))
def test_write_uniform_evutils(benchmark, benchmark_rounds, reference_events, tmp_path, fmt):
    benchmark.group = f"write-{fmt}"
    ev = reference_events
    if fmt == "aer":
        ev = ev.copy()
        ev["x"] &= 0x1FF
        ev["y"] &= 0x1FF
    out = tmp_path / f"out{_SUFFIX[fmt]}"

    def write():
        with EventWriter(out, width=1280, height=720) as w:
            w.write(ev)

    benchmark.pedantic(write, rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert _read_all(out) == len(ev)
    benchmark.extra_info.update(library="evutils", n_events=len(ev), fmt=fmt)


# --------------------------------------------------------------------------- #
# Third-party comparisons on the identical files
# --------------------------------------------------------------------------- #
# NOTE: keep every benchmark parametrized with `fmt` -- the suite is commonly
# run with --benchmark-group-by=param:fmt, and pytest-benchmark crashes on
# benchmarks without params (see also the guard hook in conftest.py).
@pytest.mark.benchmark(group="read-dat")
@pytest.mark.parametrize("fmt", ["dat"])
def test_read_dat_expelliarmus(benchmark, benchmark_rounds, uniform_files, fmt):
    try:
        from expelliarmus import Wizard
    except ImportError as exc:
        pytest.skip(f"expelliarmus not available: {exc}")
    wizard = Wizard(encoding=fmt)
    n = benchmark.pedantic(lambda: len(wizard.read(str(uniform_files[fmt]))),
                           rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="expelliarmus", n_events=n, fmt=fmt)


@pytest.mark.benchmark(group="write-dat")
@pytest.mark.parametrize("fmt", ["dat"])
def test_write_dat_expelliarmus(benchmark, benchmark_rounds, reference_events, tmp_path, fmt):
    try:
        from expelliarmus import Wizard
    except ImportError as exc:
        pytest.skip(f"expelliarmus not available: {exc}")
    wizard = Wizard(encoding=fmt)
    out = tmp_path / "out.dat"
    benchmark.pedantic(lambda: wizard.save(str(out), reference_events),
                       rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert len(wizard.read(str(out))) == len(reference_events)
    benchmark.extra_info.update(library="expelliarmus", n_events=len(reference_events), fmt=fmt)


def _evlib_count(path):
    """Read with evlib, skipping if it is absent or rejects the file."""
    try:
        import evlib
        import polars as pl
    except ImportError as exc:
        pytest.skip(f"evlib not available: {exc}")
    try:
        df = evlib.load_events(str(path))
        if hasattr(df, "collect"):
            try:
                return df.select(pl.len()).collect(engine="streaming").item()
            except TypeError:
                return df.select(pl.len()).collect(streaming=True).item()
        return len(df)
    except Exception as exc:  # noqa: BLE001 - any rejection is a skip, not a failure
        pytest.skip(f"evlib could not read {path.name}: {exc}")


@pytest.mark.parametrize("fmt", ["hdf5", "aedat4"])
def test_read_uniform_evlib(benchmark, benchmark_rounds, uniform_files, fmt):
    benchmark.group = READ_FORMATS[fmt]
    n = benchmark.pedantic(lambda: _evlib_count(uniform_files[fmt]),
                           rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evlib", n_events=n, fmt=fmt)
