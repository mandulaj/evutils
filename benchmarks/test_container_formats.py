"""Read/write throughput benchmarks for the container formats: NPZ, HDF5,
CSV and AEDAT4 (read only -- there is no AEDAT writer yet).

All use the shared 5M-event ``reference_events`` payload (real EVT3 events).
The AEDAT4 file is synthesized with the spec-accurate builder shared with the
correctness tests. Where a third-party library can read the same file
(evlib: HDF5, AEDAT4), it is benchmarked on it for comparison; those
benchmarks skip if the library is missing or rejects the file.

    pytest benchmarks/test_container_formats.py --benchmark-group-by=group
"""
import numpy as np
import pytest

from evutils.io import EventReader, EventWriter

from aedat_synth import make_aedat4


@pytest.fixture(scope="session")
def npz_file(reference_events, tmp_path_factory):
    path = tmp_path_factory.mktemp("npz") / "ref.npz"
    with EventWriter(path) as w:
        w.write(reference_events)
    return path


@pytest.fixture(scope="session")
def hdf5_file(reference_events, tmp_path_factory):
    path = tmp_path_factory.mktemp("hdf5") / "ref.h5"
    with EventWriter(path, width=1280, height=720) as w:
        w.write(reference_events)
    return path


@pytest.fixture(scope="session")
def csv_file(reference_events, tmp_path_factory):
    path = tmp_path_factory.mktemp("csv") / "ref.csv"
    with EventWriter(path) as w:
        w.write(reference_events)
    return path


@pytest.fixture(scope="session")
def aedat4_file(reference_events, tmp_path_factory):
    path = tmp_path_factory.mktemp("aedat") / "ref.aedat4"
    ev = reference_events
    path.write_bytes(
        make_aedat4(ev["t"], ev["x"], ev["y"], ev["p"], events_per_packet=65536)
    )
    return path


def _read_all(path):
    with EventReader(path) as reader:
        return len(reader.read_all())


def _write(path, events):
    with EventWriter(path) as writer:
        writer.write(events)


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


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
@pytest.mark.benchmark(group="read-npz")
def test_read_npz_evutils(benchmark, benchmark_rounds, npz_file):
    n = benchmark.pedantic(lambda: _read_all(npz_file), rounds=benchmark_rounds,
                           iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evutils", n_events=n)


@pytest.mark.benchmark(group="read-hdf5")
def test_read_hdf5_evutils(benchmark, benchmark_rounds, hdf5_file):
    n = benchmark.pedantic(lambda: _read_all(hdf5_file), rounds=benchmark_rounds,
                           iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evutils", n_events=n)


@pytest.mark.benchmark(group="read-hdf5")
def test_read_hdf5_evlib(benchmark, benchmark_rounds, hdf5_file):
    n = benchmark.pedantic(lambda: _evlib_count(hdf5_file), rounds=benchmark_rounds,
                           iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evlib", n_events=n)


@pytest.mark.benchmark(group="read-csv")
def test_read_csv_evutils(benchmark, benchmark_rounds, csv_file):
    n = benchmark.pedantic(lambda: _read_all(csv_file), rounds=benchmark_rounds,
                           iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evutils", n_events=n)


@pytest.mark.benchmark(group="read-aedat4")
def test_read_aedat4_evutils(benchmark, benchmark_rounds, aedat4_file):
    n = benchmark.pedantic(lambda: _read_all(aedat4_file), rounds=benchmark_rounds,
                           iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evutils", n_events=n)


@pytest.mark.benchmark(group="read-aedat4")
def test_read_aedat4_evlib(benchmark, benchmark_rounds, aedat4_file):
    n = benchmark.pedantic(lambda: _evlib_count(aedat4_file), rounds=benchmark_rounds,
                           iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evlib", n_events=n)


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
@pytest.mark.benchmark(group="write-npz")
def test_write_npz_evutils(benchmark, benchmark_rounds, reference_events, tmp_path):
    out = tmp_path / "out.npz"
    benchmark.pedantic(lambda: _write(out, reference_events),
                       rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert _read_all(out) == len(reference_events)
    benchmark.extra_info.update(library="evutils", n_events=len(reference_events))


@pytest.mark.benchmark(group="write-hdf5")
def test_write_hdf5_evutils(benchmark, benchmark_rounds, reference_events, tmp_path):
    out = tmp_path / "out.h5"
    benchmark.pedantic(lambda: _write(out, reference_events),
                       rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert _read_all(out) == len(reference_events)
    benchmark.extra_info.update(library="evutils", n_events=len(reference_events))


@pytest.mark.benchmark(group="write-csv")
def test_write_csv_evutils(benchmark, benchmark_rounds, reference_events, tmp_path):
    out = tmp_path / "out.csv"
    benchmark.pedantic(lambda: _write(out, reference_events),
                       rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert _read_all(out) == len(reference_events)
    benchmark.extra_info.update(library="evutils", n_events=len(reference_events))
