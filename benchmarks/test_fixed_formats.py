"""Read/write throughput benchmarks for the fixed-record formats: DAT and AER.

DAT reuses the shared ``reference_events`` (1280x720 fits its 14-bit coords).
AER only has 9-bit coordinates and no timestamps, so it uses a separate small
(GenX320-sized) synthetic fixture.

    pytest benchmarks/test_fixed_formats.py

The read benchmarks decode a file generated once per session; the write
benchmarks measure encoding of the in-memory events. expelliarmus (if installed)
is compared on the DAT read.
"""
import numpy as np
import pytest

from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype

AER_N = 5_000_000


@pytest.fixture(scope="session")
def aer_events():
    """Small-resolution (GenX320) events for AER, which is 9-bit / timestamp-less."""
    rng = np.random.default_rng(0)
    ev = np.zeros(AER_N, dtype=Event_dtype)
    ev["x"] = rng.integers(0, 320, AER_N)
    ev["y"] = rng.integers(0, 320, AER_N)
    ev["p"] = rng.integers(0, 2, AER_N)
    return ev


@pytest.fixture(scope="session")
def dat_file(reference_events, tmp_path_factory):
    path = tmp_path_factory.mktemp("dat") / "ref.dat"
    with EventWriter(path) as writer:
        writer.write(reference_events)
    return path


@pytest.fixture(scope="session")
def aer_file(aer_events, tmp_path_factory):
    path = tmp_path_factory.mktemp("aer") / "ref.aer"
    with EventWriter(path) as writer:
        writer.write(aer_events)
    return path


def _read_all(path):
    return len(EventReader(path).read_all())


def _write(path, events):
    with EventWriter(path) as writer:
        writer.write(events)


# --------------------------------------------------------------------------- #
# DAT
# --------------------------------------------------------------------------- #
@pytest.mark.benchmark(group="read-dat")
def test_read_dat_evutils(benchmark, benchmark_rounds, dat_file):
    n = benchmark.pedantic(lambda: _read_all(dat_file), rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evutils", n_events=n)


@pytest.mark.benchmark(group="read-dat")
def test_read_dat_expelliarmus(benchmark, benchmark_rounds, dat_file):
    try:
        from expelliarmus import Wizard
    except ImportError as exc:
        pytest.skip(f"expelliarmus not available: {exc}")
    wizard = Wizard(encoding="dat")
    n = benchmark.pedantic(lambda: len(wizard.read(str(dat_file))),
                           rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="expelliarmus", n_events=n)


@pytest.mark.benchmark(group="write-dat")
def test_write_dat_evutils(benchmark, benchmark_rounds, reference_events, tmp_path):
    out = tmp_path / "out.dat"
    benchmark.pedantic(lambda: _write(out, reference_events),
                       rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert len(EventReader(out).read_all()) == len(reference_events)
    benchmark.extra_info.update(library="evutils", n_events=len(reference_events))


@pytest.mark.benchmark(group="write-dat")
def test_write_dat_expelliarmus(benchmark, benchmark_rounds, reference_events, tmp_path):
    try:
        from expelliarmus import Wizard
    except ImportError as exc:
        pytest.skip(f"expelliarmus not available: {exc}")
    wizard = Wizard(encoding="dat")
    out = tmp_path / "out.dat"
    benchmark.pedantic(lambda: wizard.save(str(out), reference_events),
                       rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert len(wizard.read(str(out))) == len(reference_events)
    benchmark.extra_info.update(library="expelliarmus", n_events=len(reference_events))


# --------------------------------------------------------------------------- #
# AER
# --------------------------------------------------------------------------- #
@pytest.mark.benchmark(group="read-aer")
def test_read_aer_evutils(benchmark, benchmark_rounds, aer_file):
    n = benchmark.pedantic(lambda: _read_all(aer_file), rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    assert n > 0
    benchmark.extra_info.update(library="evutils", n_events=n)


@pytest.mark.benchmark(group="write-aer")
def test_write_aer_evutils(benchmark, benchmark_rounds, aer_events, tmp_path):
    out = tmp_path / "out.aer"
    benchmark.pedantic(lambda: _write(out, aer_events),
                       rounds=benchmark_rounds, iterations=1, warmup_rounds=1)
    benchmark.extra_info.update(library="evutils", n_events=len(aer_events))
