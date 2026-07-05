"""Read-throughput benchmarks for the native EVT2 / EVT2.1 / EVT3 decoders.

Run with::

    pytest benchmarks/test_read.py

To compare evutils against the optional reference libraries side by side, group
the results by format::

    pytest benchmarks/ --benchmark-group-by=param:fmt
"""
import pytest

from evutils.io import EventReader

FORMATS = ["evt3", "evt2", "evt21"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_read_evutils(benchmark, benchmark_rounds, real_event_files, fmt):
    benchmark.group = f"read-{fmt}"
    ef = real_event_files[fmt]

    def decode():
        return len(EventReader(ef.path).read_all())

    n = benchmark.pedantic(decode, rounds=benchmark_rounds, iterations=1, warmup_rounds=1)

    # Sanity: the benchmarked path decodes the expected number of events.
    assert n == ef.count
    benchmark.extra_info["n_events"] = n
    benchmark.extra_info["library"] = "evutils"
