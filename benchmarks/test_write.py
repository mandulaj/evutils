"""Write-throughput benchmarks for the EVT2 / EVT2.1 / EVT3 encoders.

Run with::

    pytest benchmarks/test_write.py

The payload is the shared ``reference_events`` array, so all three formats
encode the same events. ``warmup_rounds=1`` absorbs the numba JIT compile of
each encoder so it isn't counted in the timed rounds.
"""
import pytest

from evutils.io import EventWriter

FORMATS = ["evt3", "evt2", "evt21"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_write_evutils(benchmark, reference_events, tmp_path, fmt):
    events = reference_events
    out = tmp_path / f"out_{fmt}.raw"

    def write():
        with EventWriter(out, format=fmt) as writer:
            writer.write(events)

    benchmark.pedantic(write, rounds=3, iterations=1, warmup_rounds=1)

    assert out.stat().st_size > 0
    benchmark.extra_info["n_events"] = len(events)
    benchmark.extra_info["library"] = "evutils"
