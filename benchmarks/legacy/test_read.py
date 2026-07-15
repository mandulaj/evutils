"""Read-throughput benchmarks for the native EVT2 / EVT2.1 / EVT3 decoders.

Run with::

    pytest benchmarks/test_read.py

To compare evutils against the optional reference libraries side by side, group
the results by the benchmark ``group`` attribute (``read-<fmt>`` / ``write-<fmt>``)
so reads and writes stay in separate buckets::

    pytest benchmarks/ --benchmark-group-by=group

(Grouping by ``param:fmt`` instead would merge the read and write benchmarks of
each format, since neither is distinguished by a parameter.)
"""
import pytest

from typing import Any

from evutils.io import EventReader

FORMATS = ["evt3", "evt2", "evt21"]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("read_mode", ["sync", "async", "read_all"])
def test_read_evutils(benchmark: Any, benchmark_rounds: int, real_event_files: dict[str, list[Any]], fmt: str, read_mode: str) -> None:
    benchmark.group = f"read-{fmt}"
    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")
    ef = next((f for f in real_event_files[fmt] if 'hand' in f.path.name), real_event_files[fmt][0])

    def decode() -> tuple[int, int]:
        total = 0
        n_pos = 0
        import numpy as np
        
        if read_mode == "read_all":
            with EventReader(ef.path) as reader:
                events = reader.read_all()
                if isinstance(events, tuple):
                    events = events[0]
                total = len(events)
                n_pos = int(np.count_nonzero(events.p == 1))
        else:
            is_async = (read_mode == "async")
            with EventReader(ef.path, async_read=is_async) as reader:
                for chunk in reader:
                    total += len(chunk)
                    n_pos += int(np.count_nonzero(chunk.p == 1))
        return total, n_pos

    n, n_pos = benchmark.pedantic(decode, rounds=benchmark_rounds, iterations=1, warmup_rounds=1)

    # Sanity: the benchmarked path decodes the expected number of events.
    assert n == ef.count
    benchmark.extra_info["n_events"] = n
    benchmark.extra_info["library"] = f"evutils-{read_mode}"
