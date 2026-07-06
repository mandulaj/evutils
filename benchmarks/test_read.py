"""Read-throughput benchmarks for the native EVT2 / EVT2.1 / EVT3 decoders.

Run with::

    pytest benchmarks/test_read.py

To compare evutils against the optional reference libraries side by side, group
the results by format::

    pytest benchmarks/ --benchmark-group-by=param:fmt
"""
import pytest

from typing import Any

from evutils.io import EventReader  # type: ignore

FORMATS = ["evt3", "evt2", "evt21"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_read_evutils(benchmark: Any, benchmark_rounds: int, real_event_files: dict[str, list[Any]], fmt: str) -> None:
    benchmark.group = f"read-{fmt}"
    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")
    ef = next((f for f in real_event_files[fmt] if 'hand' in f.path.name), real_event_files[fmt][0])

    def decode() -> int:
        total = 0
        with EventReader(ef.path) as reader:
            for chunk in reader:
                total += len(chunk)
        return total

    n = benchmark.pedantic(decode, rounds=benchmark_rounds, iterations=1, warmup_rounds=1)

    # Sanity: the benchmarked path decodes the expected number of events.
    assert n == ef.count
    benchmark.extra_info["n_events"] = n
    benchmark.extra_info["library"] = "evutils"
