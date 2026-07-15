"""Optional cross-library read benchmarks (run on demand).

Each third-party reader lives in ``readers.py``. A benchmark **skips
automatically** when the library isn't importable, so this file is inert by
default. Enable libraries by installing them::

    pip install evutils[compare]     # expelliarmus, tonic, evlib
    # OpenEB / Metavision: not on PyPI -- see benchmarks/docker/

Then line every library up against evutils per format::

    pytest benchmarks/ --benchmark-group-by=group

evutils itself is benchmarked in test_read.py; it shares the same ``read-<fmt>``
group, so it appears in the same grouped table as the readers below. (Grouping
by ``group`` rather than ``param:fmt`` also keeps reads and writes -- from
test_write.py -- in separate buckets when the whole suite is run together.)
"""
import pytest

from typing import Any

from readers import ALL_FORMATS, READERS  # type: ignore


@pytest.mark.parametrize("fmt", ALL_FORMATS)
@pytest.mark.parametrize("reader", READERS, ids=lambda r: r.name)
def test_read_compare(benchmark: Any, benchmark_rounds: int, real_event_files: dict[str, list[Any]], reader: Any, fmt: str) -> None:
    if fmt not in reader.formats:
        pytest.skip(f"{reader.name} does not support {fmt}")

    benchmark.group = f"read-{fmt}"
    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")
    ef = next((f for f in real_event_files[fmt] if 'hand' in f.path.name), real_event_files[fmt][0])

    try:
        n, n_pos = benchmark.pedantic(
            lambda: reader.read(ef.path, fmt),
            rounds=benchmark_rounds, iterations=1, warmup_rounds=1,
        )
    except ImportError as exc:
        pytest.skip(f"{reader.name} not available: {exc}")

    assert n > 0
    benchmark.extra_info.update(library=reader.name, fmt=fmt, n_events=n, n_pos=n_pos)
