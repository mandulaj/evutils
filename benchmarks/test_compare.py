"""Optional cross-library read benchmarks (run on demand).

Each third-party reader lives in ``readers.py``. A benchmark **skips
automatically** when the library isn't importable, so this file is inert by
default. Enable libraries by installing them::

    pip install evutils[compare]     # expelliarmus, tonic, evlib
    # OpenEB / Metavision: not on PyPI -- see benchmarks/docker/

Then line every library up against evutils per format::

    pytest benchmarks/ --benchmark-group-by=param:fmt

evutils itself is benchmarked in test_read.py; it shares the ``fmt`` param, so
it appears in the same grouped table as the readers below.
"""
import pytest

from readers import ALL_FORMATS, READERS


@pytest.mark.parametrize("fmt", ALL_FORMATS)
@pytest.mark.parametrize("reader", READERS, ids=lambda r: r.name)
def test_read_compare(benchmark, benchmark_rounds, real_event_files, reader, fmt):
    if fmt not in reader.formats:
        pytest.skip(f"{reader.name} does not support {fmt}")

    benchmark.group = f"read-{fmt}"
    ef = real_event_files[fmt]

    try:
        n = benchmark.pedantic(
            lambda: reader.read(ef.path, fmt),
            rounds=benchmark_rounds, iterations=1, warmup_rounds=1,
        )
    except ImportError as exc:
        pytest.skip(f"{reader.name} not available: {exc}")

    assert n > 0
    benchmark.extra_info.update(library=reader.name, fmt=fmt, n_events=n)
