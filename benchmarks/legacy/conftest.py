"""Benchmark fixtures.

The download/discovery helpers live in ``tests/conftest_utils.py`` and are
shared with ``tests/conftest.py`` (pytest only shares conftest *fixtures*
downward, and ``benchmarks/`` is a sibling of ``tests/``, so the fixtures
themselves are re-declared here -- but they call the same underlying helpers,
so the two cannot drift on data source, caching, or metadata).

The ``--dataset {small,normal,large}`` option is registered in the repo-root
``conftest.py`` and shared with the test suite; the default is ``normal`` (the
same GitHub-release tarball the tests use).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from typing import Any, cast

from evutils.io import EventReader

sys.path.append(str(Path(__file__).parent.parent.parent / "tests"))
from conftest_utils import (  # type: ignore
    fetch_real_event_files_for,
    load_event_files,
    register_dataset_option,
)

#: Cap on how many events are held in memory for the write benchmarks. Large
#: enough for a stable throughput measurement, small enough to stay light.
N_REFERENCE = 5_000_000


@pytest.fixture(scope='session')
def real_event_files(request: Any, dataset_size: str) -> dict[str, list[Any]]:
    """Real Prophesee recordings (downloaded + cached on first use).

    Returns ``{format: [EventFile(path, count, metadata), ...]}`` for the tier
    selected by ``--dataset`` (default ``normal`` -- identical to the test
    suite, sharing the same cache dir + sentinel so it downloads at most once
    for both). Each tier extracts into its own cache subdir.

    Set ``EVUTILS_BENCH_DATA`` to a directory that already contains the
    extracted recordings (+ JSON metadata) to skip the download entirely --
    useful in offline environments such as the OpenEB benchmark container
    (mount the host cache there).
    """
    override = os.environ.get("EVUTILS_BENCH_DATA")
    if override:
        data_dir = Path(override)
        if not data_dir.is_dir():
            pytest.skip(f"EVUTILS_BENCH_DATA={override} is not a directory")
        return cast(dict[str, list[Any]], load_event_files(data_dir))

    return cast(dict[str, list[Any]],
                fetch_real_event_files_for(dataset_size, request.config.cache))


@pytest.fixture(scope='session')
def reference_events(real_event_files: dict[str, list[Any]]) -> Any:
    """First ``N_REFERENCE`` events of the real EVT3 recording as an AoS array.

    Decoded once and reused as the payload for every write benchmark, so the
    write benchmarks measure encoding cost only (not decoding).
    """
    parts = []
    total = 0
    if 'evt3' not in real_event_files or not real_event_files['evt3']:
        pytest.skip("No evt3 files available for reference events")
    # Use the hand_evt3.raw file for reference events
    ef = next((f for f in real_event_files['evt3'] if 'hand' in f.path.name), real_event_files['evt3'][0])
    with EventReader(ef.path, n_events=1_000_000) as reader:
        for chunk in reader:
            parts.append(chunk.to_aos())
            total += len(chunk)
            if total >= N_REFERENCE:
                break
    return np.concatenate(parts)[:N_REFERENCE]


def pytest_addoption(parser: Any) -> None:
    # --dataset is shared with the test suite (registered idempotently so it
    # works whether tests/ or benchmarks/ is collected first). --rounds is
    # benchmark-only.
    register_dataset_option(parser)
    parser.addoption(
        "--rounds", action="store", default=4, type=int, help="Number of benchmark rounds"
    )

@pytest.fixture(scope="session")
def benchmark_rounds(request: Any) -> int:
    return int(request.config.getoption("--rounds"))

@pytest.fixture(scope="session")
def dataset_size(request: Any) -> str:
    return str(request.config.getoption("--dataset"))


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    """Append a throughput table (M events/s) to the benchmark output.

    pytest-benchmark reports wall-clock time, which is only comparable across
    benchmarks that process the *same* number of events. Every benchmark here
    instead records the event count it handled (``extra_info["n_events"]``), so
    the scale-invariant metric is throughput = events / time: it stays directly
    comparable across recordings of wildly different sizes (33 M vs 2.3 B
    events) and across formats/libraries. Both reads and writes count events,
    so both get a rate. ``peak`` uses the fastest round (min time, matching
    pytest-benchmark's headline); ``mean`` uses the mean round.
    """
    session = getattr(config, "_benchmarksession", None)
    if session is None or not getattr(session, "benchmarks", None):
        return

    rows = []
    for bench in session.benchmarks:
        # A benchmark that was skipped (or registered but never run) has no
        # timing samples; as_dict() would raise on min([]). Skip those.
        stats_obj = getattr(bench, "stats", None)
        if stats_obj is None or not getattr(stats_obj, "data", None):
            continue
        d = bench.as_dict()
        extra = d.get("extra_info") or {}
        stats = d.get("stats") or {}
        n = extra.get("n_events")
        mean = stats.get("mean")
        best = stats.get("min")
        if not n or not mean or not best:
            continue
        rows.append((
            d.get("group") or "",
            extra.get("library") or d.get("name", ""),
            int(n),
            n / best / 1e6,   # peak throughput (fastest round)
            n / mean / 1e6,   # mean throughput
        ))
    if not rows:
        return

    rows.sort(key=lambda r: (r[0], -r[3]))  # group, then peak desc
    tw = terminalreporter
    tw.write_line("")
    tw.write_line("----------- Throughput: M events/s (higher is better) -----------")
    tw.write_line(f"{'group':17s} {'library':24s} {'events':>13s} {'peak':>8s} {'mean':>8s}")
    for group, lib, n, peak, mean_tp in rows:
        tw.write_line(f"{group:17s} {lib:24s} {n:>13,d} {peak:>8.1f} {mean_tp:>8.1f}")


@pytest.hookimpl(hookwrapper=True)
def pytest_benchmark_group_stats(config: Any, benchmarks: list[dict[str, Any]], group_by: str) -> Any:
    """Guard against a pytest-benchmark crash with ``--benchmark-group-by=param:<name>``.

    Benchmarks collected from non-parametrized tests have ``params = None``,
    which the plugin's param-grouping code cannot handle (TypeError) -- and the
    crash also swallows the whole error/summary report. Give such benchmarks an
    explicit null param instead, so they group under ``<name>=None``.
    """
    if isinstance(group_by, str) and group_by.startswith("param:"):
        names = group_by.split(":", 1)[1].split(",")
        for bench in benchmarks:
            if bench.get("params") is None:
                bench["params"] = {}
            for name in names:
                bench["params"].setdefault(name, None)
    yield


@pytest.fixture(scope="session")
def uniform_files(reference_events: Any, tmp_path_factory: Any) -> dict[str, Any]:
    """The same ``N_REFERENCE`` real events transcoded into every format.

    All per-format read/write benchmarks (benchmarks/test_formats.py) operate
    on these files, so throughput numbers are comparable across formats: same
    events, same count. AER is the one lossy exception -- it has no timestamps
    and 9-bit coordinates, so its file carries the same events with coordinates
    masked to 0..511 (the event count is identical).
    """
    from evutils.io import EventWriter

    sys.path.append(str(Path(__file__).parent.parent.parent / "tests"))
    from aedat_synth import make_aedat4  # type: ignore

    d = tmp_path_factory.mktemp("uniform")
    ev = reference_events
    files = {}

    for fmt in ("evt3", "evt21", "evt2"):
        files[fmt] = d / f"ref_{fmt}.raw"
        with EventWriter(files[fmt], width=1280, height=720, format=fmt) as w:
            w.write(ev)

    for fmt, name in (("dat", "ref.dat"), ("npz", "ref.npz"),
                      ("hdf5", "ref.h5"), ("csv", "ref.csv")):
        files[fmt] = d / name
        with EventWriter(files[fmt], width=1280, height=720) as w:
            w.write(ev)

    aer = ev.copy()
    aer["x"] &= 0x1FF
    aer["y"] &= 0x1FF
    files["aer"] = d / "ref.aer"
    with EventWriter(files["aer"]) as w:
        w.write(aer)

    files["aedat4"] = d / "ref.aedat4"
    files["aedat4"].write_bytes(
        make_aedat4(ev["t"], ev["x"], ev["y"], ev["p"], events_per_packet=65536)
    )
    return files
