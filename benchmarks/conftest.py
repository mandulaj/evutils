"""Benchmark fixtures.

``EventFile`` and ``real_event_files`` are duplicated from ``tests/conftest.py``
on purpose: pytest only shares conftest fixtures downward, and ``benchmarks/``
is a sibling of ``tests/``, so there is no clean way to reuse them without a
repo-root conftest. Keep the two copies in sync.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from typing import Any, cast

from evutils.io import EventReader  # type: ignore

sys.path.append(str(Path(__file__).parent.parent / "tests"))
from conftest_utils import EventFile, download_and_extract_gdrive, load_event_files  # type: ignore

#: Cap on how many events are held in memory for the write benchmarks. Large
#: enough for a stable throughput measurement, small enough to stay light.
N_REFERENCE = 5_000_000


@pytest.fixture(scope='session')
def real_event_files(request: Any, benchmark_dataset: str) -> dict[str, list[Any]]:
    """Real Prophesee recordings (downloaded + cached on first use).

    Returns ``{format: [EventFile(path, count)]}``.

    Set ``EVUTILS_BENCH_DATA`` to a directory that already contains the
    extracted recordings (+ JSON metadata) to skip the download entirely --
    useful in offline environments such as the OpenEB benchmark container
    (mount the host cache there).
    """
    import os
    override = os.environ.get("EVUTILS_BENCH_DATA")
    if override:
        data_dir = Path(override)
        if not data_dir.is_dir():
            pytest.skip(f"EVUTILS_BENCH_DATA={override} is not a directory")
        return cast(dict[str, list[Any]], load_event_files(data_dir))

    if benchmark_dataset == "large":
        temp_dir = request.config.cache.mkdir("event_files_huge")
        file_id = "1QPuilR1VD0rKyhVOlu1Y-HtkO4ZBc2Cz"
        
        json_files = list(temp_dir.glob("*.json"))
        if not json_files:
            download_and_extract_gdrive(file_id, temp_dir, "huge.tar.zst")
            
        return cast(dict[str, list[Any]], load_event_files(temp_dir))
    else:
        temp_dir = request.config.cache.mkdir("event_files")
        file_id = "1uhOsWbp2o3CktsHrFkzGCNFbx0bQLsct"
        filenames = {
            'evt3': "hand_evt3.raw",
            'evt21': "hand_evt21.raw",
            'evt2': "hand_evt2.raw",
        }
        paths = {fmt: temp_dir / name for fmt, name in filenames.items()}

        for key, path in paths.items():
            if not path.exists():
                download_and_extract_gdrive(file_id, temp_dir, "hand.tar.zst")
                break
                
        return cast(dict[str, list[Any]], load_event_files(temp_dir))


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
    parser.addoption(
        "--rounds", action="store", default=4, type=int, help="Number of benchmark rounds"
    )
    parser.addoption(
        "--dataset", action="store", default="small", choices=["small", "large"], 
        help="Dataset size to use for benchmarks ('small' or 'large')"
    )

@pytest.fixture(scope="session")
def benchmark_rounds(request: Any) -> int:
    return int(request.config.getoption("--rounds"))

@pytest.fixture(scope="session")
def benchmark_dataset(request: Any) -> str:
    return str(request.config.getoption("--dataset"))


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

    sys.path.append(str(Path(__file__).parent.parent / "tests"))
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
