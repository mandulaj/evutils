"""Benchmark fixtures.

``EventFile`` and ``real_event_files`` are duplicated from ``tests/conftest.py``
on purpose: pytest only shares conftest fixtures downward, and ``benchmarks/``
is a sibling of ``tests/``, so there is no clean way to reuse them without a
repo-root conftest. Keep the two copies in sync.
"""
import subprocess
from collections import namedtuple

import numpy as np
import pytest

from evutils.io import EventReader

#: (see tests/conftest.py) path + reference OpenEB event count for a recording.
EventFile = namedtuple("EventFile", ["path", "count"])

#: Cap on how many events are held in memory for the write benchmarks. Large
#: enough for a stable throughput measurement, small enough to stay light.
N_REFERENCE = 5_000_000


@pytest.fixture(scope='session')
def real_event_files(request):
    """Real Prophesee recordings (downloaded + cached on first use).

    Returns ``{format: EventFile(path, count)}``. Duplicated from
    tests/conftest.py -- keep in sync.
    """
    event_counts = {
        'evt3': 33494595,
        'evt21': 8214341,
        'evt2': 33494595,
    }

    temp_dir = request.config.cache.mkdir("event_files")

    filenames = {
        'evt3': "hand_evt3.raw",
        'evt21': "hand_evt21.raw",
        'evt2': "hand_evt2.raw",
    }
    paths = {fmt: temp_dir / name for fmt, name in filenames.items()}

    for key, path in paths.items():
        if not path.exists():
            tar_url = "https://drive.usercontent.google.com/download?id=18LbJljYr5dqBmbrkm0EJs0EcddCqMKTv&confirm=t"
            tar_file = temp_dir / "hand.tar.gz"
            subprocess.run(["wget", str(tar_url), "-O", str(tar_file)])
            subprocess.run(["tar", "zxv", "-C", str(temp_dir), "-f", str(tar_file)])
            break

    return {
        fmt: EventFile(path=paths[fmt], count=event_counts[fmt])
        for fmt in filenames
    }


@pytest.fixture(scope='session')
def reference_events(real_event_files):
    """First ``N_REFERENCE`` events of the real EVT3 recording as an AoS array.

    Decoded once and reused as the payload for every write benchmark, so the
    write benchmarks measure encoding cost only (not decoding).
    """
    parts = []
    total = 0
    with EventReader(real_event_files['evt3'].path, n_events=1_000_000) as reader:
        for chunk in reader:
            parts.append(chunk.to_aos())
            total += len(chunk)
            if total >= N_REFERENCE:
                break
    return np.concatenate(parts)[:N_REFERENCE]


def pytest_addoption(parser):
    parser.addoption(
        "--rounds", action="store", default=4, type=int, help="Number of benchmark rounds"
    )

@pytest.fixture(scope="session")
def benchmark_rounds(request):
    return request.config.getoption("--rounds")
