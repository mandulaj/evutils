"""Shared fixtures for the test suite (visible to everything under ``tests/``).

The reference-data fixtures (``test_events``, ``real_event_files``) are also
needed by the benchmarks. pytest only shares conftest fixtures *downward*, and
``benchmarks/`` is a sibling of ``tests/``, so those two fixtures are duplicated
in ``benchmarks/conftest.py`` -- keep them in sync.
"""
import subprocess
from collections import namedtuple

import numpy as np
import pytest

from evutils.types import Event_dtype

#: A real reference recording: its path plus the event count reported by the
#: reference OpenEB implementation (implementations differ slightly, so counts
#: are expected to be close but not necessarily identical).
EventFile = namedtuple("EventFile", ["path", "count"])


@pytest.fixture(scope='session')
def test_events():
    """Small synthetic, time-sorted event array for correctness tests."""
    N_EVENTS = 1000
    np.random.seed(42)

    test_events = np.zeros(N_EVENTS, dtype=Event_dtype)
    test_events['t'] = np.random.randint(0, 10000, N_EVENTS)
    test_events.sort(order='t')
    test_events['x'] = np.random.randint(0, 1280, N_EVENTS)
    test_events['y'] = np.random.randint(0, 720, N_EVENTS)
    test_events['p'] = np.random.randint(0, 2, N_EVENTS)

    return test_events


@pytest.fixture(scope='session')
def real_event_files(request):
    """Real Prophesee recordings (downloaded + cached on first use).

    Returns ``{format: EventFile(path, count)}`` where ``count`` is the
    reference OpenEB event count.
    """
    # Reference event counts from the OpenEB implementation.
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
            # Download + extract the reference recordings on first use.
            tar_url = "https://drive.usercontent.google.com/download?id=18LbJljYr5dqBmbrkm0EJs0EcddCqMKTv&confirm=t"
            tar_file = temp_dir / "hand.tar.gz"
            subprocess.run(["wget", str(tar_url), "-O", str(tar_file)])
            subprocess.run(["tar", "zxv", "-C", str(temp_dir), "-f", str(tar_file)])
            break

    return {
        fmt: EventFile(path=paths[fmt], count=event_counts[fmt])
        for fmt in filenames
    }
