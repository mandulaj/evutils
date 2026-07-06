"""Shared fixtures for the test suite (visible to everything under ``tests/``).

The reference-data fixtures (``test_events``, ``real_event_files``) are also
needed by the benchmarks. pytest only shares conftest fixtures *downward*, and
``benchmarks/`` is a sibling of ``tests/``, so those two fixtures are duplicated
in ``benchmarks/conftest.py`` -- keep them in sync.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from evutils.types import Event_dtype

# Add tests dir to path to import conftest_utils
sys.path.append(str(Path(__file__).parent))
from conftest_utils import download_and_extract_gdrive, load_event_files # type: ignore


from typing import Any
@pytest.fixture(scope='session')
def test_events() -> Any:
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
def real_event_files(request: Any) -> Any:
    """Real Prophesee recordings (downloaded + cached on first use).

    Returns ``{format: EventFile(path, count)}`` where ``count`` is the
    reference OpenEB event count (from the JSON sidecars).

    Set ``EVUTILS_BENCH_DATA`` to a directory that already contains the
    extracted recordings + JSON sidecars to skip the download (offline
    environments, e.g. the OpenEB Docker container).
    """
    import os
    override = os.environ.get("EVUTILS_BENCH_DATA")
    if override:
        data_dir = Path(override)
        if not data_dir.is_dir():
            pytest.skip(f"EVUTILS_BENCH_DATA={override} is not a directory")
        return load_event_files(data_dir)

    temp_dir = request.config.cache.mkdir("event_files")

    filenames = {
        'evt3': "hand_evt3.raw",
        'evt21': "hand_evt21.raw",
        'evt2': "hand_evt2.raw",
    }
    paths = {fmt: temp_dir / name for fmt, name in filenames.items()}

    for key, path in paths.items():
        if not path.exists():
            download_and_extract_gdrive("1uhOsWbp2o3CktsHrFkzGCNFbx0bQLsct", temp_dir, "hand.tar.zst")
            break

    return load_event_files(temp_dir)
