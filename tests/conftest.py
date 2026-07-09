"""Shared fixtures for the test suite (visible to everything under ``tests/``).

The reference-data fixtures (``test_events``, ``real_event_files``,
``real_event_file``) are also needed by the benchmarks. pytest only shares
conftest fixtures *downward*, and ``benchmarks/`` is a sibling of ``tests/``,
so those fixtures are duplicated in ``benchmarks/conftest.py`` -- keep them in
sync.
"""
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evutils.types import Event_dtype

# Add tests dir to path to import conftest_utils
sys.path.append(str(Path(__file__).parent))
from conftest_utils import download_and_extract_github, load_event_files # type: ignore

#: Release tarball holding the reference recordings + JSON sidecars.
_RELEASE_URL = "https://github.com/mandulaj/evutils/releases/download/v0.3.14/testfiles.tar.xz"

#: Cached ``{format: [EventFile, ...]}`` for the whole session (the download /
#: extraction happens at most once, even across collection and fixtures).
_REAL_FILES_CACHE: dict[str, list] | None = None


def _load_real_event_files(config: Any) -> dict[str, list]:
    """Resolve, download-if-needed, and parse the reference recordings.

    Returns ``{format: [EventFile, ...]}`` (empty if the data is unavailable).
    Set ``EVUTILS_BENCH_DATA`` to a directory that already holds the extracted
    recordings + JSON sidecars to skip the download (offline environments).
    """
    global _REAL_FILES_CACHE
    if _REAL_FILES_CACHE is not None:
        return _REAL_FILES_CACHE

    override = os.environ.get("EVUTILS_BENCH_DATA")
    if override:
        data_dir = Path(override)
        _REAL_FILES_CACHE = load_event_files(data_dir) if data_dir.is_dir() else {}
        return _REAL_FILES_CACHE

    temp_dir = config.cache.mkdir("event_files")
    download_and_extract_github(_RELEASE_URL, temp_dir, "testfiles.tar.xz")
    _REAL_FILES_CACHE = load_event_files(temp_dir)
    return _REAL_FILES_CACHE


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
    """Real Prophesee recordings grouped by format (downloaded + cached).

    Returns ``{format: [EventFile(path, count, metadata), ...]}``. Skips the
    test when no reference data is available.
    """
    files = _load_real_event_files(request.config)
    if not files:
        pytest.skip("No real event files available (set EVUTILS_BENCH_DATA or allow the download)")
    return files


def pytest_generate_tests(metafunc: Any) -> None:
    """Parametrize ``real_event_file`` with one case per discovered recording.

    Any test that requests the ``real_event_file`` fixture is run once for every
    file in the reference tarball, regardless of format -- so dropping a new
    recording + JSON sidecar into the tarball extends coverage with zero code
    changes. Each case is identified by the file name.
    """
    if "real_event_file" not in metafunc.fixturenames:
        return

    files_by_format = _load_real_event_files(metafunc.config)
    flat = [ef for efs in files_by_format.values() for ef in efs]

    if flat:
        metafunc.parametrize("real_event_file", flat, ids=[ef.path.name for ef in flat])
    else:
        metafunc.parametrize(
            "real_event_file",
            [pytest.param(None, marks=pytest.mark.skip(reason="no real event files available"))],
        )
