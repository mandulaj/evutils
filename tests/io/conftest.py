import os

import pytest

# ``test_events`` and ``real_event_files`` live in the repo-root conftest.py so
# they are shared with the benchmarks; only io-test-specific fixtures live here.


from typing import Any, Callable

@pytest.fixture
def dummy_file_factory(tmp_path: Any) -> Callable[[str], str]:
    # Create a temporary dummy (non-event) file.
    def _filefactory(name: str) -> str:
        test_file_path = os.path.join(tmp_path, name)
        with open(test_file_path, 'w') as f:
            f.write('This is a dummy test file.')
        return test_file_path

    return _filefactory


@pytest.fixture(scope='session')
def event_files(tmp_path_factory: Any, test_events: Any) -> dict[str, Any]:
    event_file_paths: dict[str, Any] = {}

    temp_dir = tmp_path_factory.mktemp("data")
    event_file_paths['csv'] = temp_dir / "events.csv"
    event_file_paths['csv_noheader'] = temp_dir / "events_noheader.csv"
    event_file_paths['csv_shuffled_columns'] = temp_dir / "events_noheader_xypt.csv"
    event_file_paths['evt2'] = temp_dir / "events_evt2.raw"
    event_file_paths['evt21'] = temp_dir / "events_evt21.raw"
    event_file_paths['evt3'] = temp_dir / "events_evt3.raw"
    event_file_paths['hdf5'] = temp_dir / "events.h5"
    event_file_paths['dat'] = temp_dir / "events.dat"
    event_file_paths['npz'] = temp_dir / "events.npz"
    event_file_paths['bin'] = temp_dir / "events.bin"
    event_file_paths['txt'] = temp_dir / "events.txt"

    from evutils.io import EventWriter
    with EventWriter(event_file_paths['csv']) as writer:
        writer.write(test_events)

    with EventWriter(event_file_paths['evt3'], format="EVT3") as writer:
        writer.write(test_events)

    return event_file_paths
