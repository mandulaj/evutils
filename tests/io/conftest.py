import pytest
import os
import tempfile
from evutils.types import Event_dtype

import numpy as np

@pytest.fixture(scope='session')
def test_events():

    ## Setup test data

    N_EVENTS = 1000
    np.random.seed(42)

    test_events = np.zeros(N_EVENTS, dtype=Event_dtype)
    test_events['t'] = np.random.randint(0, 10000, N_EVENTS)
    test_events.sort(order='t')
    test_events['x'] = np.random.randint(0, 1280, N_EVENTS)
    test_events['y'] = np.random.randint(0, 720, N_EVENTS)
    test_events['p'] = np.random.randint(0, 2, N_EVENTS)

    return test_events


@pytest.fixture
def dummy_file_factory(tmp_path):
    # Create a temporary directory

    def _filefactory(name):
        test_file_path = os.path.join(tmp_path, name)
        with open(test_file_path, 'w') as f:
            f.write('This is a dummy test file.')
        return test_file_path
    # Yield the path to the test file
    return _filefactory



@pytest.fixture(scope='session')
def event_files(tmp_path_factory, test_events):
    event_file_paths = {}

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
    

    from evutils.io.writer import EventWriter_Csv
    with EventWriter_Csv(event_file_paths['csv']) as writer:
        writer.write(test_events)
    
    from evutils.io.writer import EventWriter_RAW
    with EventWriter_RAW(event_file_paths['evt3'], format="EVT3") as writer:
        writer.write(test_events)
    
    # event_file_paths['hdf5'] = temp_dir / "events.h5"
    # from evutils.io.writer import EventWriter_HDF5
    # with EventWriter_HDF5(event_file_paths['hdf5']) as writer:
    #     writer.write(test_events)
    
    
    return event_file_paths


@pytest.fixture()
def real_event_files(cache):
    event_file_paths = {}

    temp_dir = cache.mkdir("event_files")


    event_file_paths['evt3'] = temp_dir / "events_evt3.raw"


    print(temp_dir)

    return event_file_paths

    # Download file if it does not exist
           

