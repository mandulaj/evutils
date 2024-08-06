import pytest
import os
import tempfile
from evutils.types import Events

import numpy as np

@pytest.fixture(scope='session')
def test_events():

    ## Setup test data

    N_EVENTS = 1000
    np.random.seed(42)

    test_events = np.zeros(N_EVENTS, dtype=Events)
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