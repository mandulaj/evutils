

from evutils.types import Events

import numpy as np


N_EVENTS = 1000
np.random.seed(42)

TEST_EVENTS = np.zeros(N_EVENTS, dtype=Events)
TEST_EVENTS['t'] = np.random.randint(0, 10000, N_EVENTS)
TEST_EVENTS.sort(order='t')
TEST_EVENTS['x'] = np.random.randint(0, 1280, N_EVENTS)
TEST_EVENTS['y'] = np.random.randint(0, 720, N_EVENTS)
TEST_EVENTS['p'] = np.random.randint(0, 2, N_EVENTS)


def test_CSV_writer(tmp_path):
    from evutils.io.writer import EventWriter_Csv
    from evutils.io.reader import EventReader_Csv


    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    writer = EventWriter_Csv(p)
    writer.write(TEST_EVENTS)
    writer.close()

    # Check if the file is created
    assert p.is_file()

    # load last line:
    with open(p, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        assert last_line == f"{TEST_EVENTS[-1]['t']},{TEST_EVENTS[-1]['x']},{TEST_EVENTS[-1]['y']},{TEST_EVENTS[-1]['p']}\n"

        assert len(lines) == N_EVENTS + 1


    reader = EventReader_Csv(p)
    events = reader.read()
    assert np.array_equal(events, TEST_EVENTS)

def test_RAW_writer(tmp_path):
    from evutils.io.writer import EventWriter_RAW
    from evutils.io.reader import EventReader_RAW

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.raw"
    writer = EventWriter_RAW(p)
    writer.write(TEST_EVENTS)
    writer.close()

    # Check if the file is created
    assert p.is_file()

    reader = EventReader_RAW(p)
    events = reader.read()

    assert np.array_equal(events, TEST_EVENTS)