import numpy as np
import pytest







####################################
####################################
# Test CSV module
####################################
####################################





from typing import Any
def test_CSV_writer(tmp_path: Any, test_events: Any) -> None:
    from evutils.io import EventWriter
    from evutils.io import EventReader


    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    writer = EventWriter(p)
    writer.write(test_events)
    writer.close()

    # Check if the file is created
    assert p.is_file()

    # load last line:
    with open(p, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        assert last_line == f"{test_events[-1]['t']},{test_events[-1]['x']},{test_events[-1]['y']},{test_events[-1]['p']}\n"

        assert len(lines) == len(test_events) + 1


    reader = EventReader(p, n_events=len(test_events))
    events = reader.read()
    assert np.array_equal(events, test_events)

def test_CSV_reader_nevents(event_files: Any, test_events: Any) -> None:
    csv_file_path = event_files['csv']

    from evutils.io import EventReader

    STEP=100

    with EventReader(csv_file_path, n_events=STEP) as reader:

        for i in range(0, len(test_events), STEP):
            events = reader.read()
            assert np.array_equal(events, test_events[i:i+STEP])


def test_CSV_reader_gen(event_files: Any, test_events: Any) -> None:
    csv_file_path = event_files['csv']

    from evutils.io import EventReader

    STEP=100
    i = 0

    for ev in EventReader(csv_file_path, n_events=STEP):

        assert np.array_equal(ev, test_events[i:i+STEP])
        i += STEP

