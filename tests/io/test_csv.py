import numpy as np


####################################
####################################
# Test CSV module
####################################
####################################



def test_CSV_writer_import():
    from evutils.io.writer import EventWriter_Csv
    assert EventWriter_Csv is not None
    test_writer = EventWriter_Csv("test.csv")
    assert test_writer is not None

def test_CSV_reader_import(dummy_file_factory):
    test_file = dummy_file_factory("test.csv")

    from evutils.io.reader import EventReader_Csv
    assert EventReader_Csv is not None
    test_reader = EventReader_Csv(test_file)
    assert test_reader is not None


def test_CSV_writer(tmp_path, test_events):
    from evutils.io.writer import EventWriter_Csv
    from evutils.io.reader import EventReader_Csv


    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    writer = EventWriter_Csv(p)
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


    reader = EventReader_Csv(p)
    events = reader.read()
    assert np.array_equal(events, test_events)