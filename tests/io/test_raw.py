# import pytest
import numpy as np




####################################
####################################
# Test RAW module
####################################
####################################


def test_RAW_writer_import():
    from evutils.io.writer import EventWriter_RAW
    assert EventWriter_RAW is not None
    test_writer = EventWriter_RAW("test.raw")
    assert test_writer is not None


def test_RAW_reader_import(dummy_file_factory):
    test_file = dummy_file_factory("test.raw")

    from evutils.io.reader import EventReader_RAW
    assert EventReader_RAW is not None
    test_reader = EventReader_RAW(test_file)
    assert test_reader is not None



def test_RAW_writer(tmp_path, test_events):
    from evutils.io.writer import EventWriter_RAW
    from evutils.io.reader import EventReader_RAW

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.raw"
    writer = EventWriter_RAW(p)
    writer.write(test_events)
    writer.close()

    # Check if the file is created
    assert p.is_file()

    reader = EventReader_RAW(p)
    events = reader.read()

    assert np.array_equal(events, test_events)