# import pytest
import numpy as np




####################################
####################################
# Test RAW module
####################################
####################################


def test_RAW_writer_import():
    '''
    Testing if the RAW writer can be imported
    '''
    from evutils.io.writer import EventWriter_RAW
    assert EventWriter_RAW is not None
    test_writer = EventWriter_RAW("test.raw")
    assert test_writer is not None


def test_RAW_reader_import(dummy_file_factory):
    '''
    Testing if the RAW reader can be imported
    '''
    test_file = dummy_file_factory("test.raw")

    from evutils.io.reader import EventReader_RAW
    assert EventReader_RAW is not None
    test_reader = EventReader_RAW(test_file)
    assert test_reader is not None


### Test RAW writer

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

    # Check if the file is not empty
    assert p.stat().st_size > 0

    # Check if the file can be read
    reader = EventReader_RAW(p)
    events = reader.read()

    assert np.array_equal(events, test_events)



def test_RAW_writer_evt2(tmp_path, test_events):
    from evutils.io.writer import EventWriter_RAW
    from evutils.io.reader import EventReader_RAW

    p = tmp_path / "evt2.raw"
    writer = EventWriter_RAW(p, format='evt2')
    writer.write(test_events)
    writer.close()

    # Check if the file is created
    assert p.is_file()

    # Check if the file is not empty
    assert p.stat().st_size > 0

    # Check if the file can be read
    reader = EventReader_RAW(p)
    events = reader.read()

    assert np.array_equal(events, test_events)

def test_RAW_writer_evt21(tmp_path, test_events):
    from evutils.io.writer import EventWriter_RAW
    from evutils.io.reader import EventReader_RAW

    d = tmp_path / "sub"
    d.mkdir()
    

def test_RAW_writer_evt3(tmp_path, test_events):
    from evutils.io.writer import EventWriter_RAW
    from evutils.io.reader import EventReader_RAW

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.evt3"
    writer = EventWriter_RAW(p)
    writer.write(test_events)
    writer.close()

    # Check if the file is created
    assert p.is_file()

    # Check if the file is not empty
    assert p.stat().st_size > 0

    # Check if the file can be read
    reader = EventReader_RAW(p)
    events = reader.read()

    assert np.array_equal(events, test_events)


def test_RAW_real_read(real_event_files):
    # from evutils.io.reader import EventReader_RAW

    print(real_event_files)
    print("Hello")
    assert True

    # reader = EventReader_RAW(real_event_files['evt3'])
    # events = reader.read()

    # assert len(events) > 0
    # assert len(events) == 1000
    # assert np.array_equal(events, test_events)