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
    from evutils.io import EventWriter
    assert EventWriter is not None
    test_writer = EventWriter("test.raw")
    assert test_writer is not None
    test_writer.close()


def test_RAW_reader_import(dummy_file_factory):
    '''
    Testing if the RAW reader can be imported
    '''
    test_file = dummy_file_factory("test.raw")

    from evutils.io import EventReader
    assert EventReader is not None
    test_reader = EventReader(test_file)
    assert test_reader is not None
    test_reader.close()

### Test RAW writer

def test_RAW_writer(tmp_path, test_events):
    from evutils.io import EventWriter
    from evutils.io import EventReader

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.raw"
    with EventWriter(p) as writer:
        writer.write(test_events)
    

    # Check if the file is created
    assert p.is_file()

    # Check if the file is not empty
    assert p.stat().st_size > 0

    # Check if the file can be read
    with EventReader(p) as reader:
        events = reader.read()

    assert np.array_equal(events, test_events)



def test_RAW_writer_evt2(tmp_path, test_events):
    from evutils.io import EventWriter
    from evutils.io import EventReader

    return

    p = tmp_path / "evt2.raw"
    with EventWriter_RAW(p, format='evt2') as writer:
        writer.write(test_events)

    # Check if the file is created
    assert p.is_file()

    # Check if the file is not empty
    assert p.stat().st_size > 0

    # Check if the file can be read
    with EventReader_RAW(p) as reader:
        events = reader.read()

    assert np.array_equal(events, test_events)

def test_RAW_writer_evt21(tmp_path, test_events):
    from evutils.io import EventWriter
    from evutils.io import EventReader

    d = tmp_path / "sub"
    d.mkdir()
    

def test_RAW_writer_evt3(tmp_path, test_events):
    from evutils.io import EventWriter
    from evutils.io import EventReader

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.raw"
    with EventWriter(p, format='evt3') as writer:
        writer.write(test_events)

    # Check if the file is created
    assert p.is_file()

    # Check if the file is not empty
    assert p.stat().st_size > 0

    # Check if the file can be read
    with EventReader(p) as reader:
        events = reader.read()

    assert np.array_equal(events, test_events)


def test_RAW_real_read(real_event_files):
    from evutils.io import EventReader


    for format in ['evt3']:
        with EventReader(real_event_files[format]) as reader:
            assert reader._is_initialized == False
            events = reader.read()
            assert reader._is_initialized == True

            print(events)
            # assert format == reader.format
            assert reader.shape() == (1280, 720)


    # assert len(events) > 0
    # assert len(events) == 1000
    # assert np.array_equal(events, test_events)


def test_RAW_nevents_read(real_event_files):
    from evutils.io import EventReader

    return

    for format in ['evt3']:

        for length in [10, 100, 1000]:
            
            with EventReader(real_event_files[format], n_events=length) as reader:
                events = reader.read()

                assert len(events) == length

def test_RAW_delta_t_read(real_event_files):
    from evutils.io import EventReader

    return

    for format in ['evt3']:

        for delta_t in [10, 100, 1000]:
            with EventReader(real_event_files[format], delta_t=delta_t) as reader:
                events = reader.read()

                assert len(events) > 0
                assert len(events) == 1000
                # assert np.array_equal(events, test_events)