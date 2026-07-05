import pytest
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
    from evutils.io import EventWriter, EventReader

    p = tmp_path / "evt2.raw"
    with EventWriter(p, format='evt2') as writer:
        writer.write(test_events)

    assert p.is_file()
    assert p.stat().st_size > 0

    with EventReader(p) as reader:
        events = reader.read()
    assert np.array_equal(events, test_events)


def test_RAW_writer_evt21(tmp_path, test_events):
    from evutils.io import EventWriter, EventReader

    p = tmp_path / "evt21.raw"
    with EventWriter(p, format='evt21') as writer:
        writer.write(test_events)

    assert p.is_file()
    assert p.stat().st_size > 0

    with EventReader(p) as reader:
        events = reader.read()
    assert np.array_equal(events, test_events)



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

    for format in ['evt3', 'evt2', 'evt21']:
        with EventReader(real_event_files[format].path) as reader:
            assert reader._is_initialized == False
            events = reader.read()
            assert reader._is_initialized == True

            assert reader.shape() == (1280, 720)
            assert len(events) > 0

            # Decoded events must be valid and time-ordered.
            assert int(events.x.max()) < 1280
            assert int(events.y.max()) < 720
            assert bool(np.all(np.diff(events.t) >= 0))


def test_RAW_real_count(real_event_files):
    '''The reader's total event count should closely match the reference count
    from the OpenEB implementation. The decoders differ slightly (e.g. edge
    handling around the first/last time base), so we allow a small relative
    tolerance instead of requiring an exact match.'''
    from evutils.io import EventReader

    REL_TOL = 0.01  # 1%

    for format in ['evt3', 'evt2', 'evt21']:
        ref = real_event_files[format].count
        n = sum(len(e) for e in
                EventReader(real_event_files[format].path, n_events=5_000_000))
        diff = n - ref
        assert abs(diff) <= REL_TOL * ref, (
            f"{format}: reader returned {n:,} events, reference {ref:,} "
            f"(diff {diff:+,}, {100 * diff / ref:+.3f}%)"
        )


def test_EVT2_matches_EVT3(real_event_files):
    '''EVT2 and EVT3 are the same recording transcoded, so decoding them must
    yield identical events (timestamps equal up to a constant offset).'''
    from evutils.io import EventReader

    def head(fmt, n=200_000):
        parts = []
        with EventReader(real_event_files[fmt].path, n_events=100_000) as reader:
            for e in reader:
                parts.append(e)
                if sum(len(p) for p in parts) >= n:
                    break
        t = np.concatenate([p.t for p in parts])[:n]
        x = np.concatenate([p.x for p in parts])[:n]
        y = np.concatenate([p.y for p in parts])[:n]
        p_ = np.concatenate([p.p for p in parts])[:n]
        return t, x, y, p_

    t2, x2, y2, p2 = head('evt2')
    t3, x3, y3, p3 = head('evt3')

    assert np.array_equal(x2, x3)
    assert np.array_equal(y2, y3)
    assert np.array_equal(p2, p3)
    # Timestamps differ only by a constant (per-file time origin).
    assert np.ptp(t3.astype(np.int64) - t2.astype(np.int64)) == 0


def test_RAW_nevents_read(real_event_files):
    from evutils.io import EventReader

    for format in ['evt3', 'evt2', 'evt21']:
        for length in [10, 100, 1000]:
            with EventReader(real_event_files[format].path, n_events=length) as reader:
                events = reader.read()
                assert len(events) == length


def test_RAW_delta_t_read(real_event_files):
    from evutils.io import EventReader

    for format in ['evt3', 'evt2', 'evt21']:
        for delta_t in [100, 1000, 10000]:
            with EventReader(real_event_files[format].path, delta_t=delta_t) as reader:
                events = reader.read()
                assert len(events) > 0
                # All events fall within the requested time window.
                assert int(events.t.max()) - int(events.t.min()) <= delta_t