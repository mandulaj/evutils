import numpy as np

from evutils.types import Event_dtype


def test_event_size() -> None:
    a = np.zeros(1, dtype=Event_dtype)
    assert a.itemsize == 13


def test_event_dtype() -> None:
    a = np.zeros(1, dtype=Event_dtype)
    a['t'] = 1
    a['x'] = 2
    a['y'] = 3
    a['p'] = 4


def test_soa_array_indexing() -> None:
    from evutils.types import EventArray, TriggerArray
    
    # EventArray
    t = [10, 20, 30]
    x = [1, 2, 3]
    y = [4, 5, 6]
    p = [0, 1, 0]
    
    events = EventArray(t, x, y, p)
    
    # Test column access
    assert np.array_equal(events['t'], t)
    assert np.array_equal(events.t, t)
    
    # Test slice indexing
    ev_slice = events[1:]
    assert isinstance(ev_slice, EventArray)
    assert len(ev_slice) == 2
    assert ev_slice.t[0] == 20
    
    # Test scalar indexing (should return a void record, just like AoS)
    ev_single = events[1]
    assert isinstance(ev_single, np.void)
    assert ev_single['t'] == 20
    assert ev_single['x'] == 2
    
    # TriggerArray
    trig_t = [100, 200]
    trig_p = [1, 0]
    trig_id = [5, 6]
    
    triggers = TriggerArray(trig_t, trig_p, trig_id)
    
    # Test slice indexing
    trig_slice = triggers[:1]
    assert isinstance(trig_slice, TriggerArray)
    assert len(trig_slice) == 1
    
    # Test scalar indexing
    trig_single = triggers[0]
    assert isinstance(trig_single, np.void)
    assert trig_single['t'] == 100
    assert trig_single['id'] == 5
    