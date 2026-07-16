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


def test_event_array_empty() -> None:
    from evutils.types import EventArray
    empty_events = EventArray.empty()
    assert len(empty_events) == 0
    assert len(empty_events.t) == 0
    assert empty_events.to_aos().shape == (0,)
    assert repr(empty_events) == "EventArray(empty)"


def test_soa_array_negative_indexing() -> None:
    from evutils.types import EventArray
    t = [10, 20, 30]
    x = [1, 2, 3]
    y = [4, 5, 6]
    p = [0, 1, 0]
    
    events = EventArray(t, x, y, p)
    
    # Negative scalar index
    ev_last = events[-1]
    assert isinstance(ev_last, np.void)
    assert ev_last['t'] == 30
    assert ev_last['x'] == 3
    
    # Negative slice
    ev_slice = events[-2:]
    assert isinstance(ev_slice, EventArray)
    assert len(ev_slice) == 2
    assert ev_slice.t[0] == 20
    assert ev_slice.t[1] == 30


def test_soa_array_bool_indexing() -> None:
    from evutils.types import EventArray
    t = [10, 20, 30]
    x = [1, 2, 3]
    y = [4, 5, 6]
    p = [0, 1, 0]
    
    events = EventArray(t, x, y, p)
    
    # Bool index
    mask = events.p == 1
    ev_bool = events[mask]
    assert isinstance(ev_bool, EventArray)
    assert len(ev_bool) == 1
    assert ev_bool.t[0] == 20


def test_soa_array_repr_long() -> None:
    from evutils.types import EventArray
    t = np.arange(20)
    x = np.arange(20)
    y = np.arange(20)
    p = np.zeros(20)
    
    events = EventArray(t, x, y, p)
    rep = repr(events)
    assert "..." in rep
    assert "EventArray(n=20)" in rep


def test_trigger_array_empty() -> None:
    from evutils.types import TriggerArray
    empty_trig = TriggerArray.empty()
    assert len(empty_trig) == 0
    assert len(empty_trig.t) == 0
    assert empty_trig.to_aos().shape == (0,)


def test_event_array_scalar_init() -> None:
    from evutils.types import EventArray
    ev = EventArray(t=1, x=2, y=3, p=1)
    assert len(ev) == 1
    assert ev.t[0] == 1


def test_soa_array_multi_field_indexing() -> None:
    from evutils.types import EventArray, SoaArray
    events = EventArray(t=[1, 2], x=[10, 20], y=[30, 40], p=[1, 0])
    sub = events[['x', 'y']]
    assert isinstance(sub, SoaArray)
    assert len(sub) == 2
    assert np.array_equal(sub.x, [10, 20])
    assert np.array_equal(sub.y, [30, 40])
    assert not hasattr(sub, 't')

    