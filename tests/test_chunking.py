import numpy as np
import pytest
from evutils.chunking import window_delta_t, sliding_window, sort_events, get_dt_events
from evutils.random import random_events
from evutils.types import Event_dtype

def test_window_delta_t():
    # Empty array
    events = np.array([], dtype=Event_dtype)
    chunks = list(window_delta_t(events, delta_t=1000))
    assert len(chunks) == 0

    # Normal case
    events = random_events(100, start_ts=0, end_ts=30_000)
    chunks = list(window_delta_t(events, delta_t=10_000))
    for chunk in chunks:
        if len(chunk) > 0:
            assert chunk['t'].max() - chunk['t'].min() <= 10_000

    # Events exactly on boundary
    events = np.array([(0,0,0,0), (10_000,0,0,0), (20_000,0,0,0)], dtype=Event_dtype)
    chunks = list(window_delta_t(events, delta_t=10_000))
    assert len(chunks) == 3
    assert len(chunks[0]) == 1
    assert len(chunks[1]) == 1
    assert len(chunks[2]) == 1

def test_sliding_window():
    events = np.array([], dtype=Event_dtype)
    chunks = list(sliding_window(events))
    assert len(chunks) == 0

    events = random_events(100, start_ts=0, end_ts=50_000)
    
    # full_window = False
    chunks = list(sliding_window(events, delta_t=10_000, window_size=20_000, full_window=False))
    assert len(chunks) > 0

    # full_window = True
    chunks_full = list(sliding_window(events, delta_t=10_000, window_size=20_000, full_window=True))
    # Number of full chunks should be less than or equal to total chunks
    assert len(chunks_full) <= len(chunks)

def test_sort_events():
    events = np.array([], dtype=Event_dtype)
    sorted_events = sort_events(events)
    assert len(sorted_events) == 0

    events = random_events(10)
    events['t'] = np.arange(10, 0, -1)
    sorted_events = sort_events(events)
    assert sorted_events['t'][0] == 1
    assert sorted_events['t'][-1] == 10

def test_get_dt_events():
    events = np.array([], dtype=Event_dtype)
    assert len(get_dt_events(events)) == 0

    events = random_events(100, start_ts=0, end_ts=50_000)
    sub = get_dt_events(events, dt=10_000)
    if len(sub) > 0:
        assert (sub['t'] <= events['t'][0] + 10_000).all()

    # dt larger than stream
    sub2 = get_dt_events(events, dt=100_000)
    assert len(sub2) == len(events)
