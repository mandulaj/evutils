

from evutils.random import random_events, random_events_generator, event_jitter, event_jitter_n

from evutils.types import is_monotonically_increasing

import numpy as np

N = 1000

def test_random_events():
    ev = random_events(N, 1280, 720, 10, 10000)

    assert len(ev) == N
    assert ev["x"].min() >= 0 and ev["x"].max() < 1280
    assert ev["y"].min() >= 0 and ev["y"].max() < 720
    assert ev["p"].min() >= 0 and ev["p"].max() <= 1
    assert ev["t"].min() >= 10 and ev["t"].max() < 10000


def test_random_events_generator():
    for ev in random_events_generator(N, 1280, 720, 10, 10000, chunk_size=100):
        assert len(ev) == 100
        assert ev["x"].min() >= 0 and ev["x"].max() < 1280
        assert ev["y"].min() >= 0 and ev["y"].max() < 720
        assert ev["p"].min() >= 0 and ev["p"].max() <= 1
        assert ev["t"].min() >= 10 and ev["t"].max() < 10000


def test_event_jitter():
    ev = random_events(N, 1280, 720, 10, 10000)
    ev_jitter = event_jitter(ev.copy(), jitter=100, sort=False)

    assert len(ev) == len(ev_jitter)
    assert np.array_equal(ev["x"], ev_jitter["x"])
    assert np.array_equal(ev["y"], ev_jitter["y"])
    assert np.array_equal(ev["p"], ev_jitter["p"])

    assert not is_monotonically_increasing(ev_jitter)

    ev_s_jitter = event_jitter(ev.copy(), jitter=100, sort=True)

    print(np.diff(ev_s_jitter["t"]))
    
    assert is_monotonically_increasing(ev_s_jitter)