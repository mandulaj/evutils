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
    