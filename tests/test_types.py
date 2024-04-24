import numpy as np

from evutils.types import Events


def test_event_size():
    a = np.zeros(1, dtype=Events)
    assert a.itemsize == 13


def test_event_dtype():
    a = np.zeros(1, dtype=Events)
    a['t'] = 1
    a['x'] = 2
    a['y'] = 3
    a['p'] = 4
    