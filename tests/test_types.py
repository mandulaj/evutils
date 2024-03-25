import numpy as np

from evutils.types import Events


def test_event_size():
    a = np.zeros(1, dtype=Events)
    assert a.itemsize == 13
