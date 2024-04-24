


import numpy as np


Events = np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)])

Triggers = np.dtype([('t', np.int64), ('p', np.uint8), ('id', np.uint8)])


def is_monotonically_increasing(events: np.ndarray) -> bool:
    '''Checks if the event ts is monotonically increasing'''
    return np.all(np.diff(events['t']) >= 0)

