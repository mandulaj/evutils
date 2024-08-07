


import numpy as np



__all__ = ['Event_dtype', 'Trigger_dtype']


#: A structured numpy dtype for event data.
#:
#: Fields:
#:
#: - `t` (np.int64): Timestamp of the event (us).
#: - `x` (np.uint16): X-coordinate.
#: - `y` (np.uint16): Y-coordinate.
#: - `p` (np.uint8): Polarity (0: off, 1: on).
Event_dtype = np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)])


#: A structured numpy dtype for trigger data.
#:
#: Fields:
#:
#: - `t` (np.int64): Timestamp of the event (us).
#: - `p` (np.uint8):  Polarity (0: off, 1: on).
#: - `id` (np.uint8): Identifier.
Trigger_dtype = np.dtype([('t', np.int64), ('p', np.uint8), ('id', np.uint8)])

def is_monotonically_increasing(events: np.ndarray) -> bool:
    '''Checks if the event ts is monotonically increasing'''
    return np.all(np.diff(events['t']) >= 0)

