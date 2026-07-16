
"""Module for generating time surface representations from events."""

import numpy as np
from typing import Any 
from ..jit import lazy_njit_unwrapped_events
from ..types import EventArray

@lazy_njit_unwrapped_events
def _timesurface_jit(t, x, y, p, buffer, t_ref, tau):
    height, width = buffer.shape
    for i in range(len(t)):
        xi = x[i]
        yi = y[i]
        ti = t[i]
        pi = p[i]
        if 0 <= xi < width and 0 <= yi < height:
            dt = max(0.0, float(t_ref - ti))
            value = np.exp(-dt / tau)
            if pi == 0:
                value = -value
            buffer[yi, xi] = value

def timesurface(events: 'np.ndarray | EventArray', width: int = 1280, height: int = 720, tau: int = 10_000, dtype: Any = np.float32) -> np.ndarray:
    """Generate a time surface frame from the events.

    Parameters
    ----------
    events : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format.
    width : int, optional
        Width of the time surface frame, by default 1280.
    height : int, optional
        Height of the time surface frame, by default 720.
    tau : int, optional
        Time constant for the exponential decay, by default 10_000 (10 ms).
    dtype : np.dtype, optional
        Data type of the output frame, by default np.uint8.

    Returns
    -------
    np.ndarray
        A numpy array with the time surface frame (height, width).

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.repr import timesurface
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = timesurface(events, width=100, height=100)
    >>> frame.shape
    (100, 100)

    [1] Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern recognition https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476

    """
    # Initialize the buffer
    buffer = np.zeros((height, width), dtype=dtype)

    if len(events) < 1:
        return buffer

    # Last event timestamp
    t_ref = events['t'][-1]
    
    _timesurface_jit(events, buffer, t_ref, tau)

    return buffer

    """"""
