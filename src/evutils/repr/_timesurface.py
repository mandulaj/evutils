
"""Module for generating time surface representations from events."""

import numpy as np
import numba
from typing import Any 

@numba.njit
def timesurface(events: np.ndarray, width: int = 1280, height: int = 720, tau: int = 10_000, dtype: Any = np.float32) -> np.ndarray:
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

    [1] Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern recognition https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476

    """
    # Initialize the buffer
    buffer = np.zeros((height, width), dtype=dtype)

    if len(events) < 1:
        return buffer

    # Last event timestamp
    t_ref = events[-1]['t']

    for e in events:
        x = e['x']
        y = e['y']
        t = e['t']
        p = e['p']

        value = np.exp(-(t_ref - t) / tau)

        # If polarity is 1, we keep the value as is, flip it if polarity is 0
        if p == 0:
            value = -value

        # Update the buffer
        buffer[y, x] = value


    return buffer

    """"""
