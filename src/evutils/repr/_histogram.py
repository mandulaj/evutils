"""Module for generating histogram-based representations from events."""

import numba 
import numpy as np
from typing import Any


@numba.njit
def histogram(events: np.ndarray, width: int = 1280, height: int = 720, fill: bool = False, dtype: Any = np.uint8) -> np.ndarray:
    """Generate a histogram frame from the events.

    Parameters
    ----------
    events : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int, optional
        Width of the frame, by default 1280
    height : int, optional
        Height of the frame, by default 720
    fill : bool, optional
        If True, the non-zero values are set to 255, by default False
    dtype : np.dtype, optional
        Data type of the output array, by default np.uint8
    
    Returns
    -------
    np.ndarray
        A numpy array with the histogram frame (height, width, 3)

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.repr import histogram
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = histogram(events, width=100, height=100)
    >>> frame.shape
    (100, 100, 3)
    """
    buffer = np.zeros((height, width, 3), dtype=dtype)

    clip = 255

    # clip = 255 if dtype == np.uint8 else 65535

    for e in events:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            p = 0 # Red
        else:
            p = 2 # Blue
        


        if buffer[y, x, p] < clip:
            buffer[y, x, p] += 1

    # If the fill flag is set, we fill the non-zero values with 255
    if fill:
        for x in range(0, width):
            for y in range(0, height):
                for c in range(0, 3):
                    if buffer[y, x, c] > 0:
                        buffer[y, x, c] = 255

    return buffer


@numba.njit
def wedge_histogram(events: np.ndarray, width: int = 1280, height: int = 720, tl: float = 30e6, dtype: Any = np.uint8) -> np.ndarray:   
    """Generate a wedge histogram frame from the events.

    Parameters
    ----------
    events : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int, optional
        Width of the frame, by default 1280
    height : int, optional
        Height of the frame, by default 720
    tl : float, optional
        Time limit for the frame in us, by default 30e6
    dtype : np.dtype, optional
        Data type of the output array, by default np.uint8
    
    Returns
    -------
    np.ndarray
        A numpy array with the wedge frame (height, width, 3)

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.repr import wedge_histogram
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = wedge_histogram(events, width=100, height=100)
    >>> frame.shape
    (100, 100, 3)
    """
    buffer = np.zeros((height, width, 3), dtype=dtype)

    ts_norm = (events['t'] - events['t'][0])/tl

    # print(len(ev))

    sel = (height-events['y'])/height > ts_norm - 0.1
    ev_sel = events[sel]
    # print(len(ev_sel))
    # print((720-ev['y'])/720, ts_norm, sel)


    for e in ev_sel:
        x = e['x']
        y = e['y']
        p = e['p']


        if p == 1:
            p = 0
        else:
            p = 2
        
        if buffer[y, x, p] < 128:
            buffer[y, x, p] += 255

    return buffer