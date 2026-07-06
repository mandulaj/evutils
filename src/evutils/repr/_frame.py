"""Module for generating frame-based representations from events."""

import numba 
import numpy as np
from typing import Any


@numba.njit
def frame_gray(events: np.ndarray, width: int = 1280, height: int = 720, dtype: Any = np.uint8) -> np.ndarray:
    """Generate a grayscale frame from the events.

    Parameters
    ----------
    events : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int, optional
        Width of the frame, by default 1280
    height : int, optional
        Height of the frame, by default 720
    dtype : np.dtype, optional
        Data type of the output array, by default np.uint8
    
    Returns
    -------
    np.ndarray
        A numpy array with the frame (height, width)

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.repr import frame_gray
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = frame_gray(events, width=100, height=100)
    >>> frame.shape
    (100, 100)
    """
    buffer = np.full((height, width), 128, dtype=dtype)


    for e in events:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            p = 255
        else:
            p = 0

        buffer[y, x] = p 

    return buffer


@numba.njit
def frame_rgb(ev: np.ndarray, width: int = 1280, height: int = 720, bg_color: Any = np.array((0, 0, 0)), pos_color: Any = np.array((255, 0, 0)), neg_color: Any = np.array((0, 0, 255))) -> np.ndarray:
    """Generate an RGB frame from the events.

    Parameters
    ----------
    ev : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int, optional
        Width of the frame, by default 1280
    height : int, optional
        Height of the frame, by default 720
    bg_color : np.ndarray, optional
        Background color, by default np.array((0, 0, 0))
    pos_color : np.ndarray, optional
        Color for positive events, by default np.array((255, 0, 0))
    neg_color : np.ndarray, optional
        Color for negative events, by default np.array((0, 0, 255))

    Returns
    -------
    np.ndarray
        A numpy array with the frame (height, width, 3)

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.repr import frame_rgb
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = frame_rgb(events, width=100, height=100)
    >>> frame.shape
    (100, 100, 3)
    """
    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    buffer[:, :] = bg_color

    for e in ev:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            buffer[y, x] = pos_color
        else:
            buffer[y, x] = neg_color

    return buffer


@numba.njit
def frame_diff(ev: np.ndarray, width: int = 1280, height: int = 720, dtype: Any = np.int8) -> np.ndarray:
    """Generate a differential frame from the events.

    Parameters
    ----------
    ev : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int, optional
        Width of the frame, by default 1280
    height : int, optional
        Height of the frame, by default 720
    dtype : np.dtype, optional
        Data type of the output array, by default np.int8
        
    Returns
    -------
    np.ndarray
        A numpy array with the frame (height, width)

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.repr import frame_diff
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = frame_diff(events, width=100, height=100)
    >>> frame.shape
    (100, 100)
    """
    buffer = np.zeros((height, width), dtype=dtype)

    for e in ev:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            buffer[y, x] += 1
        else:
            buffer[y, x] -= 1

    return buffer

