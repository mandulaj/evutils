"""Module for generating frame-based representations from events."""

import numpy as np
from typing import Any
from ..jit import lazy_njit_unwrapped_events
from ..types import EventArray

@lazy_njit_unwrapped_events
def _frame_gray_jit(t, x, y, p, buffer):
    height, width = buffer.shape
    for i in range(len(t)):
        xi = x[i]
        yi = y[i]
        pi = p[i]
        if 0 <= xi < width and 0 <= yi < height:
            if pi == 1:
                buffer[yi, xi] = 255
            else:
                buffer[yi, xi] = 0

def frame_gray(events: 'np.ndarray | EventArray', width: int = 1280, height: int = 720, dtype: Any = np.uint8) -> np.ndarray:
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
    if len(events) > 0:
        _frame_gray_jit(events, buffer)
    return buffer


@lazy_njit_unwrapped_events
def _frame_rgb_jit(t, x, y, p, buffer, pos_color, neg_color):
    height, width, _ = buffer.shape
    for i in range(len(t)):
        xi = x[i]
        yi = y[i]
        pi = p[i]
        if 0 <= xi < width and 0 <= yi < height:
            if pi == 1:
                buffer[yi, xi, 0] = pos_color[0]
                buffer[yi, xi, 1] = pos_color[1]
                buffer[yi, xi, 2] = pos_color[2]
            else:
                buffer[yi, xi, 0] = neg_color[0]
                buffer[yi, xi, 1] = neg_color[1]
                buffer[yi, xi, 2] = neg_color[2]

def frame_rgb(ev: 'np.ndarray | EventArray', width: int = 1280, height: int = 720, bg_color: Any = None, pos_color: Any = None, neg_color: Any = None) -> np.ndarray:
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
    if bg_color is None: bg_color = np.array([0, 0, 0], dtype=np.uint8)
    if pos_color is None: pos_color = np.array([255, 0, 0], dtype=np.uint8)
    if neg_color is None: neg_color = np.array([0, 0, 255], dtype=np.uint8)

    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    buffer[:, :] = bg_color

    if len(ev) > 0:
        _frame_rgb_jit(ev, buffer, pos_color, neg_color)

    return buffer


@lazy_njit_unwrapped_events
def _frame_diff_jit(t, x, y, p, buffer):
    height, width = buffer.shape
    for i in range(len(t)):
        xi = x[i]
        yi = y[i]
        pi = p[i]
        if 0 <= xi < width and 0 <= yi < height:
            if pi == 1:
                buffer[yi, xi] += 1
            else:
                buffer[yi, xi] -= 1

def frame_diff(ev: 'np.ndarray | EventArray', width: int = 1280, height: int = 720, dtype: Any = np.int8) -> np.ndarray:
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
    if len(ev) > 0:
        _frame_diff_jit(ev, buffer)
    return buffer

