"""Module for generating histogram-based representations from events."""

import numpy as np
from ..jit import lazy_njit_unwrapped_events
from ..types import EventArray

@lazy_njit_unwrapped_events
def _histogram_jit(t, x, y, p, buffer, clip):
    height, width, _ = buffer.shape
    for i in range(len(t)):
        xi = x[i]
        yi = y[i]
        pi = p[i]
        if 0 <= xi < width and 0 <= yi < height:
            pi_mapped = 0 if pi == 1 else 2
            if buffer[yi, xi, pi_mapped] < clip:
                buffer[yi, xi, pi_mapped] += 1

def histogram(events: 'np.ndarray | EventArray', width: int = 1280, height: int = 720, fill: bool = False, dtype: np.dtype | type = np.uint8) -> np.ndarray:
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
    >>> from evutils.dense import histogram
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = histogram(events, width=100, height=100)
    >>> frame.shape
    (100, 100, 3)
    """
    buffer = np.zeros((height, width, 3), dtype=dtype)
    if len(events) == 0:
        return buffer

    try:
        clip = np.iinfo(dtype).max
    except ValueError:
        clip = np.finfo(dtype).max

    _histogram_jit(events, buffer, clip)

    if fill:
        buffer[buffer > 0] = 255

    return buffer

@lazy_njit_unwrapped_events
def _wedge_histogram_jit(t, x, y, p, buffer):
    height, width, _ = buffer.shape
    for i in range(len(t)):
        xi = x[i]
        yi = y[i]
        pi = p[i]
        if 0 <= xi < width and 0 <= yi < height:
            pi_mapped = 0 if pi == 1 else 2
            buffer[yi, xi, pi_mapped] = 255

def wedge_histogram(events: 'np.ndarray | EventArray', width: int = 1280, height: int = 720, tl: float = 30e6, dtype: np.dtype | type = np.uint8) -> np.ndarray:   
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
    >>> from evutils.dense import wedge_histogram
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = wedge_histogram(events, width=100, height=100)
    >>> frame.shape
    (100, 100, 3)
    """
    buffer = np.zeros((height, width, 3), dtype=dtype)
    if len(events) == 0:
        return buffer

    ts_norm = (events['t'] - events['t'][0])/tl
    sel = (height - events['y'])/height > ts_norm - 0.1
    ev_sel = events[sel]

    if len(ev_sel) > 0:
        _wedge_histogram_jit(ev_sel, buffer)

    return buffer