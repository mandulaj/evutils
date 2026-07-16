"""Module for generating Time-Ordered Recent Event (TORE) representations from events."""

import numpy as np
from typing import Any
from ..jit import lazy_njit_unwrapped_events
from ..types import EventArray

@lazy_njit_unwrapped_events
def _tore_jit(t, x, y, p, tore_fifo, tore_fifo_idx, t_res, tau):
    height, width, n_events, _ = tore_fifo.shape
    for i in range(len(t) - 1, -1, -1):
        xi = x[i]
        yi = y[i]
        pi = p[i]
        ti = t[i]
        if 0 <= xi < width and 0 <= yi < height:
            k_idx = tore_fifo_idx[yi, xi, pi]
            tore_fifo_idx[yi, xi, pi] -= 1
            if k_idx >= 0:
                dt = max(0.0, float(t_res - ti))
                tore_fifo[yi, xi, k_idx, pi] = np.exp(-dt / tau)

def tore(events: 'np.ndarray | EventArray', width: int = 1280, height: int = 720, n_events: int = 4, tau: int = 10_000, dtype: Any = np.uint8) -> np.ndarray:
    """Generate a TORE from the events.

    Parameters
    ----------
    events : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int, optional
        Width of the frame, by default 1280
    height : int, optional
        Height of the frame, by default 720
    n_events : int, optional
        Number of events to keep in the TORE, by default 4
    tau : int, optional
        Time constant for the exponential decay, by default 10_000
    dtype : np.dtype, optional
        Data type of the output array, by default np.uint8
    
    Returns
    -------
    np.ndarray
        A numpy array with the TORE representation (height, width, n_events, 2)

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.repr import tore
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> frame = tore(events, width=100, height=100, n_events=4)
    >>> frame.shape
    (100, 100, 4, 2)

    [1] Baldwin, R. W., Liu, R., Almatrafi, M., Asari, V., & Hirakawa, K. (2022). Time-ordered recent event (tore) volumes for event cameras. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2), 2519-2532.

    """
    tore_fifo = np.zeros((height, width, n_events, 2), dtype=np.float32)

    if len(events) == 0:
        return tore_fifo

    tore_fifo_idx = np.full((height, width, 2), n_events - 1, dtype=np.int32)

    t_res = events['t'][-1]
    
    _tore_jit(events, tore_fifo, tore_fifo_idx, t_res, tau)
    
    return tore_fifo